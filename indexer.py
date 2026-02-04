"""
LocalVibe Audio Indexer
Scans folders, extracts CLaMP 3 embeddings, analyzes BPM/key, stores in LanceDB.

Usage:
    python indexer.py <folder> [--db ./localvibe.lance] [--recursive]

Activate venv first:
    .\\venv\\Scripts\\activate
"""
import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

import database as db
from validate_core import analyze_audio
import autotagger
from model_wrapper import ModelWrapper

# Supported audio extensions
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def find_audio_files(folder: Path, recursive: bool = False) -> list[Path]:
    """Find all audio files in folder."""
    files = []
    pattern = "**/*" if recursive else "*"
    for path in folder.glob(pattern):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(path)
    return sorted(files)


def get_indexed_filenames(table) -> set[str]:
    """Get set of already indexed filenames from database."""
    try:
        # Query all filenames
        results = table.search().select(["filename"]).limit(1_000_000).to_list()
        return {r["filename"] for r in results}
    except Exception:
        return set()


def analyze_audio_safe(audio_path: Path) -> dict:
    """Analyze audio with error handling."""
    try:
        return analyze_audio(str(audio_path))
    except Exception as e:
        return {"error": str(e), "bpm": 0.0, "key": "", "scale": "", "duration_sec": 0.0}


def index_folder(
    folder: Path,
    db_path: str = "./localvibe.lance",
    recursive: bool = False,
    batch_size: int = 50,
    max_workers: int = 4,
    progress_callback=None,
) -> dict:
    """
    Index all audio files in a folder.

    Args:
        folder: Path to folder containing audio files
        db_path: Path to LanceDB database
        recursive: Search subfolders recursively
        batch_size: Number of files to process per CLaMP batch
        max_workers: Number of parallel workers for librosa analysis
        progress_callback: Optional callable(processed, total, status_msg)

    Returns:
        Dict with indexing statistics
    """
    stats = {"total": 0, "new": 0, "skipped": 0, "errors": 0}

    # Find all audio files
    if progress_callback:
        progress_callback(0, 0, f"Scanning {folder}...")
    print(f"Scanning {folder}...")
    audio_files = find_audio_files(folder, recursive)
    stats["total"] = len(audio_files)
    print(f"Found {len(audio_files)} audio files")

    if not audio_files:
        return stats

    # Connect to database
    lance_db = db.get_db(db_path)
    table = db.create_samples_table(lance_db)

    # Load tag embeddings for auto-tagging
    print("Loading tag embeddings...")
    tag_embeddings = autotagger.load_tag_embeddings()

    # Get already indexed files
    indexed = get_indexed_filenames(table)
    print(f"Already indexed: {len(indexed)} files")

    # Filter to new files only
    new_files = [f for f in audio_files if f.name not in indexed]
    stats["skipped"] = len(audio_files) - len(new_files)
    stats["new"] = len(new_files)

    if not new_files:
        print("No new files to index")
        if progress_callback:
            progress_callback(len(audio_files), len(audio_files), "No new files.")
        return stats

    print(f"Indexing {len(new_files)} new files...")

    # Initialize ModelWrapper (Loads MERT + CLaMP 3)
    # This avoids reloading models for every batch
    model = ModelWrapper(quantize=True)

    processed_count = 0
    try:
        for batch_start in range(0, len(new_files), batch_size):
            batch_files = new_files[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(new_files) + batch_size - 1) // batch_size

            msg = f"Batch {batch_num}/{total_batches} ({len(batch_files)} files)"
            print(f"\n{msg}")
            if progress_callback:
                progress_callback(processed_count, len(new_files), msg)

            # Step 1: Extract CLaMP embeddings (batch)
            print("  Extracting embeddings...")
            if progress_callback:
                progress_callback(processed_count, len(new_files), f"Extracting embeddings (Batch {batch_num})...")
            
            try:
                embeddings = model.compute_embeddings(batch_files)
            except Exception as e:
                print(f"  CLaMP batch failed: {e}")
                stats["errors"] += len(batch_files)
                processed_count += len(batch_files)
                continue

            # Step 2: Analyze BPM/key in parallel
            print("  Analyzing audio (BPM/key)...")
            if progress_callback:
                progress_callback(processed_count, len(new_files), f"Analyzing audio (Batch {batch_num})...")

            analyses = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(analyze_audio_safe, f): f
                    for f in batch_files
                }
                # Disable tqdm if stderr is not available (e.g. in GUI)
                disable_tqdm = sys.stderr is None
                for future in tqdm(as_completed(futures), total=len(futures), desc="  Analyzing", disable=disable_tqdm):
                    audio_file = futures[future]
                    analyses[audio_file.name] = future.result()

            # Step 3: Prepare records for batch insert
            samples = []
            for audio_file in batch_files:
                filename = audio_file.name

                # Find embedding using full path key
                embedding = embeddings.get(str(audio_file))

                if embedding is None:
                    print(f"  Warning: No embedding for {filename}")
                    stats["errors"] += 1
                    continue

                analysis = analyses.get(filename, {})
                if "error" in analysis:
                    print(f"  Warning: Analysis failed for {filename}: {analysis['error']}")

                key_str = f"{analysis.get('key', '')} {analysis.get('scale', '')}".strip()

                # Generate tags
                tags = autotagger.suggest_tags(embedding, tag_embeddings, top_k=5)

                metadata = db.build_metadata(
                    filename=filename,
                    path=str(audio_file.resolve()),
                    bpm=analysis.get("bpm", 0.0),
                    key=key_str,
                    duration_sec=analysis.get("duration_sec", 0.0),
                    file_size_bytes=audio_file.stat().st_size,
                )
                
                # Add tags to metadata
                metadata["tags"] = tags

                samples.append({
                    "filename": filename,
                    "embedding": embedding,
                    "metadata": metadata,
                })

            # Step 4: Batch insert to LanceDB
            if samples:
                print(f"  Storing {len(samples)} embeddings...")
                db.add_samples_batch(table, samples)
            
            processed_count += len(batch_files)
            if progress_callback:
                progress_callback(processed_count, len(new_files), f"Completed Batch {batch_num}")

    finally:
        # Cleanup
        pass

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="LocalVibe Audio Indexer - Index audio samples for semantic search"
    )
    parser.add_argument(
        "folder",
        help="Path to folder containing audio files"
    )
    parser.add_argument(
        "--db",
        default="./localvibe.lance",
        help="Path to LanceDB database (default: ./localvibe.lance)"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search subfolders recursively"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Files per CLaMP batch (default: 50)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for audio analysis (default: 4)"
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  LocalVibe Audio Indexer")
    print("=" * 60)
    print(f"  Folder:    {folder}")
    print(f"  Database:  {args.db}")
    print(f"  Recursive: {args.recursive}")
    print("=" * 60)
    print()

    stats = index_folder(
        folder,
        db_path=args.db,
        recursive=args.recursive,
        batch_size=args.batch_size,
        max_workers=args.workers,
    )

    print()
    print("=" * 60)
    print("  Indexing Complete")
    print("=" * 60)
    print(f"  Total files:   {stats['total']}")
    print(f"  New indexed:   {stats['new'] - stats['errors']}")
    print(f"  Already in DB: {stats['skipped']}")
    print(f"  Errors:        {stats['errors']}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
