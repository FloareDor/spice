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


def run_clamp_batch(audio_files: list[Path], output_dir: Path) -> dict[str, np.ndarray]:
    """
    Run CLaMP 3 on a batch of audio files.

    Returns:
        Dict mapping filename -> 768-dim embedding
    """
    if not audio_files:
        return {}

    clamp_script = Path("clamp3/clamp3_embd.py").resolve()
    if not clamp_script.exists():
        raise FileNotFoundError(f"CLaMP 3 not found at {clamp_script}")

    # Create temp input directory with all files
    temp_input = output_dir / "clamp_batch_input"
    temp_input.mkdir(exist_ok=True)

    # Copy all audio files to temp input (preserving names)
    for audio_file in audio_files:
        dest = temp_input / audio_file.name
        # Handle duplicate names by adding parent folder
        if dest.exists():
            dest = temp_input / f"{audio_file.parent.name}_{audio_file.name}"
        shutil.copy(audio_file, dest)

    temp_output = output_dir / "clamp_batch_output"
    if temp_output.exists():
        shutil.rmtree(temp_output)

    # Run CLaMP 3
    cmd = [
        sys.executable,
        str(clamp_script),
        str(temp_input.resolve()),
        str(temp_output.resolve()),
        "--get_global"
    ]

    result = subprocess.run(
        cmd,
        cwd=str(clamp_script.parent),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"CLaMP error: {result.stderr}")
        raise RuntimeError(f"CLaMP extraction failed: {result.returncode}")

    # Load all embeddings
    embeddings = {}
    for npy_file in temp_output.rglob("*.npy"):
        embedding = np.load(npy_file)
        if embedding.ndim > 1:
            embedding = embedding.flatten()

        # Map back to original filename
        stem = npy_file.stem
        embeddings[stem] = embedding

    # Cleanup
    shutil.rmtree(temp_input, ignore_errors=True)
    shutil.rmtree(temp_output, ignore_errors=True)

    return embeddings


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
) -> dict:
    """
    Index all audio files in a folder.

    Args:
        folder: Path to folder containing audio files
        db_path: Path to LanceDB database
        recursive: Search subfolders recursively
        batch_size: Number of files to process per CLaMP batch
        max_workers: Number of parallel workers for librosa analysis

    Returns:
        Dict with indexing statistics
    """
    stats = {"total": 0, "new": 0, "skipped": 0, "errors": 0}

    # Find all audio files
    print(f"Scanning {folder}...")
    audio_files = find_audio_files(folder, recursive)
    stats["total"] = len(audio_files)
    print(f"Found {len(audio_files)} audio files")

    if not audio_files:
        return stats

    # Connect to database
    lance_db = db.get_db(db_path)
    table = db.create_samples_table(lance_db)

    # Get already indexed files
    indexed = get_indexed_filenames(table)
    print(f"Already indexed: {len(indexed)} files")

    # Filter to new files only
    new_files = [f for f in audio_files if f.name not in indexed]
    stats["skipped"] = len(audio_files) - len(new_files)
    stats["new"] = len(new_files)

    if not new_files:
        print("No new files to index")
        return stats

    print(f"Indexing {len(new_files)} new files...")

    # Process in batches
    temp_dir = Path(db_path).parent / ".indexer_temp"
    temp_dir.mkdir(exist_ok=True)

    try:
        for batch_start in range(0, len(new_files), batch_size):
            batch_files = new_files[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(new_files) + batch_size - 1) // batch_size

            print(f"\nBatch {batch_num}/{total_batches} ({len(batch_files)} files)")

            # Step 1: Extract CLaMP embeddings (batch)
            print("  Extracting embeddings...")
            try:
                embeddings = run_clamp_batch(batch_files, temp_dir)
            except Exception as e:
                print(f"  CLaMP batch failed: {e}")
                stats["errors"] += len(batch_files)
                continue

            # Step 2: Analyze BPM/key in parallel
            print("  Analyzing audio (BPM/key)...")
            analyses = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(analyze_audio_safe, f): f
                    for f in batch_files
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="  Analyzing"):
                    audio_file = futures[future]
                    analyses[audio_file.name] = future.result()

            # Step 3: Prepare records for batch insert
            samples = []
            for audio_file in batch_files:
                filename = audio_file.name
                stem = audio_file.stem

                # Find embedding (handle name variations)
                embedding = embeddings.get(stem)
                if embedding is None:
                    # Try with parent folder prefix
                    alt_stem = f"{audio_file.parent.name}_{stem}"
                    embedding = embeddings.get(alt_stem)

                if embedding is None:
                    print(f"  Warning: No embedding for {filename}")
                    stats["errors"] += 1
                    continue

                analysis = analyses.get(filename, {})
                if "error" in analysis:
                    print(f"  Warning: Analysis failed for {filename}: {analysis['error']}")

                key_str = f"{analysis.get('key', '')} {analysis.get('scale', '')}".strip()

                metadata = db.build_metadata(
                    filename=filename,
                    path=str(audio_file.resolve()),
                    bpm=analysis.get("bpm", 0.0),
                    key=key_str,
                    duration_sec=analysis.get("duration_sec", 0.0),
                    file_size_bytes=audio_file.stat().st_size,
                )

                samples.append({
                    "filename": filename,
                    "embedding": embedding,
                    "metadata": metadata,
                })

            # Step 4: Batch insert to LanceDB
            if samples:
                print(f"  Storing {len(samples)} embeddings...")
                db.add_samples_batch(table, samples)

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

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
