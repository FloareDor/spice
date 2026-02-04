"""
LocalVibe Search CLI
Search your indexed audio samples by audio file or text description.

Usage:
    python search.py --text "warm bass"                    # Text search
    python search.py --text "heavy drums" --bpm 140        # Text + BPM filter
    python search.py --text "dark pad" --key "C minor"     # Text + key filter
    python search.py --audio <file.wav>                    # Find similar to audio
    python search.py --audio <file.wav> --bpm 120-140      # Similar + BPM range
    python search.py --list                                # List all indexed

Activate venv first:
    .\\venv\\Scripts\\activate
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from . import database as db
from .model_wrapper import get_model

# Default database path
DEFAULT_DB = "./localvibe.lance"

# BPM tolerance for single value (±5 BPM)
BPM_TOLERANCE = 5


def parse_bpm_filter(bpm_str: str) -> tuple[float, float]:
    """
    Parse BPM filter string into (min, max) range.

    Formats:
        "140"       -> (135, 145)  # ±5 tolerance
        "120-150"   -> (120, 150)  # explicit range
        "120+"      -> (120, inf)  # minimum
        "-150"      -> (0, 150)    # maximum
    """
    bpm_str = bpm_str.strip()

    # Range format: "120-150"
    if "-" in bpm_str and not bpm_str.startswith("-"):
        parts = bpm_str.split("-")
        if len(parts) == 2:
            low = float(parts[0]) if parts[0] else 0
            high = float(parts[1]) if parts[1] else float("inf")
            return (low, high)

    # Minimum format: "120+"
    if bpm_str.endswith("+"):
        return (float(bpm_str[:-1]), float("inf"))

    # Maximum format: "-150" (handled above if no leading -)
    if bpm_str.startswith("-"):
        return (0, float(bpm_str[1:]))

    # Single value with tolerance
    bpm = float(bpm_str)
    return (bpm - BPM_TOLERANCE, bpm + BPM_TOLERANCE)


def parse_key_filter(key_str: str) -> tuple[str | None, str | None]:
    """
    Parse key filter string into (root, scale).

    Formats:
        "C"         -> ("C", None)      # Any C (major or minor)
        "C minor"   -> ("C", "minor")   # Exact match
        "C#"        -> ("C#", None)     # Any C#
        "Cm"        -> ("C", "minor")   # Shorthand
        "CM"        -> ("C", "major")   # Shorthand
        "minor"     -> (None, "minor")  # Any minor key
        "major"     -> (None, "major")  # Any major key
    """
    key_str = key_str.strip().lower()

    # Scale only: "minor", "major"
    if key_str in ("minor", "major"):
        return (None, key_str)

    # Shorthand: "Cm", "CM", "C#m", "C#M"
    if key_str.endswith("m") and not key_str.endswith("inor"):
        return (key_str[:-1].upper(), "minor")
    if key_str.endswith("M") and not key_str.endswith("ajor"):
        return (key_str[:-1].upper(), "major")

    # Full format: "C minor", "C# major"
    parts = key_str.split()
    if len(parts) == 2:
        return (parts[0].upper(), parts[1].lower())

    # Just root: "C", "C#"
    return (key_str.upper(), None)


def matches_bpm(metadata: dict, bpm_range: tuple[float, float]) -> bool:
    """Check if sample BPM is within range."""
    bpm = metadata.get("bpm", 0)
    if bpm == 0:
        return True  # Include samples with unknown BPM
    return bpm_range[0] <= bpm <= bpm_range[1]


def matches_key(metadata: dict, key_filter: tuple[str | None, str | None]) -> bool:
    """Check if sample key matches filter."""
    key_str = metadata.get("key", "")
    if not key_str:
        return True  # Include samples with unknown key

    root, scale = key_filter

    # Check root note (if specified)
    if root and not key_str.upper().startswith(root.upper()):
        return False

    # Check scale (if specified)
    if scale and scale.lower() not in key_str.lower():
        return False

    return True


def filter_results(
    results: list[dict],
    bpm_range: tuple[float, float] | None = None,
    key_filter: tuple[str | None, str | None] | None = None,
) -> list[dict]:
    """Filter search results by BPM and/or key."""
    filtered = []
    for r in results:
        meta = r["metadata"]

        if bpm_range and not matches_bpm(meta, bpm_range):
            continue
        if key_filter and not matches_key(meta, key_filter):
            continue

        filtered.append(r)

    return filtered


def get_audio_embedding(
    audio_path: Path,
    temp_dir: Path = None,
    status_callback=None
) -> np.ndarray:
    """Extract CLaMP 3 embedding from audio file using shared ModelWrapper."""
    model = get_model(quantize=True, status_callback=status_callback)

    results = model.compute_embeddings([audio_path])
    if str(audio_path) in results:
        return results[str(audio_path)]

    raise RuntimeError("Failed to generate embedding")


def get_text_embedding(
    text: str,
    temp_dir: Path = None,
    status_callback=None
) -> np.ndarray:
    """Extract CLaMP 3 embedding from text description using shared ModelWrapper."""
    model = get_model(quantize=True, status_callback=status_callback)

    embeddings = model.compute_text_embeddings([text])
    if embeddings:
        return embeddings[0]

    raise RuntimeError("Failed to generate text embedding")


def search_by_audio(
    audio_path: Path,
    db_path: str,
    limit: int,
    bpm_range: tuple[float, float] | None = None,
    key_filter: tuple[str | None, str | None] | None = None,
    status_callback=None,
) -> list[dict]:
    """Search for samples similar to an audio file."""
    if status_callback:
        status_callback(f"Extracting embedding from {audio_path.name}...")
    else:
        print(f"Extracting embedding from {audio_path.name}...")

    embedding = get_audio_embedding(audio_path, status_callback=status_callback)

    lance_db = db.get_db(db_path)
    table = db.create_samples_table(lance_db)

    # Fetch more results if filtering, then trim
    fetch_limit = limit * 5 if (bpm_range or key_filter) else limit
    results = db.search_by_embedding(table, embedding, limit=fetch_limit)

    if bpm_range or key_filter:
        results = filter_results(results, bpm_range, key_filter)

    return results[:limit]


def search_by_text(
    text: str,
    db_path: str,
    limit: int,
    bpm_range: tuple[float, float] | None = None,
    key_filter: tuple[str | None, str | None] | None = None,
    status_callback=None,
) -> list[dict]:
    """Search for samples matching a text description."""
    if status_callback:
        status_callback(f"Generating embedding for: \"{text}\"...")
    else:
        print(f"Generating embedding for: \"{text}\"...")

    embedding = get_text_embedding(text, status_callback=status_callback)

    lance_db = db.get_db(db_path)
    table = db.create_samples_table(lance_db)

    # Fetch more results if filtering, then trim
    fetch_limit = limit * 5 if (bpm_range or key_filter) else limit
    results = db.search_by_embedding(table, embedding, limit=fetch_limit)

    if bpm_range or key_filter:
        results = filter_results(results, bpm_range, key_filter)

    return results[:limit]


def search_similar_to_sample(
    file_path: str,
    db_path: str,
    limit: int,
    bpm_range: tuple[float, float] | None = None,
    key_filter: tuple[str | None, str | None] | None = None,
) -> list[dict]:
    """
    Search for samples similar to an already-indexed sample.
    Uses the stored embedding instead of recomputing.

    Args:
        file_path: Full path to the sample (as stored in metadata)
        db_path: Path to LanceDB database
        limit: Max results to return
        bpm_range: Optional BPM filter
        key_filter: Optional key filter

    Returns:
        List of similar samples (excluding the query sample itself)
    """
    lance_db = db.get_db(db_path)
    table = db.create_samples_table(lance_db)

    # Get the sample's stored embedding
    sample = db.get_sample(table, Path(file_path).name)
    if sample is None:
        raise ValueError(f"Sample not found in database: {file_path}")

    embedding = sample["embedding"]

    # Search for similar (fetch extra to account for filtering + excluding self)
    fetch_limit = (limit + 1) * 5 if (bpm_range or key_filter) else limit + 1
    results = db.search_by_embedding(table, embedding, limit=fetch_limit)

    # Filter out the query sample itself
    results = [r for r in results if r["metadata"].get("path") != file_path]

    if bpm_range or key_filter:
        results = filter_results(results, bpm_range, key_filter)

    return results[:limit]


def list_samples(
    db_path: str,
    limit: int,
    bpm_range: tuple[float, float] | None = None,
    key_filter: tuple[str | None, str | None] | None = None,
) -> None:
    """List all indexed samples, optionally filtered."""
    import json

    lance_db = db.get_db(db_path)
    table = db.create_samples_table(lance_db)

    total = db.count_samples(table)
    print(f"Total indexed samples: {total}")

    if total == 0:
        return

    # Get samples
    if bpm_range or key_filter:
        fetch_limit = max(limit * 10, 1000)
    else:
        fetch_limit = limit

    rows = table.search().limit(fetch_limit).to_list()

    # Convert to result format and filter
    results = []
    for row in rows:
        meta = json.loads(row["metadata"])
        results.append({"metadata": meta})

    if bpm_range or key_filter:
        results = filter_results(results, bpm_range, key_filter)
        print(f"Filtered samples: {len(results)}")

    results = results[:limit]
    print()

    print(f"{'Filename':<40} {'BPM':>6} {'Key':<12}")
    print("-" * 60)

    for r in results:
        meta = r["metadata"]
        filename = meta.get("filename", "?")[:39]
        bpm = meta.get("bpm", 0)
        key = meta.get("key", "")[:11]
        print(f"{filename:<40} {bpm:>6.1f} {key:<12}")


def print_results(results: list[dict], query_type: str) -> None:
    """Print search results in a formatted table."""
    if not results:
        print("No results found.")
        return

    print()
    print(f"{'#':<3} {'Filename':<35} {'Distance':>10} {'BPM':>6} {'Key':<12}")
    print("-" * 70)

    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        filename = meta.get("filename", "?")[:34]
        distance = r["_distance"]
        bpm = meta.get("bpm", 0)
        key = meta.get("key", "")[:11]
        print(f"{i:<3} {filename:<35} {distance:>10.4f} {bpm:>6.1f} {key:<12}")

    print()
    print(f"Showing top {len(results)} results")


def main():
    parser = argparse.ArgumentParser(
        description="LocalVibe Search - Find audio samples by similarity or description",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python search.py --text "heavy bass"
  python search.py --text "bright synth" --bpm 128
  python search.py --text "dark pad" --bpm 100-140 --key "C minor"
  python search.py --audio kick.wav --bpm 140
  python search.py --list --key "E"
        """
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--audio", "-a",
        type=str,
        help="Path to audio file to find similar samples"
    )
    group.add_argument(
        "--text", "-t",
        type=str,
        help="Text description to search for (e.g., 'warm bass', 'bright synth lead')"
    )
    group.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all indexed samples"
    )
    parser.add_argument(
        "--bpm", "-b",
        type=str,
        help="Filter by BPM: '140' (±5), '120-150' (range), '120+' (min), '-150' (max)"
    )
    parser.add_argument(
        "--key", "-k",
        type=str,
        help="Filter by key: 'C', 'C minor', 'C#', 'Cm' (shorthand for C minor)"
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"Path to LanceDB database (default: {DEFAULT_DB})"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )
    args = parser.parse_args()

    # Parse filters
    bpm_range = parse_bpm_filter(args.bpm) if args.bpm else None
    key_filter = parse_key_filter(args.key) if args.key else None

    # Check database exists
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {args.db}")
        print("Run 'python indexer.py <folder>' first to index your samples.")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  LocalVibe Search")
    print("=" * 60)

    # Show active filters
    if bpm_range:
        if bpm_range[1] == float("inf"):
            print(f"  BPM filter: {bpm_range[0]}+")
        elif bpm_range[0] == 0:
            print(f"  BPM filter: -{bpm_range[1]}")
        else:
            print(f"  BPM filter: {bpm_range[0]}-{bpm_range[1]}")
    if key_filter:
        parts = []
        if key_filter[0]: parts.append(key_filter[0])
        if key_filter[1]: parts.append(key_filter[1])
        key_str = " ".join(parts)
        print(f"  Key filter: {key_str}")

    if args.list:
        print("=" * 60)
        list_samples(args.db, args.limit, bpm_range, key_filter)

    elif args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"Error: Audio file not found: {args.audio}")
            sys.exit(1)

        print(f"  Query: {audio_path.name}")
        print("=" * 60)

        results = search_by_audio(audio_path, args.db, args.limit, bpm_range, key_filter)
        print_results(results, "audio")

    elif args.text:
        print(f"  Query: \"{args.text}\"")
        print("=" * 60)

        results = search_by_text(args.text, args.db, args.limit, bpm_range, key_filter)
        print_results(results, "text")


if __name__ == "__main__":
    main()
