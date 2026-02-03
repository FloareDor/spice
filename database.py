"""
LocalVibe LanceDB Vector Storage Module
Stores CLaMP 3 embeddings for semantic audio search.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import lancedb
import numpy as np
import pyarrow as pa


# CLaMP 3 embedding dimension
EMBEDDING_DIM = 768


def get_db(db_path: str = "./localvibe.lance") -> lancedb.DBConnection:
    """
    Initialize or connect to a LanceDB database.

    Args:
        db_path: Path to the LanceDB database directory.

    Returns:
        LanceDB connection object.
    """
    return lancedb.connect(db_path)


def create_samples_table(db: lancedb.DBConnection, table_name: str = "samples") -> lancedb.table.Table:
    """
    Create the samples table with the appropriate schema for storing audio embeddings.

    If the table already exists, returns the existing table.

    Args:
        db: LanceDB connection.
        table_name: Name of the table to create.

    Returns:
        LanceDB Table object.
    """
    # Check if table already exists
    if table_name in db.table_names():
        return db.open_table(table_name)

    # Define schema using PyArrow
    schema = pa.schema([
        pa.field("filename", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
        pa.field("metadata", pa.string()),  # JSON-encoded metadata
    ])

    # Create empty table with schema
    return db.create_table(table_name, schema=schema)


def add_sample(
    table: lancedb.table.Table,
    filename: str,
    embedding: np.ndarray,
    metadata: dict
) -> None:
    """
    Add a single sample embedding to the table.

    Args:
        table: LanceDB table.
        filename: Unique identifier for the audio file.
        embedding: 768-dimensional CLaMP 3 embedding.
        metadata: Dictionary containing file metadata (bpm, key, path, etc.)
    """
    if embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(f"Expected embedding shape ({EMBEDDING_DIM},), got {embedding.shape}")

    # Ensure embedding is float32
    embedding = embedding.astype(np.float32)

    record = {
        "filename": filename,
        "vector": embedding.tolist(),
        "metadata": json.dumps(metadata),
    }

    table.add([record])


def add_samples_batch(table: lancedb.table.Table, samples: list[dict]) -> None:
    """
    Add multiple samples in a batch operation.

    Args:
        table: LanceDB table.
        samples: List of dicts with keys: filename, embedding, metadata
                 Each embedding should be a 768-dim numpy array.
    """
    records = []
    for sample in samples:
        embedding = sample["embedding"]
        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Expected embedding shape ({EMBEDDING_DIM},), got {embedding.shape} "
                f"for file {sample['filename']}"
            )

        records.append({
            "filename": sample["filename"],
            "vector": embedding.astype(np.float32).tolist(),
            "metadata": json.dumps(sample["metadata"]),
        })

    if records:
        table.add(records)


def search_by_embedding(
    table: lancedb.table.Table,
    query_embedding: np.ndarray,
    limit: int = 50
) -> list[dict]:
    """
    Search for similar samples using a query embedding.

    Args:
        table: LanceDB table.
        query_embedding: 768-dimensional query embedding.
        limit: Maximum number of results to return.

    Returns:
        List of dicts with filename, metadata, and distance score.
    """
    if query_embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(f"Expected embedding shape ({EMBEDDING_DIM},), got {query_embedding.shape}")

    query_embedding = query_embedding.astype(np.float32)

    results = (
        table.search(query_embedding.tolist())
        .limit(limit)
        .to_list()
    )

    # Parse metadata JSON and format results
    formatted = []
    for row in results:
        formatted.append({
            "filename": row["filename"],
            "metadata": json.loads(row["metadata"]),
            "_distance": row["_distance"],
        })

    return formatted


def delete_sample(table: lancedb.table.Table, filename: str) -> int:
    """
    Delete a sample by filename.

    Args:
        table: LanceDB table.
        filename: The filename to delete.

    Returns:
        Number of rows deleted.
    """
    # LanceDB delete uses SQL-like predicate
    table.delete(f'filename = "{filename}"')
    # LanceDB delete doesn't return count, so we just return success
    return 1


def get_sample(table: lancedb.table.Table, filename: str) -> Optional[dict]:
    """
    Retrieve a single sample by filename.

    Args:
        table: LanceDB table.
        filename: The filename to retrieve.

    Returns:
        Dict with filename, embedding, and metadata, or None if not found.
    """
    results = table.search().where(f'filename = "{filename}"').limit(1).to_list()

    if not results:
        return None

    row = results[0]
    return {
        "filename": row["filename"],
        "embedding": np.array(row["vector"], dtype=np.float32),
        "metadata": json.loads(row["metadata"]),
    }


def count_samples(table: lancedb.table.Table) -> int:
    """
    Get the total number of samples in the table.

    Args:
        table: LanceDB table.

    Returns:
        Number of samples.
    """
    return table.count_rows()


def build_metadata(
    filename: str,
    path: str,
    bpm: float = 0.0,
    key: str = "",
    duration_sec: float = 0.0,
    file_size_bytes: int = 0,
) -> dict:
    """
    Helper to build a standardized metadata dict for a sample.

    Args:
        filename: Name of the audio file.
        path: Full path to the audio file.
        bpm: Detected BPM (0 if unknown).
        key: Detected key (e.g., "C minor").
        duration_sec: Duration in seconds.
        file_size_bytes: File size in bytes.

    Returns:
        Metadata dictionary with indexed_at timestamp.
    """
    return {
        "filename": filename,
        "path": path,
        "bpm": bpm,
        "key": key,
        "duration_sec": duration_sec,
        "file_size_bytes": file_size_bytes,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
