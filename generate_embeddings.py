"""
generate_embeddings.py
======================
Intent-Aware and Explainable Hybrid Retrieval System
----------------------------------------------------
Generate Sentence-BERT embeddings for semantic retrieval chunks.

This script only performs embedding generation:
    1. Load chunk CSV files
    2. Load SentenceTransformer
    3. Generate normalized float32 embeddings in batches
    4. Save embeddings.npy
    5. Save metadata.csv
    6. Print statistics

FAISS indexing, OpenSearch indexing, and retrieval are intentionally handled by
separate pipeline stages.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "embeddings"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64

TEXT_COLUMN = "text"
METADATA_COLUMNS = [
    "chunk_id",
    "parent_id",
    "source",
    "chunk_type",
    "title",
    "location",
    "industry",
]


@dataclass(frozen=True)
class ChunkDataset:
    """Configuration for one chunk file and its embedding output directory."""

    name: str
    display_name: str
    input_filename: str
    output_subdir: str


CHUNK_DATASETS = [
    ChunkDataset(
        name="profiles",
        display_name="Profiles",
        input_filename="profiles_chunks.csv",
        output_subdir="profiles",
    ),
    ChunkDataset(
        name="demands",
        display_name="Demands",
        input_filename="demands_chunks.csv",
        output_subdir="demands",
    ),
    ChunkDataset(
        name="jd",
        display_name="JD",
        input_filename="jd_chunks.csv",
        output_subdir="jd",
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for reusable local runs."""
    parser = argparse.ArgumentParser(
        description="Generate normalized Sentence-BERT embeddings for chunk CSV files."
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DEFAULT_CHUNKS_DIR,
        help="Directory containing profiles_chunks.csv, demands_chunks.csv, and jd_chunks.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where embedding arrays and metadata files will be written.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name or local model path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for SentenceTransformer.encode().",
    )
    return parser.parse_args()


def validate_batch_size(batch_size: int) -> None:
    """Ensure batch size is usable before starting model inference."""
    if batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")


def load_chunk_csv(chunks_dir: Path, dataset: ChunkDataset) -> pd.DataFrame:
    """Load and validate one chunk CSV."""
    input_path = chunks_dir / dataset.input_filename
    if not input_path.exists():
        raise FileNotFoundError(f"Missing chunk file: {input_path}")

    df = pd.read_csv(input_path)
    required_columns = METADATA_COLUMNS + [TEXT_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{input_path} is missing required columns: {missing}")

    if df.empty:
        raise ValueError(f"{input_path} has no rows to embed")

    if df["chunk_id"].duplicated().any():
        duplicate_count = int(df["chunk_id"].duplicated().sum())
        raise ValueError(f"{input_path} contains {duplicate_count} duplicate chunk_id values")

    if df[TEXT_COLUMN].isna().any():
        missing_count = int(df[TEXT_COLUMN].isna().sum())
        raise ValueError(f"{input_path} contains {missing_count} rows with missing text")

    return df


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load the SentenceTransformer model once for all chunk datasets."""
    print(f"Loading model: {model_name}")
    return SentenceTransformer(model_name)


def generate_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """Generate normalized float32 embeddings using batched SentenceTransformer encoding."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return np.asarray(embeddings, dtype=np.float32)


def save_dataset_outputs(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: Path,
    dataset: ChunkDataset,
) -> None:
    """Save embeddings.npy and metadata.csv for one dataset."""
    if embeddings.shape[0] != len(df):
        raise ValueError(
            f"Embedding row mismatch for {dataset.name}: "
            f"{embeddings.shape[0]} embeddings for {len(df)} metadata rows"
        )

    dataset_output_dir = output_dir / dataset.output_subdir
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    np.save(dataset_output_dir / "embeddings.npy", embeddings)

    # Keep metadata lightweight. Embeddings stay only in the .npy array.
    metadata = df.loc[:, METADATA_COLUMNS].fillna("")
    metadata.to_csv(dataset_output_dir / "metadata.csv", index=False, encoding="utf-8")


def process_dataset(
    model: SentenceTransformer,
    chunks_dir: Path,
    output_dir: Path,
    dataset: ChunkDataset,
    batch_size: int,
) -> tuple[int, int]:
    """Load, embed, and save outputs for one chunk dataset."""
    df = load_chunk_csv(chunks_dir, dataset)
    texts = df[TEXT_COLUMN].astype(str).tolist()

    print(f"\nEmbedding {dataset.display_name}: {len(texts)} chunks")
    embeddings = generate_embeddings(
        model=model,
        texts=texts,
        batch_size=batch_size,
    )

    save_dataset_outputs(
        df=df,
        embeddings=embeddings,
        output_dir=output_dir,
        dataset=dataset,
    )

    return len(df), int(embeddings.shape[1])


def print_statistics(
    counts: dict[str, int],
    embedding_dimension: int,
    model_name: str,
    output_dir: Path,
) -> None:
    """Print the requested embedding generation summary."""
    print("\nEmbedding generation complete")
    print("-----------------------------")
    print(f"Profiles Embedded : {counts.get('profiles', 0)}")
    print(f"Demands Embedded : {counts.get('demands', 0)}")
    print(f"JD Embedded : {counts.get('jd', 0)}")
    print(f"Embedding Dimension : {embedding_dimension}")
    print(f"Model Used : {model_name}")
    try:
        output_display = output_dir.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        output_display = output_dir.as_posix()

    print(f"Output Folder : {output_display.rstrip('/')}/")


def main() -> int:
    """Run the embedding generation pipeline."""
    args = parse_args()

    chunks_dir = args.chunks_dir.resolve()
    output_dir = args.output_dir.resolve()
    validate_batch_size(args.batch_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_embedding_model(args.model_name)

    counts: dict[str, int] = {}
    embedding_dimensions: set[int] = set()

    for dataset in CHUNK_DATASETS:
        row_count, embedding_dimension = process_dataset(
            model=model,
            chunks_dir=chunks_dir,
            output_dir=output_dir,
            dataset=dataset,
            batch_size=args.batch_size,
        )
        counts[dataset.name] = row_count
        embedding_dimensions.add(embedding_dimension)

    if len(embedding_dimensions) != 1:
        raise ValueError(f"Inconsistent embedding dimensions found: {sorted(embedding_dimensions)}")

    print_statistics(
        counts=counts,
        embedding_dimension=embedding_dimensions.pop(),
        model_name=args.model_name,
        output_dir=output_dir,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nEmbedding generation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
