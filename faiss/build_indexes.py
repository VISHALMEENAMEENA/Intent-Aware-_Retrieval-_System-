"""
build_indexes.py
================
Intent-Aware and Explainable Hybrid Retrieval System
----------------------------------------------------
Build FAISS semantic indexes from precomputed Sentence-BERT embeddings.

This module only handles dense semantic indexing:
    1. Load normalized embedding arrays
    2. Load chunk metadata
    3. Build one IndexFlatIP index per dataset
    4. Persist FAISS indexes and FAISS-ID mapping files

It does not implement BM25, hybrid retrieval, RRF, reranking, or intent
detection.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import faiss
except ImportError:  # Dependency is checked at runtime so --help still works.
    faiss = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
DEFAULT_INDEXES_DIR = Path(__file__).resolve().parent / "indexes"
EMBEDDING_DIMENSION = 384

MAPPING_METADATA_COLUMNS = [
    "chunk_id",
    "parent_id",
    "source",
    "chunk_type",
    "title",
    "location",
    "industry",
]

MAPPING_COLUMNS = ["faiss_id", *MAPPING_METADATA_COLUMNS]


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for one embedding dataset and its FAISS outputs."""

    name: str
    display_name: str
    embeddings_subdir: str
    index_filename: str
    mapping_filename: str
    stats_label: str


DATASETS = [
    DatasetConfig(
        name="profiles",
        display_name="Profiles",
        embeddings_subdir="profiles",
        index_filename="profile_index.bin",
        mapping_filename="profile_mapping.csv",
        stats_label="Profile vectors indexed",
    ),
    DatasetConfig(
        name="demands",
        display_name="Demands",
        embeddings_subdir="demands",
        index_filename="demand_index.bin",
        mapping_filename="demand_mapping.csv",
        stats_label="Demand vectors indexed",
    ),
    DatasetConfig(
        name="jd",
        display_name="JD",
        embeddings_subdir="jd",
        index_filename="jd_index.bin",
        mapping_filename="jd_mapping.csv",
        stats_label="JD vectors indexed",
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse FAISS indexing command-line options."""
    parser = argparse.ArgumentParser(
        description="Build FAISS IndexFlatIP indexes from normalized embedding arrays."
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=DEFAULT_EMBEDDINGS_DIR,
        help="Directory containing profiles, demands, and jd embedding subdirectories.",
    )
    parser.add_argument(
        "--indexes-dir",
        type=Path,
        default=DEFAULT_INDEXES_DIR,
        help="Directory where FAISS index and mapping files will be written.",
    )
    return parser.parse_args()


def ensure_faiss_available() -> Any:
    """Return the FAISS module or raise a clear dependency error."""
    if faiss is None:
        raise ImportError("Missing dependency: install it with `pip install faiss-cpu`.")

    if not hasattr(faiss, "IndexFlatIP"):
        raise ImportError(
            "Imported module named 'faiss' does not expose IndexFlatIP. "
            "Run this script directly and ensure faiss-cpu is installed."
        )

    return faiss


def load_embeddings(embeddings_dir: Path, dataset: DatasetConfig) -> np.ndarray:
    """Load and validate one embedding array."""
    embeddings_path = embeddings_dir / dataset.embeddings_subdir / "embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")

    vectors = np.load(embeddings_path)
    if vectors.ndim != 2:
        raise ValueError(f"{embeddings_path} must be a 2D array, found shape {vectors.shape}")

    row_count, dimension = vectors.shape
    if row_count == 0:
        raise ValueError(f"{embeddings_path} contains no vectors")

    if dimension != EMBEDDING_DIMENSION:
        raise ValueError(
            f"{embeddings_path} has dimension {dimension}; expected {EMBEDDING_DIMENSION}"
        )

    if not np.isfinite(vectors).all():
        raise ValueError(f"{embeddings_path} contains NaN or infinite values")

    return np.ascontiguousarray(vectors, dtype=np.float32)


def load_mapping_metadata(embeddings_dir: Path, dataset: DatasetConfig) -> pd.DataFrame:
    """Load metadata and create a FAISS-ID mapping dataframe."""
    metadata_path = embeddings_dir / dataset.embeddings_subdir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    missing_columns = [
        column for column in MAPPING_METADATA_COLUMNS if column not in metadata.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{metadata_path} is missing required columns: {missing}")

    if metadata.empty:
        raise ValueError(f"{metadata_path} has no metadata rows")

    if metadata["chunk_id"].duplicated().any():
        duplicate_count = int(metadata["chunk_id"].duplicated().sum())
        raise ValueError(f"{metadata_path} contains {duplicate_count} duplicate chunk_id values")

    mapping = metadata.loc[:, MAPPING_METADATA_COLUMNS].fillna("").copy()
    mapping.insert(0, "faiss_id", range(len(mapping)))
    return mapping.loc[:, MAPPING_COLUMNS]


def build_index(vectors: np.ndarray) -> Any:
    """Build an inner-product FAISS index for normalized vectors."""
    faiss_module = ensure_faiss_available()
    index = faiss_module.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(vectors)
    return index


def write_index_outputs(
    index: Any,
    mapping: pd.DataFrame,
    indexes_dir: Path,
    dataset: DatasetConfig,
) -> None:
    """Persist one FAISS index and its ID mapping CSV."""
    faiss_module = ensure_faiss_available()
    indexes_dir.mkdir(parents=True, exist_ok=True)

    index_path = indexes_dir / dataset.index_filename
    mapping_path = indexes_dir / dataset.mapping_filename

    faiss_module.write_index(index, str(index_path))
    mapping.to_csv(mapping_path, index=False, encoding="utf-8")


def process_dataset(
    embeddings_dir: Path,
    indexes_dir: Path,
    dataset: DatasetConfig,
) -> int:
    """Build and save FAISS outputs for one dataset."""
    vectors = load_embeddings(embeddings_dir, dataset)
    mapping = load_mapping_metadata(embeddings_dir, dataset)

    if len(mapping) != vectors.shape[0]:
        raise ValueError(
            f"Row mismatch for {dataset.name}: {vectors.shape[0]} vectors but "
            f"{len(mapping)} metadata rows"
        )

    index = build_index(vectors)
    if index.ntotal != vectors.shape[0]:
        raise ValueError(
            f"FAISS index for {dataset.name} contains {index.ntotal} vectors after add(); "
            f"expected {vectors.shape[0]}"
        )

    write_index_outputs(
        index=index,
        mapping=mapping,
        indexes_dir=indexes_dir,
        dataset=dataset,
    )

    return int(index.ntotal)


def main() -> int:
    """Build FAISS indexes for all semantic retrieval datasets."""
    args = parse_args()
    embeddings_dir = args.embeddings_dir.resolve()
    indexes_dir = args.indexes_dir.resolve()

    ensure_faiss_available()

    counts: dict[str, int] = {}
    for dataset in DATASETS:
        counts[dataset.name] = process_dataset(
            embeddings_dir=embeddings_dir,
            indexes_dir=indexes_dir,
            dataset=dataset,
        )

    for dataset in DATASETS:
        print(f"{dataset.stats_label} : {counts[dataset.name]}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nFAISS index build failed: {error}", file=sys.stderr)
        raise SystemExit(1)
