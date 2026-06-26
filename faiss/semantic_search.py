"""
semantic_search.py
==================
Reusable FAISS semantic search over intent-specific dense indexes.

This module only handles dense semantic retrieval:
    1. Load the dataset-specific FAISS index
    2. Encode the query with the same Sentence-BERT model
    3. Search by inner product over normalized vectors
    4. Convert FAISS IDs back to chunk metadata
    5. Attach retrieved chunk text from the chunk CSV

It does not implement OpenSearch, BM25, hybrid retrieval, RRF, reranking, or
intent detection.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

try:
    import faiss
except ImportError:  # Dependency is checked at runtime so --help still works.
    faiss = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAISS_DIR = Path(__file__).resolve().parent
DEFAULT_INDEXES_DIR = FAISS_DIR / "indexes"
DEFAULT_CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

MAPPING_COLUMNS = [
    "faiss_id",
    "chunk_id",
    "parent_id",
    "source",
    "chunk_type",
    "title",
    "location",
    "industry",
]


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for one semantic-search dataset."""

    name: str
    display_name: str
    index_filename: str
    mapping_filename: str
    chunk_filename: str


DATASETS = {
    "profiles": DatasetConfig(
        name="profiles",
        display_name="Profiles",
        index_filename="profile_index.bin",
        mapping_filename="profile_mapping.csv",
        chunk_filename="profiles_chunks.csv",
    ),
    "demands": DatasetConfig(
        name="demands",
        display_name="Demands",
        index_filename="demand_index.bin",
        mapping_filename="demand_mapping.csv",
        chunk_filename="demands_chunks.csv",
    ),
    "jd": DatasetConfig(
        name="jd",
        display_name="JD",
        index_filename="jd_index.bin",
        mapping_filename="jd_mapping.csv",
        chunk_filename="jd_chunks.csv",
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse semantic search command-line options."""
    parser = argparse.ArgumentParser(
        description="Run FAISS semantic search against one dense retrieval dataset."
    )
    parser.add_argument("--query", required=True, help="Search query text.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS),
        help="Dataset to search: profiles, demands, or jd.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--indexes-dir",
        type=Path,
        default=DEFAULT_INDEXES_DIR,
        help="Directory containing FAISS index and mapping files.",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DEFAULT_CHUNKS_DIR,
        help="Directory containing chunk CSV files used for retrieved text.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name or local model path.",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow SentenceTransformer to download the query encoder if it is not cached.",
    )
    return parser.parse_args()


def ensure_faiss_available() -> Any:
    """Return the FAISS module or raise a clear dependency error."""
    if faiss is None:
        raise ImportError("Missing dependency: install it with `pip install faiss-cpu`.")

    if not hasattr(faiss, "read_index"):
        raise ImportError(
            "Imported module named 'faiss' does not expose read_index. "
            "Run this script directly and ensure faiss-cpu is installed."
        )

    return faiss


def ensure_sentence_transformer_available() -> Any:
    """Return SentenceTransformer or raise a clear dependency error."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "Missing dependency: install it with `pip install sentence-transformers`."
        ) from error

    return SentenceTransformer


def get_dataset_config(dataset: str) -> DatasetConfig:
    """Resolve and validate a dataset name."""
    dataset_key = dataset.strip().lower()
    if dataset_key not in DATASETS:
        supported = ", ".join(sorted(DATASETS))
        raise ValueError(f"dataset must be one of: {supported}")

    return DATASETS[dataset_key]


def validate_search_inputs(query: str, top_k: int) -> str:
    """Validate reusable search() inputs and return a cleaned query."""
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query must be a non-empty string")

    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    return cleaned_query


def load_index(indexes_dir: Path, dataset: DatasetConfig) -> Any:
    """Load one persisted FAISS index from disk."""
    faiss_module = ensure_faiss_available()
    index_path = indexes_dir / dataset.index_filename
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing FAISS index: {index_path}. Run faiss/build_indexes.py first."
        )

    index = faiss_module.read_index(str(index_path))
    if index.d != EMBEDDING_DIMENSION:
        raise ValueError(
            f"{index_path} has dimension {index.d}; expected {EMBEDDING_DIMENSION}"
        )

    if index.ntotal == 0:
        raise ValueError(f"{index_path} contains no vectors")

    return index


def load_mapping(indexes_dir: Path, dataset: DatasetConfig) -> dict[int, dict[str, Any]]:
    """Load a FAISS-ID-to-metadata mapping CSV."""
    mapping_path = indexes_dir / dataset.mapping_filename
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Missing FAISS mapping: {mapping_path}. Run faiss/build_indexes.py first."
        )

    mapping = pd.read_csv(mapping_path, dtype=str, keep_default_na=False)
    missing_columns = [column for column in MAPPING_COLUMNS if column not in mapping.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{mapping_path} is missing required columns: {missing}")

    if mapping.empty:
        raise ValueError(f"{mapping_path} has no mapping rows")

    if mapping["faiss_id"].duplicated().any():
        duplicate_count = int(mapping["faiss_id"].duplicated().sum())
        raise ValueError(f"{mapping_path} contains {duplicate_count} duplicate faiss_id values")

    try:
        mapping["faiss_id"] = pd.to_numeric(mapping["faiss_id"], errors="raise").astype(int)
    except Exception as error:
        raise ValueError(f"{mapping_path} contains non-integer faiss_id values") from error

    rows = mapping.loc[:, MAPPING_COLUMNS].to_dict(orient="records")
    return {int(row["faiss_id"]): row for row in rows}


def load_chunk_texts(chunks_dir: Path, dataset: DatasetConfig) -> dict[str, str]:
    """Load chunk text by chunk_id for retrieved evidence text."""
    chunk_path = chunks_dir / dataset.chunk_filename
    if not chunk_path.exists():
        raise FileNotFoundError(
            f"Missing chunk text file: {chunk_path}. Run chunk_data.py first."
        )

    chunks = pd.read_csv(chunk_path, dtype=str, keep_default_na=False)
    required_columns = ["chunk_id", "text"]
    missing_columns = [column for column in required_columns if column not in chunks.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{chunk_path} is missing required columns: {missing}")

    if chunks["chunk_id"].duplicated().any():
        duplicate_count = int(chunks["chunk_id"].duplicated().sum())
        raise ValueError(f"{chunk_path} contains {duplicate_count} duplicate chunk_id values")

    return dict(zip(chunks["chunk_id"], chunks["text"]))


def load_embedding_model(model_name: str, allow_model_download: bool = False) -> Any:
    """Load the SentenceTransformer query encoder."""
    model_cls = ensure_sentence_transformer_available()
    with contextlib.suppress(Exception):
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return model_cls(
                model_name,
                local_files_only=not allow_model_download,
            )
    except Exception as error:
        if allow_model_download:
            load_mode = "from the local cache or model hub"
        else:
            load_mode = "from the local cache"

        raise RuntimeError(
            f"Could not load SentenceTransformer model '{model_name}' {load_mode}. "
            "Use a local model path, pre-cache the model, or pass --allow-model-download "
            "when network access is available."
        ) from error


def encode_query(model: Any, query: str) -> np.ndarray:
    """Encode and L2-normalize a query for IndexFlatIP cosine-equivalent search."""
    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_vector = np.asarray(query_vector, dtype=np.float32)
    if query_vector.shape != (1, EMBEDDING_DIMENSION):
        raise ValueError(
            f"Query embedding has shape {query_vector.shape}; "
            f"expected (1, {EMBEDDING_DIMENSION})"
        )

    return np.ascontiguousarray(query_vector, dtype=np.float32)


def format_result(
    *,
    rank: int,
    faiss_id: int,
    score: float,
    mapping: dict[int, dict[str, Any]],
    chunk_texts: dict[str, str],
) -> dict[str, Any]:
    """Convert one FAISS hit into the semantic retrieval result schema."""
    metadata = mapping.get(faiss_id)
    if metadata is None:
        raise KeyError(f"FAISS ID {faiss_id} was not found in the mapping CSV")

    chunk_id = str(metadata.get("chunk_id", ""))
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "similarity_score": float(score),
        "title": metadata.get("title", ""),
        "source": metadata.get("source", ""),
        "chunk_type": metadata.get("chunk_type", ""),
        "location": metadata.get("location", ""),
        "retrieved_text": chunk_texts.get(chunk_id, ""),
    }


def search(
    query: str,
    dataset: str,
    top_k: int = 10,
    *,
    indexes_dir: Path = DEFAULT_INDEXES_DIR,
    chunks_dir: Path = DEFAULT_CHUNKS_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    allow_model_download: bool = False,
) -> list[dict[str, Any]]:
    """
    Search one dataset with FAISS semantic retrieval.

    Args:
        query: Natural-language query text.
        dataset: Target dataset: profiles, demands, or jd.
        top_k: Number of nearest semantic neighbors to return.

    Returns:
        A list of dictionaries containing rank, chunk_id, similarity_score,
        title, source, chunk_type, location, and retrieved_text.
    """
    cleaned_query = validate_search_inputs(query=query, top_k=top_k)
    dataset_config = get_dataset_config(dataset)

    if os.getenv("HYBRIDMIND_USE_FAISS", "0").strip().lower() not in {"1", "true", "yes"}:
        from graph.local_store import semantic_like_search
        return semantic_like_search(dataset_config.name, cleaned_query, top_k=top_k)

    resolved_indexes_dir = indexes_dir.resolve()
    resolved_chunks_dir = chunks_dir.resolve()

    try:
        index = load_index(resolved_indexes_dir, dataset_config)
        mapping = load_mapping(resolved_indexes_dir, dataset_config)
        chunk_texts = load_chunk_texts(resolved_chunks_dir, dataset_config)

        if index.ntotal != len(mapping):
            raise ValueError(
                f"Index/mapping mismatch for {dataset_config.name}: "
                f"{index.ntotal} indexed vectors but {len(mapping)} mapping rows"
            )

        model = load_embedding_model(
            model_name=model_name,
            allow_model_download=allow_model_download,
        )
        query_vector = encode_query(model, cleaned_query)

        effective_top_k = min(top_k, int(index.ntotal))
        scores, ids = index.search(query_vector, effective_top_k)

        results: list[dict[str, Any]] = []
        for rank, (score, faiss_id) in enumerate(zip(scores[0], ids[0]), start=1):
            if int(faiss_id) < 0:
                continue

            results.append(
                format_result(
                    rank=rank,
                    faiss_id=int(faiss_id),
                    score=float(score),
                    mapping=mapping,
                    chunk_texts=chunk_texts,
                )
            )

        return results
    except Exception as error:
        print(f"[semantic] FAISS semantic search unavailable, using local semantic fallback: {error}", file=sys.stderr)
        from graph.local_store import semantic_like_search
        return semantic_like_search(dataset_config.name, cleaned_query, top_k=top_k)


def print_results(results: list[dict[str, Any]]) -> None:
    """Print semantic search results in a readable CLI format."""
    if not results:
        print("No results found.")
        return

    for result in results:
        print("-" * 80)
        print(f"Rank : {result['rank']}")
        print(f"Chunk ID : {result['chunk_id']}")
        print(f"Similarity Score : {result['similarity_score']:.6f}")
        print(f"Title : {result['title']}")
        print(f"Source : {result['source']}")
        print(f"Chunk Type : {result['chunk_type']}")
        print(f"Location : {result['location']}")
        print("Retrieved Text :")
        print(result["retrieved_text"])


def main() -> int:
    """Run FAISS semantic search from the command line."""
    args = parse_args()
    results = search(
        query=args.query,
        dataset=args.dataset,
        top_k=args.top_k,
        indexes_dir=args.indexes_dir,
        chunks_dir=args.chunks_dir,
        model_name=args.model_name,
        allow_model_download=args.allow_model_download,
    )
    print_results(results)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nSemantic search failed: {error}", file=sys.stderr)
        raise SystemExit(1)
