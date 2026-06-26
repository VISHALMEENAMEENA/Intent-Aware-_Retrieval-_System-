"""
rrf.py
======
Intent-Aware and Explainable Hybrid Retrieval System
----------------------------------------------------
Merge BM25 and FAISS semantic retrieval results with Reciprocal Rank Fusion.

This module only performs hybrid retrieval fusion:
    1. Accept a query and manually provided intent
    2. Map intent to the correct BM25 index and FAISS dataset
    3. Retrieve BM25 and semantic candidates
    4. Merge duplicate chunk_ids using RRF
    5. Return a final ranked list

It does not implement intent detection, knowledge graphs, reranking, LLM calls,
response generation, or explainability.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENSEARCH_DIR = PROJECT_ROOT / "opensearch"
FAISS_DIR = PROJECT_ROOT / "faiss"
DEFAULT_CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"

for module_dir in (OPENSEARCH_DIR, FAISS_DIR):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

try:
    import bm25_search
    import semantic_search
except ImportError as error:
    raise ImportError(
        "Could not import existing BM25/semantic search modules. "
        "Expected opensearch/bm25_search.py and faiss/semantic_search.py."
    ) from error


CHUNK_METADATA_COLUMNS = [
    "chunk_id",
    "parent_id",
    "source",
    "chunk_type",
    "title",
    "location",
    "industry",
    "text",
]

FINAL_RESULT_FIELDS = [
    "rank",
    "chunk_id",
    "parent_id",
    "source",
    "chunk_type",
    "title",
    "location",
    "industry",
    "bm25_rank",
    "semantic_rank",
    "bm25_score",
    "semantic_score",
    "rrf_score",
    "retrieved_text",
]


@dataclass(frozen=True)
class IntentConfig:
    """Manual intent routing for the current pre-intent-detection pipeline."""

    intent: str
    dataset: str
    index_name: str
    chunk_filename: str


INTENT_CONFIGS = {
    "profile_search": IntentConfig(
        intent="profile_search",
        dataset="profiles",
        index_name="profiles_index",
        chunk_filename="profiles_chunks.csv",
    ),
    "job_search": IntentConfig(
        intent="job_search",
        dataset="demands",
        index_name="demands_index",
        chunk_filename="demands_chunks.csv",
    ),
    "jd_search": IntentConfig(
        intent="jd_search",
        dataset="jd",
        index_name="jd_index",
        chunk_filename="jd_chunks.csv",
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse RRF command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run hybrid retrieval by fusing BM25 and FAISS results with RRF."
    )
    parser.add_argument(
        "--intent",
        required=True,
        choices=sorted(INTENT_CONFIGS),
        help="Manual retrieval intent. Dataset/index routing is handled internally.",
    )
    parser.add_argument("--query", required=True, help="Search query text.")
    parser.add_argument("--top-k", type=int, default=10, help="Final fused result count.")
    parser.add_argument("--bm25-k", type=int, default=20, help="BM25 candidates to retrieve.")
    parser.add_argument(
        "--semantic-k",
        type=int,
        default=20,
        help="Semantic candidates to retrieve.",
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant.")
    return parser.parse_args()


def get_intent_config(intent: str) -> IntentConfig:
    """Map a manually provided intent to the correct retrieval backends."""
    intent_key = intent.strip().lower()
    if intent_key not in INTENT_CONFIGS:
        supported = ", ".join(sorted(INTENT_CONFIGS))
        raise ValueError(f"intent must be one of: {supported}")

    return INTENT_CONFIGS[intent_key]


def validate_search_inputs(
    query: str,
    top_k: int,
    bm25_k: int,
    semantic_k: int,
    rrf_k: int,
) -> str:
    """Validate public search arguments and return a cleaned query."""
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query must be a non-empty string")

    for name, value in (
        ("top_k", top_k),
        ("bm25_k", bm25_k),
        ("semantic_k", semantic_k),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if rrf_k < 0:
        raise ValueError("rrf_k must be zero or a positive integer")

    return cleaned_query


def load_chunk_metadata(
    config: IntentConfig,
    chunks_dir: Path = DEFAULT_CHUNKS_DIR,
) -> dict[str, dict[str, str]]:
    """Load chunk metadata and text for final result enrichment."""
    chunk_path = chunks_dir / config.chunk_filename
    if not chunk_path.exists():
        raise FileNotFoundError(f"Missing chunk metadata file: {chunk_path}")

    with chunk_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        missing_columns = [
            column for column in CHUNK_METADATA_COLUMNS if column not in fieldnames
        ]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"{chunk_path} is missing required columns: {missing}")

        metadata_by_chunk_id: dict[str, dict[str, str]] = {}
        duplicate_ids: set[str] = set()

        for row in reader:
            chunk_id = row.get("chunk_id", "")
            if not chunk_id:
                continue

            if chunk_id in metadata_by_chunk_id:
                duplicate_ids.add(chunk_id)
                continue

            metadata_by_chunk_id[chunk_id] = {
                column: row.get(column, "") for column in CHUNK_METADATA_COLUMNS
            }

    if duplicate_ids:
        preview = ", ".join(sorted(duplicate_ids)[:10])
        raise ValueError(f"{chunk_path} contains duplicate chunk_id values: {preview}")

    return metadata_by_chunk_id


def rrf_contribution(rank: int, rrf_k: int) -> float:
    """Calculate the Reciprocal Rank Fusion contribution for one source rank."""
    return 1.0 / (rrf_k + rank)


def first_present(*values: Any) -> Any:
    """Return the first value that is neither None nor an empty string."""
    for value in values:
        if value is not None and value != "":
            return value

    return None


def optional_float(value: Any) -> float | None:
    """Convert a score-like value to float while preserving missing values."""
    if value is None or value == "":
        return None

    return float(value)


def extract_chunk_id(result: dict[str, Any]) -> str:
    """Return a result chunk_id or fail with a clear message."""
    chunk_id = str(result.get("chunk_id", "")).strip()
    if not chunk_id:
        raise ValueError(f"retrieval result is missing chunk_id: {result}")

    return chunk_id


def empty_fusion_row(chunk_id: str) -> dict[str, Any]:
    """Create the internal accumulator row for one chunk_id."""
    row = {field: None for field in FINAL_RESULT_FIELDS}
    row.update(
        {
            "rank": 0,
            "chunk_id": chunk_id,
            "rrf_score": 0.0,
            "_source_count": 0,
            "_best_source_rank": sys.maxsize,
        }
    )
    return row


def merge_metadata(
    row: dict[str, Any],
    *,
    result: dict[str, Any],
    chunk_metadata: dict[str, str] | None,
) -> None:
    """Merge result metadata and chunk CSV metadata into a fusion row."""
    metadata = chunk_metadata or {}

    for field in ("parent_id", "source", "chunk_type", "title", "location", "industry"):
        row[field] = first_present(row.get(field), result.get(field), metadata.get(field))

    row["retrieved_text"] = first_present(
        row.get("retrieved_text"),
        result.get("retrieved_text"),
        result.get("text"),
        metadata.get("text"),
    )


def apply_bm25_results(
    fused_by_chunk_id: dict[str, dict[str, Any]],
    bm25_results: Iterable[dict[str, Any]],
    metadata_by_chunk_id: dict[str, dict[str, str]],
    rrf_k: int,
) -> None:
    """Add BM25 ranks, scores, and RRF contributions to the fusion map."""
    seen_chunk_ids: set[str] = set()

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = extract_chunk_id(result)
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)

        row = fused_by_chunk_id.setdefault(chunk_id, empty_fusion_row(chunk_id))
        row["bm25_rank"] = rank
        row["bm25_score"] = optional_float(result.get("score"))
        row["rrf_score"] += rrf_contribution(rank=rank, rrf_k=rrf_k)
        row["_source_count"] += 1
        row["_best_source_rank"] = min(row["_best_source_rank"], rank)
        merge_metadata(
            row,
            result=result,
            chunk_metadata=metadata_by_chunk_id.get(chunk_id),
        )


def apply_semantic_results(
    fused_by_chunk_id: dict[str, dict[str, Any]],
    semantic_results: Iterable[dict[str, Any]],
    metadata_by_chunk_id: dict[str, dict[str, str]],
    rrf_k: int,
) -> None:
    """Add semantic ranks, scores, and RRF contributions to the fusion map."""
    seen_chunk_ids: set[str] = set()

    for rank, result in enumerate(semantic_results, start=1):
        chunk_id = extract_chunk_id(result)
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)

        row = fused_by_chunk_id.setdefault(chunk_id, empty_fusion_row(chunk_id))
        row["semantic_rank"] = rank
        row["semantic_score"] = optional_float(result.get("similarity_score"))
        row["rrf_score"] += rrf_contribution(rank=rank, rrf_k=rrf_k)
        row["_source_count"] += 1
        row["_best_source_rank"] = min(row["_best_source_rank"], rank)
        merge_metadata(
            row,
            result=result,
            chunk_metadata=metadata_by_chunk_id.get(chunk_id),
        )


def finalize_fused_rows(
    fused_by_chunk_id: dict[str, dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Sort fused rows by RRF score and return the requested final schema."""
    sorted_rows = sorted(
        fused_by_chunk_id.values(),
        key=lambda row: (
            -float(row["rrf_score"]),
            -int(row["_source_count"]),
            int(row["_best_source_rank"]),
            str(row["chunk_id"]),
        ),
    )

    final_results: list[dict[str, Any]] = []
    for rank, row in enumerate(sorted_rows[:top_k], start=1):
        row["rank"] = rank
        final_results.append({field: row.get(field) for field in FINAL_RESULT_FIELDS})

    return final_results


def fuse_results(
    bm25_results: list[dict[str, Any]],
    semantic_results: list[dict[str, Any]],
    *,
    metadata_by_chunk_id: dict[str, dict[str, str]] | None = None,
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse already-retrieved BM25 and semantic results using RRF."""
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if rrf_k < 0:
        raise ValueError("rrf_k must be zero or a positive integer")

    metadata = metadata_by_chunk_id or {}
    fused_by_chunk_id: dict[str, dict[str, Any]] = {}

    apply_bm25_results(
        fused_by_chunk_id=fused_by_chunk_id,
        bm25_results=bm25_results,
        metadata_by_chunk_id=metadata,
        rrf_k=rrf_k,
    )
    apply_semantic_results(
        fused_by_chunk_id=fused_by_chunk_id,
        semantic_results=semantic_results,
        metadata_by_chunk_id=metadata,
        rrf_k=rrf_k,
    )

    return finalize_fused_rows(fused_by_chunk_id=fused_by_chunk_id, top_k=top_k)


def search(
    query: str,
    intent: str,
    top_k: int = 10,
    bm25_k: int = 20,
    semantic_k: int = 20,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """
    Search with BM25 and FAISS semantic retrieval, then fuse with RRF.

    Args:
        query: Natural-language query text.
        intent: Manual intent: profile_search, job_search, or jd_search.
        top_k: Number of final fused results to return.
        bm25_k: Number of BM25 candidates to retrieve before fusion.
        semantic_k: Number of semantic candidates to retrieve before fusion.
        rrf_k: RRF smoothing constant. The standard default is 60.

    Returns:
        Final ranked hybrid retrieval results without duplicate chunk_ids.
    """
    cleaned_query = validate_search_inputs(
        query=query,
        top_k=top_k,
        bm25_k=bm25_k,
        semantic_k=semantic_k,
        rrf_k=rrf_k,
    )
    config = get_intent_config(intent)

    bm25_results = bm25_search.search(
        query=cleaned_query,
        index_name=config.index_name,
        top_k=bm25_k,
    )
    semantic_results = semantic_search.search(
        query=cleaned_query,
        dataset=config.dataset,
        top_k=semantic_k,
    )

    if not bm25_results and not semantic_results:
        return []

    metadata_by_chunk_id = load_chunk_metadata(config)
    return fuse_results(
        bm25_results=bm25_results,
        semantic_results=semantic_results,
        metadata_by_chunk_id=metadata_by_chunk_id,
        top_k=top_k,
        rrf_k=rrf_k,
    )


def format_value(value: Any, decimal_places: int | None = None) -> str:
    """Format CLI values, printing None for missing retrieval-side fields."""
    if value is None:
        return "None"

    if decimal_places is not None:
        return f"{float(value):.{decimal_places}f}"

    return str(value)


def print_results(results: list[dict[str, Any]]) -> None:
    """Print hybrid retrieval results in a readable CLI format."""
    print("=" * 50)
    print("Hybrid Retrieval (RRF)")
    print("=" * 50)

    if not results:
        print("\nNo hybrid results found.")
        return

    for result in results:
        print(f"\nRank : {format_value(result['rank'])}")
        print(f"\nChunk ID :\n{format_value(result['chunk_id'])}")
        print(f"\nParent ID :\n{format_value(result['parent_id'])}")
        print(f"\nBM25 Rank :\n{format_value(result['bm25_rank'])}")
        print(f"\nSemantic Rank :\n{format_value(result['semantic_rank'])}")
        print(f"\nBM25 Score :\n{format_value(result['bm25_score'], 6)}")
        print(f"\nSemantic Score :\n{format_value(result['semantic_score'], 6)}")
        print(f"\nRRF Score :\n{format_value(result['rrf_score'], 6)}")
        print(f"\nTitle :\n{format_value(result['title'])}")
        print(f"\nSource :\n{format_value(result['source'])}")
        print(f"\nChunk Type :\n{format_value(result['chunk_type'])}")
        print(f"\nLocation :\n{format_value(result['location'])}")
        print(f"\nIndustry :\n{format_value(result['industry'])}")
        print("\n" + "-" * 50)
        print("\nRetrieved Text\n")
        print(format_value(result["retrieved_text"]))


def main() -> int:
    """Run hybrid RRF retrieval from the command line."""
    args = parse_args()

    try:
        results = search(
            query=args.query,
            intent=args.intent,
            top_k=args.top_k,
            bm25_k=args.bm25_k,
            semantic_k=args.semantic_k,
            rrf_k=args.rrf_k,
        )
        print_results(results)
    except Exception as error:
        print(f"Hybrid retrieval failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
