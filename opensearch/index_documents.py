"""
index_documents.py
==================
Bulk index semantic chunk CSV files into intent-specific OpenSearch indexes.

This script stores only lexical BM25 documents:
    chunk_id, parent_id, source, chunk_type, title, location, industry, text

It does not read or store embeddings, vectors, k-NN fields, or FAISS artifacts.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from opensearchpy import OpenSearch, helpers
    from opensearchpy.exceptions import OpenSearchException
except ImportError:  # Dependency is checked at runtime so --help still works.
    OpenSearch = None
    helpers = None
    OpenSearchException = Exception


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"

DOCUMENT_COLUMNS = [
    "chunk_id",
    "parent_id",
    "source",
    "chunk_type",
    "title",
    "location",
    "industry",
    "text",
]


@dataclass(frozen=True)
class OpenSearchConfig:
    """Connection settings for an OpenSearch cluster."""

    host: str
    port: int
    username: str | None
    password: str | None
    use_ssl: bool
    verify_certs: bool
    timeout: int


@dataclass(frozen=True)
class IndexJob:
    """Configuration for one CSV-to-index bulk indexing job."""

    display_name: str
    input_filename: str
    index_name: str
    stats_label: str


INDEX_JOBS = [
    IndexJob(
        display_name="Profiles",
        input_filename="profiles_chunks.csv",
        index_name="profiles_index",
        stats_label="Profiles Indexed",
    ),
    IndexJob(
        display_name="Demands",
        input_filename="demands_chunks.csv",
        index_name="demands_index",
        stats_label="Demands Indexed",
    ),
    IndexJob(
        display_name="JD",
        input_filename="jd_chunks.csv",
        index_name="jd_index",
        stats_label="JD Indexed",
    ),
]


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    """Parse boolean CLI/environment values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    """Parse OpenSearch connection and indexing settings."""
    parser = argparse.ArgumentParser(
        description="Bulk index chunk CSV files into BM25-only OpenSearch indexes."
    )
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--host", default=os.getenv("OPENSEARCH_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("OPENSEARCH_PORT", "9200")))
    parser.add_argument("--username", default=os.getenv("OPENSEARCH_USER"))
    parser.add_argument("--password", default=os.getenv("OPENSEARCH_PASSWORD"))
    parser.add_argument(
        "--use-ssl",
        action="store_true",
        default=parse_bool(os.getenv("OPENSEARCH_USE_SSL"), False),
        help="Use HTTPS when connecting to OpenSearch.",
    )
    parser.add_argument(
        "--verify-certs",
        action="store_true",
        default=parse_bool(os.getenv("OPENSEARCH_VERIFY_CERTS"), False),
        help="Verify TLS certificates.",
    )
    parser.add_argument("--timeout", type=int, default=int(os.getenv("OPENSEARCH_TIMEOUT", "30")))
    parser.add_argument("--bulk-size", type=int, default=500)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh each index after bulk indexing so documents are immediately searchable.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> OpenSearchConfig:
    """Build a typed config object from parsed arguments."""
    return OpenSearchConfig(
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        use_ssl=args.use_ssl,
        verify_certs=args.verify_certs,
        timeout=args.timeout,
    )


def create_client(config: OpenSearchConfig) -> Any:
    """Create and validate an OpenSearch client connection."""
    if OpenSearch is None:
        raise ImportError("Missing dependency: install it with `pip install opensearch-py`.")

    auth = None
    if config.username and config.password:
        auth = (config.username, config.password)

    client = OpenSearch(
        hosts=[{"host": config.host, "port": config.port}],
        http_auth=auth,
        use_ssl=config.use_ssl,
        verify_certs=config.verify_certs,
        ssl_show_warn=False,
        timeout=config.timeout,
        max_retries=3,
        retry_on_timeout=True,
    )

    try:
        client.info()
    except Exception as error:
        raise ConnectionError(
            f"Could not connect to OpenSearch at {config.host}:{config.port}. "
            "Check that the cluster is running and the credentials are correct."
        ) from error

    return client


def validate_bulk_size(bulk_size: int) -> None:
    """Validate bulk chunk size before indexing starts."""
    if bulk_size <= 0:
        raise ValueError("--bulk-size must be a positive integer")


def ensure_index_exists(client: Any, index_name: str) -> None:
    """Fail early if the target index was not created yet."""
    if not client.indices.exists(index=index_name):
        raise ValueError(
            f"Index '{index_name}' does not exist. Run opensearch/create_indexes.py first."
        )


def load_chunk_documents(chunks_dir: Path, job: IndexJob) -> pd.DataFrame:
    """Load and validate one chunk CSV for lexical indexing."""
    input_path = chunks_dir / job.input_filename
    if not input_path.exists():
        raise FileNotFoundError(f"Missing chunk file: {input_path}")

    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    missing_columns = [column for column in DOCUMENT_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{input_path} is missing required columns: {missing}")

    df = df.loc[:, DOCUMENT_COLUMNS].fillna("")
    if df.empty:
        raise ValueError(f"{input_path} has no rows to index")

    if df["chunk_id"].duplicated().any():
        duplicate_count = int(df["chunk_id"].duplicated().sum())
        raise ValueError(f"{input_path} contains {duplicate_count} duplicate chunk_id values")

    empty_text_count = int((df["text"].str.strip() == "").sum())
    if empty_text_count:
        raise ValueError(f"{input_path} contains {empty_text_count} rows with empty text")

    return df


def iter_bulk_actions(df: pd.DataFrame, index_name: str) -> Iterable[dict[str, Any]]:
    """Yield OpenSearch bulk actions for one dataframe."""
    for record in df.to_dict(orient="records"):
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": record["chunk_id"],
            "_source": record,
        }


def bulk_index_dataframe(
    client: Any,
    df: pd.DataFrame,
    index_name: str,
    bulk_size: int,
) -> tuple[int, int]:
    """Bulk index a dataframe and return success/failure counts."""
    if helpers is None:
        raise ImportError("Missing dependency: install it with `pip install opensearch-py`.")

    success_count = 0
    failure_count = 0

    for ok, item in helpers.streaming_bulk(
        client=client,
        actions=iter_bulk_actions(df, index_name),
        chunk_size=bulk_size,
        raise_on_error=False,
        raise_on_exception=False,
    ):
        if ok:
            success_count += 1
        else:
            failure_count += 1
            if failure_count <= 5:
                print(f"Indexing failure sample: {item}", file=sys.stderr)

    return success_count, failure_count


def index_job(
    client: Any,
    chunks_dir: Path,
    job: IndexJob,
    bulk_size: int,
    refresh: bool,
) -> int:
    """Run one CSV-to-OpenSearch indexing job."""
    ensure_index_exists(client, job.index_name)
    df = load_chunk_documents(chunks_dir, job)

    print(f"Indexing {job.display_name}: {len(df)} chunks -> {job.index_name}")
    success_count, failure_count = bulk_index_dataframe(
        client=client,
        df=df,
        index_name=job.index_name,
        bulk_size=bulk_size,
    )

    if failure_count:
        raise RuntimeError(
            f"{job.display_name} indexing completed with {failure_count} failed documents"
        )

    if success_count != len(df):
        raise RuntimeError(
            f"{job.display_name} indexing mismatch: {success_count} indexed for {len(df)} rows"
        )

    if refresh:
        client.indices.refresh(index=job.index_name)

    return success_count


def print_statistics(counts: dict[str, int]) -> None:
    """Print indexing statistics in the requested format."""
    print("\nOpenSearch BM25 indexing complete")
    print("---------------------------------")
    for job in INDEX_JOBS:
        print(f"{job.stats_label} : {counts.get(job.display_name, 0)}")


def main() -> int:
    """Index profiles, demands, and JD chunks into separate BM25 indexes."""
    args = parse_args()
    chunks_dir = args.chunks_dir.resolve()

    try:
        validate_bulk_size(args.bulk_size)
        client = create_client(build_config(args))

        counts = {}
        for job in INDEX_JOBS:
            counts[job.display_name] = index_job(
                client=client,
                chunks_dir=chunks_dir,
                job=job,
                bulk_size=args.bulk_size,
                refresh=args.refresh,
            )

        print_statistics(counts)
    except (ImportError, ConnectionError, FileNotFoundError, ValueError, RuntimeError, OpenSearchException) as error:
        print(f"Document indexing failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
