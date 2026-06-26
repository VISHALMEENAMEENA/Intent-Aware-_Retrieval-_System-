"""
bm25_search.py
==============
Reusable BM25 lexical search over intent-specific OpenSearch indexes.

This script does not use embeddings, vector search, k-NN, FAISS, RRF, or
reranking. It is a pure OpenSearch BM25 retrieval utility.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any

try:
    from opensearchpy import OpenSearch
    from opensearchpy.exceptions import OpenSearchException
except ImportError:  # Dependency is checked at runtime so --help still works.
    OpenSearch = None
    OpenSearchException = Exception

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


DEFAULT_INDEXES = ["profiles_index", "demands_index", "jd_index"]
RESULT_FIELDS = [
    "chunk_id",
    "title",
    "source",
    "chunk_type",
    "location",
    "industry",
    "text",
]

HIGHLIGHT_FIELDS = ["text", "title", "location", "industry"]


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


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    """Parse boolean CLI/environment values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    """Parse BM25 search CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run BM25 lexical search against one OpenSearch index."
    )
    parser.add_argument("--query", required=True, help="Search query text.")
    parser.add_argument(
        "--index-name",
        required=True,
        choices=DEFAULT_INDEXES,
        help="Intent-specific OpenSearch index to search.",
    )
    parser.add_argument("--top-k", type=int, default=10)
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


def get_default_client() -> Any:
    """Create a client from environment variables for programmatic search()."""
    config = OpenSearchConfig(
        host=os.getenv("OPENSEARCH_HOST", "localhost"),
        port=int(os.getenv("OPENSEARCH_PORT", "9200")),
        username=os.getenv("OPENSEARCH_USER"),
        password=os.getenv("OPENSEARCH_PASSWORD"),
        use_ssl=parse_bool(os.getenv("OPENSEARCH_USE_SSL"), False),
        verify_certs=parse_bool(os.getenv("OPENSEARCH_VERIFY_CERTS"), False),
        timeout=int(os.getenv("OPENSEARCH_TIMEOUT", "30")),
    )
    return create_client(config)


def build_bm25_query(query: str, top_k: int) -> dict[str, Any]:
    """Build a pure BM25 lexical query with OpenSearch highlight snippets."""
    return {
        "size": top_k,
        "_source": RESULT_FIELDS,
        "query": {
            "multi_match": {
                "query": query,
                "fields": [
                    "text^3",
                    "title^2",
                    "location",
                    "industry",
                    "chunk_type",
                ],
                "type": "best_fields",
                "operator": "or",
            }
        },
        "highlight": {
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
            "require_field_match": False,
            "fields": {
                "text": {
                    "fragment_size": 240,
                    "number_of_fragments": 3,
                },
                "title": {
                    "number_of_fragments": 0,
                },
                "location": {
                    "number_of_fragments": 0,
                },
                "industry": {
                    "number_of_fragments": 0,
                },
            },
        },
    }


def extract_highlighted_text(hit: dict[str, Any]) -> str:
    """
    Return the best OpenSearch highlight snippet for explainable debugging.

    Text highlights are preferred because they show why the chunk matched.
    If the query matched metadata fields such as title or location, those
    snippets are still useful and are returned as a fallback.
    """
    highlights = hit.get("highlight", {})

    for field in HIGHLIGHT_FIELDS:
        snippets = highlights.get(field)
        if snippets:
            return " ... ".join(snippets)

    source = hit.get("_source", {})
    return truncate_text(source.get("text", ""))


def format_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Convert one OpenSearch hit into the retrieval result schema."""
    source = hit.get("_source", {})
    return {
        "chunk_id": source.get("chunk_id", hit.get("_id", "")),
        "score": float(hit.get("_score", 0.0)),
        "matched_index": hit.get("_index", ""),
        "title": source.get("title", ""),
        "source": source.get("source", ""),
        "chunk_type": source.get("chunk_type", ""),
        "matched_chunk_type": source.get("chunk_type", ""),
        "location": source.get("location", ""),
        "text": source.get("text", ""),
        "highlighted_text": extract_highlighted_text(hit),
    }


def search(query: str, index_name: str, top_k: int = 10) -> list[dict[str, Any]]:
    """
    Search one intent-specific OpenSearch index with BM25.

    Args:
        query: User search query.
        index_name: Target index, e.g. profiles_index, demands_index, or jd_index.
        top_k: Number of results to return.

    Returns:
        A list of result dictionaries sorted by BM25 score descending.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    dataset_by_index = {
        "profiles_index": "profiles",
        "demands_index": "demands",
        "jd_index": "jd",
    }

    if OpenSearch is None:
        from graph.local_store import bm25_like_search
        return bm25_like_search(dataset_by_index[index_name], query, top_k=top_k)

    try:
        client = get_default_client()

        if not client.indices.exists(index=index_name):
            raise ValueError(f"Index '{index_name}' does not exist. Run create_indexes.py and index_documents.py first.")

        response = client.search(
            index=index_name,
            body=build_bm25_query(query=query.strip(), top_k=top_k),
        )

        hits = response.get("hits", {}).get("hits", [])
        return [format_hit(hit) for hit in hits]
    except Exception as error:
        print(f"[bm25] OpenSearch unavailable, using local BM25 fallback: {error}", file=sys.stderr)
        from graph.local_store import bm25_like_search
        return bm25_like_search(dataset_by_index[index_name], query, top_k=top_k)


def search_with_client(
    client: Any,
    query: str,
    index_name: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Search with an explicit client, useful for services and tests."""
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not client.indices.exists(index=index_name):
        raise ValueError(f"Index '{index_name}' does not exist. Run create_indexes.py and index_documents.py first.")

    response = client.search(
        index=index_name,
        body=build_bm25_query(query=query.strip(), top_k=top_k),
    )

    hits = response.get("hits", {}).get("hits", [])
    return [format_hit(hit) for hit in hits]


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Shorten long chunk text for terminal display only."""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned

    return cleaned[: max_chars - 3].rstrip() + "..."


def pretty_print_results(results: list[dict[str, Any]]) -> None:
    """Pretty-print BM25 results for manual testing."""
    if not results:
        print("No BM25 results found.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print("-" * 60)
        print(f"Chunk ID           : {result['chunk_id']}")
        print(f"BM25 Score         : {result['score']:.4f}")
        print(f"Matched Index      : {result['matched_index']}")
        print(f"Matched Chunk Type : {result['matched_chunk_type']}")
        print(f"Title              : {result['title']}")
        print(f"Highlighted Text   : {truncate_text(result['highlighted_text'])}")


def main() -> int:
    """Run BM25 search from the command line."""
    args = parse_args()

    try:
        client = create_client(build_config(args))
        results = search_with_client(
            client=client,
            query=args.query,
            index_name=args.index_name,
            top_k=args.top_k,
        )
        pretty_print_results(results)
    except (ImportError, ConnectionError, ValueError, OpenSearchException) as error:
        print(f"BM25 search failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
