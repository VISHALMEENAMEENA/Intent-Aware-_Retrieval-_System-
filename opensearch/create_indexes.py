"""
create_indexes.py
=================
Create intent-aware OpenSearch indexes for BM25 lexical retrieval.

This script creates three independent indexes:
    - profiles_index
    - demands_index
    - jd_index

No vector fields, embeddings, k-NN, FAISS, or retrieval logic are included here.
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


INDEX_NAMES = {
    "profiles": "profiles_index",
    "demands": "demands_index",
    "jd": "jd_index",
}


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
    """Parse OpenSearch connection and index settings."""
    parser = argparse.ArgumentParser(
        description="Create BM25-only OpenSearch indexes for semantic chunks."
    )
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
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--replicas", type=int, default=0)
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


def build_index_body(shards: int, replicas: int) -> dict[str, Any]:
    """Return a BM25-only index configuration with standard text analysis."""
    return {
        "settings": {
            "index": {
                "number_of_shards": shards,
                "number_of_replicas": replicas,
                "similarity": {
                    "default": {
                        "type": "BM25",
                    }
                },
            },
            "analysis": {
                "analyzer": {
                    "standard_analyzer": {
                        "type": "standard",
                    }
                }
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "chunk_id": {"type": "keyword"},
                "parent_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "chunk_type": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                },
                "location": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                },
                "industry": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                },
                "text": {"type": "text", "analyzer": "standard"},
            },
        },
    }


def create_index_if_missing(
    client: Any,
    index_name: str,
    index_body: dict[str, Any],
) -> bool:
    """Create one index if it does not already exist."""
    if client.indices.exists(index=index_name):
        print(f"Index already exists : {index_name}")
        return False

    client.indices.create(index=index_name, body=index_body)
    print(f"Index created        : {index_name}")
    return True


def create_all_indexes(client: Any, shards: int, replicas: int) -> dict[str, bool]:
    """Create the three intent-aware BM25 indexes."""
    index_body = build_index_body(shards=shards, replicas=replicas)
    creation_status = {}

    for index_name in INDEX_NAMES.values():
        creation_status[index_name] = create_index_if_missing(
            client=client,
            index_name=index_name,
            index_body=index_body,
        )

    return creation_status


def main() -> int:
    """Create OpenSearch indexes and print a concise summary."""
    args = parse_args()
    config = build_config(args)

    try:
        client = create_client(config)
        creation_status = create_all_indexes(
            client=client,
            shards=args.shards,
            replicas=args.replicas,
        )
    except (ImportError, ConnectionError, OpenSearchException) as error:
        print(f"Index creation failed: {error}", file=sys.stderr)
        return 1

    created_count = sum(1 for created in creation_status.values() if created)
    existing_count = len(creation_status) - created_count

    print("\nOpenSearch index setup complete")
    print("--------------------------------")
    print(f"Created indexes : {created_count}")
    print(f"Existing indexes: {existing_count}")
    print("Vector fields   : none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
