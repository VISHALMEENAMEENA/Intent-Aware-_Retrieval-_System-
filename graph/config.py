"""
graph/config.py
===============
Configuration for the Knowledge Graph module.
Loads Neo4j credentials from api/neo4j.env and exposes
all tunable constants for traversal, scoring, and expansion.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEO4J_ENV_PATH = PROJECT_ROOT / "api" / "neo4j.env"

if NEO4J_ENV_PATH.exists():
    load_dotenv(dotenv_path=NEO4J_ENV_PATH)

# ── Neo4j connection ───────────────────────────────────────────────────────────
NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

# ── Data paths ─────────────────────────────────────────────────────────────────
PROFILES_CSV = PROJECT_ROOT / "data" / "cleaned" / "profiles_cleaned.csv"
DEMANDS_CSV  = PROJECT_ROOT / "data" / "cleaned" / "demands_cleaned.csv"
JD_CSV       = PROJECT_ROOT / "data" / "cleaned" / "jd_cleaned.csv"

# ── Graph expansion ────────────────────────────────────────────────────────────
DEFAULT_EXPANSION_DEPTH: int = 2      # max ontology hops
DEFAULT_EXPANSION_LIMIT: int = 5      # max expanded entities returned
MIN_COOCCURRENCE_WEIGHT: int = 2      # legacy only; semantic graph uses confidence
MIN_EDGE_CONFIDENCE: float = 0.60

# ── Scoring weights ────────────────────────────────────────────────────────────
RRF_WEIGHT: float = 0.65              # weight assigned to RRF score
GRAPH_WEIGHT: float = 0.20           # graph must not dominate retrieval
CROSS_ENCODER_WEIGHT: float = 0.15

# ── Retrieval tuning ───────────────────────────────────────────────────────────
RETRIEVAL_BM25_K: int = 50            # BM25 candidates to retrieve
RETRIEVAL_SEMANTIC_K: int = 50        # Semantic candidates to retrieve
RETRIEVAL_RRF_K: int = 60             # RRF smoothing constant
RETRIEVAL_FUSED_K: int = 50           # Number of fused results to pass to graph ranking

# ── Builder batching ───────────────────────────────────────────────────────────
BATCH_SIZE: int = 200                 # Cypher UNWIND batch size for performance

# ── Cross-encoder reranker ─────────────────────────────────────────────────────
CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_POOL_SIZE: int = 20     # Max candidates fed into the cross-encoder
FINAL_TOP_K: int = 10                 # Default number of final results to return
ENABLE_CROSS_ENCODER: bool = os.getenv(
    "HYBRIDMIND_USE_CROSS_ENCODER", "0"
).strip().lower() in {"1", "true", "yes"}

# ── Score fusion weights for the unified pipeline ──────────────────────────────
# Final score = GRAPH_REL_WEIGHT * graph_rel_score + RRF_WEIGHT * rrf_score
# Cross-encoder score replaces this when the reranker is active
GRAPH_REL_WEIGHT: float = 0.35       # relationship confidence component
GRAPH_TEXT_WEIGHT: float = 0.65      # ontology/entity coverage component


def validate_config() -> None:
    """Raise if any required credential is missing."""
    missing = [k for k, v in {
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USER": NEO4J_USER,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
    }.items() if not v]
    if missing:
        raise ValueError(
            f"Missing Neo4j config: {', '.join(missing)}. "
            f"Check {NEO4J_ENV_PATH}"
        )
