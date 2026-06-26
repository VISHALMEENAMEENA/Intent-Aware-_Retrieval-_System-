"""
reranker/cross_encoder.py
=========================
Cross-Encoder Re-ranking Module.

Uses a cross-encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2) to re-score
(query, document) pairs. Unlike bi-encoders used in FAISS semantic search,
cross-encoders attend jointly to both inputs and produce much more precise
relevance scores.

Architecture position:
    Graph-Aware Ranking results
              │
              ▼
    Cross-Encoder Reranker  ← computes score(query, retrieved_text) for each candidate
              │
              ▼
    Final Top-K Results (sorted by cross-encoder score)

Performance note:
    Cross-encoders are slower than bi-encoders but more accurate. They are applied
    to a small candidate pool (top-K from graph ranking, not the full corpus).
    Lazy model loading ensures the model is only downloaded/loaded once per process.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Lazy model loading ─────────────────────────────────────────────────────────
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder = None


def _get_model():
    """Load the cross-encoder model on first use (singleton)."""
    global _cross_encoder  # noqa: PLW0603
    if _cross_encoder is None:
        from graph.config import ENABLE_CROSS_ENCODER
        if not ENABLE_CROSS_ENCODER:
            raise RuntimeError(
                "Cross-encoder is disabled by default to avoid model download/load delays. "
                "Set HYBRIDMIND_USE_CROSS_ENCODER=1 to enable it."
            )
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
            _cross_encoder = CrossEncoder(_MODEL_NAME, max_length=512)
        except ImportError as exc:
            raise RuntimeError(
                "sentence_transformers is required for cross-encoder reranking. "
                "Run: pip install sentence-transformers"
            ) from exc
    return _cross_encoder


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_document_text(result: dict[str, Any]) -> str:
    """
    Build a single text string from the most informative fields of a candidate.

    Priority order: retrieved_text → text → title → highlighted_text.
    Falls back to joining all non-empty string values if nothing found.
    """
    # Try richest text fields first
    for field in ("retrieved_text", "text", "highlighted_text"):
        val = str(result.get(field) or "").strip()
        if val:
            return val[:1500]  # Truncate to avoid exceeding model max length

    # Fallback: join title and source metadata
    parts = [
        str(result.get("title") or ""),
        str(result.get("source") or ""),
        str(result.get("industry") or ""),
        str(result.get("location") or ""),
    ]
    return " | ".join(p for p in parts if p)[:512]


# ── Public API ─────────────────────────────────────────────────────────────────

def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Re-rank candidates using a cross-encoder model.

    Scores each (query, document) pair jointly using a cross-encoder
    and returns candidates sorted by cross-encoder score descending.

    Attaches to each candidate:
        cross_encoder_score  float  – raw cross-encoder logit score (higher = more relevant)
        cross_encoder_rank   int    – 1-based rank after cross-encoder re-ranking

    Args:
        query:       The original user query (or expanded query).
        candidates:  Candidate list from graph ranking.
        top_k:       If provided, return only top_k results. Otherwise return all.

    Returns:
        Re-ranked list of result dicts (best first).
    """
    if not candidates:
        return []

    model = _get_model()

    # Build (query, doc_text) pairs for batch scoring
    pairs = [(query, _build_document_text(c)) for c in candidates]

    # Score all pairs in one batch call for efficiency
    scores: list[float] = model.predict(pairs, show_progress_bar=False).tolist()

    from graph.config import CROSS_ENCODER_WEIGHT

    # Attach scores to candidates
    for candidate, score in zip(candidates, scores):
        raw_score = float(score)
        normalized_score = 1.0 / (1.0 + pow(2.718281828, -raw_score))
        candidate["cross_encoder_score"] = round(raw_score, 6)
        candidate["cross_encoder_normalized_score"] = round(normalized_score, 6)
        candidate["pre_cross_encoder_score"] = candidate.get("final_score", 0.0)
        candidate["final_score"] = round(
            float(candidate.get("final_score") or 0.0)
            + CROSS_ENCODER_WEIGHT * normalized_score,
            6,
        )
        candidate["final_combined_score"] = candidate["final_score"]

    # Sort by cross-encoder score descending
    ranked = sorted(candidates, key=lambda r: r["final_score"], reverse=True)

    # Assign 1-based ranks
    for idx, result in enumerate(ranked, start=1):
        result["cross_encoder_rank"] = idx

    if top_k is not None:
        ranked = ranked[:top_k]

    return ranked


def rerank_safe(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Safe wrapper around rerank() that falls back to the current order
    if cross-encoder loading or scoring fails.

    Always attaches cross_encoder_score (0.0 on failure) and cross_encoder_rank
    so that downstream consumers don't need to handle missing keys.

    Args:
        query:       User query string.
        candidates:  Candidate list.
        top_k:       Desired final result count.

    Returns:
        Re-ranked candidates (or original order if model fails).
    """
    try:
        return rerank(query, candidates, top_k=top_k)
    except Exception as exc:
        print(f"[reranker] Cross-encoder reranking failed, using prior ranking: {exc}",
              file=sys.stderr)
        # Attach placeholder values so downstream code doesn't break
        for idx, c in enumerate(candidates, start=1):
            c.setdefault("cross_encoder_score", 0.0)
            c.setdefault("cross_encoder_rank", idx)
        if top_k is not None:
            return candidates[:top_k]
        return candidates
