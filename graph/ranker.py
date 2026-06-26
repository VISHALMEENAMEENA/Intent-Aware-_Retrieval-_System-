"""
graph/ranker.py
===============
Graph-aware scoring for retrieved candidates.

After BM25 + FAISS → RRF fusion, each result is scored against the
expanded graph context. The final score combines:

    final_score = rrf_weight * rrf_score + graph_weight * graph_score

Graph relevance is computed by measuring how many of the expanded entities
appear in the candidate document's text fields.

All weights are configurable via graph/config.py.
"""

from __future__ import annotations

from typing import Any

from graph.config import (
    GRAPH_REL_WEIGHT,
    GRAPH_TEXT_WEIGHT,
    GRAPH_WEIGHT,
    RETRIEVAL_RRF_K,
    RRF_WEIGHT,
)


# ── Text normalisation ─────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def _document_text(result: dict[str, Any]) -> str:
    """Collect all searchable text fields from a retrieval result."""
    fields = [
        "title", "retrieved_text", "text", "highlighted_text",
        "source", "chunk_type", "location", "industry",
    ]
    parts = [str(result.get(f, "")) for f in fields if result.get(f)]
    return _normalize(" ".join(parts))


# ── Graph relevance score ──────────────────────────────────────────────────────

def compute_graph_score(
    result: dict[str, Any],
    seed_entities: list[str],
    expanded_entities: list[str],
) -> float:
    """
    Score one retrieval result against the expanded graph context.

    Strategy:
    - seed_score = matched_seeds / total_seeds (defaults to 1.0 if no seeds)
    - expanded_score = matched_expanded / total_expanded (defaults to 1.0 if no expanded)
    - Combined score is weighted (0.7 for seeds, 0.3 for expanded) to prevent dilution.

    Returns 0.0 if neither exists.
    """
    if not seed_entities and not expanded_entities:
        return 0.0

    doc_text = _document_text(result)

    matched_seeds = [s for s in seed_entities if _normalize(s) in doc_text]
    matched_expanded = [s for s in expanded_entities if _normalize(s) in doc_text]

    seed_score = len(matched_seeds) / len(seed_entities) if seed_entities else 1.0
    expanded_score = len(matched_expanded) / len(expanded_entities) if expanded_entities else 1.0

    if seed_entities and expanded_entities:
        score = 0.7 * seed_score + 0.3 * expanded_score
    elif seed_entities:
        score = seed_score
    else:
        score = expanded_score

    return round(score, 6)


def combine_scores(
    rrf_score: float,
    graph_score: float,
    rrf_weight: float = RRF_WEIGHT,
    graph_weight: float = GRAPH_WEIGHT,
) -> float:
    """
    Combine RRF and graph relevance scores into a single final score.

    Args:
        rrf_score:    Normalised RRF score (typically 0.01 – 0.03 range).
        graph_score:  Graph relevance score in [0, 1].
        rrf_weight:   Weight for the RRF component (default 0.70).
        graph_weight: Weight for the graph component (default 0.30).

    Returns:
        Combined score. Since RRF scores are already small, they are scaled
        to [0,1] by dividing by a typical max RRF value of 1/60 ≈ 0.0167
        before weighting.
    """
    # Scale RRF score using the two-source max contribution: BM25 rank 1 + semantic rank 1.
    RRF_SCALE = 2.0 / (RETRIEVAL_RRF_K + 1.0)
    normalised_rrf = min(rrf_score / RRF_SCALE, 1.0) if RRF_SCALE > 0 else 0.0

    return round(
        rrf_weight * normalised_rrf + graph_weight * graph_score,
        6,
    )


def rank_results(
    results: list[dict[str, Any]],
    seed_entities: list[str],
    expanded_entities: list[str],
    rrf_weight: float = RRF_WEIGHT,
    graph_weight: float = GRAPH_WEIGHT,
) -> list[dict[str, Any]]:
    """
    Compute graph scores for all results, attach scores, and re-rank.

    Adds to each result:
        graph_score    float
        final_score    float
        matched_seeds      list[str]
        matched_expanded   list[str]

    Returns list sorted by final_score descending.
    """
    for result in results:
        rrf_score = float(result.get("rrf_score") or 0.0)
        doc_text = _document_text(result)

        matched_seeds: list[str] = []
        matched_expanded: list[str] = []

        for entity in seed_entities:
            if _normalize(entity) in doc_text:
                matched_seeds.append(entity)

        for entity in expanded_entities:
            if _normalize(entity) in doc_text:
                matched_expanded.append(entity)

        seed_score = len(matched_seeds) / len(seed_entities) if seed_entities else 1.0
        expanded_score = len(matched_expanded) / len(expanded_entities) if expanded_entities else 1.0

        if seed_entities and expanded_entities:
            graph_text_score = 0.7 * seed_score + 0.3 * expanded_score
        elif seed_entities:
            graph_text_score = seed_score
        elif expanded_entities:
            graph_text_score = expanded_score
        else:
            graph_text_score = 0.0

        graph_rel_score = float(result.get("graph_rel_score") or 0.0)
        graph_score_weight_total = GRAPH_TEXT_WEIGHT + GRAPH_REL_WEIGHT
        if graph_score_weight_total > 0:
            graph_score = (
                GRAPH_TEXT_WEIGHT * graph_text_score
                + GRAPH_REL_WEIGHT * graph_rel_score
            ) / graph_score_weight_total
        else:
            graph_score = graph_text_score

        final_score = combine_scores(
            rrf_score=rrf_score,
            graph_score=graph_score,
            rrf_weight=rrf_weight,
            graph_weight=graph_weight,
        )

        result["graph_text_score"] = round(graph_text_score, 6)
        result["graph_score"] = round(graph_score, 6)
        result["final_score"] = final_score
        result["final_combined_score"] = final_score
        result["matched_seeds"] = matched_seeds
        result["matched_expanded"] = matched_expanded

    results.sort(key=lambda r: r["final_score"], reverse=True)

    # Re-assign ranks after re-sorting
    for idx, result in enumerate(results, start=1):
        result["graph_rank"] = idx

    return results
