"""
graph/explainer.py
==================
Builds a human-readable, structured explanation for every retrieved result.

Each explanation tells the user:
  - Which original (seed) entities matched the document
  - Which graph-expanded entities matched the document
  - Which graph traversal paths contributed to expansion
  - The RRF score, graph score, and combined final score
  - A natural-language summary sentence

The explanation is stored as a nested dict inside each result under
the key "explanation".
"""

from __future__ import annotations

from typing import Any


def _score_label(score: float) -> str:
    """Return a qualitative label for a final_score in [0, 1]."""
    if score >= 0.80:
        return "excellent"
    if score >= 0.60:
        return "strong"
    if score >= 0.40:
        return "moderate"
    if score >= 0.20:
        return "partial"
    return "weak"


def build_explanation(
    result: dict[str, Any],
    original_query: str,
    intent: str,
    seed_entities: list[str],
    expanded_entities: list[str],
    expansion_paths: list[dict[str, Any]],
    rrf_score: float,
    graph_score: float,
    final_score: float,
) -> dict[str, Any]:
    """
    Construct a rich explanation dict for one retrieved result.

    Returns a dict suitable for JSON serialisation.
    """
    matched_seeds: list[str] = result.get("matched_seeds", [])
    matched_expanded: list[str] = result.get("matched_expanded", [])
    graph_direct: list[str] = result.get("graph_direct_matches", [])
    graph_neighbours: list[str] = result.get("graph_neighbour_matches", [])

    # Filter paths to only those whose 'to' node is in the matched expanded set
    matched_set = set(matched_expanded)
    contributing_paths = [
        p for p in expansion_paths
        if p.get("to") in matched_set
    ]

    # Build natural-language summary
    parts: list[str] = []
    if matched_seeds:
        parts.append(f"Directly matched: {', '.join(matched_seeds[:5])}")
    if matched_expanded:
        parts.append(f"Graph-expanded matches: {', '.join(matched_expanded[:5])}")
    if graph_direct:
        parts.append(f"Graph relationship direct matches: {', '.join(graph_direct[:5])}")
    if graph_neighbours:
        parts.append(f"Graph relationship neighbour matches: {', '.join(graph_neighbours[:5])}")
    if not matched_seeds and not matched_expanded and not graph_direct and not graph_neighbours:
        parts.append("No direct or expanded entity matches found.")

    quality = _score_label(final_score)
    summary = (
        f"This result has a {quality} relevance to your query "
        f"'{original_query}'. " + " | ".join(parts) + "."
    )

    return {
        "summary": summary,
        "intent": intent,
        "original_query": original_query,
        "seed_entities": seed_entities,
        "expanded_entities": expanded_entities,
        "matched_original_entities": matched_seeds,
        "matched_expanded_entities": matched_expanded,
        "matched_original_skills": graph_direct or matched_seeds,
        "matched_graph_skills": graph_neighbours or matched_expanded,
        "reason": "Candidate matches explicit user requirements and graph-related technologies."
                  if (matched_seeds or matched_expanded or graph_direct or graph_neighbours)
                  else "Candidate was retained by hybrid retrieval, but graph evidence is weak.",
        "contributing_graph_paths": contributing_paths[:10],  # cap for readability
        "scores": {
            "rrf_score": round(rrf_score, 6),
            "graph_score": round(graph_score, 6),
            "graph_text_score": round(result.get("graph_text_score") or 0.0, 6),
            "graph_rel_score": round(result.get("graph_rel_score") or 0.0, 6),
            "cross_encoder_score": round(result.get("cross_encoder_score") or 0.0, 6),
            "cross_encoder_normalized_score": round(result.get("cross_encoder_normalized_score") or 0.0, 6),
            "pre_cross_encoder_score": round(result.get("pre_cross_encoder_score") or 0.0, 6),
            "final_score": round(final_score, 6),
            "final_combined_score": round(result.get("final_combined_score") or final_score, 6),
            "bm25_score": round(result.get("bm25_score") or 0.0, 6),
            "semantic_score": round(result.get("semantic_score") or 0.0, 6),
            "quality_label": quality,
        },
    }


def attach_explanations(
    results: list[dict[str, Any]],
    original_query: str,
    intent: str,
    seed_entities: list[str],
    expanded_entities: list[str],
    expansion_paths: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Attach an 'explanation' key to every result dict in-place.

    Args:
        results:          Re-ranked result list (from ranker.rank_results)
        original_query:   Raw user query string
        intent:           Classified intent (profile_search / job_search / jd_search)
        seed_entities:    Flat list of seed skill/term names
        expanded_entities: Flat list of graph-expanded skill names
        expansion_paths:  [{from, to, weight}] records

    Returns:
        The same list with 'explanation' dicts attached.
    """
    for result in results:
        rrf_score = float(result.get("rrf_score") or 0.0)
        graph_score = float(result.get("graph_score") or 0.0)
        final_score = float(result.get("final_score") or 0.0)

        result["explanation"] = build_explanation(
            result=result,
            original_query=original_query,
            intent=intent,
            seed_entities=seed_entities,
            expanded_entities=expanded_entities,
            expansion_paths=expansion_paths,
            rrf_score=rrf_score,
            graph_score=graph_score,
            final_score=final_score,
        )

    return results
