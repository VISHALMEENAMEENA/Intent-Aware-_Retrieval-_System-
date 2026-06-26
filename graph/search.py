"""
graph/search.py
===============
Public search API for the explainable, graph-aware hybrid retrieval system.

Orchestrates the entire pipeline:
  1. Gemini Query Understanding (intent detection + entity extraction)
  2. Knowledge Graph Expansion (RELATED_TO traversal)
  3. Expanded Retrieval Query Construction
  4. Hybrid BM25 + Semantic Search via OpenSearch and FAISS (fused with RRF)
  5. Graph-aware Scoring & Re-ranking
  6. Structured Explanation Attachment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure import paths work correctly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "retrieval"))

from llm.query_understanding import understand_query
from graph.expander import expand_query
from graph.ranker import rank_results
from graph.explainer import attach_explanations
import rrf

from graph.config import (
    RETRIEVAL_BM25_K,
    RETRIEVAL_SEMANTIC_K,
    RETRIEVAL_RRF_K,
    RETRIEVAL_FUSED_K,
    RRF_WEIGHT,
    GRAPH_WEIGHT,
)


def search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """
    Execute the complete graph-aware explainable hybrid retrieval search.

    Args:
        query: Natural language query.
        top_k: Number of final ranked results to return.

    Returns:
        List of result dictionaries, each containing retrieved metadata,
        scores, and a structured 'explanation' dict.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query string cannot be empty.")

    # 1. Gemini Query Understanding
    llm_json = understand_query(cleaned_query)
    intent = llm_json.get("intent", "jd_search")

    # 2. Knowledge Graph Query Expansion
    expansion_result = expand_query(llm_json)
    expanded_skills = expansion_result.get("expanded_skills", [])

    # 3. Expanded Retrieval Query Construction
    original_terms = set(cleaned_query.lower().split())
    new_terms = [s for s in expanded_skills if s.lower() not in original_terms]
    if new_terms:
        expanded_query = f"{cleaned_query} {' '.join(new_terms)}"
    else:
        expanded_query = cleaned_query

    # 4. Retrieve candidate pool from OpenSearch and FAISS, fused via RRF
    # We retrieve a larger pool (RETRIEVAL_FUSED_K) so graph re-ranking can select the best
    fused_results = rrf.search(
        query=expanded_query,
        intent=intent,
        top_k=RETRIEVAL_FUSED_K,
        bm25_k=RETRIEVAL_BM25_K,
        semantic_k=RETRIEVAL_SEMANTIC_K,
        rrf_k=RETRIEVAL_RRF_K,
    )

    if not fused_results:
        return []

    # 5. Extract flat lists of entities for graph scoring
    seed_dict = expansion_result.get("seed_entities") or {}
    seed_entities: list[str] = []
    for val in seed_dict.values():
        if isinstance(val, list):
            seed_entities.extend(str(v).strip().lower() for v in val if str(v).strip())
        elif isinstance(val, str) and val.strip():
            seed_entities.append(val.strip().lower())
    seed_entities = list(dict.fromkeys(seed_entities))

    expanded_entities = [s.lower() for s in expanded_skills if s]

    # 6. Graph-aware Ranking
    ranked_candidates = rank_results(
        results=fused_results,
        seed_entities=seed_entities,
        expanded_entities=expanded_entities,
        rrf_weight=RRF_WEIGHT,
        graph_weight=GRAPH_WEIGHT,
    )

    # Slice to top_k requested
    top_candidates = ranked_candidates[:top_k]

    # 7. Attach structured explanations
    attached_results = attach_explanations(
        results=top_candidates,
        original_query=cleaned_query,
        intent=intent,
        seed_entities=seed_entities,
        expanded_entities=expanded_entities,
        expansion_paths=expansion_result.get("expansion_paths", []),
    )

    return attached_results


def print_search_results(results: list[dict[str, Any]]) -> None:
    """Print the final search results with their explanations in a clear CLI format."""
    print("=" * 70)
    print("Graph-Aware Hybrid Retrieval Results")
    print("=" * 70)

    if not results:
        print("\nNo results matched the query.")
        return

    for result in results:
        expl = result.get("explanation") or {}
        scores = expl.get("scores") or {}

        print(f"\n[Rank {result.get('graph_rank') or result.get('rank')}] Final Score: {scores.get('final_score', 0.0):.4f} ({scores.get('quality_label', 'unknown')})")
        print(f"  Chunk ID   : {result.get('chunk_id')}")
        print(f"  Title      : {result.get('title')}")
        print(f"  Source     : {result.get('source')}")
        print(f"  Location   : {result.get('location')}")
        print(f"  Industry   : {result.get('industry')}")
        print("-" * 70)
        print("  RELEVANCE EXPLANATION:")
        print(f"    * {expl.get('summary')}")
        print(f"    * Intent detected: {expl.get('intent')}")
        
        matched_seeds = expl.get("matched_original_entities") or []
        matched_expanded = expl.get("matched_expanded_entities") or []
        paths = expl.get("contributing_graph_paths") or []

        if matched_seeds:
            print(f"    * Matched seeds: {', '.join(matched_seeds)}")
        if matched_expanded:
            print(f"    * Matched expanded: {', '.join(matched_expanded)}")
        
        if paths:
            print("    * Contributing graph paths:")
            for p in paths[:3]:
                print(f"        - ({p.get('from')}) -[{p.get('relationship_type', 'RELATED_TO')} (weight: {p.get('weight')})]-> ({p.get('to')})")
        
        print("  DETAIL SCORES:")
        print(f"    - RRF Score       : {scores.get('rrf_score', 0.0):.6f}")
        print(f"    - Graph Score     : {scores.get('graph_score', 0.0):.6f}")
        
        print("-" * 70)
        print("  Retrieved Text Preview:")
        text = str(result.get("retrieved_text") or "")
        preview = text.strip().replace("\n", " ")[:200]
        print(f"    {preview}...")
        print("=" * 70)


def main() -> int:
    """Run search from command line."""
    parser = argparse.ArgumentParser(
        description="Run graph-aware explainable hybrid search on the RAG pipeline."
    )
    parser.add_argument("--query", required=True, help="Natural language search query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final results to display.")
    args = parser.parse_args()

    try:
        results = search(args.query, top_k=args.top_k)
        print_search_results(results)
    except Exception as error:
        print(f"Search execution failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
