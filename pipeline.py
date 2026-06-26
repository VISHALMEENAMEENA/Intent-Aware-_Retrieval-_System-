"""
pipeline.py
===========
Unified End-to-End Retrieval Pipeline.

Orchestrates the complete Intent-Aware, Explainable Hybrid Retrieval flow:

                     User Query
                          │
                          ▼
                 Gemini 2.5 Flash
          (Intent + Entity Extraction)
                          │
                          ▼
                 Structured JSON
                          │
          ┌───────────────┴───────────────────────────┐
          │                                           │
          ▼                                           ▼
  KG Query Expansion                  Graph Relationship Retrieval
  (expander.py)                       (graph_retrieval.py)
  expand skills via                   score candidates by
  RELATED_TO traversal                real KG neighbourhood
          │                                           │
          ▼                                           │
  Expanded Query String                               │
          │                                           │
  ┌───────┴───────┐                                   │
  │               │                                   │
  ▼               ▼                                   │
BM25 Search   Semantic Search                         │
(OpenSearch)  (FAISS)                                 │
  │               │                                   │
  └───────┬───────┘                                   │
          ▼                                           │
   RRF Fusion (rrf.py)                               │
          │                                           │
          └──────────────┬────────────────────────────┘
                         │
                         ▼
              Graph-Aware Scoring
         (graph_score + rrf_score → final_score)
                         │
                         ▼
             Cross-Encoder Reranker
          (ms-marco-MiniLM-L-6-v2)
                         │
                         ▼
            Explainability Attachment
                         │
                         ▼
              Final Ranked Results

Usage (CLI):
    python pipeline.py --query "Senior Python engineer with FastAPI and Kubernetes"
    python pipeline.py --query "Find ML engineers in Bangalore" --top-k 10

Usage (module):
    from pipeline import run_pipeline
    results = run_pipeline("Python engineer with Kubernetes")
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ── Ensure project root is on PYTHONPATH ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "retrieval"))  # for rrf.py direct import

# ── Core pipeline modules ──────────────────────────────────────────────────────
from llm.query_understanding import understand_query
from graph.expander import expand_query
from graph.graph_retrieval import score_candidates_with_graph
from graph.ranker import rank_results
from graph.explainer import attach_explanations
from reranker.cross_encoder import rerank_safe
import rrf  # retrieval/rrf.py

from graph.config import (
    RETRIEVAL_BM25_K,
    RETRIEVAL_SEMANTIC_K,
    RETRIEVAL_RRF_K,
    RETRIEVAL_FUSED_K,
    RRF_WEIGHT,
    GRAPH_WEIGHT,
    CROSS_ENCODER_POOL_SIZE,
    FINAL_TOP_K,
)


def _log(message: str) -> None:
    print(message, file=sys.stderr)


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(
    query: str,
    top_k: int = FINAL_TOP_K,
    use_cross_encoder: bool = True,
) -> dict[str, Any]:
    """
    Execute the complete end-to-end retrieval pipeline.

    Args:
        query:             Natural language user query.
        top_k:             Number of final results to return.
        use_cross_encoder: If True, apply cross-encoder re-ranking as the final
                           ranking step. If False, graph-aware score is used.

    Returns:
        Pipeline result dict:
        {
          "query":           str          – original user query
          "intent":          str          – detected intent
          "seed_entities":   dict         – LLM-extracted entities by type
          "expanded_skills": list[str]    – graph-expanded skills
          "expansion_paths": list[dict]   – graph traversal paths
          "results":         list[dict]   – final ranked results with explanations
          "pipeline_meta":   dict         – internal stage sizes for debugging
        }
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query string cannot be empty.")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 1: LLM Query Understanding
    #   Gemini extracts structured JSON: intent, skills, role, location, etc.
    # ──────────────────────────────────────────────────────────────────────────
    _log("[pipeline] Stage 1: LLM Query Understanding...")
    llm_json: dict[str, Any] = understand_query(cleaned_query)
    intent: str = llm_json.get("intent", "jd_search")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2: KG Query Expansion
    #   Uses graph RELATED_TO traversal to discover related skills.
    #   Runs in parallel conceptually with graph_retrieval below, but is needed
    #   first to build the expanded query string for BM25/FAISS.
    # ──────────────────────────────────────────────────────────────────────────
    _log("[pipeline] Stage 2: KG Query Expansion...")
    expansion_result = expand_query(llm_json)
    expanded_skills: list[str] = expansion_result.get("expanded_skills", [])
    expansion_paths: list[dict] = expansion_result.get("expansion_paths", [])

    # Build expanded query string for retrieval
    original_terms = set(cleaned_query.lower().split())
    new_terms = [s for s in expanded_skills if s.lower() not in original_terms]
    expanded_query = f"{cleaned_query} {' '.join(new_terms)}" if new_terms else cleaned_query

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 3: BM25 + Semantic Search → RRF Fusion
    #   Retrieve a larger candidate pool for downstream ranking.
    #   The expanded query is passed to both BM25 and FAISS.
    # ──────────────────────────────────────────────────────────────────────────
    _log(f"[pipeline] Stage 3: BM25 + Semantic + RRF Fusion (intent={intent})...")
    fused_results: list[dict[str, Any]] = rrf.search(
        query=expanded_query,
        intent=intent,
        top_k=RETRIEVAL_FUSED_K,
        bm25_k=RETRIEVAL_BM25_K,
        semantic_k=RETRIEVAL_SEMANTIC_K,
        rrf_k=RETRIEVAL_RRF_K,
    )

    if not fused_results:
        return _empty_result(cleaned_query, intent, expansion_result)

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 4: Graph Relationship Retrieval
    #   Score each candidate using real Neo4j graph neighbourhood lookups.
    #   (NOT text matching — actual HAS_SKILL / REQUIRES_SKILL graph edges)
    # ──────────────────────────────────────────────────────────────────────────
    _log("[pipeline] Stage 4: Graph Relationship Retrieval scoring...")
    fused_results = score_candidates_with_graph(
        candidates=fused_results,
        llm_json=llm_json,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 5: Graph-Aware Ranking
    #   Combines RRF score with graph entity text-match score.
    #   (graph_score from ranker.py is text-based; graph_rel_score from stage 4
    #   is graph-edge-based — both are captured in the explanation)
    # ──────────────────────────────────────────────────────────────────────────
    _log("[pipeline] Stage 5: Graph-Aware Re-ranking...")
    # Flatten seed entities for ranker.py text-match graph scoring
    seed_dict = expansion_result.get("seed_entities") or {}
    seed_entities: list[str] = []
    for val in seed_dict.values():
        if isinstance(val, list):
            seed_entities.extend(str(v).strip().lower() for v in val if str(v).strip())
        elif isinstance(val, str) and val.strip():
            seed_entities.append(val.strip().lower())
    seed_entities = list(dict.fromkeys(seed_entities))
    expanded_entities = [s.lower() for s in expanded_skills if s]

    graph_ranked = rank_results(
        results=fused_results,
        seed_entities=seed_entities,
        expanded_entities=expanded_entities,
        rrf_weight=RRF_WEIGHT,
        graph_weight=GRAPH_WEIGHT,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 6: Cross-Encoder Reranker
    #   Re-score the top CROSS_ENCODER_POOL_SIZE candidates jointly with
    #   the original query. This is the final precision-boosting stage.
    #   Falls back gracefully if the model is unavailable.
    # ──────────────────────────────────────────────────────────────────────────
    # Feed a larger pool into the cross-encoder to improve recall at top-K
    pool = graph_ranked[:CROSS_ENCODER_POOL_SIZE]

    if use_cross_encoder:
        _log(f"[pipeline] Stage 6: Cross-Encoder Reranking (pool={len(pool)})...")
        final_ranked = rerank_safe(
            query=cleaned_query,
            candidates=pool,
            top_k=top_k,
        )
    else:
        _log("[pipeline] Stage 6: Cross-Encoder skipped.")
        for idx, c in enumerate(pool, start=1):
            c.setdefault("cross_encoder_score", 0.0)
            c.setdefault("cross_encoder_rank", idx)
        final_ranked = pool[:top_k]

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 7: Explainability Attachment
    #   Attach a rich structured explanation to every final result.
    # ──────────────────────────────────────────────────────────────────────────
    _log("[pipeline] Stage 7: Attaching explanations...")
    explained = attach_explanations(
        results=final_ranked,
        original_query=cleaned_query,
        intent=intent,
        seed_entities=seed_entities,
        expanded_entities=expanded_entities,
        expansion_paths=expansion_paths,
    )
    for idx, result in enumerate(explained, start=1):
        result["final_rank"] = idx

    return {
        "query": cleaned_query,
        "intent": intent,
        "seed_entities": seed_dict,
        "expanded_skills": expanded_skills,
        "expansion_paths": expansion_paths,
        "results": explained,
        "pipeline_meta": {
            "fused_pool_size": len(fused_results),
            "graph_ranked_pool_size": len(graph_ranked),
            "cross_encoder_pool_size": len(pool),
            "final_result_count": len(explained),
        },
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _empty_result(
    query: str,
    intent: str,
    expansion_result: dict[str, Any],
) -> dict[str, Any]:
    """Return a well-structured empty pipeline result."""
    return {
        "query": query,
        "intent": intent,
        "seed_entities": expansion_result.get("seed_entities", {}),
        "expanded_skills": expansion_result.get("expanded_skills", []),
        "expansion_paths": expansion_result.get("expansion_paths", []),
        "results": [],
        "pipeline_meta": {
            "fused_pool_size": 0,
            "graph_ranked_pool_size": 0,
            "cross_encoder_pool_size": 0,
            "final_result_count": 0,
        },
    }


# ── CLI output ────────────────────────────────────────────────────────────────

def _print_pipeline_output(output: dict[str, Any]) -> None:
    """Pretty-print the full pipeline output to stdout."""
    results = output.get("results", [])
    meta = output.get("pipeline_meta", {})

    print("\n" + "=" * 72)
    print("  HYBRID MIND - End-to-End Retrieval Results")
    print("=" * 72)
    print(f"  Query  : {output.get('query')}")
    print(f"  Intent : {output.get('intent')}")
    print(f"  Expanded Skills: {', '.join(output.get('expanded_skills', [])[:8]) or 'none'}")
    print(f"  Pipeline: {meta.get('fused_pool_size')} fused -> "
          f"{meta.get('graph_ranked_pool_size')} graph-ranked -> "
          f"{meta.get('cross_encoder_pool_size')} CE pool -> "
          f"{meta.get('final_result_count')} results")
    print("=" * 72)

    if not results:
        print("\n  No results matched your query.\n")
        return

    for result in results:
        expl = result.get("explanation") or {}
        scores = expl.get("scores") or {}

        rank = result.get("cross_encoder_rank") or result.get("graph_rank") or "?"
        ce_score = scores.get("cross_encoder_score", 0.0)
        final_score = scores.get("final_score", 0.0)
        quality = scores.get("quality_label", "unknown")

        print(f"\n[Rank {rank}]  CE={ce_score:.4f}  Final={final_score:.4f}  ({quality})")
        print(f"  Chunk ID   : {result.get('chunk_id')}")
        print(f"  Title      : {result.get('title')}")
        print(f"  Source     : {result.get('source')}")
        print(f"  Location   : {result.get('location')}")
        print(f"  Industry   : {result.get('industry')}")
        print("-" * 72)
        print("  EXPLANATION:")
        print(f"    {expl.get('summary', '')}")

        # Score breakdown
        print("  SCORES:")
        print(f"    CrossEnc  : {scores.get('cross_encoder_score', 0.0):.6f}")
        print(f"    Graph     : {scores.get('graph_score', 0.0):.6f}")
        print(f"    GraphText : {scores.get('graph_text_score', 0.0):.6f}")
        print(f"    GraphRel  : {scores.get('graph_rel_score', 0.0):.6f}")
        print(f"    RRF       : {scores.get('rrf_score', 0.0):.6f}")
        print(f"    BM25      : {scores.get('bm25_score', 0.0):.6f}")
        print(f"    Semantic  : {scores.get('semantic_score', 0.0):.6f}")
        print(f"    Final     : {scores.get('final_combined_score', 0.0):.6f}")

        # Matched entities
        seeds_matched = expl.get("matched_original_entities") or []
        expanded_matched = expl.get("matched_expanded_entities") or []
        if seeds_matched:
            print(f"    Seeds matched: {', '.join(seeds_matched[:6])}")
        if expanded_matched:
            print(f"    Expanded matched: {', '.join(expanded_matched[:6])}")

        # Graph paths
        paths = expl.get("contributing_graph_paths") or []
        if paths:
            print("    Graph paths:")
            for p in paths[:3]:
                print(f"      ({p.get('from')}) --[{p.get('relationship_type', 'RELATED_TO')} w={p.get('weight')}]--> ({p.get('to')})")

        # Preview text
        text = str(result.get("retrieved_text") or result.get("text") or "").strip()
        if text:
            print("-" * 72)
            print(f"  Preview: {text[:250].replace(chr(10), ' ')}...")

        print("=" * 72)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> int:
    """Command-line interface for the unified pipeline."""
    parser = argparse.ArgumentParser(
        description="HybridMind - Intent-Aware, Explainable Hybrid Retrieval Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Natural language search query.",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=FINAL_TOP_K,
        help=f"Number of final results to return (default {FINAL_TOP_K}).",
    )
    parser.add_argument(
        "--no-cross-encoder",
        action="store_true",
        help="Disable cross-encoder re-ranking (use graph score as final rank).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text.",
    )
    args = parser.parse_args()

    try:
        output = run_pipeline(
            query=args.query,
            top_k=args.top_k,
            use_cross_encoder=not args.no_cross_encoder,
        )

        if args.json:
            print(json.dumps(output, indent=2, default=str))
        else:
            _print_pipeline_output(output)

    except Exception as error:
        print(f"\n[pipeline] ERROR: {error}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
