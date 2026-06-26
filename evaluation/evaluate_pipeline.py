"""
evaluate_pipeline.py
====================
Stage-wise evaluation framework for the HybridMind retrieval pipeline.

This script is evaluation-only. It does not tune, reweight, or modify retrieval
behavior. It runs every pipeline stage separately, writes CSV reports, and
produces a markdown summary that highlights where quality is lost.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_FILE = EVALUATION_DIR / "benchmark_queries.json"
DEFAULT_OUTPUT_DIR = EVALUATION_DIR

for module_dir in (
    PROJECT_ROOT,
    PROJECT_ROOT / "opensearch",
    PROJECT_ROOT / "faiss",
    PROJECT_ROOT / "retrieval",
):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from llm.query_understanding import understand_query
from graph.expander import expand_query
from graph.graph_retrieval import score_candidates_with_graph
from graph.ranker import rank_results
from graph.explainer import attach_explanations
from graph.local_store import DATASET_CHUNKS, load_all_chunks, load_chunks, skill_graph
from graph.config import (
    CROSS_ENCODER_POOL_SIZE,
    GRAPH_WEIGHT,
    RETRIEVAL_RRF_K,
    RRF_WEIGHT,
)
from reranker.cross_encoder import rerank_safe
import bm25_search
import semantic_search
import rrf


INTENT_ROUTING = {
    "profile_search": {"dataset": "profiles", "index_name": "profiles_index"},
    "job_search": {"dataset": "demands", "index_name": "demands_index"},
    "jd_search": {"dataset": "jd", "index_name": "jd_index"},
}

K_VALUES = (5, 10, 20)
REPORT_NAMES = [
    "llm_report.csv",
    "graph_report.csv",
    "bm25_report.csv",
    "semantic_report.csv",
    "rrf_report.csv",
    "graph_ranking_report.csv",
    "cross_encoder_report.csv",
    "benchmark_summary.csv",
    "error_analysis.csv",
]


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    query: str
    intent: str
    expected_entities: list[str]
    expected_relevant_candidates: list[str]
    expected_relevant_jobs: list[str]
    expected_relevant_jd: list[str]
    expected_keywords: list[str]
    notes: str = ""


def normalize(text: Any) -> str:
    return " ".join(str(text or "").casefold().strip().split())


def result_text(result: dict[str, Any]) -> str:
    fields = [
        "chunk_id", "title", "source", "chunk_type", "location", "industry",
        "retrieved_text", "text", "highlighted_text",
    ]
    return normalize(" ".join(str(result.get(field, "")) for field in fields))


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


def extract_result_ids(results: list[dict[str, Any]]) -> list[str]:
    return [str(r.get("chunk_id") or "") for r in results if r.get("chunk_id")]


def relevant_ids_for(query: BenchmarkQuery) -> set[str]:
    return set(
        query.expected_relevant_candidates
        + query.expected_relevant_jobs
        + query.expected_relevant_jd
    )


def is_relevant(result: dict[str, Any], query: BenchmarkQuery) -> bool:
    chunk_id = str(result.get("chunk_id") or "")
    ids = relevant_ids_for(query)
    if ids:
        return chunk_id in ids

    text = result_text(result)
    keywords = [normalize(k) for k in query.expected_keywords]
    if not keywords:
        return False
    matches = sum(1 for keyword in keywords if keyword and keyword in text)
    return matches >= max(1, math.ceil(len(keywords) * 0.35))


def relevance_vector(results: list[dict[str, Any]], query: BenchmarkQuery) -> list[int]:
    return [1 if is_relevant(result, query) else 0 for result in results]


def precision_at(results: list[dict[str, Any]], query: BenchmarkQuery, k: int) -> float:
    if k <= 0:
        return 0.0
    labels = relevance_vector(results[:k], query)
    return sum(labels) / k


def recall_at(results: list[dict[str, Any]], query: BenchmarkQuery, k: int) -> float:
    labels = relevance_vector(results[:k], query)
    relevant_total = len(relevant_ids_for(query))
    if relevant_total == 0:
        relevant_total = max(sum(relevance_vector(results, query)), 1)
    return min(sum(labels) / relevant_total, 1.0)


def mrr(results: list[dict[str, Any]], query: BenchmarkQuery) -> float:
    for index, result in enumerate(results, start=1):
        if is_relevant(result, query):
            return 1.0 / index
    return 0.0


def ndcg_at(results: list[dict[str, Any]], query: BenchmarkQuery, k: int) -> float:
    labels = relevance_vector(results[:k], query)
    dcg = sum(label / math.log2(index + 2) for index, label in enumerate(labels))
    ideal = sorted(relevance_vector(results, query), reverse=True)[:k]
    idcg = sum(label / math.log2(index + 2) for index, label in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def coverage(results: list[dict[str, Any]], query: BenchmarkQuery, k: int) -> float:
    text = " ".join(result_text(result) for result in results[:k])
    keywords = [normalize(k) for k in query.expected_keywords if normalize(k)]
    if not keywords:
        return 0.0
    return sum(1 for keyword in keywords if keyword in text) / len(keywords)


def average_score(results: list[dict[str, Any]], fields: list[str]) -> float:
    values: list[float] = []
    for result in results:
        for field in fields:
            try:
                if result.get(field) is not None:
                    values.append(float(result[field]))
                    break
            except (TypeError, ValueError):
                continue
    return statistics.fmean(values) if values else 0.0


def timed(callable_obj):
    start = time.perf_counter()
    value = callable_obj()
    return value, (time.perf_counter() - start) * 1000.0


def safe_stage(callable_obj, default):
    start = time.perf_counter()
    try:
        return callable_obj(), (time.perf_counter() - start) * 1000.0, ""
    except Exception as error:
        return default, (time.perf_counter() - start) * 1000.0, str(error)


def bootstrap_benchmark(path: Path, min_queries: int = 120) -> None:
    """Create an annotatable benchmark dataset with 100+ queries."""
    seeds = [
        ("profile_search", "python backend developer", ["python", "backend", "developer", "fastapi"]),
        ("profile_search", "java spring boot developer", ["java", "spring boot", "developer"]),
        ("profile_search", "devops engineer cloud ci cd", ["devops", "cloud", "ci/cd", "aws"]),
        ("profile_search", "qa automation tester selenium", ["qa", "automation", "selenium"]),
        ("profile_search", "data analyst sql power bi", ["data", "sql", "power bi"]),
        ("profile_search", "regulatory affairs compliance specialist", ["regulatory affairs", "compliance", "fda"]),
        ("job_search", "backend jobs in pune", ["backend", "pune", "developer"]),
        ("job_search", "servicenow technical lead", ["servicenow", "technical lead", "itbm"]),
        ("job_search", "test manager bengaluru", ["test manager", "bengaluru", "selenium"]),
        ("job_search", "data engineer jobs python sql", ["data engineer", "python", "sql"]),
        ("jd_search", "python developer responsibilities", ["python", "developer", "backend"]),
        ("jd_search", "qa engineer test cases regression", ["qa", "test cases", "regression"]),
        ("jd_search", "project manager stakeholder delivery", ["project manager", "stakeholder", "delivery"]),
        ("jd_search", "servicenow developer requirements", ["servicenow", "developer", "workflow"]),
        ("jd_search", "java microservices job description", ["java", "microservices", "developer"]),
    ]
    modifiers = [
        "", "with 0 to 2 years", "senior", "in pune", "in bengaluru",
        "remote", "full time", "with docker", "with sql", "with cloud",
    ]

    rows: list[dict[str, Any]] = []
    index = 1
    while len(rows) < min_queries:
        for intent, query, entities in seeds:
            for modifier in modifiers:
                if len(rows) >= min_queries:
                    break
                full_query = normalize(f"{modifier} {query}") if modifier else query
                rows.append(
                    {
                        "query_id": f"Q{index:03d}",
                        "query": full_query,
                        "intent": intent,
                        "expected_entities": unique(entities + modifier.split()),
                        "expected_relevant_candidates": [],
                        "expected_relevant_jobs": [],
                        "expected_relevant_jd": [],
                        "expected_keywords": unique(entities + query.split()[:3]),
                        "notes": "Bootstrap query. Add relevant chunk_ids manually for strict evaluation.",
                    }
                )
                index += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def load_benchmark(path: Path, max_queries: int | None = None) -> list[BenchmarkQuery]:
    if not path.exists():
        bootstrap_benchmark(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    queries: list[BenchmarkQuery] = []
    for idx, item in enumerate(raw, start=1):
        query = str(item.get("query", "")).strip()
        intent = str(item.get("intent", "")).strip()
        if not query or intent not in INTENT_ROUTING:
            raise ValueError(f"Invalid benchmark item {idx}: query/intent missing")
        queries.append(
            BenchmarkQuery(
                query_id=str(item.get("query_id") or f"Q{idx:03d}"),
                query=query,
                intent=intent,
                expected_entities=[normalize(v) for v in item.get("expected_entities", [])],
                expected_relevant_candidates=[str(v) for v in item.get("expected_relevant_candidates", [])],
                expected_relevant_jobs=[str(v) for v in item.get("expected_relevant_jobs", [])],
                expected_relevant_jd=[str(v) for v in item.get("expected_relevant_jd", [])],
                expected_keywords=[normalize(v) for v in item.get("expected_keywords", [])],
                notes=str(item.get("notes") or ""),
            )
        )
    return queries[:max_queries] if max_queries else queries


def flatten_llm_entities(llm_json: dict[str, Any]) -> list[str]:
    entities: list[str] = []
    for key in (
        "role", "skills", "technologies", "tools", "frameworks",
        "programming_languages", "soft_skills", "certifications", "education",
        "industry",
    ):
        value = llm_json.get(key)
        if isinstance(value, list):
            entities.extend(normalize(v) for v in value)
        elif value:
            entities.append(normalize(value))
    location = llm_json.get("location") or {}
    if isinstance(location, dict):
        entities.extend(normalize(v) for v in location.values() if v)
    return unique(entities)


def evaluate_llm(query: BenchmarkQuery, llm_json: dict[str, Any], error: str) -> dict[str, Any]:
    predicted = set(flatten_llm_entities(llm_json))
    expected = set(query.expected_entities)
    missing = sorted(expected - predicted)
    incorrect = sorted(predicted - expected)
    matched = sorted(expected & predicted)
    entity_precision = len(matched) / max(len(predicted), 1)
    entity_recall = len(matched) / max(len(expected), 1)
    return {
        "query_id": query.query_id,
        "query": query.query,
        "expected_intent": query.intent,
        "predicted_intent": llm_json.get("intent", ""),
        "intent_correct": int(llm_json.get("intent") == query.intent),
        "expected_entities": "; ".join(sorted(expected)),
        "predicted_entities": "; ".join(sorted(predicted)),
        "matched_entities": "; ".join(matched),
        "missing_entities": "; ".join(missing),
        "incorrect_entities": "; ".join(incorrect),
        "hallucinated_entities": "; ".join(incorrect),
        "entity_precision": round(entity_precision, 6),
        "entity_recall": round(entity_recall, 6),
        "error": error,
    }


def evaluate_graph(query: BenchmarkQuery, expansion: dict[str, Any], llm_json: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = flatten_llm_entities(llm_json)
    expanded = [normalize(v) for v in expansion.get("expanded_skills", [])]
    paths = expansion.get("expansion_paths", [])
    expected_terms = set(query.expected_entities + query.expected_keywords)
    useful = [e for e in expanded if any(t and (t in e or e in t) for t in expected_terms)]
    irrelevant = [e for e in expanded if e not in set(useful)]
    rows: list[dict[str, Any]] = []
    if paths:
        for path in paths:
            target = normalize(path.get("to"))
            rows.append(
                {
                    "query_id": query.query_id,
                    "query": query.query,
                    "original_entities": "; ".join(seeds),
                    "expanded_entity": target,
                    "expansion_path": f"{path.get('from')} -> {path.get('to')}",
                    "relationship_type": "RELATED_TO",
                    "relationship_weight": path.get("weight", 0),
                    "expansion_depth": path.get("hops", 1),
                    "useful_expansion": int(target in useful),
                    "irrelevant_expansion": int(target in irrelevant),
                    "expansion_precision": round(len(useful) / max(len(expanded), 1), 6),
                    "noise_ratio": round(len(irrelevant) / max(len(expanded), 1), 6),
                    "expansion_size": len(expanded),
                    "row_type": "query_expansion",
                }
            )
    else:
        rows.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "original_entities": "; ".join(seeds),
                "expanded_entity": "",
                "expansion_path": "",
                "relationship_type": "",
                "relationship_weight": 0,
                "expansion_depth": 0,
                "useful_expansion": 0,
                "irrelevant_expansion": 0,
                "expansion_precision": 0,
                "noise_ratio": 0,
                "expansion_size": 0,
                "row_type": "query_expansion",
            }
        )
    return rows


def metric_rows(
    query: BenchmarkQuery,
    stage: str,
    results: list[dict[str, Any]],
    latency_ms: float,
    score_fields: list[str],
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for k in K_VALUES:
        rows.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "intent": query.intent,
                "stage": stage,
                "k": k,
                "precision_at_k": round(precision_at(results, query, k), 6),
                "recall_at_k": round(recall_at(results, query, k), 6),
                "mrr": round(mrr(results, query), 6),
                "ndcg_at_k": round(ndcg_at(results, query, k), 6),
                "coverage": round(coverage(results, query, k), 6),
                "average_score": round(average_score(results[:k], score_fields), 6),
                "latency_ms": round(latency_ms, 3),
                "top_ids": "; ".join(extract_result_ids(results[:k])),
                **(extra or {}),
            }
        )
    return rows


def result_source_trace(results: list[dict[str, Any]], k: int = 20) -> str:
    traces = []
    for result in results[:k]:
        sources = []
        if result.get("bm25_rank") is not None:
            sources.append("BM25")
        if result.get("semantic_rank") is not None:
            sources.append("Semantic")
        traces.append(f"{result.get('chunk_id')}:{'+'.join(sources) or 'unknown'}")
    return "; ".join(traces)


def fuse_with_existing_rrf(
    query: BenchmarkQuery,
    bm25_results: list[dict[str, Any]],
    semantic_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reuse retrieval/rrf.py fusion without rerunning retrieval backends."""
    config = rrf.get_intent_config(query.intent)
    metadata = rrf.load_chunk_metadata(config)
    return rrf.fuse_results(
        bm25_results=bm25_results,
        semantic_results=semantic_results,
        metadata_by_chunk_id=metadata,
        top_k=20,
        rrf_k=RETRIEVAL_RRF_K,
    )


def evaluate_graph_ranking(
    query: BenchmarkQuery,
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    expansion_paths: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    before_rank = {str(r.get("chunk_id")): idx for idx, r in enumerate(before, start=1)}
    rows = []
    for idx, result in enumerate(after, start=1):
        chunk_id = str(result.get("chunk_id"))
        old_rank = before_rank.get(chunk_id, idx)
        score_diff = float(result.get("final_score") or 0.0) - float(result.get("rrf_score") or 0.0)
        rank_delta = old_rank - idx
        matched_expanded = set(result.get("matched_expanded", []))
        matched_paths = [
            f"{p.get('from')} -> {p.get('to')} ({p.get('weight')})"
            for p in expansion_paths
            if normalize(p.get("to")) in matched_expanded
        ]
        rows.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "chunk_id": chunk_id,
                "old_rank": old_rank,
                "new_rank": idx,
                "rank_delta_positive_is_improvement": rank_delta,
                "rrf_score": result.get("rrf_score", 0.0),
                "graph_score": result.get("graph_score", 0.0),
                "final_score": result.get("final_score", 0.0),
                "score_difference": round(score_diff, 6),
                "graph_contribution": round(float(result.get("graph_score") or 0.0) * GRAPH_WEIGHT, 6),
                "matched_original_entities": "; ".join(result.get("matched_seeds", [])),
                "matched_expanded_entities": "; ".join(result.get("matched_expanded", [])),
                "matched_graph_paths": "; ".join(matched_paths[:10]),
                "is_relevant": int(is_relevant(result, query)),
                "graph_reduced_quality": int(rank_delta < 0 and is_relevant(result, query)),
            }
        )
    return rows


def evaluate_cross_encoder(query: BenchmarkQuery, before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, Any]]:
    before_rank = {str(r.get("chunk_id")): idx for idx, r in enumerate(before, start=1)}
    rows = []
    for idx, result in enumerate(after, start=1):
        chunk_id = str(result.get("chunk_id"))
        old_rank = before_rank.get(chunk_id, idx)
        rows.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "chunk_id": chunk_id,
                "before_rank": old_rank,
                "after_rank": idx,
                "rank_change_positive_is_improvement": old_rank - idx,
                "cross_encoder_score": result.get("cross_encoder_score", 0.0),
                "final_score_before_ce": result.get("final_score", 0.0),
                "is_relevant": int(is_relevant(result, query)),
                "before_mrr": round(mrr(before, query), 6),
                "after_mrr": round(mrr(after, query), 6),
                "before_ndcg_10": round(ndcg_at(before, query, 10), 6),
                "after_ndcg_10": round(ndcg_at(after, query, 10), 6),
                "before_precision_10": round(precision_at(before, query, 10), 6),
                "after_precision_10": round(precision_at(after, query, 10), 6),
            }
        )
    return rows


def classify_errors(
    query: BenchmarkQuery,
    llm_row: dict[str, Any],
    expansion_rows: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    semantic_results: list[dict[str, Any]],
    rrf_results: list[dict[str, Any]],
    graph_ranked: list[dict[str, Any]],
    cross_ranked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    errors = []

    def add(category: str, detail: str) -> None:
        errors.append({"query_id": query.query_id, "query": query.query, "category": category, "detail": detail})

    if not llm_row["intent_correct"]:
        add("Wrong intent", f"expected={query.intent}; predicted={llm_row['predicted_intent']}")
    if llm_row["missing_entities"]:
        add("Missing entity", llm_row["missing_entities"])
    if llm_row["incorrect_entities"]:
        add("Hallucinated/incorrect entity", llm_row["incorrect_entities"])
    noise_values = [float(r.get("noise_ratio") or 0) for r in expansion_rows]
    if noise_values and max(noise_values) > 0.70:
        add("Incorrect graph expansion", f"noise_ratio={max(noise_values):.3f}")
    if precision_at(bm25_results, query, 10) == 0:
        add("Wrong BM25 retrieval", "Precision@10 is 0")
    if precision_at(semantic_results, query, 10) == 0:
        add("Weak semantic retrieval", "Precision@10 is 0")
    if precision_at(rrf_results, query, 10) < max(precision_at(bm25_results, query, 10), precision_at(semantic_results, query, 10)):
        add("Poor RRF fusion", "Hybrid Precision@10 lower than one source")
    if precision_at(graph_ranked, query, 10) < precision_at(rrf_results, query, 10):
        add("Incorrect graph score", "Graph ranking reduced Precision@10")
    if precision_at(cross_ranked, query, 10) < precision_at(graph_ranked, query, 10):
        add("Cross encoder mistake", "Cross encoder reduced Precision@10")
    return errors


def analyze_graph_quality(graph_ranking_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    graph = skill_graph()
    rows: list[dict[str, Any]] = []
    degrees = {node: len(neighbours) for node, neighbours in graph.items()}
    edge_weights = [
        (a, b, weight)
        for a, neighbours in graph.items()
        for b, weight in neighbours.items()
        if a < b
    ]
    avg_degree = statistics.fmean(degrees.values()) if degrees else 0.0
    communities = detect_graph_communities(graph)

    for node, degree in sorted(degrees.items(), key=lambda item: item[1], reverse=True)[:20]:
        rows.append(
            {
                "query_id": "GRAPH",
                "query": "",
                "original_entities": "",
                "expanded_entity": node,
                "expansion_path": "",
                "relationship_type": "NODE_DEGREE",
                "relationship_weight": degree,
                "expansion_depth": "",
                "useful_expansion": "",
                "irrelevant_expansion": "",
                "expansion_precision": "",
                "noise_ratio": "",
                "expansion_size": len(graph),
                "row_type": "hub_node",
                "graph_nodes": len(graph),
                "relationship_types": "RELATED_TO",
                "average_node_degree": round(avg_degree, 6),
            }
        )

    for community_id, members in enumerate(communities[:20], start=1):
        rows.append(
            {
                "query_id": "GRAPH",
                "query": "",
                "original_entities": "",
                "expanded_entity": "; ".join(members[:25]),
                "expansion_path": "",
                "relationship_type": "COMMUNITY",
                "relationship_weight": len(members),
                "expansion_depth": "",
                "useful_expansion": "",
                "irrelevant_expansion": "",
                "expansion_precision": "",
                "noise_ratio": "",
                "expansion_size": "",
                "row_type": f"community_{community_id}",
                "graph_nodes": len(graph),
                "relationship_types": "RELATED_TO",
                "average_node_degree": round(avg_degree, 6),
            }
        )

    for a, b, weight in sorted(edge_weights, key=lambda item: item[2], reverse=True)[:20]:
        rows.append(
            {
                "query_id": "GRAPH",
                "query": "",
                "original_entities": "",
                "expanded_entity": b,
                "expansion_path": f"{a} -> {b}",
                "relationship_type": "RELATED_TO_COMMON",
                "relationship_weight": weight,
                "expansion_depth": 1,
                "useful_expansion": "",
                "irrelevant_expansion": "",
                "expansion_precision": "",
                "noise_ratio": "",
                "expansion_size": "",
                "row_type": "common_edge",
                "graph_nodes": len(graph),
                "relationship_types": "RELATED_TO",
                "average_node_degree": round(avg_degree, 6),
            }
        )

    for a, b, weight in sorted(edge_weights, key=lambda item: item[2])[:20]:
        rows.append(
            {
                "query_id": "GRAPH",
                "query": "",
                "original_entities": "",
                "expanded_entity": b,
                "expansion_path": f"{a} -> {b}",
                "relationship_type": "RELATED_TO_LOW_WEIGHT",
                "relationship_weight": weight,
                "expansion_depth": 1,
                "useful_expansion": "",
                "irrelevant_expansion": "",
                "expansion_precision": "",
                "noise_ratio": "",
                "expansion_size": "",
                "row_type": "potential_noisy_edge",
                "graph_nodes": len(graph),
                "relationship_types": "RELATED_TO",
                "average_node_degree": round(avg_degree, 6),
            }
        )

    negative_edges = Counter()
    for row in graph_ranking_rows:
        if int(row.get("graph_reduced_quality") or 0):
            for path in str(row.get("matched_graph_paths") or "").split("; "):
                if path:
                    negative_edges[path] += 1
    for edge, count in negative_edges.most_common(20):
        rows.append(
            {
                "query_id": "GRAPH",
                "query": "",
                "original_entities": "",
                "expanded_entity": "",
                "expansion_path": edge,
                "relationship_type": "QUALITY_REDUCING_EDGE",
                "relationship_weight": count,
                "expansion_depth": "",
                "useful_expansion": "",
                "irrelevant_expansion": "",
                "expansion_precision": "",
                "noise_ratio": "",
                "expansion_size": "",
                "row_type": "downweight_candidate",
                "graph_nodes": len(graph),
                "relationship_types": "RELATED_TO",
                "average_node_degree": round(avg_degree, 6),
            }
        )
    return rows


def detect_graph_communities(graph: dict[str, Counter[str]]) -> list[list[str]]:
    """Return connected components as lightweight graph communities."""
    seen: set[str] = set()
    communities: list[list[str]] = []
    for node in graph:
        if node in seen:
            continue
        queue: deque[str] = deque([node])
        seen.add(node)
        component: list[str] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbour in graph.get(current, {}):
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        communities.append(sorted(component))
    communities.sort(key=len, reverse=True)
    return communities


def unrelated_expansion_producers(graph_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Top 20 source nodes that produce the most irrelevant expansions."""
    counts: Counter[str] = Counter()
    for row in graph_rows:
        if row.get("row_type") != "query_expansion":
            continue
        if int(row.get("irrelevant_expansion") or 0) != 1:
            continue
        source = str(row.get("expansion_path") or "").split(" -> ", 1)[0].strip()
        if source:
            counts[source] += 1
    return [
        {
            "query_id": "GRAPH",
            "query": "",
            "original_entities": source,
            "expanded_entity": "",
            "expansion_path": "",
            "relationship_type": "UNRELATED_EXPANSION_PRODUCER",
            "relationship_weight": count,
            "expansion_depth": "",
            "useful_expansion": "",
            "irrelevant_expansion": "",
            "expansion_precision": "",
            "noise_ratio": "",
            "expansion_size": "",
            "row_type": "top_unrelated_expansion_source",
        }
        for source, count in counts.most_common(20)
    ]


def graph_ranking_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_rows = [r for r in rows if r.get("chunk_id") and not r.get("row_type")]
    if not candidate_rows:
        return []
    gains = [float(r.get("score_difference") or 0.0) for r in candidate_rows]
    rank_deltas = [int(r.get("rank_delta_positive_is_improvement") or 0) for r in candidate_rows]
    return [
        {
            "query_id": "ALL",
            "query": "",
            "chunk_id": "",
            "old_rank": "",
            "new_rank": "",
            "rank_delta_positive_is_improvement": "",
            "rrf_score": "",
            "graph_score": "",
            "final_score": "",
            "score_difference": "",
            "graph_contribution": "",
            "matched_original_entities": "",
            "matched_expanded_entities": "",
            "matched_graph_paths": "",
            "is_relevant": "",
            "graph_reduced_quality": "",
            "row_type": "graph_ranking_summary",
            "average_graph_gain": round(statistics.fmean(max(v, 0.0) for v in gains), 6),
            "average_graph_loss": round(statistics.fmean(min(v, 0.0) for v in gains), 6),
            "rankings_improved": sum(1 for v in rank_deltas if v > 0),
            "rankings_worse": sum(1 for v in rank_deltas if v < 0),
            "graph_reduced_quality_cases": sum(int(r.get("graph_reduced_quality") or 0) for r in candidate_rows),
        }
    ]


def cross_encoder_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_rows = [r for r in rows if r.get("chunk_id") and not r.get("row_type")]
    if not candidate_rows:
        return []
    rank_changes = [int(r.get("rank_change_positive_is_improvement") or 0) for r in candidate_rows]
    before_mrr = [float(r.get("before_mrr") or 0.0) for r in candidate_rows]
    after_mrr = [float(r.get("after_mrr") or 0.0) for r in candidate_rows]
    return [
        {
            "query_id": "ALL",
            "query": "",
            "chunk_id": "",
            "before_rank": "",
            "after_rank": "",
            "rank_change_positive_is_improvement": "",
            "cross_encoder_score": "",
            "final_score_before_ce": "",
            "is_relevant": "",
            "before_mrr": "",
            "after_mrr": "",
            "before_ndcg_10": "",
            "after_ndcg_10": "",
            "before_precision_10": "",
            "after_precision_10": "",
            "row_type": "cross_encoder_summary",
            "average_rank_change": round(statistics.fmean(rank_changes), 6),
            "rankings_improved": sum(1 for v in rank_changes if v > 0),
            "rankings_worse": sum(1 for v in rank_changes if v < 0),
            "average_score_improvement": round(statistics.fmean(a - b for a, b in zip(after_mrr, before_mrr)), 6),
        }
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_stage(rows: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    filtered = [r for r in rows if r.get("stage") == stage and int(r.get("k") or 0) == 10]
    return {
        "stage": stage,
        "queries": len(filtered),
        "precision_at_10": statistics.fmean(float(r["precision_at_k"]) for r in filtered) if filtered else 0.0,
        "recall_at_10": statistics.fmean(float(r["recall_at_k"]) for r in filtered) if filtered else 0.0,
        "mrr": statistics.fmean(float(r["mrr"]) for r in filtered) if filtered else 0.0,
        "ndcg_at_10": statistics.fmean(float(r["ndcg_at_k"]) for r in filtered) if filtered else 0.0,
        "coverage": statistics.fmean(float(r["coverage"]) for r in filtered) if filtered else 0.0,
        "latency_ms": statistics.fmean(float(r["latency_ms"]) for r in filtered) if filtered else 0.0,
    }


def aggregate_llm_stage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage": "LLM Query Understanding",
        "queries": len(rows),
        "intent_accuracy": statistics.fmean(float(r["intent_correct"]) for r in rows) if rows else 0.0,
        "entity_precision": statistics.fmean(float(r["entity_precision"]) for r in rows) if rows else 0.0,
        "entity_recall": statistics.fmean(float(r["entity_recall"]) for r in rows) if rows else 0.0,
        "missing_entity_queries": sum(1 for r in rows if r.get("missing_entities")),
        "incorrect_entity_queries": sum(1 for r in rows if r.get("incorrect_entities")),
        "latency_ms": statistics.fmean(float(r["latency_ms"]) for r in rows) if rows else 0.0,
    }


def aggregate_graph_stage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    query_rows = [r for r in rows if r.get("row_type") == "query_expansion"]
    per_query: dict[str, dict[str, Any]] = {}
    for row in query_rows:
        per_query.setdefault(row["query_id"], row)
    return {
        "stage": "Knowledge Graph Expansion",
        "queries": len(per_query),
        "expansion_precision": statistics.fmean(float(r.get("expansion_precision") or 0.0) for r in per_query.values()) if per_query else 0.0,
        "noise_ratio": statistics.fmean(float(r.get("noise_ratio") or 0.0) for r in per_query.values()) if per_query else 0.0,
        "average_expansion_size": statistics.fmean(float(r.get("expansion_size") or 0.0) for r in per_query.values()) if per_query else 0.0,
        "useful_expansions": sum(int(r.get("useful_expansion") or 0) for r in query_rows),
        "irrelevant_expansions": sum(int(r.get("irrelevant_expansion") or 0) for r in query_rows),
    }


def write_summary_md(
    output_dir: Path,
    benchmark_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    graph_ranking_rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
) -> None:
    by_stage = {row["stage"]: row for row in benchmark_rows}
    error_counts = Counter(row["category"] for row in error_rows)
    avg_gain = statistics.fmean(
        max(float(r.get("score_difference") or 0.0), 0.0) for r in graph_ranking_rows
    ) if graph_ranking_rows else 0.0
    avg_loss = statistics.fmean(
        min(float(r.get("score_difference") or 0.0), 0.0) for r in graph_ranking_rows
    ) if graph_ranking_rows else 0.0
    reduced = sum(int(r.get("graph_reduced_quality") or 0) for r in graph_ranking_rows)
    hub_nodes = [r for r in graph_rows if r.get("row_type") == "hub_node"][:10]

    lines = [
        "# Evaluation Summary",
        "",
        "## Strengths",
        f"- Hybrid/RRF Precision@10: {by_stage.get('Hybrid RRF', {}).get('precision_at_10', 0):.3f}",
        f"- Graph-ranked Precision@10: {by_stage.get('Graph-aware Ranking', {}).get('precision_at_10', 0):.3f}",
        f"- Cross-encoder Precision@10: {by_stage.get('Cross Encoder', {}).get('precision_at_10', 0):.3f}",
        "",
        "## Weaknesses",
        f"- Most common failure categories: {', '.join(f'{k} ({v})' for k, v in error_counts.most_common(5)) or 'none'}",
        f"- Average graph score gain: {avg_gain:.4f}",
        f"- Average graph score loss: {avg_loss:.4f}",
        f"- Graph reduced quality cases: {reduced}",
        "",
        "## Stage Metrics",
        "| Stage | P@10 | R@10 | MRR | nDCG@10 | Coverage | Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in ["BM25", "Semantic", "Hybrid RRF", "Graph-aware Ranking", "Cross Encoder"]:
        row = by_stage.get(stage, {})
        lines.append(
            f"| {stage} | {row.get('precision_at_10', 0):.3f} | {row.get('recall_at_10', 0):.3f} | "
            f"{row.get('mrr', 0):.3f} | {row.get('ndcg_at_10', 0):.3f} | "
            f"{row.get('coverage', 0):.3f} | {row.get('latency_ms', 0):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Hub Nodes",
            *[f"- {row.get('expanded_entity')}: degree {row.get('relationship_weight')}" for row in hub_nodes],
            "",
            "## Failure Cases",
            *[f"- {row['query_id']} [{row['category']}]: {row['detail']}" for row in error_rows[:20]],
            "",
            "## Suggested Optimizations",
            "- Review hub nodes and low-weight RELATED_TO edges before changing weights.",
            "- Compare BM25 vs semantic failures to decide whether lexical or embedding recall is the first bottleneck.",
            "- Down-weight graph relationships listed as QUALITY_REDUCING_EDGE in graph_report.csv only after manual review.",
            "- Add strict relevant chunk IDs to benchmark_queries.json to replace keyword-based relevance estimates.",
            "",
            "## Prioritized Improvements",
            "1. Complete manual annotations for the 100+ benchmark queries.",
            "2. Inspect graph expansion noise and hub nodes.",
            "3. Evaluate cross-encoder with HYBRIDMIND_USE_CROSS_ENCODER=1 once model weights are available.",
            "4. Tune retrieval only after reviewing these reports.",
        ]
    )
    (output_dir / "evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(queries: list[BenchmarkQuery], output_dir: Path) -> None:
    llm_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    bm25_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    rrf_rows: list[dict[str, Any]] = []
    graph_ranking_rows: list[dict[str, Any]] = []
    cross_encoder_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for query in queries:
        routing = INTENT_ROUTING[query.intent]

        llm_json, llm_latency, llm_error = safe_stage(lambda: understand_query(query.query), {})
        llm_row = evaluate_llm(query, llm_json, llm_error)
        llm_row["latency_ms"] = round(llm_latency, 3)
        llm_rows.append(llm_row)

        expansion, _, _ = safe_stage(lambda: expand_query(llm_json), {})
        query_graph_rows = evaluate_graph(query, expansion, llm_json)
        graph_rows.extend(query_graph_rows)
        expanded_terms = expansion.get("expanded_skills", [])
        expanded_query = query.query + (" " + " ".join(expanded_terms) if expanded_terms else "")

        bm25_results, bm25_latency, bm25_error = safe_stage(
            lambda: bm25_search.search(expanded_query, routing["index_name"], top_k=20),
            [],
        )
        semantic_results, semantic_latency, semantic_error = safe_stage(
            lambda: semantic_search.search(expanded_query, routing["dataset"], top_k=20),
            [],
        )
        hybrid_results, fusion_latency, hybrid_error = safe_stage(
            lambda: fuse_with_existing_rrf(query, bm25_results, semantic_results),
            [],
        )
        hybrid_latency = bm25_latency + semantic_latency + fusion_latency

        bm25_rows.extend(metric_rows(query, "BM25", bm25_results, bm25_latency, ["score"], {"error": bm25_error}))
        semantic_rows.extend(metric_rows(query, "Semantic", semantic_results, semantic_latency, ["similarity_score"], {"error": semantic_error}))
        rrf_rows.extend(
            metric_rows(
                query,
                "BM25 only",
                bm25_results,
                bm25_latency,
                ["score"],
                {"source_trace": "BM25", "error": bm25_error},
            )
        )
        rrf_rows.extend(
            metric_rows(
                query,
                "Semantic only",
                semantic_results,
                semantic_latency,
                ["similarity_score"],
                {"source_trace": "Semantic", "error": semantic_error},
            )
        )
        rrf_rows.extend(
            metric_rows(
                query,
                "Hybrid RRF",
                hybrid_results,
                hybrid_latency,
                ["rrf_score"],
                {"source_trace": result_source_trace(hybrid_results), "error": hybrid_error},
            )
        )

        scored = score_candidates_with_graph(hybrid_results, llm_json)
        seed_entities = unique([e for values in (expansion.get("seed_entities") or {}).values() for e in (values if isinstance(values, list) else [values])])
        graph_ranked = rank_results(
            scored,
            seed_entities=seed_entities,
            expanded_entities=[normalize(v) for v in expanded_terms],
            rrf_weight=RRF_WEIGHT,
            graph_weight=GRAPH_WEIGHT,
        )
        graph_ranking_rows.extend(evaluate_graph_ranking(query, hybrid_results, graph_ranked, expansion.get("expansion_paths", [])))

        cross_ranked, cross_latency, _ = safe_stage(
            lambda: rerank_safe(query.query, graph_ranked[:CROSS_ENCODER_POOL_SIZE], top_k=20),
            graph_ranked[:20],
        )
        attached_cross = attach_explanations(
            cross_ranked,
            original_query=query.query,
            intent=query.intent,
            seed_entities=seed_entities,
            expanded_entities=[normalize(v) for v in expanded_terms],
            expansion_paths=expansion.get("expansion_paths", []),
        )
        cross_encoder_rows.extend(evaluate_cross_encoder(query, graph_ranked[:20], attached_cross))

        graph_metric_latency = hybrid_latency
        bm = metric_rows(query, "Graph-aware Ranking", graph_ranked, graph_metric_latency, ["final_score"])
        ce = metric_rows(query, "Cross Encoder", attached_cross, cross_latency, ["cross_encoder_score", "final_score"])
        # Store aggregate rows in stage-specific CSVs too, because they are useful for joins.
        graph_ranking_rows.extend(
            {**row, "chunk_id": "", "row_type": "metric_summary"} for row in bm
        )
        cross_encoder_rows.extend(
            {**row, "chunk_id": "", "row_type": "metric_summary"} for row in ce
        )

        error_rows.extend(
            classify_errors(
                query,
                llm_row,
                query_graph_rows,
                bm25_results,
                semantic_results,
                hybrid_results,
                graph_ranked,
                attached_cross,
            )
        )

    graph_rows.extend(unrelated_expansion_producers(graph_rows))
    graph_ranking_rows.extend(graph_ranking_summary_rows(graph_ranking_rows))
    cross_encoder_rows.extend(cross_encoder_summary_rows(cross_encoder_rows))
    graph_rows.extend(analyze_graph_quality(graph_ranking_rows))

    benchmark_rows = []
    benchmark_rows.append(aggregate_llm_stage(llm_rows))
    benchmark_rows.append(aggregate_graph_stage(graph_rows))
    benchmark_rows.append(aggregate_stage(bm25_rows, "BM25"))
    benchmark_rows.append(aggregate_stage(semantic_rows, "Semantic"))
    benchmark_rows.append(aggregate_stage(rrf_rows, "BM25 only"))
    benchmark_rows.append(aggregate_stage(rrf_rows, "Semantic only"))
    benchmark_rows.append(aggregate_stage(rrf_rows, "Hybrid RRF"))
    metric_graph_rows = [r for r in graph_ranking_rows if r.get("row_type") == "metric_summary"]
    metric_ce_rows = [r for r in cross_encoder_rows if r.get("row_type") == "metric_summary"]
    benchmark_rows.append(aggregate_stage(metric_graph_rows, "Graph-aware Ranking"))
    benchmark_rows.append(aggregate_stage(metric_ce_rows, "Cross Encoder"))

    write_csv(output_dir / "llm_report.csv", llm_rows)
    write_csv(output_dir / "graph_report.csv", graph_rows)
    write_csv(output_dir / "bm25_report.csv", bm25_rows)
    write_csv(output_dir / "semantic_report.csv", semantic_rows)
    write_csv(output_dir / "rrf_report.csv", rrf_rows)
    write_csv(output_dir / "graph_ranking_report.csv", graph_ranking_rows)
    write_csv(output_dir / "cross_encoder_report.csv", cross_encoder_rows)
    write_csv(output_dir / "benchmark_summary.csv", benchmark_rows)
    write_csv(output_dir / "error_analysis.csv", error_rows)
    write_summary_md(output_dir, benchmark_rows, error_rows, graph_ranking_rows, graph_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-wise HybridMind retrieval evaluation.")
    parser.add_argument("--benchmark-file", type=Path, default=DEFAULT_BENCHMARK_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--bootstrap-benchmark", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.bootstrap_benchmark or not args.benchmark_file.exists():
            bootstrap_benchmark(args.benchmark_file)
        queries = load_benchmark(args.benchmark_file, max_queries=args.max_queries)
        run_evaluation(queries, args.output_dir)
        print("Evaluation complete.")
        print(f"Benchmark queries: {len(queries)}")
        for name in REPORT_NAMES:
            print(args.output_dir / name)
        print(args.output_dir / "evaluation_summary.md")
    except Exception as error:
        print(f"Pipeline evaluation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
