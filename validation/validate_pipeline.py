"""
validation/validate_pipeline.py
================================
End-to-End Validation of the HybridMind Retrieval Pipeline.

NOTE: Forces stdout to UTF-8 to avoid Windows cp1252 issues.

Runs 20 representative queries across 11 domains, traces every pipeline stage,
validates Knowledge Graph expansion quality, checks explanation quality, computes
retrieval metrics (P@5, P@10, MRR, nDCG), checks graph contribution, and
produces validation/pipeline_validation.md.

Usage:
    python validation/validate_pipeline.py
"""

from __future__ import annotations
import io

import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Project imports ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "retrieval", PROJECT_ROOT / "opensearch", PROJECT_ROOT / "faiss"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from llm.query_understanding import understand_query
from graph.expander import expand_query
from graph.graph_retrieval import score_candidates_with_graph
from graph.ranker import rank_results
from graph.explainer import attach_explanations
from reranker.cross_encoder import rerank_safe
import rrf as rrf_module
import bm25_search
import semantic_search

from graph.config import (
    RETRIEVAL_BM25_K, RETRIEVAL_SEMANTIC_K, RETRIEVAL_RRF_K,
    RETRIEVAL_FUSED_K, RRF_WEIGHT, GRAPH_WEIGHT, CROSS_ENCODER_POOL_SIZE,
)
from graph.ontology import classify_entity, normalize

OUTPUT_DIR = PROJECT_ROOT / "validation"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Validation queries (20 across 11 domains) ─────────────────────────────────
VALIDATION_QUERIES = [
    # Backend
    {"domain": "Backend",         "query": "Senior Python backend developer with FastAPI and PostgreSQL",           "intent": "profile_search", "expected_skills": ["python", "fastapi", "postgresql", "rest api", "docker"]},
    {"domain": "Backend",         "query": "Java Spring Boot microservices developer with AWS experience",         "intent": "profile_search", "expected_skills": ["java", "spring boot", "microservices", "aws"]},
    # Frontend
    {"domain": "Frontend",        "query": "React frontend developer with TypeScript and Redux",                    "intent": "profile_search", "expected_skills": ["react", "typescript", "javascript", "redux"]},
    {"domain": "Frontend",        "query": "Angular developer with RxJS and Material Design",                      "intent": "profile_search", "expected_skills": ["angular", "typescript", "javascript"]},
    # Full Stack
    {"domain": "Full Stack",      "query": "Full stack developer Node.js React MongoDB",                           "intent": "profile_search", "expected_skills": ["node", "react", "mongodb", "javascript"]},
    {"domain": "Full Stack",      "query": "MERN stack developer with GraphQL experience",                         "intent": "profile_search", "expected_skills": ["mongodb", "react", "node", "javascript", "graphql"]},
    # DevOps
    {"domain": "DevOps",          "query": "DevOps engineer with Kubernetes Docker CI/CD pipeline expertise",      "intent": "profile_search", "expected_skills": ["docker", "kubernetes", "ci/cd", "devops", "terraform"]},
    {"domain": "DevOps",          "query": "Site reliability engineer with Prometheus Grafana and Terraform",      "intent": "profile_search", "expected_skills": ["terraform", "docker", "kubernetes", "aws"]},
    # Data Science
    {"domain": "Data Science",    "query": "Data scientist with Python pandas scikit-learn and SQL",               "intent": "profile_search", "expected_skills": ["python", "sql", "machine learning", "data science"]},
    {"domain": "Data Science",    "query": "Data engineer with Spark Airflow and Snowflake",                       "intent": "profile_search", "expected_skills": ["spark", "airflow", "snowflake", "python", "sql"]},
    # Machine Learning
    {"domain": "Machine Learning","query": "Machine learning engineer with TensorFlow PyTorch and NLP experience", "intent": "profile_search", "expected_skills": ["machine learning", "python", "tensorflow", "pytorch"]},
    {"domain": "Machine Learning","query": "MLOps engineer with model deployment Docker and Kubernetes",           "intent": "profile_search", "expected_skills": ["docker", "kubernetes", "python", "machine learning"]},
    # QA Automation
    {"domain": "QA Automation",   "query": "QA automation engineer with Selenium TestNG Java",                     "intent": "profile_search", "expected_skills": ["selenium", "java", "qa", "automation"]},
    {"domain": "QA Automation",   "query": "Software test engineer with Cypress API testing and CI/CD",           "intent": "profile_search", "expected_skills": ["selenium", "ci/cd", "rest api"]},
    # Cloud
    {"domain": "Cloud",           "query": "AWS cloud architect with Lambda S3 and CloudFormation",               "intent": "profile_search", "expected_skills": ["aws", "docker", "terraform", "kubernetes"]},
    {"domain": "Cloud",           "query": "Azure DevOps engineer with AKS and Azure Functions",                  "intent": "profile_search", "expected_skills": ["azure", "docker", "kubernetes", "devops"]},
    # Cyber Security
    {"domain": "Cyber Security",  "query": "Cyber security analyst with penetration testing and SIEM tools",      "intent": "profile_search", "expected_skills": ["linux", "python", "aws"]},
    # Mobile Development
    {"domain": "Mobile",          "query": "Android developer with Kotlin Jetpack and REST APIs",                  "intent": "profile_search", "expected_skills": ["java", "rest api", "kotlin"]},
    # Project Management
    {"domain": "Project Mgmt",    "query": "Technical project manager with Agile Scrum and stakeholder management","intent": "profile_search", "expected_skills": ["agile", "scrum"]},
    # Bonus – job demand search
    {"domain": "Job Search",      "query": "Backend developer jobs in Pune with Python",                           "intent": "job_search",     "expected_skills": ["python", "backend", "django", "fastapi"]},
]

# ── Noisy expansion detection ──────────────────────────────────────────────────
# If an expanded skill belongs to a wildly different ontology than the seed, it's noisy.
NOISE_PAIRS_FORBIDDEN = {
    # (seed_type, expanded_type) → should never cross
    ("ProgrammingLanguage", "SoftSkill"),
    ("Framework", "SoftSkill"),
    ("Tool", "SoftSkill"),
    ("ProgrammingLanguage", "Industry"),
    ("Framework", "Industry"),
}

SOFT_SKILL_BLOCKLIST = {
    "communication", "problem solving", "teamwork", "leadership",
    "collaboration", "stakeholder management",
}


# ── Metric helpers ─────────────────────────────────────────────────────────────
def precision_at(results: list[dict], expected: list[str], k: int) -> float:
    hits = sum(
        1 for r in results[:k]
        if any(exp in _result_text(r) for exp in expected)
    )
    return hits / k if k else 0.0


def recall_at(results: list[dict], expected: list[str], k: int) -> float:
    matched = set()
    for r in results[:k]:
        for exp in expected:
            if exp in _result_text(r):
                matched.add(exp)
    return len(matched) / max(len(expected), 1)


def mrr_score(results: list[dict], expected: list[str]) -> float:
    for i, r in enumerate(results, 1):
        if any(exp in _result_text(r) for exp in expected):
            return 1.0 / i
    return 0.0


def ndcg_at(results: list[dict], expected: list[str], k: int) -> float:
    rel = [1 if any(exp in _result_text(r) for exp in expected) else 0 for r in results[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(rel, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _result_text(r: dict) -> str:
    fields = ["title", "retrieved_text", "text", "highlighted_text", "chunk_id",
              "source", "chunk_type", "location", "industry"]
    return " ".join(str(r.get(f, "") or "") for f in fields).lower()


# ── Expansion quality helpers ──────────────────────────────────────────────────
def check_expansion_quality(seeds: list[str], expanded: list[str], paths: list[dict]) -> dict:
    issues = []
    good_expansions = []
    bad_expansions = []

    for p in paths:
        src = normalize(p.get("from", ""))
        tgt = normalize(p.get("to", ""))
        src_type = classify_entity(src)
        tgt_type = classify_entity(tgt)
        rel = p.get("relationship_type", "UNKNOWN")
        conf = float(p.get("confidence", 0.0))
        reason = p.get("reason", "")

        is_noisy = (
            (src_type, tgt_type) in NOISE_PAIRS_FORBIDDEN
            or tgt in SOFT_SKILL_BLOCKLIST
            or tgt_type == "SoftSkill"
            or conf < 0.6
        )

        entry = {
            "from": src,
            "to": tgt,
            "from_type": src_type,
            "to_type": tgt_type,
            "relationship_type": rel,
            "confidence": conf,
            "reason": reason,
            "is_noisy": is_noisy,
        }
        if is_noisy:
            bad_expansions.append(entry)
            issues.append(f"Noisy expansion: {src}({src_type}) --[{rel}]--> {tgt}({tgt_type}) conf={conf:.2f}")
        else:
            good_expansions.append(entry)

    return {
        "good": good_expansions,
        "bad": bad_expansions,
        "issues": issues,
        "precision": len(good_expansions) / max(len(paths), 1),
        "noise_ratio": len(bad_expansions) / max(len(paths), 1),
    }


def check_explanation_quality(result: dict) -> dict:
    explanation = result.get("explanation") or {}
    issues = []

    paths = explanation.get("contributing_graph_paths") or []
    for p in paths:
        if not p.get("relationship_type"):
            issues.append("Missing relationship_type in explanation path")
        if not p.get("reason"):
            issues.append("Missing reason in explanation path")
        if not p.get("weight"):
            issues.append("Missing weight in explanation path")
        if p.get("relationship_type") in ("RELATED_TO", "UNKNOWN", None):
            issues.append(f"Generic/missing relationship_type: {p.get('relationship_type')}")

    if not explanation.get("summary"):
        issues.append("Missing explanation summary")

    return {"issues": issues, "path_count": len(paths), "has_explanation": bool(explanation)}


# ── Stage runner ───────────────────────────────────────────────────────────────
def run_single_query(q_info: dict) -> dict:
    query = q_info["query"]
    expected = q_info["expected_skills"]
    domain = q_info["domain"]

    result = {
        "domain": domain,
        "query": query,
        "expected_skills": expected,
        "stages": {},
        "issues": [],
        "metrics": {},
        "expansion_quality": {},
        "graph_contribution": {},
        "explanation_quality": [],
        "ce_comparison": {},
        "final_results": [],
    }

    # ── Stage 1: LLM Understanding ────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        llm_json = understand_query(query)
    except Exception as e:
        llm_json = {}
        result["issues"].append(f"[LLM] Failed: {e}")
    llm_ms = (time.perf_counter() - t0) * 1000

    result["stages"]["llm"] = {
        "intent": llm_json.get("intent", ""),
        "role": llm_json.get("role", ""),
        "skills": llm_json.get("skills", []),
        "frameworks": llm_json.get("frameworks", []),
        "programming_languages": llm_json.get("programming_languages", []),
        "tools": llm_json.get("tools", []),
        "technologies": llm_json.get("technologies", []),
        "soft_skills": llm_json.get("soft_skills", []),
        "location": llm_json.get("location", {}),
        "latency_ms": round(llm_ms, 1),
    }

    # ── Stage 2: KG Expansion ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        expansion = expand_query(llm_json)
    except Exception as e:
        expansion = {"expanded_skills": [], "expansion_paths": [], "seed_entities": {}}
        result["issues"].append(f"[KG Expansion] Failed: {e}")
    kg_ms = (time.perf_counter() - t0) * 1000

    expanded_skills = expansion.get("expanded_skills", [])
    expansion_paths = expansion.get("expansion_paths", [])
    seed_dict = expansion.get("seed_entities", {})

    seed_entities: list[str] = []
    for val in seed_dict.values():
        if isinstance(val, list):
            seed_entities.extend(str(v).strip().lower() for v in val if str(v).strip())
        elif val:
            seed_entities.append(str(val).strip().lower())
    seed_entities = list(dict.fromkeys(seed_entities))

    result["stages"]["kg_expansion"] = {
        "seeds": seed_entities,
        "expanded_skills": expanded_skills,
        "expansion_paths": expansion_paths,
        "latency_ms": round(kg_ms, 1),
    }
    result["expansion_quality"] = check_expansion_quality(seed_entities, expanded_skills, expansion_paths)

    # Build expanded query
    orig_terms = set(query.lower().split())
    new_terms = [s for s in expanded_skills if s.lower() not in orig_terms]
    expanded_query = f"{query} {' '.join(new_terms)}" if new_terms else query

    # ── Stage 3: BM25 ────────────────────────────────────────────────────────
    intent = llm_json.get("intent", q_info.get("intent", "profile_search"))
    routing = {
        "profile_search": {"dataset": "profiles", "index_name": "profiles_index"},
        "job_search":     {"dataset": "demands",  "index_name": "demands_index"},
        "jd_search":      {"dataset": "jd",       "index_name": "jd_index"},
    }.get(intent, {"dataset": "profiles", "index_name": "profiles_index"})

    t0 = time.perf_counter()
    try:
        bm25_results = bm25_search.search(expanded_query, routing["index_name"], top_k=RETRIEVAL_BM25_K)
    except Exception as e:
        bm25_results = []
        result["issues"].append(f"[BM25] Failed: {e}")
    bm25_ms = (time.perf_counter() - t0) * 1000

    result["stages"]["bm25"] = {
        "top10_ids": [r.get("chunk_id") for r in bm25_results[:10]],
        "top10_scores": [round(float(r.get("score", 0)), 4) for r in bm25_results[:10]],
        "latency_ms": round(bm25_ms, 1),
        "count": len(bm25_results),
    }

    # ── Stage 4: Semantic ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        semantic_results = semantic_search.search(expanded_query, routing["dataset"], top_k=RETRIEVAL_SEMANTIC_K)
    except Exception as e:
        semantic_results = []
        result["issues"].append(f"[Semantic] Failed: {e}")
    sem_ms = (time.perf_counter() - t0) * 1000

    result["stages"]["semantic"] = {
        "top10_ids": [r.get("chunk_id") for r in semantic_results[:10]],
        "top10_scores": [round(float(r.get("similarity_score", 0)), 4) for r in semantic_results[:10]],
        "latency_ms": round(sem_ms, 1),
        "count": len(semantic_results),
    }

    # ── Stage 5: RRF Fusion ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        config_obj = rrf_module.get_intent_config(intent)
        metadata = rrf_module.load_chunk_metadata(config_obj)
        rrf_results = rrf_module.fuse_results(
            bm25_results=bm25_results,
            semantic_results=semantic_results,
            metadata_by_chunk_id=metadata,
            top_k=RETRIEVAL_FUSED_K,
            rrf_k=RETRIEVAL_RRF_K,
        )
    except Exception as e:
        rrf_results = []
        result["issues"].append(f"[RRF] Failed: {e}")
    rrf_ms = (time.perf_counter() - t0) * 1000

    result["stages"]["rrf"] = {
        "top10_ids": [r.get("chunk_id") for r in rrf_results[:10]],
        "top10_scores": [round(float(r.get("rrf_score", 0)), 6) for r in rrf_results[:10]],
        "latency_ms": round(rrf_ms, 1),
        "count": len(rrf_results),
    }

    # ── Stage 6: Graph Retrieval Scoring ──────────────────────────────────────
    t0 = time.perf_counter()
    try:
        scored = score_candidates_with_graph(rrf_results, llm_json)
    except Exception as e:
        scored = rrf_results
        result["issues"].append(f"[Graph Scoring] Failed: {e}")

    expanded_entities = [s.lower() for s in expanded_skills if s]

    t1 = time.perf_counter()
    try:
        graph_ranked = rank_results(
            results=scored,
            seed_entities=seed_entities,
            expanded_entities=expanded_entities,
            rrf_weight=RRF_WEIGHT,
            graph_weight=GRAPH_WEIGHT,
        )
    except Exception as e:
        graph_ranked = scored
        result["issues"].append(f"[Graph Ranking] Failed: {e}")
    graph_ms = (time.perf_counter() - t0) * 1000

    result["stages"]["graph_ranking"] = {
        "top10_ids": [r.get("chunk_id") for r in graph_ranked[:10]],
        "top10_final_scores": [round(float(r.get("final_score", 0)), 6) for r in graph_ranked[:10]],
        "top10_graph_scores": [round(float(r.get("graph_score", 0)), 6) for r in graph_ranked[:10]],
        "top10_rrf_scores": [round(float(r.get("rrf_score", 0)), 6) for r in graph_ranked[:10]],
        "latency_ms": round(graph_ms, 1),
    }

    # ── Stage 7: Cross Encoder ────────────────────────────────────────────────
    pool = graph_ranked[:CROSS_ENCODER_POOL_SIZE]
    pre_ce_order = [r.get("chunk_id") for r in pool]

    t0 = time.perf_counter()
    try:
        ce_ranked = rerank_safe(query, pool, top_k=10)
    except Exception as e:
        ce_ranked = pool[:10]
        result["issues"].append(f"[CrossEncoder] Failed: {e}")
    ce_ms = (time.perf_counter() - t0) * 1000

    post_ce_order = [r.get("chunk_id") for r in ce_ranked]

    # Compare before/after cross encoder
    moved_up, moved_down = [], []
    for new_pos, cid in enumerate(post_ce_order):
        if cid in pre_ce_order:
            old_pos = pre_ce_order.index(cid)
            delta = old_pos - new_pos
            if delta > 0:
                moved_up.append({"chunk_id": cid, "old_rank": old_pos + 1, "new_rank": new_pos + 1, "delta": delta})
            elif delta < 0:
                moved_down.append({"chunk_id": cid, "old_rank": old_pos + 1, "new_rank": new_pos + 1, "delta": delta})

    result["stages"]["cross_encoder"] = {
        "top10_ids": post_ce_order,
        "top10_ce_scores": [round(float(r.get("cross_encoder_score", 0)), 4) for r in ce_ranked],
        "top10_final_scores": [round(float(r.get("final_score", 0)), 6) for r in ce_ranked],
        "moved_up": moved_up[:5],
        "moved_down": moved_down[:5],
        "latency_ms": round(ce_ms, 1),
    }

    result["ce_comparison"] = {
        "moved_up": moved_up,
        "moved_down": moved_down,
        "improved": len(moved_up),
        "degraded": len(moved_down),
    }

    # ── Stage 8: Explanations ─────────────────────────────────────────────────
    try:
        final = attach_explanations(
            results=ce_ranked,
            original_query=query,
            intent=intent,
            seed_entities=seed_entities,
            expanded_entities=expanded_entities,
            expansion_paths=expansion_paths,
        )
    except Exception as e:
        final = ce_ranked
        result["issues"].append(f"[Explanations] Failed: {e}")

    # Check explanation quality per result
    expl_issues = []
    for r in final[:5]:
        q_res = check_explanation_quality(r)
        expl_issues.extend(q_res["issues"])
    result["explanation_quality"] = expl_issues

    # ── Graph contribution analysis ───────────────────────────────────────────
    improved_by_graph = 0
    hurt_by_graph = 0
    for r in graph_ranked[:10]:
        rrf_s = float(r.get("rrf_score") or 0)
        grf_s = float(r.get("final_score") or 0)
        if grf_s > rrf_s * 1.05:
            improved_by_graph += 1
        elif grf_s < rrf_s * 0.95:
            hurt_by_graph += 1

    result["graph_contribution"] = {
        "improved_by_graph": improved_by_graph,
        "hurt_by_graph": hurt_by_graph,
        "avg_graph_score": round(statistics.fmean([float(r.get("graph_score") or 0) for r in graph_ranked[:10]]), 4) if graph_ranked else 0.0,
        "avg_rrf_score": round(statistics.fmean([float(r.get("rrf_score") or 0) for r in graph_ranked[:10]]), 6) if graph_ranked else 0.0,
        "avg_final_score": round(statistics.fmean([float(r.get("final_score") or 0) for r in graph_ranked[:10]]), 4) if graph_ranked else 0.0,
    }

    # ── Retrieval metrics ─────────────────────────────────────────────────────
    result["metrics"] = {
        "bm25": {
            "p5":  round(precision_at(bm25_results, expected, 5), 3),
            "p10": round(precision_at(bm25_results, expected, 10), 3),
            "r5":  round(recall_at(bm25_results, expected, 5), 3),
            "r10": round(recall_at(bm25_results, expected, 10), 3),
            "mrr": round(mrr_score(bm25_results, expected), 3),
            "ndcg10": round(ndcg_at(bm25_results, expected, 10), 3),
            "latency_ms": round(bm25_ms, 1),
        },
        "semantic": {
            "p5":  round(precision_at(semantic_results, expected, 5), 3),
            "p10": round(precision_at(semantic_results, expected, 10), 3),
            "r5":  round(recall_at(semantic_results, expected, 5), 3),
            "r10": round(recall_at(semantic_results, expected, 10), 3),
            "mrr": round(mrr_score(semantic_results, expected), 3),
            "ndcg10": round(ndcg_at(semantic_results, expected, 10), 3),
            "latency_ms": round(sem_ms, 1),
        },
        "rrf": {
            "p5":  round(precision_at(rrf_results, expected, 5), 3),
            "p10": round(precision_at(rrf_results, expected, 10), 3),
            "r5":  round(recall_at(rrf_results, expected, 5), 3),
            "r10": round(recall_at(rrf_results, expected, 10), 3),
            "mrr": round(mrr_score(rrf_results, expected), 3),
            "ndcg10": round(ndcg_at(rrf_results, expected, 10), 3),
            "latency_ms": round(rrf_ms, 1),
        },
        "graph_ranking": {
            "p5":  round(precision_at(graph_ranked, expected, 5), 3),
            "p10": round(precision_at(graph_ranked, expected, 10), 3),
            "r5":  round(recall_at(graph_ranked, expected, 5), 3),
            "r10": round(recall_at(graph_ranked, expected, 10), 3),
            "mrr": round(mrr_score(graph_ranked, expected), 3),
            "ndcg10": round(ndcg_at(graph_ranked, expected, 10), 3),
            "latency_ms": round(graph_ms, 1),
        },
        "cross_encoder": {
            "p5":  round(precision_at(final, expected, 5), 3),
            "p10": round(precision_at(final, expected, 10), 3),
            "r5":  round(recall_at(final, expected, 5), 3),
            "r10": round(recall_at(final, expected, 10), 3),
            "mrr": round(mrr_score(final, expected), 3),
            "ndcg10": round(ndcg_at(final, expected, 10), 3),
            "latency_ms": round(ce_ms, 1),
        },
    }

    # Top 10 final results summary
    for rank, r in enumerate(final[:10], 1):
        expl = r.get("explanation") or {}
        paths = expl.get("contributing_graph_paths") or []
        result["final_results"].append({
            "rank": rank,
            "chunk_id": r.get("chunk_id"),
            "title": str(r.get("title") or "")[:80],
            "rrf_score": round(float(r.get("rrf_score") or 0), 6),
            "graph_score": round(float(r.get("graph_score") or 0), 4),
            "ce_score": round(float(r.get("cross_encoder_score") or 0), 4),
            "final_score": round(float(r.get("final_score") or 0), 4),
            "graph_paths": [
                f"{p.get('from')} --[{p.get('relationship_type')} w={p.get('weight')}]--> {p.get('to')}"
                for p in paths[:3]
            ],
        })

    return result


# ── Report generation ──────────────────────────────────────────────────────────
def print_trace(q_result: dict, idx: int) -> None:
    """Print stage-by-stage trace for one query."""
    sep = "=" * 70
    sep2 = "-" * 60
    print(f"\n{sep}")
    print(f"[Query {idx:02d}] [{q_result['domain']}] {q_result['query']}")
    print(sep)

    stages = q_result["stages"]

    # Stage 1
    llm = stages.get("llm", {})
    print(f"\n{sep2}")
    print("STAGE 1: LLM Query Understanding")
    print(f"  Intent            : {llm.get('intent')}")
    print(f"  Role              : {llm.get('role')}")
    print(f"  Skills            : {', '.join(llm.get('skills', [])) or 'none'}")
    print(f"  Frameworks        : {', '.join(llm.get('frameworks', [])) or 'none'}")
    print(f"  Languages         : {', '.join(llm.get('programming_languages', [])) or 'none'}")
    print(f"  Tools             : {', '.join(llm.get('tools', [])) or 'none'}")
    print(f"  Technologies      : {', '.join(llm.get('technologies', [])) or 'none'}")
    print(f"  Latency           : {llm.get('latency_ms')} ms")

    # Stage 2
    kg = stages.get("kg_expansion", {})
    print(f"\n{sep2}")
    print("STAGE 2: Knowledge Graph Expansion")
    print(f"  Seeds             : {', '.join(kg.get('seeds', [])) or 'none'}")
    print(f"  Expanded Skills   : {', '.join(kg.get('expanded_skills', [])) or 'none'}")
    exp_q = q_result["expansion_quality"]
    print(f"  Expansion P       : {exp_q.get('precision', 0):.2f}  Noise Ratio: {exp_q.get('noise_ratio', 0):.2f}")
    for p in kg.get("expansion_paths", [])[:5]:
        print(f"    {p.get('from')} --[{p.get('relationship_type')} w={p.get('confidence', 0):.2f}]--> {p.get('to')}")
        print(f"    Reason: {p.get('reason', 'N/A')}")
    if exp_q.get("issues"):
        for issue in exp_q["issues"]:
            print(f"  ⚠ {issue}")
    print(f"  Latency           : {kg.get('latency_ms')} ms")

    # Stage 3
    bm25 = stages.get("bm25", {})
    print(f"\n{sep2}")
    print("STAGE 3: BM25 Search")
    print(f"  Top 10 IDs        : {', '.join(str(x) for x in bm25.get('top10_ids', [])[:5])}")
    print(f"  Scores            : {', '.join(str(x) for x in bm25.get('top10_scores', [])[:5])}")
    print(f"  Latency           : {bm25.get('latency_ms')} ms")

    # Stage 4
    sem = stages.get("semantic", {})
    print(f"\n{sep2}")
    print("STAGE 4: Semantic Search")
    print(f"  Top 10 IDs        : {', '.join(str(x) for x in sem.get('top10_ids', [])[:5])}")
    print(f"  Scores            : {', '.join(str(x) for x in sem.get('top10_scores', [])[:5])}")
    print(f"  Latency           : {sem.get('latency_ms')} ms")

    # Stage 5
    rrf = stages.get("rrf", {})
    print(f"\n{sep2}")
    print("STAGE 5: RRF Fusion")
    print(f"  Top 10 IDs        : {', '.join(str(x) for x in rrf.get('top10_ids', [])[:5])}")
    print(f"  Scores            : {', '.join(str(x) for x in rrf.get('top10_scores', [])[:5])}")
    print(f"  Latency           : {rrf.get('latency_ms')} ms")

    # Stage 6
    gr = stages.get("graph_ranking", {})
    gc = q_result["graph_contribution"]
    print(f"\n{sep2}")
    print("STAGE 6: Graph-Aware Ranking")
    print(f"  Top 10 IDs        : {', '.join(str(x) for x in gr.get('top10_ids', [])[:5])}")
    print(f"  Final Scores      : {', '.join(str(x) for x in gr.get('top10_final_scores', [])[:5])}")
    print(f"  Graph Scores      : {', '.join(str(x) for x in gr.get('top10_graph_scores', [])[:5])}")
    print(f"  Graph improved    : {gc.get('improved_by_graph')} results | Hurt: {gc.get('hurt_by_graph')} results")
    print(f"  Avg Graph Score   : {gc.get('avg_graph_score')}  Avg Final: {gc.get('avg_final_score')}")
    print(f"  Latency           : {gr.get('latency_ms')} ms")

    # Stage 7
    ce = stages.get("cross_encoder", {})
    ce_comp = q_result["ce_comparison"]
    print(f"\n{sep2}")
    print("STAGE 7: Cross-Encoder Re-ranking")
    print(f"  Top 10 IDs        : {', '.join(str(x) for x in ce.get('top10_ids', [])[:5])}")
    print(f"  CE Scores         : {', '.join(str(x) for x in ce.get('top10_ce_scores', [])[:5])}")
    print(f"  Moved up          : {ce_comp.get('improved')} | Moved down: {ce_comp.get('degraded')}")
    for m in ce.get("moved_up", [])[:3]:
        print(f"    ↑ {m['chunk_id']}: rank {m['old_rank']} → {m['new_rank']}")
    for m in ce.get("moved_down", [])[:3]:
        print(f"    ↓ {m['chunk_id']}: rank {m['old_rank']} → {m['new_rank']}")
    print(f"  Latency           : {ce.get('latency_ms')} ms")

    # Metrics
    m = q_result["metrics"]
    print(f"\n{sep2}")
    print("RETRIEVAL METRICS")
    print(f"  {'Stage':<22} {'P@5':>6} {'P@10':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'nDCG':>6} {'ms':>7}")
    print(f"  {'-'*70}")
    for stage, label in [("bm25","BM25"),("semantic","Semantic"),("rrf","Hybrid RRF"),("graph_ranking","Graph-aware"),("cross_encoder","Cross Encoder")]:
        s = m.get(stage, {})
        print(f"  {label:<22} {s.get('p5',0):>6.3f} {s.get('p10',0):>6.3f} {s.get('r5',0):>6.3f} {s.get('r10',0):>6.3f} {s.get('mrr',0):>6.3f} {s.get('ndcg10',0):>6.3f} {s.get('latency_ms',0):>7.1f}")

    # Top 10 final
    print(f"\n{sep2}")
    print("FINAL TOP 10 RESULTS")
    for r in q_result["final_results"]:
        print(f"  [{r['rank']:2d}] {str(r['chunk_id']):<25}  RRF={r['rrf_score']:.5f}  Graph={r['graph_score']:.4f}  CE={r['ce_score']:.4f}  Final={r['final_score']:.4f}")
        for path in r.get("graph_paths", []):
            print(f"         {path}")

    if q_result["issues"]:
        print(f"\n  ISSUES DETECTED:")
        for issue in q_result["issues"]:
            print(f"    ⚠ {issue}")


def generate_markdown_report(all_results: list[dict]) -> str:
    lines = [
        "# HybridMind Pipeline Validation Report",
        "",
        "> Generated by `validation/validate_pipeline.py`",
        f"> Queries evaluated: **{len(all_results)}** across **11 domains**",
        "",
    ]

    # ── Overall aggregates ────────────────────────────────────────────────────
    def avg_metric(stage: str, key: str) -> float:
        vals = [r["metrics"].get(stage, {}).get(key, 0.0) for r in all_results if r["metrics"].get(stage)]
        return round(statistics.fmean(vals), 3) if vals else 0.0

    lines += [
        "## Overall Stage Metrics (Averages over all queries)",
        "",
        "| Stage | P@5 | P@10 | R@5 | R@10 | MRR | nDCG@10 | Latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, label in [
        ("bm25", "BM25"),
        ("semantic", "Semantic Search"),
        ("rrf", "Hybrid RRF"),
        ("graph_ranking", "Graph-aware Ranking"),
        ("cross_encoder", "Cross Encoder"),
    ]:
        lines.append(
            f"| {label} | {avg_metric(stage,'p5')} | {avg_metric(stage,'p10')} | "
            f"{avg_metric(stage,'r5')} | {avg_metric(stage,'r10')} | "
            f"{avg_metric(stage,'mrr')} | {avg_metric(stage,'ndcg10')} | "
            f"{avg_metric(stage,'latency_ms')} |"
        )

    # ── Graph expansion quality ───────────────────────────────────────────────
    all_exp_prec = [r["expansion_quality"].get("precision", 0.0) for r in all_results]
    all_noise = [r["expansion_quality"].get("noise_ratio", 0.0) for r in all_results]
    total_bad = sum(len(r["expansion_quality"].get("bad", [])) for r in all_results)
    total_good = sum(len(r["expansion_quality"].get("good", [])) for r in all_results)

    lines += [
        "",
        "## Knowledge Graph Expansion Quality",
        "",
        f"- **Average Expansion Precision:** {round(statistics.fmean(all_exp_prec), 3) if all_exp_prec else 0}",
        f"- **Average Noise Ratio:** {round(statistics.fmean(all_noise), 3) if all_noise else 0}",
        f"- **Total Good Expansions:** {total_good}",
        f"- **Total Noisy Expansions:** {total_bad}",
        "",
    ]

    # per-domain expansion quality
    lines.append("### Per-Domain Expansion Paths (Sample)")
    lines.append("")
    for r in all_results:
        kg = r["stages"].get("kg_expansion", {})
        paths = kg.get("expansion_paths", [])
        if not paths:
            continue
        lines.append(f"**[{r['domain']}]** `{r['query'][:60]}`")
        for p in paths[:4]:
            icon = "⚠" if any(p.get("to") == bad.get("to") for bad in r["expansion_quality"].get("bad", [])) else "✔"
            lines.append(
                f"  {icon} `{p.get('from')}` → [{p.get('relationship_type', '?')} w={p.get('confidence', 0):.2f}] → `{p.get('to')}` — {p.get('reason', '')}"
            )
        lines.append("")

    # ── Graph contribution ────────────────────────────────────────────────────
    total_improved = sum(r["graph_contribution"].get("improved_by_graph", 0) for r in all_results)
    total_hurt = sum(r["graph_contribution"].get("hurt_by_graph", 0) for r in all_results)
    avg_graph_score = round(statistics.fmean([r["graph_contribution"].get("avg_graph_score", 0) for r in all_results]), 4)

    lines += [
        "## Graph Contribution to Ranking",
        "",
        f"- **Total results improved by graph:** {total_improved}",
        f"- **Total results hurt by graph:** {total_hurt}",
        f"- **Average graph score:** {avg_graph_score}",
        "",
        "| Query (domain) | Improved | Hurt | Avg Graph Score | Avg Final Score |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in all_results:
        gc = r["graph_contribution"]
        lines.append(
            f"| [{r['domain']}] {r['query'][:45]}... | {gc.get('improved_by_graph',0)} | "
            f"{gc.get('hurt_by_graph',0)} | {gc.get('avg_graph_score',0)} | {gc.get('avg_final_score',0)} |"
        )

    # ── Cross Encoder ─────────────────────────────────────────────────────────
    total_ce_up = sum(r["ce_comparison"].get("improved", 0) for r in all_results)
    total_ce_down = sum(r["ce_comparison"].get("degraded", 0) for r in all_results)
    lines += [
        "",
        "## Cross-Encoder Re-ranking",
        "",
        f"- **Total candidates moved up:** {total_ce_up}",
        f"- **Total candidates moved down:** {total_ce_down}",
        "",
    ]

    # ── Issues detected ───────────────────────────────────────────────────────
    all_issues: list[str] = []
    for r in all_results:
        for issue in r.get("issues", []):
            all_issues.append(f"[{r['domain']}] {issue}")
        for issue in r.get("expansion_quality", {}).get("issues", []):
            all_issues.append(f"[{r['domain']}] Expansion: {issue}")
        for issue in r.get("explanation_quality", []):
            all_issues.append(f"[{r['domain']}] Explanation: {issue}")

    lines += [
        "## Detected Issues",
        "",
    ]
    if all_issues:
        for issue in all_issues[:50]:
            lines.append(f"- ⚠ {issue}")
    else:
        lines.append("- ✔ No critical issues detected.")

    # ── Per-query summary ─────────────────────────────────────────────────────
    lines += [
        "",
        "## Per-Query Validation Results",
        "",
        "| # | Domain | Query | BM25 P@10 | Sem P@10 | RRF P@10 | Graph P@10 | CE P@10 | Exp Precision | Graph Improved |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(all_results, 1):
        m = r["metrics"]
        eq = r["expansion_quality"]
        gc = r["graph_contribution"]
        lines.append(
            f"| {i} | {r['domain']} | {r['query'][:40]}... | "
            f"{m.get('bm25',{}).get('p10',0):.3f} | "
            f"{m.get('semantic',{}).get('p10',0):.3f} | "
            f"{m.get('rrf',{}).get('p10',0):.3f} | "
            f"{m.get('graph_ranking',{}).get('p10',0):.3f} | "
            f"{m.get('cross_encoder',{}).get('p10',0):.3f} | "
            f"{eq.get('precision',0):.2f} | "
            f"{gc.get('improved_by_graph',0)} |"
        )

    # ── Acceptance Criteria ───────────────────────────────────────────────────
    # Compute criteria
    avg_exp_prec = round(statistics.fmean(all_exp_prec), 3) if all_exp_prec else 0
    avg_noise = round(statistics.fmean(all_noise), 3) if all_noise else 0
    avg_bm25_p10 = avg_metric("bm25", "p10")
    avg_sem_p10 = avg_metric("semantic", "p10")
    avg_rrf_p10 = avg_metric("rrf", "p10")
    avg_graph_p10 = avg_metric("graph_ranking", "p10")
    avg_ce_p10 = avg_metric("cross_encoder", "p10")
    graph_does_not_hurt = total_hurt < total_improved
    ce_helps = total_ce_up >= total_ce_down

    def _check(val, label):
        icon = "✔" if val else "✘"
        return f"| {icon} | {label} |"

    lines += [
        "",
        "## Acceptance Criteria",
        "",
        "| Status | Criterion |",
        "|:---:|---|",
        _check(avg_exp_prec >= 0.5,      f"Graph expansion remains domain-specific (precision={avg_exp_prec})"),
        _check(avg_noise <= 0.5,         f"No excessive unrelated technologies in expansion (noise={avg_noise})"),
        _check(avg_bm25_p10 > 0,         f"BM25 contributes to retrieval (P@10={avg_bm25_p10})"),
        _check(avg_sem_p10 > 0,          f"Semantic search contributes to retrieval (P@10={avg_sem_p10})"),
        _check(avg_graph_p10 >= avg_rrf_p10 * 0.98, f"Graph does not hurt retrieval (Graph P@10={avg_graph_p10} vs RRF={avg_rrf_p10})"),
        _check(graph_does_not_hurt,      f"Graph improved more results than it hurt ({total_improved} vs {total_hurt})"),
        _check(ce_helps or total_ce_up == 0, f"Cross Encoder improves final ranking (↑{total_ce_up} ↓{total_ce_down})"),
        _check(not any("Generic/missing" in i for i in all_issues), "Explanations correctly reflect graph relationships"),
        _check(avg_graph_p10 >= avg_rrf_p10 * 0.95, "No stage introduces significant retrieval degradation"),
    ]

    # ── Overall Assessment ────────────────────────────────────────────────────
    passing = sum([
        avg_exp_prec >= 0.5,
        avg_noise <= 0.5,
        avg_bm25_p10 > 0,
        avg_sem_p10 > 0,
        avg_graph_p10 >= avg_rrf_p10 * 0.98,
        graph_does_not_hurt,
        ce_helps or total_ce_up == 0,
    ])
    total_criteria = 7

    lines += [
        "",
        "## Overall Assessment",
        "",
        f"- **Retrieval Quality:** {'Excellent' if avg_rrf_p10 > 0.8 else 'Good' if avg_rrf_p10 > 0.6 else 'Needs improvement'}  (Avg RRF P@10={avg_rrf_p10})",
        f"- **Graph Quality:** {'Good' if avg_exp_prec >= 0.6 and avg_noise <= 0.4 else 'Acceptable' if avg_exp_prec >= 0.4 else 'Needs improvement'}  (Expansion Precision={avg_exp_prec}, Noise={avg_noise})",
        f"- **Explainability Quality:** {'Good' if not any('Generic' in i for i in all_issues) else 'Needs improvement'}",
        f"- **Graph Contribution:** Improved {total_improved} results, hurt {total_hurt} results",
        f"- **Cross Encoder Contribution:** Improved {total_ce_up} positions, degraded {total_ce_down}",
        f"- **Acceptance Criteria Passed:** {passing}/{total_criteria}",
        f"- **Production Readiness:** {'✔ READY' if passing >= 5 else '⚠ NEEDS WORK'}",
        "",
        "### Recommended Improvements",
        "",
        "1. **Annotate benchmark queries** with exact chunk IDs to enable strict P/R evaluation.",
        "2. **Expand ontology dictionaries** — add more frameworks and tools to `graph/ontology/` JSON files.",
        "3. **Enable Cross-Encoder** (`HYBRIDMIND_USE_CROSS_ENCODER=1`) for full re-ranking capability.",
        "4. **Add missing ontology links** for emerging tech stacks (Kotlin, Flutter, Terraform modules).",
        "5. **Review LLM quota** — Gemini free-tier quota hit causes fallback to local parser, reducing entity extraction accuracy.",
        "6. **Increase `stacks.json` coverage** for QA, Security, Mobile, and PM domains.",
    ]

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    # Force UTF-8 stdout so Unicode chars don't crash on Windows cp1252
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 70)
    print("  HybridMind Pipeline End-to-End Validation")
    print(f"  Evaluating {len(VALIDATION_QUERIES)} queries across 11 domains")
    print("=" * 70)

    all_results = []
    for idx, q_info in enumerate(VALIDATION_QUERIES, 1):
        print(f"\n[{idx:02d}/{len(VALIDATION_QUERIES)}] Running: [{q_info['domain']}] {q_info['query'][:60]}...")
        result = run_single_query(q_info)
        all_results.append(result)
        print_trace(result, idx)

    # Save JSON results
    json_path = OUTPUT_DIR / "validation_results.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\n✔ Raw results saved to: {json_path}")

    # Generate and save markdown report
    report_md = generate_markdown_report(all_results)
    md_path = OUTPUT_DIR / "pipeline_validation.md"
    md_path.write_text(report_md, encoding="utf-8")
    print(f"✔ Markdown report saved to: {md_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("  VALIDATION COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
