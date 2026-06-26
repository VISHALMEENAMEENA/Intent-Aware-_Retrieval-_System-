"""
graph/local_store.py
====================
Small local graph/retrieval helper used when Neo4j/OpenSearch/model services
are unavailable. It reads the existing chunk CSV metadata and builds a bounded
co-occurrence graph from the stored skill lists.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from graph.ontology import classify_entity, semantic_adjacency, semantic_stack_edges

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"

DATASET_CHUNKS = {
    "profiles": CHUNKS_DIR / "profiles_chunks.csv",
    "demands": CHUNKS_DIR / "demands_chunks.csv",
    "jd": CHUNKS_DIR / "jd_chunks.csv",
}

SOURCE_TO_DATASET = {
    "profile": "profiles",
    "demand": "demands",
    "jd": "jd",
}

TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


def normalize(text: Any) -> str:
    return " ".join(str(text or "").lower().strip().split())


def tokenize(text: Any) -> list[str]:
    return TOKEN_RE.findall(normalize(text))


def _metadata_skills(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    skills = data.get("skills") or []
    if isinstance(skills, list):
        return [normalize(s) for s in skills if normalize(s)]
    return []


@lru_cache(maxsize=8)
def load_chunks(dataset: str) -> list[dict[str, Any]]:
    path = DATASET_CHUNKS[dataset]
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row = dict(row)
            row["skills"] = _metadata_skills(row.get("metadata", ""))
            row["search_text"] = " ".join(
                str(row.get(field, ""))
                for field in ("title", "location", "industry", "chunk_type", "text")
            )
            rows.append(row)
    return rows


@lru_cache(maxsize=1)
def load_all_chunks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_CHUNKS:
        rows.extend(load_chunks(dataset))
    return rows


@lru_cache(maxsize=1)
def skill_graph() -> dict[str, Counter[str]]:
    """Compatibility view for evaluation: semantic edge adjacency as weighted counters."""
    graph: dict[str, Counter[str]] = defaultdict(Counter)
    for edge in semantic_stack_edges():
        graph[edge["source"]][edge["target"]] = int(round(edge["confidence"] * 100))
    return graph


@lru_cache(maxsize=1)
def skills_by_chunk_id() -> dict[str, set[str]]:
    return {
        str(row.get("chunk_id", "")): set(row.get("skills", []))
        for row in load_all_chunks()
        if row.get("chunk_id")
    }


def related_skills(seeds: list[str], depth: int = 2, limit: int = 5) -> tuple[list[str], list[dict[str, Any]]]:
    """Ontology-constrained expansion with max 2 hops, max 5 expansions, min confidence .6."""
    adjacency = semantic_adjacency()
    seed_set = {normalize(s) for s in seeds if normalize(s)}
    seen = set(seed_set)
    frontier = [
        {
            "node": seed,
            "type": classify_entity(seed),
            "path_confidence": 1.0,
            "hops": 0,
        }
        for seed in seed_set
        if classify_entity(seed) != "SoftSkill"
    ]
    scores: dict[str, float] = {}
    paths: list[dict[str, Any]] = []

    max_depth = min(max(depth, 1), 2)
    for hop in range(1, max_depth + 1):
        next_frontier: list[dict[str, Any]] = []
        for item in frontier:
            source = item["node"]
            for edge in adjacency.get(source, []):
                if edge["confidence"] < 0.6:
                    continue
                target = edge["target"]
                if target in seed_set or target in seen:
                    continue
                if edge["target_type"] == "SoftSkill":
                    continue
                path_confidence = float(item["path_confidence"]) * float(edge["confidence"])
                scores[target] = max(scores.get(target, 0.0), path_confidence)
                paths.append({
                    "from": source,
                    "to": target,
                    "weight": edge["weight"],
                    "confidence": round(edge["confidence"], 3),
                    "path_confidence": round(path_confidence, 3),
                    "hops": hop,
                    "relationship_type": edge["relationship_type"],
                    "source_type": edge["source_type"],
                    "target_type": edge["target_type"],
                    "reason": edge["reason"],
                    "stack": edge.get("stack", ""),
                })
                next_frontier.append(
                    {
                        "node": target,
                        "type": edge["target_type"],
                        "path_confidence": path_confidence,
                        "hops": hop,
                    }
                )
                seen.add(target)
        frontier = next_frontier
        if not frontier:
            break

    expanded = [skill for skill, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]]
    path_targets = set(expanded)
    return expanded, [p for p in paths if p["to"] in path_targets][:limit]


def bm25_like_search(dataset: str, query: str, top_k: int = 10) -> list[dict[str, Any]]:
    rows = load_chunks(dataset)
    query_terms = tokenize(query)
    if not query_terms:
        return []

    doc_freq: Counter[str] = Counter()
    doc_tokens: list[list[str]] = []
    for row in rows:
        tokens = tokenize(row.get("search_text", ""))
        doc_tokens.append(tokens)
        doc_freq.update(set(tokens))

    avgdl = sum(len(t) for t in doc_tokens) / max(len(doc_tokens), 1)
    k1 = 1.5
    b = 0.75
    scored: list[tuple[float, dict[str, Any]]] = []
    for row, tokens in zip(rows, doc_tokens):
        counts = Counter(tokens)
        score = 0.0
        for term in query_terms:
            if counts[term] == 0:
                continue
            idf = math.log(1 + (len(rows) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = counts[term] + k1 * (1 - b + b * len(tokens) / max(avgdl, 1))
            score += idf * counts[term] * (k1 + 1) / denom
        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "chunk_id": row.get("chunk_id", ""),
            "score": float(score),
            "matched_index": f"{dataset}_local_bm25",
            "title": row.get("title", ""),
            "source": row.get("source", ""),
            "chunk_type": row.get("chunk_type", ""),
            "matched_chunk_type": row.get("chunk_type", ""),
            "location": row.get("location", ""),
            "industry": row.get("industry", ""),
            "text": row.get("text", ""),
            "highlighted_text": str(row.get("text", ""))[:500],
        }
        for score, row in scored[:top_k]
    ]


def semantic_like_search(dataset: str, query: str, top_k: int = 10) -> list[dict[str, Any]]:
    rows = load_chunks(dataset)
    query_terms = set(tokenize(query))
    if not query_terms:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        text_terms = set(tokenize(row.get("search_text", "")))
        skill_terms = set(token for skill in row.get("skills", []) for token in tokenize(skill))
        doc_terms = text_terms | skill_terms
        overlap = len(query_terms & doc_terms)
        if overlap == 0:
            continue
        score = overlap / math.sqrt(max(len(query_terms), 1) * max(len(doc_terms), 1))
        scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "rank": rank,
            "chunk_id": row.get("chunk_id", ""),
            "similarity_score": float(score),
            "title": row.get("title", ""),
            "source": row.get("source", ""),
            "chunk_type": row.get("chunk_type", ""),
            "location": row.get("location", ""),
            "industry": row.get("industry", ""),
            "retrieved_text": row.get("text", ""),
        }
        for rank, (score, row) in enumerate(scored[:top_k], start=1)
    ]
