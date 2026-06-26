"""
graph/graph_retrieval.py
========================
Graph Relationship Retrieval.

This module receives the structured LLM JSON and a list of retrieved candidates.
It looks up the Knowledge Graph to determine how well each candidate's entities
are connected to the query entities.

IMPORTANT:
- This module does NOT retrieve candidates.
- It only scores already-retrieved candidates using graph lookups.
- Uses actual Neo4j graph queries, not text matching.

Architecture position:
    Structured JSON ──> Graph Relationship Retrieval
                              │
                    (runs in parallel with BM25+Semantic+RRF)
                              │
                    Graph-aware Relevance Scoring
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.local_store import related_skills, skills_by_chunk_id
from graph.ontology import classify_entity


def _get_session():
    from graph.client import get_session
    return get_session()


def _extract_query_skill_names(llm_json: dict[str, Any]) -> list[str]:
    """Extract all skill-related terms directly from LLM structured JSON."""
    def safe_list(val: Any) -> list[str]:
        if isinstance(val, list):
            return [str(v).strip().lower() for v in val if str(v).strip()]
        return []

    terms: list[str] = []
    for key in ("skills", "technologies", "tools", "frameworks",
                "programming_languages"):
        terms.extend(safe_list(llm_json.get(key)))

    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _get_candidate_skills_from_graph(
    chunk_id: str,
    source: str,
) -> set[str]:
    """
    Look up a candidate's direct skill neighbourhood in Neo4j.

    Queries are intent-aware:
      - Profile  → HAS_SKILL
      - JobPosting → HAS_SKILL
      - JobDescription → REQUIRES_SKILL

    Returns a set of lowercase skill names attached to this candidate.
    """
    if not chunk_id or not source:
        return set()

    try:
        with _get_session() as session:
            # Determine the node type and relationship from the source
            if source == "profile":
                cypher = """
                MATCH (c:Candidate {candidate_id: $node_id})-[rel]->(s)
                WHERE type(rel) IN ['HAS_SKILL', 'USES_FRAMEWORK', 'USES_TOOL', 'USES_DATABASE', 'USES_CLOUD', 'CERTIFIED_IN']
                RETURN s.name AS skill_name
                """
                # Profile chunk IDs look like "profile_12345" — extract numeric ID
                node_id = chunk_id.replace("profile_", "").split("_")[0]
            elif source == "demand":
                cypher = """
                MATCH (j:Job {job_id: $node_id})-[rel]->(s)
                WHERE type(rel) IN ['REQUIRES_SKILL', 'REQUIRES_FRAMEWORK', 'REQUIRES_TOOL', 'REQUIRES_DATABASE', 'REQUIRES_CLOUD', 'MATCHES_ROLE']
                RETURN s.name AS skill_name
                """
                node_id = chunk_id.replace("demand_", "").split("_")[0]
            elif source == "jd":
                cypher = """
                MATCH (d:JobDescription {jd_id: $node_id})-[rel]->(s)
                WHERE type(rel) IN ['REQUIRES_SKILL', 'REQUIRES_FRAMEWORK', 'REQUIRES_TOOL', 'REQUIRES_DATABASE', 'REQUIRES_CLOUD', 'CERTIFIED_IN', 'HAS_EXPERIENCE', 'MATCHES_ROLE']
                RETURN s.name AS skill_name
                """
                node_id = chunk_id.replace("jd_", "").split("_")[0]
            else:
                return set()

            result = session.run(cypher, node_id=node_id)
            return {rec["skill_name"].lower() for rec in result if rec["skill_name"]}

    except Exception:
        return skills_by_chunk_id().get(chunk_id, set())


def _get_related_skills_for_query(query_skills: list[str], limit: int = 5) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """
    For the given query skills, find all directly related skills in the graph
    (1-hop RELATED_TO neighbours). Used to assess neighbourhood overlap.
    """
    if not query_skills:
        return set(), {}
    try:
        with _get_session() as session:
            result = session.run(
                """
                UNWIND $seeds AS seedName
                MATCH (s {name: seedName})-[r]->(related)
                WHERE NOT any(lbl IN labels(related) WHERE lbl IN ['Candidate', 'Job', 'JobDescription'])
                  AND NOT any(lbl IN labels(s) WHERE lbl IN ['Candidate', 'Job', 'JobDescription'])
                  AND coalesce(r.confidence, 0.0) >= 0.6
                  AND NOT 'SoftSkill' IN labels(related)
                RETURN related.name AS skill_name,
                       type(r) AS relationship_type,
                       coalesce(r.confidence, 0.0) AS confidence,
                       coalesce(r.reason, '') AS reason
                ORDER BY confidence DESC
                LIMIT $limit
                """,
                seeds=query_skills,
                limit=limit,
            )
            meta = {}
            for rec in result:
                if rec["skill_name"]:
                    name = rec["skill_name"].lower()
                    meta[name] = {
                        "relationship_type": rec["relationship_type"],
                        "confidence": float(rec["confidence"] or 0.0),
                        "reason": rec["reason"],
                    }
            return set(meta), meta
    except Exception:
        expanded, paths = related_skills(query_skills, depth=2, limit=limit)
        meta = {
            p["to"]: {
                "relationship_type": p.get("relationship_type", ""),
                "confidence": float(p.get("confidence") or p.get("path_confidence") or 0.0),
                "reason": p.get("reason", ""),
                "path_confidence": float(p.get("path_confidence") or 0.0),
            }
            for p in paths
        }
        return set(expanded), meta


def compute_graph_relationship_score(
    candidate: dict[str, Any],
    query_skills: list[str],
    related_skills: set[str],
    related_meta: dict[str, dict[str, Any]] | None = None,
) -> tuple[float, list[str], list[str]]:
    """
    Compute the graph relevance score for one candidate using actual graph lookups.

    Strategy:
    - Fetch the candidate's skills from the graph (not from text).
    - Compute direct match: how many query skills appear in the candidate's skill set.
    - Compute neighbour match: how many of the candidate's skills appear in the
      1-hop neighbourhood of the query skills.
    - Final graph score = 0.6 * direct_ratio + 0.4 * neighbour_ratio

    Returns:
        (graph_score, matched_query_skills, matched_neighbour_skills)
    """
    chunk_id = str(candidate.get("chunk_id") or "")
    source = str(candidate.get("source") or "")

    # Look up this candidate's skills from the graph
    candidate_skills = _get_candidate_skills_from_graph(chunk_id, source)

    if not candidate_skills:
        return 0.0, [], []

    query_skill_set = set(s.lower() for s in query_skills)

    related_meta = related_meta or {}

    # Direct matches: query skills present in candidate's graph skills
    direct_matches = list(candidate_skills & query_skill_set)

    # Neighbour matches: candidate skills in the query's 1-hop neighbourhood
    neighbour_matches = list(candidate_skills & related_skills - query_skill_set)

    ontology_matches = [
        skill for skill in candidate_skills
        if classify_entity(skill) in {"ProgrammingLanguage", "Framework", "Tool", "Database", "CloudPlatform", "Skill"}
    ]
    ontology_match = min(len(ontology_matches) / max(len(candidate_skills), 1), 1.0)
    direct_ratio = len(direct_matches) / len(query_skill_set) if query_skill_set else 0.0
    entity_coverage = min((len(direct_matches) + len(neighbour_matches)) / max(len(query_skill_set), 1), 1.0)
    relationship_confidence = (
        sum(float(related_meta.get(skill, {}).get("confidence") or 0.0) for skill in neighbour_matches)
        / max(len(neighbour_matches), 1)
    )
    path_confidence = (
        sum(float(related_meta.get(skill, {}).get("path_confidence") or related_meta.get(skill, {}).get("confidence") or 0.0) for skill in neighbour_matches)
        / max(len(neighbour_matches), 1)
    )

    score = round(
        0.30 * ontology_match
        + 0.25 * relationship_confidence
        + 0.25 * entity_coverage
        + 0.20 * path_confidence,
        6,
    )
    return score, direct_matches, neighbour_matches


def score_candidates_with_graph(
    candidates: list[dict[str, Any]],
    llm_json: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Score all retrieved candidates using real Neo4j graph neighbourhood lookups.

    Attaches to each candidate:
        graph_rel_score         float  – graph relationship relevance score
        graph_direct_matches    list[str]  – query skills found in candidate graph
        graph_neighbour_matches list[str]  – neighbour skills found in candidate graph

    This is the Graph Relationship Retrieval module. It does NOT change the
    ordering — it only computes and attaches graph-based scores for use by
    the downstream ranking stage.

    Args:
        candidates: List of RRF-fused result dicts.
        llm_json:   Structured output from understand_query() — NOT the expanded query.

    Returns:
        Same list of candidates with graph_rel_score attached.
    """
    query_skills = _extract_query_skill_names(llm_json)

    if not query_skills:
        # No skills in query — attach zeros and return
        for c in candidates:
            c["graph_rel_score"] = 0.0
            c["graph_direct_matches"] = []
            c["graph_neighbour_matches"] = []
        return candidates

    # Pre-fetch 1-hop neighbourhood for all query skills in one Cypher call
    related_skills, related_meta = _get_related_skills_for_query(query_skills, limit=5)

    for candidate in candidates:
        score, direct, neighbours = compute_graph_relationship_score(
            candidate=candidate,
            query_skills=query_skills,
            related_skills=related_skills,
            related_meta=related_meta,
        )
        candidate["graph_rel_score"] = score
        candidate["graph_direct_matches"] = direct
        candidate["graph_neighbour_matches"] = neighbours
        candidate["graph_relationship_evidence"] = {
            skill: related_meta.get(skill, {}) for skill in neighbours
        }

    return candidates
