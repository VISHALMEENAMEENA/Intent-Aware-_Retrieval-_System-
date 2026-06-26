"""
graph/expander.py
=================
Graph Query Expansion.

Given the structured JSON from the LLM Query Understanding module,
uses the Neo4j Knowledge Graph to discover semantically related entities
via RELATED_TO (skill co-occurrence) traversal.

Returns both the expanded entity set and the traversal paths used,
which are stored for downstream explainability.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.config import (
    DEFAULT_EXPANSION_DEPTH,
    DEFAULT_EXPANSION_LIMIT,
    MIN_EDGE_CONFIDENCE,
    MIN_COOCCURRENCE_WEIGHT,
)
from graph.local_store import related_skills


def _get_session():
    from graph.client import get_session
    return get_session()


def _extract_seed_entities(llm_json: dict[str, Any]) -> dict[str, list[str]]:
    """
    Pull the seed entity names from the LLM query understanding output.
    Returns a categorised dict: {entity_type: [names]}.
    """
    def safe_list(val: Any) -> list[str]:
        if isinstance(val, list):
            return [str(v).strip().lower() for v in val if str(v).strip()]
        return []

    seeds: dict[str, list[str]] = {
        "skills": safe_list(llm_json.get("skills")),
        "technologies": safe_list(llm_json.get("technologies")),
        "tools": safe_list(llm_json.get("tools")),
        "frameworks": safe_list(llm_json.get("frameworks")),
        "programming_languages": safe_list(llm_json.get("programming_languages")),
        "soft_skills": safe_list(llm_json.get("soft_skills")),
        "roles": [],
        "locations": [],
    }

    # Role
    role = str(llm_json.get("role") or "").strip().lower()
    if role:
        seeds["roles"] = [role]

    # Location city
    location = llm_json.get("location") or {}
    city = str(location.get("city") or "").strip().lower()
    if city:
        seeds["locations"] = [city]

    return seeds


def _all_seed_skill_names(seeds: dict[str, list[str]]) -> list[str]:
    """Flatten all skill-like seeds into one deduplicated list."""
    combined: list[str] = []
    for key in ("skills", "technologies", "tools", "frameworks",
                "programming_languages", "soft_skills"):
        combined.extend(seeds.get(key, []))
    # deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for s in combined:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def expand_skills(
    skill_names: list[str],
    depth: int = DEFAULT_EXPANSION_DEPTH,
    limit: int = DEFAULT_EXPANSION_LIMIT,
    min_weight: int = MIN_COOCCURRENCE_WEIGHT,
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Traverse RELATED_TO edges up to `depth` hops from the seed skills.

    Returns:
        expanded_skills: deduplicated list of expanded skill names (excluding seeds)
        paths: list of {from, to, weight, hops} path records for explainability
    """
    if not skill_names:
        return [], []

    depth = min(depth, DEFAULT_EXPANSION_DEPTH)
    limit = min(limit, DEFAULT_EXPANSION_LIMIT)

    cypher = """
    UNWIND $seeds AS seedName
    MATCH (seed {name: seedName})
    WHERE any(label IN labels(seed) WHERE label IN ['Skill','Framework','ProgrammingLanguage','Tool','Database','CloudPlatform','Role','Certification','Education','Industry','Location'])
    CALL apoc.path.expandConfig(seed, {
        minLevel: 1,
        maxLevel: $depth,
        uniqueness: 'NODE_PATH'
    })
    YIELD path
    WITH seedName, path, last(nodes(path)) AS expanded, relationships(path) AS rels
    WHERE expanded.name <> seedName
      AND all(n IN nodes(path) WHERE NOT any(lbl IN labels(n) WHERE lbl IN ['Candidate', 'Job', 'JobDescription']))
      AND all(r IN rels WHERE coalesce(r.confidence, 0.0) >= $min_confidence)
      AND NOT 'SoftSkill' IN labels(expanded)
    RETURN seedName,
           expanded.name AS expandedName,
           type(last(rels)) AS relationshipType,
           coalesce(last(rels).confidence, 0.0) AS confidence,
           coalesce(last(rels).reason, '') AS reason,
           length(path) AS hops,
           reduce(pc = 1.0, rel IN rels | pc * coalesce(rel.confidence, 0.0)) AS pathConfidence
    ORDER BY pathConfidence DESC
    LIMIT $limit
    """

    # Fallback Cypher without APOC (standard variable-length path)
    cypher_no_apoc = """
    UNWIND $seeds AS seedName
    MATCH (seed:Skill {name: seedName})
    MATCH (seed)-[rels:RELATED_TO*1..""" + str(depth) + """]-(expanded:Skill)
    WHERE expanded.name <> seedName
      AND NOT expanded.name IN $seeds
      AND ALL(r IN rels WHERE r.weight >= $min_weight)
    WITH seedName,
         expanded.name AS expandedName,
         reduce(w = 0, r IN rels | w + r.weight) AS totalWeight
    RETURN seedName, expandedName, totalWeight
    ORDER BY totalWeight DESC
    LIMIT $limit
    """

    # Simpler flat Cypher that always works (depth 1 first for reliability)
    cypher_simple = """
    UNWIND $seeds AS seedName
    MATCH (seed:Skill {name: seedName})-[r:RELATED_TO]-(expanded:Skill)
    WHERE expanded.name <> seedName
      AND r.weight >= $min_weight
    RETURN seedName, expanded.name AS expandedName, r.weight AS totalWeight
    ORDER BY r.weight DESC
    LIMIT $limit
    """

    paths: list[dict[str, Any]] = []
    expanded_set: set[str] = set()
    seed_set = set(skill_names)

    try:
        with _get_session() as session:
            # Try APOC first; fall back to multi-hop Cypher, then to simple 1-hop
            try:
                result = session.run(
                    cypher,
                    seeds=skill_names,
                    depth=depth,
                    min_confidence=MIN_EDGE_CONFIDENCE,
                    limit=limit,
                )
                records = list(result)
            except Exception:
                return related_skills(skill_names, depth=depth, limit=limit)

            for rec in records:
                exp_name: str = rec["expandedName"]
                if exp_name and exp_name not in seed_set:
                    expanded_set.add(exp_name)
                    paths.append({
                        "from": rec["seedName"],
                        "to": exp_name,
                        "weight": rec["confidence"],
                        "confidence": rec["confidence"],
                        "path_confidence": rec["pathConfidence"],
                        "hops": rec["hops"],
                        "relationship_type": rec["relationshipType"],
                        "reason": rec["reason"],
                    })

    except Exception as exc:
        print(f"[expander] Neo4j unavailable, using local graph expansion: {exc}", file=sys.stderr)
        return related_skills(skill_names, depth=depth, limit=limit)

    return list(expanded_set)[:limit], paths


def expand_roles(role_names: list[str]) -> list[str]:
    """Find Roles in the graph that exist as nodes (validates roles are real)."""
    if not role_names:
        return []
    try:
        with _get_session() as session:
            result = session.run(
                "UNWIND $names AS n MATCH (r:Role {name: n}) RETURN r.name AS name",
                names=role_names,
            )
            return [rec["name"] for rec in result]
    except Exception:
        return []


def expand_locations(city_names: list[str]) -> list[str]:
    """Find Location nodes matching given city names."""
    if not city_names:
        return []
    try:
        with _get_session() as session:
            result = session.run(
                "UNWIND $names AS n MATCH (l:Location {city: n}) RETURN l.city AS city",
                names=city_names,
            )
            return [rec["city"] for rec in result]
    except Exception:
        return []


def expand_query(
    llm_json: dict[str, Any],
    depth: int = DEFAULT_EXPANSION_DEPTH,
    limit: int = DEFAULT_EXPANSION_LIMIT,
) -> dict[str, Any]:
    """
    Main expansion function called by search.py.

    Args:
        llm_json: Structured output from understand_query()
        depth:    Max RELATED_TO hops (default 2)
        limit:    Max total expanded entities

    Returns:
        {
          "original_query":    str,
          "intent":            str,
          "seed_entities":     {type: [names]},
          "expanded_skills":   [str],
          "confirmed_roles":   [str],
          "confirmed_locations": [str],
          "expansion_paths":   [{from, to, weight}],
          "all_query_terms":   [str],   # seeds + expanded, for retrieval
        }
    """
    seeds = _extract_seed_entities(llm_json)
    seed_skills = _all_seed_skill_names(seeds)

    expanded_skills, paths = expand_skills(seed_skills, depth=depth, limit=limit)
    confirmed_roles = expand_roles(seeds["roles"])
    confirmed_locations = expand_locations(seeds["locations"])

    # Build the flat list of terms for retrieval query construction
    all_terms: list[str] = list(dict.fromkeys(
        seed_skills + expanded_skills + seeds["roles"] + seeds["locations"]
    ))

    return {
        "original_query": str(llm_json.get("original_query", "")),
        "intent": str(llm_json.get("intent", "")),
        "seed_entities": seeds,
        "expanded_skills": expanded_skills,
        "confirmed_roles": confirmed_roles,
        "confirmed_locations": confirmed_locations,
        "expansion_paths": paths,
        "all_query_terms": all_terms,
    }
