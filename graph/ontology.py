"""
graph/ontology.py
=================
Typed ontology utilities for the semantic HybridMind knowledge graph.

This replaces raw co-occurrence assumptions with explicit entity classes and
curated semantic relationships used for graph build, expansion, and scoring.
"""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any


ONTOLOGY_DIR = Path(__file__).resolve().parent / "ontology"

NODE_TYPES = {
    "skill": "Skill",
    "framework": "Framework",
    "language": "ProgrammingLanguage",
    "tool": "Tool",
    "database": "Database",
    "cloud": "CloudPlatform",
    "certification": "Certification",
    "soft_skill": "SoftSkill",
    "role": "Role",
    "concept": "Skill",
    "education": "Education",
}

ENTITY_RELATIONSHIPS = {
    ("ProgrammingLanguage", "Framework"): ("USES_FRAMEWORK", "Framework commonly used with language", 0.92),
    ("Framework", "ProgrammingLanguage"): ("FRAMEWORK_FOR_LANGUAGE", "Framework is built for language", 0.92),
    ("Framework", "Skill"): ("SPECIALIZES_IN", "Framework specialization", 0.86),
    ("Skill", "Framework"): ("SUPPORTED_BY_FRAMEWORK", "Concept implemented by framework", 0.78),
    ("Framework", "Tool"): ("USES_TOOL", "Deployment or development tool", 0.78),
    ("Tool", "Framework"): ("SUPPORTS_FRAMEWORK", "Tool supports framework workflows", 0.72),
    ("ProgrammingLanguage", "Database"): ("USES_DATABASE", "Common database used with language stack", 0.72),
    ("Database", "ProgrammingLanguage"): ("DATABASE_USED_WITH", "Database commonly used by language stack", 0.70),
    ("ProgrammingLanguage", "Tool"): ("USES_TOOL", "Common tool used with language stack", 0.70),
    ("Tool", "ProgrammingLanguage"): ("TOOL_FOR_LANGUAGE", "Tool commonly used with language stack", 0.70),
    ("Tool", "Database"): ("DEPLOYS_WITH_DATABASE", "Deployment tool commonly runs database-backed services", 0.62),
    ("ProgrammingLanguage", "CloudPlatform"): ("USES_CLOUD", "Common cloud platform used with language stack", 0.64),
    ("Tool", "CloudPlatform"): ("DEPLOYS_TO_CLOUD", "Deployment tool targets cloud platform", 0.68),
    ("Role", "ProgrammingLanguage"): ("MATCHES_ROLE", "Role commonly requires language", 0.80),
    ("Role", "Framework"): ("MATCHES_ROLE", "Role commonly requires framework", 0.78),
    ("Role", "Tool"): ("MATCHES_ROLE", "Role commonly requires tool", 0.68),
    ("Role", "Database"): ("MATCHES_ROLE", "Role commonly requires database", 0.66),
    ("Role", "Skill"): ("SPECIALIZES_IN", "Role specialization", 0.74),
}

ALLOWED_EXPANSION_TARGETS = {
    "ProgrammingLanguage": {"Framework", "Tool", "Database", "CloudPlatform", "Skill"},
    "Framework": {"ProgrammingLanguage", "Tool", "Database", "Skill"},
    "Tool": {"Framework", "ProgrammingLanguage", "Database", "CloudPlatform", "Skill"},
    "Database": {"ProgrammingLanguage", "Framework", "Tool", "CloudPlatform"},
    "CloudPlatform": {"Tool", "ProgrammingLanguage", "Database"},
    "Role": {"ProgrammingLanguage", "Framework", "Tool", "Database", "Skill"},
    "Skill": {"Framework", "Tool", "Database", "ProgrammingLanguage"},
    "SoftSkill": set(),
}

SOFT_SKILL_BLOCKLIST = {
    "communication",
    "problem solving",
    "teamwork",
    "leadership",
    "collaboration",
    "stakeholder management",
}


def normalize(text: Any) -> str:
    return " ".join(str(text or "").lower().strip().split())


def _load_json(name: str) -> Any:
    path = ONTOLOGY_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def ontology_terms() -> dict[str, set[str]]:
    return {
        "framework": set(map(normalize, _load_json("frameworks.json"))),
        "language": set(map(normalize, _load_json("languages.json"))),
        "cloud": set(map(normalize, _load_json("cloud.json"))),
        "database": set(map(normalize, _load_json("databases.json"))),
        "tool": set(map(normalize, _load_json("tools.json"))),
        "role": set(map(normalize, _load_json("roles.json"))),
        "soft_skill": set(map(normalize, _load_json("soft_skills.json"))),
        "certification": set(map(normalize, _load_json("certifications.json"))),
    }


@lru_cache(maxsize=1)
def stack_definitions() -> dict[str, dict[str, Any]]:
    raw = _load_json("stacks.json")
    return {
        normalize(name): {
            key: [normalize(v) for v in value] if isinstance(value, list) else normalize(value)
            for key, value in stack.items()
        }
        for name, stack in raw.items()
    }


def classify_entity(entity: Any, metadata_category: str | None = None) -> str:
    name = normalize(entity)
    terms = ontology_terms()
    category = normalize(metadata_category)

    if not name:
        return "Skill"
    if name in terms["language"]:
        return "ProgrammingLanguage"
    if name in terms["framework"]:
        return "Framework"
    if name in terms["database"]:
        return "Database"
    if name in terms["cloud"]:
        return "CloudPlatform"
    if name in terms["tool"]:
        return "Tool"
    if name in terms["certification"] or "certification" in name or "certified" in name:
        return "Certification"
    if name in terms["soft_skill"] or category == "soft":
        return "SoftSkill"
    if name in terms["role"] or any(token in name for token in ("developer", "engineer", "manager", "analyst", "lead", "architect")):
        return "Role"
    if "degree" in name or "bachelor" in name or "master" in name:
        return "Education"
    if category in {"tool"}:
        return "Tool"
    if category in {"technology", "must_have_skill", "preferred_skill", "core", "secondary", "primary"}:
        return "Skill"
    return "Skill"


def relationship_between(source_type: str, target_type: str) -> tuple[str, str, float] | None:
    return ENTITY_RELATIONSHIPS.get((source_type, target_type))


def can_expand(source_type: str, target_type: str, explicit_soft_skill: bool = False) -> bool:
    if target_type == "SoftSkill" and not explicit_soft_skill:
        return False
    return target_type in ALLOWED_EXPANSION_TARGETS.get(source_type, set())


def semantic_stack_edges() -> list[dict[str, Any]]:
    """Curated semantic relationships between ontology entities."""
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add(source: str, target: str, source_type: str, target_type: str, stack: str) -> None:
        rel = relationship_between(source_type, target_type)
        if rel is None:
            return
        rel_type, reason, confidence = rel
        key = (source, target, rel_type)
        current = edges.get(key)
        if current is None or confidence > current["confidence"]:
            edges[key] = {
                "source": source,
                "target": target,
                "source_type": source_type,
                "target_type": target_type,
                "relationship_type": rel_type,
                "reason": reason,
                "confidence": confidence,
                "stack": stack,
                "weight": round(confidence, 3),
            }

    for stack_name, stack in stack_definitions().items():
        role = stack.get("role")
        concepts = stack.get("concepts", [])
        languages = stack.get("languages", [])
        frameworks = stack.get("frameworks", [])
        tools = stack.get("tools", [])
        databases = stack.get("databases", [])
        clouds = stack.get("cloud", [])

        for language in languages:
            if role:
                add(role, language, "Role", "ProgrammingLanguage", stack_name)
                add(language, role, "ProgrammingLanguage", "Role", stack_name)
            for framework in frameworks:
                add(language, framework, "ProgrammingLanguage", "Framework", stack_name)
                add(framework, language, "Framework", "ProgrammingLanguage", stack_name)
                for concept in concepts:
                    add(framework, concept, "Framework", "Skill", stack_name)
                    add(concept, framework, "Skill", "Framework", stack_name)
            for tool in tools:
                add(language, tool, "ProgrammingLanguage", "Tool", stack_name)
                add(tool, language, "Tool", "ProgrammingLanguage", stack_name)
            for database in databases:
                add(language, database, "ProgrammingLanguage", "Database", stack_name)
                add(database, language, "Database", "ProgrammingLanguage", stack_name)
            for cloud in clouds:
                add(language, cloud, "ProgrammingLanguage", "CloudPlatform", stack_name)
                add(cloud, language, "CloudPlatform", "ProgrammingLanguage", stack_name)

        for framework in frameworks:
            if role:
                add(role, framework, "Role", "Framework", stack_name)
                add(framework, role, "Framework", "Role", stack_name)
            for tool in tools:
                add(framework, tool, "Framework", "Tool", stack_name)
                add(tool, framework, "Tool", "Framework", stack_name)
            for database in databases:
                add(framework, database, "Framework", "Database", stack_name)
                add(database, framework, "Database", "Framework", stack_name)

        for tool in tools:
            if role:
                add(role, tool, "Role", "Tool", stack_name)
                add(tool, role, "Tool", "Role", stack_name)
            for cloud in clouds:
                add(tool, cloud, "Tool", "CloudPlatform", stack_name)
                add(cloud, tool, "CloudPlatform", "Tool", stack_name)
            for database in databases:
                add(tool, database, "Tool", "Database", stack_name)
                add(database, tool, "Database", "Tool", stack_name)

        for database in databases:
            if role:
                add(role, database, "Role", "Database", stack_name)
                add(database, role, "Database", "Role", stack_name)

        for concept in concepts:
            if role:
                add(role, concept, "Role", "Skill", stack_name)
                add(concept, role, "Skill", "Role", stack_name)

    # Explicit high-value backend path requested by evaluation findings.
    add("python", "fastapi", "ProgrammingLanguage", "Framework", "python backend")
    add("fastapi", "python", "Framework", "ProgrammingLanguage", "python backend")
    add("fastapi", "rest api", "Framework", "Skill", "python backend")
    add("rest api", "fastapi", "Skill", "Framework", "python backend")
    add("fastapi", "docker", "Framework", "Tool", "python backend")
    add("docker", "fastapi", "Tool", "Framework", "python backend")
    add("rest api", "docker", "Skill", "Tool", "python backend")
    add("docker", "rest api", "Tool", "Skill", "python backend")
    add("docker", "postgresql", "Tool", "Database", "python backend")
    add("postgresql", "docker", "Database", "Tool", "python backend")
    return list(edges.values())


@lru_cache(maxsize=1)
def semantic_adjacency() -> dict[str, list[dict[str, Any]]]:
    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in semantic_stack_edges():
        adjacency[edge["source"]].append(edge)
    for edges in adjacency.values():
        edges.sort(key=lambda e: e["confidence"], reverse=True)
    return dict(adjacency)
