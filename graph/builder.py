"""
graph/builder.py
================
Semantic Knowledge Graph builder for HybridMind.

This builder intentionally avoids raw co-occurrence edges. It creates typed
ontology nodes and semantically meaningful relationships from the cleaned
datasets plus curated ontology stack rules.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.client import get_session, verify_connection
from graph.config import BATCH_SIZE, DEMANDS_CSV, JD_CSV, PROFILES_CSV
from graph.ontology import classify_entity, normalize, semantic_stack_edges
from graph.schema import create_schema, get_schema_summary


ENTITY_LABELS = {
    "Skill",
    "Framework",
    "ProgrammingLanguage",
    "Tool",
    "Database",
    "CloudPlatform",
    "Certification",
    "SoftSkill",
    "Role",
    "Industry",
    "Location",
    "Education",
}

RELATIONSHIP_TYPES = {
    "HAS_SKILL",
    "USES_FRAMEWORK",
    "USES_TOOL",
    "USES_DATABASE",
    "USES_CLOUD",
    "REQUIRES_SKILL",
    "REQUIRES_FRAMEWORK",
    "REQUIRES_TOOL",
    "REQUIRES_DATABASE",
    "REQUIRES_CLOUD",
    "SIMILAR_ROLE",
    "LOCATED_IN",
    "WORKED_AS",
    "BELONGS_TO_DOMAIN",
    "PART_OF_STACK",
    "PREREQUISITE",
    "CERTIFIED_IN",
    "SPECIALIZES_IN",
    "HAS_EXPERIENCE",
    "MATCHES_ROLE",
    "FRAMEWORK_FOR_LANGUAGE",
    "SUPPORTED_BY_FRAMEWORK",
    "SUPPORTS_FRAMEWORK",
    "DATABASE_USED_WITH",
    "TOOL_FOR_LANGUAGE",
    "DEPLOYS_WITH_DATABASE",
    "DEPLOYS_TO_CLOUD",
}


def split_csv_field(value: Any) -> list[str]:
    if not value or str(value).strip() in ("nan", ""):
        return []
    return [normalize(part) for part in str(value).split(",") if normalize(part)]


def parse_list_field(value: Any) -> list[str]:
    if not value or str(value).strip() in ("nan", "[]", ""):
        return []
    text = str(value).strip()
    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [normalize(item) for item in parsed if normalize(item)]
        except Exception:
            pass
    return split_csv_field(value)


def parse_json_dict(value: Any) -> dict[str, Any]:
    if not value or str(value).strip() in ("nan", "{}"):
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def short_entities(items: list[str], max_len: int = 80) -> list[str]:
    return [item for item in items if item and len(item) <= max_len]


def run_batched(session: Any, cypher: str, rows: list[dict[str, Any]]) -> None:
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        if batch:
            session.run(cypher, rows=batch)


def merge_entities(session: Any, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = row["label"]
        if label not in ENTITY_LABELS:
            continue
        grouped.setdefault(label, []).append(row)

    for label, label_rows in grouped.items():
        run_batched(
            session,
            f"""UNWIND $rows AS r
                MERGE (e:{label} {{name: r.name}})
                SET e.ontology_type = r.label,
                    e.source = coalesce(e.source, r.source)""",
            label_rows,
        )


def merge_relationships(session: Any, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["source_label"], row["relationship_type"], row["target_label"])
        if row["source_label"] in ENTITY_LABELS and row["target_label"] in ENTITY_LABELS and row["relationship_type"] in RELATIONSHIP_TYPES:
            grouped.setdefault(key, []).append(row)

    for (source_label, relationship_type, target_label), rel_rows in grouped.items():
        run_batched(
            session,
            f"""UNWIND $rows AS r
                MATCH (a:{source_label} {{name: r.source}})
                MATCH (b:{target_label} {{name: r.target}})
                MERGE (a)-[rel:{relationship_type}]->(b)
                SET rel.confidence = r.confidence,
                    rel.reason = r.reason,
                    rel.source = r.source_dataset,
                    rel.weight = r.confidence""",
            rel_rows,
        )


def entity_row(name: str, source: str, metadata_category: str | None = None) -> dict[str, Any]:
    label = classify_entity(name, metadata_category)
    return {"name": normalize(name), "label": label, "source": source}


def rel_row(
    source: str,
    source_label: str,
    relationship_type: str,
    target: str,
    target_label: str,
    reason: str,
    confidence: float,
    source_dataset: str,
) -> dict[str, Any]:
    return {
        "source": normalize(source),
        "source_label": source_label,
        "relationship_type": relationship_type,
        "target": normalize(target),
        "target_label": target_label,
        "reason": reason,
        "confidence": round(confidence, 3),
        "source_dataset": source_dataset,
    }


def build_profile_graph(session: Any) -> None:
    df = pd.read_csv(PROFILES_CSV, dtype=str)
    df = df.where(pd.notna(df), None)
    candidate_rows: list[dict[str, Any]] = []
    entity_rows: list[dict[str, Any]] = []
    rel_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        candidate_id = str(row["id"]).strip()
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "years_of_experience": float(row.get("years_of_experience") or 0.0),
                "summary": str(row.get("skill_summary") or "")[:2000],
            }
        )
        metadata = parse_json_dict(row.get("skill_metadata"))
        for skill in split_csv_field(row.get("all_skills")):
            category = metadata.get(skill, {}).get("category") if isinstance(metadata.get(skill), dict) else None
            label = classify_entity(skill, category)
            entity_rows.append(entity_row(skill, "profile", category))
            rel_type = "HAS_SKILL"
            if label == "Framework":
                rel_type = "USES_FRAMEWORK"
            elif label == "Tool":
                rel_type = "USES_TOOL"
            elif label == "Database":
                rel_type = "USES_DATABASE"
            elif label == "CloudPlatform":
                rel_type = "USES_CLOUD"
            elif label == "Certification":
                rel_type = "CERTIFIED_IN"
            elif label == "SoftSkill":
                rel_type = "HAS_SKILL"
            rel_rows.append(
                {
                    "candidate_id": candidate_id,
                    "target": skill,
                    "target_label": label,
                    "relationship_type": rel_type,
                    "confidence": 0.85 if label != "SoftSkill" else 0.55,
                    "reason": f"Candidate profile lists {skill}",
                }
            )
        for role in split_csv_field(row.get("potential_roles")):
            entity_rows.append(entity_row(role, "profile"))
            rel_rows.append(
                {
                    "candidate_id": candidate_id,
                    "target": role,
                    "target_label": "Role",
                    "relationship_type": "WORKED_AS",
                    "confidence": 0.70,
                    "reason": "Role inferred from candidate profile",
                }
            )

    run_batched(
        session,
        """UNWIND $rows AS r
           MERGE (c:Candidate {candidate_id: r.candidate_id})
           SET c.years_of_experience = r.years_of_experience,
               c.summary = r.summary""",
        candidate_rows,
    )
    merge_entities(session, entity_rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rel_rows:
        if row["target_label"] in ENTITY_LABELS and row["relationship_type"] in RELATIONSHIP_TYPES:
            grouped.setdefault((row["target_label"], row["relationship_type"]), []).append(row)
    for (label, relationship_type), rows in grouped.items():
        run_batched(
            session,
            f"""UNWIND $rows AS r
                MATCH (c:Candidate {{candidate_id: r.candidate_id}})
                MATCH (e:{label} {{name: r.target}})
                MERGE (c)-[rel:{relationship_type}]->(e)
                SET rel.confidence = r.confidence,
                    rel.reason = r.reason""",
            rows,
        )
    print(f"  Candidates: {len(candidate_rows)} | candidate relationships: {len(rel_rows)}")


def build_demand_graph(session: Any) -> None:
    df = pd.read_csv(DEMANDS_CSV, dtype=str)
    df = df.where(pd.notna(df), None)
    job_rows: list[dict[str, Any]] = []
    entity_rows: list[dict[str, Any]] = []
    rels_by_label: dict[str, list[dict[str, Any]]] = {}

    for _, row in df.iterrows():
        job_id = str(row["id"]).strip()
        title = normalize(row.get("job_title") or "")
        job_rows.append(
            {
                "job_id": job_id,
                "job_title": title,
                "experience_lower": float(row.get("experience_lower") or 0.0),
                "experience_upper": float(row.get("experience_upper") or 0.0),
            }
        )
        if title:
            entity_rows.append(entity_row(title, "demand"))
            rels_by_label.setdefault("Role", []).append(
                {"job_id": job_id, "target": title, "relationship_type": "MATCHES_ROLE", "confidence": 0.82, "reason": "Job title maps to role"}
            )
        city = normalize(row.get("city") or row.get("location") or "")
        if city:
            entity_rows.append({"name": city, "label": "Location", "source": "demand"})
            rels_by_label.setdefault("Location", []).append(
                {"job_id": job_id, "target": city, "relationship_type": "LOCATED_IN", "confidence": 0.95, "reason": "Job location"}
            )
        metadata = parse_json_dict(row.get("skill_metadata"))
        for skill in split_csv_field(row.get("all_skills")):
            category = metadata.get(skill, {}).get("category") if isinstance(metadata.get(skill), dict) else None
            label = classify_entity(skill, category)
            entity_rows.append(entity_row(skill, "demand", category))
            rel_type = {
                "Framework": "REQUIRES_FRAMEWORK",
                "Tool": "REQUIRES_TOOL",
                "Database": "REQUIRES_DATABASE",
                "CloudPlatform": "REQUIRES_CLOUD",
            }.get(label, "REQUIRES_SKILL")
            rels_by_label.setdefault(label, []).append(
                {"job_id": job_id, "target": skill, "relationship_type": rel_type, "confidence": 0.88, "reason": "Job posting explicitly requires entity"}
            )

    run_batched(
        session,
        """UNWIND $rows AS r
           MERGE (j:Job {job_id: r.job_id})
           SET j.job_title = r.job_title,
               j.experience_lower = r.experience_lower,
               j.experience_upper = r.experience_upper""",
        job_rows,
    )
    merge_entities(session, entity_rows)
    for label, rows in rels_by_label.items():
        rel_type = rows[0]["relationship_type"] if rows else "REQUIRES_SKILL"
        if rel_type not in RELATIONSHIP_TYPES:
            rel_type = "REQUIRES_SKILL"
        run_batched(
            session,
            f"""UNWIND $rows AS r
                MATCH (j:Job {{job_id: r.job_id}})
                MATCH (e:{label} {{name: r.target}})
                MERGE (j)-[rel:{rel_type}]->(e)
                SET rel.confidence = r.confidence,
                    rel.reason = r.reason""",
            rows,
        )
    print(f"  Jobs: {len(job_rows)}")


def build_jd_graph(session: Any) -> None:
    df = pd.read_csv(JD_CSV, dtype=str)
    df = df.where(pd.notna(df), None)
    jd_rows: list[dict[str, Any]] = []
    entity_rows: list[dict[str, Any]] = []
    rels_by_label: dict[str, list[dict[str, Any]]] = {}

    for _, row in df.iterrows():
        jd_id = str(row["jd_id"]).strip()
        title = normalize(row.get("job_title") or "")
        industry = normalize(row.get("industry") or "")
        jd_rows.append({"jd_id": jd_id, "job_title": title, "industry": industry})
        if title:
            entity_rows.append(entity_row(title, "jd"))
            rels_by_label.setdefault("Role", []).append(
                {"jd_id": jd_id, "target": title, "relationship_type": "MATCHES_ROLE", "confidence": 0.82, "reason": "JD title maps to role"}
            )
        if industry:
            entity_rows.append({"name": industry, "label": "Industry", "source": "jd"})
            rels_by_label.setdefault("Industry", []).append(
                {"jd_id": jd_id, "target": industry, "relationship_type": "BELONGS_TO_DOMAIN", "confidence": 0.90, "reason": "JD industry"}
            )
        for field, confidence in (("must_have_skills", 0.90), ("preferred_skills", 0.76), ("technologies", 0.82), ("tools", 0.82), ("certifications", 0.72), ("education", 0.70)):
            for entity in short_entities(parse_list_field(row.get(field))):
                label = classify_entity(entity, field)
                entity_rows.append(entity_row(entity, "jd", field))
                rel_type = {
                    "Framework": "REQUIRES_FRAMEWORK",
                    "Tool": "REQUIRES_TOOL",
                    "Database": "REQUIRES_DATABASE",
                    "CloudPlatform": "REQUIRES_CLOUD",
                    "Certification": "CERTIFIED_IN",
                    "Education": "HAS_EXPERIENCE",
                }.get(label, "REQUIRES_SKILL")
                rels_by_label.setdefault(label, []).append(
                    {"jd_id": jd_id, "target": entity, "relationship_type": rel_type, "confidence": confidence, "reason": f"JD {field} lists entity"}
                )

    run_batched(
        session,
        """UNWIND $rows AS r
           MERGE (d:JobDescription {jd_id: r.jd_id})
           SET d.job_title = r.job_title,
               d.industry = r.industry""",
        jd_rows,
    )
    merge_entities(session, entity_rows)
    for label, rows in rels_by_label.items():
        rel_type = rows[0]["relationship_type"] if rows else "REQUIRES_SKILL"
        if rel_type not in RELATIONSHIP_TYPES:
            rel_type = "REQUIRES_SKILL"
        run_batched(
            session,
            f"""UNWIND $rows AS r
                MATCH (d:JobDescription {{jd_id: r.jd_id}})
                MATCH (e:{label} {{name: r.target}})
                MERGE (d)-[rel:{rel_type}]->(e)
                SET rel.confidence = r.confidence,
                    rel.reason = r.reason""",
            rows,
        )
    print(f"  JobDescriptions: {len(jd_rows)}")


def build_semantic_stack(session: Any) -> None:
    entity_rows = []
    relationship_rows = []
    for edge in semantic_stack_edges():
        entity_rows.append({"name": edge["source"], "label": edge["source_type"], "source": "ontology"})
        entity_rows.append({"name": edge["target"], "label": edge["target_type"], "source": "ontology"})
        relationship_rows.append(
            rel_row(
                edge["source"],
                edge["source_type"],
                edge["relationship_type"],
                edge["target"],
                edge["target_type"],
                edge["reason"],
                edge["confidence"],
                "ontology",
            )
        )
    merge_entities(session, entity_rows)
    merge_relationships(session, relationship_rows)
    print(f"  Semantic ontology edges: {len(relationship_rows)}")


def build_graph() -> None:
    print("=" * 60)
    print("Semantic Knowledge Graph Builder")
    print("=" * 60)

    if not verify_connection():
        print("ERROR: Cannot connect to Neo4j.", file=sys.stderr)
        sys.exit(1)

    create_schema()
    with get_session() as session:
        build_profile_graph(session)
        build_demand_graph(session)
        build_jd_graph(session)
        build_semantic_stack(session)

    summary = get_schema_summary()
    print("\nGraph Build Complete")
    print("Nodes:")
    for label, count in summary["nodes"].items():
        print(f"  {label:<22}: {count:>6}")
    print("Relationships:")
    for rel, count in summary["relationships"].items():
        print(f"  {rel:<22}: {count:>6}")


def main() -> int:
    build_graph()
    return 0


if __name__ == "__main__":
    sys.exit(main())
