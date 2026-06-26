"""
graph/schema.py
===============
Idempotent creation of Neo4j uniqueness constraints and indexes.
Run once before building the graph (or safely re-run at any time).

Usage:
    python graph/schema.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.client import get_session, verify_connection

# ── Constraint & index definitions ────────────────────────────────────────────
CONSTRAINTS: list[str] = [
    # Uniqueness constraints — also automatically create an index
    "CREATE CONSTRAINT skill_name IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT framework_name IF NOT EXISTS FOR (f:Framework) REQUIRE f.name IS UNIQUE",
    "CREATE CONSTRAINT language_name IF NOT EXISTS FOR (l:ProgrammingLanguage) REQUIRE l.name IS UNIQUE",
    "CREATE CONSTRAINT tool_name IF NOT EXISTS FOR (t:Tool) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT database_name IF NOT EXISTS FOR (d:Database) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT cloud_name IF NOT EXISTS FOR (c:CloudPlatform) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT certification_name IF NOT EXISTS FOR (c:Certification) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT soft_skill_name IF NOT EXISTS FOR (s:SoftSkill) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT role_name  IF NOT EXISTS FOR (r:Role)  REQUIRE r.name IS UNIQUE",
    "CREATE CONSTRAINT education_name IF NOT EXISTS FOR (e:Education) REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT industry_name IF NOT EXISTS FOR (i:Industry) REQUIRE i.name IS UNIQUE",
    "CREATE CONSTRAINT location_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
    "CREATE CONSTRAINT candidate_id IF NOT EXISTS FOR (c:Candidate) REQUIRE c.candidate_id IS UNIQUE",
    "CREATE CONSTRAINT job_id IF NOT EXISTS FOR (j:Job) REQUIRE j.job_id IS UNIQUE",
    "CREATE CONSTRAINT jd_id IF NOT EXISTS FOR (d:JobDescription) REQUIRE d.jd_id IS UNIQUE",
]

INDEXES: list[str] = [
    # Extra lookup indexes (constraints already cover the unique properties)
    "CREATE INDEX skill_freq_idx IF NOT EXISTS FOR (s:Skill) ON (s.frequency)",
    "CREATE INDEX job_title_idx IF NOT EXISTS FOR (j:Job) ON (j.job_title)",
    "CREATE INDEX jd_title_idx IF NOT EXISTS FOR (d:JobDescription) ON (d.job_title)",
    "CREATE INDEX candidate_exp_idx IF NOT EXISTS FOR (c:Candidate) ON (c.years_of_experience)",
]


def create_schema() -> None:
    """Create all constraints and indexes idempotently."""
    with get_session() as session:
        for statement in CONSTRAINTS:
            session.run(statement)
        for statement in INDEXES:
            session.run(statement)
    print(f"Schema created: {len(CONSTRAINTS)} constraints, {len(INDEXES)} indexes.")


def drop_all_data() -> None:
    """
    DANGER: Wipe all nodes and relationships.
    Only used for development resets — not called during normal builds.
    """
    with get_session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("All graph data deleted.")


def get_schema_summary() -> dict:
    """Return counts of nodes and relationships currently in the graph."""
    with get_session() as session:
        node_counts: dict[str, int] = {}
        node_labels = (
            "Candidate", "Job", "JobDescription", "Skill", "Framework",
            "ProgrammingLanguage", "Tool", "Database", "CloudPlatform",
            "Certification", "SoftSkill", "Role", "Industry", "Location", "Education"
        )
        for label in node_labels:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()
            node_counts[label] = result["c"] if result else 0

        rel_counts: dict[str, int] = {}
        rel_types = (
            "HAS_SKILL", "USES_FRAMEWORK", "USES_TOOL", "USES_DATABASE", "USES_CLOUD",
            "REQUIRES_SKILL", "REQUIRES_FRAMEWORK", "REQUIRES_TOOL", "REQUIRES_DATABASE",
            "REQUIRES_CLOUD", "SIMILAR_ROLE", "LOCATED_IN", "WORKED_AS", "BELONGS_TO_DOMAIN",
            "PART_OF_STACK", "PREREQUISITE", "CERTIFIED_IN", "SPECIALIZES_IN", "HAS_EXPERIENCE",
            "MATCHES_ROLE", "FRAMEWORK_FOR_LANGUAGE", "SUPPORTED_BY_FRAMEWORK",
            "SUPPORTS_FRAMEWORK", "DATABASE_USED_WITH", "TOOL_FOR_LANGUAGE",
            "DEPLOYS_WITH_DATABASE", "DEPLOYS_TO_CLOUD"
        )
        for rel in rel_types:
            result = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()
            rel_counts[rel] = result["c"] if result else 0

    return {"nodes": node_counts, "relationships": rel_counts}


def main() -> int:
    if not verify_connection():
        print("ERROR: Cannot connect to Neo4j.", file=sys.stderr)
        return 1
    create_schema()
    summary = get_schema_summary()
    print("\nCurrent graph statistics:")
    print("  Nodes:")
    for label, count in summary["nodes"].items():
        print(f"    {label}: {count}")
    print("  Relationships:")
    for rel, count in summary["relationships"].items():
        print(f"    {rel}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
