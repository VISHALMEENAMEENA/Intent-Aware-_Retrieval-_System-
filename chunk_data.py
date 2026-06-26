"""
chunk_data.py
=============
Intent-Aware and Explainable Hybrid Retrieval System
----------------------------------------------------
Create semantic retrieval chunks from cleaned profiles, demands, and job
descriptions.

This script intentionally avoids fixed-size token chunking. Profiles and
demands remain atomic because each row already represents one retrieval unit.
Job descriptions are split by semantic sections so downstream dense retrieval,
BM25, RRF, graph linking, reranking, and explanations can work with meaningful
evidence units.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "chunks"

CHUNK_COLUMNS = [
    "chunk_id",
    "parent_id",
    "source",
    "chunk_type",
    "title",
    "location",
    "industry",
    "text",
    "metadata",
]

PROFILE_REQUIRED_COLUMNS = [
    "id",
    "profile_text",
    "years_of_experience",
    "all_skills",
    "potential_roles",
]

DEMAND_REQUIRED_COLUMNS = [
    "id",
    "job_title",
    "location",
    "all_skills",
    "job_text",
]

JD_REQUIRED_COLUMNS = [
    "jd_id",
    "job_title",
    "industry",
    "location",
    "responsibilities",
    "must_have_skills",
    "preferred_skills",
    "soft_skills",
    "technologies",
    "tools",
    "education",
    "experience",
    "certifications",
    "other_requirements",
]


def clean_scalar(value: Any) -> str:
    """Return a compact string for CSV values, treating nulls as empty."""
    if value is None or pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_identifier(value: Any) -> str:
    """Return a stable identifier without pandas numeric artifacts."""
    if value is None or pd.isna(value):
        return ""

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return clean_scalar(value)


def split_comma_values(value: Any) -> list[str]:
    """Split a comma-separated cleaned field into ordered unique values."""
    text = clean_scalar(value)
    if not text:
        return []

    values: list[str] = []
    seen: set[str] = set()

    for item in text.split(","):
        cleaned = clean_scalar(item)
        key = cleaned.lower()
        if cleaned and key not in seen:
            values.append(cleaned)
            seen.add(key)

    return values


def parse_section_items(value: Any) -> list[str]:
    """
    Parse semantic section fields from cleaned JD CSV rows.

    The cleaned JD data stores sections as JSON lists in CSV cells. This helper
    also accepts Python literal lists and simple fallback strings so the
    chunking step remains resilient if the cleaning pipeline evolves later.
    """
    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        raw_items = value
    else:
        text = clean_scalar(value)
        if not text or text == "[]":
            return []

        raw_items = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue

            if isinstance(parsed, list):
                raw_items = parsed
                break

        if raw_items is None:
            raw_items = [text]

    items: list[str] = []
    seen: set[str] = set()

    for item in raw_items:
        cleaned = clean_scalar(item)
        key = cleaned.lower()
        if cleaned and key not in seen:
            items.append(cleaned)
            seen.add(key)

    return items


def ensure_required_columns(df: pd.DataFrame, required_columns: list[str], dataset_name: str) -> None:
    """Raise a clear error if a cleaned input file is missing required fields."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def metadata_json(**values: Any) -> str:
    """Serialize metadata consistently for downstream retrieval systems."""
    return json.dumps(values, ensure_ascii=False, sort_keys=True)


def build_chunk(
    *,
    chunk_id: str,
    parent_id: str,
    source: str,
    chunk_type: str,
    title: str,
    location: str,
    industry: str,
    text: str,
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Create one row in the unified chunk schema."""
    return {
        "chunk_id": clean_scalar(chunk_id),
        "parent_id": clean_scalar(parent_id),
        "source": source,
        "chunk_type": chunk_type,
        "title": clean_scalar(title),
        "location": clean_scalar(location),
        "industry": clean_scalar(industry),
        "text": clean_scalar(text),
        "metadata": metadata_json(**metadata),
    }


def format_labeled_values(fields: list[tuple[str, str]]) -> str:
    """Format scalar fields as readable labeled text."""
    sections = []

    for label, value in fields:
        cleaned = clean_scalar(value)
        if cleaned:
            sections.append(f"{label}:\n{cleaned}")

    return "\n\n".join(sections)


def format_labeled_list_sections(sections: list[tuple[str, list[str]]]) -> str:
    """Format semantic list sections as readable labeled bullet lists."""
    formatted_sections = []

    for label, items in sections:
        cleaned_items = [clean_scalar(item) for item in items if clean_scalar(item)]
        if not cleaned_items:
            continue

        bullets = "\n".join(f"- {item}" for item in cleaned_items)
        formatted_sections.append(f"{label}:\n{bullets}")

    return "\n\n".join(formatted_sections)


def load_cleaned_csv(cleaned_dir: Path, filename: str) -> pd.DataFrame:
    """Load one cleaned CSV file from the configured cleaned data directory."""
    path = cleaned_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing cleaned input file: {path}")

    return pd.read_csv(path)


def create_profile_chunks(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Create exactly one semantic chunk per candidate profile."""
    ensure_required_columns(profiles_df, PROFILE_REQUIRED_COLUMNS, "profiles_cleaned.csv")

    chunks: list[dict[str, str]] = []

    for _, row in profiles_df.iterrows():
        candidate_id = normalize_identifier(row["id"])
        profile_text = clean_scalar(row["profile_text"])
        skills = split_comma_values(row["all_skills"])
        roles = split_comma_values(row["potential_roles"])
        experience = clean_scalar(row["years_of_experience"])
        role_title = ", ".join(roles)

        chunks.append(
            build_chunk(
                chunk_id=f"profile_{candidate_id}",
                parent_id=candidate_id,
                source="profile",
                chunk_type="profile",
                title=role_title,
                location="",
                industry="",
                text=profile_text,
                metadata={
                    "source": "profile",
                    "parent_id": candidate_id,
                    "chunk_type": "profile",
                    "candidate_id": candidate_id,
                    "experience": experience,
                    "skills": skills,
                    "roles": roles,
                },
            )
        )

    return pd.DataFrame(chunks, columns=CHUNK_COLUMNS)


def create_demand_chunks(demands_df: pd.DataFrame) -> pd.DataFrame:
    """Create exactly one semantic chunk per demand row."""
    ensure_required_columns(demands_df, DEMAND_REQUIRED_COLUMNS, "demands_cleaned.csv")

    chunks: list[dict[str, str]] = []

    for _, row in demands_df.iterrows():
        demand_id = normalize_identifier(row["id"])
        job_title = clean_scalar(row["job_title"])
        location = clean_scalar(row["location"])
        skills = split_comma_values(row["all_skills"])
        job_text = clean_scalar(row["job_text"])

        chunks.append(
            build_chunk(
                chunk_id=f"demand_{demand_id}",
                parent_id=demand_id,
                source="demand",
                chunk_type="demand",
                title=job_title,
                location=location,
                industry="",
                text=job_text,
                metadata={
                    "source": "demand",
                    "parent_id": demand_id,
                    "chunk_type": "demand",
                    "job_title": job_title,
                    "location": location,
                    "skills": skills,
                },
            )
        )

    return pd.DataFrame(chunks, columns=CHUNK_COLUMNS)


def create_jd_header_chunk(row: pd.Series, jd_id: str) -> dict[str, str] | None:
    """Create the semantic header chunk for one JD when header content exists."""
    job_title = clean_scalar(row["job_title"])
    industry = clean_scalar(row["industry"])
    location = clean_scalar(row["location"])

    text = format_labeled_values(
        [
            ("Job Title", job_title),
            ("Industry", industry),
            ("Location", location),
        ]
    )

    if not text:
        return None

    return build_chunk(
        chunk_id=f"jd_{jd_id}_header",
        parent_id=jd_id,
        source="jd",
        chunk_type="jd_header",
        title=job_title,
        location=location,
        industry=industry,
        text=text,
        metadata={
            "source": "jd",
            "parent_id": jd_id,
            "chunk_type": "jd_header",
            "job_title": job_title,
            "industry": industry,
            "location": location,
        },
    )


def create_jd_responsibilities_chunk(row: pd.Series, jd_id: str) -> dict[str, str] | None:
    """Create the responsibilities chunk for one JD when content exists."""
    responsibilities = parse_section_items(row["responsibilities"])
    if not responsibilities:
        return None

    job_title = clean_scalar(row["job_title"])
    industry = clean_scalar(row["industry"])
    location = clean_scalar(row["location"])
    text = format_labeled_list_sections([("Responsibilities", responsibilities)])

    return build_chunk(
        chunk_id=f"jd_{jd_id}_responsibilities",
        parent_id=jd_id,
        source="jd",
        chunk_type="jd_responsibilities",
        title=job_title,
        location=location,
        industry=industry,
        text=text,
        metadata={
            "source": "jd",
            "parent_id": jd_id,
            "chunk_type": "jd_responsibilities",
            "job_title": job_title,
            "industry": industry,
            "location": location,
            "section": "responsibilities",
            "item_count": len(responsibilities),
        },
    )


def create_jd_skills_chunk(row: pd.Series, jd_id: str) -> dict[str, str] | None:
    """Create the merged skills chunk for one JD when skill content exists."""
    skill_sections = [
        ("Must Have Skills", parse_section_items(row["must_have_skills"])),
        ("Preferred Skills", parse_section_items(row["preferred_skills"])),
        ("Soft Skills", parse_section_items(row["soft_skills"])),
        ("Technologies", parse_section_items(row["technologies"])),
        ("Tools", parse_section_items(row["tools"])),
    ]

    if not any(items for _, items in skill_sections):
        return None

    job_title = clean_scalar(row["job_title"])
    industry = clean_scalar(row["industry"])
    location = clean_scalar(row["location"])
    text = format_labeled_list_sections(skill_sections)

    return build_chunk(
        chunk_id=f"jd_{jd_id}_skills",
        parent_id=jd_id,
        source="jd",
        chunk_type="jd_skills",
        title=job_title,
        location=location,
        industry=industry,
        text=text,
        metadata={
            "source": "jd",
            "parent_id": jd_id,
            "chunk_type": "jd_skills",
            "job_title": job_title,
            "industry": industry,
            "location": location,
            "must_have_skills": skill_sections[0][1],
            "preferred_skills": skill_sections[1][1],
            "soft_skills": skill_sections[2][1],
            "technologies": skill_sections[3][1],
            "tools": skill_sections[4][1],
        },
    )


def create_jd_qualification_chunk(row: pd.Series, jd_id: str) -> dict[str, str] | None:
    """Create the merged qualifications chunk for one JD when content exists."""
    qualification_sections = [
        ("Education", parse_section_items(row["education"])),
        ("Experience", parse_section_items(row["experience"])),
        ("Certifications", parse_section_items(row["certifications"])),
    ]

    if not any(items for _, items in qualification_sections):
        return None

    job_title = clean_scalar(row["job_title"])
    industry = clean_scalar(row["industry"])
    location = clean_scalar(row["location"])
    text = format_labeled_list_sections(qualification_sections)

    return build_chunk(
        chunk_id=f"jd_{jd_id}_qualification",
        parent_id=jd_id,
        source="jd",
        chunk_type="jd_qualification",
        title=job_title,
        location=location,
        industry=industry,
        text=text,
        metadata={
            "source": "jd",
            "parent_id": jd_id,
            "chunk_type": "jd_qualification",
            "job_title": job_title,
            "industry": industry,
            "location": location,
            "education": qualification_sections[0][1],
            "experience": qualification_sections[1][1],
            "certifications": qualification_sections[2][1],
        },
    )


def create_jd_other_chunk(row: pd.Series, jd_id: str) -> dict[str, str] | None:
    """Create the other requirements chunk for one JD when content exists."""
    other_requirements = parse_section_items(row["other_requirements"])
    if not other_requirements:
        return None

    job_title = clean_scalar(row["job_title"])
    industry = clean_scalar(row["industry"])
    location = clean_scalar(row["location"])
    text = format_labeled_list_sections([("Other Requirements", other_requirements)])

    return build_chunk(
        chunk_id=f"jd_{jd_id}_other",
        parent_id=jd_id,
        source="jd",
        chunk_type="jd_other",
        title=job_title,
        location=location,
        industry=industry,
        text=text,
        metadata={
            "source": "jd",
            "parent_id": jd_id,
            "chunk_type": "jd_other",
            "job_title": job_title,
            "industry": industry,
            "location": location,
            "section": "other_requirements",
            "item_count": len(other_requirements),
        },
    )


def create_jd_chunks(jd_df: pd.DataFrame) -> pd.DataFrame:
    """Create semantic section chunks for all job descriptions."""
    ensure_required_columns(jd_df, JD_REQUIRED_COLUMNS, "jd_cleaned.csv")

    chunks: list[dict[str, str]] = []
    chunk_builders = [
        create_jd_header_chunk,
        create_jd_responsibilities_chunk,
        create_jd_skills_chunk,
        create_jd_qualification_chunk,
        create_jd_other_chunk,
    ]

    for _, row in jd_df.iterrows():
        jd_id = normalize_identifier(row["jd_id"])

        for builder in chunk_builders:
            chunk = builder(row, jd_id)
            if chunk is not None:
                chunks.append(chunk)

    return pd.DataFrame(chunks, columns=CHUNK_COLUMNS)


def validate_unique_chunk_ids(*chunk_frames: pd.DataFrame) -> None:
    """Ensure chunk IDs are globally unique before writing outputs."""
    all_chunk_ids = pd.concat([frame["chunk_id"] for frame in chunk_frames], ignore_index=True)
    duplicate_ids = all_chunk_ids[all_chunk_ids.duplicated()].unique().tolist()

    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:10])
        raise ValueError(f"Duplicate chunk_id values found: {preview}")


def average_word_length(chunks_df: pd.DataFrame) -> float:
    """Calculate average chunk length in whitespace-tokenized words."""
    if chunks_df.empty:
        return 0.0

    word_counts = chunks_df["text"].fillna("").apply(lambda text: len(str(text).split()))
    return float(word_counts.mean())


def write_chunk_outputs(
    profiles_chunks: pd.DataFrame,
    demands_chunks: pd.DataFrame,
    jd_chunks: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Write per-source chunk files plus the unified all_chunks.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = pd.concat(
        [profiles_chunks, demands_chunks, jd_chunks],
        ignore_index=True,
    )

    profiles_chunks.to_csv(output_dir / "profiles_chunks.csv", index=False, encoding="utf-8")
    demands_chunks.to_csv(output_dir / "demands_chunks.csv", index=False, encoding="utf-8")
    jd_chunks.to_csv(output_dir / "jd_chunks.csv", index=False, encoding="utf-8")
    all_chunks.to_csv(output_dir / "all_chunks.csv", index=False, encoding="utf-8")

    return all_chunks


def print_statistics(
    profiles_chunks: pd.DataFrame,
    demands_chunks: pd.DataFrame,
    jd_chunks: pd.DataFrame,
    all_chunks: pd.DataFrame,
) -> None:
    """Print generation statistics for pipeline visibility."""
    print("\nChunk generation complete")
    print("-------------------------")
    print(f"Total profile chunks: {len(profiles_chunks)}")
    print(f"Total demand chunks:  {len(demands_chunks)}")
    print(f"Total JD chunks:      {len(jd_chunks)}")
    print(f"Total chunks:         {len(all_chunks)}")

    print("\nChunk type distribution:")
    distribution = all_chunks["chunk_type"].value_counts().sort_index()
    for chunk_type, count in distribution.items():
        print(f"  {chunk_type}: {count}")

    print(f"\nAverage chunk length (words): {average_word_length(all_chunks):.2f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create semantic retrieval chunks from cleaned project datasets."
    )
    parser.add_argument(
        "--cleaned-dir",
        type=Path,
        default=DEFAULT_CLEANED_DIR,
        help="Directory containing profiles_cleaned.csv, demands_cleaned.csv, and jd_cleaned.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where chunk CSV files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    """Load cleaned data, create semantic chunks, validate, write, and report."""
    args = parse_args()
    cleaned_dir = args.cleaned_dir.resolve()
    output_dir = args.output_dir.resolve()

    profiles_df = load_cleaned_csv(cleaned_dir, "profiles_cleaned.csv")
    demands_df = load_cleaned_csv(cleaned_dir, "demands_cleaned.csv")
    jd_df = load_cleaned_csv(cleaned_dir, "jd_cleaned.csv")

    profiles_chunks = create_profile_chunks(profiles_df)
    demands_chunks = create_demand_chunks(demands_df)
    jd_chunks = create_jd_chunks(jd_df)

    validate_unique_chunk_ids(profiles_chunks, demands_chunks, jd_chunks)

    all_chunks = write_chunk_outputs(
        profiles_chunks=profiles_chunks,
        demands_chunks=demands_chunks,
        jd_chunks=jd_chunks,
        output_dir=output_dir,
    )

    print_statistics(
        profiles_chunks=profiles_chunks,
        demands_chunks=demands_chunks,
        jd_chunks=jd_chunks,
        all_chunks=all_chunks,
    )


if __name__ == "__main__":
    main()
