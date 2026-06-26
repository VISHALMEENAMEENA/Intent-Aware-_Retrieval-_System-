"""
clean_jd.py
===========
Intent-Aware and Explainable Hybrid Retrieval System
------------------------------------------------------
Responsibility  : Load every raw JD markdown file, parse it via
                  parse_sections.structure_jd(), normalise all fields,
                  build derived fields, and write jd_cleaned.csv.

Design Principles
-----------------
- All parsing logic stays in parse_sections.py.
- All cleaning / normalisation logic lives here.
- Functions are small, named, and independently testable.
- The output CSV schema is fixed and explicitly enforced.

Output Schema (per row)
-----------------------
    jd_id, job_title, industry, city, country, location,
    responsibilities, must_have_skills, preferred_skills,
    soft_skills, technologies, tools, education, experience,
    certifications, other_requirements,
    retrieval_context, metadata, job_text
"""

import os
import json
import re

import pandas as pd

from parse_one_jd import parse_markdown
from parse_sections import structure_jd


# ==============================================================
# PATHS
# ==============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_JD_DIR = os.path.join(BASE_DIR, "data", "raw", "jd_dataset")

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "jd_cleaned.csv")


# ==============================================================
# FINAL COLUMN ORDER
# Must match the spec exactly.
# ==============================================================

EXPECTED_COLUMNS = [
    "jd_id",
    "job_title",
    "industry",
    "city",
    "country",
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
    "retrieval_context",
    "metadata",
    "job_text",
]


# ==============================================================
# ❶ TEXT CLEANING HELPERS
# ==============================================================

def clean_text(text) -> str:
    """
    Normalise a scalar string:
        - Cast to str
        - Replace newlines with spaces
        - Collapse multiple spaces
        - Strip edges
        - Lowercase

    Returns "" for None / empty.
    """
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def normalize_list(items: list) -> list:
    """
    For a list of strings:
        - Clean each item with clean_text()
        - Drop empty strings
        - Deduplicate while preserving insertion order

    Returns a clean list ready for JSON serialisation.
    """
    seen    = set()
    cleaned = []
    for item in items:
        norm = clean_text(item)
        if norm and norm not in seen:
            cleaned.append(norm)
            seen.add(norm)
    return cleaned


# ==============================================================
# ❷ RETRIEVAL CONTEXT BUILDER
# ==============================================================

def build_retrieval_context(
    responsibilities:   list,
    must_have_skills:   list,
    preferred_skills:   list,
    soft_skills:        list,
    technologies:       list,
    tools:              list,
    education:          list,
    experience:         list,
    other_requirements: list,
) -> list:
    """
    Merge all semantic content into one flat, deduplicated list.

    This field replaces the old `all_keywords` field.
    It is NOT just keywords — it is the *full retrieval context*
    that BM25 / FAISS / OpenSearch will index against:

        responsibilities + must_have_skills + preferred_skills
        + soft_skills + technologies + tools
        + education + experience + other_requirements

    Deduplication is case-insensitive.
    """
    retrieval_context = []
    seen = set()

    all_sources = (
        responsibilities
        + must_have_skills
        + preferred_skills
        + soft_skills
        + technologies
        + tools
        + education
        + experience
        + other_requirements
    )

    for item in all_sources:
        norm = item.strip()
        key  = norm.lower()
        if norm and key not in seen:
            retrieval_context.append(norm)
            seen.add(key)

    return retrieval_context


# ==============================================================
# ❸ METADATA JSON BUILDER
# ==============================================================

def build_metadata(
    must_have_skills:   list,
    preferred_skills:   list,
    soft_skills:        list,
    certifications:     list,
    technologies:       list,
    tools:              list,
    education:          list,
    experience:         list,
    other_requirements: list,
) -> str:
    """
    Build a structured category metadata JSON object.

    Every unique item across all classified fields gets an entry
    recording which semantic category it belongs to.

    Example output:
        {
            "python":                   {"category": "technology"},
            "communication":            {"category": "soft_skill"},
            "aws certified developer":  {"category": "certification"},
            "bachelor's degree":        {"category": "education"},
            "5+ years of experience":   {"category": "experience"},
            "docker":                   {"category": "tool"},
            "react":                    {"category": "preferred_skill"},
        }

    If the same string appears in multiple categories, the first
    category assigned wins (priority order matches the loop below).
    """
    meta = {}

    category_sources = [
        ("must_have_skill",  must_have_skills),
        ("preferred_skill",  preferred_skills),
        ("soft_skill",       soft_skills),
        ("certification",    certifications),
        ("technology",       technologies),
        ("tool",             tools),
        ("education",        education),
        ("experience",       experience),
        ("other_requirement", other_requirements),
    ]

    for category, items in category_sources:
        for item in items:
            if item and item not in meta:
                meta[item] = {"category": category}

    return json.dumps(meta, ensure_ascii=False)


# ==============================================================
# ❹ JOB TEXT BUILDER  (Sentence-BERT embedding input)
# ==============================================================

def build_job_text(jd: dict) -> str:
    """
    Construct a rich, structured natural-language representation of the JD.

    This string is what Sentence-BERT will embed into a dense vector.
    Every semantically meaningful field is included so the embedding
    captures the full intent of the role.

    Sections included:
        Job Title | Industry | Location
        Responsibilities | Must Have Skills | Preferred Skills
        Soft Skills | Technologies | Tools
        Education | Experience | Certifications
        Other Requirements

    The text is cleaned (lowercased, whitespace-normalised) before return.
    """

    def _join(items: list) -> str:
        """Join a list into a readable sentence fragment."""
        return ". ".join(i for i in items if i)

    def _section(label: str, content: str) -> str:
        """Format a named section only if content is non-empty."""
        if content.strip():
            return f"{label}:\n{content}"
        return ""

    parts = [
        _section("Job Title",        jd["job_title"]),
        _section("Industry",         jd["industry"]),
        _section("Location",         f"{jd['city']}, {jd['country']}".strip(", ")),
        _section("Responsibilities", _join(jd["responsibilities"])),
        _section("Must Have Skills", _join(jd["must_have_skills"])),
        _section("Preferred Skills", _join(jd["preferred_skills"])),
        _section("Soft Skills",      _join(jd["soft_skills"])),
        _section("Technologies",     _join(jd["technologies"])),
        _section("Tools",            _join(jd["tools"])),
        _section("Education",        _join(jd["education"])),
        _section("Experience",       _join(jd["experience"])),
        _section("Certifications",   _join(jd["certifications"])),
        _section("Other Requirements", _join(jd["other_requirements"])),
    ]

    text = "\n\n".join(p for p in parts if p)
    return clean_text(text)


# ==============================================================
# ❺ SINGLE JD PROCESSOR
# ==============================================================

def process_one_jd(folder_name: str, md_file: str) -> dict:
    """
    Full pipeline for a single JD markdown file:
        read → parse_markdown → structure_jd → normalise → build derived fields

    Returns a flat record dict matching EXPECTED_COLUMNS.
    """
    with open(md_file, "r", encoding="utf-8") as f:
        markdown = f.read()

    # ── Parse markdown into section dict ──────────────────────
    parsed = parse_markdown(markdown)

    # ── Structure sections into classified fields ──────────────
    jd = structure_jd(parsed)

    # ── Normalise all list fields ─────────────────────────────
    responsibilities   = normalize_list(jd["responsibilities"])
    must_have_skills   = normalize_list(jd["must_have_skills"])
    preferred_skills   = normalize_list(jd["preferred_skills"])
    soft_skills        = normalize_list(jd["soft_skills"])
    technologies       = normalize_list(jd["technologies"])
    tools              = normalize_list(jd["tools"])
    education          = normalize_list(jd["education"])
    experience         = normalize_list(jd["experience"])
    certifications     = normalize_list(jd["certifications"])
    other_requirements = normalize_list(jd["other_requirements"])

    # ── Normalise scalar fields ────────────────────────────────
    job_title = clean_text(jd["job_title"])
    industry  = clean_text(jd["industry"])
    city      = clean_text(jd["city"])
    country   = clean_text(jd["country"])

    # ── Build derived fields ──────────────────────────────────
    retrieval_context = build_retrieval_context(
        responsibilities,
        must_have_skills,
        preferred_skills,
        soft_skills,
        technologies,
        tools,
        education,
        experience,
        other_requirements,
    )

    metadata = build_metadata(
        must_have_skills,
        preferred_skills,
        soft_skills,
        certifications,
        technologies,
        tools,
        education,
        experience,
        other_requirements,
    )

    job_text = build_job_text({
        "job_title":          job_title,
        "industry":           industry,
        "city":               city,
        "country":            country,
        "responsibilities":   responsibilities,
        "must_have_skills":   must_have_skills,
        "preferred_skills":   preferred_skills,
        "soft_skills":        soft_skills,
        "technologies":       technologies,
        "tools":              tools,
        "education":          education,
        "experience":         experience,
        "certifications":     certifications,
        "other_requirements": other_requirements,
    })

    # ── Assemble flat record ──────────────────────────────────
    record = {
        "jd_id":              folder_name,
        "job_title":          job_title,
        "industry":           industry,
        "city":               city,
        "country":            country,
        "location":           clean_text(f"{city}, {country}").strip(", "),
        "responsibilities":   json.dumps(responsibilities,   ensure_ascii=False),
        "must_have_skills":   json.dumps(must_have_skills,   ensure_ascii=False),
        "preferred_skills":   json.dumps(preferred_skills,   ensure_ascii=False),
        "soft_skills":        json.dumps(soft_skills,        ensure_ascii=False),
        "technologies":       json.dumps(technologies,       ensure_ascii=False),
        "tools":              json.dumps(tools,              ensure_ascii=False),
        "education":          json.dumps(education,          ensure_ascii=False),
        "experience":         json.dumps(experience,         ensure_ascii=False),
        "certifications":     json.dumps(certifications,     ensure_ascii=False),
        "other_requirements": json.dumps(other_requirements, ensure_ascii=False),
        "retrieval_context":  json.dumps(retrieval_context,  ensure_ascii=False),
        "metadata":           metadata,
        "job_text":           job_text,
    }

    return record


# ==============================================================
# ❻ MAIN
# ==============================================================

def main():

    records = []

    # Sort numerically so output rows are in natural order (1, 2, ... 289)
    folders = sorted(
        os.listdir(RAW_JD_DIR),
        key=lambda x: int(x) if x.isdigit() else x
    )

    print("=" * 60)
    print("  Cleaning JD Dataset")
    print("=" * 60)

    errors = []

    for folder in folders:

        md_file = os.path.join(RAW_JD_DIR, folder, "enhanced_job_description.md")

        if not os.path.exists(md_file):
            continue

        try:
            record = process_one_jd(folder, md_file)
            records.append(record)
            print(f"  [OK]  JD {folder:>4}")

        except Exception as exc:
            errors.append((folder, str(exc)))
            print(f"  [ERR] JD {folder:>4}  →  {exc}")

    # ── Build DataFrame ───────────────────────────────────────
    df = pd.DataFrame(records)

    # Guarantee all expected columns exist in the right order
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[EXPECTED_COLUMNS]

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # ── Summary ───────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Completed Successfully")
    print("=" * 60)
    print(f"  Saved     : {OUTPUT_FILE}")
    print(f"  Total JDs : {len(df)}")

    if errors:
        print(f"  Errors    : {len(errors)}")
        for folder, msg in errors:
            print(f"    JD {folder}: {msg}")


if __name__ == "__main__":
    main()