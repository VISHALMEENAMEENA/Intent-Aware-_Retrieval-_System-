"""
llm/query_understanding.py
==========================
Main module exposing understand_query and CLI interface.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any
from llm.prompt import SYSTEM_INSTRUCTION
from llm.parser import parse_and_validate_response


KNOWN_SKILLS = {
    "python", "java", "javascript", "typescript", "react", "angular", "node",
    "fastapi", "django", "flask", "spring", "spring boot", "springboot",
    "sql", "mysql", "postgresql", "mongodb", "redis", "docker", "kubernetes",
    "aws", "azure", "gcp", "devops", "ci/cd", "selenium", "rest", "rest api",
    "microservices", "machine learning", "ml", "data science", "snowflake",
    "databricks", "spark", "airflow", "git", "linux", "terraform",
}

KNOWN_CITIES = {
    "pune", "bengaluru", "bangalore", "chennai", "hyderabad", "noida",
    "mumbai", "delhi", "gurgaon", "gurugram", "kolkata", "texas",
    "virginia", "manila", "taguig",
}


def _fallback_understand_query(query: str) -> dict[str, Any]:
    """Deterministic local fallback used when Gemini is unavailable."""
    q = query.lower()
    skills = [skill for skill in sorted(KNOWN_SKILLS, key=len, reverse=True) if skill in q]
    languages = [s for s in skills if s in {"python", "java", "javascript", "typescript"}]
    frameworks = [s for s in skills if s in {"fastapi", "django", "flask", "react", "angular", "spring", "spring boot", "springboot"}]
    tools = [s for s in skills if s in {"docker", "kubernetes", "git", "terraform", "selenium"}]
    technologies = [s for s in skills if s not in set(languages + frameworks + tools)]

    role = None
    role_patterns = [
        r"([a-z ]+\bdeveloper)",
        r"([a-z ]+\bengineer)",
        r"([a-z ]+\barchitect)",
        r"([a-z ]+\blead)",
        r"([a-z ]+\bmanager)",
    ]
    for pattern in role_patterns:
        match = re.search(pattern, q)
        if match:
            role = " ".join(match.group(1).split()[-4:])
            break

    city = next((city for city in KNOWN_CITIES if re.search(rf"\b{re.escape(city)}\b", q)), None)
    if city == "bangalore":
        city = "bengaluru"

    intent = "profile_search"
    if any(word in q for word in ("job", "opening", "requirement", "hiring", "jd", "description")):
        intent = "jd_search" if "jd" in q or "description" in q else "job_search"
    if any(word in q for word in ("candidate", "profile", "resume", "developer", "engineer")):
        intent = "profile_search"

    return {
        "intent": intent,
        "original_query": query,
        "role": role,
        "skills": skills,
        "technologies": technologies,
        "tools": tools,
        "frameworks": frameworks,
        "programming_languages": languages,
        "soft_skills": [],
        "certifications": [],
        "education": [],
        "industry": [],
        "location": {"city": city, "state": None, "country": "india" if city else None},
        "experience": {"min_years": None, "max_years": None},
        "filters": {"remote": None, "full_time": None, "internship": None},
    }

def understand_query(query: str) -> dict[str, Any]:
    """
    Convert a user's natural language query into a structured JSON dictionary.
    Exposes classification of intent and extraction of query entities.
    
    Args:
        query: Natural language search query.
        
    Returns:
        A dictionary matching the query understanding schema.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query string cannot be empty.")

    try:
        from llm.client import get_gemini_model
        import google.generativeai as genai
        from llm.config import MODEL_NAME

        get_gemini_model()
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_INSTRUCTION
        )
        response = model.generate_content(
            cleaned_query,
            generation_config={"response_mime_type": "application/json"}
        )
        response_text = response.text
    except Exception as error:
        print(f"[llm] Gemini unavailable, using local query parser: {error}", file=sys.stderr)
        return _fallback_understand_query(cleaned_query)

    # Parse and validate the response
    return parse_and_validate_response(response_text, cleaned_query)

def main() -> int:
    """CLI loop for interactive query understanding testing."""
    print("=" * 60)
    print("LLM Query Understanding CLI")
    print("=" * 60)
    
    try:
        # Check config right away so we give a clean error if API key is missing
        from llm.config import validate_config
        validate_config()
    except Exception as error:
        print(f"Configuration Error: {error}", file=sys.stderr)
        return 1

    print("Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        try:
            query = input("Enter Query:\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
            
        if not query:
            continue
            
        if query.lower() in ("exit", "quit"):
            print("Exiting.")
            break
            
        print("\nAnalyzing query with Gemini 2.5 Flash...")
        try:
            result = understand_query(query)
            print("\nOutput:")
            print(json.dumps(result, indent=4))
        except Exception as error:
            print(f"\nError: {error}", file=sys.stderr)
            
        print("-" * 60)
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
