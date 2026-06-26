"""
llm/parser.py
=============
Parser and validator for raw responses from the Gemini API.
"""

from __future__ import annotations

import json
from typing import Any

ALLOWED_INTENTS = {"profile_search", "job_search", "jd_search"}

def clean_response_text(text: str) -> str:
    """
    Remove accidental markdown fences or prefixes from LLM response.
    """
    cleaned = text.strip()
    
    # Remove code block fences if present
    if cleaned.startswith("```"):
        # Find the end of the first line (e.g., ```json or ```)
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline:].strip()
        else:
            cleaned = cleaned[3:].strip()
            
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
            
    return cleaned

def validate_and_normalize_schema(data: dict[str, Any], original_query: str) -> dict[str, Any]:
    """
    Validate fields and populate defaults for missing schema elements.
    Ensures the dictionary structure matches the specified schema exactly.
    """
    if not isinstance(data, dict):
        raise ValueError("Parsed JSON must be a dictionary object.")

    # Validate intent
    intent = data.get("intent")
    if intent not in ALLOWED_INTENTS:
        raise ValueError(
            f"Invalid intent '{intent}'. Must be one of: {', '.join(ALLOWED_INTENTS)}"
        )

    # Normalize structure to guarantee all fields are present
    normalized = {
        "intent": intent,
        "original_query": str(data.get("original_query", original_query)),
        "role": data.get("role") if data.get("role") is not None else None,
        "skills": list(data.get("skills", [])),
        "technologies": list(data.get("technologies", [])),
        "tools": list(data.get("tools", [])),
        "frameworks": list(data.get("frameworks", [])),
        "programming_languages": list(data.get("programming_languages", [])),
        "soft_skills": list(data.get("soft_skills", [])),
        "certifications": list(data.get("certifications", [])),
        "education": list(data.get("education", [])),
        "industry": list(data.get("industry", [])),
    }

    # Normalize location
    location_data = data.get("location") or {}
    if not isinstance(location_data, dict):
        location_data = {}
    normalized["location"] = {
        "city": location_data.get("city") if location_data.get("city") is not None else None,
        "state": location_data.get("state") if location_data.get("state") is not None else None,
        "country": location_data.get("country") if location_data.get("country") is not None else None,
    }

    # Normalize experience
    experience_data = data.get("experience") or {}
    if not isinstance(experience_data, dict):
        experience_data = {}
    
    # Parse years as float/int or keep as null
    def parse_years(val: Any) -> float | int | None:
        if val is None or val == "":
            return None
        try:
            if isinstance(val, (int, float)):
                return val
            # Try to cast string
            if "." in str(val):
                return float(val)
            return int(val)
        except (ValueError, TypeError):
            return None

    normalized["experience"] = {
        "min_years": parse_years(experience_data.get("min_years")),
        "max_years": parse_years(experience_data.get("max_years")),
    }

    # Normalize filters
    filters_data = data.get("filters") or {}
    if not isinstance(filters_data, dict):
        filters_data = {}
        
    def parse_bool(val: Any) -> bool | None:
        if val is None or val == "":
            return None
        if isinstance(val, bool):
            return val
        if str(val).lower() in ("true", "1", "yes"):
            return True
        if str(val).lower() in ("false", "0", "no"):
            return False
        return None

    normalized["filters"] = {
        "remote": parse_bool(filters_data.get("remote")),
        "full_time": parse_bool(filters_data.get("full_time")),
        "internship": parse_bool(filters_data.get("internship")),
    }

    return normalized

def parse_and_validate_response(text: str, original_query: str) -> dict[str, Any]:
    """
    Clean, parse, and validate LLM output.
    """
    cleaned_text = clean_response_text(text)
    try:
        data = json.loads(cleaned_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Failed to parse Gemini response as JSON. Raw response: {text}"
        ) from error

    return validate_and_normalize_schema(data, original_query)
