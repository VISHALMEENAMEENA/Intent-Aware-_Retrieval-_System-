"""
llm/client.py
=============
Client initialization for Gemini Generative AI.
"""

from __future__ import annotations

from typing import Any
from llm.config import GEMINI_API_KEY, MODEL_NAME, validate_config

_model_cache: Any | None = None

def get_gemini_model() -> Any:
    """
    Get or configure and return the GenerativeModel client.
    Uses caching to avoid configuring genai repeatedly.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai is not installed. Install it to use Gemini, "
            "or rely on the local query-understanding fallback."
        ) from exc

    validate_config()
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Configure the model with default model name
    _model_cache = genai.GenerativeModel(
        model_name=MODEL_NAME
    )
    return _model_cache
