"""
llm/config.py
=============
Configuration settings for the LLM Query Understanding module.
Loads the Gemini API key from api/llm.env.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Define directory paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE_PATH = PROJECT_ROOT / "api" / "llm.env"

# Load the environment variables from api/llm.env
if ENV_FILE_PATH.exists():
    load_dotenv(dotenv_path=ENV_FILE_PATH)
else:
    # Fallback to system env if llm.env does not exist
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API")
MODEL_NAME = "gemini-2.5-flash"

def validate_config() -> None:
    """Validate that required environment variables are present."""
    if not GEMINI_API_KEY:
        raise ValueError(
            f"GEMINI_API_KEY (or GEMINI_API) is not set. Please ensure it is defined in "
            f"'{ENV_FILE_PATH}' or as a system environment variable."
        )
