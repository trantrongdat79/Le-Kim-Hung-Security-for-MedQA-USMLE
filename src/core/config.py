"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_1_MODEL = "deepseek/deepseek-v4-flash"


@dataclass(frozen=True)
class LLMConfig:
    model: str
    api_base: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 512


def load_llm_1_config() -> LLMConfig:
    """Load the primary LLM config used by Sprint 1."""

    load_dotenv()

    api_key = os.getenv("LLM_1_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing required environment variable: LLM_1_API_KEY")

    return LLMConfig(
        model=os.getenv("LLM_1_MODEL", DEFAULT_LLM_1_MODEL).strip(),
        api_base=os.getenv("LLM_1_API_BASE", DEFAULT_OPENROUTER_BASE_URL).strip(),
        api_key=api_key,
    )
