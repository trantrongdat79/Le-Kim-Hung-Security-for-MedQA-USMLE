"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_LLM_1_MODEL= "deepseek/deepseek-v4-flash"
DEFAULT_LLM_1_API_BASE= "https://openrouter.ai/api/v1"

DEFAULT_LLM_2_MODEL= "cx/gpt-5.4-mini"
DEFAULT_LLM_2_API_BASE= "http://localhost:20128/v1"

DEFAULT_LLM_3_MODEL= "deepseek/deepseek-v4-flash"
DEFAULT_LLM_3_API_BASE= "https://openrouter.ai/api/v1"

@dataclass(frozen=True)
class LLMConfig:
    model: str
    api_base: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 1024


def load_llm_1_config() -> LLMConfig:
    """Load the config of LLM 1."""

    load_dotenv()

    api_key = os.getenv("LLM_1_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing required environment variable: LLM_1_API_KEY")

    return LLMConfig(
        model=os.getenv("LLM_1_MODEL", DEFAULT_LLM_1_MODEL).strip(),
        api_base=os.getenv("LLM_1_API_BASE", DEFAULT_LLM_1_API_BASE).strip(),
        api_key=api_key,
    )


def load_llm_2_config() -> LLMConfig:
    """Load the config of LLM 2."""

    load_dotenv()

    api_key = os.getenv("LLM_2_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing required environment variable: LLM_2_API_KEY")

    return LLMConfig(
        model=os.getenv("LLM_2_MODEL", DEFAULT_LLM_2_MODEL).strip(),
        api_base=os.getenv("LLM_2_API_BASE", DEFAULT_LLM_2_API_BASE).strip(),
        api_key=api_key,
    )


def load_llm_3_config() -> LLMConfig:
    """Load the config of LLM 3."""

    load_dotenv()

    api_key = os.getenv("LLM_3_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing required environment variable: LLM_3_API_KEY")

    return LLMConfig(
        model=os.getenv("LLM_3_MODEL", DEFAULT_LLM_3_MODEL).strip(),
        api_base=os.getenv("LLM_3_API_BASE", DEFAULT_LLM_3_API_BASE).strip(),
        api_key=api_key,
    )