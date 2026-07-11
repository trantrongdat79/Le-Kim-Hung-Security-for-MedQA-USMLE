"""Lightweight data containers shared by the agent and evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_CHOICES = {"A", "B", "C", "D"}


@dataclass
class MedQAExample:
    question: str
    options: dict[str, str]
    answer: str | None = None
    answer_idx: str | None = None
    meta_info: str | None = None
    metamap_phrases: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MedQAExample":
        known_fields = {
            "question",
            "options",
            "answer",
            "answer_idx",
            "meta_info",
            "metamap_phrases",
        }
        options = data.get("options") or {}
        return cls(
            question=str(data.get("question", "")).strip(),
            options={str(key).upper(): str(value) for key, value in options.items()},
            answer=_optional_str(data.get("answer")),
            answer_idx=normalize_choice(data.get("answer_idx")),
            meta_info=_optional_str(data.get("meta_info")),
            metamap_phrases=list(data.get("metamap_phrases") or []),
            extra={key: value for key, value in data.items() if key not in known_fields},
        )


@dataclass
class AgentPrediction:
    answer_idx: str | None
    answer: str | None = None
    explanation: str = ""
    raw_response: str | None = None
    is_valid: bool = True
    error: str | None = None


def normalize_choice(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    return normalized if normalized else None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
