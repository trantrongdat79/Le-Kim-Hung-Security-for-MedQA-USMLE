"""Shared data structures for MedQA examples and agent predictions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

AnswerChoice = Literal["A", "B", "C", "D"]
VALID_CHOICES = {"A", "B", "C", "D"}


class MedQAExample(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: str
    options: dict[str, str]
    answer: str | None = None
    answer_idx: AnswerChoice | None = None
    meta_info: str | None = None
    metamap_phrases: list[str] = Field(default_factory=list)

    @field_validator("options")
    @classmethod
    def validate_options(cls, options: dict[str, str]) -> dict[str, str]:
        normalized = {str(key).upper(): str(value) for key, value in options.items()}
        missing = VALID_CHOICES.difference(normalized)
        if missing:
            raise ValueError(f"Missing answer options: {sorted(missing)}")
        return {choice: normalized[choice] for choice in sorted(VALID_CHOICES)}

    @field_validator("answer_idx", mode="before")
    @classmethod
    def normalize_answer_idx(cls, answer_idx: Any) -> str | None:
        if answer_idx is None:
            return None
        normalized = str(answer_idx).strip().upper()
        if normalized not in VALID_CHOICES:
            raise ValueError(f"Invalid answer_idx: {answer_idx}")
        return normalized


class AgentPrediction(BaseModel):
    answer_idx: AnswerChoice | None
    answer: str | None = None
    explanation: str = ""
    raw_response: str | None = None
    is_valid: bool = True
    error: str | None = None

    @field_validator("answer_idx", mode="before")
    @classmethod
    def normalize_answer_idx(cls, answer_idx: Any) -> str | None:
        if answer_idx is None:
            return None
        normalized = str(answer_idx).strip().upper()
        if normalized not in VALID_CHOICES:
            raise ValueError(f"Invalid answer_idx: {answer_idx}")
        return normalized
