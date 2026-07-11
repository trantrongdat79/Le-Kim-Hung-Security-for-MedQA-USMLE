"""Medical reasoning agent for MedQA-USMLE multiple-choice questions."""

from __future__ import annotations

import json
import re
from typing import Any

from llama_index.llms.openrouter import OpenRouter

from src.core.config import LLMConfig, load_llm_1_config
from src.core.schema import AgentPrediction, MedQAExample, VALID_CHOICES


SYSTEM_INSTRUCTIONS = """You are a careful medical exam reasoning assistant.
Answer USMLE-style multiple-choice questions.
Return only valid JSON with answer_idx, answer, and explanation.
The explanation must be short and must not include hidden chain-of-thought."""


class MedicalReasoningAgent:
    """Single-agent medical reasoning baseline for Sprint 1."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or load_llm_1_config()
        self.llm = OpenRouter(
            model=self.config.model,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            system_prompt=SYSTEM_INSTRUCTIONS,
        )

    def answer(self, example: MedQAExample) -> AgentPrediction:
        prompt = self._build_prompt(example)
        try:
            response = self.llm.complete(prompt)
            raw_response = str(getattr(response, "text", response)).strip()
        except Exception as exc:
            return AgentPrediction(
                answer_idx=None,
                answer=None,
                explanation="",
                is_valid=False,
                error=f"LLM call failed: {exc}",
            )

        return self._parse_prediction(raw_response, example)

    def _build_prompt(self, example: MedQAExample) -> str:
        options = "\n".join(
            f"{choice}. {example.options[choice]}" for choice in sorted(VALID_CHOICES)
        )
        return f"""Question:
{example.question}

Options:
{options}

Choose exactly one option: A, B, C, or D.
Return only JSON in this exact shape:
{{
  "answer_idx": "A",
  "answer": "answer text from the selected option",
  "explanation": "one or two short sentences"
}}"""

    def _parse_prediction(
        self,
        raw_response: str,
        example: MedQAExample,
    ) -> AgentPrediction:
        try:
            payload = _extract_json_object(raw_response)
            answer_idx = str(payload.get("answer_idx", "")).strip().upper()
            if answer_idx not in VALID_CHOICES:
                raise ValueError(f"Invalid answer_idx: {answer_idx!r}")

            answer = example.options.get(answer_idx) or _clean_text(payload.get("answer"))
            explanation = _clean_text(payload.get("explanation"))

            return AgentPrediction(
                answer_idx=answer_idx,
                answer=answer,
                explanation=explanation,
                raw_response=raw_response,
                is_valid=True,
            )
        except Exception as exc:
            return AgentPrediction(
                answer_idx=None,
                answer=None,
                explanation="",
                raw_response=raw_response,
                is_valid=False,
                error=str(exc),
            )


def _extract_json_object(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in model response")
        text = text[start : end + 1]

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON is not an object")
    return parsed


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
