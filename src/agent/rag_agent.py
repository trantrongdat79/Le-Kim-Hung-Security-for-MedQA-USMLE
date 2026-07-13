"""RAG-only medical reasoning agent for Sprint 2."""

from __future__ import annotations

from llama_index.llms.openrouter import OpenRouter

from src.agent.med_agent import (
    SYSTEM_INSTRUCTIONS,
    _clean_text,
    _extract_json_object,
    _format_options,
)
from src.core.config import LLMConfig, load_llm_1_config
from src.core.schema import (
    AgentPrediction,
    EvidenceChunk,
    MedQAExample,
    VALID_CHOICES,
    normalize_choice,
)
from src.retrieval.retriever import MedicalEvidenceRetriever


class RAGMedicalReasoningAgent:
    """Retrieve textbook evidence before answering a MedQA-USMLE question."""

    def __init__(
        self,
        *,
        config: LLMConfig | None = None,
        retriever: MedicalEvidenceRetriever | None = None,
        top_k: int = 5,
    ) -> None:
        self.config = config or load_llm_1_config()
        self.retriever = retriever or MedicalEvidenceRetriever()
        self.top_k = top_k
        self.llm = OpenRouter(
            model=self.config.model,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            system_prompt=SYSTEM_INSTRUCTIONS,
        )

    def answer(self, example: MedQAExample) -> AgentPrediction:
        try:
            evidence = self.retriever.retrieve(example.question, top_k=self.top_k)
        except Exception as exc:
            return AgentPrediction(
                answer_idx=None,
                answer=None,
                explanation="",
                is_valid=False,
                error=f"Retrieval failed: {exc}",
            )

        return self.answer_with_evidence(example, evidence)

    def answer_with_evidence(
        self,
        example: MedQAExample,
        evidence: list[EvidenceChunk],
    ) -> AgentPrediction:
        prompt = self._build_prompt(example, evidence)
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

    def _build_prompt(
        self,
        example: MedQAExample,
        evidence: list[EvidenceChunk],
    ) -> str:
        options = "\n".join(_format_options(example.options))
        evidence_text = _format_evidence(evidence)
        return f"""Use the retrieved evidence when it is relevant.
If the evidence is irrelevant or conflicting, answer from medical knowledge.

Retrieved evidence:
{evidence_text}

Question:
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
            answer_idx = normalize_choice(payload.get("answer_idx"))
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


def _format_evidence(evidence: list[EvidenceChunk]) -> str:
    if not evidence:
        return "No retrieved evidence."

    blocks: list[str] = []
    for index, chunk in enumerate(evidence, start=1):
        title = chunk.title or "unknown source"
        source = chunk.source or "unknown corpus"
        chunk_id = chunk.chunk_id or "unknown id"
        score = f"{chunk.score:.6f}" if chunk.score is not None else "unknown"
        blocks.append(
            f"[{index}] title={title}; source={source}; id={chunk_id}; distance={score}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(blocks)
