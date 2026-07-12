"""LangGraph-orchestrated RAG workflow for Sprint 3."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent.rag_agent import RAGMedicalReasoningAgent
from src.core.config import LLMConfig
from src.core.schema import AgentPrediction, EvidenceChunk, MedQAExample
from src.retrieval.retriever import MedicalEvidenceRetriever


class LangGraphRAGState(TypedDict, total=False):
    example: MedQAExample
    evidence: list[EvidenceChunk]
    prediction: AgentPrediction


class LangGraphRAGAgent:
    """Run the Sprint 2 RAG flow through a simple LangGraph workflow."""

    def __init__(
        self,
        *,
        config: LLMConfig | None = None,
        retriever: MedicalEvidenceRetriever | None = None,
        top_k: int = 5,
    ) -> None:
        self.top_k = top_k
        self.retriever = retriever or MedicalEvidenceRetriever()
        self.answerer = RAGMedicalReasoningAgent(
            config=config,
            retriever=self.retriever,
            top_k=top_k,
        )
        self.graph = self._build_graph()

    def answer(self, example: MedQAExample) -> AgentPrediction:
        initial_state: LangGraphRAGState = {"example": example}
        final_state = self.graph.invoke(initial_state)
        prediction = final_state.get("prediction")
        if prediction is None:
            return AgentPrediction(
                answer_idx=None,
                answer=None,
                explanation="",
                is_valid=False,
                error="LangGraph workflow completed without a prediction",
            )
        return prediction

    def _build_graph(self):
        workflow = StateGraph(LangGraphRAGState)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("answer", self._answer_node)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", END)
        return workflow.compile()

    def _retrieve_node(self, state: LangGraphRAGState) -> LangGraphRAGState:
        example = state["example"]
        try:
            evidence = self.retriever.retrieve(example.question, top_k=self.top_k)
            return {"evidence": evidence}
        except Exception as exc:
            return {
                "evidence": [],
                "prediction": AgentPrediction(
                    answer_idx=None,
                    answer=None,
                    explanation="",
                    is_valid=False,
                    error=f"Retrieval failed: {exc}",
                ),
            }

    def _answer_node(self, state: LangGraphRAGState) -> LangGraphRAGState:
        if prediction := state.get("prediction"):
            return {"prediction": prediction}

        example = state["example"]
        evidence = state.get("evidence", [])
        prediction = self.answerer.answer_with_evidence(example, evidence)
        return {"prediction": prediction}
