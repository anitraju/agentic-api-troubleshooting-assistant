"""LangGraph state machine for agentic API troubleshooting."""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.classification import IssueClassifier, merge_retrieval_filters
from app.agent.models import IssueClassification
from app.rag.answers import (
    Confidence,
    GeneratedTroubleshootingAnalysis,
    TroubleshootingAnswer,
)
from app.rag.generation import TroubleshootingGenerator
from app.rag.pipeline import (
    UngroundedGenerationError,
    insufficient_evidence_answer,
    validate_grounding,
)
from app.rag.retrieval import RetrievalFilters, RetrievalResult, SemanticRetriever


class TroubleshootingAgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    query: str
    top_k: int
    requested_filters: RetrievalFilters | None
    classification: IssueClassification
    retrieval_filters: RetrievalFilters
    evidence: list[RetrievalResult]
    retrieval_fallback_used: bool
    analysis: GeneratedTroubleshootingAnalysis | None
    answer: TroubleshootingAnswer
    generation_attempts: int
    grounding_error: str | None


class TroubleshootingAgent:
    """Four-node LangGraph workflow with grounding-aware conditional routing."""

    def __init__(
        self,
        classifier: IssueClassifier,
        retriever: SemanticRetriever,
        generator: TroubleshootingGenerator,
        *,
        default_top_k: int = 5,
        max_generation_attempts: int = 2,
    ) -> None:
        if default_top_k < 1:
            raise ValueError("default_top_k must be at least 1")
        if max_generation_attempts < 1:
            raise ValueError("max_generation_attempts must be at least 1")

        self.classifier = classifier
        self.retriever = retriever
        self.generator = generator
        self.default_top_k = default_top_k
        self.max_generation_attempts = max_generation_attempts
        self.graph = self._build_graph()

    def run(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> TroubleshootingAnswer:
        """Run the graph and return the final validated answer."""
        final_state = self.run_with_state(
            query,
            top_k=top_k,
            filters=filters,
        )
        answer = final_state.get("answer")
        if answer is None:
            raise RuntimeError("agent graph completed without producing an answer")
        return answer

    def run_with_state(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> TroubleshootingAgentState:
        """Run the graph and expose final state for diagnostics and tests."""
        if not query.strip():
            raise ValueError("query cannot be empty")

        result_limit = self.default_top_k if top_k is None else top_k
        if result_limit < 1:
            raise ValueError("top_k must be at least 1")

        result = self.graph.invoke(
            {
                "query": query.strip(),
                "top_k": result_limit,
                "requested_filters": filters,
                "generation_attempts": 0,
                "grounding_error": None,
                "retrieval_fallback_used": False,
            }
        )
        return TroubleshootingAgentState(**result)

    def _build_graph(self):
        workflow = StateGraph(TroubleshootingAgentState)

        workflow.add_node("classify", self._classify_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("diagnose", self._diagnose_node)
        workflow.add_node("verify", self._verify_node)

        workflow.add_edge(START, "classify")
        workflow.add_edge("classify", "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self._route_after_retrieval,
            {
                "diagnose": "diagnose",
                "verify": "verify",
            },
        )
        workflow.add_edge("diagnose", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._route_after_verification,
            {
                "retry": "diagnose",
                "end": END,
            },
        )

        return workflow.compile()

    def _classify_node(self, state: TroubleshootingAgentState) -> dict:
        classification = self.classifier.classify(state["query"])
        retrieval_filters = merge_retrieval_filters(
            classification,
            state.get("requested_filters"),
        )
        return {
            "classification": classification,
            "retrieval_filters": retrieval_filters,
        }

    def _retrieve_node(self, state: TroubleshootingAgentState) -> dict:
        """Retrieve with classifier-derived filters, then safely relax inferred filters.

        Classifier-extracted endpoint/status/method/service values are useful retrieval hints,
        but they can be more specific than the metadata stored in the knowledge index. For
        example, a user may report `/api/v1/orders/123` while documentation is indexed under
        `/api/v1/orders/{order_id}`.

        We therefore try the merged filters first. If they return no evidence, we retry using
        only filters explicitly supplied by the caller. Explicit caller constraints are never
        discarded.
        """
        primary_filters = state["retrieval_filters"]
        evidence = self.retriever.retrieve(
            state["query"],
            top_k=state["top_k"],
            filters=primary_filters,
        )

        fallback_used = False
        requested_filters = state.get("requested_filters")
        fallback_filters = (
            requested_filters
            if requested_filters is not None
            else RetrievalFilters()
        )

        if not evidence and primary_filters != fallback_filters:
            evidence = self.retriever.retrieve(
                state["query"],
                top_k=state["top_k"],
                filters=fallback_filters,
            )
            fallback_used = True

        return {
            "evidence": evidence,
            "retrieval_fallback_used": fallback_used,
        }

    @staticmethod
    def _route_after_retrieval(
        state: TroubleshootingAgentState,
    ) -> Literal["diagnose", "verify"]:
        return "diagnose" if state.get("evidence") else "verify"

    def _diagnose_node(self, state: TroubleshootingAgentState) -> dict:
        attempts = state.get("generation_attempts", 0) + 1
        analysis = self.generator.generate(
            state["query"],
            state["evidence"],
            feedback=state.get("grounding_error"),
        )
        return {
            "analysis": analysis,
            "generation_attempts": attempts,
            "grounding_error": None,
        }

    def _verify_node(self, state: TroubleshootingAgentState) -> dict:
        evidence = state.get("evidence", [])
        if not evidence:
            return {
                "answer": insufficient_evidence_answer(state["query"]),
                "grounding_error": None,
            }

        analysis = state.get("analysis")
        if analysis is None:
            return {
                "answer": _generation_failure_answer(
                    state["query"],
                    evidence,
                    "The diagnosis node did not return an analysis.",
                )
            }

        try:
            validate_grounding(
                analysis,
                {item.rank for item in evidence},
            )
        except UngroundedGenerationError as exc:
            error = str(exc)
            if state.get("generation_attempts", 0) < self.max_generation_attempts:
                return {
                    "analysis": None,
                    "grounding_error": error,
                }

            return {
                "analysis": None,
                "grounding_error": error,
                "answer": _generation_failure_answer(
                    state["query"],
                    evidence,
                    error,
                ),
            }

        return {
            "answer": TroubleshootingAnswer(
                query=state["query"],
                analysis=analysis,
                evidence=evidence,
            ),
            "grounding_error": None,
        }

    @staticmethod
    def _route_after_verification(
        state: TroubleshootingAgentState,
    ) -> Literal["retry", "end"]:
        if state.get("answer") is not None:
            return "end"
        if state.get("grounding_error"):
            return "retry"
        return "end"


def _generation_failure_answer(
    query: str,
    evidence: list[RetrievalResult],
    reason: str,
) -> TroubleshootingAnswer:
    """Return a safe answer when grounded generation cannot be completed."""
    return TroubleshootingAnswer(
        query=query.strip(),
        analysis=GeneratedTroubleshootingAnalysis(
            summary=(
                "Relevant evidence was retrieved, but the agent could not produce a "
                "grounding-validated diagnosis."
            ),
            confidence=Confidence.LOW,
            limitations=[
                reason,
                "No unsupported cause or remediation is returned.",
            ],
        ),
        evidence=evidence,
    )
