"""Tests for the four-node LangGraph troubleshooting workflow."""

from __future__ import annotations

from typing import Any

from app.agent.graph import TroubleshootingAgent
from app.agent.models import IssueCategory, IssueClassification
from app.ingestion.models import SourceType
from app.rag.answers import (
    Confidence,
    EvidenceBackedStatement,
    GeneratedTroubleshootingAnalysis,
)
from app.rag.retrieval import RetrievalFilters, RetrievalResult


class FakeClassifier:
    def __init__(self, classification: IssueClassification) -> None:
        self.classification = classification
        self.calls = 0

    def classify(self, query: str) -> IssueClassification:
        self.calls += 1
        return self.classification


class FakeRetriever:
    def __init__(self, evidence: list[RetrievalResult]) -> None:
        self.evidence = evidence
        self.calls = 0
        self.query: str | None = None
        self.top_k: int | None = None
        self.filters: Any = None

    def retrieve(self, query: str, *, top_k=None, filters=None):
        self.calls += 1
        self.query = query
        self.top_k = top_k
        self.filters = filters
        return self.evidence[:top_k]


class FilterSensitiveRetriever:
    """Return nothing for inferred hard filters and evidence after fallback relaxation."""

    def __init__(
        self,
        evidence: list[RetrievalResult],
        expected_fallback: RetrievalFilters,
    ) -> None:
        self.evidence = evidence
        self.expected_fallback = expected_fallback
        self.calls: list[RetrievalFilters] = []

    def retrieve(self, query: str, *, top_k=None, filters=None):
        self.calls.append(filters)
        if filters == self.expected_fallback:
            return self.evidence[:top_k]
        return []


class SequenceGenerator:
    def __init__(self, analyses: list[GeneratedTroubleshootingAnalysis]) -> None:
        self.analyses = analyses
        self.calls = 0
        self.feedback: list[str | None] = []

    def generate(self, query: str, evidence, *, feedback=None):
        self.feedback.append(feedback)
        analysis = self.analyses[min(self.calls, len(self.analyses) - 1)]
        self.calls += 1
        return analysis


def _classification(
    *,
    status_code: int | None = 401,
    endpoint: str | None = None,
) -> IssueClassification:
    return IssueClassification(
        category=IssueCategory.AUTHENTICATION,
        summary="The order API returns 401.",
        endpoint=endpoint,
        http_method="GET" if endpoint else None,
        status_code=status_code,
        signals=["401"] if status_code else [],
    )


def _evidence(rank: int = 1) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        chunk_id=f"chunk-{rank}",
        content="Invalid or expired access tokens return HTTP 401.",
        source="docs/authentication.md",
        source_type=SourceType.DOCUMENTATION,
        distance=0.1,
        metadata={"section": "Authentication"},
    )


def _analysis(rank: int = 1) -> GeneratedTroubleshootingAnalysis:
    return GeneratedTroubleshootingAnalysis(
        summary="The request is failing access-token validation.",
        likely_causes=[
            EvidenceBackedStatement(
                text="The token may be invalid or expired.",
                evidence_ranks=[rank],
            )
        ],
        diagnostic_steps=[
            EvidenceBackedStatement(
                text="Validate the access token.",
                evidence_ranks=[rank],
            )
        ],
        confidence=Confidence.HIGH,
    )


def test_agent_runs_classify_retrieve_diagnose_verify() -> None:
    classifier = FakeClassifier(_classification())
    retriever = FakeRetriever([_evidence()])
    generator = SequenceGenerator([_analysis()])
    agent = TroubleshootingAgent(classifier, retriever, generator)

    state = agent.run_with_state("Why is the order API returning 401?")

    assert classifier.calls == 1
    assert retriever.calls == 1
    assert generator.calls == 1
    assert state["classification"].category == IssueCategory.AUTHENTICATION
    assert state["retrieval_filters"].status_code == 401
    assert state["answer"].analysis.confidence == Confidence.HIGH
    assert state["generation_attempts"] == 1
    assert state["retrieval_fallback_used"] is False


def test_agent_skips_generation_when_retrieval_is_empty() -> None:
    classifier = FakeClassifier(_classification(status_code=None))
    retriever = FakeRetriever([])
    generator = SequenceGenerator([_analysis()])
    agent = TroubleshootingAgent(classifier, retriever, generator)

    state = agent.run_with_state("Why is the order API failing?")

    assert generator.calls == 0
    assert state["generation_attempts"] == 0
    assert state["answer"].analysis.confidence == Confidence.LOW
    assert state["answer"].evidence == []
    assert state["retrieval_fallback_used"] is False


def test_agent_retries_after_grounding_failure_then_succeeds() -> None:
    classifier = FakeClassifier(_classification())
    retriever = FakeRetriever([_evidence(1)])
    generator = SequenceGenerator(
        [
            _analysis(rank=9),
            _analysis(rank=1),
        ]
    )
    agent = TroubleshootingAgent(
        classifier,
        retriever,
        generator,
        max_generation_attempts=2,
    )

    state = agent.run_with_state("Why is the order API returning 401?")

    assert classifier.calls == 1
    assert retriever.calls == 1
    assert generator.calls == 2
    assert generator.feedback[0] is None
    assert "unavailable evidence rank" in generator.feedback[1]
    assert state["generation_attempts"] == 2
    assert state["answer"].analysis.confidence == Confidence.HIGH


def test_agent_returns_safe_answer_when_grounding_retries_are_exhausted() -> None:
    classifier = FakeClassifier(_classification())
    retriever = FakeRetriever([_evidence(1)])
    generator = SequenceGenerator([_analysis(rank=9)])
    agent = TroubleshootingAgent(
        classifier,
        retriever,
        generator,
        max_generation_attempts=2,
    )

    state = agent.run_with_state("Why is the order API returning 401?")

    assert generator.calls == 2
    assert state["answer"].analysis.confidence == Confidence.LOW
    assert state["answer"].analysis.likely_causes == []
    assert "could not produce" in state["answer"].analysis.summary
    assert state["answer"].evidence[0].rank == 1


def test_agent_preserves_explicit_filters_over_classification() -> None:
    classifier = FakeClassifier(_classification(status_code=401))
    retriever = FakeRetriever([_evidence()])
    generator = SequenceGenerator([_analysis()])
    agent = TroubleshootingAgent(classifier, retriever, generator)

    requested = RetrievalFilters(
        status_code=403,
        service="order-service",
    )
    agent.run(
        "The request is failing",
        top_k=1,
        filters=requested,
    )

    assert retriever.top_k == 1
    assert retriever.filters.status_code == 403
    assert retriever.filters.service == "order-service"


def test_agent_relaxes_classifier_filters_when_they_return_no_evidence() -> None:
    classifier = FakeClassifier(
        _classification(
            status_code=401,
            endpoint="/api/v1/orders/123",
        )
    )
    fallback_filters = RetrievalFilters()
    retriever = FilterSensitiveRetriever(
        [_evidence()],
        expected_fallback=fallback_filters,
    )
    generator = SequenceGenerator([_analysis()])
    agent = TroubleshootingAgent(classifier, retriever, generator)

    state = agent.run_with_state(
        "GET /api/v1/orders/123 returns 401."
    )

    assert len(retriever.calls) == 2
    assert retriever.calls[0].endpoint == "/api/v1/orders/123"
    assert retriever.calls[0].status_code == 401
    assert retriever.calls[1] == RetrievalFilters()
    assert state["retrieval_fallback_used"] is True
    assert len(state["evidence"]) == 1
    assert state["answer"].analysis.confidence == Confidence.HIGH


def test_agent_fallback_keeps_explicit_caller_filters() -> None:
    classifier = FakeClassifier(
        _classification(
            status_code=401,
            endpoint="/api/v1/orders/123",
        )
    )
    requested = RetrievalFilters(
        service="order-service",
    )
    retriever = FilterSensitiveRetriever(
        [_evidence()],
        expected_fallback=requested,
    )
    generator = SequenceGenerator([_analysis()])
    agent = TroubleshootingAgent(classifier, retriever, generator)

    state = agent.run_with_state(
        "GET /api/v1/orders/123 returns 401.",
        filters=requested,
    )

    assert len(retriever.calls) == 2
    assert retriever.calls[0].service == "order-service"
    assert retriever.calls[0].endpoint == "/api/v1/orders/123"
    assert retriever.calls[1] == requested
    assert state["retrieval_fallback_used"] is True
