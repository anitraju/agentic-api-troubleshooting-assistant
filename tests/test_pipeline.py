"""Tests for the baseline retrieve-then-generate RAG pipeline."""

from typing import Any

import pytest

from app.ingestion.models import SourceType
from app.rag.answers import (
    Confidence,
    EvidenceBackedStatement,
    GeneratedTroubleshootingAnalysis,
)
from app.rag.pipeline import GroundedRAGPipeline, UngroundedGenerationError
from app.rag.retrieval import RetrievalFilters, RetrievalResult


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.query: str | None = None
        self.top_k: int | None = None
        self.filters: Any = None

    def retrieve(self, query: str, *, top_k=None, filters=None):
        self.query = query
        self.top_k = top_k
        self.filters = filters
        return self.results[:top_k]


class FakeGenerator:
    def __init__(self, analysis: GeneratedTroubleshootingAnalysis) -> None:
        self.analysis = analysis
        self.called = False
        self.query: str | None = None
        self.evidence: list[RetrievalResult] = []

    def generate(self, query: str, evidence):
        self.called = True
        self.query = query
        self.evidence = list(evidence)
        return self.analysis


def _evidence(rank: int = 1) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        chunk_id=f"chunk-{rank}",
        content="The service validates JWT expiration and returns 401 for invalid tokens.",
        source="docs/authentication.md",
        source_type=SourceType.DOCUMENTATION,
        distance=0.1,
        metadata={"section": "Authentication"},
    )


def _analysis(rank: int = 1) -> GeneratedTroubleshootingAnalysis:
    return GeneratedTroubleshootingAnalysis(
        summary="The request is failing token validation.",
        likely_causes=[
            EvidenceBackedStatement(
                text="The token may be invalid or expired.",
                evidence_ranks=[rank],
            )
        ],
        diagnostic_steps=[
            EvidenceBackedStatement(
                text="Inspect the token expiration and validation result.",
                evidence_ranks=[rank],
            )
        ],
        confidence=Confidence.MEDIUM,
    )


def test_pipeline_retrieves_generates_and_preserves_evidence() -> None:
    retriever = FakeRetriever([_evidence()])
    generator = FakeGenerator(_analysis())
    pipeline = GroundedRAGPipeline(retriever, generator, default_top_k=5)

    answer = pipeline.run("Why is the order API returning 401?")

    assert retriever.top_k == 5
    assert generator.called is True
    assert generator.evidence[0].chunk_id == "chunk-1"
    assert answer.analysis.confidence == Confidence.MEDIUM
    assert answer.evidence[0].rank == 1


def test_pipeline_forwards_top_k_and_filters() -> None:
    retriever = FakeRetriever([_evidence()])
    generator = FakeGenerator(_analysis())
    pipeline = GroundedRAGPipeline(retriever, generator)
    filters = RetrievalFilters(service="order-service", status_code=401)

    pipeline.run("Why 401?", top_k=1, filters=filters)

    assert retriever.top_k == 1
    assert retriever.filters is filters


def test_pipeline_returns_grounded_fallback_when_retrieval_is_empty() -> None:
    retriever = FakeRetriever([])
    generator = FakeGenerator(_analysis())
    pipeline = GroundedRAGPipeline(retriever, generator)

    answer = pipeline.run("Unknown production failure")

    assert generator.called is False
    assert answer.analysis.confidence == Confidence.LOW
    assert answer.evidence == []
    assert "did not return evidence" in answer.analysis.summary


def test_pipeline_rejects_generation_that_cites_unretrieved_rank() -> None:
    retriever = FakeRetriever([_evidence(1)])
    generator = FakeGenerator(_analysis(rank=2))
    pipeline = GroundedRAGPipeline(retriever, generator)

    with pytest.raises(UngroundedGenerationError, match="unavailable evidence"):
        pipeline.run("Why 401?")


def test_pipeline_rejects_blank_query() -> None:
    retriever = FakeRetriever([_evidence()])
    generator = FakeGenerator(_analysis())
    pipeline = GroundedRAGPipeline(retriever, generator)

    with pytest.raises(ValueError, match="query"):
        pipeline.run("   ")


def test_pipeline_rejects_invalid_top_k() -> None:
    retriever = FakeRetriever([_evidence()])
    generator = FakeGenerator(_analysis())
    pipeline = GroundedRAGPipeline(retriever, generator)

    with pytest.raises(ValueError, match="top_k"):
        pipeline.run("Why 401?", top_k=0)
