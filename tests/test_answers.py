"""Tests for structured troubleshooting answer models and rendering."""

import pytest
from pydantic import ValidationError

from app.ingestion.models import SourceType
from app.rag.answers import (
    Confidence,
    EvidenceBackedStatement,
    GeneratedTroubleshootingAnalysis,
    TroubleshootingAnswer,
)
from app.rag.retrieval import RetrievalResult


def _evidence(rank: int = 1) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        chunk_id=f"chunk-{rank}",
        content="Expired access tokens return HTTP 401.",
        source="docs/authentication.md",
        source_type=SourceType.DOCUMENTATION,
        distance=0.1,
        metadata={"section": "Expired tokens"},
    )


def test_evidence_statement_deduplicates_ranks() -> None:
    statement = EvidenceBackedStatement(text="Check the token.", evidence_ranks=[1, 1, 2])
    assert statement.evidence_ranks == [1, 2]


def test_evidence_statement_rejects_non_positive_rank() -> None:
    with pytest.raises(ValidationError, match="positive integers"):
        EvidenceBackedStatement(text="Check the token.", evidence_ranks=[0])


def test_answer_renderer_includes_inline_citations_and_sources() -> None:
    answer = TroubleshootingAnswer(
        query="Why 401?",
        analysis=GeneratedTroubleshootingAnalysis(
            summary="The request is failing authentication.",
            likely_causes=[
                EvidenceBackedStatement(
                    text="The token may be expired.",
                    evidence_ranks=[1],
                )
            ],
            confidence=Confidence.MEDIUM,
        ),
        evidence=[_evidence()],
    )

    rendered = answer.render_markdown()
    assert "The token may be expired. [1]" in rendered
    assert "[1] docs/authentication.md — Expired tokens" in rendered
    assert "**Confidence:** medium" in rendered
