"""Tests for evidence formatting and generation validation."""

from app.ingestion.models import SourceType
from app.rag.generation import OpenAITroubleshootingGenerator, format_evidence_context
from app.rag.retrieval import RetrievalResult


def _result() -> RetrievalResult:
    return RetrievalResult(
        rank=1,
        chunk_id="chunk-1",
        content="Expired access tokens return HTTP 401.",
        source="docs/authentication.md",
        source_type=SourceType.DOCUMENTATION,
        distance=0.1,
        metadata={"section": "Expired tokens", "service": "order-service"},
    )


def test_format_evidence_context_preserves_rank_source_and_content() -> None:
    context = format_evidence_context([_result()])

    assert "=== EVIDENCE [1] ===" in context
    assert "Source: docs/authentication.md" in context
    assert "Source type: documentation" in context
    assert "service=order-service" in context
    assert "Expired access tokens return HTTP 401." in context


def test_generator_rejects_blank_model_name() -> None:
    try:
        OpenAITroubleshootingGenerator(model_name=" ", api_key="key")
    except ValueError as exc:
        assert "model_name" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_generator_rejects_blank_api_key() -> None:
    try:
        OpenAITroubleshootingGenerator(model_name="test-model", api_key=" ")
    except ValueError as exc:
        assert "api_key" in str(exc)
    else:
        raise AssertionError("expected ValueError")
