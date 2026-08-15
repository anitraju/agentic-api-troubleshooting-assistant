"""Tests for the LLM generation boundary without making network calls."""

import pytest

from app.ingestion.models import SourceType
from app.rag.generation import OpenAITroubleshootingGenerator, format_evidence_context
from app.rag.retrieval import RetrievalResult


def _evidence() -> RetrievalResult:
    return RetrievalResult(
        rank=1,
        chunk_id="chunk-1",
        content="Every protected request must include an Authorization bearer token.",
        source="docs/authentication.md",
        source_type=SourceType.DOCUMENTATION,
        distance=0.2,
        metadata={"section": "Overview", "service": "order-service"},
    )


def test_format_evidence_context_preserves_rank_source_and_content() -> None:
    context = format_evidence_context([_evidence()])
    assert "=== EVIDENCE [1] ===" in context
    assert "Source: docs/authentication.md" in context
    assert "service=order-service" in context
    assert "Authorization bearer token" in context


def test_openai_generator_rejects_blank_api_key_without_network_call() -> None:
    with pytest.raises(ValueError, match="api_key"):
        OpenAITroubleshootingGenerator(model_name="gpt-5-nano", api_key=" ")


def test_openai_generator_rejects_empty_evidence_before_model_loading() -> None:
    generator = OpenAITroubleshootingGenerator(
        model_name="gpt-5-nano",
        api_key="test-key",
    )
    with pytest.raises(ValueError, match="evidence"):
        generator.generate("Why 401?", [])
