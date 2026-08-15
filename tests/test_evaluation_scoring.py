"""Tests for deterministic evaluation scoring."""

from app.agent.models import IssueCategory, IssueClassification
from app.evaluation.models import EvaluationCase
from app.evaluation.scoring import score_agent_state
from app.ingestion.models import SourceType
from app.rag.answers import (
    Confidence,
    EvidenceBackedStatement,
    GeneratedTroubleshootingAnalysis,
    TroubleshootingAnswer,
)
from app.rag.retrieval import RetrievalResult


def _case() -> EvaluationCase:
    return EvaluationCase(
        case_id="auth-401",
        query="GET /api/v1/orders/123 returns 401",
        expected_category=IssueCategory.AUTHENTICATION,
        expected_status_code=401,
        expected_http_method="GET",
        expected_endpoint="/api/v1/orders/123",
        expected_source_hints=["authentication.md"],
        expected_answer_terms=["token", "401"],
        minimum_score=0.75,
    )


def _evidence() -> RetrievalResult:
    return RetrievalResult(
        rank=1,
        chunk_id="chunk-1",
        content="Invalid or expired access tokens return HTTP 401.",
        source="docs/authentication.md",
        source_type=SourceType.DOCUMENTATION,
        distance=0.1,
        metadata={},
    )


def _state(*, cited_rank: int = 1) -> dict:
    evidence = [_evidence()]
    analysis = GeneratedTroubleshootingAnalysis(
        summary="The token is invalid or expired and the request returns 401.",
        likely_causes=[
            EvidenceBackedStatement(
                text="The token may be invalid or expired.",
                evidence_ranks=[cited_rank],
            )
        ],
        confidence=Confidence.HIGH,
    )

    return {
        "classification": IssueClassification(
            category=IssueCategory.AUTHENTICATION,
            summary="GET request returns 401.",
            endpoint="/api/v1/orders/123",
            http_method="GET",
            status_code=401,
            signals=["GET", "401"],
        ),
        "evidence": evidence,
        "answer": TroubleshootingAnswer(
            query=_case().query,
            analysis=analysis,
            evidence=evidence,
        ),
        "generation_attempts": 1,
    }


def test_scoring_passes_fully_grounded_matching_case() -> None:
    result = score_agent_state(_case(), _state())

    assert result.passed is True
    assert result.overall_score == 1.0
    assert result.scores["grounding"] == 1.0
    assert result.scores["retrieval_source_hit"] == 1.0
    assert result.scores["answer_term_recall"] == 1.0


def test_scoring_rejects_ungrounded_evidence_rank() -> None:
    result = score_agent_state(_case(), _state(cited_rank=9))

    assert result.passed is False
    assert result.scores["grounding"] == 0.0
    assert any("outside retrieved ranks" in item for item in result.failures)


def test_scoring_records_classification_mismatch() -> None:
    state = _state()
    state["classification"] = IssueClassification(
        category=IssueCategory.AUTHORIZATION,
        summary="Wrong category.",
        endpoint="/api/v1/orders/123",
        http_method="GET",
        status_code=401,
    )

    result = score_agent_state(_case(), state)

    assert result.passed is False
    assert result.scores["category"] == 0.0
    assert any("category mismatch" in item for item in result.failures)
