"""Tests for evaluation dataset loading and runner error isolation."""

import json
from pathlib import Path

import pytest

from app.agent.models import IssueCategory, IssueClassification
from app.evaluation.models import EvaluationCase
from app.evaluation.runner import EvaluationRunner, load_evaluation_cases
from app.ingestion.models import SourceType
from app.rag.answers import Confidence, GeneratedTroubleshootingAnalysis, TroubleshootingAnswer
from app.rag.retrieval import RetrievalResult


class PassingAgent:
    def run_with_state(self, query: str, *, top_k=None, filters=None):
        evidence = [
            RetrievalResult(
                rank=1,
                chunk_id="chunk-1",
                content="Access tokens are required.",
                source="docs/authentication.md",
                source_type=SourceType.DOCUMENTATION,
                distance=0.1,
                metadata={},
            )
        ]
        return {
            "classification": IssueClassification(
                category=IssueCategory.AUTHENTICATION,
                summary="Authentication failure.",
                status_code=401,
            ),
            "evidence": evidence,
            "answer": TroubleshootingAnswer(
                query=query,
                analysis=GeneratedTroubleshootingAnalysis(
                    summary="Check the access token for the 401 response.",
                    confidence=Confidence.HIGH,
                ),
                evidence=evidence,
            ),
            "generation_attempts": 1,
        }


class FailingAgent:
    def run_with_state(self, query: str, *, top_k=None, filters=None):
        raise RuntimeError("synthetic agent failure")


def _case() -> EvaluationCase:
    return EvaluationCase(
        case_id="case-1",
        query="Order API returns 401",
        expected_category=IssueCategory.AUTHENTICATION,
        expected_status_code=401,
        expected_source_hints=["authentication.md"],
        expected_answer_terms=["token", "401"],
        minimum_score=0.75,
    )


def test_runner_returns_report_for_successful_case() -> None:
    report = EvaluationRunner(PassingAgent()).run([_case()])

    assert report.total_cases == 1
    assert report.results[0].passed is True


def test_runner_captures_agent_exception_as_case_failure() -> None:
    report = EvaluationRunner(FailingAgent()).run([_case()])

    assert report.total_cases == 1
    assert report.results[0].passed is False
    assert "synthetic agent failure" in report.results[0].error


def test_load_evaluation_cases_validates_unique_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    case = {
        "case_id": "duplicate",
        "query": "q",
        "expected_category": "authentication",
    }
    path.write_text(
        json.dumps({"cases": [case, case]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique"):
        load_evaluation_cases(path)


def test_load_evaluation_cases_reads_curated_json(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "one",
                        "query": "GET /orders returns 401",
                        "expected_category": "authentication",
                        "expected_status_code": 401,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = load_evaluation_cases(path)

    assert len(cases) == 1
    assert cases[0].expected_category == IssueCategory.AUTHENTICATION
    assert cases[0].expected_status_code == 401
