"""Tests for evaluation data and aggregate reporting models."""

from app.agent.models import IssueCategory
from app.evaluation.models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationReport,
)


def test_evaluation_case_normalizes_reference_fields() -> None:
    case = EvaluationCase(
        case_id="case-1",
        query="GET /orders returns 401",
        expected_category=IssueCategory.AUTHENTICATION,
        expected_http_method="get",
        expected_source_hints=[" auth.md ", "auth.md"],
        expected_answer_terms=[" token ", "token"],
    )

    assert case.expected_http_method == "GET"
    assert case.expected_source_hints == ["auth.md"]
    assert case.expected_answer_terms == ["token"]


def test_evaluation_report_calculates_summary_metrics() -> None:
    report = EvaluationReport(
        results=[
            EvaluationCaseResult(
                case_id="one",
                query="q1",
                passed=True,
                overall_score=1.0,
            ),
            EvaluationCaseResult(
                case_id="two",
                query="q2",
                passed=False,
                overall_score=0.5,
            ),
        ]
    )

    assert report.total_cases == 2
    assert report.passed_cases == 1
    assert report.pass_rate == 0.5
    assert report.average_score == 0.75
    assert "[PASS] one" in report.render_text()
    assert "[FAIL] two" in report.render_text()
