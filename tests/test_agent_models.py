"""Tests for structured agent classification models."""

from app.agent.models import IssueCategory, IssueClassification


def test_issue_classification_normalizes_fields() -> None:
    classification = IssueClassification(
        category=IssueCategory.AUTHENTICATION,
        summary="Request returns 401.",
        endpoint=" /api/v1/orders/123 ",
        http_method="get",
        status_code=401,
        signals=["401", " GET ", "401"],
    )

    assert classification.endpoint == "/api/v1/orders/123"
    assert classification.http_method == "GET"
    assert classification.signals == ["401", "GET"]
