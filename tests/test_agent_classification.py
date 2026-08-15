"""Tests for classifier-derived retrieval filtering."""

from app.agent.classification import merge_retrieval_filters
from app.agent.models import IssueCategory, IssueClassification
from app.ingestion.models import SourceType
from app.rag.retrieval import RetrievalFilters


def _classification() -> IssueClassification:
    return IssueClassification(
        category=IssueCategory.SERVICE_AVAILABILITY,
        summary="POST /api/v1/orders returns 503.",
        service="order-service",
        endpoint="/api/v1/orders",
        http_method="POST",
        status_code=503,
        signals=["POST", "/api/v1/orders", "503"],
    )


def test_classification_fills_missing_retrieval_filters() -> None:
    filters = merge_retrieval_filters(_classification(), None)

    assert filters.service == "order-service"
    assert filters.endpoint == "/api/v1/orders"
    assert filters.http_method == "POST"
    assert filters.status_code == 503


def test_explicit_filters_override_classifier_values() -> None:
    requested = RetrievalFilters(
        source_types=(SourceType.RUNBOOK,),
        service="manual-service",
        status_code=502,
        metadata={"environment": "production"},
    )

    filters = merge_retrieval_filters(_classification(), requested)

    assert filters.source_types == (SourceType.RUNBOOK,)
    assert filters.service == "manual-service"
    assert filters.endpoint == "/api/v1/orders"
    assert filters.http_method == "POST"
    assert filters.status_code == 502
    assert filters.metadata == {"environment": "production"}
