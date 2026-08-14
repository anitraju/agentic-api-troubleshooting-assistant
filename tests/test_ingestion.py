"""Tests for knowledge-base document loading."""

import json
from pathlib import Path

import pytest

from app.ingestion.loaders import (
    load_incidents_file,
    load_knowledge_base,
    load_markdown_file,
    load_openapi_file,
)
from app.ingestion.models import SourceType


def test_load_markdown_file_extracts_title_and_metadata(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    path = docs_dir / "authentication.md"
    path.write_text(
        "# Authentication\n\nUse a bearer token for protected endpoints.\n",
        encoding="utf-8",
    )

    document = load_markdown_file(
        path,
        source_type=SourceType.DOCUMENTATION,
        root=tmp_path,
    )

    assert document.source == "docs/authentication.md"
    assert document.source_type == SourceType.DOCUMENTATION
    assert document.metadata["title"] == "Authentication"
    assert document.metadata["format"] == "markdown"
    assert "bearer token" in document.content


def test_load_incidents_creates_one_document_per_incident(tmp_path: Path) -> None:
    incident_dir = tmp_path / "incidents"
    incident_dir.mkdir()

    path = incident_dir / "known_incidents.json"
    path.write_text(
        json.dumps(
            {
                "service": "order-service",
                "incidents": [
                    {
                        "incident_id": "INC-1",
                        "title": "Expired token",
                        "date": "2026-01-01",
                        "status": "resolved",
                        "symptoms": ["401", "invalid_token"],
                        "affected_endpoint": "POST /api/v1/orders",
                        "root_cause": "Token expired.",
                        "resolution": "Refresh token.",
                        "signatures": ["token_expired"],
                    },
                    {
                        "incident_id": "INC-2",
                        "title": "Payment timeout",
                        "date": "2026-01-02",
                        "status": "resolved",
                        "symptoms": ["504"],
                        "affected_endpoint": "POST /api/v1/orders",
                        "root_cause": "Payment latency exceeded timeout.",
                        "resolution": "Optimize downstream query.",
                        "signatures": ["payment_timeout"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    documents = load_incidents_file(path, root=tmp_path)

    assert len(documents) == 2
    assert documents[0].source == "incidents/known_incidents.json#INC-1"
    assert documents[0].source_type == SourceType.INCIDENT
    assert documents[0].metadata["service"] == "order-service"
    assert documents[0].metadata["incident_id"] == "INC-1"
    assert "Root cause: Token expired." in documents[0].content
    assert documents[1].metadata["incident_id"] == "INC-2"


def test_load_incidents_rejects_non_list_incidents(tmp_path: Path) -> None:
    path = tmp_path / "incidents.json"
    path.write_text(
        json.dumps({"incidents": {"incident_id": "INC-1"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="'incidents' must be a list"):
        load_incidents_file(path)


def test_load_openapi_extracts_spec_metadata(tmp_path: Path) -> None:
    path = tmp_path / "service.yaml"
    path.write_text(
        """
openapi: 3.1.0
info:
  title: Test API
  version: 1.2.3
paths:
  /api/v1/orders:
    post:
      responses:
        "201":
          description: Created
  /health:
    get:
      responses:
        "200":
          description: Healthy
""".strip(),
        encoding="utf-8",
    )

    documents = load_openapi_file(path, root=tmp_path)

    assert len(documents) == 1
    document = documents[0]
    assert document.source == "service.yaml"
    assert document.source_type == SourceType.OPENAPI
    assert document.metadata["title"] == "Test API"
    assert document.metadata["api_version"] == "1.2.3"
    assert document.metadata["openapi_version"] == "3.1.0"
    assert document.metadata["path_count"] == 2
    assert document.metadata["paths"] == ["/api/v1/orders", "/health"]


def test_load_knowledge_base_loads_expected_repository_sources() -> None:
    documents = load_knowledge_base(Path("data"))

    assert len(documents) == 15

    counts = {
        source_type: sum(document.source_type == source_type for document in documents)
        for source_type in SourceType
    }

    assert counts == {
        SourceType.DOCUMENTATION: 4,
        SourceType.RUNBOOK: 4,
        SourceType.INCIDENT: 6,
        SourceType.OPENAPI: 1,
    }

    sources = {document.source for document in documents}
    assert "docs/authentication.md" in sources
    assert "runbooks/service_unavailable.md" in sources
    assert "incidents/known_incidents.json#INC-2026-014" in sources
    assert "openapi/order-service.yaml" in sources

    assert not any(document.source.startswith("logs/") for document in documents)
