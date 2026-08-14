"""Tests for metadata-aware knowledge chunking."""

from pathlib import Path

from app.ingestion import (
    KnowledgeDocument,
    SourceType,
    chunk_document,
    chunk_documents,
    load_knowledge_base,
)


def test_markdown_chunk_preserves_section_and_api_signals() -> None:
    document = KnowledgeDocument(
        content="""
# Order Service API

## Create Order

POST /api/v1/orders returns 422 when the request violates the schema.

## Authentication

POST /api/v1/orders requires the orders:write scope and can return 403.
""".strip(),
        source="docs/orders.md",
        source_type=SourceType.DOCUMENTATION,
        metadata={"title": "Order Service API"},
    )

    chunks = chunk_document(document)

    create_order = next(
        chunk for chunk in chunks if chunk.metadata.get("section") == "Create Order"
    )

    assert create_order.metadata["endpoint"] == "/api/v1/orders"
    assert create_order.metadata["http_method"] == "POST"
    assert create_order.metadata["status_code"] == 422
    assert create_order.metadata["service"] == "order-service"
    assert create_order.metadata["chunk_count"] == len(chunks)


def test_long_markdown_section_is_split_with_repeated_section_context() -> None:
    body = " ".join(f"diagnostic-token-{index}" for index in range(180))
    document = KnowledgeDocument(
        content=f"# Runbook\n\n## Investigation\n\n{body}",
        source="runbooks/long.md",
        source_type=SourceType.RUNBOOK,
    )

    chunks = chunk_document(document, max_chars=320, overlap_chars=40)

    investigation_chunks = [
        chunk for chunk in chunks if chunk.metadata.get("section") == "Investigation"
    ]

    assert len(investigation_chunks) > 1
    assert all(chunk.content.startswith("## Investigation") for chunk in investigation_chunks)
    assert all(len(chunk.content) <= 320 for chunk in investigation_chunks)


def test_incident_chunk_preserves_incident_metadata() -> None:
    document = KnowledgeDocument(
        content="""
Incident ID: INC-14
Title: Database pool exhaustion
Affected endpoint: POST /api/v1/orders
Symptoms: 503, service_unavailable
Root cause: No database connections were available.
""".strip(),
        source="incidents/known_incidents.json#INC-14",
        source_type=SourceType.INCIDENT,
        metadata={
            "incident_id": "INC-14",
            "service": "order-service",
            "affected_endpoint": "POST /api/v1/orders",
        },
    )

    chunks = chunk_document(document)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.metadata["incident_id"] == "INC-14"
    assert chunk.metadata["service"] == "order-service"
    assert chunk.metadata["endpoint"] == "/api/v1/orders"
    assert chunk.metadata["http_method"] == "POST"
    assert chunk.metadata["status_code"] == 503


def test_openapi_is_chunked_by_operation_and_resolves_local_refs() -> None:
    document = KnowledgeDocument(
        content="""
openapi: 3.1.0
info:
  title: Order Service API
  version: 1.0.0
paths:
  /api/v1/orders:
    post:
      operationId: createOrder
      summary: Create order
      parameters:
        - $ref: '#/components/parameters/IdempotencyKey'
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateOrderRequest'
      responses:
        '201':
          description: Created
        '422':
          description: Invalid request
components:
  parameters:
    IdempotencyKey:
      name: Idempotency-Key
      in: header
      required: true
      schema:
        type: string
  schemas:
    CreateOrderRequest:
      type: object
      required: [customer_id]
      properties:
        customer_id:
          type: string
""".strip(),
        source="openapi/order-service.yaml",
        source_type=SourceType.OPENAPI,
        metadata={"title": "Order Service API"},
    )

    chunks = chunk_document(document, max_chars=4000)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.metadata["endpoint"] == "/api/v1/orders"
    assert chunk.metadata["http_method"] == "POST"
    assert chunk.metadata["operation_id"] == "createOrder"
    assert chunk.metadata["status_codes"] == "201,422"
    assert "Idempotency-Key" in chunk.content
    assert "customer_id" in chunk.content


def test_repository_knowledge_base_produces_unique_retrieval_chunks() -> None:
    documents = load_knowledge_base(Path("data"))
    chunks = chunk_documents(documents)

    assert len(chunks) > len(documents)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(chunk.content for chunk in chunks)
    assert all(chunk.metadata["chunk_count"] >= 1 for chunk in chunks)
    assert not any(chunk.source.startswith("logs/") for chunk in chunks)

    openapi_chunks = [
        chunk for chunk in chunks if chunk.source_type == SourceType.OPENAPI
    ]
    assert {chunk.metadata["endpoint"] for chunk in openapi_chunks} == {
        "/api/v1/orders",
        "/api/v1/orders/{order_id}",
        "/health",
    }


def test_chunk_settings_are_validated() -> None:
    document = KnowledgeDocument(
        content="A" * 200,
        source="docs/example.md",
        source_type=SourceType.DOCUMENTATION,
    )

    try:
        chunk_document(document, max_chars=99)
    except ValueError as exc:
        assert "max_chars" in str(exc)
    else:
        raise AssertionError("Expected max_chars validation to fail")

    try:
        chunk_document(document, max_chars=200, overlap_chars=200)
    except ValueError as exc:
        assert "overlap_chars" in str(exc)
    else:
        raise AssertionError("Expected overlap validation to fail")
