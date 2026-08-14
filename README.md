# Agentic API Troubleshooting Assistant Using RAG

A portfolio project for building an AI assistant that investigates API failures using
retrieval-augmented generation (RAG), structured troubleshooting tools, and an agentic
workflow.

## Current status

**Commit 2 — Sample Order Service troubleshooting environment**

The repository now contains a deterministic fictional Order Service environment that will
serve as the knowledge base and ground-truth dataset for later RAG and agentic commits.

The sample data includes:

- API documentation
- authentication and rate-limit guidance
- error-code reference material
- operational runbooks
- historical incident records
- request-correlated structured logs
- an OpenAPI 3.1 specification

No RAG framework has been introduced yet. Commit 3 will implement document ingestion and
normalize this source material into application models.

## Why use a deterministic sample environment?

A controlled fictional service gives later evaluation code known answers. We can measure
whether retrieval and agent behavior find the right evidence instead of relying on subjective
demo questions.

| Failure | Ground-truth signal |
|---|---|
| `400 malformed_request` | malformed JSON |
| `401 invalid_token` | expired/invalid JWT |
| `403 insufficient_scope` | missing OAuth scope |
| `404 order_not_found` | unknown order ID |
| `409 duplicate_idempotency_key` | key reused with another payload |
| `422 schema_validation_failed` | request violates schema |
| `429 rate_limit_exceeded` | caller exceeded quota |
| `500 database_unavailable` | database failure |
| `502 payment_dependency_failed` | Payment Service failed |
| `503 service_unavailable` | dependency/service saturation |
| `504 payment_timeout` | payment call exceeded timeout |

## Requirements

- Python 3.11+
- `pip`
- Git

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
python -m app.main
```

Expected output is similar to:

```text
INFO | __main__ | Starting Agentic API Troubleshooting Assistant
INFO | __main__ | Environment: development
INFO | __main__ | Project initialization completed successfully.
```

## Code quality

```bash
ruff check .
pytest
```

Commit 2 adds sample knowledge-base content only, so Commit 1 application behavior remains
unchanged.

## Knowledge-base structure

```text
data/
├── docs/
│   ├── authentication.md
│   ├── error_codes.md
│   ├── orders_api.md
│   └── rate_limits.md
├── incidents/
│   └── known_incidents.json
├── logs/
│   └── order_service.jsonl
├── openapi/
│   └── order-service.yaml
└── runbooks/
    ├── authentication_failures.md
    ├── downstream_timeouts.md
    ├── order_creation_failures.md
    └── service_unavailable.md
```

## Known request IDs

| Request ID | Expected diagnosis |
|---|---|
| `req-auth-401` | expired JWT |
| `req-scope-403` | missing `orders:write` |
| `req-json-400` | malformed JSON |
| `req-schema-422` | invalid item quantity |
| `req-idem-409` | idempotency conflict |
| `req-rate-429` | rate limit exceeded |
| `req-db-500` | database connection failure |
| `req-pay-502` | Payment Service returned 500 |
| `req-pay-504` | Payment Service exceeded 3000 ms timeout |
| `req-pool-503` | database connection pool exhausted |
| `req-notfound-404` | unknown order ID |

## Planned progression

Later commits will add document ingestion, chunking, embeddings, retrieval, baseline RAG,
OpenAPI inspection, log/incident tools, LangGraph orchestration, evidence synthesis,
FastAPI, evaluation, observability, Docker, and CI.

## Commit message

```text
feat: add sample order service API and troubleshooting knowledge base
```
