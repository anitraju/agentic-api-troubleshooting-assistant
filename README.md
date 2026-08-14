# Agentic API Troubleshooting Assistant Using RAG

A portfolio project for building an AI assistant that investigates API failures using
retrieval-augmented generation (RAG), structured troubleshooting tools, and an agentic
workflow.

## Current status

**Commit 4 — Metadata-aware document chunking**

The ingestion layer now converts normalized knowledge documents into retrieval-ready
`KnowledgeChunk` objects while preserving troubleshooting context.

Chunk metadata can include:

- source and source type
- section heading
- endpoint
- HTTP method
- HTTP status code(s)
- service
- incident ID
- OpenAPI operation ID
- chunk index and chunk count

## Why metadata-aware chunking matters

Naive fixed-character splitting can separate an error explanation from the heading or
endpoint that gives it meaning. The current chunker first respects document structure and
then applies bounded splitting only when a section is too large.

For Markdown, section headings are repeated on split chunks so each retrieval unit remains
understandable in isolation.

For OpenAPI, the raw specification is not treated as one giant YAML document. Each API
operation becomes its own retrieval unit.

For example:

```text
POST /api/v1/orders
GET /api/v1/orders/{order_id}
GET /health
```

OpenAPI chunks also include locally referenced parameters, schemas, and response
definitions so a retrieved endpoint chunk contains useful contract details instead of only
unresolved `$ref` values.

## Retrieval-ready model

```python
KnowledgeChunk(
    chunk_id="chunk-...",
    content="...",
    source="docs/orders_api.md",
    source_type=SourceType.DOCUMENTATION,
    metadata={
        "section": "Create Order",
        "endpoint": "/api/v1/orders",
        "http_method": "POST",
        "status_code": 409,
        "service": "order-service",
        "chunk_index": 0,
        "chunk_count": 1,
    },
)
```

Chunk IDs are deterministic SHA-256-derived identifiers based on source, chunk position,
and content.

## Project structure

Commit 4 adds or changes:

```text
app/ingestion/
├── __init__.py
├── chunking.py
└── models.py

scripts/
└── inspect_chunks.py

tests/
└── test_chunking.py
```

No new dependency is introduced in this commit.

## Inspect the chunks

```bash
python -m scripts.inspect_chunks
```

With the Commit 2 sample knowledge base and default chunk settings, the expected summary is:

```text
Loaded 15 knowledge documents
Created 61 retrieval chunks
- documentation: 29
- incident: 6
- openapi: 9
- runbook: 17
```

The script also prints sample chunk IDs and metadata.

## Validate

```bash
ruff check .
pytest
python -m scripts.inspect_chunks
```

The chunking test suite covers:

- Markdown section preservation
- endpoint/method/status extraction
- long-section splitting
- incident metadata preservation
- OpenAPI operation-level chunking
- local OpenAPI `$ref` resolution
- deterministic unique chunk IDs
- full-repository chunk generation
- invalid chunk-size configuration

## Architecture so far

```text
Raw knowledge sources
        ↓
Document loaders
        ↓
KnowledgeDocument
        ↓
Metadata-aware chunker
        ↓
KnowledgeChunk
        ↓
Embedding + vector index (next)
```

Operational logs remain outside this flow. They will later be accessed through a dedicated
log-search tool.

## Next commit

Commit 5 will add the embedding model and Chroma vector store so these chunks can be
indexed for semantic retrieval.

## Commit message

```text
feat: add metadata-aware document chunking
```
