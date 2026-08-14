# Agentic API Troubleshooting Assistant Using RAG

A portfolio project for building an AI assistant that investigates API failures using
retrieval-augmented generation (RAG), structured troubleshooting tools, and an agentic
workflow.

## Current status

**Commit 3 — Knowledge-base document loaders**

The project now has an ingestion layer that converts the heterogeneous troubleshooting
knowledge base into a common `KnowledgeDocument` representation.

Supported RAG knowledge sources:

- Markdown API documentation
- Markdown operational runbooks
- historical incidents stored as JSON
- the Order Service OpenAPI YAML specification

Structured request logs are intentionally **not** loaded into the RAG knowledge base. They
are request-specific operational evidence and will later be accessed through a dedicated
log-search tool.

## Ingestion model

Every loaded source is normalized to:

```python
KnowledgeDocument(
    content="...",
    source="...",
    source_type=SourceType.DOCUMENTATION,
    metadata={...},
)
```

The source types are:

- `documentation`
- `runbook`
- `incident`
- `openapi`

Historical incident records are loaded as independent documents, which makes it possible
for retrieval to return one specific known incident rather than the entire incident file.

## Current knowledge-base counts

The controlled sample environment produces:

| Source type | Documents |
|---|---:|
| Documentation | 4 |
| Runbooks | 4 |
| Historical incidents | 6 |
| OpenAPI specifications | 1 |
| **Total** | **15** |

## Project structure

```text
app/
└── ingestion/
    ├── __init__.py
    ├── loaders.py
    └── models.py

scripts/
└── inspect_knowledge_base.py

tests/
└── test_ingestion.py
```

## Requirements

- Python 3.11 or newer
- `pip`
- Git

Commit 3 introduces `PyYAML` for safe OpenAPI YAML parsing.

## Install or update dependencies

With the virtual environment active:

```bash
pip install -r requirements.txt
```

Because `requirements.txt` installs the project in editable mode, the new `PyYAML`
dependency declared in `pyproject.toml` will be installed automatically.

## Run the application

```bash
python -m app.main
```

## Inspect the ingestion pipeline

```bash
python -m scripts.inspect_knowledge_base
```

Expected output:

```text
Loaded 15 knowledge documents
- documentation: 4
- incident: 6
- openapi: 1
- runbook: 4
```

It will also print every normalized source.

## Validate

```bash
ruff check .
pytest
```

The ingestion tests verify:

- Markdown title and metadata extraction
- one-document-per-incident normalization
- malformed incident structure handling
- OpenAPI metadata parsing
- expected full-repository knowledge counts
- exclusion of structured request logs from the RAG corpus

## Why logs are excluded from RAG ingestion

API documentation, runbooks, and historical incidents are relatively stable knowledge.
Request logs are dynamic evidence tied to a specific troubleshooting investigation.

Keeping these concerns separate allows the final agent to combine:

```text
retrieved knowledge
        +
request-specific log evidence
        +
OpenAPI inspection
        +
historical incident lookup
        ↓
evidence-based diagnosis
```

rather than embedding every operational log line into the knowledge vector store.

## Next commit

Commit 4 will add metadata-aware chunking so long documents can be split into retrieval
units without losing source, section, endpoint, HTTP status, and other useful context.

## Commit message

```text
feat: implement knowledge base document loaders
```
