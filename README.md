# Agentic API Troubleshooting Assistant Using RAG

A portfolio project for building an AI assistant that investigates API failures using
retrieval-augmented generation (RAG), structured troubleshooting tools, and an agentic
workflow.

## Current status

**Commit 6 — Metadata-aware semantic retrieval with source attribution**

The project can now turn a natural-language troubleshooting question into a query embedding,
search the persistent Chroma knowledge index, apply structured metadata filters, and return
ranked retrieval results with source attribution that later RAG and agent stages can cite.

The retrieval flow is:

```text
User troubleshooting query
          ↓
SentenceTransformerEmbedder.embed_query(...)
          ↓
Normalized query embedding
          ↓
Optional metadata filters
          ↓
Persistent Chroma collection
          ↓
Nearest-neighbor ranking
          ↓
VectorSearchHit
          ↓
RetrievalResult
          ↓
Citation-ready context for later RAG stages
```

## Retrieval design

The retrieval layer stays separated from Chroma-specific storage details.

`VectorSearcher` is the small nearest-neighbor search contract consumed by the high-level
retriever. `ChromaVectorStore` implements that contract and returns `VectorSearchHit` objects.

`SemanticRetriever` is responsible for:

- validating the user query and result limit;
- creating the query embedding through the existing `Embedder` abstraction;
- converting troubleshooting filters into a Chroma `where` expression;
- requesting nearest-neighbor results from the vector store;
- converting stored source metadata into typed, citation-ready `RetrievalResult` objects.

This keeps later RAG and LangGraph code independent of Chroma's raw query-result structure.

## Retrieval result model

Each `RetrievalResult` contains:

```text
rank
chunk_id
content
source
source_type
distance
metadata
citation
```

`source` and `source_type` are promoted out of raw vector-store metadata so that downstream
reasoning code cannot accidentally lose source attribution.

The computed `citation` field produces a compact label such as:

```text
[1] docs/authentication.md — Expired access tokens
```

The retrieval layer keeps Chroma's distance value rather than inventing a provider-specific
similarity score. Lower distance means the result was ranked closer to the query.

## Metadata-aware retrieval

`RetrievalFilters` supports the troubleshooting signals already created during ingestion and
chunking:

```text
source_types
service
endpoint
http_method
status_code
metadata
```

`metadata` can contain additional scalar equality filters when a later workflow needs a field
that is not yet modeled explicitly.

Multiple filters are combined with Chroma's `$and` expression. Multiple source types use
`$in`.

Examples of useful retrieval constraints include:

```text
source_type = runbook
service = order-service
endpoint = /orders
http_method = POST
status_code = 503
```

Semantic ranking still happens inside the filtered candidate set.

## Configuration

Commit 6 adds one retrieval setting:

```text
RETRIEVAL_TOP_K=5
```

The complete retrieval/index configuration is now:

```text
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=32
CHROMA_PERSIST_DIR=chroma_db
CHROMA_COLLECTION_NAME=order-service-knowledge
VECTOR_UPSERT_BATCH_SIZE=64
RETRIEVAL_TOP_K=5
```

Copy any new value you want from `.env.example` into your local `.env`. The default also works
without explicitly adding it.

## Build the index

Commit 6 reads the persistent index created in Commit 5, so build or rebuild it first whenever
knowledge or chunking changes:

```bash
python -m scripts.build_index
```

With the current sample knowledge base, the index should contain 61 chunks.

## Query the index

A small CLI is included for manual retrieval checks.

Basic semantic search:

```bash
python -m scripts.query_index "Why am I getting 401 when calling the order API?"
```

Return only three results:

```bash
python -m scripts.query_index \
  "Why is order creation returning 503?" \
  --top-k 3
```

Restrict the candidate set to runbooks:

```bash
python -m scripts.query_index \
  "How should I troubleshoot expired authentication tokens?" \
  --source-type runbook
```

Combine API metadata filters:

```bash
python -m scripts.query_index \
  "Order creation is failing" \
  --service order-service \
  --endpoint /orders \
  --http-method POST \
  --status-code 503
```

Multiple source types can be supplied by repeating the option:

```bash
python -m scripts.query_index \
  "Find evidence for a database pool outage" \
  --source-type runbook \
  --source-type incident
```

## Tests

The test suite continues to avoid Sentence Transformer downloads by using deterministic or
fixed in-memory test embedders.

Run:

```bash
ruff check .
pytest
```

Commit 6 tests cover:

- nearest-neighbor vector queries;
- ranked hit conversion;
- metadata filtering;
- multiple source-type filters;
- query and top-k validation;
- empty collections;
- source and source-type attribution;
- citation generation;
- custom scalar metadata filters.

## Files added or changed

```text
app/
├── config.py
└── rag/
    ├── __init__.py
    ├── retrieval.py
    └── vector_store.py

scripts/
└── query_index.py

tests/
├── test_retrieval.py
└── test_vector_store.py

.env.example
README.md
```

No new third-party dependency is required for this commit.

## Architecture boundary

Commit 6 is responsible for **retrieving evidence**.

It deliberately does not yet generate a troubleshooting answer, call an LLM, inspect live logs,
or introduce LangGraph orchestration. The output of this commit is a ranked collection of
source-attributed evidence that those later stages can consume.

## Next commit

Commit 7 will build the baseline RAG troubleshooting pipeline on top of these retrieval results,
so the system can turn retrieved evidence into a grounded diagnosis and remediation response.

## Commit message

```text
add metadata-aware semantic retrieval
```
