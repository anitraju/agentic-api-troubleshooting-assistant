# Agentic API Troubleshooting Assistant Using RAG

A portfolio project for building an AI assistant that investigates API failures using
retrieval-augmented generation (RAG), structured troubleshooting tools, and an agentic
workflow.

## Current status

**Commit 5 — Local embeddings and persistent Chroma vector index**

The project can now convert metadata-aware `KnowledgeChunk` objects into normalized dense
embeddings and persist them in a local Chroma collection.

The indexing flow is:

```text
Raw knowledge
      ↓
Document loaders
      ↓
KnowledgeDocument
      ↓
Metadata-aware chunking
      ↓
KnowledgeChunk
      ↓
Sentence Transformer
      ↓
Dense embedding
      ↓
Persistent Chroma collection
```

## Embedding design

The application uses a small internal `Embedder` protocol instead of coupling the rest of
the code directly to Sentence Transformers.

The production implementation is:

```text
SentenceTransformerEmbedder
```

Default model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are normalized before storage.

This abstraction will also let later code replace the local model with another embedding
provider without changing the vector-store interface.

## Persistent vector store

`ChromaVectorStore` uses a local Chroma persistent client.

The default index is stored under:

```text
chroma_db/
```

This directory is already ignored by Git.

The collection name defaults to:

```text
order-service-knowledge
```

Index writes use `upsert`, making repeated indexing safe for deterministic chunk IDs.

## Metadata compatibility

Knowledge chunks can contain useful non-scalar metadata such as historical-incident
symptoms or signatures.

Before metadata is written to Chroma:

- strings, integers, floats, and booleans are preserved;
- `None` values are omitted;
- lists and dictionaries are serialized to deterministic JSON strings;
- `source` and `source_type` are always added.

This keeps the vector database representation stable while retaining troubleshooting
context.

## Configuration

Commit 5 adds:

```text
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=32
CHROMA_PERSIST_DIR=chroma_db
CHROMA_COLLECTION_NAME=order-service-knowledge
VECTOR_UPSERT_BATCH_SIZE=64
```

Copy any new values you want from `.env.example` into your local `.env`. Existing defaults
also work without explicitly adding them.

## Install dependencies

Commit 5 introduces:

- `sentence-transformers`
- `chromadb`

Update the active virtual environment with:

```bash
pip install -r requirements.txt
```

The first real index build may download the configured Sentence Transformer model if it is
not already present in the local model cache.

## Build the index

```bash
python -m scripts.build_index
```

With the current sample knowledge base, the summary should include:

```text
Loaded 15 knowledge documents
Created 61 retrieval chunks
Indexed 61 chunks
Collection count: 61
Collection: order-service-knowledge
Persistence directory: chroma_db
Embedding model: sentence-transformers/all-MiniLM-L6-v2
```

The build script deliberately performs a clean collection reset before indexing. This makes
local development reproducible when chunking or source data changes.

## Tests

Vector-store tests use a deterministic in-memory test embedder instead of downloading an ML
model. This keeps the test suite fast and repeatable.

Run:

```bash
ruff check .
pytest
```

Commit 5 tests cover:

- embedding input validation
- lazy model loading behavior
- Chroma persistence across store instances
- idempotent upserts
- metadata sanitization
- collection reset
- invalid embedding output handling

## Files added or changed

```text
app/
├── config.py
└── rag/
    ├── __init__.py
    ├── embeddings.py
    └── vector_store.py

scripts/
└── build_index.py

tests/
├── test_embeddings.py
└── test_vector_store.py

.env.example
pyproject.toml
README.md
```

## Architecture boundary

Commit 5 is responsible for **index construction and persistence**.

It does not yet implement semantic retrieval. Query embedding, nearest-neighbor retrieval,
metadata filters, ranking, and citation-ready results are intentionally reserved for
Commit 6.

## Next commit

Commit 6 will implement metadata-aware semantic retrieval with source attribution and
retrieval result models.

## Commit message

```text
feat: add Chroma vector store and knowledge indexing
```
