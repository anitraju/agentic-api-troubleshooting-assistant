"""Tests for metadata-aware semantic retrieval."""

from typing import Any

import pytest

from app.ingestion.models import SourceType
from app.rag.retrieval import RetrievalFilters, SemanticRetriever
from app.rag.vector_store import VectorSearchHit


class FixedEmbedder:
    """Return a fixed query embedding without loading an external model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class CapturingVectorStore:
    """Small vector-store double that records retrieval arguments."""

    def __init__(self, hits: list[VectorSearchHit]) -> None:
        self.hits = hits
        self.query_embedding: list[float] | None = None
        self.n_results: int | None = None
        self.where: dict[str, Any] | None = None

    def query(
        self,
        query_embedding: list[float],
        *,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorSearchHit]:
        self.query_embedding = query_embedding
        self.n_results = n_results
        self.where = where
        return self.hits[:n_results]


def _hit(
    chunk_id: str = "chunk-1",
    *,
    distance: float = 0.1,
    source: str = "docs/authentication.md",
    source_type: str = "documentation",
    section: str = "Expired tokens",
) -> VectorSearchHit:
    return VectorSearchHit(
        chunk_id=chunk_id,
        document="Expired access tokens return HTTP 401.",
        metadata={
            "source": source,
            "source_type": source_type,
            "section": section,
            "service": "order-service",
        },
        distance=distance,
    )


def test_retriever_returns_ranked_source_attributed_results() -> None:
    store = CapturingVectorStore(
        [
            _hit("chunk-1", distance=0.05),
            _hit(
                "chunk-2",
                distance=0.2,
                source="runbooks/authentication.md",
                source_type="runbook",
                section="Token recovery",
            ),
        ]
    )
    retriever = SemanticRetriever(store, FixedEmbedder(), default_top_k=2)

    results = retriever.retrieve("Why is the API returning 401?")

    assert store.query_embedding == [1.0, 0.0]
    assert store.n_results == 2
    assert results[0].rank == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].source == "docs/authentication.md"
    assert results[0].source_type == SourceType.DOCUMENTATION
    assert results[0].citation == "[1] docs/authentication.md — Expired tokens"
    assert results[0].metadata["service"] == "order-service"
    assert "source" not in results[0].metadata
    assert "source_type" not in results[0].metadata
    assert results[1].rank == 2


def test_retriever_builds_combined_chroma_metadata_filter() -> None:
    store = CapturingVectorStore([_hit()])
    retriever = SemanticRetriever(store, FixedEmbedder())
    filters = RetrievalFilters(
        source_types=(SourceType.RUNBOOK, SourceType.INCIDENT),
        service="order-service",
        endpoint="/orders",
        http_method="post",
        status_code=503,
        metadata={"environment": "production"},
    )

    retriever.retrieve("Order creation is failing", filters=filters)

    assert store.where == {
        "$and": [
            {"source_type": {"$in": ["runbook", "incident"]}},
            {"service": "order-service"},
            {"endpoint": "/orders"},
            {"http_method": "POST"},
            {"status_code": 503},
            {"environment": "production"},
        ]
    }


def test_retriever_uses_single_filter_without_unnecessary_and() -> None:
    store = CapturingVectorStore([_hit()])
    retriever = SemanticRetriever(store, FixedEmbedder())

    retriever.retrieve(
        "Show runbook guidance",
        filters=RetrievalFilters(source_types=(SourceType.RUNBOOK,)),
    )

    assert store.where == {"source_type": "runbook"}


def test_retriever_allows_empty_filters() -> None:
    store = CapturingVectorStore([_hit()])
    retriever = SemanticRetriever(store, FixedEmbedder())

    retriever.retrieve("Authentication failure", filters=RetrievalFilters())

    assert store.where is None


def test_retriever_rejects_blank_query_before_embedding() -> None:
    store = CapturingVectorStore([])
    retriever = SemanticRetriever(store, FixedEmbedder())

    with pytest.raises(ValueError, match="query cannot be empty"):
        retriever.retrieve("   ")


def test_retriever_rejects_invalid_top_k() -> None:
    store = CapturingVectorStore([])
    retriever = SemanticRetriever(store, FixedEmbedder())

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("Authentication failure", top_k=0)


def test_retriever_rejects_empty_custom_metadata_key() -> None:
    store = CapturingVectorStore([_hit()])
    retriever = SemanticRetriever(store, FixedEmbedder())

    with pytest.raises(ValueError, match="metadata filter keys"):
        retriever.retrieve(
            "Authentication failure",
            filters=RetrievalFilters(metadata={"   ": "value"}),
        )


def test_retriever_requires_source_attribution_from_vector_store() -> None:
    hit = VectorSearchHit(
        chunk_id="chunk-1",
        document="Troubleshooting content",
        metadata={"source_type": "documentation"},
        distance=0.1,
    )
    store = CapturingVectorStore([hit])
    retriever = SemanticRetriever(store, FixedEmbedder())

    with pytest.raises(ValueError, match="missing source attribution"):
        retriever.retrieve("Authentication failure")
