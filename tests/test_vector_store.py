"""Tests for persistent vector indexing and nearest-neighbor search."""

from pathlib import Path

import pytest

from app.ingestion.models import KnowledgeChunk, SourceType
from app.rag.vector_store import ChromaVectorStore


class DeterministicEmbedder:
    """Small deterministic embedder used to avoid model downloads in tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(len(text)),
            float(lowered.count("token")),
            float(lowered.count("payment")),
            float(lowered.count("database")),
        ]


def _chunk(
    chunk_id: str,
    content: str,
    *,
    metadata: dict | None = None,
    source: str = "docs/example.md",
    source_type: SourceType = SourceType.DOCUMENTATION,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        content=content,
        source=source,
        source_type=source_type,
        metadata=metadata or {},
    )


def test_vector_store_persists_records_across_instances(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma"
    first = ChromaVectorStore(persist_dir, "test-knowledge")
    first.upsert_chunks(
        [
            _chunk("chunk-1", "Expired token troubleshooting."),
            _chunk("chunk-2", "Payment service troubleshooting."),
        ],
        DeterministicEmbedder(),
    )

    assert first.count() == 2

    reopened = ChromaVectorStore(persist_dir, "test-knowledge")

    assert reopened.count() == 2
    assert reopened.get_record("chunk-1")["document"] == "Expired token troubleshooting."


def test_upsert_is_idempotent_for_existing_chunk_ids(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "test-knowledge")
    embedder = DeterministicEmbedder()

    store.upsert_chunks([_chunk("chunk-1", "Original text.")], embedder)
    store.upsert_chunks([_chunk("chunk-1", "Updated text.")], embedder)

    assert store.count() == 1
    assert store.get_record("chunk-1")["document"] == "Updated text."


def test_vector_store_sanitizes_complex_metadata(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "test-knowledge")
    store.upsert_chunks(
        [
            _chunk(
                "chunk-1",
                "Database pool incident.",
                metadata={
                    "status_code": 503,
                    "service": "order-service",
                    "symptoms": ["503", "db_pool_exhausted"],
                    "details": {"available_connections": 0},
                },
            )
        ],
        DeterministicEmbedder(),
    )

    metadata = store.get_record("chunk-1")["metadata"]
    assert metadata["source"] == "docs/example.md"
    assert metadata["source_type"] == "documentation"
    assert metadata["status_code"] == 503
    assert metadata["service"] == "order-service"
    assert metadata["symptoms"] == '["503","db_pool_exhausted"]'
    assert metadata["details"] == '{"available_connections":0}'


def test_query_returns_nearest_neighbor_hits_in_ranked_order(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "test-knowledge")
    embedder = DeterministicEmbedder()
    store.upsert_chunks(
        [
            _chunk("chunk-token", "Expired token troubleshooting."),
            _chunk("chunk-payment", "Payment database troubleshooting."),
        ],
        embedder,
    )

    hits = store.query(
        embedder.embed_query("Expired token troubleshooting."),
        n_results=2,
    )

    assert [hit.chunk_id for hit in hits] == ["chunk-token", "chunk-payment"]
    assert hits[0].document == "Expired token troubleshooting."
    assert hits[0].distance <= hits[1].distance
    assert hits[0].metadata["source"] == "docs/example.md"


def test_query_applies_metadata_filter_before_ranking(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "test-knowledge")
    embedder = DeterministicEmbedder()
    store.upsert_chunks(
        [
            _chunk(
                "chunk-doc",
                "Expired token troubleshooting.",
                metadata={"service": "order-service"},
            ),
            _chunk(
                "chunk-runbook",
                "Expired token troubleshooting runbook.",
                metadata={"service": "order-service"},
                source="runbooks/authentication.md",
                source_type=SourceType.RUNBOOK,
            ),
        ],
        embedder,
    )

    hits = store.query(
        embedder.embed_query("Expired token troubleshooting."),
        n_results=5,
        where={"source_type": "runbook"},
    )

    assert [hit.chunk_id for hit in hits] == ["chunk-runbook"]
    assert hits[0].metadata["source_type"] == "runbook"


def test_query_returns_empty_list_for_empty_collection(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "test-knowledge")

    assert store.query([1.0, 0.0]) == []


def test_query_rejects_empty_embedding(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "test-knowledge")

    with pytest.raises(ValueError, match="query_embedding"):
        store.query([])


def test_query_rejects_invalid_result_count(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "test-knowledge")

    with pytest.raises(ValueError, match="n_results"):
        store.query([1.0], n_results=0)


def test_reset_recreates_empty_collection(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "test-knowledge")
    store.upsert_chunks(
        [_chunk("chunk-1", "Some troubleshooting knowledge.")],
        DeterministicEmbedder(),
    )

    assert store.count() == 1

    store.reset()

    assert store.count() == 0


def test_vector_store_rejects_embedding_count_mismatch(tmp_path: Path) -> None:
    class BrokenEmbedder:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return []

        def embed_query(self, text: str) -> list[float]:
            return [1.0]

    store = ChromaVectorStore(tmp_path / "chroma", "test-knowledge")
    with pytest.raises(ValueError, match="embedding count"):
        store.upsert_chunks(
            [_chunk("chunk-1", "Troubleshooting knowledge.")],
            BrokenEmbedder(),
        )


def test_vector_store_rejects_empty_collection_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="collection_name"):
        ChromaVectorStore(tmp_path / "chroma", " ")
