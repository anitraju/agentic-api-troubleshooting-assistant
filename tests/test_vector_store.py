"""Tests for persistent vector indexing."""

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
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        content=content,
        source="docs/example.md",
        source_type=SourceType.DOCUMENTATION,
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
