"""Tests for embedding abstractions."""

import pytest

from app.rag.embeddings import SentenceTransformerEmbedder


def test_sentence_transformer_embedder_validates_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        SentenceTransformerEmbedder(batch_size=0)


def test_sentence_transformer_embedder_rejects_empty_query_without_loading_model() -> None:
    embedder = SentenceTransformerEmbedder()

    with pytest.raises(ValueError, match="query text"):
        embedder.embed_query("   ")


def test_sentence_transformer_embedder_handles_empty_document_batch_without_loading_model() -> None:
    embedder = SentenceTransformerEmbedder()

    assert embedder.embed_documents([]) == []
