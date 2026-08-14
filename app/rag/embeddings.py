"""Embedding abstractions used by the RAG pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    """Minimal embedding contract used by the vector-store layer."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed knowledge-base documents."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query."""


class SentenceTransformerEmbedder:
    """Local Sentence Transformers implementation of the embedding contract."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return normalized dense embeddings for knowledge documents."""
        if not texts:
            return []

        embeddings = self._get_model().encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Return a normalized dense embedding for one query."""
        if not text.strip():
            raise ValueError("query text cannot be empty")

        return self.embed_documents([text])[0]

    def _get_model(self):
        """Load the model lazily so importing the application remains lightweight."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

        return self._model
