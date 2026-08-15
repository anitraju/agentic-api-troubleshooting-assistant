"""Persistent Chroma vector-store integration."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import chromadb

from app.ingestion.models import KnowledgeChunk
from app.rag.embeddings import Embedder


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    """One ranked record returned by the vector store."""

    chunk_id: str
    document: str
    metadata: dict[str, Any]
    distance: float


class VectorSearcher(Protocol):
    """Minimal nearest-neighbor search contract used by the retrieval layer."""

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorSearchHit]:
        """Return ranked vector-search hits."""


class ChromaVectorStore:
    """Persist, manage, and query retrieval chunks in a local Chroma collection."""

    def __init__(
        self,
        persist_dir: Path | str,
        collection_name: str,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty")

        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def upsert_chunks(
        self,
        chunks: Iterable[KnowledgeChunk],
        embedder: Embedder,
        *,
        batch_size: int = 64,
    ) -> int:
        """Embed and upsert chunks in deterministic batches.

        Upsert semantics make index construction safe to rerun because identical chunk IDs
        are updated rather than duplicated.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        chunk_list = list(chunks)
        if not chunk_list:
            return 0

        indexed = 0
        for start in range(0, len(chunk_list), batch_size):
            batch = chunk_list[start : start + batch_size]
            documents = [chunk.content for chunk in batch]
            embeddings = embedder.embed_documents(documents)

            if len(embeddings) != len(batch):
                raise ValueError(
                    "embedding count does not match the number of chunks being indexed"
                )

            dimensions = {len(vector) for vector in embeddings}
            if not dimensions or 0 in dimensions:
                raise ValueError("embeddings must contain at least one numeric dimension")
            if len(dimensions) != 1:
                raise ValueError("all embeddings in a batch must have the same dimension")

            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=documents,
                embeddings=embeddings,
                metadatas=[_build_chroma_metadata(chunk) for chunk in batch],
            )
            indexed += len(batch)

        return indexed

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorSearchHit]:
        """Return nearest-neighbor records for one pre-computed query embedding."""
        if n_results < 1:
            raise ValueError("n_results must be at least 1")
        if len(query_embedding) == 0:
            raise ValueError("query_embedding cannot be empty")
        if self.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = _first_query_result(result.get("ids"))
        documents = _first_query_result(result.get("documents"))
        metadatas = _first_query_result(result.get("metadatas"))
        distances = _first_query_result(result.get("distances"))

        hits: list[VectorSearchHit] = []
        for index, chunk_id in enumerate(ids):
            document = _required_result_value(documents, index, "document")
            metadata = _required_result_value(metadatas, index, "metadata")
            distance = _required_result_value(distances, index, "distance")

            if not isinstance(document, str):
                raise ValueError("Chroma returned a non-string document")
            if not isinstance(metadata, dict):
                raise ValueError("Chroma returned invalid metadata")
            try:
                numeric_distance = float(distance)
            except (TypeError, ValueError) as exc:
                raise ValueError("Chroma returned an invalid distance") from exc

            hits.append(
                VectorSearchHit(
                    chunk_id=str(chunk_id),
                    document=document,
                    metadata=dict(metadata),
                    distance=numeric_distance,
                )
            )

        return hits

    def count(self) -> int:
        """Return the number of records in the current collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection for a clean index rebuild."""
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.get_or_create_collection(name=self.collection_name)

    def get_record(self, chunk_id: str) -> dict[str, Any]:
        """Return one indexed record by chunk ID for diagnostics and tests."""
        result = self._collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"],
        )

        ids = result.get("ids") or []
        if not ids:
            raise KeyError(f"chunk not found: {chunk_id}")

        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return {
            "id": ids[0],
            "document": documents[0] if documents else None,
            "metadata": metadatas[0] if metadatas else {},
        }


def _first_query_result(value: Any) -> list[Any]:
    if not value:
        return []
    first = value[0]
    return list(first) if first is not None else []


def _required_result_value(values: list[Any], index: int, field_name: str) -> Any:
    if index >= len(values) or values[index] is None:
        raise ValueError(f"Chroma query result is missing {field_name} data")
    return values[index]


def _build_chroma_metadata(chunk: KnowledgeChunk) -> dict[str, str | int | float | bool]:
    """Convert chunk metadata into scalar values accepted consistently by Chroma."""
    raw_metadata: dict[str, Any] = {
        **chunk.metadata,
        "source": chunk.source,
        "source_type": chunk.source_type.value,
    }
    sanitized: dict[str, str | int | float | bool] = {}

    for key, value in raw_metadata.items():
        if value is None:
            continue

        if isinstance(value, bool | int | float | str):
            sanitized[key] = value
        else:
            sanitized[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))

    return sanitized
