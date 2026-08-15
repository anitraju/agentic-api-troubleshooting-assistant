"""Metadata-aware semantic retrieval with source attribution."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, computed_field

from app.ingestion.models import SourceType
from app.rag.embeddings import Embedder
from app.rag.vector_store import VectorSearcher, VectorSearchHit

MetadataScalar = str | int | float | bool


class RetrievalFilters(BaseModel):
    """Troubleshooting-oriented metadata filters applied before vector ranking."""

    source_types: tuple[SourceType, ...] = ()
    service: str | None = None
    endpoint: str | None = None
    http_method: str | None = None
    status_code: int | None = Field(default=None, ge=100, le=599)
    metadata: dict[str, MetadataScalar] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Citation-ready retrieval result returned to later RAG and agent stages."""

    rank: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_type: SourceType
    distance: float = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def citation(self) -> str:
        """Return a compact human-readable citation label for this result."""
        location = self.metadata.get("section") or self.metadata.get("title")
        suffix = f" — {location}" if location else ""
        return f"[{self.rank}] {self.source}{suffix}"


class SemanticRetriever:
    """Embed user queries and retrieve ranked, attributed knowledge chunks."""

    def __init__(
        self,
        vector_store: VectorSearcher,
        embedder: Embedder,
        *,
        default_top_k: int = 5,
    ) -> None:
        if default_top_k < 1:
            raise ValueError("default_top_k must be at least 1")

        self.vector_store: VectorSearcher = vector_store
        self.embedder = embedder
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalResult]:
        """Return semantic matches ordered from nearest to farthest."""
        if not query.strip():
            raise ValueError("query cannot be empty")

        result_limit = self.default_top_k if top_k is None else top_k
        if result_limit < 1:
            raise ValueError("top_k must be at least 1")

        query_embedding = self.embedder.embed_query(query)
        hits = self.vector_store.query(
            query_embedding,
            n_results=result_limit,
            where=_build_chroma_where(filters),
        )

        return [_to_retrieval_result(hit, rank) for rank, hit in enumerate(hits, start=1)]


def _to_retrieval_result(hit: VectorSearchHit, rank: int) -> RetrievalResult:
    metadata = dict(hit.metadata)
    source = metadata.pop("source", None)
    source_type = metadata.pop("source_type", None)

    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"retrieved chunk {hit.chunk_id} is missing source attribution")
    if not isinstance(source_type, str):
        raise ValueError(f"retrieved chunk {hit.chunk_id} is missing source_type attribution")

    try:
        normalized_source_type = SourceType(source_type)
    except ValueError as exc:
        raise ValueError(
            f"retrieved chunk {hit.chunk_id} has unsupported source_type: {source_type}"
        ) from exc

    return RetrievalResult(
        rank=rank,
        chunk_id=hit.chunk_id,
        content=hit.document,
        source=source,
        source_type=normalized_source_type,
        distance=hit.distance,
        metadata=metadata,
    )


def _build_chroma_where(filters: RetrievalFilters | None) -> dict[str, Any] | None:
    if filters is None:
        return None

    conditions: list[dict[str, Any]] = []

    if filters.source_types:
        source_values = list(
            dict.fromkeys(source_type.value for source_type in filters.source_types)
        )
        if len(source_values) == 1:
            conditions.append({"source_type": source_values[0]})
        else:
            conditions.append({"source_type": {"$in": source_values}})

    if filters.service:
        conditions.append({"service": filters.service})
    if filters.endpoint:
        conditions.append({"endpoint": filters.endpoint})
    if filters.http_method:
        conditions.append({"http_method": filters.http_method.upper()})
    if filters.status_code is not None:
        conditions.append({"status_code": filters.status_code})

    for key, value in filters.metadata.items():
        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError("metadata filter keys cannot be empty")
        conditions.append({normalized_key: value})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
