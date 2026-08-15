"""Query the persistent troubleshooting knowledge index from the command line."""

from __future__ import annotations

import argparse

from app.config import get_settings
from app.ingestion.models import SourceType
from app.rag import (
    ChromaVectorStore,
    RetrievalFilters,
    SemanticRetriever,
    SentenceTransformerEmbedder,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run metadata-aware semantic retrieval against the local Chroma index."
    )
    parser.add_argument("query", help="Natural-language troubleshooting query")
    parser.add_argument("--top-k", type=int, default=None, help="Maximum results to return")
    parser.add_argument(
        "--source-type",
        action="append",
        choices=[source_type.value for source_type in SourceType],
        default=[],
        help="Restrict retrieval to one or more source types; repeat the option as needed",
    )
    parser.add_argument("--service", help="Filter by service metadata")
    parser.add_argument("--endpoint", help="Filter by exact endpoint metadata")
    parser.add_argument("--http-method", help="Filter by HTTP method, for example POST")
    parser.add_argument("--status-code", type=int, help="Filter by exact HTTP status code")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    settings = get_settings()

    embedder = SentenceTransformerEmbedder(
        model_name=settings.embedding_model_name,
        batch_size=settings.embedding_batch_size,
    )
    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )
    retriever = SemanticRetriever(
        vector_store,
        embedder,
        default_top_k=settings.retrieval_top_k,
    )

    filters = RetrievalFilters(
        source_types=tuple(SourceType(value) for value in args.source_type),
        service=args.service,
        endpoint=args.endpoint,
        http_method=args.http_method,
        status_code=args.status_code,
    )
    results = retriever.retrieve(args.query, top_k=args.top_k, filters=filters)

    if not results:
        print("No retrieval results found.")
        return

    for result in results:
        print(result.citation)
        print(f"source_type={result.source_type.value} | distance={result.distance:.6f}")
        print(f"chunk_id={result.chunk_id}")
        print(result.content)
        print("-" * 80)


if __name__ == "__main__":
    main()
