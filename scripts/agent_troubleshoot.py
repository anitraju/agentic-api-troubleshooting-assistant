"""Run the LangGraph API troubleshooting agent from the command line."""

from __future__ import annotations

import argparse

from app.agent import OpenAIIssueClassifier, TroubleshootingAgent
from app.config import get_settings
from app.ingestion.models import SourceType
from app.rag import (
    ChromaVectorStore,
    OpenAITroubleshootingGenerator,
    RetrievalFilters,
    SemanticRetriever,
    SentenceTransformerEmbedder,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the four-node LangGraph API troubleshooting agent."
    )
    parser.add_argument("query", help="Natural-language troubleshooting query")
    parser.add_argument("--top-k", type=int, default=None, help="Maximum evidence chunks")
    parser.add_argument(
        "--source-type",
        action="append",
        choices=[source_type.value for source_type in SourceType],
        default=[],
        help="Restrict retrieval to one or more source types; repeat as needed",
    )
    parser.add_argument("--service", help="Filter by service metadata")
    parser.add_argument("--endpoint", help="Filter by exact endpoint metadata")
    parser.add_argument("--http-method", help="Filter by HTTP method, for example POST")
    parser.add_argument("--status-code", type=int, help="Filter by exact HTTP status code")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    settings = get_settings()

    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value().strip():
        raise SystemExit(
            "OPENAI_API_KEY is required. Add it to your local .env file before running "
            "the LangGraph troubleshooting agent."
        )

    api_key = settings.openai_api_key.get_secret_value()

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
    classifier = OpenAIIssueClassifier(
        model_name=settings.openai_model,
        api_key=api_key,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    generator = OpenAITroubleshootingGenerator(
        model_name=settings.openai_model,
        api_key=api_key,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    agent = TroubleshootingAgent(
        classifier,
        retriever,
        generator,
        default_top_k=settings.retrieval_top_k,
        max_generation_attempts=settings.agent_max_generation_attempts,
    )

    filters = RetrievalFilters(
        source_types=tuple(SourceType(value) for value in args.source_type),
        service=args.service,
        endpoint=args.endpoint,
        http_method=args.http_method,
        status_code=args.status_code,
    )

    state = agent.run_with_state(
        args.query,
        top_k=args.top_k,
        filters=filters,
    )

    classification = state["classification"]
    evidence = state.get("evidence", [])
    attempts = state.get("generation_attempts", 0)

    print("## Agent trace")
    print()
    print(f"- Category: {classification.category.value}")
    print(f"- Summary: {classification.summary}")
    print(f"- Retrieved evidence: {len(evidence)}")
    print(f"- Retrieval fallback used: {state.get('retrieval_fallback_used', False)}")
    print(f"- Generation attempts: {attempts}")
    print("- Workflow: classify -> retrieve -> diagnose -> verify")
    print()
    print(state["answer"].render_markdown())


if __name__ == "__main__":
    main()
