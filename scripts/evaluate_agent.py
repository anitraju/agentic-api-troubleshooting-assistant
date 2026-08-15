"""Run curated live evaluations against the LangGraph troubleshooting agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.agent import OpenAIIssueClassifier, TroubleshootingAgent
from app.config import get_settings
from app.evaluation import EvaluationRunner, load_evaluation_cases
from app.rag import (
    ChromaVectorStore,
    OpenAITroubleshootingGenerator,
    SemanticRetriever,
    SentenceTransformerEmbedder,
)

DEFAULT_DATASET = Path("data/evaluation/scenarios.json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the LangGraph troubleshooting agent on curated scenarios."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Evaluation dataset path (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only the named case; repeat to select multiple cases",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N selected cases",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON report",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with status 1 when one or more cases fail",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    settings = get_settings()
    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value().strip():
        raise SystemExit(
            "OPENAI_API_KEY is required for live evaluation. "
            "Add it to your local .env file."
        )

    cases = load_evaluation_cases(args.dataset)

    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case.case_id in selected]

        missing = selected - {case.case_id for case in cases}
        if missing:
            raise SystemExit(
                "Unknown evaluation case ID(s): " + ", ".join(sorted(missing))
            )

    if args.limit is not None:
        cases = cases[: args.limit]

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

    report = EvaluationRunner(agent).run(cases)
    print(report.render_text())

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        print()
        print(f"JSON report: {args.output}")

    if args.fail_on_regression and report.passed_cases != report.total_cases:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
