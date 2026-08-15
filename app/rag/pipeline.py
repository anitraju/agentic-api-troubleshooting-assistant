"""Baseline retrieve-then-generate RAG pipeline with grounding validation."""

from __future__ import annotations

from collections.abc import Iterable

from app.rag.answers import (
    Confidence,
    EvidenceBackedStatement,
    GeneratedTroubleshootingAnalysis,
    TroubleshootingAnswer,
)
from app.rag.generation import TroubleshootingGenerator
from app.rag.retrieval import RetrievalFilters, SemanticRetriever


class UngroundedGenerationError(ValueError):
    """Raised when generated findings cite evidence that was not retrieved."""


class GroundedRAGPipeline:
    """Retrieve troubleshooting evidence, generate analysis, and validate grounding."""

    def __init__(
        self,
        retriever: SemanticRetriever,
        generator: TroubleshootingGenerator,
        *,
        default_top_k: int = 5,
    ) -> None:
        if default_top_k < 1:
            raise ValueError("default_top_k must be at least 1")

        self.retriever = retriever
        self.generator = generator
        self.default_top_k = default_top_k

    def run(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> TroubleshootingAnswer:
        """Run one baseline RAG troubleshooting request."""
        if not query.strip():
            raise ValueError("query cannot be empty")

        result_limit = self.default_top_k if top_k is None else top_k
        if result_limit < 1:
            raise ValueError("top_k must be at least 1")

        evidence = self.retriever.retrieve(
            query,
            top_k=result_limit,
            filters=filters,
        )

        if not evidence:
            return _insufficient_evidence_answer(query)

        analysis = self.generator.generate(query, evidence)
        _validate_grounding(analysis, {item.rank for item in evidence})

        return TroubleshootingAnswer(
            query=query.strip(),
            analysis=analysis,
            evidence=evidence,
        )


def _validate_grounding(
    analysis: GeneratedTroubleshootingAnalysis,
    valid_ranks: set[int],
) -> None:
    for statement in _iter_evidence_backed_statements(analysis):
        invalid = [rank for rank in statement.evidence_ranks if rank not in valid_ranks]
        if invalid:
            raise UngroundedGenerationError(
                "generated statement referenced unavailable evidence rank(s): "
                + ", ".join(str(rank) for rank in invalid)
            )


def _iter_evidence_backed_statements(
    analysis: GeneratedTroubleshootingAnalysis,
) -> Iterable[EvidenceBackedStatement]:
    yield from analysis.likely_causes
    yield from analysis.diagnostic_steps
    yield from analysis.remediation_steps


def _insufficient_evidence_answer(query: str) -> TroubleshootingAnswer:
    return TroubleshootingAnswer(
        query=query.strip(),
        analysis=GeneratedTroubleshootingAnalysis(
            summary=(
                "The knowledge index did not return evidence that can support a grounded "
                "diagnosis for this query."
            ),
            confidence=Confidence.LOW,
            limitations=[
                "No retrieved evidence was available, so no cause or remediation is asserted."
            ],
        ),
        evidence=[],
    )
