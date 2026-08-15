"""Structured troubleshooting answer models for grounded RAG."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from app.rag.retrieval import RetrievalResult


class Confidence(StrEnum):
    """Coarse confidence level for a generated troubleshooting diagnosis."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EvidenceBackedStatement(BaseModel):
    """One troubleshooting statement tied to one or more retrieved evidence ranks."""

    text: str = Field(min_length=1)
    evidence_ranks: list[int] = Field(min_length=1)

    @field_validator("evidence_ranks")
    @classmethod
    def validate_evidence_ranks(cls, ranks: list[int]) -> list[int]:
        if any(rank < 1 for rank in ranks):
            raise ValueError("evidence ranks must be positive integers")
        return list(dict.fromkeys(ranks))


class GeneratedTroubleshootingAnalysis(BaseModel):
    """Structured payload produced by the LLM before grounding validation."""

    summary: str = Field(min_length=1)
    likely_causes: list[EvidenceBackedStatement] = Field(default_factory=list)
    diagnostic_steps: list[EvidenceBackedStatement] = Field(default_factory=list)
    remediation_steps: list[EvidenceBackedStatement] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW
    limitations: list[str] = Field(default_factory=list)


class TroubleshootingAnswer(BaseModel):
    """Final RAG answer containing validated analysis and the evidence used to create it."""

    query: str = Field(min_length=1)
    analysis: GeneratedTroubleshootingAnalysis
    evidence: list[RetrievalResult] = Field(default_factory=list)

    def render_markdown(self) -> str:
        """Render a human-readable answer while preserving evidence-rank citations."""
        lines = [
            "## Diagnosis",
            "",
            self.analysis.summary,
            "",
            f"**Confidence:** {self.analysis.confidence.value}",
        ]

        _append_statement_section(lines, "Likely causes", self.analysis.likely_causes)
        _append_statement_section(lines, "Diagnostic steps", self.analysis.diagnostic_steps)
        _append_statement_section(lines, "Remediation", self.analysis.remediation_steps)

        if self.analysis.limitations:
            lines.extend(["", "### Limitations"])
            lines.extend(f"- {item}" for item in self.analysis.limitations)

        if self.evidence:
            lines.extend(["", "### Sources"])
            lines.extend(f"- {item.citation}" for item in self.evidence)

        return "\n".join(lines)


def _append_statement_section(
    lines: list[str],
    title: str,
    statements: list[EvidenceBackedStatement],
) -> None:
    if not statements:
        return

    lines.extend(["", f"### {title}"])
    for statement in statements:
        citations = " ".join(f"[{rank}]" for rank in statement.evidence_ranks)
        lines.append(f"- {statement.text} {citations}")
