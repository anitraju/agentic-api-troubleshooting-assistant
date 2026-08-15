"""Pydantic models for repeatable agent evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.agent.models import IssueCategory
from app.rag.answers import Confidence


class EvaluationCase(BaseModel):
    """One curated troubleshooting scenario with reference expectations."""

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_category: IssueCategory
    expected_status_code: int | None = Field(default=None, ge=100, le=599)
    expected_http_method: str | None = None
    expected_endpoint: str | None = None
    expected_service: str | None = None
    expected_source_hints: list[str] = Field(default_factory=list)
    expected_answer_terms: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)
    minimum_score: float = Field(default=0.75, ge=0.0, le=1.0)

    @field_validator(
        "expected_http_method",
        "expected_endpoint",
        "expected_service",
    )
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = value.strip()
        return normalized or None

    @field_validator("expected_http_method")
    @classmethod
    def normalize_http_method(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.upper()

    @field_validator("expected_source_hints", "expected_answer_terms")
    @classmethod
    def normalize_reference_lists(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        return list(dict.fromkeys(normalized))


class EvaluationCaseResult(BaseModel):
    """Measured result for one evaluation case."""

    case_id: str
    query: str
    passed: bool
    overall_score: float = Field(ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    actual_category: IssueCategory | None = None
    actual_status_code: int | None = None
    actual_http_method: str | None = None
    actual_endpoint: str | None = None
    actual_service: str | None = None
    retrieved_sources: list[str] = Field(default_factory=list)
    answer_confidence: Confidence | None = None
    generation_attempts: int = Field(default=0, ge=0)
    error: str | None = None


class EvaluationReport(BaseModel):
    """Aggregate evaluation report for a complete scenario run."""

    results: list[EvaluationCaseResult] = Field(default_factory=list)

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def passed_cases(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.passed_cases / self.total_cases

    @property
    def average_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.overall_score for result in self.results) / self.total_cases

    def render_text(self) -> str:
        """Render a compact terminal-friendly evaluation report."""
        lines = [
            "## Agent evaluation",
            "",
            f"Cases: {self.total_cases}",
            f"Passed: {self.passed_cases}",
            f"Pass rate: {self.pass_rate:.1%}",
            f"Average score: {self.average_score:.3f}",
            "",
        ]

        for result in self.results:
            marker = "PASS" if result.passed else "FAIL"
            lines.append(
                f"[{marker}] {result.case_id} "
                f"score={result.overall_score:.3f}"
            )

            if result.failures:
                for failure in result.failures:
                    lines.append(f"  - {failure}")

            if result.error:
                lines.append(f"  - error: {result.error}")

        return "\n".join(lines)
