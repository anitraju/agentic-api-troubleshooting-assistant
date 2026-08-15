"""Scenario loading and evaluation execution."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from app.evaluation.models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationReport,
)
from app.evaluation.scoring import score_agent_state


class SupportsAgentRun(Protocol):
    """Small contract required by the evaluation runner."""

    def run_with_state(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: Any = None,
    ) -> dict[str, Any]:
        """Run one troubleshooting query and expose the final graph state."""


class EvaluationRunner:
    """Execute a curated dataset against a troubleshooting agent."""

    def __init__(self, agent: SupportsAgentRun) -> None:
        self.agent = agent

    def run(
        self,
        cases: Sequence[EvaluationCase],
    ) -> EvaluationReport:
        """Run each case independently and return an aggregate report."""
        results = [self.run_case(case) for case in cases]
        return EvaluationReport(results=results)

    def run_case(
        self,
        case: EvaluationCase,
    ) -> EvaluationCaseResult:
        """Run and score one evaluation case without aborting the full suite on errors."""
        try:
            state = self.agent.run_with_state(
                case.query,
                top_k=case.top_k,
            )
        except Exception as exc:  # noqa: BLE001 - eval reports should capture case failures
            message = f"{type(exc).__name__}: {exc}"
            return EvaluationCaseResult(
                case_id=case.case_id,
                query=case.query,
                passed=False,
                overall_score=0.0,
                failures=[message],
                error=message,
            )

        return score_agent_state(case, state)


def load_evaluation_cases(path: Path | str) -> list[EvaluationCase]:
    """Load evaluation cases from a JSON object containing a `cases` array."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("evaluation dataset must be a JSON object")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("evaluation dataset must contain a 'cases' array")

    cases = [
        EvaluationCase.model_validate(item)
        for item in raw_cases
    ]

    if not cases:
        raise ValueError("evaluation dataset cannot be empty")

    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation case IDs must be unique")

    return cases
