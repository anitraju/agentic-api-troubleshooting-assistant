"""Deterministic evaluators for troubleshooting-agent outputs."""

from __future__ import annotations

from collections.abc import Iterable
from statistics import mean
from typing import Any

from app.evaluation.models import EvaluationCase, EvaluationCaseResult
from app.rag.answers import EvidenceBackedStatement


def score_agent_state(
    case: EvaluationCase,
    state: dict[str, Any],
) -> EvaluationCaseResult:
    """Score one final LangGraph state against a curated evaluation case."""
    classification = state.get("classification")
    answer = state.get("answer")
    evidence = state.get("evidence", [])

    if classification is None:
        return _failed_result(case, "agent state is missing classification")
    if answer is None:
        return _failed_result(case, "agent state is missing final answer")

    scores: dict[str, float] = {}
    failures: list[str] = []

    category_score = float(classification.category == case.expected_category)
    scores["category"] = category_score
    if category_score == 0:
        failures.append(
            "category mismatch: "
            f"expected {case.expected_category.value}, "
            f"got {classification.category.value}"
        )

    _score_expected_field(
        scores,
        failures,
        "status_code",
        case.expected_status_code,
        classification.status_code,
    )
    _score_expected_field(
        scores,
        failures,
        "http_method",
        case.expected_http_method,
        classification.http_method,
    )
    _score_expected_field(
        scores,
        failures,
        "endpoint",
        case.expected_endpoint,
        classification.endpoint,
    )
    _score_expected_field(
        scores,
        failures,
        "service",
        case.expected_service,
        classification.service,
    )

    sources = [item.source for item in evidence]
    if case.expected_source_hints:
        retrieval_score = _source_hint_score(
            case.expected_source_hints,
            sources,
        )
        scores["retrieval_source_hit"] = retrieval_score
        if retrieval_score == 0:
            failures.append(
                "retrieval missed expected source hints: "
                + ", ".join(case.expected_source_hints)
            )

    rendered_answer = answer.render_markdown()
    if case.expected_answer_terms:
        answer_term_score = _term_recall(
            case.expected_answer_terms,
            rendered_answer,
        )
        scores["answer_term_recall"] = answer_term_score
        if answer_term_score == 0:
            failures.append(
                "answer contained none of the expected terms: "
                + ", ".join(case.expected_answer_terms)
            )

    grounding_score = float(
        _is_grounded(
            answer.analysis.likely_causes,
            answer.analysis.diagnostic_steps,
            answer.analysis.remediation_steps,
            valid_ranks={item.rank for item in evidence},
        )
    )
    scores["grounding"] = grounding_score
    if grounding_score == 0:
        failures.append("answer contains evidence citations outside retrieved ranks")

    overall_score = mean(scores.values()) if scores else 0.0

    # Category accuracy and citation grounding are hard requirements. Other dimensions are
    # combined into the case-level score so retrieval/generation quality can improve gradually.
    passed = (
        category_score == 1.0
        and grounding_score == 1.0
        and overall_score >= case.minimum_score
    )

    if overall_score < case.minimum_score:
        failures.append(
            f"overall score {overall_score:.3f} is below "
            f"minimum {case.minimum_score:.3f}"
        )

    return EvaluationCaseResult(
        case_id=case.case_id,
        query=case.query,
        passed=passed,
        overall_score=overall_score,
        scores=scores,
        failures=failures,
        actual_category=classification.category,
        actual_status_code=classification.status_code,
        actual_http_method=classification.http_method,
        actual_endpoint=classification.endpoint,
        actual_service=classification.service,
        retrieved_sources=sources,
        answer_confidence=answer.analysis.confidence,
        generation_attempts=state.get("generation_attempts", 0),
    )


def _score_expected_field(
    scores: dict[str, float],
    failures: list[str],
    name: str,
    expected: Any,
    actual: Any,
) -> None:
    if expected is None:
        return

    score = float(actual == expected)
    scores[name] = score
    if score == 0:
        failures.append(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _source_hint_score(
    expected_hints: list[str],
    sources: list[str],
) -> float:
    """Return 1 when any expected source hint is present in any retrieved source."""
    normalized_sources = [source.casefold() for source in sources]

    for hint in expected_hints:
        normalized_hint = hint.casefold()
        if any(normalized_hint in source for source in normalized_sources):
            return 1.0

    return 0.0


def _term_recall(
    expected_terms: list[str],
    rendered_answer: str,
) -> float:
    normalized_answer = rendered_answer.casefold()
    matched = sum(
        1
        for term in expected_terms
        if term.casefold() in normalized_answer
    )
    return matched / len(expected_terms)


def _is_grounded(
    *statement_groups: Iterable[EvidenceBackedStatement],
    valid_ranks: set[int],
) -> bool:
    for statements in statement_groups:
        for statement in statements:
            if not set(statement.evidence_ranks).issubset(valid_ranks):
                return False
    return True


def _failed_result(
    case: EvaluationCase,
    error: str,
) -> EvaluationCaseResult:
    return EvaluationCaseResult(
        case_id=case.case_id,
        query=case.query,
        passed=False,
        overall_score=0.0,
        failures=[error],
        error=error,
    )
