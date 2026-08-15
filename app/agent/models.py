"""Structured models used by the LangGraph troubleshooting workflow."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class IssueCategory(StrEnum):
    """High-level API failure category used to guide troubleshooting."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    SERVICE_AVAILABILITY = "service_availability"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"


class IssueClassification(BaseModel):
    """Structured interpretation of the user's reported API failure."""

    category: IssueCategory = IssueCategory.UNKNOWN
    summary: str = Field(min_length=1)
    service: str | None = None
    endpoint: str | None = None
    http_method: str | None = None
    status_code: int | None = Field(default=None, ge=100, le=599)
    signals: list[str] = Field(default_factory=list)

    @field_validator("service", "endpoint")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("http_method")
    @classmethod
    def normalize_http_method(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = value.strip().upper()
        return normalized or None

    @field_validator("signals")
    @classmethod
    def deduplicate_signals(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        return list(dict.fromkeys(normalized))
