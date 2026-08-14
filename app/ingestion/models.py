"""Models shared by the knowledge-base ingestion pipeline."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    """Supported knowledge-base source categories."""

    DOCUMENTATION = "documentation"
    RUNBOOK = "runbook"
    INCIDENT = "incident"
    OPENAPI = "openapi"


class KnowledgeDocument(BaseModel):
    """Normalized representation of a knowledge-base source."""

    content: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_type: SourceType
    metadata: dict[str, Any] = Field(default_factory=dict)
