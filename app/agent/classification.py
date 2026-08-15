"""Intent classification and retrieval-filter preparation for the agent."""

from __future__ import annotations

import ssl
from typing import Protocol

import httpx
import truststore

from app.agent.models import IssueClassification
from app.rag.retrieval import RetrievalFilters

_CLASSIFICATION_SYSTEM_PROMPT = """You classify API troubleshooting requests.

Return a structured classification for the user's report.

Rules:
- Categorize the primary issue as authentication, authorization, validation, not_found,
  rate_limit, service_availability, dependency, or unknown.
- Extract status_code only when it is explicitly present in the user query.
- Extract endpoint and HTTP method only when explicitly present.
- Extract service only when the user clearly names a service.
- Never invent an endpoint, service, HTTP method, status code, log value, or dependency.
- The summary must describe only what the user reported.
- signals should contain short explicit clues from the query, such as "401", "POST",
  "/api/v1/orders", "timeout", or "connection refused".
"""


class IssueClassifier(Protocol):
    """Classification contract consumed by the LangGraph workflow."""

    def classify(self, query: str) -> IssueClassification:
        """Interpret one troubleshooting query."""


class OpenAIIssueClassifier:
    """Structured OpenAI implementation of the issue classifier."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if not api_key.strip():
            raise ValueError("api_key cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        self.model_name = model_name
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._structured_model = None

    def classify(self, query: str) -> IssueClassification:
        """Classify a query without inventing missing API details."""
        if not query.strip():
            raise ValueError("query cannot be empty")

        result = self._get_structured_model().invoke(
            [
                ("system", _CLASSIFICATION_SYSTEM_PROMPT),
                ("human", query.strip()),
            ]
        )

        if isinstance(result, IssueClassification):
            return result
        return IssueClassification.model_validate(result)

    def _get_structured_model(self):
        if self._structured_model is None:
            from langchain_openai import ChatOpenAI

            ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            http_client = httpx.Client(
                verify=ssl_context,
                timeout=self.timeout_seconds,
                follow_redirects=True,
            )
            model = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
                http_client=http_client,
            )
            self._structured_model = model.with_structured_output(IssueClassification)

        return self._structured_model


def merge_retrieval_filters(
    classification: IssueClassification,
    requested: RetrievalFilters | None,
) -> RetrievalFilters:
    """Merge explicit caller filters with facts extracted by classification.

    Explicit caller-provided filters win. The classifier only fills fields that were not
    supplied by the caller.
    """
    requested = requested or RetrievalFilters()

    return RetrievalFilters(
        source_types=requested.source_types,
        service=requested.service or classification.service,
        endpoint=requested.endpoint or classification.endpoint,
        http_method=requested.http_method or classification.http_method,
        status_code=(
            requested.status_code
            if requested.status_code is not None
            else classification.status_code
        ),
        metadata=dict(requested.metadata),
    )
