"""LLM generation layer for evidence-grounded API troubleshooting."""

from __future__ import annotations

import ssl
from collections.abc import Sequence
from typing import Protocol

import httpx
import truststore

from app.rag.answers import GeneratedTroubleshootingAnalysis
from app.rag.retrieval import RetrievalResult

_SYSTEM_PROMPT = """You are an API troubleshooting assistant.

Use ONLY the supplied retrieved evidence. Do not use outside knowledge to assert facts about
this system. Every likely cause, diagnostic step, and remediation step must include one or
more evidence_ranks that point to the evidence items supporting that statement.

Rules:
- Do not invent endpoints, status codes, configuration values, logs, dependencies, or fixes.
- Prefer precise diagnostic checks before destructive or broad remediation.
- If evidence is incomplete or conflicting, say so in limitations and lower confidence.
- HIGH confidence requires strong, directly relevant evidence.
- MEDIUM confidence means the evidence is plausible but incomplete.
- LOW confidence means the retrieved evidence is weak, indirect, or insufficient.
- Keep the summary concise and actionable.
"""


class TroubleshootingGenerator(Protocol):
    """Generation contract consumed by the baseline RAG pipeline."""

    def generate(
        self,
        query: str,
        evidence: Sequence[RetrievalResult],
    ) -> GeneratedTroubleshootingAnalysis:
        """Generate a structured troubleshooting analysis grounded in retrieved evidence."""


class OpenAITroubleshootingGenerator:
    """LangChain ChatOpenAI implementation using provider-supported structured output."""

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

    def generate(
        self,
        query: str,
        evidence: Sequence[RetrievalResult],
    ) -> GeneratedTroubleshootingAnalysis:
        """Generate a schema-validated analysis from the supplied evidence only."""
        if not query.strip():
            raise ValueError("query cannot be empty")
        if not evidence:
            raise ValueError("evidence cannot be empty")

        messages = [
            ("system", _SYSTEM_PROMPT),
            (
                "human",
                "Troubleshooting query:\n"
                f"{query.strip()}\n\n"
                "Retrieved evidence:\n"
                f"{format_evidence_context(evidence)}",
            ),
        ]
        result = self._get_structured_model().invoke(messages)

        if isinstance(result, GeneratedTroubleshootingAnalysis):
            return result
        return GeneratedTroubleshootingAnalysis.model_validate(result)

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
            self._structured_model = model.with_structured_output(
                GeneratedTroubleshootingAnalysis
            )

        return self._structured_model


def format_evidence_context(evidence: Sequence[RetrievalResult]) -> str:
    """Serialize ranked retrieval results into a citation-stable LLM context."""
    blocks: list[str] = []
    for item in evidence:
        metadata = ", ".join(
            f"{key}={value}"
            for key, value in sorted(item.metadata.items())
            if value not in (None, "")
        )
        metadata_line = metadata or "none"
        blocks.append(
            "\n".join(
                [
                    f"=== EVIDENCE [{item.rank}] ===",
                    f"Source: {item.source}",
                    f"Source type: {item.source_type.value}",
                    f"Metadata: {metadata_line}",
                    "Content:",
                    item.content,
                ]
            )
        )

    return "\n\n".join(blocks)
