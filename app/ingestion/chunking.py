"""Metadata-aware chunking for troubleshooting knowledge."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any

import yaml

from app.ingestion.models import KnowledgeChunk, KnowledgeDocument, SourceType

DEFAULT_MAX_CHARS = 1400
DEFAULT_OVERLAP_CHARS = 160

_HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options", "trace"}
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_API_OPERATION = re.compile(
    r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|TRACE)\s+"
    r"(`?/[A-Za-z0-9_./{}\-]+`?)",
    flags=re.IGNORECASE,
)
_STATUS_CODE = re.compile(r"(?<!\d)([1-5]\d{2})(?!\d)")


def chunk_documents(
    documents: Iterable[KnowledgeDocument],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[KnowledgeChunk]:
    """Chunk many normalized documents in deterministic input order."""
    _validate_chunk_settings(max_chars, overlap_chars)

    chunks: list[KnowledgeChunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )
    return chunks


def chunk_document(
    document: KnowledgeDocument,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[KnowledgeChunk]:
    """Create retrieval-ready chunks while preserving source-specific context."""
    _validate_chunk_settings(max_chars, overlap_chars)

    if document.source_type == SourceType.OPENAPI:
        drafts = _chunk_openapi_document(document, max_chars=max_chars, overlap_chars=overlap_chars)
    elif document.source_type in {SourceType.DOCUMENTATION, SourceType.RUNBOOK}:
        drafts = _chunk_markdown_document(
            document,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
    else:
        drafts = _chunk_plain_document(
            document,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

    total = len(drafts)
    chunks: list[KnowledgeChunk] = []

    for index, (content, metadata) in enumerate(drafts):
        enriched_metadata = {
            **document.metadata,
            **metadata,
            "chunk_index": index,
            "chunk_count": total,
        }
        enriched_metadata.update(_extract_api_signals(content, enriched_metadata))

        chunks.append(
            KnowledgeChunk(
                chunk_id=_build_chunk_id(document.source, index, content),
                content=content,
                source=document.source,
                source_type=document.source_type,
                metadata=enriched_metadata,
            )
        )

    return chunks


def _chunk_markdown_document(
    document: KnowledgeDocument,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[tuple[str, dict[str, Any]]]:
    sections = _split_markdown_sections(document.content)
    drafts: list[tuple[str, dict[str, Any]]] = []

    for heading, level, section_content in sections:
        prefix = f"{'#' * level} {heading}\n\n" if heading else ""
        available_chars = max_chars - len(prefix)

        if available_chars <= 0:
            available_chars = max_chars

        body = section_content.removeprefix(prefix).strip()
        pieces = _split_text(
            body if body else section_content.strip(),
            max_chars=available_chars,
            overlap_chars=min(overlap_chars, max(0, available_chars // 3)),
        )

        for piece in pieces:
            content = f"{prefix}{piece}".strip() if prefix else piece.strip()
            metadata: dict[str, Any] = {}
            if heading:
                metadata["section"] = heading
                metadata["heading_level"] = level
            drafts.append((content, metadata))

    return drafts


def _chunk_plain_document(
    document: KnowledgeDocument,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[tuple[str, dict[str, Any]]]:
    pieces = _split_text(
        document.content,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    return [(piece, {}) for piece in pieces]


def _chunk_openapi_document(
    document: KnowledgeDocument,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[tuple[str, dict[str, Any]]]:
    spec = yaml.safe_load(document.content)

    if not isinstance(spec, dict):
        raise ValueError(f"OpenAPI source must contain a YAML object: {document.source}")

    paths = spec.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError(f"OpenAPI 'paths' must be an object: {document.source}")

    service = _infer_service(document.content, document.metadata)
    drafts: list[tuple[str, dict[str, Any]]] = []

    for endpoint in sorted(paths):
        path_item = paths[endpoint]
        if not isinstance(path_item, dict):
            continue

        for method in _HTTP_METHODS:
            operation = path_item.get(method)
            if not isinstance(operation, dict):
                continue

            operation_text = _render_openapi_operation(
                spec=spec,
                endpoint=endpoint,
                method=method.upper(),
                operation=operation,
            )

            base_metadata: dict[str, Any] = {
                "section": f"{method.upper()} {endpoint}",
                "endpoint": endpoint,
                "endpoints": endpoint,
                "http_method": method.upper(),
                "http_methods": method.upper(),
            }

            operation_id = operation.get("operationId")
            if operation_id:
                base_metadata["operation_id"] = str(operation_id)

            tags = operation.get("tags")
            if isinstance(tags, list) and tags:
                base_metadata["tags"] = ", ".join(str(tag) for tag in tags)

            responses = operation.get("responses")
            status_codes = _response_status_codes(responses)
            if status_codes:
                base_metadata["status_codes"] = ",".join(status_codes)
                if len(status_codes) == 1:
                    base_metadata["status_code"] = int(status_codes[0])

            if service:
                base_metadata["service"] = service

            pieces = _split_text(
                operation_text,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )

            for piece in pieces:
                drafts.append((piece, base_metadata))

    if drafts:
        return drafts

    return _chunk_plain_document(
        document,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )


def _render_openapi_operation(
    *,
    spec: dict[str, Any],
    endpoint: str,
    method: str,
    operation: dict[str, Any],
) -> str:
    lines = [f"OpenAPI operation: {method} {endpoint}"]

    summary = operation.get("summary")
    if summary:
        lines.append(f"Summary: {summary}")

    description = operation.get("description")
    if description:
        lines.append(f"Description: {description}")

    operation_id = operation.get("operationId")
    if operation_id:
        lines.append(f"Operation ID: {operation_id}")

    lines.append("Operation definition:")
    lines.append(yaml.safe_dump(operation, sort_keys=True).strip())

    referenced_components = _collect_referenced_components(operation, spec)
    if referenced_components:
        lines.append("Referenced OpenAPI components:")
        for ref, value in sorted(referenced_components.items()):
            lines.append(f"{ref}:")
            lines.append(yaml.safe_dump(value, sort_keys=True).strip())

    return "\n".join(lines).strip()


def _collect_referenced_components(
    value: Any,
    spec: dict[str, Any],
) -> dict[str, Any]:
    found: dict[str, Any] = {}
    pending = list(_find_local_refs(value))

    while pending:
        ref = pending.pop()
        if ref in found:
            continue

        resolved = _resolve_local_ref(spec, ref)
        if resolved is None:
            continue

        found[ref] = resolved
        pending.extend(_find_local_refs(resolved))

    return found


def _find_local_refs(value: Any) -> set[str]:
    refs: set[str] = set()

    if isinstance(value, dict):
        ref = value.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            refs.add(ref)

        for child in value.values():
            refs.update(_find_local_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.update(_find_local_refs(child))

    return refs


def _resolve_local_ref(spec: dict[str, Any], ref: str) -> Any | None:
    if not ref.startswith("#/"):
        return None

    current: Any = spec
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]

    return current


def _split_markdown_sections(content: str) -> list[tuple[str | None, int, str]]:
    lines = content.splitlines()
    sections: list[tuple[str | None, int, str]] = []

    current_heading: str | None = None
    current_level = 0
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((current_heading, current_level, text))
        current_lines = []

    for line in lines:
        match = _MARKDOWN_HEADING.match(line)
        if match:
            flush()
            current_level = len(match.group(1))
            current_heading = match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    flush()

    if not sections and content.strip():
        return [(None, 0, content.strip())]

    return sections


def _split_text(text: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []

    if len(normalized) <= max_chars:
        return [normalized]

    pieces: list[str] = []
    start = 0
    length = len(normalized)

    while start < length:
        hard_end = min(start + max_chars, length)
        end = hard_end

        if hard_end < length:
            search_start = start + max_chars // 2
            paragraph_break = normalized.rfind("\n\n", search_start, hard_end)
            whitespace_break = normalized.rfind(" ", search_start, hard_end)

            if paragraph_break > start:
                end = paragraph_break
            elif whitespace_break > start:
                end = whitespace_break

        piece = normalized[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= length:
            break

        next_start = max(end - overlap_chars, start + 1)
        while next_start < end and not normalized[next_start].isspace():
            next_start += 1
        while next_start < length and normalized[next_start].isspace():
            next_start += 1

        if next_start <= start:
            next_start = end

        start = next_start

    return pieces


def _extract_api_signals(
    content: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    signals: dict[str, Any] = {}

    operations = {
        (method.upper(), endpoint.strip("`"))
        for method, endpoint in _API_OPERATION.findall(content)
    }
    methods = sorted({method for method, _ in operations})
    endpoints = sorted({endpoint for _, endpoint in operations})

    existing_endpoint = metadata.get("endpoint")
    existing_method = metadata.get("http_method")

    if existing_endpoint:
        endpoints = sorted({*endpoints, str(existing_endpoint)})
    if existing_method:
        methods = sorted({*methods, str(existing_method).upper()})

    if endpoints:
        signals["endpoints"] = ",".join(endpoints)
        if len(endpoints) == 1:
            signals["endpoint"] = endpoints[0]

    if methods:
        signals["http_methods"] = ",".join(methods)
        if len(methods) == 1:
            signals["http_method"] = methods[0]

    status_codes = sorted(set(_STATUS_CODE.findall(content)))
    existing_codes = metadata.get("status_codes")
    if isinstance(existing_codes, str):
        status_codes = sorted(
            {*status_codes, *(code for code in existing_codes.split(",") if code)}
        )

    if status_codes:
        signals["status_codes"] = ",".join(status_codes)
        if len(status_codes) == 1:
            signals["status_code"] = int(status_codes[0])

    service = metadata.get("service") or _infer_service(content, metadata)
    if service:
        signals["service"] = str(service)

    return signals


def _infer_service(content: str, metadata: dict[str, Any]) -> str | None:
    existing = metadata.get("service")
    if existing:
        return str(existing)

    title = str(metadata.get("title", ""))
    filename = str(metadata.get("filename", ""))
    searchable = f"{content}\n{title}\n{filename}".lower()

    if "order-service" in searchable or "order service" in searchable:
        return "order-service"

    return None


def _response_status_codes(responses: Any) -> list[str]:
    if not isinstance(responses, dict):
        return []

    return sorted(
        str(code)
        for code in responses
        if re.fullmatch(r"[1-5]\d{2}", str(code))
    )


def _build_chunk_id(source: str, index: int, content: str) -> str:
    payload = f"{source}|{index}|{content}".encode()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"chunk-{digest}"


def _validate_chunk_settings(max_chars: int, overlap_chars: int) -> None:
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    if overlap_chars < 0:
        raise ValueError("overlap_chars cannot be negative")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")
