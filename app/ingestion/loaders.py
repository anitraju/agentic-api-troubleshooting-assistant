"""Load raw troubleshooting knowledge into normalized documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from app.ingestion.models import KnowledgeDocument, SourceType


def load_knowledge_base(data_dir: Path | str = "data") -> list[KnowledgeDocument]:
    """Load all RAG knowledge sources from the configured data directory.

    Structured request logs are intentionally excluded. They are operational evidence and
    will be queried later through a dedicated log-search tool rather than embedded into the
    long-lived knowledge base.
    """
    root = Path(data_dir)

    documents: list[KnowledgeDocument] = []
    documents.extend(
        load_markdown_directory(
            root / "docs",
            source_type=SourceType.DOCUMENTATION,
            root=root,
        )
    )
    documents.extend(
        load_markdown_directory(
            root / "runbooks",
            source_type=SourceType.RUNBOOK,
            root=root,
        )
    )
    documents.extend(load_incidents_file(root / "incidents" / "known_incidents.json", root=root))
    documents.extend(load_openapi_file(root / "openapi" / "order-service.yaml", root=root))

    return documents


def load_markdown_directory(
    directory: Path | str,
    *,
    source_type: SourceType,
    root: Path | None = None,
) -> list[KnowledgeDocument]:
    """Load every Markdown file in a directory in deterministic filename order."""
    directory_path = Path(directory)

    if not directory_path.exists():
        return []

    return [
        load_markdown_file(path, source_type=source_type, root=root)
        for path in sorted(directory_path.glob("*.md"))
        if path.is_file()
    ]


def load_markdown_file(
    path: Path | str,
    *,
    source_type: SourceType,
    root: Path | None = None,
) -> KnowledgeDocument:
    """Load one Markdown document and extract basic metadata."""
    file_path = Path(path)
    content = _read_text(file_path)
    title = _extract_markdown_title(content)

    metadata: dict[str, Any] = {
        "filename": file_path.name,
        "format": "markdown",
    }
    if title:
        metadata["title"] = title

    return KnowledgeDocument(
        content=content,
        source=_source_name(file_path, root),
        source_type=source_type,
        metadata=metadata,
    )


def load_incidents_file(
    path: Path | str,
    *,
    root: Path | None = None,
) -> list[KnowledgeDocument]:
    """Load each historical incident as an independent knowledge document."""
    file_path = Path(path)

    if not file_path.exists():
        return []

    payload = json.loads(_read_text(file_path))
    service = payload.get("service")
    incidents = payload.get("incidents", [])

    if not isinstance(incidents, list):
        raise ValueError(f"'incidents' must be a list in {file_path}")

    documents: list[KnowledgeDocument] = []

    for incident in incidents:
        if not isinstance(incident, dict):
            raise ValueError(f"Each incident must be an object in {file_path}")

        incident_id = _required_string(incident, "incident_id", file_path)
        title = _required_string(incident, "title", file_path)

        content = _incident_to_text(incident)

        metadata: dict[str, Any] = {
            "filename": file_path.name,
            "format": "json",
            "incident_id": incident_id,
            "title": title,
        }

        for key in ("date", "status", "affected_endpoint"):
            value = incident.get(key)
            if value is not None:
                metadata[key] = value

        if service is not None:
            metadata["service"] = service

        symptoms = incident.get("symptoms")
        if isinstance(symptoms, list):
            metadata["symptoms"] = symptoms

        signatures = incident.get("signatures")
        if isinstance(signatures, list):
            metadata["signatures"] = signatures

        documents.append(
            KnowledgeDocument(
                content=content,
                source=f"{_source_name(file_path, root)}#{incident_id}",
                source_type=SourceType.INCIDENT,
                metadata=metadata,
            )
        )

    return documents


def load_openapi_file(
    path: Path | str,
    *,
    root: Path | None = None,
) -> list[KnowledgeDocument]:
    """Load an OpenAPI specification as a normalized knowledge document."""
    file_path = Path(path)

    if not file_path.exists():
        return []

    content = _read_text(file_path)
    payload = yaml.safe_load(content)

    if not isinstance(payload, dict):
        raise ValueError(f"OpenAPI file must contain a YAML object: {file_path}")

    info = payload.get("info", {})
    title = info.get("title") if isinstance(info, dict) else None
    version = info.get("version") if isinstance(info, dict) else None
    openapi_version = payload.get("openapi")
    paths = payload.get("paths", {})

    metadata: dict[str, Any] = {
        "filename": file_path.name,
        "format": "yaml",
    }

    if title is not None:
        metadata["title"] = title
    if version is not None:
        metadata["api_version"] = str(version)
    if openapi_version is not None:
        metadata["openapi_version"] = str(openapi_version)
    if isinstance(paths, dict):
        metadata["path_count"] = len(paths)
        metadata["paths"] = sorted(paths.keys())

    return [
        KnowledgeDocument(
            content=content,
            source=_source_name(file_path, root),
            source_type=SourceType.OPENAPI,
            metadata=metadata,
        )
    ]


def _incident_to_text(incident: dict[str, Any]) -> str:
    """Convert a structured incident record into deterministic retrieval-friendly text."""
    lines = [
        f"Incident ID: {incident.get('incident_id', '')}",
        f"Title: {incident.get('title', '')}",
        f"Date: {incident.get('date', '')}",
        f"Status: {incident.get('status', '')}",
        f"Affected endpoint: {incident.get('affected_endpoint', '')}",
        f"Symptoms: {_join_values(incident.get('symptoms'))}",
        f"Root cause: {incident.get('root_cause', '')}",
        f"Resolution: {incident.get('resolution', '')}",
        f"Signatures: {_join_values(incident.get('signatures'))}",
    ]
    return "\n".join(lines).strip()


def _join_values(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _required_string(payload: dict[str, Any], key: str, file_path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid '{key}' in {file_path}")
    return value


def _extract_markdown_title(content: str) -> str | None:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _source_name(path: Path, root: Path | None) -> str:
    if root is None:
        return path.as_posix()

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
