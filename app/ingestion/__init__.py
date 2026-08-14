"""Knowledge-base ingestion utilities."""

from app.ingestion.loaders import load_knowledge_base
from app.ingestion.models import KnowledgeDocument, SourceType

__all__ = ["KnowledgeDocument", "SourceType", "load_knowledge_base"]
