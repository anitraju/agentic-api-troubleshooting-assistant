"""RAG infrastructure for indexing, retrieval, and grounded answer generation."""

from app.rag.answers import (
    Confidence,
    EvidenceBackedStatement,
    GeneratedTroubleshootingAnalysis,
    TroubleshootingAnswer,
)
from app.rag.embeddings import Embedder, SentenceTransformerEmbedder
from app.rag.generation import OpenAITroubleshootingGenerator, TroubleshootingGenerator
from app.rag.pipeline import GroundedRAGPipeline, UngroundedGenerationError
from app.rag.retrieval import RetrievalFilters, RetrievalResult, SemanticRetriever
from app.rag.vector_store import ChromaVectorStore, VectorSearcher, VectorSearchHit

__all__ = [
    "ChromaVectorStore",
    "Confidence",
    "Embedder",
    "EvidenceBackedStatement",
    "GeneratedTroubleshootingAnalysis",
    "GroundedRAGPipeline",
    "OpenAITroubleshootingGenerator",
    "RetrievalFilters",
    "RetrievalResult",
    "SemanticRetriever",
    "SentenceTransformerEmbedder",
    "TroubleshootingAnswer",
    "TroubleshootingGenerator",
    "UngroundedGenerationError",
    "VectorSearcher",
    "VectorSearchHit",
]
