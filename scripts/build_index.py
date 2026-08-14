"""Build the persistent semantic-search index from the troubleshooting knowledge base."""

from app.config import get_settings
from app.ingestion import chunk_documents, load_knowledge_base
from app.rag import ChromaVectorStore, SentenceTransformerEmbedder


def main() -> None:
    settings = get_settings()

    documents = load_knowledge_base()
    chunks = chunk_documents(documents)

    embedder = SentenceTransformerEmbedder(
        model_name=settings.embedding_model_name,
        batch_size=settings.embedding_batch_size,
    )
    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )

    vector_store.reset()

    indexed = vector_store.upsert_chunks(
        chunks,
        embedder,
        batch_size=settings.vector_upsert_batch_size,
    )

    print(f"Loaded {len(documents)} knowledge documents")
    print(f"Created {len(chunks)} retrieval chunks")
    print(f"Indexed {indexed} chunks")
    print(f"Collection count: {vector_store.count()}")
    print(f"Collection: {settings.chroma_collection_name}")
    print(f"Persistence directory: {settings.chroma_persist_dir}")
    print(f"Embedding model: {settings.embedding_model_name}")


if __name__ == "__main__":
    main()
