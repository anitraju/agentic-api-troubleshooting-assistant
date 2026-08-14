"""Inspect retrieval-ready chunks created from the knowledge base."""

from collections import Counter

from app.ingestion import chunk_documents, load_knowledge_base


def main() -> None:
    documents = load_knowledge_base()
    chunks = chunk_documents(documents)

    counts = Counter(chunk.source_type.value for chunk in chunks)

    print(f"Loaded {len(documents)} knowledge documents")
    print(f"Created {len(chunks)} retrieval chunks")

    for source_type, count in sorted(counts.items()):
        print(f"- {source_type}: {count}")

    print("\nSample chunk metadata:")
    for chunk in chunks[:10]:
        print(f"- {chunk.chunk_id} | {chunk.source} | {chunk.metadata}")


if __name__ == "__main__":
    main()
