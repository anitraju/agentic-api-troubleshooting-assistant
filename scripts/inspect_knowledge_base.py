"""Inspect the normalized knowledge-base documents produced by the ingestion layer."""

from collections import Counter

from app.ingestion import load_knowledge_base


def main() -> None:
    documents = load_knowledge_base()

    counts = Counter(document.source_type.value for document in documents)

    print(f"Loaded {len(documents)} knowledge documents")
    for source_type, count in sorted(counts.items()):
        print(f"- {source_type}: {count}")

    print("\nSources:")
    for document in documents:
        print(f"- [{document.source_type.value}] {document.source}")


if __name__ == "__main__":
    main()
