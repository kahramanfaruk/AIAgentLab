from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from agent.ingestion.loader import Document


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    content: str
    source: str
    metadata: dict


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for doc in documents:
            doc_chunks = self._split_text(doc.content)
            for idx, piece in enumerate(doc_chunks):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}-chunk-{idx}",
                        content=piece,
                        source=doc.source,
                        metadata={
                            **doc.metadata,
                            "parent_doc_id": doc.doc_id,
                            "chunk_index": idx,
                            "chunk_size": len(piece),
                        },
                    )
                )

        return chunks

    def _split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            candidate = text[start:end]

            if end < text_length:
                split_at = self._find_split_point(candidate)
                if split_at > 0:
                    end = start + split_at
                    candidate = text[start:end]

            candidate = candidate.strip()
            if candidate:
                chunks.append(candidate)

            if end >= text_length:
                break

            start = max(end - self.chunk_overlap, start + 1)

        return chunks

    def _find_split_point(self, text: str) -> int:
        for sep in self.separators:
            pos = text.rfind(sep)
            if pos > self.chunk_size * 0.5:
                return pos + len(sep)
        return len(text)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.split_documents(documents)


if __name__ == "__main__":
    from agent.ingestion.loader import load_documents

    docs = load_documents()
    chunks = chunk_documents(docs)

    print(f"Loaded {len(docs)} documents/pages.")
    print(f"Created {len(chunks)} chunks.")

    for chunk in chunks[:3]:
        print("-" * 80)
        print(chunk.chunk_id)
        print(chunk.metadata)
        print(chunk.content[:500])