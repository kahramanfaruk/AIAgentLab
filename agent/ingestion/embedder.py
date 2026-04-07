from __future__ import annotations

import os
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from agent.ingestion.chunker import Chunk


@dataclass(slots=True)
class EmbeddedChunk:
    chunk_id: str
    content: str
    source: str
    embedding: list[float]
    metadata: dict


class LocalEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        embedded_chunks: list[EmbeddedChunk] = []
        for chunk, vector in zip(chunks, vectors):
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    source=chunk.source,
                    embedding=vector.tolist(),
                    metadata=chunk.metadata,
                )
            )

        return embedded_chunks

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.tolist()


if __name__ == "__main__":
    from agent.ingestion.chunker import chunk_documents
    from agent.ingestion.loader import load_documents

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    docs = load_documents()
    chunks = chunk_documents(docs)
    embedder = LocalEmbedder(device="cpu")
    embedded = embedder.embed_chunks(chunks[:5])

    print(f"Embedded {len(embedded)} chunks.")
    print(f"Embedding dimension: {len(embedded[0].embedding) if embedded else 0}")