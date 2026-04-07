from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.ingestion.embedder import LocalEmbedder
from agent.retrieval.vector_store import ChromaVectorStore


@dataclass(slots=True)
class RetrievalResult:
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    distance: float | None


class Retriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedder: LocalEmbedder,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_embedding = self.embedder.embed_query(query)

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        retrieval_results: list[RetrievalResult] = []
        for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            retrieval_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=metadata or {},
                    distance=distance,
                )
            )

        return retrieval_results


if __name__ == "__main__":
    store = ChromaVectorStore()
    embedder = LocalEmbedder(device="cpu")
    retriever = Retriever(vector_store=store, embedder=embedder)

    query = "What do the authors claim about gating after scaled dot-product attention?"
    results = retriever.retrieve(query=query, top_k=3)

    print(f"Retrieved {len(results)} results.")
    for idx, result in enumerate(results, start=1):
        print("-" * 80)
        print(f"Result {idx}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Distance: {result.distance}")
        print(f"Metadata: {result.metadata}")
        print(result.content[:700])