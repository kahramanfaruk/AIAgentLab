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
    score: float


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

        raw_k = max(top_k * 3, 12)
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=raw_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        retrieval_results: list[RetrievalResult] = []
        for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            score = self._score_result(query, metadata, content, distance)

            retrieval_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=metadata,
                    distance=distance,
                    score=score,
                )
            )

        retrieval_results.sort(key=lambda x: x.score, reverse=True)
        return retrieval_results[:top_k]

    def _score_result(
        self,
        query: str,
        metadata: dict[str, Any],
        content: str,
        distance: float | None,
    ) -> float:
        query_lower = query.lower()
        content_lower = content.lower()

        base_score = -(distance if distance is not None else 999.0)

        page_number = metadata.get("page_number")
        try:
            page_number = int(page_number)
        except (TypeError, ValueError):
            page_number = None

        bonus = 0.0

        purpose_keywords = ["purpose", "objective", "goal", "main point", "contribution", "abstract"]
        if any(keyword in query_lower for keyword in purpose_keywords):
            if page_number == 1:
                bonus += 2.5
            if page_number == 2:
                bonus += 1.0
            if "abstract" in content_lower:
                bonus += 2.0
            if "introduction" in content_lower:
                bonus += 1.2
            if "in this work" in content_lower:
                bonus += 1.5
            if "our central finding" in content_lower:
                bonus += 1.5

        method_keywords = ["how", "method", "approach", "mechanism"]
        if any(keyword in query_lower for keyword in method_keywords):
            if "we study" in content_lower or "we investigate" in content_lower:
                bonus += 1.0

        finding_keywords = ["finding", "result", "improve", "benefit", "claim"]
        if any(keyword in query_lower for keyword in finding_keywords):
            if "our central finding" in content_lower:
                bonus += 2.0
            if "improves performance" in content_lower:
                bonus += 1.2

        return base_score + bonus


if __name__ == "__main__":
    store = ChromaVectorStore()
    embedder = LocalEmbedder(device="cpu")
    retriever = Retriever(vector_store=store, embedder=embedder)

    query = "What is the main objective of this paper?"
    results = retriever.retrieve(query=query, top_k=5)

    print(f"Retrieved {len(results)} results.")
    for idx, result in enumerate(results, start=1):
        print("-" * 80)
        print(f"Result {idx}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Distance: {result.distance}")
        print(f"Score: {result.score}")
        print(f"Metadata: {result.metadata}")
        print(result.content[:700])