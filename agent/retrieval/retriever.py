"""Backend-agnostic retrieval over a vector store.

The retriever over-fetches candidates from any :class:`VectorStore` backend and
applies a lightweight heuristic re-ranking tuned for research-paper questions
(purpose, method, and findings). It depends only on the vector-store and
embedder protocols, so it works identically with Chroma or OpenSearch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.ingestion.base import Embedder
from agent.retrieval.base import VectorStore


@dataclass(slots=True)
class RetrievalResult:
    """A retrieved chunk with its re-ranking score.

    Attributes
    ----------
    chunk_id : str
        Identifier of the matched chunk.
    content : str
        Chunk text.
    metadata : dict
        Chunk metadata.
    distance : float or None
        Backend distance; smaller means more similar.
    score : float
        Heuristic re-ranking score; larger is better.
    """

    chunk_id: str
    content: str
    metadata: dict[str, Any]
    distance: float | None
    score: float


class Retriever:
    """Retrieve and re-rank chunks for a query.

    Parameters
    ----------
    vector_store : VectorStore
        Backend used for nearest-neighbour search.
    embedder : Embedder
        Backend used to embed the query.
    """

    def __init__(self, vector_store: VectorStore, embedder: Embedder) -> None:
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve the ``top_k`` best chunks for ``query``.

        Parameters
        ----------
        query : str
            Natural-language query.
        top_k : int, optional
            Number of results to return after re-ranking.

        Returns
        -------
        list of RetrievalResult
            Results ordered by descending re-ranking score.
        """
        query_embedding = self.embedder.embed_query(query)

        raw_k = max(top_k * 3, 12)
        matches = self.vector_store.search(query_embedding, top_k=raw_k)

        results = [
            RetrievalResult(
                chunk_id=match.chunk_id,
                content=match.content,
                metadata=match.metadata,
                distance=match.distance,
                score=self._score_result(
                    query, match.metadata, match.content, match.distance
                ),
            )
            for match in matches
        ]

        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def _score_result(
        self,
        query: str,
        metadata: dict[str, Any],
        content: str,
        distance: float | None,
    ) -> float:
        """Compute a heuristic re-ranking score.

        Parameters
        ----------
        query : str
            The query text.
        metadata : dict
            Chunk metadata.
        content : str
            Chunk text.
        distance : float or None
            Backend distance; smaller means more similar.

        Returns
        -------
        float
            Re-ranking score; larger is better.
        """
        query_lower = query.lower()
        content_lower = content.lower()

        base_score = -(distance if distance is not None else 999.0)

        raw_page: Any = metadata.get("page_number")
        try:
            page_number: int | None = int(raw_page)
        except (TypeError, ValueError):
            page_number = None

        bonus = 0.0

        purpose_keywords = [
            "purpose",
            "objective",
            "goal",
            "main point",
            "contribution",
            "abstract",
        ]
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
