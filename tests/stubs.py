"""Reusable test doubles for AIAgentLab.

These stubs let the orchestration, API, and ingestion paths be tested without an
LLM provider, an embedding model download, or any AWS calls.
"""

from __future__ import annotations

from typing import Any

from agent.ingestion.chunker import Chunk
from agent.ingestion.embedder import EmbeddedChunk
from agent.retrieval.base import VectorMatch
from agent.retrieval.retriever import RetrievalResult


class StubLLM:
    """An LLM client that replays scripted responses.

    Parameters
    ----------
    responses : list of str
        Responses returned in order; the last response repeats once exhausted.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return the next scripted response and record the prompt."""
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self._responses) - 1)
        return self._responses[index]


class FakeEmbedder:
    """A deterministic embedder that needs no model.

    Parameters
    ----------
    dim : int, optional
        Embedding dimensionality.
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        seed = sum(ord(char) for char in text) or 1
        return [((seed * (i + 1)) % 97) / 97.0 for i in range(self.dim)]

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed chunks deterministically."""
        return [
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                source=chunk.source,
                embedding=self._vector(chunk.content),
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]

    def embed_query(self, query: str) -> list[float]:
        """Embed a query deterministically."""
        return self._vector(query)


class FakeVectorStore:
    """An in-memory vector store that returns inserted chunks in order."""

    def __init__(self) -> None:
        self._chunks: list[EmbeddedChunk] = []

    def add_embeddings(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """Store chunks and return the count added."""
        self._chunks.extend(embedded_chunks)
        return len(embedded_chunks)

    def search(self, query_embedding: list[float], top_k: int) -> list[VectorMatch]:
        """Return the first ``top_k`` stored chunks as matches."""
        return [
            VectorMatch(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
                distance=float(index),
            )
            for index, chunk in enumerate(self._chunks[:top_k])
        ]

    def reset(self) -> None:
        """Clear all stored chunks."""
        self._chunks.clear()

    def count(self) -> int:
        """Return the number of stored chunks."""
        return len(self._chunks)


class FakeRetriever:
    """A retriever that returns fixed results regardless of the query.

    Parameters
    ----------
    results : list of RetrievalResult or None, optional
        Results to return; defaults to a single insurance-style result.
    """

    def __init__(self, results: list[RetrievalResult] | None = None) -> None:
        self._results = results or [
            RetrievalResult(
                chunk_id="c1",
                content="The deductible is 250 EUR per claim.",
                metadata={"page_number": 1, "file_name": "policy.pdf"},
                distance=0.1,
                score=1.0,
            )
        ]

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return the fixed results, truncated to ``top_k``."""
        return list(self._results[:top_k])


def make_agent_plan(**fields: Any) -> str:
    """Render a planner-style JSON object for scripting the agent.

    Parameters
    ----------
    **fields : Any
        Keys to include in the JSON object (for example ``thought``,
        ``action``, ``final_answer``, ``confidence``).

    Returns
    -------
    str
        A compact JSON string.
    """
    import json

    return json.dumps(fields)
