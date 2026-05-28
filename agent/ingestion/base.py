"""Provider-agnostic interface for embedding backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent.ingestion.chunker import Chunk
from agent.ingestion.embedder import EmbeddedChunk


@runtime_checkable
class Embedder(Protocol):
    """Contract for turning text into dense vectors.

    Implemented by the local Sentence Transformers backend (default) and the
    Bedrock Titan backend, so ingestion and retrieval stay backend-agnostic.
    """

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a batch of chunks.

        Parameters
        ----------
        chunks : list of Chunk
            Chunks to embed.

        Returns
        -------
        list of EmbeddedChunk
            One embedded chunk per input chunk, preserving order.
        """
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Parameters
        ----------
        query : str
            The query to embed.

        Returns
        -------
        list of float
            The query embedding vector.
        """
        ...
