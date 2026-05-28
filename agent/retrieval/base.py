"""Provider-agnostic interface for vector store backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agent.ingestion.embedder import EmbeddedChunk


@dataclass(slots=True)
class VectorMatch:
    """A single nearest-neighbour match returned by a vector store.

    Attributes
    ----------
    chunk_id : str
        Identifier of the matched chunk.
    content : str
        Chunk text.
    metadata : dict
        Chunk metadata (source, page number, and similar).
    distance : float or None
        Backend distance score; smaller means more similar. ``None`` if the
        backend does not report a distance.
    """

    chunk_id: str
    content: str
    metadata: dict
    distance: float | None


@runtime_checkable
class VectorStore(Protocol):
    """Contract for indexing and searching embedded chunks.

    Implemented by the Chroma backend (local default) and the OpenSearch
    backend, so retrieval is independent of the storage engine.
    """

    def add_embeddings(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """Index embedded chunks.

        Parameters
        ----------
        embedded_chunks : list of EmbeddedChunk
            Chunks with precomputed embeddings.

        Returns
        -------
        int
            The number of chunks indexed.
        """
        ...

    def search(
        self, query_embedding: list[float], top_k: int
    ) -> list[VectorMatch]:
        """Return the ``top_k`` nearest neighbours to ``query_embedding``.

        Parameters
        ----------
        query_embedding : list of float
            The query vector.
        top_k : int
            Maximum number of matches to return.

        Returns
        -------
        list of VectorMatch
            Matches ordered from most to least similar.
        """
        ...

    def reset(self) -> None:
        """Remove all indexed data and recreate an empty store."""
        ...

    def count(self) -> int:
        """Return the number of indexed chunks.

        Returns
        -------
        int
            The current document count.
        """
        ...
