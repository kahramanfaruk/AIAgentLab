"""Chroma vector store backend.

Chroma is the free, local-development default vector store. It implements the
:class:`~agent.retrieval.base.VectorStore` protocol so it is interchangeable
with the OpenSearch backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.api.models.Collection import Collection

from agent.ingestion.embedder import EmbeddedChunk
from agent.retrieval.base import VectorMatch


class ChromaVectorStore:
    """Persistent local vector store backed by Chroma.

    Parameters
    ----------
    persist_directory : str or Path, optional
        Directory where Chroma persists its data.
    collection_name : str, optional
        Name of the Chroma collection.
    """

    _COLLECTION_METADATA = {"description": "AIAgentLab RAG document store"}

    def __init__(
        self,
        persist_directory: str | Path = "data/vector_db",
        collection_name: str = "rag_documents",
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection_name = collection_name
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=self._COLLECTION_METADATA,
        )

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
        if not embedded_chunks:
            return 0

        embeddings = cast(
            list[Sequence[float]],
            [chunk.embedding for chunk in embedded_chunks],
        )
        self.collection.add(
            ids=[chunk.chunk_id for chunk in embedded_chunks],
            documents=[chunk.content for chunk in embedded_chunks],
            embeddings=embeddings,
            metadatas=[
                self._sanitize_metadata(chunk.metadata, chunk.source)
                for chunk in embedded_chunks
            ],
        )
        return len(embedded_chunks)

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
        results = self.collection.query(
            query_embeddings=cast(list[Sequence[float]], [query_embedding]),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        return [
            VectorMatch(
                chunk_id=chunk_id,
                content=content,
                metadata=dict(metadata) if metadata else {},
                distance=distance,
            )
            for chunk_id, content, metadata, distance in zip(
                ids, documents, metadatas, distances
            )
        ]

    def reset(self) -> None:
        """Remove all indexed data and recreate an empty collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=self._COLLECTION_METADATA,
        )

    def count(self) -> int:
        """Return the number of indexed chunks.

        Returns
        -------
        int
            The current collection count.
        """
        return self.collection.count()

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any], source: str) -> dict[str, Any]:
        """Coerce metadata into Chroma-compatible scalar values.

        Parameters
        ----------
        metadata : dict
            Raw chunk metadata.
        source : str
            Document source path, added under the ``"source"`` key.

        Returns
        -------
        dict
            Metadata with only ``str``, ``int``, ``float``, ``bool``, or
            ``None`` values.
        """
        clean: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                clean[key] = value
            else:
                clean[key] = str(value)
        clean["source"] = source
        return clean
