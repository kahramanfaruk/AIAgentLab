"""Retrieval package: vector stores, retriever, and the vector-store factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.retrieval.base import VectorMatch, VectorStore
from agent.retrieval.retriever import RetrievalResult, Retriever
from agent.retrieval.vector_store import ChromaVectorStore

if TYPE_CHECKING:
    from config.settings import Settings

__all__ = [
    "VectorMatch",
    "VectorStore",
    "RetrievalResult",
    "Retriever",
    "ChromaVectorStore",
    "build_vector_store",
]


def build_vector_store(settings: Settings) -> VectorStore:
    """Construct the vector store selected by ``settings.vector_backend``.

    Parameters
    ----------
    settings : Settings
        Application configuration.

    Returns
    -------
    VectorStore
        A store implementing the :class:`VectorStore` protocol.

    Raises
    ------
    ValueError
        If ``settings.vector_backend`` is not a supported backend.
    """
    if settings.vector_backend == "chroma":
        return ChromaVectorStore(
            persist_directory=settings.vector_db_path,
            collection_name=settings.chroma_collection_name,
        )

    if settings.vector_backend == "opensearch":
        from agent.retrieval.opensearch_store import OpenSearchVectorStore

        return OpenSearchVectorStore.from_settings(settings)

    raise ValueError(f"Unsupported vector_backend: {settings.vector_backend!r}")
