"""Ingestion package: loaders, chunkers, embedders, and the embedder factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.ingestion.base import Embedder
from agent.ingestion.embedder import EmbeddedChunk, LocalEmbedder

if TYPE_CHECKING:
    from config.settings import Settings

__all__ = [
    "Embedder",
    "EmbeddedChunk",
    "LocalEmbedder",
    "build_embedder",
]


def build_embedder(settings: Settings) -> Embedder:
    """Construct the embedder selected by ``settings.embedding_provider``.

    Parameters
    ----------
    settings : Settings
        Application configuration.

    Returns
    -------
    Embedder
        An embedder implementing the :class:`Embedder` protocol.

    Raises
    ------
    ValueError
        If ``settings.embedding_provider`` is not a supported backend.
    """
    if settings.embedding_provider == "local":
        return LocalEmbedder(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )

    if settings.embedding_provider == "bedrock":
        from agent.ingestion.bedrock_embedder import BedrockEmbedder

        return BedrockEmbedder.from_settings(settings)

    raise ValueError(
        f"Unsupported embedding_provider: {settings.embedding_provider!r}"
    )
