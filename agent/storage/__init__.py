"""Storage package: document stores and the document-store factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.storage.base import DocumentStore
from agent.storage.local_store import LocalDocumentStore

if TYPE_CHECKING:
    from config.settings import Settings

__all__ = [
    "DocumentStore",
    "LocalDocumentStore",
    "build_document_store",
]


def build_document_store(settings: Settings) -> DocumentStore:
    """Construct the document store selected by ``settings.storage_backend``.

    Parameters
    ----------
    settings : Settings
        Application configuration.

    Returns
    -------
    DocumentStore
        A store implementing the :class:`DocumentStore` protocol.

    Raises
    ------
    ValueError
        If ``settings.storage_backend`` is not a supported backend.
    """
    if settings.storage_backend == "local":
        return LocalDocumentStore()

    if settings.storage_backend == "s3":
        from agent.storage.s3_store import S3DocumentStore

        return S3DocumentStore.from_settings(settings)

    raise ValueError(f"Unsupported storage_backend: {settings.storage_backend!r}")
