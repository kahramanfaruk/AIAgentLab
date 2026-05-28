"""Provider-agnostic interface for document storage backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DocumentStore(Protocol):
    """Contract for persisting and retrieving raw document bytes.

    Implemented by the local filesystem backend (default) and the S3 backend,
    so ingestion does not depend on where documents live.
    """

    def save(self, name: str, data: bytes) -> str:
        """Persist ``data`` under ``name``.

        Parameters
        ----------
        name : str
            Document name (used as the object key or file name).
        data : bytes
            Raw document content.

        Returns
        -------
        str
            A locator for the stored document (a path or an ``s3://`` URI).
        """
        ...

    def load(self, name: str) -> bytes:
        """Load the document stored under ``name``.

        Parameters
        ----------
        name : str
            Document name.

        Returns
        -------
        bytes
            The stored content.

        Raises
        ------
        FileNotFoundError
            If no document is stored under ``name``.
        """
        ...

    def list_documents(self) -> list[str]:
        """List the names of all stored documents.

        Returns
        -------
        list of str
            Document names, sorted.
        """
        ...

    def exists(self, name: str) -> bool:
        """Report whether a document is stored under ``name``.

        Parameters
        ----------
        name : str
            Document name.

        Returns
        -------
        bool
            ``True`` if the document exists.
        """
        ...
