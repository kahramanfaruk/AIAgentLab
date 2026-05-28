"""Local filesystem document storage backend (free local default)."""

from __future__ import annotations

from pathlib import Path

from agent.ingestion.loader import SUPPORTED_EXTENSIONS


class LocalDocumentStore:
    """Store documents as files under a base directory.

    Parameters
    ----------
    base_dir : str or Path, optional
        Directory where documents are stored. Created if it does not exist.
    """

    def __init__(self, base_dir: str | Path = "data/raw") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, data: bytes) -> str:
        """Write ``data`` to ``base_dir / name``.

        Parameters
        ----------
        name : str
            File name.
        data : bytes
            Raw document content.

        Returns
        -------
        str
            The absolute path to the written file.
        """
        path = self._resolve(name)
        path.write_bytes(data)
        return str(path)

    def load(self, name: str) -> bytes:
        """Read the document stored under ``name``.

        Parameters
        ----------
        name : str
            File name.

        Returns
        -------
        bytes
            File content.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = self._resolve(name)
        if not path.is_file():
            raise FileNotFoundError(f"Document not found: {path}")
        return path.read_bytes()

    def list_documents(self) -> list[str]:
        """List names of supported documents in ``base_dir``.

        Returns
        -------
        list of str
            Sorted file names whose extensions are supported by the loader.
        """
        return sorted(
            path.name
            for path in self.base_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def exists(self, name: str) -> bool:
        """Report whether ``base_dir / name`` exists.

        Parameters
        ----------
        name : str
            File name.

        Returns
        -------
        bool
            ``True`` if the file exists.
        """
        return self._resolve(name).is_file()

    def _resolve(self, name: str) -> Path:
        """Resolve ``name`` to a path inside ``base_dir``, rejecting traversal.

        Parameters
        ----------
        name : str
            Candidate file name.

        Returns
        -------
        Path
            The resolved path within ``base_dir``.

        Raises
        ------
        ValueError
            If ``name`` resolves outside ``base_dir``.
        """
        candidate = (self.base_dir / name).resolve()
        base = self.base_dir.resolve()
        if base != candidate and base not in candidate.parents:
            raise ValueError(f"Document name escapes the storage directory: {name!r}")
        return candidate
