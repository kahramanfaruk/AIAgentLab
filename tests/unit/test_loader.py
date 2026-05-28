"""Tests for the document loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.ingestion.loader import load_documents


def test_loads_txt_file(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("Hello world.\n\nSecond line.", encoding="utf-8")
    documents = load_documents(tmp_path)
    assert len(documents) == 1
    assert "Hello world." in documents[0].content
    assert documents[0].metadata["file_type"] == "txt"


def test_ignores_unsupported_extensions(tmp_path: Path) -> None:
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")
    (tmp_path / "keep.txt").write_text("content", encoding="utf-8")
    documents = load_documents(tmp_path)
    assert len(documents) == 1


def test_missing_directory_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_documents("/nonexistent/path/aiagentlab")
