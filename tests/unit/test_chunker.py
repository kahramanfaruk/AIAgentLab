"""Tests for the text chunker."""

from __future__ import annotations

import pytest

from agent.ingestion.chunker import TextChunker, chunk_documents
from agent.ingestion.loader import Document


def _document(content: str) -> Document:
    return Document(
        content=content,
        source="mem",
        doc_id="doc",
        metadata={"file_name": "doc.txt"},
    )


def test_short_text_is_single_chunk() -> None:
    chunks = chunk_documents(
        [_document("short text")], chunk_size=100, chunk_overlap=20
    )
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "doc-chunk-0"


def test_long_text_is_split_with_overlap() -> None:
    text = "word " * 400
    chunks = chunk_documents([_document(text)], chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(len(chunk.content) <= 200 for chunk in chunks)


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_metadata_is_propagated() -> None:
    chunks = chunk_documents([_document("x" * 500)], chunk_size=120, chunk_overlap=20)
    assert chunks[0].metadata["parent_doc_id"] == "doc"
    assert chunks[0].metadata["file_name"] == "doc.txt"
