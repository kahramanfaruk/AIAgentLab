"""Tests for the Chroma vector store backend."""

from __future__ import annotations

from pathlib import Path

from agent.ingestion.embedder import EmbeddedChunk
from agent.retrieval.vector_store import ChromaVectorStore


def _embedded(chunk_id: str, vector: list[float]) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        content=f"content {chunk_id}",
        source="policy.pdf",
        embedding=vector,
        metadata={"page_number": 1, "file_name": "policy.pdf"},
    )


def test_add_count_and_search(tmp_path: Path) -> None:
    store = ChromaVectorStore(
        persist_directory=tmp_path / "vdb", collection_name="test_collection"
    )
    added = store.add_embeddings(
        [
            _embedded("c1", [1.0, 0.0, 0.0]),
            _embedded("c2", [0.0, 1.0, 0.0]),
        ]
    )
    assert added == 2
    assert store.count() == 2

    matches = store.search([1.0, 0.0, 0.0], top_k=1)
    assert len(matches) == 1
    assert matches[0].chunk_id == "c1"
    assert matches[0].metadata["file_name"] == "policy.pdf"


def test_reset_clears_collection(tmp_path: Path) -> None:
    store = ChromaVectorStore(
        persist_directory=tmp_path / "vdb", collection_name="test_collection"
    )
    store.add_embeddings([_embedded("c1", [1.0, 0.0, 0.0])])
    store.reset()
    assert store.count() == 0


def test_add_empty_is_noop(tmp_path: Path) -> None:
    store = ChromaVectorStore(
        persist_directory=tmp_path / "vdb", collection_name="test_collection"
    )
    assert store.add_embeddings([]) == 0
