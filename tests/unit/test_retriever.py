"""Tests for the backend-agnostic retriever."""

from __future__ import annotations

from agent.ingestion.embedder import EmbeddedChunk
from agent.retrieval.retriever import Retriever
from tests.stubs import FakeEmbedder, FakeVectorStore


def _embedded(chunk_id: str, content: str, page: int) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        content=content,
        source="policy.pdf",
        embedding=[0.0] * 8,
        metadata={"page_number": page, "file_name": "policy.pdf"},
    )


def test_retrieve_returns_results_through_protocol() -> None:
    store = FakeVectorStore()
    store.add_embeddings(
        [
            _embedded("c1", "In this work we study the abstract objective.", 1),
            _embedded("c2", "Unrelated body text on a later page.", 5),
        ]
    )
    retriever = Retriever(vector_store=store, embedder=FakeEmbedder())
    results = retriever.retrieve("What is the purpose?", top_k=2)
    assert len(results) == 2
    # The page-1 abstract chunk should outrank the later-page chunk.
    assert results[0].chunk_id == "c1"


def test_retrieve_respects_top_k() -> None:
    store = FakeVectorStore()
    store.add_embeddings([_embedded(f"c{i}", f"text {i}", i) for i in range(10)])
    retriever = Retriever(vector_store=store, embedder=FakeEmbedder())
    assert len(retriever.retrieve("query", top_k=3)) == 3
