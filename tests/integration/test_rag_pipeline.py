"""End-to-end RAG pipeline test on the local stack (no model, no AWS).

Exercises load -> chunk -> embed -> index -> retrieve -> generate using the
deterministic fake embedder, a real Chroma store, and a scripted LLM.
"""

from __future__ import annotations

from pathlib import Path

from agent.generation.chain import RAGChain
from agent.ingestion.pipeline import index_document_bytes
from agent.retrieval.retriever import Retriever
from agent.retrieval.vector_store import ChromaVectorStore
from tests.stubs import FakeEmbedder, StubLLM


def test_full_local_pipeline(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    store = ChromaVectorStore(
        persist_directory=tmp_path / "vdb", collection_name="test_collection"
    )

    summary = index_document_bytes(
        name="policy.txt",
        data=b"The deductible is 250 EUR per claim. Coverage includes fire. " * 10,
        embedder=embedder,
        vector_store=store,
        chunk_size=200,
        chunk_overlap=40,
    )
    assert summary.chunks_indexed > 0
    assert store.count() == summary.chunks_indexed

    retriever = Retriever(vector_store=store, embedder=embedder)
    chain = RAGChain(
        retriever=retriever,
        llm_client=StubLLM(["The deductible is 250 EUR."]),
    )
    result = chain.ask("What is the deductible?", top_k=3)
    assert result.answer == "The deductible is 250 EUR."
    assert len(result.context_chunks) >= 1
