"""Tests for the RAG chain."""

from __future__ import annotations

from agent.generation.chain import RAGChain
from tests.stubs import FakeRetriever, StubLLM


def test_ask_returns_answer_and_context() -> None:
    chain = RAGChain(
        retriever=FakeRetriever(),
        llm_client=StubLLM(["The deductible is 250 EUR [c1]."]),
    )
    result = chain.ask("What is the deductible?", top_k=3)
    assert result.answer == "The deductible is 250 EUR [c1]."
    assert len(result.context_chunks) == 1
    assert result.context_chunks[0]["chunk_id"] == "c1"


def test_prompt_includes_question() -> None:
    llm = StubLLM(["answer"])
    RAGChain(retriever=FakeRetriever(), llm_client=llm).ask("unique-question-token")
    assert "unique-question-token" in llm.prompts[0]
