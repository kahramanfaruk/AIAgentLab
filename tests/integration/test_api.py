"""Integration tests for the FastAPI layer using an injected context."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.generation.chain import RAGChain
from agent.memory.chat_history import InMemoryConversationStore
from agent.orchestration.agent import Agent
from agent.orchestration.guardrails import Guardrails
from agent.orchestration.tools import (
    EscalateToHumanTool,
    LookupPolicyClauseTool,
    RetrieveDocumentsTool,
    ToolRegistry,
)
from agent.storage.local_store import LocalDocumentStore
from api.context import AppContext
from api.main import create_app
from config.settings import Settings
from tests.stubs import FakeEmbedder, FakeRetriever, FakeVectorStore, StubLLM


def _agent(responses: list[str], memory: InMemoryConversationStore) -> Agent:
    tools = ToolRegistry(
        [
            RetrieveDocumentsTool(FakeRetriever()),
            LookupPolicyClauseTool({"deductible": "Deductible is 250 EUR."}),
            EscalateToHumanTool(),
        ]
    )
    return Agent(
        llm_client=StubLLM(responses),
        tools=tools,
        guardrails=Guardrails(blocked_topics=["self-harm"]),
        memory=memory,
        max_steps=4,
        confidence_threshold=0.45,
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    memory = InMemoryConversationStore()
    retriever = FakeRetriever()
    context = AppContext(
        settings=Settings(),
        document_store=LocalDocumentStore(base_dir=tmp_path),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        retriever=retriever,
        memory=memory,
        rag_chain=RAGChain(
            retriever=retriever,
            llm_client=StubLLM(["The deductible is 250 EUR [c1]."]),
        ),
        agent=_agent(
            [
                '{"thought": "ctx", "action": {"tool": "retrieve_documents", '
                '"input": {"query": "d"}}}',
                '{"thought": "ok", "final_answer": "Deductible is 250 EUR [c1].", '
                '"confidence": 0.9}',
            ],
            memory,
        ),
        llm_error=None,
    )
    app = create_app()
    app.state.context = context
    return TestClient(app)


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_ask(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "What is the deductible?"})
    assert response.status_code == 200
    assert "250" in response.json()["answer"]


def test_agent(client: TestClient) -> None:
    response = client.post("/agent", json={"question": "deductible?"})
    body = response.json()
    assert response.status_code == 200
    assert body["escalated"] is False
    assert len(body["trace"]) == 2
    assert len(body["context_chunks"]) == 1


def test_ingest_and_list_documents(client: TestClient) -> None:
    files = {"file": ("note.txt", b"hello world " * 30, "text/plain")}
    ingest = client.post("/ingest", files=files)
    assert ingest.status_code == 200
    assert ingest.json()["chunks_indexed"] >= 1
    assert "note.txt" in client.get("/documents").json()["documents"]


def test_ask_returns_503_without_llm(tmp_path: Path) -> None:
    context = AppContext(
        settings=Settings(),
        document_store=LocalDocumentStore(base_dir=tmp_path),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        retriever=FakeRetriever(),
        memory=InMemoryConversationStore(),
        rag_chain=None,
        agent=None,
        llm_error="GROQ_API_KEY missing",
    )
    app = create_app()
    app.state.context = context
    client = TestClient(app)
    assert client.post("/ask", json={"question": "x"}).status_code == 503
