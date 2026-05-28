"""Tests for the backend selection factories."""

from __future__ import annotations

from agent.ingestion import build_embedder
from agent.ingestion.embedder import LocalEmbedder
from agent.memory import build_memory
from agent.memory.chat_history import InMemoryConversationStore
from agent.retrieval import build_vector_store
from agent.retrieval.vector_store import ChromaVectorStore
from agent.storage import build_document_store
from agent.storage.local_store import LocalDocumentStore
from config.settings import Settings


def test_build_vector_store_local(settings: Settings) -> None:
    assert isinstance(build_vector_store(settings), ChromaVectorStore)


def test_build_document_store_local(settings: Settings) -> None:
    assert isinstance(build_document_store(settings), LocalDocumentStore)


def test_build_memory_local(settings: Settings) -> None:
    assert isinstance(build_memory(settings), InMemoryConversationStore)


def test_build_embedder_local_is_lazy_type(settings: Settings) -> None:
    # The factory returns the local embedder type without forcing AWS imports.
    embedder = build_embedder(settings)
    assert isinstance(embedder, LocalEmbedder)
