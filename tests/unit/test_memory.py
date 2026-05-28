"""Tests for the conversation memory backends."""

from __future__ import annotations

from agent.memory.chat_history import (
    DynamoDBConversationStore,
    InMemoryConversationStore,
)


def test_in_memory_history_roundtrip() -> None:
    store = InMemoryConversationStore()
    store.add_message("s1", "user", "hello")
    store.add_message("s1", "assistant", "hi")
    history = store.get_history("s1")
    assert [m.role for m in history] == ["user", "assistant"]
    store.clear("s1")
    assert store.get_history("s1") == []


def test_in_memory_escalations_newest_first() -> None:
    store = InMemoryConversationStore()
    first = store.save_escalation({"reason": "a", "created_at": "2026-01-01T00:00:00"})
    second = store.save_escalation({"reason": "b", "created_at": "2026-02-01T00:00:00"})
    escalations = store.list_escalations()
    assert [e["reason"] for e in escalations] == ["b", "a"]
    assert first and second


def test_dynamodb_history_roundtrip(dynamodb_table: str) -> None:
    store = DynamoDBConversationStore(table_name=dynamodb_table, region="eu-central-1")
    store.add_message("s1", "user", "hello")
    store.add_message("s1", "assistant", "hi")
    history = store.get_history("s1")
    assert [m.content for m in history] == ["hello", "hi"]
    store.clear("s1")
    assert store.get_history("s1") == []


def test_dynamodb_escalations_roundtrip(dynamodb_table: str) -> None:
    store = DynamoDBConversationStore(table_name=dynamodb_table, region="eu-central-1")
    escalation_id = store.save_escalation({"reason": "low_confidence", "question": "q"})
    escalations = store.list_escalations()
    assert escalation_id
    assert escalations[0]["reason"] == "low_confidence"
