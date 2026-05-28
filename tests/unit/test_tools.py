"""Tests for agent tools and the tool registry."""

from __future__ import annotations

import pytest

from agent.orchestration.tools import (
    EscalateToHumanTool,
    LookupPolicyClauseTool,
    RetrieveDocumentsTool,
    ToolRegistry,
)
from tests.stubs import FakeRetriever


def test_retrieve_documents_returns_context() -> None:
    tool = RetrieveDocumentsTool(retriever=FakeRetriever())
    result = tool.run({"query": "deductible"})
    assert result.data["context_chunks"]
    assert "250" in result.output


def test_retrieve_documents_requires_query() -> None:
    tool = RetrieveDocumentsTool(retriever=FakeRetriever())
    with pytest.raises(ValueError):
        tool.run({"query": "   "})


def test_lookup_policy_clause_matches_topic() -> None:
    tool = LookupPolicyClauseTool({"deductible": "Deductible is 250 EUR."})
    result = tool.run({"topic": "what is the deductible"})
    assert "250" in result.output
    assert result.data["clause_key"] == "deductible"


def test_lookup_policy_clause_reports_missing() -> None:
    tool = LookupPolicyClauseTool({"deductible": "x"})
    result = tool.run({"topic": "flooding"})
    assert result.data["clause_key"] is None


def test_escalate_tool_requests_escalation() -> None:
    result = EscalateToHumanTool().run({"reason": "needs adjuster"})
    assert result.requests_escalation is True
    assert result.escalation_reason == "needs adjuster"


def test_registry_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError):
        ToolRegistry([EscalateToHumanTool(), EscalateToHumanTool()])


def test_registry_get_unknown_raises() -> None:
    registry = ToolRegistry([EscalateToHumanTool()])
    with pytest.raises(KeyError):
        registry.get("missing")


def test_registry_describe_lists_tools() -> None:
    registry = ToolRegistry(
        [RetrieveDocumentsTool(FakeRetriever()), EscalateToHumanTool()]
    )
    described = registry.describe()
    assert "retrieve_documents" in described
    assert "escalate_to_human" in described
