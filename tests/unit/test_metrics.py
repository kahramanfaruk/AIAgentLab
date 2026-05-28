"""Tests for evaluation metrics."""

from __future__ import annotations

from agent.evaluation.metrics import (
    citation_validity,
    escalation_correct,
    keyword_coverage,
    retrieval_hit,
    tool_selection_correct,
)
from agent.orchestration.schemas import AgentStep, AgentTrace, ToolCall


def test_keyword_coverage() -> None:
    assert keyword_coverage("fire and theft are covered", ["fire", "theft"]) == 1.0
    assert keyword_coverage("only fire", ["fire", "theft"]) == 0.5
    assert keyword_coverage("anything", []) == 1.0


def test_retrieval_hit() -> None:
    chunks = [{"source": "data/raw/policy.pdf"}]
    assert retrieval_hit(chunks, ["policy.pdf"]) == 1.0
    assert retrieval_hit(chunks, ["other.pdf"]) == 0.0
    assert retrieval_hit([], []) == 1.0


def test_escalation_correct() -> None:
    assert escalation_correct(True, True) == 1.0
    assert escalation_correct(False, True) == 0.0


def test_tool_selection_correct() -> None:
    trace = AgentTrace(
        steps=[
            AgentStep(
                index=0,
                thought="t",
                tool_call=ToolCall(tool="retrieve_documents", tool_input={}, output=""),
            )
        ]
    )
    assert tool_selection_correct(trace, "retrieve_documents") == 1.0
    assert tool_selection_correct(trace, "escalate_to_human") == 0.0
    assert tool_selection_correct(trace, None) == 1.0


def test_citation_validity() -> None:
    chunks = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
    assert citation_validity("answer [c1] and [c2]", chunks) == 1.0
    assert citation_validity("answer [c1] and [c9]", chunks) == 0.5
    assert citation_validity("no citations here", chunks) == 1.0
