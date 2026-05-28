"""Tests for the evaluation harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.evaluation.evaluator import evaluate, load_dataset
from agent.memory.chat_history import InMemoryConversationStore
from agent.orchestration.agent import Agent
from agent.orchestration.guardrails import Guardrails
from agent.orchestration.tools import (
    EscalateToHumanTool,
    LookupPolicyClauseTool,
    RetrieveDocumentsTool,
    ToolRegistry,
)
from tests.stubs import FakeRetriever, StubLLM, make_agent_plan


def test_load_dataset_parses_items(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"question": "q1", "expected_escalation": true}\n'
        '{"question": "q2", "expected_keywords": ["a"]}\n',
        encoding="utf-8",
    )
    items = load_dataset(path)
    assert len(items) == 2
    assert items[0].expected_escalation is True
    assert items[1].expected_keywords == ["a"]


def test_load_dataset_missing_question_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"expected_escalation": true}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)


def test_evaluate_aggregates_metrics(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"question": "deductible?", "expected_keywords": ["250"], '
        '"expected_escalation": false, "expected_tool": "retrieve_documents"}\n',
        encoding="utf-8",
    )
    agent = Agent(
        llm_client=StubLLM(
            [
                make_agent_plan(
                    thought="ctx",
                    action={"tool": "retrieve_documents", "input": {"query": "d"}},
                ),
                make_agent_plan(
                    thought="ok",
                    final_answer="The deductible is 250 EUR [c1].",
                    confidence=0.9,
                ),
            ]
        ),
        tools=ToolRegistry(
            [
                RetrieveDocumentsTool(FakeRetriever()),
                LookupPolicyClauseTool({"deductible": "x"}),
                EscalateToHumanTool(),
            ]
        ),
        guardrails=Guardrails(),
        memory=InMemoryConversationStore(),
        max_steps=4,
        confidence_threshold=0.45,
    )

    report = evaluate(agent, load_dataset(path))
    assert report.aggregates["escalation_correct"] == 1.0
    assert report.aggregates["tool_selection_correct"] == 1.0
    assert report.aggregates["keyword_coverage"] == 1.0
    assert "Evaluation report" in report.format()
