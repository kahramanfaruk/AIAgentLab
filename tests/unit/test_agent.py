"""Tests for the agent loop, including escalation paths."""

from __future__ import annotations

import pytest

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


def build_agent(responses: list[str], threshold: float = 0.45) -> Agent:
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
        memory=InMemoryConversationStore(),
        max_steps=4,
        confidence_threshold=threshold,
    )


def test_happy_path_retrieves_then_answers() -> None:
    agent = build_agent(
        [
            make_agent_plan(
                thought="need context",
                action={"tool": "retrieve_documents", "input": {"query": "d"}},
            ),
            make_agent_plan(
                thought="have evidence",
                final_answer="The deductible is 250 EUR [c1].",
                confidence=0.9,
            ),
        ]
    )
    result = agent.run("What is the deductible?")
    assert result.escalated is False
    assert "250" in result.answer
    assert len(result.context_chunks) == 1
    assert len(result.trace.steps) == 2


def test_low_confidence_escalates() -> None:
    agent = build_agent(
        [make_agent_plan(thought="unsure", final_answer="maybe", confidence=0.1)]
    )
    result = agent.run("Edge question?")
    assert result.escalated is True
    assert result.escalation is not None
    assert result.escalation.reason == "low_confidence"


def test_tool_requested_escalation() -> None:
    agent = build_agent(
        [
            make_agent_plan(
                thought="needs human",
                action={
                    "tool": "escalate_to_human",
                    "input": {"reason": "complex claim"},
                },
            )
        ]
    )
    result = agent.run("Complex claim?")
    assert result.escalated is True
    assert result.escalation is not None
    assert result.escalation.reason == "complex claim"


def test_input_guardrail_escalates_before_planning() -> None:
    agent = build_agent(
        [make_agent_plan(thought="x", final_answer="y", confidence=0.9)]
    )
    result = agent.run("I want help with self-harm please")
    assert result.escalated is True
    assert result.escalation is not None
    assert result.escalation.reason == "input_guardrail"


def test_invalid_json_then_recovers() -> None:
    agent = build_agent(
        [
            "this is not valid json",
            make_agent_plan(
                thought="ok", final_answer="Recovered answer.", confidence=0.8
            ),
        ]
    )
    result = agent.run("Recover?")
    assert result.escalated is False
    assert result.answer == "Recovered answer."


def test_max_steps_exceeded_escalates() -> None:
    agent = build_agent(
        [
            make_agent_plan(
                thought="loop",
                action={"tool": "retrieve_documents", "input": {"query": "x"}},
            )
        ]
    )
    result = agent.run("Loop forever?")
    assert result.escalated is True
    assert result.escalation is not None
    assert result.escalation.reason == "max_steps_exceeded"
    assert len(result.trace.steps) == 4


def test_escalation_is_persisted_to_memory() -> None:
    agent = build_agent(
        [make_agent_plan(thought="unsure", final_answer="maybe", confidence=0.0)]
    )
    agent.run("Edge?")
    escalations = agent.memory.list_escalations()
    assert len(escalations) == 1
    assert escalations[0]["reason"] == "low_confidence"


@pytest.mark.parametrize("value,expected", [(2.0, 1.0), (-1.0, 0.0), ("x", 0.5)])
def test_confidence_is_clamped(value: object, expected: float) -> None:
    assert Agent._clamp_confidence(value) == expected
