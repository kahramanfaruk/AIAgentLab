"""Dataclasses describing agent execution, traces, and escalations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolCall:
    """A single tool invocation and its result.

    Attributes
    ----------
    tool : str
        Name of the invoked tool.
    tool_input : dict
        Arguments passed to the tool.
    output : str
        Human-readable tool output.
    """

    tool: str
    tool_input: dict[str, Any]
    output: str


@dataclass(slots=True)
class AgentStep:
    """One reasoning step in the agent loop.

    Attributes
    ----------
    index : int
        Zero-based step number.
    thought : str
        The model's stated reasoning for this step.
    tool_call : ToolCall or None
        The tool invoked this step, or ``None`` if the step produced the final
        answer.
    """

    index: int
    thought: str
    tool_call: ToolCall | None = None


@dataclass(slots=True)
class AgentTrace:
    """The ordered sequence of steps taken during a run.

    Attributes
    ----------
    steps : list of AgentStep
        Steps in execution order.
    """

    steps: list[AgentStep] = field(default_factory=list)


@dataclass(slots=True)
class Escalation:
    """A human-in-the-loop handoff record.

    Attributes
    ----------
    reason : str
        Machine-readable escalation reason.
    question : str
        The question that triggered the escalation.
    confidence : float or None
        Model confidence at handoff, if available.
    details : dict
        Additional context, for example guardrail violations.
    """

    reason: str
    question: str
    confidence: float | None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentResult:
    """The outcome of an agent run.

    Attributes
    ----------
    question : str
        The original question.
    answer : str
        The final answer or human-handoff message.
    confidence : float
        Final confidence in ``[0, 1]``.
    escalated : bool
        Whether the run was escalated to a human.
    escalation : Escalation or None
        The escalation record when ``escalated`` is ``True``.
    trace : AgentTrace
        The full reasoning trace.
    context_chunks : list of dict
        Grounding context retrieved during the run.
    """

    question: str
    answer: str
    confidence: float
    escalated: bool
    trace: AgentTrace
    escalation: Escalation | None = None
    context_chunks: list[dict[str, Any]] = field(default_factory=list)
