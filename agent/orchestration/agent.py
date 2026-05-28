"""The autonomous agent loop.

The agent plans one step at a time using structured JSON prompting, executes
tools, accumulates observations, and either produces a grounded final answer or
escalates to a human. Escalation is triggered by an input/output guardrail
block, low model confidence, an explicit ``escalate_to_human`` tool call, or
exhausting the step budget. Every run produces a full trace for inspection.

Structured prompting (rather than provider-native function calling) keeps the
loop identical across LLM backends and trivially testable with a scripted LLM.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from agent.generation.base import LLMClient
from agent.generation.prompts import AGENT_SYSTEM_PROMPT, build_agent_prompt
from agent.memory.chat_history import ConversationMemory
from agent.orchestration.guardrails import Guardrails
from agent.orchestration.schemas import (
    AgentResult,
    AgentStep,
    AgentTrace,
    Escalation,
    ToolCall,
)
from agent.orchestration.tools import ToolRegistry

_DEFAULT_CONFIDENCE = 0.5

_HANDOFF_MESSAGES = {
    "input_guardrail": (
        "This request cannot be handled automatically and has been routed to a "
        "human specialist."
    ),
    "output_guardrail": (
        "A draft answer was prepared but did not pass the safety checks, so the "
        "request has been routed to a human specialist."
    ),
    "low_confidence": (
        "I am not confident enough to answer this reliably, so I have routed it "
        "to a human specialist."
    ),
    "max_steps_exceeded": (
        "I could not resolve this within the allowed steps, so I have routed it "
        "to a human specialist."
    ),
}
_DEFAULT_HANDOFF = "This request has been routed to a human specialist."


class PlanParseError(ValueError):
    """Raised when the planner response is not a usable JSON object."""


class Agent:
    """A tool-using agent with guardrails and human-in-the-loop escalation.

    Parameters
    ----------
    llm_client : LLMClient
        Backend used for planning and answering.
    tools : ToolRegistry
        The tools available to the agent.
    guardrails : Guardrails
        Input and output safety checks.
    memory : ConversationMemory
        Store for chat history and escalations.
    max_steps : int, optional
        Maximum planning steps before forced escalation.
    confidence_threshold : float, optional
        Minimum final confidence required to answer without escalating.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        guardrails: Guardrails,
        memory: ConversationMemory,
        max_steps: int = 6,
        confidence_threshold: float = 0.45,
    ) -> None:
        self.llm_client = llm_client
        self.tools = tools
        self.guardrails = guardrails
        self.memory = memory
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold

    def run(self, question: str, session_id: str = "default") -> AgentResult:
        """Answer ``question`` autonomously or escalate.

        Parameters
        ----------
        question : str
            The user question.
        session_id : str, optional
            Conversation identifier used for memory.

        Returns
        -------
        AgentResult
            The final answer or a human-handoff result, with the full trace.
        """
        trace = AgentTrace()
        context_chunks: list[dict[str, Any]] = []

        input_check = self.guardrails.check_input(question)
        safe_question = input_check.text
        if not input_check.allowed:
            return self._escalate(
                question=safe_question,
                reason="input_guardrail",
                confidence=None,
                details={"violations": input_check.violations},
                trace=trace,
                context_chunks=context_chunks,
                session_id=session_id,
            )

        self.memory.add_message(session_id, "user", safe_question)
        scratchpad: list[str] = []

        for index in range(self.max_steps):
            prompt = build_agent_prompt(
                question=safe_question,
                tools_description=self.tools.describe(),
                scratchpad="\n".join(scratchpad),
            )
            raw = self.llm_client.generate(prompt, system=AGENT_SYSTEM_PROMPT)

            try:
                plan = self._parse_plan(raw)
            except PlanParseError as exc:
                trace.steps.append(
                    AgentStep(index=index, thought=f"invalid planner output: {exc}")
                )
                scratchpad.append(
                    f"Step {index}: planner output was not valid JSON; "
                    "respond with exactly one JSON object."
                )
                continue

            thought = str(plan.get("thought", ""))

            if "final_answer" in plan:
                return self._finalize(
                    plan=plan,
                    thought=thought,
                    index=index,
                    safe_question=safe_question,
                    trace=trace,
                    context_chunks=context_chunks,
                    session_id=session_id,
                )

            action = plan.get("action")
            if not isinstance(action, dict) or "tool" not in action:
                trace.steps.append(AgentStep(index=index, thought=thought))
                scratchpad.append(
                    f"Step {index}: response had neither a valid action nor a "
                    "final_answer."
                )
                continue

            tool_name = str(action["tool"])
            tool_input = action.get("input")
            if not isinstance(tool_input, dict):
                tool_input = {}

            observation, tool_data, escalation_reason = self._run_tool(
                tool_name, tool_input
            )
            trace.steps.append(
                AgentStep(
                    index=index,
                    thought=thought,
                    tool_call=ToolCall(
                        tool=tool_name, tool_input=tool_input, output=observation
                    ),
                )
            )

            if "context_chunks" in tool_data:
                context_chunks = tool_data["context_chunks"]

            if escalation_reason is not None:
                return self._escalate(
                    question=safe_question,
                    reason=escalation_reason,
                    confidence=None,
                    details={"tool": tool_name},
                    trace=trace,
                    context_chunks=context_chunks,
                    session_id=session_id,
                )

            scratchpad.append(
                f"Step {index}: thought: {thought}\n"
                f"Action: {tool_name}({json.dumps(tool_input)})\n"
                f"Observation: {observation}"
            )

        return self._escalate(
            question=safe_question,
            reason="max_steps_exceeded",
            confidence=None,
            details={"max_steps": self.max_steps},
            trace=trace,
            context_chunks=context_chunks,
            session_id=session_id,
        )

    def _run_tool(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> tuple[str, dict[str, Any], str | None]:
        """Execute a tool and normalize its outcome.

        Parameters
        ----------
        tool_name : str
            Name of the tool to run.
        tool_input : dict
            Tool arguments.

        Returns
        -------
        tuple of (str, dict, str or None)
            The observation text, the tool's structured data, and an escalation
            reason if the tool requested a handoff (otherwise ``None``). Tool
            errors are returned as observations so the agent can recover.
        """
        try:
            tool = self.tools.get(tool_name)
            result = tool.run(tool_input)
        except (KeyError, ValueError) as exc:
            return f"Tool error: {exc}", {}, None

        reason = result.escalation_reason if result.requests_escalation else None
        return result.output, result.data, reason

    def _finalize(
        self,
        plan: dict[str, Any],
        thought: str,
        index: int,
        safe_question: str,
        trace: AgentTrace,
        context_chunks: list[dict[str, Any]],
        session_id: str,
    ) -> AgentResult:
        """Validate a proposed final answer and answer or escalate.

        Parameters
        ----------
        plan : dict
            The parsed planner output containing ``final_answer``.
        thought : str
            The planner's reasoning for this step.
        index : int
            Current step index.
        safe_question : str
            The redacted question.
        trace : AgentTrace
            The trace to append to.
        context_chunks : list of dict
            Grounding context gathered so far.
        session_id : str
            Conversation identifier.

        Returns
        -------
        AgentResult
            A successful answer, or an escalation if guardrails block the output
            or confidence is below threshold.
        """
        trace.steps.append(AgentStep(index=index, thought=thought))
        answer = str(plan["final_answer"])
        confidence = self._clamp_confidence(plan.get("confidence"))

        output_check = self.guardrails.check_output(answer)
        if not output_check.allowed:
            return self._escalate(
                question=safe_question,
                reason="output_guardrail",
                confidence=confidence,
                details={"violations": output_check.violations},
                trace=trace,
                context_chunks=context_chunks,
                session_id=session_id,
            )

        if confidence < self.confidence_threshold:
            return self._escalate(
                question=safe_question,
                reason="low_confidence",
                confidence=confidence,
                details={"draft_answer": output_check.text},
                trace=trace,
                context_chunks=context_chunks,
                session_id=session_id,
            )

        answer = output_check.text
        self.memory.add_message(session_id, "assistant", answer)
        return AgentResult(
            question=safe_question,
            answer=answer,
            confidence=confidence,
            escalated=False,
            trace=trace,
            context_chunks=context_chunks,
        )

    def _escalate(
        self,
        question: str,
        reason: str,
        confidence: float | None,
        details: dict[str, Any],
        trace: AgentTrace,
        context_chunks: list[dict[str, Any]],
        session_id: str,
    ) -> AgentResult:
        """Record an escalation and return a human-handoff result.

        Parameters
        ----------
        question : str
            The redacted question.
        reason : str
            Machine-readable escalation reason.
        confidence : float or None
            Confidence at handoff, if known.
        details : dict
            Extra context persisted with the escalation.
        trace : AgentTrace
            The reasoning trace.
        context_chunks : list of dict
            Grounding context gathered so far.
        session_id : str
            Conversation identifier.

        Returns
        -------
        AgentResult
            A result flagged as escalated.
        """
        escalation = Escalation(
            reason=reason,
            question=question,
            confidence=confidence,
            details=details,
        )
        self.memory.save_escalation(asdict(escalation))

        answer = _HANDOFF_MESSAGES.get(reason, _DEFAULT_HANDOFF)
        self.memory.add_message(session_id, "assistant", answer)

        return AgentResult(
            question=question,
            answer=answer,
            confidence=confidence if confidence is not None else 0.0,
            escalated=True,
            trace=trace,
            escalation=escalation,
            context_chunks=context_chunks,
        )

    @staticmethod
    def _parse_plan(text: str) -> dict[str, Any]:
        """Parse a single JSON object from a planner response.

        Parameters
        ----------
        text : str
            Raw planner output, possibly wrapped in prose or a code fence.

        Returns
        -------
        dict
            The parsed object.

        Raises
        ------
        PlanParseError
            If no JSON object can be parsed.
        """
        candidate = text.strip()
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise PlanParseError("no JSON object found in planner output")

        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise PlanParseError(str(exc)) from exc

        if not isinstance(parsed, dict):
            raise PlanParseError("planner output was not a JSON object")
        return parsed

    @staticmethod
    def _clamp_confidence(value: Any) -> float:
        """Coerce a confidence value into ``[0, 1]``.

        Parameters
        ----------
        value : Any
            The raw confidence from the planner.

        Returns
        -------
        float
            The clamped confidence, defaulting to ``0.5`` when missing or
            non-numeric.
        """
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return _DEFAULT_CONFIDENCE
        return max(0.0, min(1.0, confidence))
