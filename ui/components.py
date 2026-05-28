"""Shared Streamlit render helpers for the AIAgentLab UI."""

from __future__ import annotations

from typing import Any

import streamlit as st

from agent.orchestration.schemas import AgentResult


def render_sources(context_chunks: list[dict[str, Any]]) -> None:
    """Render retrieved context blocks in an expander.

    Parameters
    ----------
    context_chunks : list of dict
        Context blocks with ``chunk_id``, ``source``, ``page_number``, and
        ``content`` keys.
    """
    if not context_chunks:
        return
    with st.expander("Retrieved sources"):
        for index, chunk in enumerate(context_chunks, start=1):
            st.markdown(
                f"**{index}. {chunk['chunk_id']}**  \n"
                f"Source: {chunk['source']}  \n"
                f"Page: {chunk['page_number']}"
            )
            st.code(chunk["content"][:1200])


def render_agent_trace(result: AgentResult) -> None:
    """Render the agent reasoning trace in an expander.

    Parameters
    ----------
    result : AgentResult
        The agent run outcome.
    """
    steps = result.trace.steps
    with st.expander(f"Agent trace ({len(steps)} steps)"):
        for step in steps:
            st.markdown(f"**Step {step.index}** - {step.thought}")
            if step.tool_call is not None:
                st.markdown(
                    f"Tool: `{step.tool_call.tool}`  \n"
                    f"Input: `{step.tool_call.tool_input}`"
                )
                st.code(step.tool_call.output[:600])


def render_escalation(result: AgentResult) -> None:
    """Render an escalation banner when the agent handed off to a human.

    Parameters
    ----------
    result : AgentResult
        The agent run outcome.
    """
    if not result.escalated or result.escalation is None:
        return
    escalation = result.escalation
    confidence = (
        f"{escalation.confidence:.2f}"
        if escalation.confidence is not None
        else "n/a"
    )
    st.warning(
        f"Escalated to a human specialist. Reason: {escalation.reason}; "
        f"confidence: {confidence}."
    )
