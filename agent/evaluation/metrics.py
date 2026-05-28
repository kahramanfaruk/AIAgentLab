"""Per-item evaluation metrics for the agent.

Each function scores a single agent run against gold labels and returns a value
in ``[0, 1]`` where higher is better. The evaluator aggregates these across a
dataset. Metrics are deliberately simple and deterministic so the evaluation
itself needs no LLM and is reproducible.
"""

from __future__ import annotations

import re
from typing import Any

from agent.orchestration.schemas import AgentTrace

_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords present in the answer.

    Parameters
    ----------
    answer : str
        The agent's answer.
    expected_keywords : list of str
        Keywords expected to appear in a correct answer.

    Returns
    -------
    float
        Coverage in ``[0, 1]``; ``1.0`` if no keywords are expected.
    """
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
    return hits / len(expected_keywords)


def retrieval_hit(
    context_chunks: list[dict[str, Any]], relevant_sources: list[str]
) -> float:
    """Whether retrieval surfaced at least one relevant source.

    Parameters
    ----------
    context_chunks : list of dict
        Context blocks retrieved during the run.
    relevant_sources : list of str
        Source file names considered relevant.

    Returns
    -------
    float
        ``1.0`` if any retrieved chunk's source matches a relevant source,
        otherwise ``0.0``; ``1.0`` if no relevant sources are specified.
    """
    if not relevant_sources:
        return 1.0
    relevant = {source.lower() for source in relevant_sources}
    for chunk in context_chunks:
        source = str(chunk.get("source", "")).lower()
        if any(rel in source or source in rel for rel in relevant):
            return 1.0
    return 0.0


def escalation_correct(escalated: bool, expected_escalation: bool) -> float:
    """Whether the escalate-or-answer decision matched the label.

    Parameters
    ----------
    escalated : bool
        Whether the run escalated.
    expected_escalation : bool
        Whether escalation was expected.

    Returns
    -------
    float
        ``1.0`` on a match, otherwise ``0.0``.
    """
    return 1.0 if escalated == expected_escalation else 0.0


def tool_selection_correct(trace: AgentTrace, expected_tool: str | None) -> float:
    """Whether the agent invoked the expected tool at least once.

    Parameters
    ----------
    trace : AgentTrace
        The run trace.
    expected_tool : str or None
        The tool expected to be used, or ``None`` if no expectation applies.

    Returns
    -------
    float
        ``1.0`` if the expected tool was used (or none was expected), otherwise
        ``0.0``.
    """
    if expected_tool is None:
        return 1.0
    used = {
        step.tool_call.tool for step in trace.steps if step.tool_call is not None
    }
    return 1.0 if expected_tool in used else 0.0


def citation_validity(answer: str, context_chunks: list[dict[str, Any]]) -> float:
    """Fraction of inline citations that match a retrieved chunk.

    This is a faithfulness proxy: it penalizes fabricated citations. Answers
    with no inline citations score ``1.0`` because some valid answers (for
    example structured policy-clause lookups) legitimately carry none.

    Parameters
    ----------
    answer : str
        The agent's answer.
    context_chunks : list of dict
        Context blocks retrieved during the run.

    Returns
    -------
    float
        Validity in ``[0, 1]``.
    """
    citations = _CITATION_PATTERN.findall(answer)
    if not citations:
        return 1.0
    valid_ids = {str(chunk.get("chunk_id", "")) for chunk in context_chunks}
    valid = sum(1 for citation in citations if citation in valid_ids)
    return valid / len(citations)
