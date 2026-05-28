"""Orchestration package: the agent, its tools, guardrails, and the factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.ingestion import build_embedder
from agent.memory import build_memory
from agent.orchestration.agent import Agent
from agent.orchestration.guardrails import Guardrails
from agent.orchestration.schemas import (
    AgentResult,
    AgentStep,
    AgentTrace,
    Escalation,
    ToolCall,
)
from agent.orchestration.tools import (
    EscalateToHumanTool,
    LookupPolicyClauseTool,
    RetrieveDocumentsTool,
    Tool,
    ToolRegistry,
    ToolResult,
)
from agent.retrieval import build_vector_store
from agent.retrieval.retriever import Retriever

if TYPE_CHECKING:
    from config.settings import Settings

__all__ = [
    "Agent",
    "Guardrails",
    "AgentResult",
    "AgentStep",
    "AgentTrace",
    "Escalation",
    "ToolCall",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "RetrieveDocumentsTool",
    "LookupPolicyClauseTool",
    "EscalateToHumanTool",
    "DEFAULT_POLICY_CLAUSES",
    "DEFAULT_BLOCKED_TOPICS",
    "build_agent",
]

# Synthetic reference clauses for the demo domain tool. In production this would
# be backed by a policy database; the structure is intentionally simple.
DEFAULT_POLICY_CLAUSES: dict[str, str] = {
    "cancellation": (
        "Cancellation: The policyholder may cancel within 14 days of inception "
        "for a full refund. After that, cancellation takes effect at the end of "
        "the current billing period."
    ),
    "liability": (
        "Liability: The insurer covers third-party bodily injury and property "
        "damage up to the limit stated in the schedule, subject to the "
        "deductible."
    ),
    "deductible": (
        "Deductible: The policyholder pays the first 250 EUR of each claim; the "
        "insurer pays the remainder up to the coverage limit."
    ),
    "coverage": (
        "Coverage: This policy covers fire, theft, water damage, and storm "
        "damage. Flooding and wear and tear are excluded."
    ),
}

# Conservative default block list; tuned to avoid false positives on normal
# insurance questions while still demonstrating the topic guardrail.
DEFAULT_BLOCKED_TOPICS: list[str] = ["self-harm", "suicide"]


def build_agent(settings: Settings) -> Agent:
    """Construct a fully wired agent from application settings.

    The agent uses the configured LLM, vector, embedding, and memory backends,
    the default tool set (document retrieval, policy-clause lookup, and
    escalation), and the rule-based guardrails.

    Parameters
    ----------
    settings : Settings
        Application configuration.

    Returns
    -------
    Agent
        A ready-to-run agent.
    """
    from agent.generation import build_llm_client

    llm_client = build_llm_client(settings)
    vector_store = build_vector_store(settings)
    embedder = build_embedder(settings)
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    memory = build_memory(settings)

    tools = ToolRegistry(
        [
            RetrieveDocumentsTool(
                retriever=retriever, default_top_k=settings.top_k_results
            ),
            LookupPolicyClauseTool(clauses=DEFAULT_POLICY_CLAUSES),
            EscalateToHumanTool(),
        ]
    )
    guardrails = Guardrails(blocked_topics=DEFAULT_BLOCKED_TOPICS)

    return Agent(
        llm_client=llm_client,
        tools=tools,
        guardrails=guardrails,
        memory=memory,
        max_steps=settings.agent_max_steps,
        confidence_threshold=settings.escalation_confidence_threshold,
    )
