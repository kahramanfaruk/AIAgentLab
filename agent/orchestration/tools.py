"""Agent tools and the registry that exposes them to the planner.

Each tool declares a name, a description, and a JSON input schema so the planner
can decide when and how to call it. Tools return a :class:`ToolResult` that the
agent loop turns into an observation. A tool may request escalation, which the
loop treats as a human-in-the-loop handoff signal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agent.retrieval.retriever import Retriever


@dataclass(slots=True)
class ToolResult:
    """The outcome of a tool invocation.

    Attributes
    ----------
    output : str
        Human-readable result used as the agent's observation.
    data : dict
        Structured payload, for example retrieved context blocks.
    requests_escalation : bool
        Whether the tool asks the agent to hand off to a human.
    escalation_reason : str or None
        Reason supplied when ``requests_escalation`` is ``True``.
    """

    output: str
    data: dict[str, Any] = field(default_factory=dict)
    requests_escalation: bool = False
    escalation_reason: str | None = None


class Tool(ABC):
    """Base class for agent tools.

    Subclasses set :attr:`name`, :attr:`description`, and :attr:`input_schema`
    as class attributes and implement :meth:`run`.
    """

    name: str
    description: str
    input_schema: dict[str, Any]

    @abstractmethod
    def run(self, payload: dict[str, Any]) -> ToolResult:
        """Execute the tool.

        Parameters
        ----------
        payload : dict
            Tool arguments, validated against :attr:`input_schema` by the
            caller's planner contract.

        Returns
        -------
        ToolResult
            The tool's output.
        """
        raise NotImplementedError


class RetrieveDocumentsTool(Tool):
    """Retrieve grounding context from the document index.

    Parameters
    ----------
    retriever : Retriever
        The retriever used to fetch context.
    default_top_k : int, optional
        Number of chunks to retrieve when the payload omits ``top_k``.
    """

    name = "retrieve_documents"
    description = (
        "Search the indexed documents for passages relevant to a query and "
        "return them as grounding context. Use this before answering any "
        "question about document content."
    )
    input_schema = {
        "query": "string: the search query",
        "top_k": "integer (optional): number of passages to retrieve",
    }

    def __init__(self, retriever: Retriever, default_top_k: int = 5) -> None:
        self.retriever = retriever
        self.default_top_k = default_top_k

    def run(self, payload: dict[str, Any]) -> ToolResult:
        """Retrieve context for ``payload['query']``.

        Parameters
        ----------
        payload : dict
            Must contain ``query``; may contain ``top_k``.

        Returns
        -------
        ToolResult
            Output with a formatted context summary and the raw blocks under
            ``data['context_chunks']``.

        Raises
        ------
        ValueError
            If ``query`` is missing or empty.
        """
        query = str(payload.get("query", "")).strip()
        if not query:
            raise ValueError("retrieve_documents requires a non-empty 'query'.")

        top_k = int(payload.get("top_k", self.default_top_k))
        results = self.retriever.retrieve(query=query, top_k=top_k)

        context_chunks = [
            {
                "chunk_id": result.chunk_id,
                "page_number": result.metadata.get("page_number", "unknown"),
                "source": result.metadata.get("file_name", "unknown"),
                "content": result.content,
                "distance": result.distance,
            }
            for result in results
        ]

        if not context_chunks:
            return ToolResult(
                output="No relevant passages were found.",
                data={"context_chunks": []},
            )

        formatted = "\n".join(
            f"[{block['chunk_id']}] (page {block['page_number']}): "
            f"{block['content'][:300]}"
            for block in context_chunks
        )
        return ToolResult(
            output=formatted,
            data={"context_chunks": context_chunks},
        )


class LookupPolicyClauseTool(Tool):
    """Look up a clause in a structured insurance policy reference.

    This demonstrates a deterministic domain tool with structured data, as
    distinct from free-text retrieval. The reference is injected so it can be
    backed by a database in production.

    Parameters
    ----------
    clauses : dict
        Mapping of clause key to clause text.
    """

    name = "lookup_policy_clause"
    description = (
        "Look up the exact text of an insurance policy clause by topic, for "
        "example 'cancellation', 'liability', 'deductible', or 'coverage'. Use "
        "this for questions about policy terms rather than document passages."
    )
    input_schema = {"topic": "string: the clause topic to look up"}

    def __init__(self, clauses: dict[str, str]) -> None:
        self.clauses = {key.lower(): value for key, value in clauses.items()}

    def run(self, payload: dict[str, Any]) -> ToolResult:
        """Return the clause matching ``payload['topic']``.

        Parameters
        ----------
        payload : dict
            Must contain ``topic``.

        Returns
        -------
        ToolResult
            The matched clause text, or a not-found message.

        Raises
        ------
        ValueError
            If ``topic`` is missing or empty.
        """
        topic = str(payload.get("topic", "")).strip().lower()
        if not topic:
            raise ValueError("lookup_policy_clause requires a non-empty 'topic'.")

        for key, text in self.clauses.items():
            if key in topic or topic in key:
                return ToolResult(
                    output=text,
                    data={"clause_key": key, "clause_text": text},
                )

        available = ", ".join(sorted(self.clauses)) or "none configured"
        return ToolResult(
            output=(
                f"No clause found for topic '{topic}'. "
                f"Available topics: {available}."
            ),
            data={"clause_key": None},
        )


class EscalateToHumanTool(Tool):
    """Request a human-in-the-loop handoff."""

    name = "escalate_to_human"
    description = (
        "Hand the request to a human when it requires judgment beyond the "
        "available tools, when information is missing, or when acting could "
        "carry material risk. Provide a clear reason."
    )
    input_schema = {"reason": "string: why a human is needed"}

    def run(self, payload: dict[str, Any]) -> ToolResult:
        """Signal that the agent should escalate.

        Parameters
        ----------
        payload : dict
            May contain ``reason``.

        Returns
        -------
        ToolResult
            A result flagged with ``requests_escalation=True``.
        """
        reason = str(payload.get("reason", "unspecified")).strip() or "unspecified"
        return ToolResult(
            output=f"Escalation requested: {reason}",
            requests_escalation=True,
            escalation_reason=reason,
        )


class ToolRegistry:
    """A name-indexed collection of tools.

    Parameters
    ----------
    tools : list of Tool
        The tools to register.

    Raises
    ------
    ValueError
        If two tools share a name.
    """

    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name: {tool.name!r}")
            self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return the tool registered under ``name``.

        Parameters
        ----------
        name : str
            Tool name.

        Returns
        -------
        Tool
            The registered tool.

        Raises
        ------
        KeyError
            If no tool is registered under ``name``.
        """
        if name not in self._tools:
            available = ", ".join(sorted(self._tools)) or "none"
            raise KeyError(
                f"Unknown tool: {name!r}. Available tools: {available}."
            )
        return self._tools[name]

    def list_tools(self) -> list[Tool]:
        """Return all registered tools.

        Returns
        -------
        list of Tool
            Registered tools in insertion order.
        """
        return list(self._tools.values())

    def describe(self) -> str:
        """Render tool descriptions for inclusion in the planner prompt.

        Returns
        -------
        str
            One block per tool with its name, description, and input schema.
        """
        blocks = []
        for tool in self._tools.values():
            schema = ", ".join(
                f"{key} ({value})" for key, value in tool.input_schema.items()
            )
            blocks.append(
                f"- {tool.name}: {tool.description}\n  input: {{{schema}}}"
            )
        return "\n".join(blocks)
