"""Application context: the wired components shared across API requests.

Components are built once at startup and shared, so the embedding model loads a
single time and the RAG chain and agent reuse the same retriever and memory. The
LLM client is built defensively: if no backend is configured (for example a
missing Groq key), the context still starts and the LLM-dependent endpoints
return a clear error instead of crashing the whole service.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.generation import build_llm_client
from agent.generation.base import LLMClient
from agent.generation.chain import RAGChain
from agent.ingestion import build_embedder
from agent.ingestion.base import Embedder
from agent.memory import build_memory
from agent.memory.chat_history import ConversationMemory
from agent.orchestration import (
    DEFAULT_BLOCKED_TOPICS,
    DEFAULT_POLICY_CLAUSES,
    Agent,
    EscalateToHumanTool,
    Guardrails,
    LookupPolicyClauseTool,
    RetrieveDocumentsTool,
    ToolRegistry,
)
from agent.retrieval import build_vector_store
from agent.retrieval.base import VectorStore
from agent.retrieval.retriever import Retriever
from agent.storage import build_document_store
from agent.storage.base import DocumentStore
from config.settings import Settings


@dataclass(slots=True)
class AppContext:
    """Container for the wired application components.

    Attributes
    ----------
    settings : Settings
        Active configuration.
    document_store : DocumentStore
        Backend for raw documents.
    embedder : Embedder
        Embedding backend.
    vector_store : VectorStore
        Vector store backend.
    retriever : Retriever
        Shared retriever.
    memory : ConversationMemory
        Shared memory and escalation store.
    rag_chain : RAGChain or None
        RAG chain, or ``None`` if the LLM backend is unavailable.
    agent : Agent or None
        Agent, or ``None`` if the LLM backend is unavailable.
    llm_error : str or None
        Message describing why the LLM backend is unavailable, if applicable.
    """

    settings: Settings
    document_store: DocumentStore
    embedder: Embedder
    vector_store: VectorStore
    retriever: Retriever
    memory: ConversationMemory
    rag_chain: RAGChain | None
    agent: Agent | None
    llm_error: str | None


def build_context(settings: Settings) -> AppContext:
    """Build the application context from settings.

    Parameters
    ----------
    settings : Settings
        Active configuration.

    Returns
    -------
    AppContext
        The wired components. LLM-dependent fields are ``None`` when the LLM
        backend cannot be constructed.
    """
    embedder = build_embedder(settings)
    vector_store = build_vector_store(settings)
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    memory = build_memory(settings)
    document_store = build_document_store(settings)

    rag_chain: RAGChain | None = None
    agent: Agent | None = None
    llm_error: str | None = None

    try:
        llm_client: LLMClient = build_llm_client(settings)
        rag_chain = RAGChain(retriever=retriever, llm_client=llm_client)
        agent = _build_agent(settings, llm_client, retriever, memory)
    except (ValueError, RuntimeError) as exc:
        llm_error = str(exc)

    return AppContext(
        settings=settings,
        document_store=document_store,
        embedder=embedder,
        vector_store=vector_store,
        retriever=retriever,
        memory=memory,
        rag_chain=rag_chain,
        agent=agent,
        llm_error=llm_error,
    )


def _build_agent(
    settings: Settings,
    llm_client: LLMClient,
    retriever: Retriever,
    memory: ConversationMemory,
) -> Agent:
    """Wire an agent that shares the given LLM, retriever, and memory.

    Parameters
    ----------
    settings : Settings
        Active configuration.
    llm_client : LLMClient
        Shared LLM backend.
    retriever : Retriever
        Shared retriever.
    memory : ConversationMemory
        Shared memory.

    Returns
    -------
    Agent
        The wired agent.
    """
    tools = ToolRegistry(
        [
            RetrieveDocumentsTool(
                retriever=retriever, default_top_k=settings.top_k_results
            ),
            LookupPolicyClauseTool(clauses=DEFAULT_POLICY_CLAUSES),
            EscalateToHumanTool(),
        ]
    )
    return Agent(
        llm_client=llm_client,
        tools=tools,
        guardrails=Guardrails(blocked_topics=DEFAULT_BLOCKED_TOPICS),
        memory=memory,
        max_steps=settings.agent_max_steps,
        confidence_threshold=settings.escalation_confidence_threshold,
    )
