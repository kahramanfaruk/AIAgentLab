"""Pydantic request and response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "ok"


class ContextChunk(BaseModel):
    """A retrieved context block returned to the client."""

    chunk_id: str
    source: str
    page_number: str | int
    content: str
    distance: float | None = None


class AskRequest(BaseModel):
    """Request body for plain retrieval-augmented question answering."""

    question: str = Field(min_length=1)
    top_k: int = Field(default=6, gt=0, le=50)


class AskResponse(BaseModel):
    """Response for a RAG question."""

    question: str
    answer: str
    context_chunks: list[ContextChunk]


class AgentRequest(BaseModel):
    """Request body for an autonomous agent run."""

    question: str = Field(min_length=1)
    session_id: str = "default"


class TraceStep(BaseModel):
    """A single step in the agent trace."""

    index: int
    thought: str
    tool: str | None = None
    tool_input: dict[str, Any] | None = None
    output: str | None = None


class EscalationModel(BaseModel):
    """An escalation record returned to the client."""

    reason: str
    question: str
    confidence: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Response for an agent run."""

    question: str
    answer: str
    confidence: float
    escalated: bool
    escalation: EscalationModel | None = None
    trace: list[TraceStep]
    context_chunks: list[ContextChunk]


class IngestResponse(BaseModel):
    """Response after ingesting a document."""

    document: str
    documents_parsed: int
    chunks_indexed: int


class DocumentsResponse(BaseModel):
    """List of stored document names."""

    documents: list[str]


class EscalationsResponse(BaseModel):
    """List of recorded escalations."""

    escalations: list[dict[str, Any]]
