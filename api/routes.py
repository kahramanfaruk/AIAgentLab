"""API route handlers."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status

from agent.ingestion.pipeline import index_document_bytes
from agent.orchestration.schemas import AgentResult
from api.context import AppContext
from api.schemas import (
    AgentRequest,
    AgentResponse,
    AskRequest,
    AskResponse,
    ContextChunk,
    DocumentsResponse,
    EscalationModel,
    EscalationsResponse,
    HealthResponse,
    IngestResponse,
    TraceStep,
)

router = APIRouter()


def get_context(request: Request) -> AppContext:
    """Return the application context stored on the app state.

    Parameters
    ----------
    request : Request
        The incoming request.

    Returns
    -------
    AppContext
        The wired application context.
    """
    return request.app.state.context


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service health.

    Returns
    -------
    HealthResponse
        Always ``status="ok"`` when the service is reachable.
    """
    return HealthResponse()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile, context: AppContext = Depends(get_context)
) -> IngestResponse:
    """Store and index an uploaded document.

    Parameters
    ----------
    file : UploadFile
        The uploaded document.
    context : AppContext
        The application context.

    Returns
    -------
    IngestResponse
        Counts of parsed units and indexed chunks.

    Raises
    ------
    HTTPException
        If the upload has no file name.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a name.",
        )

    data = await file.read()
    context.document_store.save(file.filename, data)
    summary = index_document_bytes(
        name=file.filename,
        data=data,
        embedder=context.embedder,
        vector_store=context.vector_store,
        chunk_size=context.settings.chunk_size,
        chunk_overlap=context.settings.chunk_overlap,
    )
    return IngestResponse(**asdict(summary))


@router.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest, context: AppContext = Depends(get_context)
) -> AskResponse:
    """Answer a question with retrieval-augmented generation.

    Parameters
    ----------
    payload : AskRequest
        The question and retrieval depth.
    context : AppContext
        The application context.

    Returns
    -------
    AskResponse
        The grounded answer and the context used.

    Raises
    ------
    HTTPException
        If the LLM backend is unavailable.
    """
    if context.rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM backend unavailable: {context.llm_error}",
        )

    result = context.rag_chain.ask(payload.question, top_k=payload.top_k)
    return AskResponse(
        question=result.question,
        answer=result.answer,
        context_chunks=[
            ContextChunk(**chunk) for chunk in result.context_chunks
        ],
    )


@router.post("/agent", response_model=AgentResponse)
def run_agent(
    payload: AgentRequest, context: AppContext = Depends(get_context)
) -> AgentResponse:
    """Run the autonomous agent on a question.

    Parameters
    ----------
    payload : AgentRequest
        The question and session identifier.
    context : AppContext
        The application context.

    Returns
    -------
    AgentResponse
        The answer or handoff, the trace, and any escalation.

    Raises
    ------
    HTTPException
        If the LLM backend is unavailable.
    """
    if context.agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM backend unavailable: {context.llm_error}",
        )

    result = context.agent.run(payload.question, session_id=payload.session_id)
    return _to_agent_response(result)


@router.get("/documents", response_model=DocumentsResponse)
def documents(context: AppContext = Depends(get_context)) -> DocumentsResponse:
    """List stored document names.

    Parameters
    ----------
    context : AppContext
        The application context.

    Returns
    -------
    DocumentsResponse
        The stored document names.
    """
    return DocumentsResponse(documents=context.document_store.list_documents())


@router.get("/escalations", response_model=EscalationsResponse)
def escalations(
    context: AppContext = Depends(get_context),
) -> EscalationsResponse:
    """List recorded human-in-the-loop escalations.

    Parameters
    ----------
    context : AppContext
        The application context.

    Returns
    -------
    EscalationsResponse
        The escalation payloads, newest first.
    """
    return EscalationsResponse(escalations=context.memory.list_escalations())


def _to_agent_response(result: AgentResult) -> AgentResponse:
    """Map an :class:`AgentResult` to its API response model.

    Parameters
    ----------
    result : AgentResult
        The agent run outcome.

    Returns
    -------
    AgentResponse
        The serialized response.
    """
    trace = [
        TraceStep(
            index=step.index,
            thought=step.thought,
            tool=step.tool_call.tool if step.tool_call else None,
            tool_input=step.tool_call.tool_input if step.tool_call else None,
            output=step.tool_call.output if step.tool_call else None,
        )
        for step in result.trace.steps
    ]

    escalation = (
        EscalationModel(**asdict(result.escalation))
        if result.escalation is not None
        else None
    )

    return AgentResponse(
        question=result.question,
        answer=result.answer,
        confidence=result.confidence,
        escalated=result.escalated,
        escalation=escalation,
        trace=trace,
        context_chunks=[
            ContextChunk(**chunk) for chunk in result.context_chunks
        ],
    )
