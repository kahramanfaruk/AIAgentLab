"""FastAPI application entry point.

Builds the shared application context once at startup and exposes the ingestion,
question-answering, agent, document, and escalation endpoints.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.context import build_context
from api.routes import router
from config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the application context on startup.

    Parameters
    ----------
    app : FastAPI
        The application instance.

    Yields
    ------
    None
        Control returns to the server while the context lives on app state.
    """
    app.state.context = build_context(get_settings())
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns
    -------
    FastAPI
        The configured application.
    """
    app = FastAPI(
        title="AIAgentLab API",
        version="0.1.0",
        summary="Agentic RAG over documents with AWS-pluggable backends.",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
