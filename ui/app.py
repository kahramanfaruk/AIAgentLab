"""Streamlit UI for AIAgentLab.

The UI supports two modes over the same wired backends: plain retrieval-augmented
question answering, and the autonomous agent (which plans, uses tools, and may
escalate to a human). Backends are selected by configuration, so the UI runs on
the free local stack or against AWS without code changes.
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from agent.ingestion.pipeline import index_document_bytes
from api.context import AppContext, build_context
from config import get_settings
from ui.components import render_agent_trace, render_escalation, render_sources

load_dotenv()

st.set_page_config(
    page_title="AIAgentLab - Agentic RAG",
    page_icon="brain",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_context() -> AppContext:
    """Build and cache the application context.

    Returns
    -------
    AppContext
        The wired backends shared across the session.
    """
    return build_context(get_settings())


def ingest_uploaded_file(context: AppContext, uploaded_file) -> dict:
    """Store and index an uploaded file.

    Parameters
    ----------
    context : AppContext
        The application context.
    uploaded_file : Any
        The Streamlit uploaded-file object.

    Returns
    -------
    dict
        Ingestion summary with parsed and indexed counts.
    """
    data = uploaded_file.getvalue()
    context.document_store.save(uploaded_file.name, data)
    summary = index_document_bytes(
        name=uploaded_file.name,
        data=data,
        embedder=context.embedder,
        vector_store=context.vector_store,
        chunk_size=context.settings.chunk_size,
        chunk_overlap=context.settings.chunk_overlap,
    )
    return {
        "document": summary.document,
        "documents_parsed": summary.documents_parsed,
        "chunks_indexed": summary.chunks_indexed,
    }


def reset_chat() -> None:
    """Clear the chat transcript in session state."""
    st.session_state.messages = []


context = get_context()
settings = context.settings

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi, I am your agentic RAG assistant. Upload a document, ingest "
                "it, choose a mode, and ask a question."
            ),
        }
    ]

st.title("AIAgentLab")
st.caption("Agentic retrieval-augmented generation over your documents")

with st.sidebar:
    st.header("Configuration")
    st.caption(
        f"LLM: {settings.llm_provider} | embeddings: {settings.embedding_provider} "
        f"| vectors: {settings.vector_backend} | storage: {settings.storage_backend} "
        f"| memory: {settings.memory_backend}"
    )
    if context.llm_error:
        st.error(f"LLM backend unavailable: {context.llm_error}")

    mode = st.radio("Mode", options=["Agent", "RAG"], horizontal=True)

    st.divider()
    st.header("Document setup")
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "docx"],
        help="Upload a text-based document for retrieval and question answering.",
    )
    if uploaded_file is not None and st.button(
        "Ingest document", use_container_width=True
    ):
        with st.spinner("Loading, chunking, embedding, and indexing..."):
            result = ingest_uploaded_file(context, uploaded_file)
        st.success(
            f"Indexed {result['chunks_indexed']} chunks from "
            f"{result['document']}."
        )

    if st.button("Reset chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.divider()
    st.subheader("Stored documents")
    stored = context.document_store.list_documents()
    if stored:
        for name in stored:
            st.write(f"- {name}")
    else:
        st.caption("No documents stored yet.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if mode == "Agent":
            if context.agent is None:
                answer = f"Agent unavailable: {context.llm_error}"
                st.error(answer)
            else:
                with st.spinner("Planning, using tools, and composing an answer..."):
                    result = context.agent.run(prompt)
                answer = result.answer
                st.markdown(answer)
                render_escalation(result)
                render_agent_trace(result)
                render_sources(result.context_chunks)
        else:
            if context.rag_chain is None:
                answer = f"RAG unavailable: {context.llm_error}"
                st.error(answer)
            else:
                with st.spinner("Retrieving context and generating an answer..."):
                    rag_result = context.rag_chain.ask(
                        prompt, top_k=settings.top_k_results
                    )
                answer = rag_result.answer
                st.markdown(answer)
                render_sources(rag_result.context_chunks)

    st.session_state.messages.append({"role": "assistant", "content": answer})
