from __future__ import annotations

import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent.generation.chain import RAGChain
from agent.generation.llm_client import GroqClient
from agent.ingestion.chunker import chunk_documents
from agent.ingestion.embedder import LocalEmbedder
from agent.ingestion.loader import load_documents
from agent.retrieval.retriever import Retriever
from agent.retrieval.vector_store import ChromaVectorStore


load_dotenv()

st.set_page_config(
    page_title="AIAgentLab - RAG Q&A Agent",
    page_icon="🧠",
    layout="wide",
)

DATA_RAW_DIR = Path("data/raw")
VECTOR_DB_DIR = Path("data/vector_db")
COLLECTION_NAME = "rag_documents"


@st.cache_resource(show_spinner=False)
def get_embedder() -> LocalEmbedder:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return LocalEmbedder(device="cpu")


@st.cache_resource(show_spinner=False)
def get_rag_chain() -> RAGChain:
    embedder = get_embedder()
    vector_store = ChromaVectorStore(
        persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME,
    )
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    llm_client = GroqClient()
    return RAGChain(retriever=retriever, llm_client=llm_client)


def ingest_documents() -> tuple[int, int]:
    docs = load_documents(DATA_RAW_DIR)
    chunks = chunk_documents(docs)
    embedder = get_embedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    vector_store = ChromaVectorStore(
        persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME,
    )
    vector_store.reset()
    inserted = vector_store.add_embeddings(embedded_chunks)
    return len(docs), inserted


def save_uploaded_file(uploaded_file) -> Path:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    save_path = DATA_RAW_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.last_sources = []


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi — I’m your RAG Q&A assistant. Upload a PDF, ingest it, "
                "and ask questions grounded in the document."
            ),
        }
    ]

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


st.title("🧠 AIAgentLab")
st.caption("RAG Q&A Agent over your own PDF documents")

with st.sidebar:
    st.header("Document setup")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload a text-based PDF for retrieval and question answering.",
    )

    if uploaded_file is not None:
        saved_path = save_uploaded_file(uploaded_file)
        st.success(f"Saved: {saved_path.name}")

    if st.button("Ingest documents", use_container_width=True):
        with st.spinner("Loading, chunking, embedding, and indexing documents..."):
            doc_count, chunk_count = ingest_documents()
        st.success(f"Ingested {doc_count} document pages into {chunk_count} chunks.")

    if st.button("Reset chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.divider()
    st.subheader("Current files")

    files = sorted(DATA_RAW_DIR.glob("*"))
    if files:
        for file in files:
            st.write(f"- {file.name}")
    else:
        st.caption("No files uploaded yet.")

    st.divider()
    st.subheader("Tips")
    st.caption("Ask focused questions for better retrieval.")
    st.caption("Example: What does the paper claim about sigmoid gating after SDPA?")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            rag_chain = get_rag_chain()
            result = rag_chain.ask(prompt, top_k=6)

            st.markdown(result.answer)

            if result.context_chunks:
                with st.expander("Retrieved sources"):
                    for idx, chunk in enumerate(result.context_chunks, start=1):
                        st.markdown(
                            f"**{idx}. {chunk['chunk_id']}**  \n"
                            f"Source: {chunk['source']}  \n"
                            f"Page: {chunk['page_number']}  \n"
                            f"Distance: {chunk['distance']}"
                        )
                        st.code(chunk["content"][:1200])

    st.session_state.messages.append({"role": "assistant", "content": result.answer})