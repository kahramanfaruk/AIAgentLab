"""Shared document-indexing pipeline.

This is the single place that turns raw document bytes into indexed vectors. It
is reused by the API ingest endpoint and the Lambda ingestion handler so both
paths stay consistent.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from agent.ingestion.base import Embedder
from agent.ingestion.chunker import chunk_documents
from agent.ingestion.loader import load_documents
from agent.retrieval.base import VectorStore


@dataclass(slots=True)
class IngestionSummary:
    """Result of indexing one document.

    Attributes
    ----------
    document : str
        The document name.
    documents_parsed : int
        Number of document units parsed (PDF pages count individually).
    chunks_indexed : int
        Number of chunks written to the vector store.
    """

    document: str
    documents_parsed: int
    chunks_indexed: int


def index_document_bytes(
    name: str,
    data: bytes,
    embedder: Embedder,
    vector_store: VectorStore,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestionSummary:
    """Parse, chunk, embed, and index a document given as raw bytes.

    The bytes are written to a temporary file so the existing loaders (which
    operate on paths) can parse them by extension.

    Parameters
    ----------
    name : str
        Document file name; its extension selects the loader.
    data : bytes
        Raw document content.
    embedder : Embedder
        Embedding backend.
    vector_store : VectorStore
        Vector store backend.
    chunk_size : int
        Maximum chunk size in characters.
    chunk_overlap : int
        Overlap between consecutive chunks in characters.

    Returns
    -------
    IngestionSummary
        Counts of parsed units and indexed chunks.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        (Path(tmp_dir) / name).write_bytes(data)
        documents = load_documents(tmp_dir)
        chunks = chunk_documents(
            documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        embedded = embedder.embed_chunks(chunks)
        indexed = vector_store.add_embeddings(embedded)

    return IngestionSummary(
        document=name,
        documents_parsed=len(documents),
        chunks_indexed=indexed,
    )
