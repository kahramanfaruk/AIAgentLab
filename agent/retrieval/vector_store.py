from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from agent.ingestion.embedder import EmbeddedChunk


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: str | Path = "data/vector_db",
        collection_name: str = "rag_documents",
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection_name = collection_name
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "AIAgentLab RAG document store"},
        )

    def add_embeddings(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        if not embedded_chunks:
            return 0

        ids = [chunk.chunk_id for chunk in embedded_chunks]
        documents = [chunk.content for chunk in embedded_chunks]
        embeddings = [chunk.embedding for chunk in embedded_chunks]
        metadatas = [self._sanitize_metadata(chunk.metadata, chunk.source) for chunk in embedded_chunks]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(embedded_chunks)

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "AIAgentLab RAG document store"},
        )

    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any], source: str) -> dict[str, Any]:
        clean = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                clean[key] = value
            else:
                clean[key] = str(value)
        clean["source"] = source
        return clean


if __name__ == "__main__":
    from agent.ingestion.chunker import chunk_documents
    from agent.ingestion.embedder import LocalEmbedder
    from agent.ingestion.loader import load_documents

    docs = load_documents()
    chunks = chunk_documents(docs)
    embedder = LocalEmbedder(device="cpu")
    embedded_chunks = embedder.embed_chunks(chunks)

    store = ChromaVectorStore()
    store.reset()
    inserted = store.add_embeddings(embedded_chunks)

    print(f"Inserted {inserted} chunks into Chroma.")
    print(f"Collection count: {store.count()}")