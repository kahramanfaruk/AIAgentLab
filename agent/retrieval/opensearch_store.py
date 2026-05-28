"""OpenSearch vector store backend.

OpenSearch is the AWS-aligned vector backend. For budget reasons it is run
locally via the open-source OpenSearch Docker image during development and
testing; the same client code targets a managed AWS OpenSearch domain when one
is provisioned through the gated Terraform module. Managed OpenSearch has a
standing cost, so it is opt-in rather than always on.

This adapter uses k-NN over a ``knn_vector`` field with cosine similarity and
implements the :class:`~agent.retrieval.base.VectorStore` protocol.

``opensearch-py`` is imported lazily so the local Chroma path does not require
it. Authentication here covers the local basic-auth or no-auth case; a managed
AWS domain additionally requires SigV4 request signing configured separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent.ingestion.embedder import EmbeddedChunk
from agent.retrieval.base import VectorMatch

if TYPE_CHECKING:
    from config.settings import Settings


class OpenSearchVectorStore:
    """Vector store backed by an OpenSearch k-NN index.

    Parameters
    ----------
    host : str
        OpenSearch host name.
    port : int
        OpenSearch port.
    index : str
        Index name used for the document store.
    embedding_dim : int
        Dimensionality of the stored vectors; must match the active embedder.
    use_ssl : bool, optional
        Whether to connect over HTTPS.
    user : str or None, optional
        Basic-auth user name, if the cluster requires authentication.
    password : str or None, optional
        Basic-auth password.

    Raises
    ------
    RuntimeError
        If ``opensearch-py`` is not installed.
    """

    def __init__(
        self,
        host: str,
        port: int,
        index: str,
        embedding_dim: int,
        use_ssl: bool = False,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        try:
            from opensearchpy import OpenSearch
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "opensearch-py is required for the OpenSearch backend. Install "
                "it with 'pip install \"aiagentlab-rag[aws]\"'."
            ) from exc

        self.index = index
        self.embedding_dim = embedding_dim

        http_auth = (user, password) if user and password else None
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=http_auth,
            use_ssl=use_ssl,
            verify_certs=use_ssl,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenSearchVectorStore:
        """Build a store from application settings.

        Parameters
        ----------
        settings : Settings
            Application configuration.

        Returns
        -------
        OpenSearchVectorStore
            A configured store.
        """
        return cls(
            host=settings.opensearch_host,
            port=settings.opensearch_port,
            index=settings.opensearch_index,
            embedding_dim=settings.embedding_dim,
            use_ssl=settings.opensearch_use_ssl,
            user=settings.opensearch_user,
            password=settings.opensearch_password,
        )

    def add_embeddings(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """Index embedded chunks via the bulk API.

        Parameters
        ----------
        embedded_chunks : list of EmbeddedChunk
            Chunks with precomputed embeddings.

        Returns
        -------
        int
            The number of chunks indexed.
        """
        if not embedded_chunks:
            return 0

        from opensearchpy import helpers

        self._ensure_index()
        actions = [
            {
                "_index": self.index,
                "_id": chunk.chunk_id,
                "_source": {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "source": chunk.source,
                    "embedding": chunk.embedding,
                    "metadata": dict(chunk.metadata),
                },
            }
            for chunk in embedded_chunks
        ]
        helpers.bulk(self.client, actions, refresh=True)
        return len(embedded_chunks)

    def search(
        self, query_embedding: list[float], top_k: int
    ) -> list[VectorMatch]:
        """Return the ``top_k`` nearest neighbours to ``query_embedding``.

        The reported ``distance`` is the negated relevance score, so a smaller
        value still means a closer match, matching the protocol contract.

        Parameters
        ----------
        query_embedding : list of float
            The query vector.
        top_k : int
            Maximum number of matches to return.

        Returns
        -------
        list of VectorMatch
            Matches ordered from most to least similar.
        """
        if not self.client.indices.exists(index=self.index):
            return []

        body = {
            "size": top_k,
            "query": {
                "knn": {"embedding": {"vector": query_embedding, "k": top_k}}
            },
        }
        response = self.client.search(index=self.index, body=body)
        hits = response.get("hits", {}).get("hits", [])

        matches: list[VectorMatch] = []
        for hit in hits:
            source = hit.get("_source", {})
            matches.append(
                VectorMatch(
                    chunk_id=source.get("chunk_id", hit.get("_id", "")),
                    content=source.get("content", ""),
                    metadata=source.get("metadata") or {},
                    distance=-float(hit.get("_score", 0.0)),
                )
            )
        return matches

    def reset(self) -> None:
        """Delete the index and recreate it empty."""
        if self.client.indices.exists(index=self.index):
            self.client.indices.delete(index=self.index)
        self._ensure_index()

    def count(self) -> int:
        """Return the number of indexed documents.

        Returns
        -------
        int
            The document count, or ``0`` if the index does not exist yet.
        """
        if not self.client.indices.exists(index=self.index):
            return 0
        return int(self.client.count(index=self.index).get("count", 0))

    def _ensure_index(self) -> None:
        """Create the k-NN index with a cosine-similarity mapping if missing."""
        if self.client.indices.exists(index=self.index):
            return

        mapping: dict[str, Any] = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "content": {"type": "text"},
                    "source": {"type": "keyword"},
                    "metadata": {"type": "object", "enabled": False},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",
                        },
                    },
                }
            },
        }
        self.client.indices.create(index=self.index, body=mapping)
