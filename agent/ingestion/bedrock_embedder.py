"""Amazon Bedrock Titan embedding backend.

Titan Text Embeddings v2 is pay-per-token with no idle cost. This adapter is
the AWS counterpart to :class:`~agent.ingestion.embedder.LocalEmbedder` and
implements the same :class:`~agent.ingestion.base.Embedder` protocol.

``boto3`` is imported lazily so the local stack does not require AWS libraries.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agent.ingestion.chunker import Chunk
from agent.ingestion.embedder import EmbeddedChunk

if TYPE_CHECKING:
    from config.settings import Settings


class BedrockEmbedder:
    """Embedding client backed by Amazon Bedrock Titan.

    Parameters
    ----------
    model_id : str
        Bedrock embedding model id, for example
        ``"amazon.titan-embed-text-v2:0"``.
    region : str
        AWS region of the Bedrock endpoint.
    dimensions : int
        Output embedding dimensionality requested from Titan.
    endpoint_url : str or None, optional
        Override endpoint, for example a LocalStack URL.

    Raises
    ------
    RuntimeError
        If ``boto3`` is not installed.
    """

    def __init__(
        self,
        model_id: str,
        region: str,
        dimensions: int,
        endpoint_url: str | None = None,
    ) -> None:
        try:
            import boto3
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "boto3 is required for the Bedrock embedder. Install it with "
                "'pip install \"aiagentlab-rag[aws]\"'."
            ) from exc

        self.model_id = model_id
        self.dimensions = dimensions
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            endpoint_url=endpoint_url,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> BedrockEmbedder:
        """Build an embedder from application settings.

        Parameters
        ----------
        settings : Settings
            Application configuration.

        Returns
        -------
        BedrockEmbedder
            A configured embedder.
        """
        return cls(
            model_id=settings.bedrock_embedding_model_id,
            region=settings.aws_region,
            dimensions=settings.embedding_dim,
            endpoint_url=settings.aws_endpoint_url,
        )

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a batch of chunks.

        Titan embeds one input at a time, so chunks are embedded sequentially.

        Parameters
        ----------
        chunks : list of Chunk
            Chunks to embed.

        Returns
        -------
        list of EmbeddedChunk
            One embedded chunk per input chunk, preserving order.
        """
        if not chunks:
            return []

        embedded: list[EmbeddedChunk] = []
        for chunk in chunks:
            vector = self._invoke(chunk.content)
            embedded.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    source=chunk.source,
                    embedding=vector,
                    metadata=chunk.metadata,
                )
            )
        return embedded

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Parameters
        ----------
        query : str
            The query to embed.

        Returns
        -------
        list of float
            The query embedding vector.
        """
        return self._invoke(query)

    def _invoke(self, text: str) -> list[float]:
        """Invoke Titan for a single input and return its embedding.

        Parameters
        ----------
        text : str
            Input text to embed.

        Returns
        -------
        list of float
            The embedding vector.

        Raises
        ------
        RuntimeError
            If the Bedrock invocation fails or returns no embedding.
        """
        body: dict[str, Any] = {
            "inputText": text,
            "dimensions": self.dimensions,
            "normalize": True,
        }
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(response["body"].read())
        except Exception as exc:
            raise RuntimeError(
                f"Bedrock embedding failed for model '{self.model_id}': {exc}"
            ) from exc

        embedding = payload.get("embedding")
        if not embedding:
            raise RuntimeError("Bedrock returned an empty embedding.")
        return list(embedding)
