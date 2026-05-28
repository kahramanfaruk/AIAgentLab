"""Centralized application configuration.

This module defines a single :class:`Settings` model that is the source of
truth for every runtime option in AIAgentLab. Backends are selected through
explicit switches (``llm_provider``, ``vector_backend``, ``storage_backend``,
``memory_backend``, ``embedding_provider``) so that the same code base runs on
a free local stack (Chroma, Groq, local filesystem, in-memory state, Sentence
Transformers) or on AWS (Bedrock, OpenSearch, S3, DynamoDB) without code
changes. All AWS clients honour ``aws_endpoint_url`` so the AWS path can target
LocalStack for zero-cost integration testing.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["groq", "bedrock"]
EmbeddingProvider = Literal["local", "bedrock"]
VectorBackend = Literal["chroma", "opensearch"]
StorageBackend = Literal["local", "s3"]
MemoryBackend = Literal["memory", "dynamodb"]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables and ``.env``.

    Every field has a default that keeps the application runnable on the free
    local stack, so importing this module never requires AWS credentials. AWS
    adapters validate their own required fields at construction time and raise
    explicit errors, rather than this model failing at import.

    Attributes
    ----------
    llm_provider : {"groq", "bedrock"}
        Which large language model backend to use for generation.
    embedding_provider : {"local", "bedrock"}
        Which embedding backend to use for indexing and querying.
    vector_backend : {"chroma", "opensearch"}
        Which vector store backend to use for retrieval.
    storage_backend : {"local", "s3"}
        Where raw documents are persisted.
    memory_backend : {"memory", "dynamodb"}
        Where conversation history, run traces, and escalations are persisted.
    aws_endpoint_url : str or None
        Override endpoint for all AWS clients. Set to ``http://localhost:4566``
        to target LocalStack; leave unset to use real AWS.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_env: str = "development"
    log_level: str = "INFO"

    # Backend selection
    llm_provider: LLMProvider = "groq"
    embedding_provider: EmbeddingProvider = "local"
    vector_backend: VectorBackend = "chroma"
    storage_backend: StorageBackend = "local"
    memory_backend: MemoryBackend = "memory"

    # LLM: Groq
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    # LLM: Bedrock (pay-per-token, zero idle cost)
    bedrock_model_id: str = "anthropic.claude-3-5-haiku-20241022-v1:0"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    bedrock_guardrail_id: str | None = None
    bedrock_guardrail_version: str = "DRAFT"

    # Generation parameters
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=800, gt=0)

    # Embeddings: local Sentence Transformers
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # Embedding dimensionality (must match the active embedding provider:
    # all-MiniLM-L6-v2 is 384, Titan Text Embeddings v2 defaults to 1024).
    embedding_dim: int = Field(default=384, gt=0)

    # Vector store: Chroma
    vector_db_path: Path = Path("data/vector_db")
    chroma_collection_name: str = "rag_documents"

    # Vector store: OpenSearch (open-source Docker image locally, gated IaC on AWS)
    opensearch_host: str = "localhost"
    opensearch_port: int = Field(default=9200, gt=0, lt=65536)
    opensearch_index: str = "rag_documents"
    opensearch_user: str | None = None
    opensearch_password: str | None = None
    opensearch_use_ssl: bool = False

    # Retrieval and chunking
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    top_k_results: int = Field(default=6, gt=0)

    # AWS general
    aws_region: str = "eu-central-1"
    aws_endpoint_url: str | None = None
    s3_bucket: str = "aiagentlab-documents"
    dynamodb_table: str = "aiagentlab-state"

    # Agent orchestration
    agent_max_steps: int = Field(default=6, gt=0)
    escalation_confidence_threshold: float = Field(default=0.45, ge=0.0, le=1.0)

    # Optional Hugging Face token for gated embedding models
    hf_token: str | None = None

    @model_validator(mode="after")
    def _validate_chunking(self) -> Settings:
        """Ensure chunk overlap is strictly smaller than chunk size.

        Returns
        -------
        Settings
            The validated settings instance.

        Raises
        ------
        ValueError
            If ``chunk_overlap`` is greater than or equal to ``chunk_size``,
            which would make chunking fail to advance.
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap "
                f"({self.chunk_overlap}) must be smaller than chunk_size "
                f"({self.chunk_size})."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    The instance is cached so that environment parsing happens once per
    process. Tests that need a modified configuration should construct
    :class:`Settings` directly with overrides rather than mutating the cache.

    Returns
    -------
    Settings
        The process-wide settings instance.
    """
    return Settings()
