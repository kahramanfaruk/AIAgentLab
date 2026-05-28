"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.settings import Settings


def test_defaults_select_local_stack() -> None:
    settings = Settings()
    assert settings.llm_provider == "groq"
    assert settings.embedding_provider == "local"
    assert settings.vector_backend == "chroma"
    assert settings.storage_backend == "local"
    assert settings.memory_backend == "memory"


def test_chunk_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValidationError):
        Settings(chunk_size=100, chunk_overlap=100)


def test_confidence_threshold_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Settings(escalation_confidence_threshold=1.5)


def test_invalid_backend_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(vector_backend="pinecone")
