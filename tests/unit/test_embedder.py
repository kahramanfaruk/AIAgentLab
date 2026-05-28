"""Tests for the embedding backends."""

from __future__ import annotations

import json

import pytest

from agent.ingestion.bedrock_embedder import BedrockEmbedder
from agent.ingestion.chunker import Chunk


class _FakeBody:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()


class _FakeRuntime:
    """A fake Bedrock runtime returning a fixed embedding."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector
        self.calls = 0

    def invoke_model(self, **_: object) -> dict:
        self.calls += 1
        return {"body": _FakeBody({"embedding": self._vector})}


def test_bedrock_embedder_embeds_query() -> None:
    embedder = BedrockEmbedder(model_id="m", region="eu-central-1", dimensions=3)
    embedder.client = _FakeRuntime([0.1, 0.2, 0.3])
    assert embedder.embed_query("hello") == [0.1, 0.2, 0.3]


def test_bedrock_embedder_embeds_chunks() -> None:
    embedder = BedrockEmbedder(model_id="m", region="eu-central-1", dimensions=3)
    embedder.client = _FakeRuntime([0.5, 0.5, 0.5])
    chunk = Chunk(chunk_id="c1", content="text", source="s", metadata={})
    embedded = embedder.embed_chunks([chunk])
    assert len(embedded) == 1
    assert embedded[0].embedding == [0.5, 0.5, 0.5]


def test_bedrock_embedder_empty_embedding_raises() -> None:
    embedder = BedrockEmbedder(model_id="m", region="eu-central-1", dimensions=3)
    embedder.client = _FakeRuntime([])
    with pytest.raises(RuntimeError):
        embedder.embed_query("hello")


def test_local_embedder_produces_vectors() -> None:
    # The model download requires network; skip when it is unavailable rather
    # than failing the suite in an offline environment.
    from agent.ingestion.embedder import LocalEmbedder

    try:
        embedder = LocalEmbedder(device="cpu")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"embedding model unavailable: {exc}")

    vector = embedder.embed_query("insurance policy deductible")
    assert isinstance(vector, list)
    assert len(vector) > 0
