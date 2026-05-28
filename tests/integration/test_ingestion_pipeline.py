"""Integration test for the S3 -> Lambda -> vector ingestion flow.

Uses moto for S3 and DynamoDB and fake embedding/vector backends, so the whole
flow runs in-process with no AWS cost and no embedding-model download.
"""

from __future__ import annotations

import boto3
import pytest

import agent.serverless.ingestion_handler as handler_module
from agent.serverless.ingestion_handler import ingest_object
from config.settings import Settings
from tests.stubs import FakeEmbedder, FakeVectorStore

_REGION = "eu-central-1"


@pytest.fixture
def patched_backends(monkeypatch: pytest.MonkeyPatch) -> FakeVectorStore:
    """Patch the handler to use fake embedding and vector backends."""
    store = FakeVectorStore()
    monkeypatch.setattr(
        handler_module, "build_embedder", lambda settings: FakeEmbedder()
    )
    monkeypatch.setattr(
        handler_module, "build_vector_store", lambda settings: store
    )
    return store


def test_ingest_object_indexes_and_records_metadata(
    aws_backends: tuple[str, str], patched_backends: FakeVectorStore
) -> None:
    bucket, table = aws_backends
    key = "documents/policy.txt"
    boto3.client("s3", region_name=_REGION).put_object(
        Bucket=bucket,
        Key=key,
        Body=b"The deductible is 250 EUR per claim. " * 20,
    )

    settings = Settings(
        aws_region=_REGION, s3_bucket=bucket, dynamodb_table=table, chunk_size=200
    )
    summary = ingest_object(settings, bucket, key)

    assert summary["document"] == "policy.txt"
    assert summary["chunks_indexed"] > 0
    assert patched_backends.count() == summary["chunks_indexed"]

    item = boto3.client("dynamodb", region_name=_REGION).get_item(
        TableName=table, Key={"pk": {"S": "DOCUMENT"}, "sk": {"S": "policy.txt"}}
    )
    assert item["Item"]["name"]["S"] == "policy.txt"


def test_handler_parses_direct_event(
    aws_backends: tuple[str, str],
    patched_backends: FakeVectorStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket, table = aws_backends
    key = "documents/doc.txt"
    boto3.client("s3", region_name=_REGION).put_object(
        Bucket=bucket, Key=key, Body=b"content " * 50
    )

    settings = Settings(aws_region=_REGION, s3_bucket=bucket, dynamodb_table=table)
    monkeypatch.setattr(handler_module, "get_settings", lambda: settings)
    result = handler_module.handler({"bucket": bucket, "key": key})
    assert result["ingested"][0]["document"] == "doc.txt"
