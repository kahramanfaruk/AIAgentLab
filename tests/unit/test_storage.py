"""Tests for the document storage backends."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.storage.local_store import LocalDocumentStore
from agent.storage.s3_store import S3DocumentStore


def test_local_store_roundtrip(tmp_path: Path) -> None:
    store = LocalDocumentStore(base_dir=tmp_path)
    store.save("a.txt", b"hello")
    assert store.exists("a.txt") is True
    assert store.exists("missing.txt") is False
    assert store.load("a.txt") == b"hello"
    assert store.list_documents() == ["a.txt"]


def test_local_store_load_missing_raises(tmp_path: Path) -> None:
    store = LocalDocumentStore(base_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load("missing.txt")


def test_local_store_rejects_path_traversal(tmp_path: Path) -> None:
    store = LocalDocumentStore(base_dir=tmp_path)
    with pytest.raises(ValueError):
        store.save("../escape.txt", b"x")


def test_s3_store_roundtrip(s3_bucket: str) -> None:
    store = S3DocumentStore(bucket=s3_bucket, region="eu-central-1")
    uri = store.save("a.txt", b"hello")
    assert uri == f"s3://{s3_bucket}/documents/a.txt"
    assert store.exists("a.txt") is True
    assert store.exists("missing.txt") is False
    assert store.load("a.txt") == b"hello"
    assert store.list_documents() == ["a.txt"]


def test_s3_store_load_missing_raises(s3_bucket: str) -> None:
    store = S3DocumentStore(bucket=s3_bucket, region="eu-central-1")
    with pytest.raises(FileNotFoundError):
        store.load("missing.txt")
