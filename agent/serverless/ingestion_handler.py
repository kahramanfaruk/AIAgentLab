"""AWS Lambda handler for serverless document ingestion.

The handler is triggered by an S3 object-created event (or invoked directly with
``{"bucket": ..., "key": ...}``). For each object it downloads the document,
runs the ``load -> chunk -> embed -> index`` pipeline using the configured
backends, and records ingestion metadata in DynamoDB.

In production the Lambda is configured with ``embedding_provider=bedrock`` and
``vector_backend=opensearch`` (no heavy local model in the package). The same
code runs in tests against moto or LocalStack with the local embedder and Chroma
backend, so the full flow is exercised without AWS inference cost.

The importable core is :func:`ingest_object`; :func:`handler` only parses the
event. The package directory is ``serverless`` rather than ``lambda`` because
``lambda`` is a reserved word and cannot be imported.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus

from agent.ingestion import build_embedder
from agent.ingestion.pipeline import index_document_bytes
from agent.retrieval import build_vector_store
from config import get_settings

if TYPE_CHECKING:
    from config.settings import Settings


def _s3_client(settings: Settings) -> Any:
    """Create an S3 client honouring the configured endpoint override.

    Parameters
    ----------
    settings : Settings
        Application configuration.

    Returns
    -------
    Any
        A boto3 S3 client.
    """
    import boto3

    return boto3.client(
        "s3",
        region_name=settings.aws_region,
        endpoint_url=settings.aws_endpoint_url,
    )


def _record_metadata(settings: Settings, name: str, chunk_count: int) -> None:
    """Write an ingestion-metadata item to DynamoDB.

    Parameters
    ----------
    settings : Settings
        Application configuration.
    name : str
        Document name.
    chunk_count : int
        Number of chunks indexed for the document.
    """
    import boto3

    resource = boto3.resource(
        "dynamodb",
        region_name=settings.aws_region,
        endpoint_url=settings.aws_endpoint_url,
    )
    resource.Table(settings.dynamodb_table).put_item(
        Item={
            "pk": "DOCUMENT",
            "sk": name,
            "name": name,
            "chunk_count": chunk_count,
            "indexed_at": datetime.now(UTC).isoformat(),
        }
    )


def ingest_object(settings: Settings, bucket: str, key: str) -> dict[str, Any]:
    """Ingest a single S3 object into the vector index.

    Parameters
    ----------
    settings : Settings
        Application configuration.
    bucket : str
        Source S3 bucket.
    key : str
        Source object key.

    Returns
    -------
    dict
        Summary with the document name, pages parsed, and chunks indexed.

    Raises
    ------
    RuntimeError
        If the object cannot be read from S3.
    """
    client = _s3_client(settings)
    try:
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read s3://{bucket}/{key}: {exc}"
        ) from exc

    name = Path(key).name
    summary = index_document_bytes(
        name=name,
        data=body,
        embedder=build_embedder(settings),
        vector_store=build_vector_store(settings),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    _record_metadata(settings, name, summary.chunks_indexed)
    return {
        "document": summary.document,
        "documents_parsed": summary.documents_parsed,
        "chunks_indexed": summary.chunks_indexed,
    }


def _iter_records(event: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract ``(bucket, key)`` pairs from a Lambda event.

    Parameters
    ----------
    event : dict
        Either an S3 notification event or a direct
        ``{"bucket": ..., "key": ...}`` payload.

    Returns
    -------
    list of tuple of (str, str)
        Bucket and key pairs to ingest.

    Raises
    ------
    ValueError
        If the event contains neither S3 records nor a bucket/key pair.
    """
    records = event.get("Records")
    if records:
        pairs: list[tuple[str, str]] = []
        for record in records:
            s3_payload = record.get("s3", {})
            bucket = s3_payload.get("bucket", {}).get("name")
            key = unquote_plus(s3_payload.get("object", {}).get("key", ""))
            if bucket and key:
                pairs.append((bucket, key))
        return pairs

    direct_bucket = event.get("bucket")
    direct_key = event.get("key")
    if direct_bucket and direct_key:
        return [(direct_bucket, direct_key)]

    raise ValueError(
        "Event must contain S3 'Records' or a 'bucket'/'key' pair."
    )


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """AWS Lambda entry point.

    Parameters
    ----------
    event : dict
        The Lambda event payload.
    context : Any, optional
        The Lambda context object (unused).

    Returns
    -------
    dict
        A summary listing the ingestion result for each object.
    """
    settings = get_settings()
    results = [
        ingest_object(settings, bucket, key)
        for bucket, key in _iter_records(event)
    ]
    return {"ingested": results}
