"""Shared pytest fixtures.

The AWS fixtures use moto, so the AWS adapter tests run fully in-process with no
real cloud calls and no LocalStack dependency.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from config.settings import Settings

_REGION = "eu-central-1"
_BUCKET = "test-documents"
_TABLE = "test-state"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Return local-stack settings pointed at a temporary vector directory."""
    return Settings(
        vector_db_path=tmp_path / "vector_db",
        chroma_collection_name="test_collection",
        aws_region=_REGION,
        s3_bucket=_BUCKET,
        dynamodb_table=_TABLE,
    )


@pytest.fixture
def aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set dummy AWS credentials so moto never touches real AWS."""
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(key, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", _REGION)


@pytest.fixture
def s3_bucket(aws_credentials: None) -> Iterator[str]:
    """Provide a mocked S3 bucket and yield its name."""
    with mock_aws():
        client = boto3.client("s3", region_name=_REGION)
        client.create_bucket(
            Bucket=_BUCKET,
            CreateBucketConfiguration={"LocationConstraint": _REGION},
        )
        yield _BUCKET


@pytest.fixture
def dynamodb_table(aws_credentials: None) -> Iterator[str]:
    """Provide a mocked DynamoDB table and yield its name."""
    with mock_aws():
        client = boto3.client("dynamodb", region_name=_REGION)
        client.create_table(
            TableName=_TABLE,
            KeySchema=[
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        yield _TABLE


@pytest.fixture
def aws_backends(aws_credentials: None) -> Iterator[tuple[str, str]]:
    """Provide both a mocked S3 bucket and DynamoDB table in one moto context."""
    with mock_aws():
        s3 = boto3.client("s3", region_name=_REGION)
        s3.create_bucket(
            Bucket=_BUCKET,
            CreateBucketConfiguration={"LocationConstraint": _REGION},
        )
        ddb = boto3.client("dynamodb", region_name=_REGION)
        ddb.create_table(
            TableName=_TABLE,
            KeySchema=[
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        yield _BUCKET, _TABLE
