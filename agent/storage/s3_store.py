"""Amazon S3 document storage backend.

S3 is pay-per-use with negligible idle cost, making it the budget-safe AWS
object store. This adapter implements the
:class:`~agent.storage.base.DocumentStore` protocol and honours an endpoint
override so it can target LocalStack for free integration testing.

``boto3`` is imported lazily so the local filesystem path does not require it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import Settings


class S3DocumentStore:
    """Store documents as objects under a single S3 bucket and prefix.

    Parameters
    ----------
    bucket : str
        Target bucket name.
    region : str
        AWS region of the bucket.
    prefix : str, optional
        Key prefix applied to every object.
    endpoint_url : str or None, optional
        Override endpoint, for example a LocalStack URL.

    Raises
    ------
    RuntimeError
        If ``boto3`` is not installed.
    """

    def __init__(
        self,
        bucket: str,
        region: str,
        prefix: str = "documents/",
        endpoint_url: str | None = None,
    ) -> None:
        try:
            import boto3
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "boto3 is required for the S3 backend. Install it with "
                "'pip install \"aiagentlab-rag[aws]\"'."
            ) from exc

        self.bucket = bucket
        self.prefix = prefix
        self.client = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint_url,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> S3DocumentStore:
        """Build a store from application settings.

        Parameters
        ----------
        settings : Settings
            Application configuration.

        Returns
        -------
        S3DocumentStore
            A configured store.
        """
        return cls(
            bucket=settings.s3_bucket,
            region=settings.aws_region,
            endpoint_url=settings.aws_endpoint_url,
        )

    def save(self, name: str, data: bytes) -> str:
        """Upload ``data`` as an object keyed by ``name``.

        Parameters
        ----------
        name : str
            Object name (combined with the prefix to form the key).
        data : bytes
            Raw document content.

        Returns
        -------
        str
            The ``s3://`` URI of the stored object.

        Raises
        ------
        RuntimeError
            If the upload fails.
        """
        key = self._key(name)
        try:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to upload to s3://{self.bucket}/{key}: {exc}"
            ) from exc
        return f"s3://{self.bucket}/{key}"

    def load(self, name: str) -> bytes:
        """Download the object keyed by ``name``.

        Parameters
        ----------
        name : str
            Object name.

        Returns
        -------
        bytes
            Object content.

        Raises
        ------
        FileNotFoundError
            If the object does not exist.
        """
        key = self._key(name)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except self.client.exceptions.NoSuchKey as exc:
            raise FileNotFoundError(
                f"Document not found: s3://{self.bucket}/{key}"
            ) from exc

    def list_documents(self) -> list[str]:
        """List object names under the configured prefix.

        Returns
        -------
        list of str
            Sorted object names with the prefix stripped.
        """
        paginator = self.client.get_paginator("list_objects_v2")
        names: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                names.append(key[len(self.prefix):])
        return sorted(names)

    def exists(self, name: str) -> bool:
        """Report whether the object keyed by ``name`` exists.

        Parameters
        ----------
        name : str
            Object name.

        Returns
        -------
        bool
            ``True`` if the object exists.
        """
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(name))
            return True
        except ClientError:
            return False

    def _key(self, name: str) -> str:
        """Combine the prefix and a document name into an object key.

        Parameters
        ----------
        name : str
            Document name.

        Returns
        -------
        str
            The full object key.
        """
        return f"{self.prefix}{name}"
