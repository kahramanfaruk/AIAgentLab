"""Conversation memory backends and the human-in-the-loop escalation store.

Memory persists three things: per-session chat history, and the escalation
queue that the agent writes to when it hands control to a human. Two backends
implement the same :class:`ConversationMemory` protocol: an in-process store
(free local default) and a DynamoDB store (pay-per-request, negligible idle
cost on AWS). Escalations are stored as JSON payloads so this module stays
decoupled from the orchestration schemas.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from config.settings import Settings

_SESSION_PREFIX = "SESSION#"
_MESSAGE_PREFIX = "MSG#"
_ESCALATION_PARTITION = "ESCALATION"


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string.

    Returns
    -------
    str
        Timezone-aware ISO 8601 timestamp.
    """
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class Message:
    """A single chat message.

    Attributes
    ----------
    role : str
        Message author, typically ``"user"`` or ``"assistant"``.
    content : str
        Message text.
    timestamp : str
        ISO 8601 creation time.
    """

    role: str
    content: str
    timestamp: str


@runtime_checkable
class ConversationMemory(Protocol):
    """Contract for chat history and escalation persistence."""

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to a session's history."""
        ...

    def get_history(self, session_id: str) -> list[Message]:
        """Return a session's messages in chronological order."""
        ...

    def clear(self, session_id: str) -> None:
        """Delete all messages for a session."""
        ...

    def save_escalation(self, escalation: dict[str, Any]) -> str:
        """Persist an escalation payload and return its identifier."""
        ...

    def list_escalations(self) -> list[dict[str, Any]]:
        """Return all escalation payloads, newest first."""
        ...


class InMemoryConversationStore:
    """Process-local conversation memory, used as the free default backend."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[Message]] = {}
        self._escalations: list[dict[str, Any]] = []

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to a session's history.

        Parameters
        ----------
        session_id : str
            Session identifier.
        role : str
            Message author.
        content : str
            Message text.
        """
        self._sessions.setdefault(session_id, []).append(
            Message(role=role, content=content, timestamp=_utc_now_iso())
        )

    def get_history(self, session_id: str) -> list[Message]:
        """Return a session's messages in chronological order.

        Parameters
        ----------
        session_id : str
            Session identifier.

        Returns
        -------
        list of Message
            Stored messages, or an empty list if the session is unknown.
        """
        return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str) -> None:
        """Delete all messages for a session.

        Parameters
        ----------
        session_id : str
            Session identifier.
        """
        self._sessions.pop(session_id, None)

    def save_escalation(self, escalation: dict[str, Any]) -> str:
        """Persist an escalation payload.

        Parameters
        ----------
        escalation : dict
            The escalation payload.

        Returns
        -------
        str
            The generated escalation identifier.
        """
        record = dict(escalation)
        record.setdefault("escalation_id", uuid.uuid4().hex)
        record.setdefault("created_at", _utc_now_iso())
        self._escalations.append(record)
        return record["escalation_id"]

    def list_escalations(self) -> list[dict[str, Any]]:
        """Return all escalation payloads, newest first.

        Returns
        -------
        list of dict
            Stored escalation payloads.
        """
        return sorted(
            self._escalations,
            key=lambda item: item.get("created_at", ""),
            reverse=True,
        )


class DynamoDBConversationStore:
    """Conversation memory backed by a single DynamoDB table.

    The table uses a composite key (``pk`` partition, ``sk`` sort). Messages are
    stored under ``pk = "SESSION#<id>"`` and escalations under
    ``pk = "ESCALATION"``, so both share one on-demand table.

    Parameters
    ----------
    table_name : str
        DynamoDB table name.
    region : str
        AWS region of the table.
    endpoint_url : str or None, optional
        Override endpoint, for example a LocalStack URL.

    Raises
    ------
    RuntimeError
        If ``boto3`` is not installed.
    """

    def __init__(
        self,
        table_name: str,
        region: str,
        endpoint_url: str | None = None,
    ) -> None:
        try:
            import boto3
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "boto3 is required for the DynamoDB backend. Install it with "
                "'pip install \"aiagentlab-rag[aws]\"'."
            ) from exc

        resource = boto3.resource(
            "dynamodb",
            region_name=region,
            endpoint_url=endpoint_url,
        )
        self.table = resource.Table(table_name)

    @classmethod
    def from_settings(cls, settings: Settings) -> DynamoDBConversationStore:
        """Build a store from application settings.

        Parameters
        ----------
        settings : Settings
            Application configuration.

        Returns
        -------
        DynamoDBConversationStore
            A configured store.
        """
        return cls(
            table_name=settings.dynamodb_table,
            region=settings.aws_region,
            endpoint_url=settings.aws_endpoint_url,
        )

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to a session's history.

        Parameters
        ----------
        session_id : str
            Session identifier.
        role : str
            Message author.
        content : str
            Message text.
        """
        timestamp = _utc_now_iso()
        self.table.put_item(
            Item={
                "pk": f"{_SESSION_PREFIX}{session_id}",
                "sk": f"{_MESSAGE_PREFIX}{timestamp}#{uuid.uuid4().hex}",
                "role": role,
                "content": content,
                "timestamp": timestamp,
            }
        )

    def get_history(self, session_id: str) -> list[Message]:
        """Return a session's messages in chronological order.

        Parameters
        ----------
        session_id : str
            Session identifier.

        Returns
        -------
        list of Message
            Stored messages.
        """
        from boto3.dynamodb.conditions import Key

        response = self.table.query(
            KeyConditionExpression=Key("pk").eq(f"{_SESSION_PREFIX}{session_id}")
            & Key("sk").begins_with(_MESSAGE_PREFIX),
            ScanIndexForward=True,
        )
        return [
            Message(
                role=item["role"],
                content=item["content"],
                timestamp=item["timestamp"],
            )
            for item in response.get("Items", [])
        ]

    def clear(self, session_id: str) -> None:
        """Delete all messages for a session.

        Parameters
        ----------
        session_id : str
            Session identifier.
        """
        from boto3.dynamodb.conditions import Key

        response = self.table.query(
            KeyConditionExpression=Key("pk").eq(f"{_SESSION_PREFIX}{session_id}")
            & Key("sk").begins_with(_MESSAGE_PREFIX),
        )
        with self.table.batch_writer() as batch:
            for item in response.get("Items", []):
                batch.delete_item(Key={"pk": item["pk"], "sk": item["sk"]})

    def save_escalation(self, escalation: dict[str, Any]) -> str:
        """Persist an escalation payload.

        Parameters
        ----------
        escalation : dict
            The escalation payload.

        Returns
        -------
        str
            The generated escalation identifier.
        """
        escalation_id = escalation.get("escalation_id") or uuid.uuid4().hex
        created_at = escalation.get("created_at") or _utc_now_iso()
        self.table.put_item(
            Item={
                "pk": _ESCALATION_PARTITION,
                "sk": f"{created_at}#{escalation_id}",
                "escalation_id": escalation_id,
                "created_at": created_at,
                "payload": json.dumps(escalation),
            }
        )
        return escalation_id

    def list_escalations(self) -> list[dict[str, Any]]:
        """Return all escalation payloads, newest first.

        Returns
        -------
        list of dict
            Stored escalation payloads.
        """
        from boto3.dynamodb.conditions import Key

        response = self.table.query(
            KeyConditionExpression=Key("pk").eq(_ESCALATION_PARTITION),
            ScanIndexForward=False,
        )
        return [json.loads(item["payload"]) for item in response.get("Items", [])]
