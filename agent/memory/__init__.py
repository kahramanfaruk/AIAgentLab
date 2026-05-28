"""Memory package: conversation/escalation stores and the memory factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.memory.chat_history import (
    ConversationMemory,
    DynamoDBConversationStore,
    InMemoryConversationStore,
    Message,
)

if TYPE_CHECKING:
    from config.settings import Settings

__all__ = [
    "ConversationMemory",
    "DynamoDBConversationStore",
    "InMemoryConversationStore",
    "Message",
    "build_memory",
]


def build_memory(settings: Settings) -> ConversationMemory:
    """Construct the memory backend selected by ``settings.memory_backend``.

    Parameters
    ----------
    settings : Settings
        Application configuration.

    Returns
    -------
    ConversationMemory
        A backend implementing the :class:`ConversationMemory` protocol.

    Raises
    ------
    ValueError
        If ``settings.memory_backend`` is not a supported backend.
    """
    if settings.memory_backend == "memory":
        return InMemoryConversationStore()

    if settings.memory_backend == "dynamodb":
        return DynamoDBConversationStore.from_settings(settings)

    raise ValueError(f"Unsupported memory_backend: {settings.memory_backend!r}")
