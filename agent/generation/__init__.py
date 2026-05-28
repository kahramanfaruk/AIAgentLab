"""Generation package: LLM backends and the factory that selects one."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.generation.base import DEFAULT_SYSTEM_PROMPT, LLMClient
from agent.generation.llm_client import GenerationConfig, GroqClient

if TYPE_CHECKING:
    from config.settings import Settings

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "LLMClient",
    "GenerationConfig",
    "GroqClient",
    "build_llm_client",
]


def build_llm_client(settings: Settings) -> LLMClient:
    """Construct the LLM client selected by ``settings.llm_provider``.

    Parameters
    ----------
    settings : Settings
        Application configuration.

    Returns
    -------
    LLMClient
        A client implementing the :class:`LLMClient` protocol.

    Raises
    ------
    ValueError
        If ``settings.llm_provider`` is not a supported backend.
    """
    if settings.llm_provider == "groq":
        config = GenerationConfig(
            model_name=settings.groq_model,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
        )
        return GroqClient(api_key=settings.groq_api_key or "", config=config)

    if settings.llm_provider == "bedrock":
        from agent.generation.bedrock_client import BedrockClient

        return BedrockClient.from_settings(settings)

    raise ValueError(f"Unsupported llm_provider: {settings.llm_provider!r}")
