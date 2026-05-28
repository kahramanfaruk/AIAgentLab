"""Provider-agnostic interface for text generation backends.

The agent and RAG chain depend only on :class:`LLMClient`, never on a concrete
provider. This keeps generation pluggable across Groq (local default) and
Amazon Bedrock without changing call sites.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

DEFAULT_SYSTEM_PROMPT = "You are a careful, grounded assistant."


@runtime_checkable
class LLMClient(Protocol):
    """Minimal text-generation contract implemented by every LLM backend.

    A single ``generate`` method is intentionally the whole surface: the agent
    loop drives tool use through structured prompting rather than provider
    specific function-calling APIs, so the same loop works against any backend.
    """

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a completion for ``prompt``.

        Parameters
        ----------
        prompt : str
            The user prompt to complete.
        system : str or None, optional
            System instruction. When ``None`` the backend applies
            :data:`DEFAULT_SYSTEM_PROMPT`.

        Returns
        -------
        str
            The generated text, stripped of surrounding whitespace.
        """
        ...
