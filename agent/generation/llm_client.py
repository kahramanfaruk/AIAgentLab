"""Groq text-generation backend.

Groq is the free, local-development default LLM provider. It implements the
:class:`~agent.generation.base.LLMClient` protocol so it is interchangeable
with the Bedrock backend.
"""

from __future__ import annotations

from dataclasses import dataclass

from groq import Groq

from agent.generation.base import DEFAULT_SYSTEM_PROMPT


@dataclass(slots=True)
class GenerationConfig:
    """Generation parameters shared by text backends.

    Attributes
    ----------
    model_name : str
        Provider-specific model identifier.
    temperature : float
        Sampling temperature.
    max_output_tokens : int
        Upper bound on generated tokens.
    """

    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2
    max_output_tokens: int = 800


class GroqClient:
    """LLM client backed by the Groq chat-completions API.

    Parameters
    ----------
    api_key : str
        Groq API key. Required and validated at construction.
    config : GenerationConfig or None, optional
        Generation parameters. Defaults to :class:`GenerationConfig`.

    Raises
    ------
    ValueError
        If ``api_key`` is empty.
    """

    def __init__(self, api_key: str, config: GenerationConfig | None = None) -> None:
        if not api_key:
            raise ValueError(
                "Groq API key is required. Set GROQ_API_KEY in the environment "
                "or switch LLM_PROVIDER to a different backend."
            )
        self.config = config or GenerationConfig()
        self.client = Groq(api_key=api_key)

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a completion for ``prompt``.

        Parameters
        ----------
        prompt : str
            The user prompt to complete.
        system : str or None, optional
            System instruction; defaults to
            :data:`~agent.generation.base.DEFAULT_SYSTEM_PROMPT`.

        Returns
        -------
        str
            The generated text, stripped.

        Raises
        ------
        RuntimeError
            If the provider returns an empty response.
        """
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_completion_tokens=self.config.max_output_tokens,
        )

        text = response.choices[0].message.content
        if not text:
            raise RuntimeError("Groq returned an empty response.")

        return text.strip()
