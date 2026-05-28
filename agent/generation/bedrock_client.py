"""Amazon Bedrock text-generation backend.

Bedrock is the AWS production LLM provider. It is pay-per-token with no idle
cost, so it is the budget-safe cloud option. This adapter targets Anthropic
Claude models through the Bedrock Runtime ``invoke_model`` API and optionally
applies a Bedrock Guardrail.

``boto3`` is imported lazily so the local stack never requires AWS libraries to
be importable at module load time.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agent.generation.base import DEFAULT_SYSTEM_PROMPT
from agent.generation.llm_client import GenerationConfig

if TYPE_CHECKING:
    from config.settings import Settings


class BedrockClient:
    """LLM client backed by Amazon Bedrock Runtime (Anthropic Claude).

    Parameters
    ----------
    config : GenerationConfig
        Generation parameters; ``model_name`` must be a Bedrock model id.
    region : str
        AWS region of the Bedrock endpoint.
    endpoint_url : str or None, optional
        Override endpoint, for example a LocalStack URL. ``None`` uses real AWS.
    guardrail_id : str or None, optional
        Bedrock Guardrail identifier. When set, the guardrail is applied to
        every request.
    guardrail_version : str, optional
        Guardrail version, defaults to ``"DRAFT"``.

    Raises
    ------
    RuntimeError
        If ``boto3`` is not installed.
    """

    _ANTHROPIC_VERSION = "bedrock-2023-05-31"

    def __init__(
        self,
        config: GenerationConfig,
        region: str,
        endpoint_url: str | None = None,
        guardrail_id: str | None = None,
        guardrail_version: str = "DRAFT",
    ) -> None:
        try:
            import boto3
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "boto3 is required for the Bedrock backend. Install it with "
                "'pip install \"aiagentlab-rag[aws]\"'."
            ) from exc

        self.config = config
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            endpoint_url=endpoint_url,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> BedrockClient:
        """Build a client from application settings.

        Parameters
        ----------
        settings : Settings
            Application configuration.

        Returns
        -------
        BedrockClient
            A configured client.
        """
        config = GenerationConfig(
            model_name=settings.bedrock_model_id,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
        )
        return cls(
            config=config,
            region=settings.aws_region,
            endpoint_url=settings.aws_endpoint_url,
            guardrail_id=settings.bedrock_guardrail_id,
            guardrail_version=settings.bedrock_guardrail_version,
        )

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a completion through Bedrock Runtime.

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
            If the Bedrock invocation fails or returns no text.
        """
        body: dict[str, Any] = {
            "anthropic_version": self._ANTHROPIC_VERSION,
            "system": system or DEFAULT_SYSTEM_PROMPT,
            "max_tokens": self.config.max_output_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        request: dict[str, Any] = {
            "modelId": self.config.model_name,
            "body": json.dumps(body),
            "contentType": "application/json",
            "accept": "application/json",
        }
        if self.guardrail_id:
            request["guardrailIdentifier"] = self.guardrail_id
            request["guardrailVersion"] = self.guardrail_version

        try:
            response = self.client.invoke_model(**request)
            payload = json.loads(response["body"].read())
        except Exception as exc:
            raise RuntimeError(
                f"Bedrock invoke_model failed for model "
                f"'{self.config.model_name}': {exc}"
            ) from exc

        text = self._extract_text(payload)
        if not text:
            raise RuntimeError("Bedrock returned an empty response.")
        return text.strip()

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        """Extract concatenated text from a Claude messages-API payload.

        Parameters
        ----------
        payload : dict
            Decoded Bedrock response body.

        Returns
        -------
        str
            Concatenated text blocks, or an empty string if none are present.
        """
        blocks = payload.get("content", [])
        return "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
