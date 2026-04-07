from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import Groq


@dataclass(slots=True)
class GenerationConfig:
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2
    max_output_tokens: int = 800


class GroqClient:
    def __init__(self, config: GenerationConfig | None = None) -> None:
        load_dotenv()
        self.config = config or GenerationConfig()
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set in the environment.")

        self.client = Groq(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful, grounded assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.config.temperature,
            max_completion_tokens=self.config.max_output_tokens,
        )

        text = response.choices[0].message.content
        if not text:
            raise RuntimeError("Groq returned an empty response.")

        return text.strip()


if __name__ == "__main__":
    client = GroqClient()
    reply = client.generate("Say hello in one short sentence.")
    print(reply)