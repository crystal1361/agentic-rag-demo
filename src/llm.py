"""
Thin provider-agnostic wrapper around Anthropic / OpenAI chat completions.

The agent loop only ever calls `LLM.complete(system, user) -> str` and
`LLM.complete_json(system, user, schema_hint) -> dict`. It never knows
or cares which vendor is behind it. That's the same "model-agnostic
pipeline, swap via config" pattern as the original RAG project -- here
it's extended to the reasoning/decision calls, not just extraction.
"""
import json
import re
from abc import ABC, abstractmethod

from src.config import CONFIG


class LLM(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        ...

    def complete_json(self, system: str, user: str) -> dict:
        """Ask for JSON, tolerate minor formatting slop from the model."""
        raw = self.complete(system, user)
        return _extract_json(raw)


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    # Strip ```json ... ``` fences if the model wrapped its output.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Last resort: grab the first {...} block.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


class AnthropicLLM(LLM):
    def __init__(self, api_key: str, model: str):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAILLM(LLM):
    def __init__(self, api_key: str, model: str):
        import openai
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content


def get_llm(provider: str | None = None) -> LLM:
    provider = (provider or CONFIG.llm_provider).lower()
    if provider == "anthropic":
        if not CONFIG.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        return AnthropicLLM(CONFIG.anthropic_api_key, CONFIG.anthropic_model)
    if provider == "openai":
        if not CONFIG.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return OpenAILLM(CONFIG.openai_api_key, CONFIG.openai_model)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")
