"""
Thin provider-agnostic wrapper around Anthropic / OpenAI.

Two calling styles, one interface:

- `complete(system, user) -> str` / `complete_json(...)` -- single-shot
  calls, used by the fixed RAG baseline and the v1 decompose/judge loop.
- `chat(system, messages, tools) -> AssistantTurn` -- native tool calling,
  used by the tool-calling agent. Messages and tool schemas are kept in a
  small provider-neutral format and translated to each vendor's wire
  format here, so the agent loop never knows which vendor is behind it.

Every call also accumulates token usage on `self.usage`, which is what
the eval harness reads to report cost per question.
"""
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.config import CONFIG


# --------------------------------------------------------------------------
# Provider-neutral types
# --------------------------------------------------------------------------

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema for the arguments


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Usage:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.llm_calls += 1
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)


# Neutral message shapes used by the agent transcript:
#   {"role": "user", "content": str}
#   {"role": "assistant", "text": str, "tool_calls": [ToolCall, ...]}
#   {"role": "tool_results", "results": [{"id": str, "name": str, "content": str}, ...]}


class LLM(ABC):
    def __init__(self) -> None:
        self.usage = Usage()

    def reset_usage(self) -> None:
        self.usage = Usage()

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        ...

    def complete_json(self, system: str, user: str) -> dict:
        """Ask for JSON, tolerate minor formatting slop from the model."""
        raw = self.complete(system, user)
        return _extract_json(raw)

    def chat(self, system: str, messages: list[dict], tools: list[ToolSpec]) -> AssistantTurn:
        raise NotImplementedError(f"{type(self).__name__} does not support tool calling")


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


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------

class AnthropicLLM(LLM):
    def __init__(self, api_key: str, model: str):
        super().__init__()
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
        self.usage.add(resp.usage.input_tokens, resp.usage.output_tokens)
        return "".join(block.text for block in resp.content if block.type == "text")

    def chat(self, system: str, messages: list[dict], tools: list[ToolSpec]) -> AssistantTurn:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            tools=[{"name": t.name, "description": t.description, "input_schema": t.parameters}
                   for t in tools],
            messages=_to_anthropic(messages),
        )
        self.usage.add(resp.usage.input_tokens, resp.usage.output_tokens)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(id=b.id, name=b.name, args=dict(b.input or {}))
                 for b in resp.content if b.type == "tool_use"]
        return AssistantTurn(text=text, tool_calls=calls)


def _to_anthropic(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            blocks = []
            if m.get("text"):
                blocks.append({"type": "text", "text": m["text"]})
            for c in m.get("tool_calls", []):
                blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.args})
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no output)"}]})
        elif m["role"] == "tool_results":
            out.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"]}
                for r in m["results"]
            ]})
    return out


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------

class OpenAILLM(LLM):
    def __init__(self, api_key: str, model: str):
        super().__init__()
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
        self._record(resp)
        return resp.choices[0].message.content or ""

    def chat(self, system: str, messages: list[dict], tools: list[ToolSpec]) -> AssistantTurn:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}] + _to_openai(messages),
            tools=[{"type": "function",
                    "function": {"name": t.name, "description": t.description,
                                 "parameters": t.parameters}}
                   for t in tools],
        )
        self._record(resp)
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
        return AssistantTurn(text=msg.content or "", tool_calls=calls)

    def _record(self, resp) -> None:
        u = getattr(resp, "usage", None)
        self.usage.add(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))


def _to_openai(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            msg = {"role": "assistant", "content": m.get("text") or None}
            if m.get("tool_calls"):
                msg["tool_calls"] = [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                    for c in m["tool_calls"]
                ]
            out.append(msg)
        elif m["role"] == "tool_results":
            for r in m["results"]:
                out.append({"role": "tool", "tool_call_id": r["id"], "content": r["content"]})
    return out


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
