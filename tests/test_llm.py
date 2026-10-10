"""
Provider adapters: neutral transcript -> Anthropic / OpenAI wire format, and
parsing tool calls + usage back out. Uses stub clients, so no network or key.
"""
import json
from types import SimpleNamespace as NS

from src.llm import AnthropicLLM, OpenAILLM, ToolCall, ToolSpec, _to_anthropic, _to_openai

TOOLS = [ToolSpec("calculate", "math", {"type": "object", "properties": {"expression": {"type": "string"}}})]
TRANSCRIPT = [
    {"role": "user", "content": "What is 94/75?"},
    {"role": "assistant", "text": "Let me compute.",
     "tool_calls": [ToolCall("call_1", "calculate", {"expression": "94/75"})]},
    {"role": "tool_results", "results": [{"id": "call_1", "name": "calculate", "content": '{"result": 1.25}'}]},
]


def test_anthropic_message_format():
    msgs = _to_anthropic(TRANSCRIPT)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"][1] == {"type": "tool_use", "id": "call_1", "name": "calculate",
                                     "input": {"expression": "94/75"}}
    assert msgs[2]["content"][0]["type"] == "tool_result"
    assert msgs[2]["content"][0]["tool_use_id"] == "call_1"


def test_openai_message_format():
    msgs = _to_openai(TRANSCRIPT)
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"]
    tc = msgs[1]["tool_calls"][0]
    assert tc["id"] == "call_1" and tc["type"] == "function"
    assert json.loads(tc["function"]["arguments"]) == {"expression": "94/75"}
    assert msgs[2]["tool_call_id"] == "call_1"


def test_anthropic_chat_parses_tool_use_and_usage():
    llm = AnthropicLLM.__new__(AnthropicLLM)
    from src.llm import Usage
    llm.usage, llm._model = Usage(), "m"
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return NS(usage=NS(input_tokens=120, output_tokens=30), content=[
            NS(type="text", text="Computing."),
            NS(type="tool_use", id="tu_1", name="calculate", input={"expression": "94/75"}),
        ])

    llm._client = NS(messages=NS(create=create))
    turn = llm.chat("sys", TRANSCRIPT[:1], TOOLS)

    assert turn.text == "Computing."
    assert turn.tool_calls == [ToolCall("tu_1", "calculate", {"expression": "94/75"})]
    assert captured["tools"][0]["input_schema"]["type"] == "object"
    assert (llm.usage.llm_calls, llm.usage.input_tokens, llm.usage.output_tokens) == (1, 120, 30)


def test_openai_chat_parses_tool_calls_and_usage():
    llm = OpenAILLM.__new__(OpenAILLM)
    from src.llm import Usage
    llm.usage, llm._model = Usage(), "m"
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        msg = NS(content=None, tool_calls=[
            NS(id="c1", function=NS(name="calculate", arguments='{"expression": "94/75"}'))])
        return NS(usage=NS(prompt_tokens=80, completion_tokens=12), choices=[NS(message=msg)])

    llm._client = NS(chat=NS(completions=NS(create=create)))
    turn = llm.chat("sys", TRANSCRIPT[:1], TOOLS)

    assert turn.text == ""
    assert turn.tool_calls == [ToolCall("c1", "calculate", {"expression": "94/75"})]
    assert captured["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["tools"][0]["function"]["name"] == "calculate"
    assert llm.usage.input_tokens == 80
