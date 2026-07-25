"""
Tests for the agentic decision layer: decompose / retrieve / judge / synthesize.

A FakeLLM stands in for Anthropic/OpenAI so these tests are fast,
deterministic, and require no API key. Each test pins the behavior of
one FakeLLM decision function to exercise a specific control-flow
path in src/agent.py.
"""
import json

from src.agent import DECOMPOSE_SYSTEM, ANSWER_SYSTEM, JUDGE_SYSTEM, SYNTHESIZE_SYSTEM
from src.agent import answer_question
from src.ingest import Chunk, VectorIndex
from src.llm import LLM


class FakeLLM(LLM):
    def __init__(self, decompose_fn=None, answer_fn=None, judge_fn=None, synthesize_fn=None):
        self.decompose_fn = decompose_fn or (lambda user: {
            "needs_decomposition": False,
            "sub_questions": [user.split("Question: ")[1].strip()],
            "reasoning": "single focused question",
        })
        self.answer_fn = answer_fn or (lambda user: {"found": False, "answer": "", "cited_pages": []})
        self.judge_fn = judge_fn or (lambda user: {"sufficient": True, "next_query": "", "reasoning": "ok"})
        self.synthesize_fn = synthesize_fn or (lambda user: {"answer": "combined", "cited_pages": []})
        self.calls = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if system == DECOMPOSE_SYSTEM:
            return json.dumps(self.decompose_fn(user))
        if system == ANSWER_SYSTEM:
            return json.dumps(self.answer_fn(user))
        if system == JUDGE_SYSTEM:
            return json.dumps(self.judge_fn(user))
        if system == SYNTHESIZE_SYSTEM:
            return json.dumps(self.synthesize_fn(user))
        raise AssertionError(f"unexpected system prompt: {system[:50]}")


def _build_index():
    chunks = [
        Chunk(text="Robotic Arms segment revenue grew nine percent in fiscal 2024", page=3, chunk_id=0),
        Chunk(text="supply chain servo motors gearboxes sourced Taiwan South Korea", page=4, chunk_id=1),
    ]
    from src.ingest import embed
    vectors = embed([c.text for c in chunks])
    return VectorIndex(chunks, vectors)


def test_simple_question_is_not_decomposed():
    index = _build_index()
    llm = FakeLLM(
        answer_fn=lambda user: {"found": True, "answer": "Revenue grew 9%.", "cited_pages": [3]},
        judge_fn=lambda user: {"sufficient": True, "next_query": "", "reasoning": "complete"},
    )

    result = answer_question("How much did Robotic Arms revenue grow?", index, llm)

    assert result.found is True
    assert result.cited_pages == [3]
    assert len(result.sub_answers) == 1
    decompose_calls = [c for c in llm.calls if c[0] == DECOMPOSE_SYSTEM]
    assert len(decompose_calls) == 1


def test_complex_question_is_decomposed_and_synthesized():
    index = _build_index()

    def decompose_fn(user):
        return {
            "needs_decomposition": True,
            "sub_questions": [
                "How much did Robotic Arms revenue grow?",
                "What supply chain risk affects Robotic Arms?",
            ],
            "reasoning": "asks about two independent facts",
        }

    def answer_fn(user):
        if "supply chain" in user.lower() or "servo" in user.lower():
            return {"found": True, "answer": "Servo motors and gearboxes from Taiwan/South Korea.", "cited_pages": [4]}
        return {"found": True, "answer": "Revenue grew 9%.", "cited_pages": [3]}

    llm = FakeLLM(
        decompose_fn=decompose_fn,
        answer_fn=answer_fn,
        judge_fn=lambda user: {"sufficient": True, "next_query": "", "reasoning": "complete"},
        synthesize_fn=lambda user: {
            "answer": "Revenue grew 9%, and the segment's margin was pressured by supply chain "
                      "concentration in servo motors and gearboxes sourced from Taiwan and South Korea.",
            "cited_pages": [3, 4],
        },
    )

    result = answer_question(
        "How much did Robotic Arms revenue grow, and what supply chain risk does that segment face?",
        index, llm,
    )

    assert result.found is True
    assert len(result.sub_answers) == 2
    assert result.cited_pages == [3, 4]
    assert "supply chain" in result.answer.lower()


def test_not_found_is_reported_honestly_not_guessed():
    index = _build_index()
    llm = FakeLLM(
        answer_fn=lambda user: {"found": False, "answer": "", "cited_pages": []},
        judge_fn=lambda user: {"sufficient": True, "next_query": "", "reasoning": "n/a"},
    )

    result = answer_question("What is the CEO's favorite color?", index, llm)

    assert result.found is False
    assert result.cited_pages == []
    assert "could not find" in result.answer.lower()


def test_max_hops_bounds_the_retrieval_loop():
    index = _build_index()
    # Judge always asks for another (novel) search -- without a hop cap this would loop forever.
    call_count = {"n": 0}

    def judge_fn(user):
        call_count["n"] += 1
        return {
            "sufficient": False,
            "next_query": f"reformulated query attempt {call_count['n']}",
            "reasoning": "keep trying",
        }

    llm = FakeLLM(
        answer_fn=lambda user: {"found": True, "answer": "partial answer", "cited_pages": [3]},
        judge_fn=judge_fn,
    )

    result = answer_question("How much did Robotic Arms revenue grow?", index, llm)

    assert result.sub_answers[0].hops_used <= 3  # CONFIG.max_hops default
    # The loop must have actually terminated (this line only reached if it did).
    assert result.found is True


def test_max_hops_is_respected_even_when_nothing_is_ever_found():
    index = _build_index()
    llm = FakeLLM(
        answer_fn=lambda user: {"found": False, "answer": "", "cited_pages": []},
    )

    result = answer_question("What is the CEO's shoe size?", index, llm)

    assert result.sub_answers[0].hops_used <= 3
    assert result.found is False
