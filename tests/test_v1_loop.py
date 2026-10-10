"""
Tests for the v1 decompose / retrieve / judge / synthesize loop (src/v1_loop.py),
plus the fixed-RAG baseline that reuses its answer step.

A FakeLLM stands in for Anthropic/OpenAI so these tests are fast,
deterministic, and require no API key. Each test pins the behavior of
one FakeLLM decision function to exercise a specific control-flow path.
"""
import json

from src import baseline
from src.ingest import Chunk, VectorIndex
from src.llm import LLM
from src.run_result import ANSWERED, NOT_FOUND
from src.v1_loop import (ANSWER_SYSTEM, DECOMPOSE_SYSTEM, JUDGE_SYSTEM, NOT_FOUND_DRAFT,
                         SYNTHESIZE_SYSTEM, answer_question)


class FakeLLM(LLM):
    def __init__(self, decompose_fn=None, answer_fn=None, judge_fn=None, synthesize_fn=None):
        super().__init__()
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
        self.usage.add(10, 5)
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


def _hops(result):
    return sum(1 for s in result.steps if s.name == "search_document")


def test_simple_question_is_not_decomposed():
    index = _build_index()
    llm = FakeLLM(
        answer_fn=lambda user: {"found": True, "answer": "Revenue grew 9%.", "cited_pages": [3]},
    )

    result = answer_question("How much did Robotic Arms revenue grow?", index, llm)

    assert result.outcome == ANSWERED
    assert result.citations == ["p3"]
    assert sum(1 for c in llm.calls if c[0] == DECOMPOSE_SYSTEM) == 1
    assert result.llm_calls == len(llm.calls)
    assert result.input_tokens == 10 * len(llm.calls)


def test_complex_question_is_decomposed_and_synthesized():
    index = _build_index()

    def answer_fn(user):
        if "supply chain" in user.lower():
            return {"found": True, "answer": "Servo motors and gearboxes from Taiwan/South Korea.",
                    "cited_pages": [4]}
        return {"found": True, "answer": "Revenue grew 9%.", "cited_pages": [3]}

    llm = FakeLLM(
        decompose_fn=lambda user: {
            "needs_decomposition": True,
            "sub_questions": ["How much did Robotic Arms revenue grow?",
                              "What supply chain risk affects Robotic Arms?"],
            "reasoning": "two independent facts",
        },
        answer_fn=answer_fn,
        synthesize_fn=lambda user: {"answer": "Revenue grew 9%; supply chain concentration risk.",
                                    "cited_pages": [3, 4]},
    )

    result = answer_question(
        "How much did Robotic Arms revenue grow, and what supply chain risk does that segment face?",
        index, llm,
    )

    assert result.outcome == ANSWERED
    assert result.citations == ["p3", "p4"]
    assert any(s.kind == "synthesize" for s in result.steps)


def test_not_found_is_reported_honestly_not_guessed():
    index = _build_index()
    llm = FakeLLM()

    result = answer_question("What is the CEO's favorite color?", index, llm)

    assert result.outcome == NOT_FOUND
    assert result.citations == []
    assert result.answer == ""


def test_not_found_retry_uses_a_real_reformulated_query():
    """Regression test for the v1 bug: the retry query used to be the original question
    with an English note appended. Now the judge proposes a genuine reformulation."""
    index = _build_index()
    judge_inputs = []

    def judge_fn(user):
        judge_inputs.append(user)
        return {"sufficient": False, "next_query": "servo motor suppliers", "reasoning": "try synonyms"}

    llm = FakeLLM(judge_fn=judge_fn)
    result = answer_question("Which parts come from Asia?", index, llm)

    queries = [s.args["query"] for s in result.steps if s.name == "search_document"]
    assert queries[1] == "servo motor suppliers"
    assert all("previous phrasing" not in q for q in queries)
    assert NOT_FOUND_DRAFT in judge_inputs[0]


def test_max_hops_bounds_the_retrieval_loop():
    index = _build_index()
    counter = {"n": 0}

    def judge_fn(user):
        counter["n"] += 1
        return {"sufficient": False, "next_query": f"reformulated query {counter['n']}",
                "reasoning": "keep trying"}

    llm = FakeLLM(
        answer_fn=lambda user: {"found": True, "answer": "partial answer", "cited_pages": [3]},
        judge_fn=judge_fn,
    )

    result = answer_question("How much did Robotic Arms revenue grow?", index, llm)

    assert _hops(result) <= 3  # CONFIG.max_hops default
    assert result.outcome == ANSWERED


def test_fixed_rag_baseline_is_a_single_pass():
    index = _build_index()
    llm = FakeLLM(answer_fn=lambda user: {"found": True, "answer": "9%", "cited_pages": [3]})

    result = baseline.answer_question("How much did Robotic Arms revenue grow?", index, llm)

    assert result.outcome == ANSWERED
    assert result.citations == ["p3"]
    assert result.llm_calls == 1
    assert [c[0] for c in llm.calls] == [ANSWER_SYSTEM]
