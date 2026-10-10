"""
v1: a bounded "decide-then-retrieve" loop (kept as a comparison point).

This was the first version of the project. The control flow is fixed by
application code -- decompose -> (retrieve -> answer -> judge)* -> synthesize --
and the LLM only makes two yes/no decisions inside it:

  1. decompose()          -- split the question into sub-questions or not?
  2. judge_sufficiency()  -- is the draft answer enough, or search again?

It has a single tool (vector search over the PDF) and cannot reach the
structured financial tables, calculate, or escalate. The eval harness runs
it side by side with the tool-calling agent (src/agent.py) and the plain
fixed-RAG baseline (src/baseline.py) to show what each step up buys.
"""
import time

from src.config import CONFIG
from src.ingest import VectorIndex
from src.llm import LLM
from src.retriever import RetrievedPassage, format_passages, retrieve
from src.run_result import ANSWERED, NOT_FOUND, RunResult, Step

SYSTEM_NAME = "v1_loop"
MAX_SUBQUESTIONS = 4

DECOMPOSE_SYSTEM = """You are a research planner. Given a user's question about a single \
source document, decide whether it can be answered with one focused search, or whether it \
genuinely requires splitting into independent sub-questions (e.g. because it asks about two \
different topics, or asks you to compare/relate two facts that likely live in different \
sections of the document).

Do not over-split. Most questions need exactly one sub-question (themselves). Only decompose \
when answering well truly requires gathering multiple independent pieces of information.

Respond with strict JSON only, no prose, in this shape:
{"needs_decomposition": true|false, "sub_questions": ["...", "..."], "reasoning": "one sentence"}

If needs_decomposition is false, sub_questions must be a list containing exactly the original \
question, unchanged. Never return more than 4 sub_questions."""

ANSWER_SYSTEM = """You answer questions using ONLY the passages provided below. Each passage is \
labeled with its source page number.

Rules:
- If the passages contain the answer, answer concisely and factually, and list the page \
number(s) you actually used.
- If the passages do NOT contain enough information to answer, you MUST set "found" to false \
and leave "answer" as an empty string. Do not guess, infer beyond what's stated, or use \
outside knowledge.

Respond with strict JSON only, no prose, in this shape:
{"found": true|false, "answer": "...", "cited_pages": [1, 2]}"""

JUDGE_SYSTEM = """You are reviewing a draft answer against the original question to decide if \
more retrieval is needed.

Respond with strict JSON only, no prose, in this shape:
{"sufficient": true|false, "next_query": "...", "reasoning": "one sentence"}

Set sufficient=true if the draft answer fully and specifically addresses the question. Set \
sufficient=false only if there's a concrete, different search query worth trying that could \
plausibly surface missing information -- in that case, next_query should be that reformulated \
search query (not a repeat of a query already tried). If you can't think of a genuinely \
different angle to search, set sufficient=true even if the answer is "not found" -- don't loop \
just to loop."""

SYNTHESIZE_SYSTEM = """You combine several sub-answers, each already grounded in cited source \
pages, into one coherent final answer to the user's original question. Do not add any \
information that isn't present in the sub-answers. If some sub-answers were not found, say so \
plainly rather than glossing over the gap. Keep citations attached to the claims they support.

Respond with strict JSON only, no prose, in this shape:
{"answer": "...", "cited_pages": [1, 2]}"""

NOT_FOUND_DRAFT = "(no answer found in the retrieved passages)"


def decompose(question: str, llm: LLM) -> tuple[bool, list[str], str]:
    result = llm.complete_json(DECOMPOSE_SYSTEM, f"Question: {question}")
    sub_qs = result.get("sub_questions") or [question]
    sub_qs = sub_qs[:MAX_SUBQUESTIONS]
    needs = bool(result.get("needs_decomposition")) and len(sub_qs) > 1
    return needs, sub_qs, result.get("reasoning", "")


def generate_answer(question: str, passages: list[RetrievedPassage], llm: LLM) -> dict:
    context = format_passages(passages)
    user = f"Question: {question}\n\nPassages:\n{context}"
    result = llm.complete_json(ANSWER_SYSTEM, user)
    return {
        "found": bool(result.get("found")),
        "answer": result.get("answer", "") or "",
        "cited_pages": [int(p) for p in (result.get("cited_pages") or []) if str(p).isdigit()],
    }


def _judge(question: str, draft_answer: str, tried_queries: list[str], llm: LLM) -> dict:
    user = (
        f"Original question: {question}\n"
        f"Draft answer: {draft_answer}\n"
        f"Queries already tried: {tried_queries}"
    )
    return llm.complete_json(JUDGE_SYSTEM, user)


def answer_sub_question(question: str, index: VectorIndex, llm: LLM,
                        max_hops: int | None = None) -> tuple[dict, list[Step]]:
    max_hops = max_hops or CONFIG.max_hops
    steps: list[Step] = []
    tried_queries: list[str] = []
    query = question
    best = {"found": False, "answer": "", "cited_pages": []}

    for hop in range(1, max_hops + 1):
        tried_queries.append(query)
        passages = retrieve(index, query)
        steps.append(Step(kind="tool", name="search_document", args={"query": query},
                          detail=f"hop {hop}: {len(passages)} passage(s), "
                                 f"pages {[p.chunk.page for p in passages]}"))

        draft = generate_answer(question, passages, llm) if passages else \
            {"found": False, "answer": "", "cited_pages": []}
        if draft["found"]:
            best = draft
        if hop >= max_hops:
            break

        # Ask the judge whether to search again. When nothing was found, the judge sees
        # that explicitly and proposes a genuinely reformulated query (bug fix: v1 used
        # to append an English note to the question and embed *that* as the next query).
        judgment = _judge(question, draft["answer"] if draft["found"] else NOT_FOUND_DRAFT,
                          tried_queries, llm)
        steps.append(Step(kind="judge",
                          detail=f"hop {hop}: sufficient={judgment.get('sufficient')} -- "
                                 f"{judgment.get('reasoning', '')}"))
        if judgment.get("sufficient", True):
            break
        next_query = (judgment.get("next_query") or "").strip()
        if not next_query or next_query in tried_queries:
            break
        query = next_query

    return best, steps


def _synthesize(question: str, sub_results: list[tuple[str, dict]], llm: LLM) -> dict:
    parts = "\n".join(
        f"- Sub-question: {q}\n  Found: {r['found']}\n  Answer: {r['answer']}\n  Pages: {r['cited_pages']}"
        for q, r in sub_results
    )
    user = f"Original question: {question}\n\nSub-answers:\n{parts}"
    return llm.complete_json(SYNTHESIZE_SYSTEM, user)


def answer_question(question: str, index: VectorIndex, llm: LLM) -> RunResult:
    usage_before = (llm.usage.llm_calls, llm.usage.input_tokens, llm.usage.output_tokens)
    started = time.perf_counter()
    result = RunResult(system=SYSTEM_NAME, question=question)

    needs_decomp, sub_qs, reasoning = decompose(question, llm)
    result.steps.append(Step(kind="decompose",
                             detail=f"needs_decomposition={needs_decomp} -- {reasoning} -- {sub_qs}"))

    sub_results: list[tuple[str, dict]] = []
    for sq in sub_qs:
        r, steps = answer_sub_question(sq, index, llm)
        sub_results.append((sq, r))
        result.steps.extend(steps)

    found_any = any(r["found"] for _, r in sub_results)
    if len(sub_results) == 1 or not found_any:
        r = sub_results[0][1] if len(sub_results) == 1 else {"found": False}
        if r.get("found"):
            result.outcome, result.answer = ANSWERED, r["answer"]
            result.citations = [f"p{p}" for p in sorted(set(r["cited_pages"]))]
        result.stop_reason = result.outcome
    else:
        synthesis = _synthesize(question, sub_results, llm)
        result.steps.append(Step(kind="synthesize", detail=str(synthesis.get("answer", ""))[:200]))
        result.outcome = ANSWERED
        result.answer = synthesis.get("answer", "")
        pages = [int(p) for p in (synthesis.get("cited_pages") or []) if str(p).isdigit()]
        result.citations = [f"p{p}" for p in sorted(set(pages))]
        result.stop_reason = ANSWERED

    if result.outcome == NOT_FOUND:
        result.answer = ""
    result.llm_calls = llm.usage.llm_calls - usage_before[0]
    result.input_tokens = llm.usage.input_tokens - usage_before[1]
    result.output_tokens = llm.usage.output_tokens - usage_before[2]
    result.latency_s = round(time.perf_counter() - started, 3)
    return result
