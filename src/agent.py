"""
The agentic layer on top of the fixed retrieve-then-generate pipeline.

This is the piece that turns "RAG pipeline" into "agent": instead of a
single hard-coded retrieve -> generate step, the LLM itself decides,
at two separate points, whether to keep going:

  1. decompose()      -- does this question need to be split into
                          independent sub-questions to be answered
                          fully, or is one focused search enough?
  2. judge_sufficiency() -- after retrieving and drafting an answer,
                          is what we have enough, or should we search
                          again with a different query?

Both decisions are made by the model, not by application code, and
both are bounded (MAX_SUBQUESTIONS, max_hops) so the loop always
terminates. That bound -- plus every dead end resolving to an honest
"not found" instead of a guess -- is the same grounding discipline as
the original fixed pipeline; only the control flow is new.
"""
from dataclasses import dataclass, field

from src.config import CONFIG
from src.ingest import VectorIndex
from src.llm import LLM
from src.retriever import RetrievedPassage, format_passages, retrieve

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


@dataclass
class TraceStep:
    kind: str      # "decompose" | "retrieve" | "judge" | "synthesize"
    detail: str


@dataclass
class SubAnswer:
    question: str
    answer: str
    found: bool
    cited_pages: list[int]
    hops_used: int


@dataclass
class AgentResult:
    question: str
    answer: str
    found: bool
    cited_pages: list[int]
    sub_answers: list[SubAnswer] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)


def decompose(question: str, llm: LLM) -> tuple[bool, list[str], str]:
    result = llm.complete_json(DECOMPOSE_SYSTEM, f"Question: {question}")
    sub_qs = result.get("sub_questions") or [question]
    sub_qs = sub_qs[:MAX_SUBQUESTIONS]
    needs = bool(result.get("needs_decomposition")) and len(sub_qs) > 1
    return needs, sub_qs, result.get("reasoning", "")


def _generate_answer(question: str, passages: list[RetrievedPassage], llm: LLM) -> dict:
    context = format_passages(passages)
    user = f"Question: {question}\n\nPassages:\n{context}"
    result = llm.complete_json(ANSWER_SYSTEM, user)
    return {
        "found": bool(result.get("found")),
        "answer": result.get("answer", "") or "",
        "cited_pages": result.get("cited_pages", []) or [],
    }


def _judge(question: str, draft_answer: str, tried_queries: list[str], llm: LLM) -> dict:
    user = (
        f"Original question: {question}\n"
        f"Draft answer: {draft_answer}\n"
        f"Queries already tried: {tried_queries}"
    )
    return llm.complete_json(JUDGE_SYSTEM, user)


def answer_sub_question(question: str, index: VectorIndex, llm: LLM,
                          max_hops: int | None = None) -> tuple[SubAnswer, list[TraceStep]]:
    max_hops = max_hops or CONFIG.max_hops
    trace: list[TraceStep] = []
    tried_queries: list[str] = []
    query = question
    best_draft = {"found": False, "answer": "", "cited_pages": []}

    for hop in range(1, max_hops + 1):
        tried_queries.append(query)
        passages = retrieve(index, query)
        trace.append(TraceStep(
            "retrieve",
            f"hop {hop}: query={query!r} -> {len(passages)} passage(s) above threshold "
            f"(pages: {[p.chunk.page for p in passages]})",
        ))

        draft = _generate_answer(question, passages, llm)
        best_draft = draft if draft["found"] else best_draft

        if not draft["found"]:
            # Nothing usable this hop. Worth one more attempt with a fresh angle,
            # otherwise stop -- an honest "not found" beats a fabricated retry loop.
            if hop < max_hops:
                query = f"{question} (previous phrasing found nothing: {tried_queries!r}; try different terms)"
                continue
            break

        judgment = _judge(question, draft["answer"], tried_queries, llm)
        trace.append(TraceStep(
            "judge",
            f"hop {hop}: sufficient={judgment.get('sufficient')} -- {judgment.get('reasoning', '')}",
        ))
        if judgment.get("sufficient", True) or hop >= max_hops:
            break
        next_query = judgment.get("next_query") or question
        if next_query in tried_queries:
            break
        query = next_query

    return (
        SubAnswer(
            question=question,
            answer=best_draft["answer"],
            found=best_draft["found"],
            cited_pages=sorted(set(best_draft["cited_pages"])),
            hops_used=len(tried_queries),
        ),
        trace,
    )


def _synthesize(question: str, sub_answers: list[SubAnswer], llm: LLM) -> dict:
    parts = "\n".join(
        f"- Sub-question: {sa.question}\n  Found: {sa.found}\n  Answer: {sa.answer}\n  Pages: {sa.cited_pages}"
        for sa in sub_answers
    )
    user = f"Original question: {question}\n\nSub-answers:\n{parts}"
    return llm.complete_json(SYNTHESIZE_SYSTEM, user)


def answer_question(question: str, index: VectorIndex, llm: LLM) -> AgentResult:
    trace: list[TraceStep] = []

    needs_decomp, sub_qs, reasoning = decompose(question, llm)
    trace.append(TraceStep(
        "decompose",
        f"needs_decomposition={needs_decomp} -- {reasoning} -- sub_questions={sub_qs}",
    ))

    sub_answers: list[SubAnswer] = []
    for sq in sub_qs:
        sa, sub_trace = answer_sub_question(sq, index, llm)
        sub_answers.append(sa)
        trace.extend(sub_trace)

    if len(sub_answers) == 1:
        sa = sub_answers[0]
        answer_text = sa.answer if sa.found else "I could not find this in the document."
        return AgentResult(
            question=question, answer=answer_text, found=sa.found,
            cited_pages=sa.cited_pages, sub_answers=sub_answers, trace=trace,
        )

    if not any(sa.found for sa in sub_answers):
        return AgentResult(
            question=question,
            answer="I could not find information about this in the document.",
            found=False, cited_pages=[], sub_answers=sub_answers, trace=trace,
        )

    synthesis = _synthesize(question, sub_answers, llm)
    trace.append(TraceStep("synthesize", synthesis.get("answer", "")[:200]))
    return AgentResult(
        question=question,
        answer=synthesis.get("answer", ""),
        found=True,
        cited_pages=sorted(set(synthesis.get("cited_pages", []))),
        sub_answers=sub_answers,
        trace=trace,
    )
