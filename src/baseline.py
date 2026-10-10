"""
Fixed RAG baseline: the classic single pass.

    embed question -> retrieve top-k above threshold -> generate grounded answer

No decisions, no retries, no tools beyond vector search. This is the
"before" picture in the eval report.
"""
import time

from src.ingest import VectorIndex
from src.llm import LLM
from src.retriever import retrieve
from src.run_result import ANSWERED, NOT_FOUND, RunResult, Step
from src.v1_loop import generate_answer

SYSTEM_NAME = "fixed_rag"


def answer_question(question: str, index: VectorIndex, llm: LLM) -> RunResult:
    usage_before = (llm.usage.llm_calls, llm.usage.input_tokens, llm.usage.output_tokens)
    started = time.perf_counter()
    result = RunResult(system=SYSTEM_NAME, question=question)

    passages = retrieve(index, question)
    result.steps.append(Step(kind="tool", name="search_document", args={"query": question},
                             detail=f"{len(passages)} passage(s), pages {[p.chunk.page for p in passages]}"))
    draft = generate_answer(question, passages, llm) if passages else {"found": False}

    if draft.get("found"):
        result.outcome = ANSWERED
        result.answer = draft["answer"]
        result.citations = [f"p{p}" for p in sorted(set(draft["cited_pages"]))]
    else:
        result.outcome = NOT_FOUND
    result.stop_reason = result.outcome

    result.llm_calls = llm.usage.llm_calls - usage_before[0]
    result.input_tokens = llm.usage.input_tokens - usage_before[1]
    result.output_tokens = llm.usage.output_tokens - usage_before[2]
    result.latency_s = round(time.perf_counter() - started, 3)
    return result
