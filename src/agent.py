"""
The tool-calling agent.

Instead of a fixed retrieve -> generate sequence, the model runs in a
loop: at each step it looks at the conversation so far and decides which
tool to call next (search the PDF, query the financial tables, calculate,
or finish / escalate). Application code never decides the order; it only
executes tools and enforces guardrails.

Guardrails (the parts the model cannot override):

1. Step cap      -- at most CONFIG.max_steps model turns; then stop with
                    an honest "not found" rather than a half-answer.
2. Citation check-- final_answer(found=true) must cite at least one
                    source, and every cited page/table must have actually
                    been returned by a tool in *this* run. Otherwise the
                    answer is rejected and the model must fix it.
3. Read-only     -- the SQL tool rejects anything but a single SELECT, and
                    the database is opened query_only.
4. Escalation    -- requests for advice or for actions route to a human
                    via escalate_to_human instead of being "answered".
5. Loop hygiene  -- an identical repeated tool call is not re-executed;
                    terminal tools must be called alone; a turn with no
                    tool call gets a nudge (and still counts as a step).

Everything the agent does is recorded as Steps on the RunResult, which
feeds the UI trace, the JSONL trace log and the eval harness.
"""
import json
import re
import sqlite3
import time

from src.config import CONFIG
from src.financials import build_database
from src.ingest import VectorIndex
from src.llm import LLM, ToolCall
from src.run_result import ANSWERED, ESCALATED, NOT_FOUND, RunResult, Step
from src.tools import TERMINAL_TOOLS, TOOL_SPECS, Toolbox

SYSTEM_NAME = "tool_agent"

AGENT_SYSTEM = """You are a careful financial research analyst answering questions about \
Northfield Robotics, a fictional company. You have two sources, reachable only through tools:
- the FY2024 annual report (search_document) -- narrative facts and annual figures, no quarterly data
- a structured financial database (query_financials) -- quarterly segment revenue and annual figures

Rules:
- Use ONLY information returned by your tools. Never use outside knowledge or assumptions.
- Use calculate for every growth rate, ratio, percentage or sum.
- Plan briefly, then call tools. Prefer the fewest calls that fully answer the question; if a \
search comes back empty or off-topic, rephrase instead of repeating it.
- Finish by calling final_answer by itself. Cite every page ("p4") or table you relied on. If the \
sources do not contain the answer, call final_answer with found=false -- do not guess.
- If the user asks for investment/legal/tax advice or a recommendation, asks you to modify data \
or take an action, or asks something a human should decide, call escalate_to_human instead. Information that is \
simply missing from the sources is NOT a reason to escalate: answer found=false."""


def answer_question(question: str, index: VectorIndex, llm: LLM,
                    db: sqlite3.Connection | None = None,
                    max_steps: int | None = None) -> RunResult:
    max_steps = max_steps or CONFIG.max_steps
    toolbox = Toolbox(index=index, db=db or build_database())
    result = RunResult(system=SYSTEM_NAME, question=question)
    messages: list[dict] = [{"role": "user", "content": question}]
    executed: dict[str, str] = {}   # (name+args) -> previous tool output, for repeat detection

    usage_before = (llm.usage.llm_calls, llm.usage.input_tokens, llm.usage.output_tokens)
    started = time.perf_counter()

    for step_no in range(1, max_steps + 1):
        t0 = time.perf_counter()
        turn = llm.chat(AGENT_SYSTEM, messages, TOOL_SPECS)
        llm_ms = (time.perf_counter() - t0) * 1000
        messages.append({"role": "assistant", "text": turn.text, "tool_calls": turn.tool_calls})
        if turn.text.strip():
            result.steps.append(Step(kind="note", detail=turn.text.strip()[:300], latency_ms=llm_ms))

        if not turn.tool_calls:
            result.steps.append(Step(kind="guardrail", detail="no tool call -> nudged to finish"))
            messages.append({"role": "user", "content":
                             "Please continue by calling a tool. To finish, call final_answer "
                             "(or escalate_to_human)."})
            continue

        terminal = [c for c in turn.tool_calls if c.name in TERMINAL_TOOLS]
        others = [c for c in turn.tool_calls if c.name not in TERMINAL_TOOLS]
        tool_results: list[dict] = []

        for call in others:
            tool_results.append(_execute(call, toolbox, executed, result))

        if terminal and others:
            # Model tried to finish in the same turn it requested new evidence.
            for call in terminal:
                result.steps.append(Step(kind="guardrail", name=call.name,
                                         detail="terminal tool called alongside other tools -> rejected"))
                tool_results.append(_reply(call, "Rejected: review the tool results first, then call "
                                                 f"{call.name} by itself."))
        elif terminal:
            call = terminal[0]
            done, feedback = _handle_terminal(call, toolbox, result)
            if done:
                break
            tool_results.append(_reply(call, feedback))
            for extra in terminal[1:]:
                tool_results.append(_reply(extra, "Ignored: only one terminal tool per turn."))

        messages.append({"role": "tool_results", "results": tool_results})
    else:
        result.outcome = NOT_FOUND
        result.answer = ""
        result.stop_reason = f"step_limit ({max_steps})"
        result.steps.append(Step(kind="guardrail",
                                 detail=f"step cap of {max_steps} reached -> stopped without an answer"))

    result.llm_calls = llm.usage.llm_calls - usage_before[0]
    result.input_tokens = llm.usage.input_tokens - usage_before[1]
    result.output_tokens = llm.usage.output_tokens - usage_before[2]
    result.latency_s = round(time.perf_counter() - started, 3)
    return result


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _reply(call: ToolCall, content: str) -> dict:
    return {"id": call.id, "name": call.name, "content": content}


def _execute(call: ToolCall, toolbox: Toolbox, executed: dict[str, str], result: RunResult) -> dict:
    key = call.name + json.dumps(call.args, sort_keys=True)
    if key in executed:
        result.steps.append(Step(kind="guardrail", name=call.name, args=call.args,
                                 detail="identical call repeated -> not re-run"))
        return _reply(call, "You already made this exact call. Same result as before:\n"
                            + executed[key][:1500] + "\nTry a different approach or finish.")
    t0 = time.perf_counter()
    outcome = toolbox.run(call.name, call.args)
    result.steps.append(Step(kind="tool", name=call.name, args=call.args, detail=outcome.summary,
                             ok=outcome.ok, latency_ms=round((time.perf_counter() - t0) * 1000, 1)))
    executed[key] = outcome.content
    return _reply(call, outcome.content)


_PAGE_RE = re.compile(r"^(?:p|page|pg\.?)?\s*(\d+)$", re.IGNORECASE)


def normalize_citation(raw) -> str:
    text = str(raw).strip().strip("[]()").strip()
    m = _PAGE_RE.match(text)
    if m:
        return f"p{int(m.group(1))}"
    return text.lower().replace("table:", "").strip()


def _handle_terminal(call: ToolCall, toolbox: Toolbox, result: RunResult) -> tuple[bool, str]:
    """Returns (done, feedback_for_model)."""
    if call.name == "escalate_to_human":
        reason = str(call.args.get("reason", "")).strip() or "no reason given"
        result.outcome = ESCALATED
        result.answer = f"Escalated to a human reviewer: {reason}"
        result.stop_reason = "escalated"
        result.steps.append(Step(kind="tool", name=call.name, args=call.args, detail=reason))
        return True, ""

    # final_answer
    found = bool(call.args.get("found"))
    answer = str(call.args.get("answer", "")).strip()
    citations = [normalize_citation(c) for c in (call.args.get("citations") or [])]
    citations = list(dict.fromkeys(c for c in citations if c))

    if not found:
        result.outcome = NOT_FOUND
        result.answer = ""
        result.stop_reason = "not_found"
        result.steps.append(Step(kind="tool", name=call.name, args=call.args,
                                 detail="found=false (not in sources)"))
        return True, ""

    problems = []
    if not answer:
        problems.append("found=true but the answer is empty")
    if not citations:
        problems.append("found=true requires at least one citation")
    unseen = [c for c in citations if c not in toolbox.seen_refs]
    if unseen:
        problems.append(f"these citations were never returned by your tools in this run: {unseen}; "
                        f"sources you actually retrieved: {sorted(toolbox.seen_refs) or 'none'}")
    if problems:
        result.steps.append(Step(kind="guardrail", name=call.name, args=call.args, ok=False,
                                 detail="citation check failed: " + "; ".join(problems)))
        return False, ("Rejected by citation check: " + "; ".join(problems)
                       + ". Retrieve the missing evidence or fix the citations, then call "
                         "final_answer again (or found=false if the sources don't support an answer).")

    result.outcome = ANSWERED
    result.answer = answer
    result.citations = citations
    result.stop_reason = "answered"
    result.steps.append(Step(kind="tool", name=call.name, args=call.args,
                             detail=f"answered, citing {citations}"))
    return True, ""
