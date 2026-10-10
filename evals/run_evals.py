"""
Eval harness: run the golden set through each system and compare.

    python -m evals.run_evals                           # all 3 systems, all questions
    python -m evals.run_evals --systems tool_agent      # just the agent
    python -m evals.run_evals --ids S04,X01 --repeats 3 # a few questions, 3 runs each
    python -m evals.run_evals --publish                 # also write evals/RESULTS.md

Systems compared on the same questions, same model, same retrieval:
  fixed_rag   single retrieve -> answer pass               (src/baseline.py)
  v1_loop     decompose + judge-driven re-retrieval loop   (src/v1_loop.py)
  tool_agent  tool-calling agent with guardrails           (src/agent.py)

Outputs (in evals/results/<timestamp>/):
  results.jsonl  one line per run: the full RunResult (answer, citations,
                 every step of the trajectory, tokens, latency) + its scores
  report.md      summary tables: overall, by question type, per question,
                 and the tool agent's failures with reasons
"""
import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evals.scoring import ItemScore, score_item  # noqa: E402
from src import agent, baseline, v1_loop  # noqa: E402
from src.config import CONFIG  # noqa: E402
from src.financials import build_database  # noqa: E402
from src.run_result import ANSWERED, RunResult  # noqa: E402
from src.tools import TOOL_SPECS  # noqa: E402

HERE = os.path.dirname(__file__)
GOLDEN_PATH = os.path.join(HERE, "golden.jsonl")
DEMO_PDF = os.path.join(HERE, "..", "data", "northfield_robotics_annual_report.pdf")

SYSTEMS = {
    "fixed_rag": lambda q, index, llm, db: baseline.answer_question(q, index, llm),
    "v1_loop": lambda q, index, llm, db: v1_loop.answer_question(q, index, llm),
    "tool_agent": lambda q, index, llm, db: agent.answer_question(q, index, llm, db=db),
}
# Trajectory (tool recall) is only scored for systems that can actually choose tools.
TOOLS_AVAILABLE = {
    "fixed_rag": set(),
    "v1_loop": set(),
    "tool_agent": {t.name for t in TOOL_SPECS},
}


def load_golden(path: str = GOLDEN_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_suite(items: list[dict], systems: list[str], llm, index, repeats: int = 1,
              on_result=None) -> list[tuple[RunResult, ItemScore]]:
    db = build_database()
    rows = []
    for rep in range(repeats):
        for item in items:
            for name in systems:
                try:
                    result = SYSTEMS[name](item["question"], index, llm, db)
                except Exception as e:  # one bad run shouldn't kill the whole eval
                    result = RunResult(system=name, question=item["question"], outcome="error",
                                       stop_reason=f"{type(e).__name__}: {e}")
                score = score_item(item, result, TOOLS_AVAILABLE[name])
                rows.append((result, score))
                if on_result:
                    on_result(rep, item, result, score)
    return rows


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def _pct(v):
    return "n/a" if v is None else f"{v * 100:.0f}%"


def _cost(results: list[RunResult]):
    if CONFIG.price_input_per_mtok is None or CONFIG.price_output_per_mtok is None:
        return None
    total = sum(r.input_tokens * CONFIG.price_input_per_mtok
                + r.output_tokens * CONFIG.price_output_per_mtok for r in results) / 1e6
    return total / len(results) if results else None


def render_report(rows: list[tuple[RunResult, ItemScore]], systems: list[str],
                  items: list[dict], model_label: str, repeats: int) -> str:
    by_sys = defaultdict(list)
    for r, s in rows:
        by_sys[s.system].append((r, s))

    out = [f"# Eval results\n",
           f"- Model: `{model_label}`",
           f"- Questions: {len(items)} · repeats: {repeats} · runs per system: {len(items) * repeats}",
           f"- Generated: {datetime.now():%Y-%m-%d %H:%M}\n",
           "## Overall\n",
           "| System | Correct | Right outcome | False answers¹ | Citation precision | Citation recall "
           "| Tool recall² | LLM calls / q | Tokens in / out per q | p50 latency | Cost / q³ |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in systems:
        pairs = by_sys[name]
        if not pairs:
            continue
        res = [r for r, _ in pairs]
        sc = [s for _, s in pairs]
        should_decline = [s for s in sc if s.expected_outcome != ANSWERED]
        cost = _cost(res)
        out.append(
            f"| {name} | {_pct(_mean([s.correct for s in sc]))} "
            f"| {_pct(_mean([s.outcome_correct for s in sc]))} "
            f"| {sum(s.false_answer for s in should_decline)}/{len(should_decline)} "
            f"| {_pct(_mean([s.citation_precision for s in sc]))} "
            f"| {_pct(_mean([s.citation_recall for s in sc]))} "
            f"| {_pct(_mean([s.tool_recall for s in sc]))} "
            f"| {_mean([r.llm_calls for r in res]):.1f} "
            f"| {_mean([r.input_tokens for r in res]):,.0f} / {_mean([r.output_tokens for r in res]):,.0f} "
            f"| {statistics.median([r.latency_s for r in res]):.1f}s "
            f"| {'n/a' if cost is None else f'${cost:.4f}'} |"
        )
    out += ["",
            "¹ Answered a question it should have declined (not in sources) or escalated.  ",
            "² Share of expected tools the trajectory actually used; only scored for the tool agent.  ",
            "³ Set PRICE_INPUT_PER_MTOK / PRICE_OUTPUT_PER_MTOK in .env to the current list prices "
            "to fill this column.\n"]

    # by type
    types = list(dict.fromkeys(i["type"] for i in items))
    out += ["## Correct by question type\n",
            "| Type | n | " + " | ".join(systems) + " |",
            "|---|---|" + "---|" * len(systems)]
    n_by_type = defaultdict(int)
    for i in items:
        n_by_type[i["type"]] += 1
    for t in types:
        cells = []
        for name in systems:
            sc = [s for _, s in by_sys[name] if s.type == t]
            cells.append(f"{sum(s.correct for s in sc)}/{len(sc)}" if sc else "-")
        out.append(f"| {t} | {n_by_type[t]} | " + " | ".join(cells) + " |")

    # per question
    out += ["", "## Per question\n",
            "✓ correct · ✗ wrong (outcome shown) · counts are correct/runs when repeats > 1\n",
            "| ID | Type | Question | " + " | ".join(systems) + " |",
            "|---|---|---|" + "---|" * len(systems)]
    for item in items:
        cells = []
        for name in systems:
            sc = [s for _, s in by_sys[name] if s.id == item["id"]]
            if not sc:
                cells.append("-")
            elif len(sc) == 1:
                cells.append("✓" if sc[0].correct else f"✗ ({sc[0].outcome})")
            else:
                cells.append(f"{sum(s.correct for s in sc)}/{len(sc)}")
        q = item["question"] if len(item["question"]) <= 70 else item["question"][:67] + "..."
        out.append(f"| {item['id']} | {item['type']} | {q} | " + " | ".join(cells) + " |")

    # failures of the agent
    if "tool_agent" in systems:
        fails = [(r, s) for r, s in by_sys["tool_agent"] if not s.correct]
        out += ["", "## Tool agent misses\n"]
        if not fails:
            out.append("None.")
        for r, s in fails:
            out.append(f"- **{s.id}** expected `{s.expected_outcome}`, got `{s.outcome}` "
                       f"({r.stop_reason}); tools: {r.tools_used}; "
                       f"{'; '.join(s.missing) or 'wrong outcome'}")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--systems", default="fixed_rag,v1_loop,tool_agent")
    p.add_argument("--provider", default=None, help="anthropic | openai (default: LLM_PROVIDER)")
    p.add_argument("--ids", default="", help="comma-separated golden ids to run")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    p.add_argument("--publish", action="store_true", help="also write evals/RESULTS.md")
    args = p.parse_args(argv)

    from src.ingest import ingest_pdf
    from src.llm import get_llm

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = set(systems) - set(SYSTEMS)
    if unknown:
        p.error(f"unknown systems: {unknown}")
    items = load_golden()
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        items = [i for i in items if i["id"] in wanted]
    if args.limit:
        items = items[:args.limit]

    llm = get_llm(args.provider)
    provider = (args.provider or CONFIG.llm_provider).lower()
    model = CONFIG.anthropic_model if provider == "anthropic" else CONFIG.openai_model
    print(f"Indexing {os.path.basename(DEMO_PDF)} ...")
    index = ingest_pdf(DEMO_PDF)

    run_dir = os.path.join(args.out, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.jsonl")
    total = len(items) * len(systems) * args.repeats
    done = {"n": 0}
    started = time.time()

    with open(results_path, "w", encoding="utf-8") as f:
        def on_result(rep, item, result, score):
            done["n"] += 1
            mark = "✓" if score.correct else "✗"
            print(f"[{done['n']:>3}/{total}] {mark} {score.system:<10} {item['id']:<4} "
                  f"{result.outcome:<9} calls={result.llm_calls} {result.latency_s:.1f}s")
            f.write(json.dumps({"repeat": rep, "result": result.to_dict(), "score": asdict(score)},
                               default=str) + "\n")
            f.flush()

        rows = run_suite(items, systems, llm, index, args.repeats, on_result)

    report = render_report(rows, systems, items, f"{provider}:{model}", args.repeats)
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    if args.publish:
        with open(os.path.join(HERE, "RESULTS.md"), "w", encoding="utf-8") as f:
            f.write(report)
    print("\n" + report)
    print(f"Done in {time.time() - started:.0f}s. Results: {run_dir}")


if __name__ == "__main__":
    main()
