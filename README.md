# Agentic RAG: a tool-calling research agent, with evals

Ask free-form questions about a (fictional) company. The agent has two
sources — an annual report PDF and a small financial database — and four
tools. At every step **the model decides what to do next**: search the
report, query the database, calculate, give a cited answer, or hand the
request to a human. Application code only executes tools and enforces
guardrails the model can't override.

The repo also contains an **eval harness** that runs a 27-question golden
set through three systems — a fixed RAG pipeline, a bounded
decompose/judge loop, and the tool-calling agent — and reports accuracy,
citation quality, trajectory correctness, tokens and latency side by side.

## Why three systems

| System | Control flow | Can reach | Decisions made by the model |
|---|---|---|---|
| `fixed_rag` (`src/baseline.py`) | retrieve → answer, once | PDF | none |
| `v1_loop` (`src/v1_loop.py`) | decompose → (retrieve → answer → judge)* → synthesize | PDF | split the question? search again? |
| `tool_agent` (`src/agent.py`) | model picks the next tool, in a loop | PDF, database, calculator, human | which tool, with what input, when to stop, whether to escalate |

The demo data is built so the difference is testable rather than
decorative: the PDF states only **annual** figures, while **quarterly**
segment revenue lives only in the database. Some questions need the PDF,
some the database, some both plus arithmetic, some have no answer at all,
and some should not be answered by an AI.

## Architecture

```
                         ┌──────────── tools ─────────────┐
question ──► LLM ──────► │ search_document(query)         │──► FAISS over PDF chunks (page-labelled)
   ▲          │          │ query_financials(sql)          │──► in-memory SQLite, read-only
   │          │          │ calculate(expression)          │──► AST-whitelisted arithmetic
   │          │          │ final_answer(found, answer,    │
   │          │          │              citations)        │──► citation check ─► done
   │          │          │ escalate_to_human(reason)      │──► done
   │          │          └────────────────────────────────┘
   └── tool results ◄────┘          (loop, max MAX_STEPS turns)
```

Native tool calling through a provider-neutral wrapper (`src/llm.py`): the
same agent runs on Claude or GPT; swap via `.env`. Embeddings run locally
(sentence-transformers), independent of the reasoning provider.

## Guardrails (what the model cannot override)

1. **Step cap** — at most `MAX_STEPS` (default 8) model turns, then stop with
   an honest "not found" instead of a half-answer.
2. **Citation verification** — `final_answer(found=true)` must cite ≥1 source,
   and every cited page/table must have actually been returned by a tool *in
   this run*. Otherwise the answer is rejected and sent back to the model.
3. **Read-only data access** — the SQL tool accepts a single `SELECT`/`WITH`
   statement only, and the database itself is opened `query_only`; row count
   is capped.
4. **Safe arithmetic** — `calculate` evaluates a whitelisted AST, never `eval()`.
5. **Escalation path** — advice requests and requests to change data or take
   actions go to `escalate_to_human` instead of being "answered". Missing
   information is *not* an escalation: it's `found=false`.
6. **Loop hygiene** — identical repeated tool calls aren't re-run; a terminal
   tool must be called alone; a turn with no tool call gets a nudge.

Every run is recorded step by step (tool, arguments, result summary,
latency, guardrail interventions, tokens) and appended to
`traces/<date>.jsonl` — the record you need to answer "why did it say that?"

## Evals

`evals/golden.jsonl` — 27 questions across 7 types:

| Type | n | What it tests |
|---|---|---|
| lookup | 8 | single fact from the PDF |
| multi_hop | 3 | two facts from different PDF sections |
| calc | 4 | arithmetic over reported figures |
| sql | 6 | quarterly data that exists only in the database |
| mixed | 1 | database + PDF in one answer |
| not_found | 3 | answer isn't in any source → must decline |
| escalate | 2 | investment advice / data-change request → must escalate |

Scoring (`evals/scoring.py`) is **deterministic and rule-based**, not
LLM-as-judge, so every number is reproducible and explainable:

- **Correct** — right outcome (answer / decline / escalate) *and*, when
  answering, every required fact present (numbers match on digit
  boundaries, so `94` matches "$94M" but not "1994").
- **False answers** — answered a question it should have declined or
  escalated. In a regulated setting this is the number that matters most.
- **Citation precision / recall** — are cited sources acceptable, and were
  all required source groups cited?
- **Tool recall (trajectory)** — did the agent use the tools the question
  needs (e.g. the database for quarterly data, `calculate` for a growth rate)?
- **Cost & latency** — LLM calls, tokens in/out, p50 latency per question
  (and $ per question if you set current prices in `.env`).

Run it:

```bash
python -m evals.run_evals                              # 3 systems × 27 questions
python -m evals.run_evals --systems tool_agent --repeats 3   # check run-to-run variance
python -m evals.run_evals --ids S04,X01,E02            # a few questions
python -m evals.run_evals --publish                    # also write evals/RESULTS.md
```

Each run writes `evals/results/<timestamp>/report.md` (summary tables, per-
question grid, the agent's misses with reasons) and `results.jsonl` (every
run's full trajectory plus its scores).

> Results: see [`evals/RESULTS.md`](evals/RESULTS.md) once generated.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY (console.anthropic.com) or OPENAI_API_KEY
streamlit run app.py
```

The UI lets you switch between the three systems and shows each answer's
sources and full trajectory. Example questions:

- *"What was Warehouse Automation revenue in Q4 FY2024, and what drove that
  segment's growth?"* — database + PDF.
- *"How much did Warehouse Automation revenue grow in Q4 FY2024 vs Q4 FY2023?"*
  — SQL, then `calculate`.
- *"What is the CEO's favorite color?"* — not in any source → declines.
- *"Should I buy Northfield stock?"* — escalates to a human.

Unit tests (61, no API key, < 1 s) cover the agent loop and every guardrail
with a scripted LLM, the tools (including SQL-write and code-injection
attempts), the provider adapters, the v1 loop and baseline, and the
scoring rules:

```bash
pytest tests/ -v
```

## Project layout

```
src/
  agent.py        tool-calling agent loop + guardrails
  tools.py        tool specs + implementations, evidence tracking, safe calculator
  financials.py   in-memory read-only SQLite (quarterly segment revenue)
  llm.py          provider-neutral chat + tool calling (Anthropic, OpenAI), token usage
  v1_loop.py      v1: decompose / judge bounded loop (comparison system)
  baseline.py     fixed single-pass RAG (comparison system)
  run_result.py   one result/trace shape for all systems
  tracing.py      JSONL run traces
  ingest.py, retriever.py, embeddings.py, config.py
evals/
  golden.jsonl    27 labelled questions
  scoring.py      deterministic scoring rules
  run_evals.py    runner + markdown report
app.py            Streamlit UI
tests/            unit tests (no API key needed)
```

## Design notes / things worth defending in an interview

- **Why give the model control at all?** Because the right sequence differs
  per question — a quarterly figure needs SQL, a growth rate needs SQL *and*
  arithmetic, a risk question needs the PDF. A fixed pipeline either can't
  answer those or hard-codes a router that breaks on the next question type.
  The eval's per-type table shows exactly where that pays off and where it
  doesn't (simple lookups gain nothing and cost more calls).
- **Why agentic is riskier in a regulated setting, and how this contains it.**
  An agent can pick the wrong tool, loop, run up cost, or answer something it
  should never answer. Each of those has a hard control here — step cap,
  citation verification against retrieved evidence, read-only data, an
  explicit escalation path — and each is covered by a unit test and measured
  in the eval (false answers, tool recall, calls per question).
- **Evaluate trajectories, not just answers.** A right number reached by
  skipping `calculate` is luck; the trajectory metric catches it.
- **Deterministic scoring first.** Rule-based checks are cheap, reproducible
  and auditable. LLM-as-judge is a reasonable next step for free-text
  answers, but it needs its own calibration against human labels.
- **The v1 loop is kept on purpose.** It's the comparison point that shows
  what the tool-calling design buys over "an LLM decides yes/no inside a
  fixed pipeline." (It also had a bug — on an empty retrieval it embedded the
  question plus an English note as the retry query; the judge now proposes a
  real reformulation, with a regression test.)

All data in this repo is fictional.
