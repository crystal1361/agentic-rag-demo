# Agentic RAG over PDFs

Ask a document free-form questions. Every answer is grounded and cited by page
number; if the document doesn't contain the answer, the agent says so instead
of guessing. The retrieval layer decides, on its own, whether a question needs
to be split into sub-questions and whether one search pass was enough — or
whether it's worth trying again with a different query.

## Why this exists

Most "RAG chatbot" demos run a single fixed pass: embed the question, retrieve
top-k chunks, generate an answer. That's fine for simple lookups, but it falls
over on two common cases:

- **Compound questions** — "How did X segment's revenue grow, and what risk
  does that segment face?" — really contain two independent lookups. A single
  retrieval pass tends to retrieve chunks relevant to only one half.
- **A first search that comes up empty or partial** — the right chunk might
  exist under different wording than the user's question. A fixed pipeline
  has no way to try again.

This project adds a thin **agentic decision layer** on top of an otherwise
standard RAG pipeline. The retrieval and generation mechanics are unchanged —
chunk with page-number metadata, embed, cosine-similarity search, ground the
answer in cited passages, return "not found" rather than hallucinate. What's
new is that two decisions that would normally be hard-coded are instead made
by the LLM itself, in a loop, with a hard cap so it always terminates:

1. **`decompose()`** — does this question need to be split into independent
   sub-questions, or is one focused search enough?
2. **`judge_sufficiency()`** — after retrieving and drafting an answer, is
   this enough to answer the question, or is a different search worth trying?

That's the difference between automation (a fixed sequence of steps a human
designed) and an agent (a model deciding, at runtime, whether to keep going).

## Architecture

```
PDF -> ingest (chunk 800 tok / 120 overlap, page metadata)
     -> embed (local sentence-transformers, provider-independent)
     -> FAISS index (cosine similarity via normalized inner product)

question -> decompose()  [LLM decision: split or not?]
         -> for each sub-question:
              loop (bounded by MAX_HOPS):
                retrieve top-k above similarity threshold
                generate grounded answer (or "not found")
                judge_sufficiency()  [LLM decision: good enough, or retry?]
         -> synthesize() if multiple sub-questions
         -> final answer + citations + full reasoning trace
```

Every LLM call in the reasoning layer is a single provider-agnostic
`LLM.complete_json()` call (see `src/llm.py`), so the same agent logic runs
unchanged against Claude or GPT — swap providers via `.env`, not code.

## Project layout

```
src/
  config.py       tunables (chunk size, top_k, similarity threshold, max_hops...)
  llm.py          provider-agnostic wrapper (Anthropic + OpenAI)
  embeddings.py   local embedding backend (no API key required)
  ingest.py       PDF -> chunks with page metadata -> FAISS index
  retriever.py    top-k + similarity-threshold retrieval
  agent.py        the agentic decision loop (decompose / judge / synthesize)
app.py            Streamlit chat UI with a visible agent reasoning trace
scripts/generate_demo_doc.py   builds the fictional demo PDF in data/
tests/            unit tests against a FakeLLM — no API key needed to run them
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY or OPENAI_API_KEY
python scripts/generate_demo_doc.py   # regenerate the demo PDF if needed
streamlit run app.py
```

Upload your own PDF from the sidebar, or ask questions against the bundled
demo document (`data/northfield_robotics_annual_report.pdf`) — a fictional
company annual report where a few natural questions genuinely require pulling
facts from two different sections, e.g.:

- *"How much did the Robotic Arms segment's revenue grow, and what supply
  chain risk does that segment face?"* — triggers decomposition into two
  sub-questions, one per section.
- *"What is the CEO's favorite color?"* — not in the document; the agent
  says so rather than inventing an answer.

Run the tests (no API key needed — they run against a `FakeLLM`):

```bash
pytest tests/ -v
```

## Design notes / things worth defending in an interview

- **Grounding is unconditional.** `ANSWER_SYSTEM` forces `found=false` with an
  empty answer when the retrieved passages don't support one — that's
  enforced by prompt contract, and additionally back-stopped by the
  similarity threshold in `retriever.py`, which skips the generation call
  entirely when nothing clears the bar.
- **Every loop is bounded.** `decompose()` caps sub-questions at 4;
  `judge_sufficiency()`'s retry loop is capped at `MAX_HOPS` (default 3). An
  agent that can decide to keep going also needs a decision it can't
  override: when to stop.
- **Embeddings are decoupled from the LLM provider.** Anthropic has no
  embeddings endpoint; routing embeddings through OpenAI just because the
  reasoning model is OpenAI would couple two independent choices. Running
  embeddings locally means retrieval quality doesn't change when you swap
  reasoning providers, and the pipeline works offline until you actually need
  the agent to reason.
- **This is the natural extension of a fixed extraction pipeline.** The
  ingest/embed/retrieve/cite mechanics here are the same shape as a
  standard field-extraction RAG pipeline; the only new surface area is
  handing two specific, bounded decisions to the model instead of hard-coding
  them.
