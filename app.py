"""
Streamlit chat UI for the agentic RAG demo.

Run: streamlit run app.py

Ask free-form questions about the demo company. Pick which system answers:
the tool-calling agent (default), the v1 decompose/judge loop, or the plain
fixed-RAG baseline -- so the difference is visible side by side. Every
answer shows its citations, and the expander shows the full trajectory:
which tools the agent chose, with what arguments, what came back, and any
guardrail that stepped in. Each run is also appended to traces/<date>.jsonl.
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src import agent, baseline, v1_loop
from src.config import CONFIG
from src.financials import build_database
from src.ingest import ingest_pdf
from src.llm import get_llm
from src.run_result import ANSWERED, ESCALATED
from src.tracing import write_trace

DEMO_PDF = os.path.join(os.path.dirname(__file__), "data", "northfield_robotics_annual_report.pdf")

SYSTEMS = {
    "Tool-calling agent": lambda q, index, llm, db: agent.answer_question(q, index, llm, db=db),
    "v1 decompose/judge loop": lambda q, index, llm, db: v1_loop.answer_question(q, index, llm),
    "Fixed RAG baseline": lambda q, index, llm, db: baseline.answer_question(q, index, llm),
}

EXAMPLES = [
    "What was Warehouse Automation revenue in Q4 FY2024, and what drove that segment's growth?",
    "How much did Warehouse Automation revenue grow in Q4 FY2024 vs Q4 FY2023, in percent?",
    "How much did the Robotic Arms segment's revenue grow, and what supply chain risk does it face?",
    "What is the CEO's favorite color?",
    "Should I buy Northfield stock right now?",
]

st.set_page_config(page_title="Agentic RAG Demo", page_icon="🔎", layout="wide")
st.title("🔎 Agentic RAG — a tool-calling research agent")
st.caption(
    "Sources: a fictional company's annual report (PDF) and a small financial database "
    "(quarterly segment revenue, which the PDF doesn't contain). The agent decides which "
    "tool to call at each step — search the report, query the database, calculate, answer, "
    "or escalate to a human — under hard guardrails."
)

with st.sidebar:
    st.header("Settings")
    system_name = st.radio("Answering system", list(SYSTEMS), index=0)
    provider = st.selectbox("LLM provider", ["anthropic", "openai"],
                            index=0 if CONFIG.llm_provider == "anthropic" else 1)
    st.caption("Requires the matching API key in your environment (.env).")
    uploaded = st.file_uploader("Or upload your own PDF (database tool still describes the demo company)",
                                type=["pdf"])
    st.divider()
    st.markdown("**Try:**")
    for ex in EXAMPLES:
        st.caption(f"• {ex}")
    st.divider()
    st.caption(
        f"top_k={CONFIG.top_k} · threshold={CONFIG.similarity_threshold} · "
        f"max_steps={CONFIG.max_steps} · max_hops(v1)={CONFIG.max_hops}"
    )


@st.cache_resource(show_spinner="Indexing document...")
def load_index(pdf_bytes_or_path, is_path: bool):
    if is_path:
        return ingest_pdf(pdf_bytes_or_path)
    tmp_path = "/tmp/_uploaded.pdf"
    with open(tmp_path, "wb") as f:
        f.write(pdf_bytes_or_path)
    return ingest_pdf(tmp_path)


@st.cache_resource
def load_db():
    return build_database()


if uploaded is not None:
    index = load_index(uploaded.getvalue(), is_path=False)
    st.sidebar.success(f"Indexed uploaded file: {uploaded.name}")
else:
    index = load_index(DEMO_PDF, is_path=True)

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_trace(trace: dict):
    st.caption(f"{trace['system']} · outcome **{trace['outcome']}** · {trace['llm_calls']} LLM call(s) · "
               f"{trace['input_tokens']:,} in / {trace['output_tokens']:,} out tokens · "
               f"{trace['latency_s']:.1f}s · stop: {trace['stop_reason']}")
    rows = []
    for i, s in enumerate(trace["steps"], 1):
        args = s["args"].get("query") or s["args"].get("sql") or s["args"].get("expression") \
            or s["args"].get("reason") or ""
        rows.append({"#": i, "kind": s["kind"], "tool": s["name"], "input": str(args)[:160],
                     "result": s["detail"][:200], "ok": "✓" if s["ok"] else "✗"})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def format_answer(result) -> str:
    if result.outcome == ANSWERED:
        cites = ", ".join(result.citations) or "n/a"
        return f"{result.answer}\n\n*Sources: {cites}*"
    if result.outcome == ESCALATED:
        return f"🙋 {result.answer}"
    return "I could not find this in the available sources."


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("Agent trajectory"):
                render_trace(msg["trace"])

question = st.chat_input("Ask a question about Northfield Robotics...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            llm = get_llm(provider)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

        with st.spinner("Working..."):
            result = SYSTEMS[system_name](question, index, llm, load_db())
        write_trace(result)

        answer_md = format_answer(result)
        st.markdown(answer_md)
        trace = result.to_dict()
        with st.expander("Agent trajectory"):
            render_trace(trace)

    st.session_state.messages.append({"role": "assistant", "content": answer_md, "trace": trace})
