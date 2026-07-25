"""
Streamlit chat UI for the agentic RAG demo.

Run: streamlit run app.py

Lets a "stakeholder" ask free-form questions about the demo document.
Every answer is grounded in cited pages, and the sidebar/expander shows
the agent's reasoning trace -- whether it decomposed the question,
how many retrieval hops it took, and why it stopped -- so the
autonomous-decision layer is actually visible, not just claimed.
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.agent import answer_question
from src.config import CONFIG
from src.ingest import ingest_pdf
from src.llm import get_llm

DEMO_PDF = os.path.join(os.path.dirname(__file__), "data", "northfield_robotics_annual_report.pdf")

st.set_page_config(page_title="Agentic RAG Demo", page_icon="🔎", layout="wide")
st.title("🔎 Agentic RAG — ask the document anything")
st.caption(
    "Demo corpus: a fictional company's annual report (see data/). "
    "The agent decides on its own whether to split your question into sub-questions "
    "and whether one retrieval pass was enough, or another is worth trying."
)

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("LLM provider", ["anthropic", "openai"],
                             index=0 if CONFIG.llm_provider == "anthropic" else 1)
    st.caption("Requires the matching API key in your environment (.env).")
    uploaded = st.file_uploader("Or upload your own PDF", type=["pdf"])
    st.divider()
    st.caption(
        f"chunk_size={CONFIG.chunk_size} · overlap={CONFIG.chunk_overlap} · "
        f"top_k={CONFIG.top_k} · threshold={CONFIG.similarity_threshold} · "
        f"max_hops={CONFIG.max_hops}"
    )


@st.cache_resource(show_spinner="Indexing document...")
def load_index(pdf_bytes_or_path, is_path: bool):
    if is_path:
        return ingest_pdf(pdf_bytes_or_path)
    tmp_path = "/tmp/_uploaded.pdf"
    with open(tmp_path, "wb") as f:
        f.write(pdf_bytes_or_path)
    return ingest_pdf(tmp_path)


if uploaded is not None:
    index = load_index(uploaded.getvalue(), is_path=False)
    st.sidebar.success(f"Indexed uploaded file: {uploaded.name}")
else:
    index = load_index(DEMO_PDF, is_path=True)
    st.sidebar.info("Using demo document: northfield_robotics_annual_report.pdf")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("Agent reasoning trace"):
                for step in msg["trace"]:
                    st.markdown(f"**{step['kind']}** — {step['detail']}")

question = st.chat_input("Ask a question about the document...")
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

        with st.spinner("Thinking..."):
            result = answer_question(question, index, llm)

        if result.found:
            pages = ", ".join(str(p) for p in result.cited_pages) or "n/a"
            answer_md = f"{result.answer}\n\n*Source page(s): {pages}*"
        else:
            answer_md = "I could not find this in the document."
        st.markdown(answer_md)

        trace_dicts = [{"kind": t.kind, "detail": t.detail} for t in result.trace]
        with st.expander("Agent reasoning trace"):
            for step in trace_dicts:
                st.markdown(f"**{step['kind']}** — {step['detail']}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_md, "trace": trace_dicts}
    )
