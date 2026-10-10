"""Eval harness: scoring rules, golden-set validity, and an end-to-end smoke run."""
import pytest

from evals.run_evals import load_golden, render_report, run_suite
from evals.scoring import matches, score_item
from src.financials import TABLE_NAMES
from src.ingest import Chunk, VectorIndex
from src.llm import LLM, AssistantTurn, ToolCall
from src.run_result import ANSWERED, ESCALATED, NOT_FOUND, RunResult, Step
from src.tools import TOOL_SPECS

ALL_TOOLS = {t.name for t in TOOL_SPECS}


@pytest.mark.parametrize("answer,alt,expected", [
    ("Revenue was $94M in Q4.", "94", True),
    ("Founded in 1994.", "94", False),
    ("It was 94.5 million.", "94", False),
    ("Grew 25.3% year over year", "25.3", True),
    ("Grew 25.3% year over year", "25", False),
    ("Total revenue was $842 million", "842", True),
    ("Revenue of 1,248 million", "1248", True),
    ("Uses PathFinder OS", "pathfinder", True),
    ("Grew 9%", "9%", True),
    ("Grew 29%", "9%", False),
])
def test_answer_matching(answer, alt, expected):
    assert matches(answer, alt) is expected


def test_golden_set_is_well_formed():
    items = load_golden()
    assert len(items) >= 25
    assert len({i["id"] for i in items}) == len(items)
    valid_sources = {f"p{n}" for n in range(1, 10)} | set(TABLE_NAMES)
    for i in items:
        assert i["expected_outcome"] in {ANSWERED, NOT_FOUND, ESCALATED}
        if i["expected_outcome"] == ANSWERED:
            assert i["must_include"] and i["citation_groups"], i["id"]
        for g in i["citation_groups"]:
            assert set(g) <= valid_sources, i["id"]
        assert set(i.get("also_allowed", [])) <= valid_sources, i["id"]
        for g in i["expected_tools"]:
            assert set(g) <= ALL_TOOLS, i["id"]


def _result(outcome, answer="", citations=(), tools=()):
    r = RunResult(system="tool_agent", question="q", outcome=outcome, answer=answer,
                  citations=list(citations))
    r.steps = [Step(kind="tool", name=t) for t in tools]
    return r


ITEM = {"id": "X01", "type": "mixed", "question": "q", "expected_outcome": ANSWERED,
        "must_include": [["94"], ["pathfinder", "attach"]],
        "citation_groups": [["segment_revenue_quarterly"], ["p4"]], "also_allowed": ["p3"],
        "expected_tools": [["query_financials"], ["search_document"]]}


def test_score_fully_correct():
    s = score_item(ITEM, _result(ANSWERED, "$94M, driven by PathFinder attach rates",
                                 ["p4", "segment_revenue_quarterly"],
                                 ["query_financials", "search_document", "final_answer"]), ALL_TOOLS)
    assert s.correct and s.citation_precision == 1 and s.citation_recall == 1 and s.tool_recall == 1


def test_score_partial_answer_and_bad_citation():
    s = score_item(ITEM, _result(ANSWERED, "$94M", ["segment_revenue_quarterly", "p9"],
                                 ["query_financials"]), ALL_TOOLS)
    assert not s.correct
    assert s.citation_precision == 0.5 and s.citation_recall == 0.5 and s.tool_recall == 0.5


def test_answering_a_should_escalate_question_is_a_false_answer():
    item = {"id": "E01", "type": "escalate", "question": "q", "expected_outcome": ESCALATED,
            "must_include": [], "citation_groups": [], "expected_tools": [["escalate_to_human"]]}
    s = score_item(item, _result(ANSWERED, "Buy it."), ALL_TOOLS)
    assert s.false_answer and not s.correct
    assert score_item(item, _result(ESCALATED), set()).tool_recall is None  # n/a for baselines


class RuleLLM(LLM):
    """Answers every question by escalating -- enough to smoke-test the harness."""

    def complete(self, system, user):
        self.usage.add(1, 1)
        return '{"found": false, "answer": "", "cited_pages": []}'

    def chat(self, system, messages, tools):
        self.usage.add(50, 10)
        return AssistantTurn(text="", tool_calls=[ToolCall("1", "escalate_to_human", {"reason": "test"})])


def test_harness_smoke_run_and_report():
    from src.ingest import embed
    chunks = [Chunk(text="Total revenue for fiscal year 2024 was 842 million", page=4, chunk_id=0)]
    index = VectorIndex(chunks, embed([c.text for c in chunks]))
    items = [i for i in load_golden() if i["id"] in {"L01", "E01", "N01"}]
    systems = ["fixed_rag", "tool_agent"]

    rows = run_suite(items, systems, RuleLLM(), index)
    assert len(rows) == 6
    report = render_report(rows, systems, items, "test:rule", 1)

    assert "## Overall" in report and "| tool_agent |" in report
    by = {(s.system, s.id): s for _, s in rows}
    assert by[("tool_agent", "E01")].correct
    assert by[("fixed_rag", "N01")].correct
    assert not by[("fixed_rag", "E01")].correct  # the baseline cannot escalate
