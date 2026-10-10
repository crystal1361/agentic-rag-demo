"""
Control-flow tests for the tool-calling agent (src/agent.py).

ScriptedLLM replays a fixed sequence of assistant turns (tool calls), so each
test pins one trajectory and checks how the loop and its guardrails respond.
These test the *harness* -- tool execution, citation checking, step caps,
escalation -- not model quality; model quality is what evals/ measures.
"""
from src.agent import answer_question, normalize_citation
from src.ingest import Chunk, VectorIndex
from src.llm import LLM, AssistantTurn, ToolCall
from src.run_result import ANSWERED, ESCALATED, NOT_FOUND


class ScriptedLLM(LLM):
    def __init__(self, turns):
        super().__init__()
        self.turns = list(turns)
        self.seen_messages = []

    def complete(self, system, user):
        raise AssertionError("the tool agent should only use chat()")

    def chat(self, system, messages, tools):
        self.seen_messages.append([dict(m) for m in messages])
        self.usage.add(100, 20)
        if not self.turns:
            return AssistantTurn(text="(script exhausted)")
        return self.turns.pop(0)


def call(name, n=1, **args):
    return ToolCall(id=f"{name}-{n}", name=name, args=args)


def turn(*calls, text=""):
    return AssistantTurn(text=text, tool_calls=list(calls))


def _index():
    chunks = [
        Chunk(text="Warehouse Automation segment revenue grew 23 percent driven by PathFinder OS "
                   "software attach rates", page=4, chunk_id=0),
        Chunk(text="supply chain servo motors gearboxes sourced Taiwan South Korea", page=5, chunk_id=1),
    ]
    from src.ingest import embed
    return VectorIndex(chunks, embed([c.text for c in chunks]))


def _last_tool_result(llm):
    return llm.seen_messages[-1][-1]["results"]


def test_sql_then_calculate_then_answer():
    llm = ScriptedLLM([
        turn(call("query_financials", sql="SELECT quarter, revenue_musd FROM segment_revenue_quarterly "
                                          "WHERE segment='Warehouse Automation' AND quarter=4")),
        turn(call("calculate", expression="round((94 / 75 - 1) * 100, 1)")),
        turn(call("final_answer", found=True, answer="Q4 FY2024 WA revenue was $94M, up 25.3% YoY.",
                  citations=["segment_revenue_quarterly"])),
    ])
    result = answer_question("How fast did WA revenue grow in Q4 FY2024 vs Q4 FY2023?", _index(), llm)

    assert result.outcome == ANSWERED
    assert result.tools_used == ["query_financials", "calculate", "final_answer"]
    assert result.citations == ["segment_revenue_quarterly"]
    assert result.llm_calls == 3 and result.input_tokens == 300


def test_mixed_sources_in_one_turn():
    llm = ScriptedLLM([
        turn(call("query_financials", sql="SELECT revenue_musd FROM segment_revenue_quarterly WHERE "
                                          "fiscal_year=2024 AND quarter=4 AND segment='Warehouse Automation'"),
             call("search_document", query="Warehouse Automation growth driver PathFinder")),
        turn(call("final_answer", found=True, answer="$94M; driven by PathFinder OS attach rates.",
                  citations=["p4", "segment_revenue_quarterly"])),
    ])
    result = answer_question("Q4 WA revenue and what drove growth?", _index(), llm)
    assert result.outcome == ANSWERED
    assert set(result.citations) == {"p4", "segment_revenue_quarterly"}


def test_citation_of_a_source_never_retrieved_is_rejected_then_fixed():
    llm = ScriptedLLM([
        turn(call("search_document", query="Warehouse Automation growth driver")),
        turn(call("final_answer", found=True, answer="Driven by PathFinder OS.", citations=["p9"])),
        turn(call("final_answer", 2, found=True, answer="Driven by PathFinder OS.", citations=["p4"])),
    ])
    result = answer_question("What drove WA growth?", _index(), llm)

    assert result.outcome == ANSWERED
    assert result.citations == ["p4"]
    guardrails = [s for s in result.steps if s.kind == "guardrail"]
    assert len(guardrails) == 1 and "p9" in guardrails[0].detail


def test_answer_without_citations_is_rejected():
    llm = ScriptedLLM([
        turn(call("final_answer", found=True, answer="Probably about $100M.", citations=[])),
        turn(call("final_answer", 2, found=False, answer="", citations=[])),
    ])
    result = answer_question("What was revenue?", _index(), llm)
    assert result.outcome == NOT_FOUND
    assert any("at least one citation" in s.detail for s in result.steps if s.kind == "guardrail")


def test_not_found_is_honest():
    llm = ScriptedLLM([
        turn(call("search_document", query="CEO favorite color")),
        turn(call("final_answer", found=False, answer="", citations=[])),
    ])
    result = answer_question("What is the CEO's favorite color?", _index(), llm)
    assert result.outcome == NOT_FOUND and result.answer == "" and result.citations == []


def test_escalation():
    llm = ScriptedLLM([turn(call("escalate_to_human", reason="investment advice request"))])
    result = answer_question("Should I buy Northfield stock?", _index(), llm)
    assert result.outcome == ESCALATED
    assert "investment advice" in result.answer
    assert result.tools_used == ["escalate_to_human"]


def test_step_cap_stops_a_runaway_agent():
    llm = ScriptedLLM([turn(call("search_document", n, query=f"attempt {n}")) for n in range(1, 50)])
    result = answer_question("Anything", _index(), llm, max_steps=4)
    assert result.outcome == NOT_FOUND
    assert result.stop_reason.startswith("step_limit")
    assert result.llm_calls == 4


def test_identical_repeated_call_is_not_re_executed():
    llm = ScriptedLLM([
        turn(call("search_document", 1, query="servo motors gearboxes sourced Taiwan")),
        turn(call("search_document", 2, query="servo motors gearboxes sourced Taiwan")),
        turn(call("final_answer", found=True, answer="Taiwan and South Korea.", citations=["p5"])),
    ])
    result = answer_question("Where do servo motors come from?", _index(), llm)
    assert result.tools_used.count("search_document") == 1
    assert any("repeated" in s.detail for s in result.steps if s.kind == "guardrail")
    assert result.outcome == ANSWERED


def test_terminal_tool_with_other_tools_in_same_turn_is_rejected():
    llm = ScriptedLLM([
        turn(call("search_document", query="servo motors gearboxes sourced Taiwan"),
             call("final_answer", found=True, answer="guess", citations=["p5"])),
        turn(call("final_answer", 2, found=True, answer="Taiwan and South Korea.", citations=["p5"])),
    ])
    result = answer_question("Where do servo motors come from?", _index(), llm)
    assert result.outcome == ANSWERED and result.answer == "Taiwan and South Korea."
    assert result.llm_calls == 2


def test_turn_without_tool_calls_gets_a_nudge():
    llm = ScriptedLLM([
        AssistantTurn(text="Let me think about this."),
        turn(call("escalate_to_human", reason="out of scope")),
    ])
    result = answer_question("Write me a poem", _index(), llm)
    assert result.outcome == ESCALATED
    assert llm.seen_messages[1][-1]["role"] == "user"


def test_rejected_sql_write_is_reported_back_to_the_model():
    llm = ScriptedLLM([
        turn(call("query_financials", sql="UPDATE segment_revenue_quarterly SET revenue_musd = 100")),
        turn(call("escalate_to_human", reason="user asked to modify data")),
    ])
    result = answer_question("Set Q4 WA revenue to 100", _index(), llm)
    assert _last_tool_result(llm)[0]["content"].startswith("Rejected")
    assert result.outcome == ESCALATED


def test_citation_normalization():
    assert normalize_citation("page 4") == "p4"
    assert normalize_citation("4") == "p4"
    assert normalize_citation("[p5]") == "p5"
    assert normalize_citation("table:Segment_Revenue_Quarterly") == "segment_revenue_quarterly"
