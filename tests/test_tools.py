"""Tools in isolation: read-only SQL enforcement, safe arithmetic, evidence tracking."""
import json

import pytest

from src.financials import QueryRejected, build_database, run_readonly_query
from src.ingest import Chunk, VectorIndex
from src.tools import Toolbox, safe_eval


def _toolbox():
    chunks = [Chunk(text="Total revenue for fiscal year 2024 was 842 million", page=4, chunk_id=0)]
    from src.ingest import embed
    return Toolbox(index=VectorIndex(chunks, embed([c.text for c in chunks])), db=build_database())


def test_quarterly_data_reconciles_to_annual_report_figures():
    db = build_database()
    rows = dict(db.execute(
        "SELECT segment || '-' || fiscal_year, SUM(revenue_musd) FROM segment_revenue_quarterly "
        "GROUP BY segment, fiscal_year").fetchall())
    assert rows["Robotic Arms-2024"] == 514
    assert rows["Warehouse Automation-2024"] == 328
    assert rows["Robotic Arms-2023"] + rows["Warehouse Automation-2023"] == 739
    assert round((514 / rows["Robotic Arms-2023"] - 1) * 100) == 9
    assert round((328 / rows["Warehouse Automation-2023"] - 1) * 100) == 23


@pytest.mark.parametrize("sql", [
    "UPDATE segment_revenue_quarterly SET revenue_musd = 100",
    "DELETE FROM annual_financials",
    "DROP TABLE annual_financials",
    "SELECT 1; DROP TABLE annual_financials",
    "PRAGMA query_only = OFF",
    "",
])
def test_sql_tool_rejects_anything_but_a_single_select(sql):
    with pytest.raises(QueryRejected):
        run_readonly_query(build_database(), sql)


def test_database_is_read_only_at_engine_level_too():
    db = build_database()
    with pytest.raises(Exception):
        db.execute("INSERT INTO annual_financials VALUES (2025, 'x', 1)")


def test_sql_tool_returns_rows_and_records_tables_as_evidence():
    tb = _toolbox()
    out = tb.run("query_financials", {"sql": "SELECT revenue_musd FROM segment_revenue_quarterly "
                                             "WHERE fiscal_year=2024 AND quarter=4 "
                                             "AND segment='Warehouse Automation'"})
    assert out.ok
    assert json.loads(out.content)["rows"] == [[94.0]]
    assert "segment_revenue_quarterly" in tb.seen_refs


def test_rejected_sql_comes_back_to_the_model_as_a_message_not_an_exception():
    tb = _toolbox()
    out = tb.run("query_financials", {"sql": "DELETE FROM annual_financials"})
    assert not out.ok and out.content.startswith("Rejected")
    assert tb.seen_tables == set()


def test_search_records_pages_as_evidence():
    tb = _toolbox()
    out = tb.run("search_document", {"query": "total revenue fiscal year 2024"})
    assert "[p4" in out.content
    assert tb.seen_refs == {"p4"}


def test_calculator():
    assert safe_eval("round((94 / 75 - 1) * 100, 1)") == 25.3
    assert safe_eval("122 + 126") == 248
    assert safe_eval("max(1, 2) - abs(-3)") == -1


@pytest.mark.parametrize("expr", [
    "__import__('os').system('ls')",
    "open('x')",
    "(1).__class__",
    "9 ** 9999",
    "x + 1",
])
def test_calculator_rejects_non_arithmetic(expr):
    with pytest.raises(Exception):
        safe_eval(expr)


def test_unknown_tool_is_an_error_message():
    out = _toolbox().run("send_email", {})
    assert not out.ok and "unknown tool" in out.content
