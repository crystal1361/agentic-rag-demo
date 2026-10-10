"""
A small structured "financial database" for the fictional Northfield
Robotics, exposed to the agent through a read-only SQL tool.

Why this exists: the annual report PDF only states *annual* segment
revenue. Quarterly segment revenue lives only here. So some questions
("What was Warehouse Automation revenue in Q4 FY2024?") can only be
answered from the table, some only from the PDF, and some need both --
which is exactly the situation where an agent has to decide which tool
to use, rather than run one fixed retrieval step.

The quarterly figures are fictional but reconcile to the annual numbers
in the PDF (FY2024: Robotic Arms $514M, Warehouse Automation $328M;
FY2023: $472M and $267M, which imply the report's 9% and 23% growth).

Safety: the database lives in memory, is opened with `PRAGMA query_only`,
and `run_readonly_query` additionally rejects anything that isn't a
single SELECT / WITH statement and caps the number of returned rows.
"""
import re
import sqlite3

from src.config import CONFIG

# (fiscal_year, quarter, segment, revenue_musd)
SEGMENT_REVENUE_QUARTERLY = [
    (2023, 1, "Robotic Arms", 113), (2023, 2, "Robotic Arms", 116),
    (2023, 3, "Robotic Arms", 119), (2023, 4, "Robotic Arms", 124),
    (2023, 1, "Warehouse Automation", 60), (2023, 2, "Warehouse Automation", 64),
    (2023, 3, "Warehouse Automation", 68), (2023, 4, "Warehouse Automation", 75),
    (2024, 1, "Robotic Arms", 122), (2024, 2, "Robotic Arms", 126),
    (2024, 3, "Robotic Arms", 130), (2024, 4, "Robotic Arms", 136),
    (2024, 1, "Warehouse Automation", 72), (2024, 2, "Warehouse Automation", 78),
    (2024, 3, "Warehouse Automation", 84), (2024, 4, "Warehouse Automation", 94),
]

# (fiscal_year, metric, value_musd) -- annual figures that also appear in the PDF.
ANNUAL_FINANCIALS = [
    (2023, "total_revenue", 739), (2024, "total_revenue", 842),
    (2023, "net_income", 71), (2024, "net_income", 96),
    (2023, "rnd_expense", 94), (2024, "rnd_expense", 112),
    (2024, "operating_expenses", 268),
]

SCHEMA_DESCRIPTION = """Tables (SQLite, all money in USD millions, fiscal years 2023-2024):
- segment_revenue_quarterly(fiscal_year INTEGER, quarter INTEGER 1-4, segment TEXT, revenue_musd REAL)
    segment is one of 'Robotic Arms', 'Warehouse Automation'
- annual_financials(fiscal_year INTEGER, metric TEXT, value_musd REAL)
    metric is one of 'total_revenue', 'net_income', 'rnd_expense', 'operating_expenses'"""

TABLE_NAMES = ("segment_revenue_quarterly", "annual_financials")

_ALLOWED_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex)\b",
    re.IGNORECASE,
)


class QueryRejected(ValueError):
    """Raised when a query is not a single read-only SELECT."""


def build_database() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("CREATE TABLE segment_revenue_quarterly "
                 "(fiscal_year INTEGER, quarter INTEGER, segment TEXT, revenue_musd REAL)")
    conn.execute("CREATE TABLE annual_financials "
                 "(fiscal_year INTEGER, metric TEXT, value_musd REAL)")
    conn.executemany("INSERT INTO segment_revenue_quarterly VALUES (?, ?, ?, ?)",
                     SEGMENT_REVENUE_QUARTERLY)
    conn.executemany("INSERT INTO annual_financials VALUES (?, ?, ?)", ANNUAL_FINANCIALS)
    conn.commit()
    conn.execute("PRAGMA query_only = ON")  # engine-level backstop: no writes at all
    return conn


def run_readonly_query(conn: sqlite3.Connection, sql: str,
                       row_limit: int | None = None
                       ) -> tuple[list[str], list[tuple], list[str], bool]:
    """Returns (columns, rows, tables_referenced, truncated). Raises QueryRejected
    for anything other than a single read-only SELECT/WITH statement."""
    row_limit = row_limit or CONFIG.sql_row_limit
    statement = sql.strip().rstrip(";").strip()
    if not statement:
        raise QueryRejected("empty query")
    if ";" in statement:
        raise QueryRejected("only a single statement is allowed")
    if not _ALLOWED_START.match(statement) or _FORBIDDEN.search(statement):
        raise QueryRejected("only read-only SELECT queries are allowed")

    cur = conn.execute(statement)
    columns = [d[0] for d in cur.description or []]
    rows = cur.fetchmany(row_limit + 1)
    truncated = len(rows) > row_limit
    rows = rows[:row_limit]
    tables = [t for t in TABLE_NAMES if re.search(rf"\b{t}\b", statement, re.IGNORECASE)]
    return columns, rows, tables, truncated
