"""
The agent's tools: what the model is allowed to do, and nothing more.

Four kinds of tool:

- search_document   -- semantic search over the PDF (returns page-labelled passages)
- query_financials  -- read-only SQL over the structured financial tables
- calculate         -- safe arithmetic (no LLM mental math on financial figures)
- final_answer / escalate_to_human -- the two *terminal* tools; every run
  must end in one of them (or hit the step cap)

A `Toolbox` instance is created per question. Besides executing tools,
it records the *evidence* the model has actually seen this run (pages
returned by search, tables successfully queried). The agent uses that
record to verify citations in final_answer -- a model may only cite
sources it really retrieved.
"""
import ast
import json
import math
import operator
import sqlite3
from dataclasses import dataclass, field

from src.financials import SCHEMA_DESCRIPTION, QueryRejected, run_readonly_query
from src.ingest import VectorIndex
from src.llm import ToolSpec
from src.retriever import retrieve

TERMINAL_TOOLS = {"final_answer", "escalate_to_human"}

TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="search_document",
        description=(
            "Semantic search over the Northfield Robotics FY2024 annual report (PDF). "
            "Returns the most relevant passages, each labelled with its page number. "
            "Use for narrative facts: company overview, segments, annual results, risk "
            "factors, R&D, executive compensation, ESG, legal. The report does NOT contain "
            "quarterly figures. Use a short, specific query; if results are empty or off-"
            "topic, try different wording rather than repeating the same query."
        ),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="query_financials",
        description=(
            "Run ONE read-only SQLite SELECT against the structured financial database. "
            "Use for quarterly segment revenue (only available here) and for exact annual "
            "figures. Writes are impossible.\n" + SCHEMA_DESCRIPTION
        ),
        parameters={
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single SELECT statement"}},
            "required": ["sql"],
        },
    ),
    ToolSpec(
        name="calculate",
        description=(
            "Evaluate an arithmetic expression exactly. Supports + - * / ** parentheses "
            "and round(x, n), abs, min, max. Use for ANY growth rate, ratio, percentage or "
            "sum instead of computing it yourself. Example: round((94 / 75 - 1) * 100, 1)"
        ),
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    ),
    ToolSpec(
        name="final_answer",
        description=(
            "Finish with the answer. Call this alone (not alongside other tools). "
            "If the sources you retrieved do not contain the answer, set found=false and "
            "leave answer empty -- never guess or use outside knowledge. When found=true, "
            "citations must list every source the answer relies on, using only sources "
            "returned to you in this conversation: pages as \"p4\", tables by name, e.g. "
            "\"segment_revenue_quarterly\"."
        ),
        parameters={
            "type": "object",
            "properties": {
                "found": {"type": "boolean"},
                "answer": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["found", "answer", "citations"],
        },
    ),
    ToolSpec(
        name="escalate_to_human",
        description=(
            "Hand the request to a human instead of answering. Use when the user asks for "
            "investment, legal or tax advice or a personal recommendation; asks you to change "
            "data or take any action (your tools are read-only); or asks something outside "
            "the scope of this company's report and financial data where a human should decide. "
            "Do NOT escalate merely because a fact is missing from the sources -- use "
            "final_answer with found=false for that."
        ),
        parameters={
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    ),
]


@dataclass
class ToolOutcome:
    content: str          # what the model sees
    ok: bool = True
    summary: str = ""     # short human-readable line for the trace


@dataclass
class Toolbox:
    index: VectorIndex
    db: sqlite3.Connection
    seen_pages: set[int] = field(default_factory=set)
    seen_tables: set[str] = field(default_factory=set)

    @property
    def seen_refs(self) -> set[str]:
        return {f"p{p}" for p in self.seen_pages} | set(self.seen_tables)

    def run(self, name: str, args: dict) -> ToolOutcome:
        handler = {
            "search_document": self._search,
            "query_financials": self._query,
            "calculate": self._calculate,
        }.get(name)
        if handler is None:
            return ToolOutcome(f"Error: unknown tool {name!r}", ok=False, summary="unknown tool")
        try:
            return handler(args)
        except Exception as e:  # tool errors go back to the model, never crash the run
            return ToolOutcome(f"Error: {type(e).__name__}: {e}", ok=False, summary=f"error: {e}")

    # -- tools -------------------------------------------------------------

    def _search(self, args: dict) -> ToolOutcome:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolOutcome("Error: query is required", ok=False, summary="empty query")
        passages = retrieve(self.index, query)
        if not passages:
            return ToolOutcome("No passages cleared the relevance threshold. Try different wording.",
                               summary="0 passages")
        pages = []
        lines = []
        for p in passages:
            self.seen_pages.add(p.chunk.page)
            pages.append(p.chunk.page)
            lines.append(f"[p{p.chunk.page} | score {p.score:.2f}] {p.chunk.text}")
        return ToolOutcome("\n\n".join(lines),
                           summary=f"{len(passages)} passage(s), pages {sorted(set(pages))}")

    def _query(self, args: dict) -> ToolOutcome:
        sql = str(args.get("sql", ""))
        try:
            columns, rows, tables, truncated = run_readonly_query(self.db, sql)
        except QueryRejected as e:
            return ToolOutcome(f"Rejected: {e}", ok=False, summary=f"rejected: {e}")
        self.seen_tables.update(tables)
        payload = {"columns": columns, "rows": [list(r) for r in rows],
                   "source_tables": tables}
        if truncated:
            payload["note"] = "result truncated; add filters or aggregation"
        return ToolOutcome(json.dumps(payload), summary=f"{len(rows)} row(s) from {tables}")

    def _calculate(self, args: dict) -> ToolOutcome:
        expr = str(args.get("expression", ""))
        value = safe_eval(expr)
        return ToolOutcome(json.dumps({"expression": expr, "result": value}),
                           summary=f"{expr} = {value}")


# --------------------------------------------------------------------------
# Safe arithmetic: an AST whitelist, never eval()
# --------------------------------------------------------------------------

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max}


def safe_eval(expression: str) -> float:
    if len(expression) > 300:
        raise ValueError("expression too long")
    tree = ast.parse(expression, mode="eval")
    result = _eval_node(tree.body)
    if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
        raise ValueError("result is not a finite number")
    return result


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")
