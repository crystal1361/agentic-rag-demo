"""
Deterministic scoring for one (golden item, RunResult) pair.

Deliberately rule-based rather than LLM-as-judge: every score is
reproducible, free to recompute, and explainable line by line. Each
golden item specifies:

  expected_outcome  answered | not_found | escalated
  must_include      list of groups; each group is a list of acceptable
                    strings, and the answer must match >= 1 string from
                    every group (numbers match on digit boundaries, so
                    "94" matches "$94M" but not "1994" or "94.5")
  citation_groups   list of groups of acceptable sources ("p4", table
                    names); recall = share of groups cited
  also_allowed      extra sources that are fine to cite (precision only)
  expected_tools    list of groups of tool names; the run must use >= 1
                    tool from every group (trajectory check -- only
                    meaningful for the tool agent, reported as n/a for
                    systems that don't have those tools)

Metrics per item:
  outcome_correct     right kind of response (answer / decline / escalate)
  correct             outcome correct AND (if answering) content correct
  false_answer        answered something it should have declined/escalated
  citation_precision  share of cited sources that are acceptable
  citation_recall     share of required source groups that were cited
  tool_recall         share of expected tool groups the trajectory used
"""
import re
from dataclasses import dataclass

from src.run_result import ANSWERED

_NUMERIC = re.compile(r"^[\d.,%$]+$")


def _normalize(text: str) -> str:
    text = text.lower().replace("–", "-").replace("—", "-")
    return re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 1,234 -> 1234


def matches(answer: str, alternative: str) -> bool:
    answer, alt = _normalize(answer), _normalize(alternative)
    if _NUMERIC.match(alt):
        pattern = rf"(?<![\d.]){re.escape(alt)}(?!\d|\.\d)"
        return re.search(pattern, answer) is not None
    return alt in answer


@dataclass
class ItemScore:
    id: str
    type: str
    system: str
    outcome: str
    expected_outcome: str
    outcome_correct: bool
    correct: bool
    false_answer: bool
    citation_precision: float | None
    citation_recall: float | None
    tool_recall: float | None
    missing: list[str]


def score_item(item: dict, result, tools_available: set[str]) -> ItemScore:
    expected = item["expected_outcome"]
    outcome_correct = result.outcome == expected
    missing: list[str] = []

    content_ok = True
    if expected == ANSWERED and result.outcome == ANSWERED:
        for group in item["must_include"]:
            if not any(matches(result.answer, alt) for alt in group):
                content_ok = False
                missing.append("answer lacks one of " + "/".join(group))

    precision = recall = None
    if expected == ANSWERED and result.outcome == ANSWERED and item["citation_groups"]:
        allowed = {s for g in item["citation_groups"] for s in g} | set(item.get("also_allowed", []))
        cited = set(result.citations)
        precision = (len(cited & allowed) / len(cited)) if cited else 0.0
        satisfied = sum(1 for g in item["citation_groups"] if cited & set(g))
        recall = satisfied / len(item["citation_groups"])
        if recall < 1:
            missing.append("missing citation group(s)")

    tool_recall = None
    groups = [g for g in item.get("expected_tools", []) if set(g) & tools_available]
    if groups and len(groups) == len(item.get("expected_tools", [])):
        used = set(result.tools_used)
        hit = sum(1 for g in groups if used & set(g))
        tool_recall = hit / len(groups)
        if tool_recall < 1:
            missing.append("trajectory skipped an expected tool")

    return ItemScore(
        id=item["id"], type=item["type"], system=result.system,
        outcome=result.outcome, expected_outcome=expected,
        outcome_correct=outcome_correct,
        correct=outcome_correct and content_ok,
        false_answer=(expected != ANSWERED and result.outcome == ANSWERED),
        citation_precision=precision, citation_recall=recall, tool_recall=tool_recall,
        missing=missing,
    )
