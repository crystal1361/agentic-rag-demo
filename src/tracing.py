"""
Structured run traces: one JSON line per question, appended to
traces/<date>.jsonl. Each line holds the full step-by-step trajectory
(every tool call with its arguments, result summary, latency, and every
guardrail intervention) plus token usage and the final outcome.

That's the minimum you'd want in production to answer "why did the
agent say that?" after the fact, and it's the same record the eval
harness scores trajectories from.
"""
import json
import os
from datetime import datetime, timezone

from src.config import CONFIG
from src.run_result import RunResult


def write_trace(result: RunResult, trace_dir: str | None = None) -> str | None:
    trace_dir = CONFIG.trace_dir if trace_dir is None else trace_dir
    if not trace_dir:
        return None
    os.makedirs(trace_dir, exist_ok=True)
    now = datetime.now(timezone.utc)
    path = os.path.join(trace_dir, f"{now:%Y-%m-%d}.jsonl")
    record = {"timestamp": now.isoformat(), **result.to_dict()}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return path
