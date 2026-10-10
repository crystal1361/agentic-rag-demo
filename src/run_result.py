"""
One result shape for every system (fixed RAG baseline, v1 loop, tool agent),
so the UI, the trace log and the eval harness can treat them identically.
"""
from dataclasses import asdict, dataclass, field

# outcome values
ANSWERED = "answered"
NOT_FOUND = "not_found"
ESCALATED = "escalated"


@dataclass
class Step:
    kind: str            # "tool" | "guardrail" | "decompose" | "retrieve" | "judge" | "synthesize" | "note"
    name: str = ""       # tool name, when kind == "tool"
    detail: str = ""     # one-line human-readable summary
    args: dict = field(default_factory=dict)
    ok: bool = True
    latency_ms: float = 0.0


@dataclass
class RunResult:
    system: str
    question: str
    outcome: str = NOT_FOUND
    answer: str = ""
    citations: list[str] = field(default_factory=list)   # "p4", "segment_revenue_quarterly", ...
    stop_reason: str = ""
    steps: list[Step] = field(default_factory=list)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    @property
    def tools_used(self) -> list[str]:
        return [s.name for s in self.steps if s.kind == "tool"]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tools_used"] = self.tools_used
        return d
