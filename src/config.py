"""
Central config, loaded from environment (.env). Keeping every tunable
in one place is what makes it easy to explain "these are the real
parameters" in an interview, and easy to swap providers without
touching the pipeline code.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _opt_float(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else None


@dataclass(frozen=True)
class Config:
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

    # v1 bounded loop (decompose / judge): max retrieval hops per sub-question.
    max_hops: int = int(os.getenv("MAX_HOPS", "3"))
    # Tool-calling agent: hard cap on model turns per question.
    max_steps: int = int(os.getenv("MAX_STEPS", "8"))
    # Max rows a single SQL tool call may return to the model.
    sql_row_limit: int = int(os.getenv("SQL_ROW_LIMIT", "50"))

    # Where per-question JSONL traces are written ("" disables tracing).
    trace_dir: str = os.getenv("TRACE_DIR", "traces")

    # Optional: USD per million tokens, used only to add a cost column to
    # eval reports. Left unset by default because list prices change --
    # fill in the current numbers for the model you run.
    price_input_per_mtok: float | None = _opt_float("PRICE_INPUT_PER_MTOK")
    price_output_per_mtok: float | None = _opt_float("PRICE_OUTPUT_PER_MTOK")


CONFIG = Config()
