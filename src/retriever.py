"""
Query -> top-k passages above a similarity threshold.

The threshold is what makes "I don't know" possible without asking the
LLM to police itself: if nothing clears the bar, we never even call
the model for that hop -- we short-circuit straight to NOT_FOUND. This
also happens to be a real cost/latency win at scale, not just a safety
one.
"""
from dataclasses import dataclass

from src.config import CONFIG
from src.embeddings import embed
from src.ingest import Chunk, VectorIndex


@dataclass
class RetrievedPassage:
    chunk: Chunk
    score: float


def retrieve(index: VectorIndex, query: str, top_k: int | None = None,
              threshold: float | None = None) -> list[RetrievedPassage]:
    top_k = top_k or CONFIG.top_k
    threshold = CONFIG.similarity_threshold if threshold is None else threshold

    query_vec = embed([query])[0]
    hits = index.search(query_vec, top_k)
    return [RetrievedPassage(chunk=c, score=s) for c, s in hits if s >= threshold]


def format_passages(passages: list[RetrievedPassage]) -> str:
    """Render retrieved passages as labeled context blocks the LLM can cite by page."""
    if not passages:
        return "(no passages retrieved)"
    return "\n\n".join(
        f"[Page {p.chunk.page}] {p.chunk.text}" for p in passages
    )
