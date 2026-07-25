"""
Embedding backend, deliberately decoupled from the LLM provider.

Anthropic has no embeddings endpoint, and forcing an OpenAI key just to
embed text would defeat the point of being provider-agnostic. Running
embeddings locally via sentence-transformers means the retrieval layer
never depends on which vendor is doing the reasoning, and the whole
thing works offline / without any API key at all until you actually
want the agent to reason.
"""
from functools import lru_cache

import numpy as np

from src.config import CONFIG


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(CONFIG.embedding_model)


def embed(texts: list[str]) -> np.ndarray:
    """Returns float32 array of shape (len(texts), dim), L2-normalized
    so that inner product == cosine similarity (what FAISS IndexFlatIP expects)."""
    vectors = _model().encode(texts, convert_to_numpy=True, show_progress_bar=False)
    vectors = vectors.astype("float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return vectors / norms
