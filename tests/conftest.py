"""
Test fixtures.

Tests never call a real LLM API or load sentence-transformers/torch --
that would make CI slow, flaky, and dependent on API keys. Instead:

- `fake_embed` (autouse) replaces the embedding backend with a cheap,
  deterministic bag-of-words hash so cosine similarity still behaves
  sensibly (shared keywords -> higher similarity) without any ML
  dependency.
- `FakeLLM` (in test files) implements the same `LLM` interface as the
  real providers, with each of the four agent decision points
  (decompose / answer / judge / synthesize) individually controllable,
  so each test can pin exactly the decision it's trying to verify.
"""
import hashlib

import numpy as np
import pytest

DIM = 64


def _hash_bow_embed(texts: list[str]) -> np.ndarray:
    vectors = np.zeros((len(texts), DIM), dtype="float32")
    for i, text in enumerate(texts):
        for word in text.lower().split():
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
            vectors[i, bucket] += 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return vectors / norms


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    monkeypatch.setattr("src.ingest.embed", _hash_bow_embed)
    monkeypatch.setattr("src.retriever.embed", _hash_bow_embed)
    yield
