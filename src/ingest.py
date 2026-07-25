"""
PDF -> chunks with page-number metadata -> FAISS index.

Same shape as the production RAG pipeline this project is modeled on:
chunk_size=800 / overlap=120 tokens (approximated here with whitespace
tokens, which is close enough for a demo corpus), cosine similarity via
a normalized inner-product FAISS index, and page numbers captured at
ingestion time so citations are a property of retrieval, not something
bolted on afterward.
"""
from dataclasses import dataclass

import faiss
import numpy as np
from pypdf import PdfReader

from src.config import CONFIG
from src.embeddings import embed


@dataclass
class Chunk:
    text: str
    page: int          # 1-indexed page number, captured at ingestion
    chunk_id: int


class VectorIndex:
    """Thin wrapper around a FAISS flat inner-product index + parallel chunk list."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = chunks
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        scores, ids = self.index.search(query_vector.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results


def _tokenize(text: str) -> list[str]:
    return text.split()


def chunk_page_text(text: str, page: int, chunk_size: int, overlap: int) -> list[str]:
    """Whitespace-token sliding window chunking with overlap. Simple and
    transparent -- easy to explain in an interview, which matters more
    here than squeezing out the last bit of retrieval quality."""
    tokens = _tokenize(text)
    if not tokens:
        return []
    chunks, start = [], 0
    step = max(chunk_size - overlap, 1)
    while start < len(tokens):
        piece = tokens[start:start + chunk_size]
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(tokens):
            break
        start += step
    return chunks


def ingest_pdf(path: str) -> VectorIndex:
    reader = PdfReader(path)
    chunks: list[Chunk] = []
    chunk_id = 0
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for piece in chunk_page_text(text, page_num, CONFIG.chunk_size, CONFIG.chunk_overlap):
            if piece.strip():
                chunks.append(Chunk(text=piece, page=page_num, chunk_id=chunk_id))
                chunk_id += 1

    if not chunks:
        raise ValueError(f"No extractable text found in {path}")

    vectors = embed([c.text for c in chunks])
    return VectorIndex(chunks, vectors)
