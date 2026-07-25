from src.ingest import Chunk, VectorIndex, chunk_page_text
import numpy as np


def test_chunking_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_page_text(text, page=1, chunk_size=800, overlap=120)

    assert len(chunks) > 1
    # First chunk should be exactly chunk_size tokens.
    assert len(chunks[0].split()) == 800
    # Consecutive chunks should share the overlap region.
    first_tail = chunks[0].split()[-120:]
    second_head = chunks[1].split()[:120]
    assert first_tail == second_head


def test_chunking_handles_short_page():
    chunks = chunk_page_text("just a few words here", page=1, chunk_size=800, overlap=120)
    assert chunks == ["just a few words here"]


def test_chunking_empty_page_returns_nothing():
    assert chunk_page_text("   ", page=1, chunk_size=800, overlap=120) == []


def test_vector_index_search_returns_best_match_first():
    chunks = [
        Chunk(text="revenue grew fourteen percent", page=3, chunk_id=0),
        Chunk(text="the board has seven directors", page=7, chunk_id=1),
    ]
    # Hand-built orthogonal-ish vectors so we know the expected ranking.
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    index = VectorIndex(chunks, vectors)

    query = np.array([0.9, 0.1], dtype="float32")
    results = index.search(query, top_k=2)

    assert results[0][0].page == 3
    assert results[0][1] > results[1][1]
