from src.ingest import Chunk, VectorIndex
from src.retriever import retrieve


def _build_index():
    chunks = [
        Chunk(text="Robotic Arms segment revenue grew nine percent", page=3, chunk_id=0),
        Chunk(text="the board has seven directors four independent", page=7, chunk_id=1),
        Chunk(text="supply chain servo motors gearboxes Taiwan South Korea", page=4, chunk_id=2),
    ]
    # fake_embed fixture (autouse) makes these deterministic bag-of-words vectors.
    from src.ingest import embed
    vectors = embed([c.text for c in chunks])
    return VectorIndex(chunks, vectors)


def test_retrieve_finds_relevant_passage_above_threshold():
    index = _build_index()
    results = retrieve(index, "How much did Robotic Arms revenue grow?", top_k=3, threshold=0.05)
    assert results, "expected at least one passage to clear a low threshold"
    assert results[0].chunk.page == 3


def test_retrieve_returns_nothing_when_threshold_too_high():
    index = _build_index()
    results = retrieve(index, "What color is the company logo?", top_k=3, threshold=0.99)
    assert results == []
