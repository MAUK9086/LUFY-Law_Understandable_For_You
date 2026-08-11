"""Tests for the in-memory hybrid vector store.

Uses hand-built fake embeddings so no embedding model is loaded.
"""

from app.core import vector_store


def _make_session():
    chunks = [
        "The tenant must pay rent on the first of each month.",
        "Either party may terminate the lease with thirty days notice.",
        "The landlord is responsible for major structural repairs.",
    ]
    # Distinct one-hot-ish vectors so dense similarity is deterministic.
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    original_text = "ORIGINAL CLEAN TEXT — no overlap duplication here."
    sections = ["RENT", "TERMINATION", "REPAIRS"]
    sid = vector_store.create_session(chunks, embeddings, original_text, sections)
    return sid, chunks, original_text


def test_session_create_retrieve_delete():
    sid, chunks, _ = _make_session()
    try:
        assert vector_store.session_exists(sid)
        # Query vector aligned with the termination chunk; query text also lexical.
        results = vector_store.retrieve(
            sid, [0.0, 1.0, 0.0], "how do I terminate the lease early", top_k=2
        )
        assert len(results) == 2
        assert results[0].text == chunks[1]
        assert results[0].score >= results[1].score
    finally:
        vector_store.delete_session(sid)
    assert not vector_store.session_exists(sid)


def test_get_full_text_returns_original_not_joined_chunks():
    sid, chunks, original_text = _make_session()
    try:
        full = vector_store.get_full_text(sid)
        assert full == original_text
        # It must NOT be the chunks re-joined (the old duplicating behaviour).
        assert full != "\n\n".join(chunks)
    finally:
        vector_store.delete_session(sid)


def test_delete_nonexistent_session_is_safe():
    vector_store.delete_session("does-not-exist")  # must not raise


def test_get_chunks_returns_original_order():
    sid, chunks, _ = _make_session()
    try:
        assert vector_store.get_chunks(sid) == chunks
    finally:
        vector_store.delete_session(sid)
