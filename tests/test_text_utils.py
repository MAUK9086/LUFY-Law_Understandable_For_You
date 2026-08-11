"""Tests for the pure text-processing utilities."""

from app.utils.text_utils import (
    clean_text,
    group_chunks_for_budget,
    split_into_chunks,
    split_into_chunks_with_sections,
)


def test_clean_text_collapses_whitespace_and_blank_lines():
    raw = "Hello   world\t\there.\n\n\n\nNext   line."
    cleaned = clean_text(raw)
    assert cleaned == "Hello world here.\n\nNext line."


def test_clean_text_normalises_unicode():
    # 'e' + combining acute accent should normalise to the single NFC codepoint.
    raw = "café"
    assert clean_text(raw) == "café"


def test_clean_text_strips_leading_and_trailing():
    assert clean_text("\n\n  padded  \n\n") == "padded"


def test_split_into_chunks_respects_size():
    text = "\n\n".join(f"Paragraph number {i} with some filler text." for i in range(20))
    chunks = split_into_chunks(text, chunk_size=120, overlap=20)
    assert len(chunks) > 1
    # Allow a little slack for the overlap carry prefix.
    assert all(len(c) <= 120 + 20 + 4 for c in chunks)


def test_split_into_chunks_overlap_carries_context():
    para_a = "A" * 90
    para_b = "B" * 90
    chunks = split_into_chunks(f"{para_a}\n\n{para_b}", chunk_size=100, overlap=15)
    assert len(chunks) == 2
    # The start of the second chunk should carry the tail of the first.
    assert "A" * 15 in chunks[1]


def test_split_long_paragraph_on_sentences():
    para = " ".join(f"Sentence {i} is here." for i in range(30))
    chunks = split_into_chunks(para, chunk_size=120, overlap=0)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_sections_align_with_chunks_and_detect_headers():
    text = (
        "DEFINITIONS\n\n"
        "The terms used in this agreement are defined as follows for clarity.\n\n"
        "1. TERMINATION\n\n"
        "Either party may terminate this agreement with thirty days written notice."
    )
    chunks, sections = split_into_chunks_with_sections(text, chunk_size=200, overlap=0)
    assert len(chunks) == len(sections)
    # Every chunk should be attributed to one of the detected headers.
    assert set(sections) <= {"DEFINITIONS", "1. TERMINATION"}
    assert "DEFINITIONS" in sections


def test_group_chunks_for_budget_packs_under_limit():
    chunks = [f"Chunk {i}: " + ("x" * 30) for i in range(10)]
    batches = group_chunks_for_budget(chunks, max_chars=100)
    assert len(batches) > 1
    assert all(len(b) <= 100 for b in batches)
    # No chunk's content should be lost across the batching.
    assert sum(b.count("Chunk") for b in batches) == len(chunks)


def test_group_chunks_for_budget_preserves_order():
    chunks = [f"Chunk-{i}" for i in range(6)]
    batches = group_chunks_for_budget(chunks, max_chars=1000)
    # Small chunks with a large budget should all land in one batch, in order.
    assert len(batches) == 1
    assert batches[0] == "\n\n".join(chunks)


def test_group_chunks_for_budget_oversized_single_chunk_stands_alone():
    huge = "x" * 500
    chunks = ["small", huge, "small-again"]
    batches = group_chunks_for_budget(chunks, max_chars=100)
    assert huge in batches
    # The oversized chunk isn't merged with its neighbours despite exceeding the budget alone.
    assert not any(huge in b and b != huge for b in batches)


def test_group_chunks_for_budget_empty_input():
    assert group_chunks_for_budget([], max_chars=100) == []
