"""Tests for the map-reduce long-document paths in llm_client, plus the
async Groq call wrapper's retry/backoff and bounded-concurrency helpers.

The Groq boundary (_call_groq / _call_groq_async / AsyncGroq / analyse_risks)
is monkeypatched or faked throughout so these tests run offline with no
network calls or API key, matching the rest of the suite's approach.
"""

import asyncio

import pytest

from app.core import llm_client


def _run(coro):
    return asyncio.run(coro)


# ── map-reduce orchestration ─────────────────────────────────────────────


def test_summarise_document_long_maps_then_reduces(monkeypatch):
    map_calls: list[str] = []
    reduce_calls: list[str] = []

    async def fake_summarise_chunk_notes_async(text, persona):
        map_calls.append(text)
        return f"NOTES[{text[:10]}]"

    def fake_summarise_document(text, persona, language):
        reduce_calls.append(text)
        return "FINAL SUMMARY"

    monkeypatch.setattr(llm_client, "_summarise_chunk_notes_async", fake_summarise_chunk_notes_async)
    monkeypatch.setattr(llm_client, "summarise_document", fake_summarise_document)

    chunks = [f"chunk-{i} " + ("x" * 50) for i in range(6)]
    result = _run(llm_client.summarise_document_long(chunks, "tenant", "English", batch_chars=100))

    assert result == "FINAL SUMMARY"
    # Multiple map calls happened (document was split into more than one batch).
    assert len(map_calls) > 1
    # The reduce step ran exactly once, over the concatenated map outputs.
    assert len(reduce_calls) == 1
    assert all(f"NOTES[chunk-{i}" in reduce_calls[0] for i in range(6))


def test_summarise_document_long_single_batch_still_reduces(monkeypatch):
    async def fake_notes(text, persona):
        return "NOTES"

    monkeypatch.setattr(llm_client, "_summarise_chunk_notes_async", fake_notes)
    monkeypatch.setattr(llm_client, "summarise_document", lambda text, persona, language: "SUMMARY")

    result = _run(
        llm_client.summarise_document_long(["one small chunk"], "general", "English", batch_chars=10_000)
    )
    assert result == "SUMMARY"


def test_summarise_document_long_respects_concurrency_setting(monkeypatch):
    peak = {"current": 0, "max": 0}

    async def fake_notes(text, persona):
        peak["current"] += 1
        peak["max"] = max(peak["max"], peak["current"])
        await asyncio.sleep(0.01)
        peak["current"] -= 1
        return "NOTES"

    monkeypatch.setattr(llm_client, "_summarise_chunk_notes_async", fake_notes)
    monkeypatch.setattr(llm_client, "summarise_document", lambda text, persona, language: "SUMMARY")
    monkeypatch.setattr(llm_client.settings, "map_reduce_max_concurrency", 2)

    chunks = [f"chunk-{i} " + ("x" * 50) for i in range(8)]
    _run(llm_client.summarise_document_long(chunks, "general", "English", batch_chars=50))

    assert peak["max"] <= 2


def test_analyse_risks_long_merges_and_dedupes_across_batches(monkeypatch):
    call_count = {"n": 0}

    async def fake_analyse_risks_async(text, persona):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {
                "red_flags": [{"clause": "Auto-renewal", "explanation": "e1", "advice": "a1"}],
                "yellow_flags": [],
                "green_flags": [],
            }
        return {
            # Same clause (different case/whitespace) re-surfaces from the
            # overlap region shared with the previous batch — must be deduped.
            "red_flags": [{"clause": "  auto-renewal  ", "explanation": "e1-dup", "advice": "a1-dup"}],
            "yellow_flags": [{"clause": "Vague notice period", "explanation": "e2", "advice": "a2"}],
            "green_flags": [],
        }

    monkeypatch.setattr(llm_client, "_analyse_risks_async", fake_analyse_risks_async)

    chunks = [f"chunk-{i} " + ("x" * 50) for i in range(6)]
    merged = _run(llm_client.analyse_risks_long(chunks, "tenant", batch_chars=100))

    assert call_count["n"] > 1
    assert len(merged["red_flags"]) == 1  # deduped
    assert merged["red_flags"][0]["explanation"] == "e1"  # first occurrence kept
    assert len(merged["yellow_flags"]) == 1
    assert merged["green_flags"] == []


def test_merge_risk_dicts_ignores_malformed_entries():
    raw_dicts = [
        {"red_flags": "not a list", "yellow_flags": [{"clause": "ok", "explanation": "x", "advice": "y"}]},
        {"red_flags": [{"clause": "fine"}], "yellow_flags": [{"not_a_clause_field": True}]},
    ]
    merged = llm_client._merge_risk_dicts(raw_dicts)
    assert merged["red_flags"] == [{"clause": "fine"}]
    assert merged["yellow_flags"] == [{"clause": "ok", "explanation": "x", "advice": "y"}]
    assert merged["green_flags"] == []


# ── _gather_bounded ───────────────────────────────────────────────────────


def test_gather_bounded_limits_concurrency_and_preserves_order():
    peak = {"current": 0, "max": 0}

    async def task(i):
        peak["current"] += 1
        peak["max"] = max(peak["max"], peak["current"])
        await asyncio.sleep(0.01)
        peak["current"] -= 1
        return i

    results = _run(llm_client._gather_bounded([task(i) for i in range(6)], max_concurrency=2))

    assert results == list(range(6))
    assert peak["max"] <= 2


# ── _call_groq_async retry/backoff ───────────────────────────────────────


class _FakeRateLimitError(Exception):
    status_code = 429


class _FakeServerError(Exception):
    status_code = 500


def _fake_async_groq(create_fn):
    """Build a fake AsyncGroq class whose chat.completions.create is create_fn."""

    class _FakeCompletions:
        async def create(self, **kwargs):
            return await create_fn(**kwargs)

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeAsyncGroq:
        def __init__(self, api_key):
            self.chat = _FakeChat()

    return _FakeAsyncGroq


def _fake_response(content: str):
    class _Message:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Message(content)

    class _Response:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    return _Response(content)


def test_call_groq_async_retries_then_succeeds_on_rate_limit(monkeypatch):
    monkeypatch.setattr(llm_client.settings, "groq_api_key", "fake-key")
    monkeypatch.setattr(llm_client, "_BASE_BACKOFF_SECONDS", 0.0)

    call_count = {"n": 0}

    async def create_fn(**kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise _FakeRateLimitError("rate limited")
        return _fake_response("OK")

    monkeypatch.setattr(llm_client, "AsyncGroq", _fake_async_groq(create_fn))

    result = _run(llm_client._call_groq_async([{"role": "user", "content": "hi"}]))
    assert result == "OK"
    assert call_count["n"] == 3


def test_call_groq_async_does_not_retry_non_rate_limit_errors(monkeypatch):
    monkeypatch.setattr(llm_client.settings, "groq_api_key", "fake-key")

    call_count = {"n": 0}

    async def create_fn(**kwargs):
        call_count["n"] += 1
        raise _FakeServerError("boom")

    monkeypatch.setattr(llm_client, "AsyncGroq", _fake_async_groq(create_fn))

    with pytest.raises(_FakeServerError):
        _run(llm_client._call_groq_async([{"role": "user", "content": "hi"}]))
    assert call_count["n"] == 1  # no retry on a non-429 error


def test_call_groq_async_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(llm_client.settings, "groq_api_key", "fake-key")
    monkeypatch.setattr(llm_client, "_BASE_BACKOFF_SECONDS", 0.0)

    call_count = {"n": 0}

    async def create_fn(**kwargs):
        call_count["n"] += 1
        raise _FakeRateLimitError("always rate limited")

    monkeypatch.setattr(llm_client, "AsyncGroq", _fake_async_groq(create_fn))

    with pytest.raises(_FakeRateLimitError):
        _run(llm_client._call_groq_async([{"role": "user", "content": "hi"}]))
    assert call_count["n"] == llm_client._MAX_RETRIES + 1
