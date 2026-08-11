"""LLM integration via the Groq API for summarisation, risk analysis, and RAG queries."""

import asyncio
import json
import logging
import random

from groq import AsyncGroq, Groq

from app.config import settings
from app.core.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)

_PERSONA_DESCRIPTIONS: dict[str, str] = {
    "tenant": "a tenant reviewing a rental or lease agreement",
    "employee": "an employee reviewing an employment contract or workplace policy",
    "freelancer": "a freelancer or independent contractor reviewing a service agreement",
    "general": "a member of the general public reviewing a legal document",
}

# Retry/backoff for the async client only: it's used exclusively by the
# map-reduce fallback path, which fires several concurrent calls per request
# and is therefore the path actually likely to hit Groq's free-tier rate
# limit (~30 requests/min, ~30K tokens/min at the time of writing). The
# single-shot sync path (_call_groq) fires one call per request and has
# never needed this.
_MAX_RETRIES = 2
_BASE_BACKOFF_SECONDS = 1.0


def _call_groq(messages: list[dict]) -> str:
    """Send a chat completion request to the Groq API.

    Args:
        messages: A list of message dicts in OpenAI chat format
            (role + content).

    Returns:
        The text content of the first completion choice.

    Raises:
        ValueError: If GROQ_API_KEY is not configured.
    """
    if not settings.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file or environment."
        )
    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


async def _call_groq_async(messages: list[dict]) -> str:
    """Async equivalent of :func:`_call_groq`, with rate-limit retry/backoff.

    Used only by the map-reduce fallback path (see ``summarise_document_long``
    / ``analyse_risks_long``), where several calls fire concurrently for one
    request and can plausibly hit a 429 even though a single call rarely
    would. Retries with exponential backoff + jitter on a 429 response;
    anything else is raised immediately.

    Args:
        messages: A list of message dicts in OpenAI chat format
            (role + content).

    Returns:
        The text content of the first completion choice.

    Raises:
        ValueError: If GROQ_API_KEY is not configured.
        Exception: Whatever the Groq client raises, after retries (for 429)
            or immediately (for anything else).
    """
    if not settings.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file or environment."
        )
    client = AsyncGroq(api_key=settings.groq_api_key)
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = await client.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as exc:
            is_rate_limited = getattr(exc, "status_code", None) == 429
            if not is_rate_limited or attempt == _MAX_RETRIES:
                raise
            backoff = _BASE_BACKOFF_SECONDS * (2**attempt) + random.uniform(0, 0.5)
            logger.warning(
                "Groq rate-limited (attempt %d/%d); backing off %.1fs",
                attempt + 1,
                _MAX_RETRIES + 1,
                backoff,
            )
            await asyncio.sleep(backoff)


async def _gather_bounded(coros, max_concurrency: int) -> list:
    """Run coroutines concurrently, capped at ``max_concurrency`` in flight.

    Plain ``asyncio.gather`` would fire every coroutine at once, which for
    the map-reduce fallback means as many concurrent Groq calls as there are
    batches — easily enough to blow through the free-tier rate limit on a
    long document. A semaphore caps how many run at once without changing
    the total amount of work.

    Args:
        coros: An iterable of coroutines to run.
        max_concurrency: Maximum number running at any one time.

    Returns:
        Results in the same order as the input coroutines.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_run(c) for c in coros))


def _build_summary_messages(text: str, persona: str, language: str) -> list[dict]:
    """Build the chat messages for the five-section summary prompt.

    Shared by the single-shot path and the reduce step of the map-reduce
    fallback, so both produce the exact same output format.

    Args:
        text: Document text to summarise.
        persona: One of tenant/employee/freelancer/general.
        language: Target language display name.

    Returns:
        A list of message dicts ready for the Groq chat-completions API.
    """
    persona_desc = _PERSONA_DESCRIPTIONS.get(persona, _PERSONA_DESCRIPTIONS["general"])
    system_prompt = (
        f"You are a friendly legal assistant helping {persona_desc}. "
        "Analyse the following legal document and provide a structured summary "
        "using EXACTLY these five sections, each on its own line:\n\n"
        "**Document Type:** (one line — what kind of document this is)\n"
        "**Key Parties:** (who is involved and their roles)\n"
        "**Main Points:**\n"
        "• (bullet point for each important clause or finding — 3 to 6 bullets)\n"
        "**Key Figures:** (important dates, deadlines, monetary amounts — write 'None' if absent)\n"
        "**What This Means For You:** (2–3 sentences of practical plain-English takeaway)\n\n"
        "Rules: avoid legal jargon; if you must use a legal term, explain it in parentheses. "
        "Do not add any sections beyond the five listed above. "
        f"Write every section in {language}."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Summarise this document:\n\n{text}"},
    ]


def summarise_document(text: str, persona: str, language: str) -> str:
    """Produce a structured plain-language summary of a legal document.

    The response is formatted into five labelled sections so the frontend
    can render it with visual hierarchy. Intended for documents that fit
    the single-shot context budget; see ``summarise_document_long`` for the
    map-reduce fallback used above that budget.

    Args:
        text: Full document text (or, for the reduce step of the map-reduce
            fallback, condensed notes standing in for it).
        persona: One of tenant/employee/freelancer/general.
        language: Target language display name (e.g. "English", "Hindi").

    Returns:
        A structured summary string with bold section headers.
    """
    return _call_groq(_build_summary_messages(text, persona, language))


def _build_chunk_notes_messages(text: str, persona: str) -> list[dict]:
    """Build the chat messages for the map-step "extract notes" prompt.

    Args:
        text: One excerpt (chunk batch) of the document.
        persona: One of tenant/employee/freelancer/general.

    Returns:
        A list of message dicts ready for the Groq chat-completions API.
    """
    persona_desc = _PERSONA_DESCRIPTIONS.get(persona, _PERSONA_DESCRIPTIONS["general"])
    system_prompt = (
        f"You are helping prepare notes for a summary aimed at {persona_desc}. "
        "You will be shown ONE EXCERPT of a larger legal document, not the whole thing. "
        "List only the factual, load-bearing points from THIS excerpt: parties named, "
        "obligations, dates, monetary amounts, and any notably one-sided or risky "
        "clause. Write terse plain-English bullet points, no headers, no preamble, no "
        "commentary about it being an excerpt. If this excerpt has nothing substantive, "
        "reply with exactly: 'No substantive content in this excerpt.'"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]


async def _summarise_chunk_notes_async(text: str, persona: str) -> str:
    """Extract terse factual notes from one excerpt of a longer document.

    This is the "map" step for long-document summarisation: it deliberately
    does not produce the final five-section format (that would repeat
    section headers once per excerpt and waste the reduce step's budget on
    formatting instead of content). It just pulls out the facts. Async so
    the map step can run concurrently across batches (see
    ``summarise_document_long``).

    Args:
        text: One excerpt (chunk batch) of the document.
        persona: One of tenant/employee/freelancer/general.

    Returns:
        A short, plain-text list of factual notes from this excerpt.
    """
    return await _call_groq_async(_build_chunk_notes_messages(text, persona))


async def summarise_document_long(
    chunks: list[str], persona: str, language: str, batch_chars: int
) -> str:
    """Summarise a document too long for the single-shot budget, via map-reduce.

    This is a fallback path only — see ``app/api/routes/summarize.py`` for
    the size threshold below which ``summarise_document`` is used directly
    in one call. It exists because even a generous single-shot budget can't
    cover every document; most real legal documents should never reach this
    function.

    Map step: each budget-sized batch of chunks is condensed into factual
    notes independently, running up to a bounded number of batches
    concurrently (see ``_gather_bounded``) rather than one at a time, since
    this path is specifically the one where per-request latency was a
    concern. Reduce step: the concatenated notes (now much shorter than the
    original document) are run through the normal five-section
    :func:`summarise_document` prompt, so the final output format is
    identical to the single-shot path regardless of document length.

    Args:
        chunks: Ordered text chunks for the full document.
        persona: One of tenant/employee/freelancer/general.
        language: Target language display name.
        batch_chars: Maximum characters per map-step LLM call.

    Returns:
        A structured five-section summary string, same shape as
        :func:`summarise_document`.
    """
    from app.utils.text_utils import group_chunks_for_budget, truncate_to_token_budget

    batches = group_chunks_for_budget(chunks, batch_chars)
    notes = await _gather_bounded(
        [_summarise_chunk_notes_async(batch, persona) for batch in batches],
        settings.map_reduce_max_concurrency,
    )
    combined_notes = "\n\n".join(notes)
    # Defensive safety net only: with reasonable batch sizes the condensed
    # notes should comfortably fit, but an extremely long document could
    # still exceed the reduce step's budget after condensing.
    combined_notes = truncate_to_token_budget(combined_notes, batch_chars * 2)
    return summarise_document(combined_notes, persona, language)


def _merge_risk_dicts(raw_dicts: list[dict]) -> dict:
    """Merge and de-duplicate risk-flag dicts produced by independent batches.

    Consecutive map-step batches overlap slightly (chunks carry a character
    overlap), so the same clause can surface in two adjacent batches' output.
    De-duplication is by normalised clause text within each category, keeping
    the first occurrence.

    Args:
        raw_dicts: One raw LLM risk-analysis dict per processed batch.

    Returns:
        A single merged dict with the same three-key shape as one batch's
        output, ready for :func:`app.core.risk_analyzer.parse_risk_response`.
    """
    merged: dict[str, list] = {"red_flags": [], "yellow_flags": [], "green_flags": []}
    seen: dict[str, set[str]] = {"red_flags": set(), "yellow_flags": set(), "green_flags": set()}

    for raw in raw_dicts:
        for key in merged:
            items = raw.get(key, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                clause = item.get("clause")
                if not isinstance(clause, str):
                    continue
                normalised = " ".join(clause.lower().split())
                if normalised in seen[key]:
                    continue
                seen[key].add(normalised)
                merged[key].append(item)

    return merged


async def analyse_risks_long(chunks: list[str], persona: str, batch_chars: int) -> dict:
    """Risk-analyse a document too long for the single-shot budget, via map-reduce.

    This is a fallback path only — see ``app/api/routes/risk.py`` for the
    size threshold below which ``analyse_risks`` is used directly in one
    call; most real legal documents should never reach this function.

    Each budget-sized batch of chunks is independently risk-analysed with
    the same JSON contract as :func:`analyse_risks`, running up to a bounded
    number of batches concurrently (see ``_gather_bounded``), and the
    resulting flag lists are merged and de-duplicated across batches. There
    is no reduce-step LLM call — flags are discrete items, not prose that
    needs re-condensing.

    Args:
        chunks: Ordered text chunks for the full document.
        persona: One of tenant/employee/freelancer/general.
        batch_chars: Maximum characters per batch LLM call.

    Returns:
        A merged dict with keys red_flags/yellow_flags/green_flags, same
        shape as :func:`analyse_risks`'s return value.
    """
    from app.utils.text_utils import group_chunks_for_budget

    batches = group_chunks_for_budget(chunks, batch_chars)
    raw_dicts = await _gather_bounded(
        [_analyse_risks_async(batch, persona) for batch in batches],
        settings.map_reduce_max_concurrency,
    )
    return _merge_risk_dicts(raw_dicts)


def _build_risk_messages(text: str, persona: str) -> list[dict]:
    """Build the chat messages for the structured risk-analysis prompt.

    Shared by the single-shot (:func:`analyse_risks`) and map-step
    (:func:`_analyse_risks_async`) paths, so both use an identical JSON
    contract.

    Args:
        text: Document text (or one batch of it) to analyse.
        persona: One of tenant/employee/freelancer/general.

    Returns:
        A list of message dicts ready for the Groq chat-completions API.
    """
    persona_desc = _PERSONA_DESCRIPTIONS.get(persona, _PERSONA_DESCRIPTIONS["general"])
    system_prompt = (
        f"You are a legal risk analyst reviewing a document for {persona_desc}. "
        "Analyse the document and return ONLY valid JSON — no other text — with exactly "
        'three keys: "red_flags", "yellow_flags", "green_flags". '
        "Each key maps to an array of objects. Every object must have exactly three "
        'string fields: "clause" (the clause or section name), "explanation" (what it '
        'means in plain language), and "advice" (what the person should do). '
        "red_flags = clauses that are unfair, risky, or heavily one-sided. "
        "yellow_flags = clauses that are missing, vague, or need clarification. "
        "green_flags = clauses that are fair and protect the reader's interests."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyse this document:\n\n{text}"},
    ]


def _parse_risk_json(raw: str) -> dict:
    """Parse an LLM risk-analysis response into a dict, with a fallback pass.

    Args:
        raw: Raw text content returned by the LLM.

    Returns:
        The parsed dict.

    Raises:
        ValueError: If the response cannot be parsed as JSON even after
            trying to extract the outermost ``{...}`` substring.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fallback: extract outermost JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"LLM returned non-JSON risk analysis response: {raw[:200]}")


def analyse_risks(text: str, persona: str) -> dict:
    """Identify risk flags in a legal document.

    Instructs the LLM to return structured JSON with exactly three keys:
    red_flags, yellow_flags, and green_flags. Intended for documents that
    fit the single-shot context budget; see ``analyse_risks_long`` for the
    map-reduce fallback used above that budget.

    Args:
        text: Full document text (or, for the map step of the map-reduce
            fallback, one batch of it).
        persona: One of tenant/employee/freelancer/general.

    Returns:
        A dict with keys red_flags, yellow_flags, green_flags. Each value is
        a list of objects with clause, explanation, and advice fields.

    Raises:
        ValueError: If the LLM response cannot be parsed as valid JSON after
            two attempts.
    """
    raw = _call_groq(_build_risk_messages(text, persona))
    return _parse_risk_json(raw)


async def _analyse_risks_async(text: str, persona: str) -> dict:
    """Async equivalent of :func:`analyse_risks`, used by the map step.

    Args:
        text: One batch of document text.
        persona: One of tenant/employee/freelancer/general.

    Returns:
        A dict with keys red_flags, yellow_flags, green_flags, same shape
        as :func:`analyse_risks`'s return value.

    Raises:
        ValueError: If the LLM response cannot be parsed as valid JSON after
            two attempts.
    """
    raw = await _call_groq_async(_build_risk_messages(text, persona))
    return _parse_risk_json(raw)


def answer_query(
    query: str,
    context_chunks: list[RetrievedChunk],
    persona: str,
    language: str,
) -> dict:
    """Answer a user question using retrieved document excerpts.

    Injects numbered source excerpts into the prompt so the LLM can cite them.

    Args:
        query: The user's natural-language question.
        context_chunks: Retrieved chunks from the vector store.
        persona: One of tenant/employee/freelancer/general.
        language: Target language display name for the answer.

    Returns:
        A dict with keys:
            answer (str): The LLM's response.
            sources (list[RetrievedChunk]): The context chunks that were provided.
    """
    persona_desc = _PERSONA_DESCRIPTIONS.get(persona, _PERSONA_DESCRIPTIONS["general"])
    excerpts = "\n\n".join(
        f"[Source {i + 1}]\n{chunk.text}"
        for i, chunk in enumerate(context_chunks)
    )
    system_prompt = (
        f"You are a precise legal assistant helping {persona_desc}. "
        "Answer ONLY from the numbered document excerpts provided. "
        "Be direct and specific — give a clear, concise answer in 1–4 sentences. "
        "When the answer is in the text, quote the exact wording briefly to support your answer. "
        "If the answer is genuinely not in the excerpts, reply with exactly: "
        "'This specific information is not covered in the retrieved sections of the document.' "
        "Never guess, invent details, or use knowledge outside the excerpts. "
        f"Reply in {language}."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Document excerpts:\n\n{excerpts}\n\n"
                f"Question: {query}"
            ),
        },
    ]
    answer = _call_groq(messages)
    return {"answer": answer, "sources": context_chunks}
