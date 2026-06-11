"""LLM integration via the Groq API for summarisation, risk analysis, and RAG queries."""

import json
import logging

from groq import Groq

from app.config import settings
from app.core.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)

_PERSONA_DESCRIPTIONS: dict[str, str] = {
    "tenant": "a tenant reviewing a rental or lease agreement",
    "employee": "an employee reviewing an employment contract or workplace policy",
    "freelancer": "a freelancer or independent contractor reviewing a service agreement",
    "general": "a member of the general public reviewing a legal document",
}


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


def summarise_document(text: str, persona: str, language: str) -> str:
    """Produce a structured plain-language summary of a legal document.

    The response is formatted into five labelled sections so the frontend
    can render it with visual hierarchy.

    Args:
        text: Full document text (pre-truncated to fit the context budget).
        persona: One of tenant/employee/freelancer/general.
        language: Target language display name (e.g. "English", "Hindi").

    Returns:
        A structured summary string with bold section headers.
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
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Summarise this document:\n\n{text}"},
    ]
    return _call_groq(messages)


def analyse_risks(text: str, persona: str) -> dict:
    """Identify risk flags in a legal document.

    Instructs the LLM to return structured JSON with exactly three keys:
    red_flags, yellow_flags, and green_flags.

    Args:
        text: Full document text (pre-truncated to fit the context budget).
        persona: One of tenant/employee/freelancer/general.

    Returns:
        A dict with keys red_flags, yellow_flags, green_flags. Each value is
        a list of objects with clause, explanation, and advice fields.

    Raises:
        ValueError: If the LLM response cannot be parsed as valid JSON after
            two attempts.
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
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyse this document:\n\n{text}"},
    ]
    raw = _call_groq(messages)

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
