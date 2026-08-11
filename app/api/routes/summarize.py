"""Document summarisation endpoint."""

import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import SummariseRequest, SummariseResponse
from app.config import settings
from app.core.llm_client import summarise_document, summarise_document_long
from app.core.translator import translate
from app.core.vector_store import get_chunks, get_full_text, session_exists

logger = logging.getLogger(__name__)

router = APIRouter()

# Documents at or under this size are summarised in a single LLM call.
# llama-3.1-8b-instant has a 128K-token (~131K) context window; 60,000
# characters (~15K tokens) leaves generous headroom for the system prompt
# and output while comfortably covering real legal documents — this repo's
# own README estimates a 20-page contract at 30,000-50,000 characters.
# Longer documents go through summarise_document_long's map-reduce path
# instead of being truncated, so content past this size is not silently
# dropped from the summary; this should be a rare fallback, not the common
# case.
_SINGLE_SHOT_CHAR_LIMIT = 60_000


@router.post("/summarize", response_model=SummariseResponse)
async def summarize(request: SummariseRequest) -> SummariseResponse:
    """Summarise the document associated with a session.

    Documents that fit in a single LLM call are summarised directly. Longer
    documents are summarised via a map-reduce pass over the full document
    (see summarise_document_long) so content is not silently truncated.

    Args:
        request: SummariseRequest with session_id, persona, and language.

    Returns:
        SummariseResponse containing the summary text and language used.

    Raises:
        HTTPException: 404 if the session does not exist.
        HTTPException: 500 if the LLM call fails.
    """
    if not session_exists(request.session_id):
        raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")

    text = get_full_text(request.session_id)

    try:
        if len(text) <= _SINGLE_SHOT_CHAR_LIMIT:
            summary = summarise_document(text, request.persona, "English")
        else:
            chunks = get_chunks(request.session_id)
            summary = await summarise_document_long(
                chunks, request.persona, "English", settings.map_reduce_batch_chars
            )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Summarisation failed for session '%s'", request.session_id)
        raise HTTPException(status_code=500, detail="Summarisation failed.") from exc

    if request.language != "English":
        summary = translate(summary, request.language)

    return SummariseResponse(summary=summary, language=request.language)
