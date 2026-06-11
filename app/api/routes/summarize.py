"""Document summarisation endpoint."""

import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import SummariseRequest, SummariseResponse
from app.core.llm_client import summarise_document
from app.core.translator import translate
from app.core.vector_store import get_full_text, session_exists
from app.utils.text_utils import truncate_to_token_budget

logger = logging.getLogger(__name__)

router = APIRouter()

_MAX_SUMMARY_CHARS = 12_000


@router.post("/summarize", response_model=SummariseResponse)
async def summarize(request: SummariseRequest) -> SummariseResponse:
    """Summarise the document associated with a session.

    Retrieves the full document text, truncates it to a safe context budget,
    calls the LLM for a plain-language summary, and optionally translates it.

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
    text = truncate_to_token_budget(text, _MAX_SUMMARY_CHARS)

    try:
        summary = summarise_document(text, request.persona, "English")
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Summarisation failed for session '%s'", request.session_id)
        raise HTTPException(status_code=500, detail="Summarisation failed.") from exc

    if request.language != "English":
        summary = translate(summary, request.language)

    return SummariseResponse(summary=summary, language=request.language)
