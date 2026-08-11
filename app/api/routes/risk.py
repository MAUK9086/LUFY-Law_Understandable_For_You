"""Risk analysis endpoint returning categorised contract flag cards."""

import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import RiskRequest, RiskResponse
from app.config import settings
from app.core.llm_client import analyse_risks, analyse_risks_long
from app.core.risk_analyzer import RiskFlag, parse_risk_response
from app.core.translator import translate
from app.core.vector_store import get_chunks, get_full_text, session_exists

logger = logging.getLogger(__name__)

router = APIRouter()

# Documents at or under this size are risk-analysed in a single LLM call.
# See summarize.py for the reasoning behind sizing this against the model's
# actual 128K-token context window rather than an arbitrary small budget.
# Longer documents go through analyse_risks_long's map-reduce path instead
# of being truncated, so clauses past this size are not silently missed;
# this should be a rare fallback, not the common case.
_SINGLE_SHOT_CHAR_LIMIT = 50_000


def _translate_flag(flag: RiskFlag, language: str) -> RiskFlag:
    """Translate the explanation and advice fields of a RiskFlag.

    Args:
        flag: The original RiskFlag with English text.
        language: Target language display name or BCP-47 code.

    Returns:
        A new RiskFlag with translated explanation and advice.
    """
    return RiskFlag(
        clause=flag.clause,
        explanation=translate(flag.explanation, language),
        advice=translate(flag.advice, language),
    )


@router.post("/risk-analysis", response_model=RiskResponse)
async def risk_analysis(request: RiskRequest) -> RiskResponse:
    """Perform a three-tier risk analysis on the uploaded document.

    Documents that fit in a single LLM call are analysed directly. Longer
    documents are analysed via a map-reduce pass over the full document (see
    analyse_risks_long) so clauses are not silently dropped by truncation —
    every clause in the document is seen by some batch.

    Args:
        request: RiskRequest with session_id, persona, and language.

    Returns:
        RiskResponse with red, yellow, and green flag lists.

    Raises:
        HTTPException: 404 if the session does not exist.
        HTTPException: 500 if LLM or parsing fails.
    """
    if not session_exists(request.session_id):
        raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")

    text = get_full_text(request.session_id)

    try:
        if len(text) <= _SINGLE_SHOT_CHAR_LIMIT:
            raw = analyse_risks(text, request.persona)
        else:
            chunks = get_chunks(request.session_id)
            raw = await analyse_risks_long(chunks, request.persona, settings.map_reduce_batch_chars)
        report = parse_risk_response(raw)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Risk analysis failed for session '%s'", request.session_id)
        raise HTTPException(status_code=500, detail="Risk analysis failed.") from exc

    need_translation = request.language != "English"

    translate_flag = (
        (lambda f: _translate_flag(f, request.language))
        if need_translation
        else (lambda f: f)
    )

    section_labels = {
        "red": translate("Red Flags", request.language) if need_translation else "Red Flags",
        "yellow": translate("Yellow Flags", request.language) if need_translation else "Yellow Flags",
        "green": translate("Green Flags", request.language) if need_translation else "Green Flags",
    }

    return RiskResponse(
        red_flags=[translate_flag(f) for f in report.red_flags],
        yellow_flags=[translate_flag(f) for f in report.yellow_flags],
        green_flags=[translate_flag(f) for f in report.green_flags],
        section_labels=section_labels,
    )
