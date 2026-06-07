"""RAG-powered document query endpoint."""

import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import QueryRequest, QueryResponse
from app.config import settings
from app.core.embedder import embed_query
from app.core.llm_client import answer_query
from app.core.translator import translate
from app.core.vector_store import retrieve, session_exists

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest) -> QueryResponse:
    """Answer a natural-language question using RAG over the uploaded document.

    Embeds the query, retrieves the most relevant chunks from the vector store,
    constructs a grounded prompt for the LLM, and returns the answer with
    source citations.

    Args:
        request: QueryRequest with session_id, query, persona, and language.

    Returns:
        QueryResponse with the LLM answer and list of source excerpt texts.

    Raises:
        HTTPException: 404 if the session does not exist.
        HTTPException: 500 if embedding or LLM call fails.
    """
    if not session_exists(request.session_id):
        raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")

    try:
        query_vector = embed_query(request.query)
        chunks = retrieve(request.session_id, query_vector, settings.retrieval_top_k)
    except Exception as exc:
        logger.exception("Retrieval failed for session '%s'", request.session_id)
        raise HTTPException(status_code=500, detail="Document retrieval failed.") from exc

    try:
        result = answer_query(request.query, chunks, request.persona, "English")
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Query answering failed for session '%s'", request.session_id)
        raise HTTPException(status_code=500, detail="Query answering failed.") from exc

    answer = result["answer"]
    if request.language != "English":
        answer = translate(answer, request.language)

    sources = [chunk.text for chunk in result["sources"]]
    return QueryResponse(answer=answer, sources=sources)
