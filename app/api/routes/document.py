"""Document upload and demo-load endpoints."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.api.schemas import UploadResponse
from app.core.document_processor import parse_document
from app.core.embedder import embed_chunks
from app.core.vector_store import create_session

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB

_SAMPLE_DOC_PATH = Path(__file__).parent.parent.parent.parent / "sample_docs" / "legal_judgement.txt"


def _process_bytes(file_bytes: bytes, filename: str) -> UploadResponse:
    """Parse, embed, and store a document, returning upload metadata.

    Args:
        file_bytes: Raw bytes of the document.
        filename: Original filename (used for type detection and metadata).

    Returns:
        An UploadResponse with session_id and document statistics.

    Raises:
        HTTPException: 400 if the document is invalid or empty.
        HTTPException: 500 for unexpected processing errors.
    """
    try:
        parsed = parse_document(file_bytes, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error parsing '%s'", filename)
        raise HTTPException(status_code=500, detail="Document parsing failed.") from exc

    try:
        embeddings = embed_chunks(parsed.chunks)
        session_id = create_session(parsed.chunks, embeddings)
    except Exception as exc:
        logger.exception("Embedding/indexing failed for '%s'", filename)
        raise HTTPException(status_code=500, detail="Document indexing failed.") from exc

    return UploadResponse(
        session_id=session_id,
        filename=parsed.filename,
        page_count=parsed.page_count,
        char_count=parsed.char_count,
        chunk_count=len(parsed.chunks),
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile) -> UploadResponse:
    """Accept a PDF, DOCX, or TXT upload and create an analysis session.

    Args:
        file: The multipart file upload.

    Returns:
        UploadResponse with session_id and document metadata.

    Raises:
        HTTPException: 400 for unsupported extension or oversized files.
        HTTPException: 500 for processing failures.
    """
    filename = file.filename or "upload"
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{suffix}'. Accepted: pdf, docx, txt.",
        )

    file_bytes = await file.read()
    if len(file_bytes) > _MAX_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds the 10 MB limit ({len(file_bytes):,} bytes received).",
        )

    logger.info("Upload received: '%s' (%d bytes)", filename, len(file_bytes))
    return _process_bytes(file_bytes, filename)


@router.post("/demo", response_model=UploadResponse)
async def load_demo() -> UploadResponse:
    """Load the bundled sample legal document to create an analysis session.

    Returns:
        UploadResponse with session_id and document metadata.

    Raises:
        HTTPException: 404 if the sample document file is missing.
        HTTPException: 500 for processing failures.
    """
    if not _SAMPLE_DOC_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Sample document not found. Check that sample_docs/ is present.",
        )
    file_bytes = _SAMPLE_DOC_PATH.read_bytes()
    logger.info("Loading demo document '%s'", _SAMPLE_DOC_PATH.name)
    return _process_bytes(file_bytes, _SAMPLE_DOC_PATH.name)
