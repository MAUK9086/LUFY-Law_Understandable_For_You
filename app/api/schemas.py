"""Pydantic request and response models for all LUFY API endpoints."""

from pydantic import BaseModel, Field

from app.core.risk_analyzer import RiskFlag


class UploadResponse(BaseModel):
    """Response returned after a successful document upload or demo load.

    Attributes:
        session_id: Opaque session identifier for subsequent API calls.
        filename: Original name of the uploaded file.
        page_count: Number of pages extracted (always 1 for DOCX/TXT).
        char_count: Total characters in the cleaned document text.
        chunk_count: Number of embedding chunks created.
    """

    session_id: str
    filename: str
    page_count: int
    char_count: int
    chunk_count: int


class SummariseRequest(BaseModel):
    """Request body for the /api/summarize endpoint.

    Attributes:
        session_id: Active session identifier from a prior upload.
        persona: Viewer persona influencing the summary framing.
        language: Target language for the summary output.
    """

    session_id: str
    persona: str = "general"
    language: str = "English"


class SummariseResponse(BaseModel):
    """Response from the /api/summarize endpoint.

    Attributes:
        summary: Plain-language summary of the document.
        language: Language in which the summary was written.
    """

    summary: str
    language: str


class RiskRequest(BaseModel):
    """Request body for the /api/risk-analysis endpoint.

    Attributes:
        session_id: Active session identifier from a prior upload.
        persona: Viewer persona influencing risk framing.
        language: Target language for flag explanations and advice.
    """

    session_id: str
    persona: str = "general"
    language: str = "English"


class RiskResponse(BaseModel):
    """Response from the /api/risk-analysis endpoint.

    Attributes:
        red_flags: Unfair or risky clauses.
        yellow_flags: Vague or missing clauses.
        green_flags: Protective or fair clauses.
        section_labels: Translated display names for the three categories,
            keyed as "red", "yellow", "green".
    """

    red_flags: list[RiskFlag]
    yellow_flags: list[RiskFlag]
    green_flags: list[RiskFlag]
    section_labels: dict[str, str] = {
        "red": "Red Flags",
        "yellow": "Yellow Flags",
        "green": "Green Flags",
    }


class QueryRequest(BaseModel):
    """Request body for the /api/query endpoint.

    Attributes:
        session_id: Active session identifier from a prior upload.
        query: User's natural-language question about the document.
        persona: Viewer persona influencing the answer framing.
        language: Target language for the answer.
    """

    session_id: str
    query: str = Field(min_length=1, max_length=1000)
    persona: str = "general"
    language: str = "English"


class QueryResponse(BaseModel):
    """Response from the /api/query endpoint.

    Attributes:
        answer: LLM-generated answer grounded in document excerpts.
        sources: List of raw text excerpts that were used as context.
    """

    answer: str
    sources: list[str]


class ErrorResponse(BaseModel):
    """Standard error envelope returned for 4xx and 5xx responses.

    Attributes:
        detail: Human-readable description of the error.
    """

    detail: str
