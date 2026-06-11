"""Pure text-processing utilities with no I/O or external service calls."""

import re
import unicodedata


def clean_text(raw: str) -> str:
    """Normalise and clean raw extracted text.

    Applies Unicode NFC normalisation, collapses runs of whitespace within
    lines, and reduces three or more consecutive newlines to two.

    Args:
        raw: Raw text string as extracted from a document parser.

    Returns:
        Cleaned text suitable for chunking and downstream processing.
    """
    text = unicodedata.normalize("NFC", raw)
    lines = []
    for line in text.splitlines():
        lines.append(re.sub(r"[ \t]+", " ", line).strip())
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_to_token_budget(text: str, max_chars: int) -> str:
    """Hard-truncate text to a character budget.

    Args:
        text: Input text to truncate.
        max_chars: Maximum number of characters to retain.

    Returns:
        The original text if it is within budget, or a truncated version
        with a trailing marker indicating content was omitted.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n… [truncated]"


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks while respecting paragraph boundaries.

    Paragraphs (separated by blank lines) are accumulated greedily until the
    running length would exceed *chunk_size*. When a single paragraph is itself
    longer than *chunk_size* it is split further on sentence boundaries
    (``". "`` delimiter). The last *overlap* characters of each chunk are
    prepended to the next chunk to preserve context across boundaries.

    Args:
        text: Cleaned input text.
        chunk_size: Target maximum length of each chunk in characters.
        overlap: Number of characters carried over from the end of one chunk
            to the start of the next.

    Returns:
        A list of non-empty text chunks.
    """
    paragraphs: list[str] = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Expand paragraphs that exceed chunk_size into sentence-level pieces.
    expanded: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            expanded.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            for sent in sentences:
                if buf and len(buf) + 1 + len(sent) > chunk_size:
                    expanded.append(buf.strip())
                    buf = sent
                else:
                    buf = (buf + " " + sent).strip() if buf else sent
            if buf:
                expanded.append(buf.strip())

    chunks: list[str] = []
    current = ""
    carry = ""

    for piece in expanded:
        candidate = (carry + "\n\n" + piece).strip() if carry else piece
        if current and len(current) + 2 + len(piece) > chunk_size:
            chunks.append(current.strip())
            carry = current[-overlap:] if overlap else ""
            current = (carry + "\n\n" + piece).strip() if carry else piece
        else:
            current = (current + "\n\n" + piece).strip() if current else piece

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]
