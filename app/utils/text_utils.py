"""Pure text-processing utilities with no I/O or external service calls."""

import re
import unicodedata

# A line is treated as a section header when it is short and looks like a
# heading: ALL-CAPS words, a numbered clause (e.g. "1." / "2.3 "), or a short
# label ending in a colon. Used to give the retriever document-structure signal.
_MAX_HEADER_LEN = 60
_NUMBERED_HEADER = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+\S")


def _is_section_header(line: str) -> bool:
    """Return True if a line looks like a legal-document section header.

    Args:
        line: A single already-stripped line of text.

    Returns:
        True if the line is short and matches a heading pattern (ALL-CAPS,
        numbered clause, or a short colon-terminated label).
    """
    if not line or len(line) > _MAX_HEADER_LEN:
        return False
    letters = [c for c in line if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return True
    if _NUMBERED_HEADER.match(line):
        return True
    if line.endswith(":") and len(line.split()) <= 8:
        return True
    return False


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


def group_chunks_for_budget(chunks: list[str], max_chars: int) -> list[str]:
    """Greedily batch pre-split chunks into groups that fit a character budget.

    Used to feed a long document to an LLM in map-reduce passes: each
    returned batch is small enough for one call, and consecutive chunks are
    kept together (in original order) so a batch reads as a coherent excerpt
    rather than disconnected fragments.

    Args:
        chunks: Ordered text chunks (e.g. from ``split_into_chunks``).
        max_chars: Maximum length of each returned batch, in characters. A
            single chunk longer than this on its own still becomes its own
            (oversized) batch rather than being split further.

    Returns:
        A list of batch strings, each the concatenation of one or more
        consecutive chunks joined by blank lines.
    """
    batches: list[str] = []
    current = ""
    for chunk in chunks:
        if current and len(current) + 2 + len(chunk) > max_chars:
            batches.append(current)
            current = chunk
        else:
            current = f"{current}\n\n{chunk}" if current else chunk
    if current:
        batches.append(current)
    return batches


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

    Thin wrapper around :func:`split_into_chunks_with_sections` that discards
    the per-chunk section labels. See that function for the chunking algorithm.

    Args:
        text: Cleaned input text.
        chunk_size: Target maximum length of each chunk in characters.
        overlap: Number of characters carried over from the end of one chunk
            to the start of the next.

    Returns:
        A list of non-empty text chunks.
    """
    chunks, _sections = split_into_chunks_with_sections(text, chunk_size, overlap)
    return chunks


def split_into_chunks_with_sections(
    text: str, chunk_size: int, overlap: int
) -> tuple[list[str], list[str]]:
    """Split text into overlapping chunks and tag each with its section header.

    Paragraphs (separated by blank lines) are accumulated greedily until the
    running length would exceed *chunk_size*. When a single paragraph is itself
    longer than *chunk_size* it is split further on sentence boundaries
    (``". "`` delimiter). The last *overlap* characters of each chunk are
    prepended to the next chunk to preserve context across boundaries.

    While walking paragraphs, lines that look like legal section headers (see
    :func:`_is_section_header`) update the "current section"; each emitted chunk
    is labelled with the section that was active when it began. This gives the
    retriever a document-structure signal it can use to boost on-topic chunks.

    Args:
        text: Cleaned input text.
        chunk_size: Target maximum length of each chunk in characters.
        overlap: Number of characters carried over from the end of one chunk
            to the start of the next.

    Returns:
        A tuple ``(chunks, sections)`` of equal length, where ``sections[i]`` is
        the section header for ``chunks[i]`` ("" if none was seen yet).
    """
    paragraphs: list[str] = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Expand paragraphs into pieces, carrying the active section header.
    # A piece is (text, section).
    expanded: list[tuple[str, str]] = []
    current_section = ""
    for para in paragraphs:
        if _is_section_header(para):
            current_section = para.rstrip(":").strip()
        if len(para) <= chunk_size:
            expanded.append((para, current_section))
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            for sent in sentences:
                if buf and len(buf) + 1 + len(sent) > chunk_size:
                    expanded.append((buf.strip(), current_section))
                    buf = sent
                else:
                    buf = (buf + " " + sent).strip() if buf else sent
            if buf:
                expanded.append((buf.strip(), current_section))

    chunks: list[str] = []
    sections: list[str] = []
    current = ""
    current_chunk_section = ""

    for piece, section in expanded:
        if current and len(current) + 2 + len(piece) > chunk_size:
            chunks.append(current.strip())
            sections.append(current_chunk_section)
            carry = current[-overlap:] if overlap else ""
            current = (carry + "\n\n" + piece).strip() if carry else piece
            current_chunk_section = section
        else:
            if not current:
                current_chunk_section = section
            current = (current + "\n\n" + piece).strip() if current else piece

    if current.strip():
        chunks.append(current.strip())
        sections.append(current_chunk_section)

    paired = [(c, s) for c, s in zip(chunks, sections) if c]
    return [c for c, _ in paired], [s for _, s in paired]
