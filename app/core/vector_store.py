"""In-memory hybrid (dense + BM25) vector store with per-session collections."""

import logging
import re
import uuid
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Number of first-stage candidates pulled from each retriever before fusion.
_CANDIDATE_POOL = 20
# RRF damping constant; 60 is the value from the original RRF paper.
_RRF_K = 60
# Extra RRF-scale boost added to a chunk whose section header overlaps the query.
_SECTION_BOOST = 1.0 / (_RRF_K + 1)

_client: chromadb.Client = chromadb.Client(
    ChromaSettings(anonymized_telemetry=False)
)


@dataclass(frozen=True)
class _Session:
    """All state retained for one uploaded document.

    Attributes:
        collection: The ChromaDB collection holding chunk embeddings.
        chunks: The raw text chunks, indexed by position.
        sections: Section header per chunk, aligned 1:1 with chunks.
        original_text: The cleaned full document text, pre-chunking. Used for
            summarisation/risk analysis so overlapping chunks are not sent
            (and duplicated) to the LLM.
        bm25: A BM25 index over the tokenised chunks for lexical retrieval.
    """

    collection: chromadb.Collection
    chunks: list[str]
    sections: list[str]
    original_text: str
    bm25: BM25Okapi


# Maps session_id -> _Session
_sessions: dict[str, _Session] = {}


@dataclass(frozen=True)
class RetrievedChunk:
    """A single chunk returned by a similarity search.

    Attributes:
        text: The raw text of the chunk.
        score: Fused relevance score; higher is more relevant.
        index: Original position of this chunk within the document.
    """

    text: str
    score: float
    index: int


def _tokenize(text: str) -> list[str]:
    """Lowercase word-tokenise text for BM25 lexical matching.

    Args:
        text: Input string.

    Returns:
        A list of lowercased alphanumeric tokens.
    """
    return re.findall(r"\w+", text.lower())


def create_session(
    chunks: list[str],
    embeddings: list[list[float]],
    original_text: str,
    sections: list[str] | None = None,
) -> str:
    """Store document chunks, embeddings, and a BM25 index in a new session.

    Args:
        chunks: List of text chunks from the parsed document.
        embeddings: Corresponding dense vectors, one per chunk.
        original_text: Cleaned full document text, pre-chunking.
        sections: Optional section header per chunk, aligned 1:1 with chunks.

    Returns:
        A unique session ID string that must be passed to subsequent calls.
    """
    sections = sections or [""] * len(chunks)
    session_id = uuid.uuid4().hex
    collection = _client.create_collection(
        name=session_id,
        metadata={"hnsw:space": "cosine"},
    )
    ids = [str(i) for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"section": s} for s in sections],
    )
    bm25 = BM25Okapi([_tokenize(c) for c in chunks])
    _sessions[session_id] = _Session(
        collection=collection,
        chunks=chunks,
        sections=sections,
        original_text=original_text,
        bm25=bm25,
    )
    logger.info("Created session '%s' with %d chunks", session_id, len(chunks))
    return session_id


def _dense_ranking(
    session: _Session, query_embedding: list[float], pool: int
) -> list[int]:
    """Return chunk indices ranked by dense (cosine) similarity, best first.

    Args:
        session: The active session.
        query_embedding: Dense query vector.
        pool: Maximum number of candidates to return.

    Returns:
        A list of chunk indices in descending similarity order.
    """
    results = session.collection.query(
        query_embeddings=[query_embedding],
        n_results=min(pool, len(session.chunks)),
    )
    return [int(doc_id) for doc_id in results["ids"][0]]


def _bm25_ranking(session: _Session, query_text: str, pool: int) -> list[int]:
    """Return chunk indices ranked by BM25 lexical score, best first.

    Args:
        session: The active session.
        query_text: Raw query string.
        pool: Maximum number of candidates to return.

    Returns:
        A list of chunk indices in descending BM25 order.
    """
    scores = session.bm25.get_scores(_tokenize(query_text))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:pool]


def _reciprocal_rank_fusion(rankings: list[list[int]], k: int = _RRF_K) -> dict[int, float]:
    """Fuse several ranked lists into one score per item via RRF.

    Reciprocal Rank Fusion scores each item as ``Σ 1 / (k + rank)`` across all
    rankings it appears in (rank is 0-based). This favours items that rank
    highly in multiple retrievers without needing comparable raw scores.

    Args:
        rankings: A list of ranked index lists (best first).
        k: RRF damping constant.

    Returns:
        A dict mapping chunk index to its fused score.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


def retrieve(
    session_id: str,
    query_embedding: list[float],
    query_text: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """Find the top-k most relevant chunks using hybrid dense + BM25 retrieval.

    Pulls a candidate pool from both a dense (cosine) retriever and a BM25
    lexical retriever, fuses them with Reciprocal Rank Fusion, applies a small
    boost to chunks whose section header shares keywords with the query, and
    returns the highest-scoring chunks.

    Args:
        session_id: Session identifier returned by create_session.
        query_embedding: Dense vector representing the user query.
        query_text: Raw query string, used for BM25 and section boosting.
        top_k: Number of chunks to return.

    Returns:
        A list of RetrievedChunk objects sorted by descending fused score.

    Raises:
        KeyError: If the session_id does not exist.
    """
    session = _sessions[session_id]

    dense = _dense_ranking(session, query_embedding, _CANDIDATE_POOL)
    lexical = _bm25_ranking(session, query_text, _CANDIDATE_POOL)
    fused = _reciprocal_rank_fusion([dense, lexical])

    query_tokens = set(_tokenize(query_text))
    for idx in fused:
        section_tokens = set(_tokenize(session.sections[idx]))
        if query_tokens & section_tokens:
            fused[idx] += _SECTION_BOOST

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        RetrievedChunk(text=session.chunks[idx], score=score, index=idx)
        for idx, score in ranked
    ]


def get_full_text(session_id: str) -> str:
    """Return the cleaned full document text (pre-chunking).

    Returns the original text rather than re-joining overlapping chunks, which
    would duplicate the overlap regions and waste the LLM context budget.

    Args:
        session_id: Session identifier returned by create_session.

    Returns:
        The cleaned full document text.

    Raises:
        KeyError: If the session_id does not exist.
    """
    return _sessions[session_id].original_text


def get_chunks(session_id: str) -> list[str]:
    """Return the ordered, non-overlapping-purpose text chunks for a session.

    Used by whole-document tasks (summarisation, risk analysis) that need to
    process a document too long for a single LLM call in map-reduce batches,
    as an alternative to truncating and silently dropping the tail of the
    document.

    Args:
        session_id: Session identifier returned by create_session.

    Returns:
        The list of text chunks in original document order.

    Raises:
        KeyError: If the session_id does not exist.
    """
    return _sessions[session_id].chunks


def session_exists(session_id: str) -> bool:
    """Check whether a session is currently active.

    Args:
        session_id: Session identifier to check.

    Returns:
        True if the session exists, False otherwise.
    """
    return session_id in _sessions


def delete_session(session_id: str) -> None:
    """Remove a session and its associated ChromaDB collection.

    Safe to call on a non-existent session — silently does nothing.

    Args:
        session_id: Session identifier to remove.
    """
    if session_id not in _sessions:
        return
    try:
        _client.delete_collection(session_id)
    except Exception:
        logger.warning("Could not delete ChromaDB collection for session '%s'", session_id)
    del _sessions[session_id]
    logger.info("Deleted session '%s'", session_id)
