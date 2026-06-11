"""In-memory ChromaDB vector store with per-session document collections."""

import logging
import uuid
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

_client: chromadb.Client = chromadb.Client(
    ChromaSettings(anonymized_telemetry=False)
)

# Maps session_id -> (collection, chunks)
_sessions: dict[str, tuple[chromadb.Collection, list[str]]] = {}


@dataclass(frozen=True)
class RetrievedChunk:
    """A single chunk returned by a similarity search.

    Attributes:
        text: The raw text of the chunk.
        score: Cosine similarity score in [0, 1]; higher is more relevant.
        index: Original position of this chunk within the document.
    """

    text: str
    score: float
    index: int


def create_session(chunks: list[str], embeddings: list[list[float]]) -> str:
    """Store document chunks and their embeddings in a new session collection.

    Args:
        chunks: List of text chunks from the parsed document.
        embeddings: Corresponding dense vectors, one per chunk.

    Returns:
        A unique session ID string that must be passed to subsequent calls.
    """
    session_id = uuid.uuid4().hex
    collection = _client.create_collection(
        name=session_id,
        metadata={"hnsw:space": "cosine"},
    )
    ids = [str(i) for i in range(len(chunks))]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks)
    _sessions[session_id] = (collection, chunks)
    logger.info("Created session '%s' with %d chunks", session_id, len(chunks))
    return session_id


def retrieve(
    session_id: str,
    query_embedding: list[float],
    top_k: int,
) -> list[RetrievedChunk]:
    """Find the top-k most relevant chunks for a query embedding.

    Args:
        session_id: Session identifier returned by create_session.
        query_embedding: Dense vector representing the user query.
        top_k: Number of chunks to retrieve.

    Returns:
        A list of RetrievedChunk objects sorted by descending similarity score.

    Raises:
        KeyError: If the session_id does not exist.
    """
    collection, chunks = _sessions[session_id]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, len(chunks)),
    )
    retrieved: list[RetrievedChunk] = []
    for doc, dist, doc_id in zip(
        results["documents"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        score = 1.0 - dist
        retrieved.append(RetrievedChunk(text=doc, score=score, index=int(doc_id)))
    return retrieved


def get_full_text(session_id: str) -> str:
    """Return the full document text by joining all stored chunks.

    Args:
        session_id: Session identifier returned by create_session.

    Returns:
        All chunks concatenated with double newlines.

    Raises:
        KeyError: If the session_id does not exist.
    """
    _, chunks = _sessions[session_id]
    return "\n\n".join(chunks)


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
