"""Sentence embedding utilities backed by a cached SentenceTransformer model."""

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load and cache the sentence embedding model.

    The model is loaded once at first call and reused for the lifetime of the
    process. Loading happens on the calling thread; in production the first
    request triggers the load.

    Returns:
        A ready-to-use SentenceTransformer instance.
    """
    logger.info("Loading embedding model '%s'", settings.embed_model)
    return SentenceTransformer(settings.embed_model)


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed a list of text chunks as dense vectors.

    Args:
        chunks: List of text strings to embed.

    Returns:
        A list of float lists, one per input chunk, suitable for storage in
        ChromaDB.
    """
    model = get_embedder()
    vectors = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string as a dense vector.

    Args:
        query: The user's natural-language query.

    Returns:
        A single float list representing the query embedding.
    """
    model = get_embedder()
    vector = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
    return vector[0].tolist()
