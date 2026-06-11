"""Sentence embedding utilities backed by a cached SentenceTransformer model."""

import logging
import os
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load and cache the sentence embedding model.

    The model is loaded once and reused for the lifetime of the process.
    Sets TOKENIZERS_PARALLELISM=false to suppress a fork-safety warning that
    appears when the tokenizer is used inside a forked process (e.g. uvicorn
    workers). Uses local_files_only when the model is already cached to skip
    the HuggingFace Hub network round-trips (15+ HEAD requests) that add
    5-10 seconds of latency on every cold load.

    Returns:
        A ready-to-use SentenceTransformer instance.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logger.info("Loading embedding model '%s'", settings.embed_model)
    try:
        model = SentenceTransformer(settings.embed_model, local_files_only=True)
        logger.info("Loaded model from local cache (offline mode)")
    except Exception:
        logger.info("Model not in local cache — downloading from HuggingFace Hub")
        model = SentenceTransformer(settings.embed_model)
    return model


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
