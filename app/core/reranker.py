"""Cross-encoder reranking to refine first-stage hybrid retrieval results."""

import logging
import os
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import settings
from app.core.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load and cache the cross-encoder reranking model.

    Mirrors :func:`app.core.embedder.get_embedder`: the model is loaded once and
    reused for the process lifetime, and ``local_files_only`` is tried first to
    skip HuggingFace Hub network round-trips when the weights are already cached.

    Returns:
        A ready-to-use CrossEncoder instance.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logger.info("Loading reranker model '%s'", settings.rerank_model)
    try:
        model = CrossEncoder(settings.rerank_model, local_files_only=True)
        logger.info("Loaded reranker from local cache (offline mode)")
    except Exception:
        logger.info("Reranker not in local cache — downloading from HuggingFace Hub")
        model = CrossEncoder(settings.rerank_model)
    return model


def rerank(
    query: str, chunks: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    """Reorder retrieved chunks by cross-encoder relevance to the query.

    A cross-encoder jointly encodes (query, chunk) pairs, giving sharper
    relevance estimates than the first-stage bi-encoder/BM25 retrieval. The
    returned chunks carry the cross-encoder score in their ``score`` field.

    Args:
        query: The user's natural-language question.
        chunks: First-stage retrieved chunks to rerank.
        top_k: Number of chunks to keep after reranking.

    Returns:
        The top_k chunks sorted by descending cross-encoder score. Returns the
        input unchanged when it is empty.
    """
    if not chunks:
        return chunks
    model = get_reranker()
    scores = model.predict([(query, chunk.text) for chunk in chunks])
    rescored = [
        RetrievedChunk(text=chunk.text, score=float(score), index=chunk.index)
        for chunk, score in zip(chunks, scores)
    ]
    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored[:top_k]
