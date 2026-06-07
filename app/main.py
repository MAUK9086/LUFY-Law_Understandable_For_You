"""FastAPI application factory for the LUFY legal document assistant."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.document import router as document_router
from app.api.routes.health import router as health_router
from app.api.routes.query import router as query_router
from app.api.routes.risk import router as risk_router
from app.api.routes.summarize import router as summarize_router
from app.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Pre-warm the embedding model so the first user request is fast.

    The SentenceTransformer model is loaded in a thread pool to avoid blocking
    the event loop during startup.

    Args:
        application: The FastAPI application instance (unused but required by
            the lifespan protocol).

    Yields:
        Control back to FastAPI after startup is complete.
    """
    loop = asyncio.get_event_loop()
    logger.info("Pre-warming embedding model at startup…")
    try:
        from app.core.embedder import get_embedder
        await loop.run_in_executor(None, get_embedder)
        logger.info("Embedding model ready.")
    except Exception as exc:
        logger.warning("Embedder pre-warm failed (will retry on first request): %s", exc)
    yield


def create_app() -> FastAPI:
    """Build and configure the LUFY FastAPI application.

    Sets up logging, middleware, API routes, and static file serving. The
    frontend/ directory is mounted at the root path so that index.html and
    app.html are served directly. The embedding model is pre-warmed during
    the lifespan startup event.

    Returns:
        A fully configured FastAPI application instance.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        level=logging.DEBUG if settings.debug else logging.INFO,
    )

    application = FastAPI(
        title=settings.app_title,
        description="RAG-powered legal document assistant with multi-language support.",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(health_router)
    application.include_router(document_router, prefix="/api")
    application.include_router(summarize_router, prefix="/api")
    application.include_router(risk_router, prefix="/api")
    application.include_router(query_router, prefix="/api")

    frontend_dir = Path(__file__).parent.parent / "frontend"
    application.mount(
        "/",
        StaticFiles(directory=str(frontend_dir), html=True),
        name="frontend",
    )

    logger.info("LUFY v2 starting — title: %s | debug: %s", settings.app_title, settings.debug)
    return application


app: FastAPI = create_app()
