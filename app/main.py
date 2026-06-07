"""FastAPI application factory for the LUFY legal document assistant."""

import logging
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


def create_app() -> FastAPI:
    """Build and configure the LUFY FastAPI application.

    Sets up logging, middleware, API routes, and static file serving. The
    frontend/ directory is mounted at the root path so that index.html and
    app.html are served directly.

    Returns:
        A fully configured FastAPI application instance.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        level=logging.DEBUG if settings.debug else logging.INFO,
    )
    logger = logging.getLogger(__name__)

    application = FastAPI(
        title=settings.app_title,
        description="RAG-powered legal document assistant with multi-language support.",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
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
