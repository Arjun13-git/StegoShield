"""Application factory.

`create_app(settings)` builds an app whose model is loaded ONCE in the
lifespan start-up (never per request, never at import). If the frozen model is
missing or fails verification the app still starts: `/health` stays green,
`/ready` returns 503 and inference endpoints return 503 `model_unavailable`.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import AppServices
from app.api.routes import router, system_router
from app.core.config import Settings
from app.core.errors import register_exception_handlers
from app.core.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from app.services.analysis_service import AnalysisService
from app.services.model_service import ModelService
from app.services.reference import load_reference

logger = logging.getLogger("stegoshield")


def _configure_logging() -> None:
    log = logging.getLogger("stegoshield")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    _configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model = ModelService(settings)
        model.load()
        reference = None
        if model.ready:
            meta = model.metadata
            from app.ml.features import FEATURE_NAMES

            reference = load_reference(
                settings.reference_path, model_sha256=meta.artifact_sha256,
                feature_names=list(FEATURE_NAMES), feature_schema_version=meta.feature_schema_version,
            )
        app.state.services = AppServices(
            settings=settings, model=model, reference=reference,
            analysis=AnalysisService(settings, model, reference),
            concurrency=asyncio.Semaphore(settings.max_concurrent_analyses),
        )
        logger.info("startup complete: model_ready=%s reference=%s", model.ready, reference is not None)
        yield

    app = FastAPI(
        title="StegoShield API",
        version=settings.api_version,
        description="Explainable ML-based image steganalysis API (LSB-focused research prototype).",
        lifespan=lifespan,
        debug=False,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )

    # add_middleware: the LAST one added is the OUTERMOST.
    app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=settings.max_request_body_bytes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Accept"],
        expose_headers=["X-Request-ID", "Content-Disposition", "X-Embedded-Bytes", "X-Capacity-Bytes"],
    )
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)
    app.include_router(system_router)
    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
