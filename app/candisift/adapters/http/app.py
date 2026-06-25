"""FastAPI application factory — the outermost driving adapter.

Wires the composition root, the durable worker (started in lifespan so one
`uvicorn` process serves API + UI + background screening), the security
middleware stack, auth-gated routers, health/readiness, and a safe error
handler that never leaks internals.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.candisift.application.screening_service import NotFoundError
from app.candisift.adapters.worker import Worker
from .container import Container, build_container
from .deps import get_container
from .routes import register_routes
from .security import (
    BodySizeLimitMiddleware, RateLimitMiddleware, RequestIDMiddleware,
    SecurityHeadersMiddleware,
)

log = logging.getLogger("candisift.app")


def create_app(container: Container | None = None) -> FastAPI:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    container = container or build_container()
    s = container.settings
    worker = Worker(container.queue, container.service,
                    lease_seconds=s.worker_lease_seconds, concurrency=s.worker_concurrency)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker.start()
        try:
            yield
        finally:
            worker.stop()

    app = FastAPI(title="CandiSift — resume screening", version="1.0.0", lifespan=lifespan)
    app.state.container = container

    # same-origin static assets (CSP script-src 'self' allows /static/app.js).
    # Mounted outside the auth-gated routers — it's public UI JS, no secrets.
    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

    # middleware: last added is outermost. Request-id outermost so every log line has it.
    # CORS: explicit allowlist only (validate_runtime rejects '*' in prod). Scope
    # methods/headers to what the UI/API actually use instead of '*' on this
    # authenticated surface. allow_credentials stays default-False (Basic auth rides
    # the Authorization header, not cookies), so a stray origin can't ride a session.
    if s.cors_list:
        app.add_middleware(CORSMiddleware, allow_origins=s.cors_list,
                           allow_methods=["GET", "POST"],
                           allow_headers=["Authorization", "Content-Type", "X-Request-ID"])
    app.add_middleware(RateLimitMiddleware, per_minute=s.rate_limit_per_min)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=s.max_request_mb * 1024 * 1024)
    app.add_middleware(SecurityHeadersMiddleware, hsts=s.hsts)
    app.add_middleware(RequestIDMiddleware)

    @app.exception_handler(NotFoundError)
    async def _not_found(request: Request, exc: NotFoundError):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        return JSONResponse({"detail": exc.errors()}, status_code=422)

    @app.exception_handler(HTTPException)
    async def _http_exc(request: Request, exc: HTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code,
                            headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", "-")
        log.exception("unhandled error rid=%s", rid)
        return JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)

    @app.get("/health", tags=["ops"])
    def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready(c: Container = Depends(get_container)):
        return {"status": "ready", "llm": c.settings.has_llm, "queue": c.queue.stats()}

    # All module routers mount here — see routes.ROUTE_MODULES for the surface.
    register_routes(app)

    return app
