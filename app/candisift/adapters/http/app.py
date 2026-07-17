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

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.candisift.application.screening_service import NotFoundError
from app.candisift.adapters.worker import Worker
from .container import Container, build_container
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
    app.add_middleware(RateLimitMiddleware, per_minute=s.rate_limit_per_min,
                       trusted_proxy_count=s.trusted_proxy_count)
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

    # ---- probes ----
    # ponytail: the engine is reached through the task queue because Container doesn't
    # expose it. Upgrade path: bind an explicit `engine` (or a HealthCheck port) in the
    # composition root and ping through that instead of a private attribute.
    engine = getattr(container.queue, "_engine", None)

    def _db_ok() -> bool:
        """One round trip to the DB. A static probe stayed green while the DB was
        corrupt, locked, or (on Turso) unreachable — the container kept serving 200s
        and no orchestrator ever restarted it."""
        if engine is None:
            return False
        try:
            with engine.connect() as conn:
                conn.exec_driver_sql("SELECT 1")
            return True
        except Exception:  # noqa: BLE001 — any failure to talk to the DB is unhealthy
            log.exception("health: DB ping failed")
            return False

    def _worker_ok() -> bool:
        """Worker threads are daemons started in lifespan. If they die the API keeps
        answering while nothing is screened — the exact silent failure a healthcheck
        exists to catch. Empty => lifespan never ran (or we are shutting down)."""
        threads = getattr(worker, "_threads", [])
        return bool(threads) and all(t.is_alive() for t in threads)

    def _probe(ok_status: str) -> tuple[dict, int]:
        db, wk = _db_ok(), _worker_ok()
        healthy = db and wk
        return ({"status": ok_status if healthy else "degraded", "db": db, "worker": wk},
                200 if healthy else 503)

    @app.get("/health", tags=["ops"])
    def health():
        """Backs the Docker HEALTHCHECK: 503 => replace the container. Both checks are
        cheap (one SELECT 1 + a thread flag), so a 30s interval costs nothing."""
        body, code = _probe("ok")
        return body if code == 200 else JSONResponse(body, status_code=code)

    @app.get("/ready", tags=["ops"])
    def ready():
        """Booleans only — this endpoint is UNAUTHENTICATED. It used to return queue
        depth and whether an LLM key was configured: free recon (pipeline volume, and
        whether spend is real) for anyone who asked. The detailed view already lives
        under auth at GET /api/queue."""
        body, code = _probe("ready")
        return body if code == 200 else JSONResponse(body, status_code=code)

    # All module routers mount here — see routes.ROUTE_MODULES for the surface.
    register_routes(app)

    return app
