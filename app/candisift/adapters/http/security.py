"""HTTP security hardening: middleware + upload validation.

Driving-side guardrails so the funnel never sees malformed or hostile requests:
  - security response headers (CSP, nosniff, frame-deny, referrer, permissions)
  - request body size cap (reject token/zip bombs before buffering)
  - per-client rate limiting (in-memory fixed window)
  - request id for traceable logs
  - upload validation (extension allowlist, per-file + count caps)

ponytail: the rate limiter is in-memory (single process). For multiple instances
put the window in Redis; the middleware interface stays the same.
"""
from __future__ import annotations

import time
import uuid
from collections import deque

from fastapi import HTTPException, UploadFile, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# image types are accepted only because the extractor OCRs them (Tesseract)
ALLOWED_EXTENSIONS = (
    ".pdf", ".docx", ".txt", ".md",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
)


# ---- middleware -----------------------------------------------------------

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, hsts: bool = False) -> None:
        super().__init__(app)
        self._hsts = hsts

    async def dispatch(self, request: Request, call_next):
        resp: Response = await call_next(request)
        # strict, self-only policy — the UI uses no inline scripts or external assets
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        if self._hsts:
            resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return resp


class BodySizeLimitMiddleware:
    """Reject oversized request bodies BEFORE a handler buffers them. Pure-ASGI (not
    BaseHTTPMiddleware) because we must rewrite the receive channel the downstream app
    actually reads from — BaseHTTPMiddleware drains the stream without propagating a
    replacement, which silently empties every request body.

    Content-Length is the fast path, but it is advisory: a chunked transfer or a
    client that omits the header would stream an unbounded body straight into
    `await upload.read()` and exhaust memory. So we drain the body here with a hard
    cap — memory is bounded to `max_bytes`, not the whole hostile body — and replay
    the buffered bytes to the app so handlers still read normally."""

    def __init__(self, app, *, max_bytes: int) -> None:
        self.app = app
        self._max = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        cl = headers.get(b"content-length")
        if cl is not None and cl.isdigit() and int(cl) > self._max:
            await self._reject(send)
            return

        # buffer the body up to the cap (covers chunked / absent Content-Length)
        body = bytearray()
        more = True
        while more:
            msg = await receive()
            if msg["type"] != "http.request":
                break
            body.extend(msg.get("body", b""))
            more = msg.get("more_body", False)
            if len(body) > self._max:
                await self._reject(send)
                return

        buffered = bytes(body)
        sent = False

        async def replay():
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": buffered, "more_body": False}

        await self.app(scope, replay, send)

    async def _reject(self, send) -> None:
        await send({"type": "http.response.start",
                    "status": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"detail":"request too large"}'})


def _strip_port(entry: str) -> str:
    """"203.0.113.7:41233" -> "203.0.113.7", so one client isn't split across buckets by
    source port. A bare IPv6 ("2001:db8::1", several colons) is left alone; only a
    trailing ":port" on an IPv4 or a bracketed [v6]:port is trimmed."""
    if entry.startswith("["):                       # [2001:db8::1]:443
        return entry[1:].split("]")[0]
    if entry.count(":") == 1:                        # ipv4:port
        return entry.split(":")[0]
    return entry


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory fixed-window limiter. ponytail: per-process; behind multiple
    instances move the window to Redis (interface unchanged). The hit map is pruned
    of stale clients each request and hard-capped so an IP-rotating flood can't grow
    it without bound (the map itself was a memory-exhaustion vector)."""

    _MAX_CLIENTS = 50_000

    def __init__(self, app, *, per_minute: int, trusted_proxy_count: int = 0) -> None:
        super().__init__(app)
        self._per_minute = per_minute
        self._trusted_proxy_count = max(0, int(trusted_proxy_count))
        self._hits: dict[str, deque] = {}

    def _client_ip(self, request: Request) -> str:
        """The bucket key. Keying on the socket peer is correct ONLY on direct
        exposure: behind a proxy/LB the peer is the proxy, so every user collapses
        into one bucket — one noisy client 429s the whole site, and the limiter stops
        being the brute-force fence on Basic auth. X-Forwarded-For is client-appendable,
        so trust it exactly as far as the operator says: with N trusted hops in front,
        the last N entries were appended by them and the Nth-from-the-right is the
        address our own edge saw. Everything left of it is attacker-controlled."""
        n = self._trusted_proxy_count
        if n:
            # getlist, not get: a proxy may append its own SEPARATE X-Forwarded-For
            # header line rather than extending the first, and get() returns only the
            # first line — so an attacker's forged line would be read as the client while
            # the real one is ignored. Flatten every line, then index from the right.
            parts = [p.strip() for line in request.headers.getlist("X-Forwarded-For")
                     for p in line.split(",") if p.strip()]
            if len(parts) >= n:
                return _strip_port(parts[-n])
            # header absent or shorter than the trusted chain (a request that bypassed
            # the proxy) -> fall back to the peer, never to a spoofable entry.
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        client = self._client_ip(request)
        now = time.monotonic()
        # opportunistic GC: drop clients whose window has fully aged out, so idle
        # IPs don't accumulate empty deques forever.
        if len(self._hits) > self._MAX_CLIENTS:
            self._hits = {k: w for k, w in self._hits.items() if w and now - w[-1] <= 60}
        window = self._hits.setdefault(client, deque())
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self._per_minute:
            return JSONResponse({"detail": "rate limit exceeded"},
                                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                headers={"Retry-After": "60"})
        window.append(now)
        return await call_next(request)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        resp = await call_next(request)
        resp.headers["X-Request-ID"] = rid
        return resp


# ---- upload validation ----------------------------------------------------

def validate_uploads(files: list[UploadFile], *, max_files: int, max_file_bytes: int) -> None:
    """Cheap checks before reading bodies: count + extension. Per-file byte size is
    enforced after read in the API (UploadFile doesn't expose length up front)."""
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "no files uploaded")
    if len(files) > max_files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"too many files ({len(files)} > {max_files})")
    for f in files:
        name = (f.filename or "").lower()
        if not name.endswith(ALLOWED_EXTENSIONS):
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                f"unsupported file type: {f.filename} (allowed: {ALLOWED_EXTENSIONS})")


def check_size(content: bytes, filename: str, max_file_bytes: int) -> None:
    if len(content) > max_file_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"{filename} exceeds {max_file_bytes // (1024 * 1024)}MB")
