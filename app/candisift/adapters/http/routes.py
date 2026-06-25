"""Single registration surface for HTTP route modules.

Add a module = add one `RouteModule` row to `ROUTE_MODULES`. The app factory
(`app.py`) never changes — it just calls `register_routes(app)`.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, Depends, FastAPI

from .api import router as api_router
from .deps import require_auth
from .ui import router as ui_router


@dataclass(frozen=True)
class RouteModule:
    name: str
    router: APIRouter
    prefix: str = ""
    auth: bool = True  # gate behind require_auth (Basic auth)


# The clean surface: one row per mounted module.
ROUTE_MODULES: list[RouteModule] = [
    RouteModule("api", api_router, prefix="/api", auth=True),
    RouteModule("ui", ui_router, auth=True),
]


def register_routes(app: FastAPI) -> None:
    """Mount every module in ROUTE_MODULES onto the app."""
    seen: set[tuple[str, str]] = set()
    for m in ROUTE_MODULES:
        key = (m.name, m.prefix)
        assert key not in seen, f"duplicate route module {key}"
        seen.add(key)
        deps = [Depends(require_auth)] if m.auth else []
        app.include_router(m.router, prefix=m.prefix, dependencies=deps)


if __name__ == "__main__":  # ponytail: registry-level self-check (FastAPI includes lazily)
    assert len({(m.name, m.prefix) for m in ROUTE_MODULES}) == len(ROUTE_MODULES), "duplicate module"
    for m in ROUTE_MODULES:
        assert m.router.routes, f"{m.name}: router has no routes"
        assert m.prefix == "" or m.prefix.startswith("/"), f"{m.name}: bad prefix {m.prefix!r}"
    register_routes(FastAPI())  # must not raise
    print(f"ok: {len(ROUTE_MODULES)} modules registered "
          f"({sum(len(m.router.routes) for m in ROUTE_MODULES)} routes)")
