"""HTTP dependencies: container access + HTTP Basic auth for the recruiter UI/API."""
from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .container import Container

_security = HTTPBasic()


def get_container(request: Request) -> Container:
    return request.app.state.container


def require_auth(
    request: Request,
    creds: HTTPBasicCredentials = Depends(_security),
) -> None:
    s = get_container(request).settings
    ok = (
        secrets.compare_digest(creds.username, s.basic_auth_user)
        and secrets.compare_digest(creds.password, s.basic_auth_pass)
    )
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
