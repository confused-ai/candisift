"""Entrypoint: `uvicorn main:app` serves the API + recruiter UI + durable worker."""
from app.candisift.adapters.http.app import create_app

app = create_app()
