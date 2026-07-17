FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OCR system binaries (Tesseract engine + Poppler for PDF->image). Without
# these the app still runs but scanned PDFs / images degrade to empty text.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY main.py .

# Durable state. The DB defaults to a file in the working dir — inside an image that
# is the container's writable layer, which is DISCARDED on every replacement (upgrade,
# crash, `docker rm`), taking every candidate, result, and audit row with it. Point it
# at /data and declare the volume so the data outlives the container:
#   docker run -v candisift-data:/data ...
# (4 slashes = absolute path. See README for the WAL-safe backup procedure.)
ENV CANDISIFT_DB_URL=sqlite+libsql:////data/candisift.db

# Refuse to boot on default creds / wildcard CORS / HSTS-off (config.validate_runtime).
# A container is a deployment, so it gets the deployment's guardrail — the operator
# must pass real credentials. Override with -e CANDISIFT_ENV=dev for a local demo.
ENV CANDISIFT_ENV=prod

# non-root runtime user; owns the working dir and the state volume (a volume
# inherits the ownership of its mount point at first use, so chown before VOLUME)
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data
VOLUME /data
USER appuser

EXPOSE 8000

# Container healthcheck. /health is unauthenticated but no longer static: it pings the
# DB and checks worker-thread liveness, so a corrupt/locked DB or a dead worker now
# fails the probe (503) instead of sitting green while nothing gets screened.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
