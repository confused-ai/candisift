#!/usr/bin/env bash
# Install deps into a local venv and launch CandiSift (API + UI + worker, one process).
# Usage:
#   ./run.sh                 # install + run on http://127.0.0.1:8000
#   ./run.sh --reload        # dev autoreload (any extra args pass through to uvicorn)
#   PORT=9000 ./run.sh       # override port
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
PY="${PYTHON:-python3}"

# 1. venv (created once, reused after)
if [[ ! -d .venv ]]; then
  echo "→ creating .venv"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 2. python deps
echo "→ installing python deps"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# 3. OCR system binaries (optional — app runs without them, scans just won't OCR)
for bin in tesseract pdftoppm; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "⚠ '$bin' not found — OCR for scanned PDFs/images disabled."
    echo "   macOS:  brew install tesseract poppler"
    echo "   Debian: sudo apt-get install tesseract-ocr poppler-utils"
  fi
done

# 4. first-run config scaffold (offline stub works with no key)
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "→ wrote .env from .env.example (edit it to set ANTHROPIC_API_KEY for real LLM)"
fi

# 5. launch
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || echo "unknown")
echo "→ http://${HOST}:${PORT}  (UI login: recruiter / change-me)"
echo "→ network: http://${LOCAL_IP}:${PORT}"
exec uvicorn main:app --host "$HOST" --port "$PORT" "$@"
