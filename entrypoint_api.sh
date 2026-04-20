#!/bin/bash
# Entrypoint cho FastAPI container
set -e

echo "[entrypoint_api] Initializing database schema..."
python -m app.db.init_db

echo "[entrypoint_api] Starting uvicorn on ${API_HOST:-0.0.0.0}:${API_PORT:-8000}..."
exec uvicorn app.api.main:app \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-8000}" \
    --log-level info
