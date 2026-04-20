#!/bin/bash
# Entrypoint cho Celery worker container
set -e

echo "[entrypoint_worker] Starting Celery worker..."
exec celery -A app.workers.celery_app worker \
    --loglevel=info \
    --concurrency="${CELERY_WORKER_CONCURRENCY:-4}"
