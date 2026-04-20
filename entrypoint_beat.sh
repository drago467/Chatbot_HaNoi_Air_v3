#!/bin/bash
# Entrypoint cho Celery Beat scheduler container
set -e

echo "[entrypoint_beat] Starting Celery Beat..."
exec celery -A app.workers.celery_app beat --loglevel=info
