"""Pytest fixtures chung.

Cấu hình mặc định:
- Celery chạy eager mode (task thực thi ngay trong test, không cần worker).
- FastAPI TestClient sẵn dùng cho mọi test API.
"""

import os

import pytest


# ── Celery eager mode cho tests ───────────────────────────────────────
# Đặt trước khi import celery_app để config có hiệu lực
os.environ.setdefault("CELERY_BROKER_URL", "memory://")
os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")


@pytest.fixture(autouse=True)
def celery_eager():
    """Chạy Celery task đồng bộ trong test — không cần broker thật."""
    from app.workers.celery_app import celery

    celery.conf.task_always_eager = True
    celery.conf.task_eager_propagates = True
    yield
    celery.conf.task_always_eager = False


@pytest.fixture
def api_client():
    """FastAPI TestClient. Import lazy để fixture không load khi test không cần."""
    from fastapi.testclient import TestClient

    from app.api.main import app

    return TestClient(app)
