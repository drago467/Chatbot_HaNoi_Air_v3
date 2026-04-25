"""Pytest fixtures chung."""

import pytest


@pytest.fixture
def api_client():
    """FastAPI TestClient. Import lazy để fixture không load khi test không cần."""
    from fastapi.testclient import TestClient

    from app.api.main import app

    return TestClient(app)
