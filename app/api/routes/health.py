"""Health & readiness endpoints.

- GET /health: liveness (process còn sống)
- GET /ready: readiness (các dependency còn hoạt động — Postgres/Redis/Ollama/LLM)
"""

import os

import httpx
from fastapi import APIRouter

from app.api.schemas import HealthResponse, ReadyResponse
from app.core.logging_config import get_logger
from app.core.redis_client import ping as redis_ping

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe. Không check dependency, chỉ trả 200 nếu process chạy."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
def ready():
    """Readiness probe. Check tất cả dependency."""
    return ReadyResponse(
        postgres=_check_postgres(),
        redis=_check_redis(),
        router=_check_ollama(),
        llm=_check_llm_config(),
    )


def _check_postgres() -> str:
    """Ping Postgres bằng SELECT 1."""
    try:
        from app.db.connection import get_db_connection, release_connection
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return "ok"
        finally:
            release_connection(conn)
    except Exception as e:
        logger.warning("Postgres check failed: %s", e)
        return "error"


def _check_redis() -> str:
    return "ok" if redis_ping() else "error"


def _check_ollama() -> str:
    """Check Ollama nếu router SLM đang bật."""
    try:
        from app.agent.router.config import OLLAMA_BASE_URL, USE_SLM_ROUTER
    except Exception:
        return "unknown"

    if not USE_SLM_ROUTER:
        return "disabled"

    try:
        r = httpx.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=3.0)
        return "ok" if r.status_code == 200 else "error"
    except Exception:
        return "error"


def _check_llm_config() -> str:
    """Check env var LLM có được set không (không gọi API thật để tránh cost)."""
    key = os.getenv("AGENT_API_KEY") or os.getenv("API_KEY")
    return "ok" if key else "missing"
