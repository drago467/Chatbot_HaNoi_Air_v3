"""Health & readiness endpoints.

- GET /health: liveness (process còn sống)
- GET /ready: readiness (các dependency còn hoạt động — Postgres/Ollama/LLM)
"""

import os

import httpx
from fastapi import APIRouter

from app.api.schemas import HealthResponse, ReadyResponse
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Trả 200 nếu FastAPI process đang chạy. Không check dependency.",
)
def health():
    """Liveness probe."""
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    description=(
        "Check các dependency: Postgres (SELECT 1), Ollama (GET /api/tags) "
        "nếu router bật, LLM API (env var AGENT_API_KEY set). Mỗi field "
        "trả 'ok' / 'error' / 'disabled' / 'missing'."
    ),
)
def ready():
    """Readiness probe — check tất cả dependency."""
    return ReadyResponse(
        postgres=_check_postgres(),
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
