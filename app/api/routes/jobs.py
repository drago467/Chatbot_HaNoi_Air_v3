"""Job endpoints — chạy ingest weather đồng bộ.

Sau R15 gỡ Celery, endpoint này chạy sync trong request thread. Client nên
chấp nhận timeout dài (60-300s cho include_history). Scope khóa luận
single-user, chạy 1 lần/ngày đủ.

Alternative: `python -m app.scripts.ingest_openweather_async` CLI/cron.
"""

from fastapi import APIRouter, HTTPException

from app.api.schemas import IngestRequest, IngestResponse
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest weather data (sync)",
    description=(
        "Gọi OpenWeather API để cập nhật dữ liệu weather cho 126 phường Hà Nội. "
        "Sau R15 gỡ Celery: endpoint chạy đồng bộ trong request thread — "
        "client cần chấp nhận timeout dài.\n\n"
        "**Modes**:\n"
        "- `include_history=false`: current + forecast (~30-60s).\n"
        "- `include_history=true`: thêm history N ngày (~2-5 phút cho 7 ngày).\n\n"
        "**Alternative**: chạy CLI `python -m app.scripts.ingest_openweather_async` "
        "hoặc `make ingest` / `make ingest-history`."
    ),
    responses={
        200: {"description": "Ingest hoàn tất"},
        500: {"description": "Ingest fail (rate limit, API key, DB, ...)"},
    },
)
def run_ingest_endpoint(req: IngestRequest):
    """Chạy ingest weather đồng bộ. Block đến khi xong."""
    logger.info(
        "Ingest job start: history=%s, days=%d",
        req.include_history, req.history_days,
    )

    try:
        from app.scripts.ingest_openweather_async import run_ingest
        run_ingest(
            include_history=req.include_history,
            history_days=req.history_days,
        )
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(500, f"Ingest failed: {e}")

    logger.info("Ingest job done")
    return IngestResponse(
        status="ok",
        include_history=req.include_history,
        history_days=req.history_days,
        message="Ingest hoàn tất",
    )
