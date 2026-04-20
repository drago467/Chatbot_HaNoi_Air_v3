"""Job endpoints — enqueue long-running task vào Celery."""

from fastapi import APIRouter

from app.api.schemas import IngestRequest, JobResponse
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/ingest", response_model=JobResponse)
def enqueue_ingest(req: IngestRequest):
    """Enqueue ingest + aggregate job. Trả task_id để client poll /tasks/{id}.

    Client UI (Streamlit) dùng endpoint này thay cho việc chạy async trực tiếp
    (giải quyết vấn đề UI block trước đây).
    """
    from app.workers.tasks.ingest import ingest_weather_task

    task = ingest_weather_task.delay(
        include_history=req.include_history,
        history_days=req.history_days,
    )
    logger.info(
        "Ingest job enqueued: task_id=%s, history=%s, days=%d",
        task.id, req.include_history, req.history_days,
    )
    return JobResponse(task_id=task.id, status_url=f"/tasks/{task.id}")
