"""Task status polling endpoint.

Trả trạng thái Celery task cho client polling. Cấu trúc response thống nhất
cho mọi loại task (chat, ingest, ...).
"""

from fastapi import APIRouter
from celery.result import AsyncResult

from app.api.schemas import TaskStatusResponse
from app.workers.celery_app import celery

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """Lấy trạng thái task. Client poll endpoint này mỗi 1-2s."""
    result = AsyncResult(task_id, app=celery)
    state = result.state
    response = TaskStatusResponse(task_id=task_id, state=state)

    if state == "PROGRESS":
        info = result.info if isinstance(result.info, dict) else {}
        response.progress = float(info.get("progress", 0.0))
        response.step = info.get("step")
    elif state == "SUCCESS":
        response.progress = 1.0
        response.result = result.result
    elif state == "FAILURE":
        response.error = str(result.info) if result.info else "unknown error"

    return response
