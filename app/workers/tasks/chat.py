"""Celery task cho chat request dài chạy nền.

Khi FastAPI endpoint /chat/async được gọi, task này được enqueue và trả task_id
cho client. Client poll /tasks/{id} để lấy kết quả.
"""

from celery import shared_task

from app.core.logging_config import get_logger

logger = get_logger(__name__)


@shared_task(bind=True, name="app.workers.tasks.chat.run_chat_task")
def run_chat_task(self, message: str, thread_id: str):
    """Chạy agent đồng bộ (không stream) cho request dài.

    Dùng khi user muốn kết quả đầy đủ thay vì stream (ví dụ: câu hỏi phức tạp
    gọi nhiều tool, hoặc batch evaluation).
    """
    # Import trong task để worker process không bắt buộc load agent lúc boot
    from app.agent.agent import run_agent_routed

    logger.info("run_chat_task start (thread_id=%s)", thread_id)
    self.update_state(state="PROGRESS", meta={"progress": 0.1})

    try:
        result = run_agent_routed(message, thread_id=thread_id)
        logger.info("run_chat_task completed (thread_id=%s)", thread_id)
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.exception("run_chat_task failed: %s", e)
        raise
