"""Celery task cho ghi telemetry (evaluation logger) nền.

FastAPI endpoint sau khi trả response có thể enqueue task này để không block
user request bởi I/O ghi CSV.
"""

from celery import shared_task

from app.core.logging_config import get_logger

logger = get_logger(__name__)


@shared_task(name="app.workers.tasks.telemetry.log_conversation_task")
def log_conversation_task(
    session_id: str,
    turn_number: int,
    user_query: str,
    llm_response: str,
    response_time_ms: float,
    resolved_location: str | None = None,
    tool_calls: list | None = None,
    error_type: str | None = None,
):
    """Ghi 1 dòng conversation vào CSV telemetry. Chạy nền, không block API."""
    from app.agent.telemetry import get_evaluation_logger

    logger_instance = get_evaluation_logger()
    logger_instance.log_conversation(
        session_id=session_id,
        turn_number=turn_number,
        user_query=user_query,
        llm_response=llm_response,
        response_time_ms=response_time_ms,
        resolved_location=resolved_location,
        tool_calls=tool_calls,
        error_type=error_type,
    )
    return {"status": "ok"}
