"""Chat endpoints.

- POST /chat           : đồng bộ, trả JSON khi agent xong
- POST /chat/stream    : SSE stream token-by-token
- POST /chat/async     : enqueue task, trả task_id để client poll /tasks/{id}
"""

import time

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from app.api.deps import get_thread_id
from app.api.schemas import ChatAsyncResponse, ChatRequest, ChatSyncResponse
from app.api.sse import sync_gen_to_sse
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatSyncResponse)
def chat_sync(req: ChatRequest, thread_id: str = Depends(get_thread_id)):
    """Chat đồng bộ: block đến khi agent trả kết quả. Dùng cho câu đơn giản."""
    from app.agent.agent import run_agent_routed

    t_start = time.time()
    logger.info("chat_sync start (thread_id=%s)", thread_id)
    result = run_agent_routed(req.message, thread_id=thread_id)
    elapsed_ms = (time.time() - t_start) * 1000

    # Enqueue telemetry nền (không block response)
    _enqueue_telemetry(
        thread_id=thread_id,
        user_query=req.message,
        llm_response=_extract_text(result),
        response_time_ms=elapsed_ms,
    )

    return ChatSyncResponse(thread_id=thread_id, result=result)


@router.post("/stream")
def chat_stream(req: ChatRequest, thread_id: str = Depends(get_thread_id)):
    """Chat streaming qua SSE. Ưu tiên dùng cho UI để có TTFT thấp.

    SSE chọn thay vì WebSocket vì:
    - Hướng server -> client đơn luồng đã đủ cho stream token.
    - Qua reverse proxy / CDN dễ hơn, tự động reconnect phía trình duyệt.
    """
    from app.agent.agent import stream_agent_routed

    logger.info("chat_stream start (thread_id=%s)", thread_id)
    sync_gen = stream_agent_routed(req.message, thread_id=thread_id)
    async_gen = sync_gen_to_sse(sync_gen)
    return EventSourceResponse(async_gen, ping=15)


@router.post("/async", response_model=ChatAsyncResponse)
def chat_async(req: ChatRequest, thread_id: str = Depends(get_thread_id)):
    """Enqueue chat task vào Celery, trả task_id cho client poll."""
    from app.workers.tasks.chat import run_chat_task

    task = run_chat_task.delay(req.message, thread_id)
    logger.info("chat_async enqueued (task_id=%s, thread_id=%s)", task.id, thread_id)
    return ChatAsyncResponse(task_id=task.id, status_url=f"/tasks/{task.id}")


def _extract_text(agent_result: dict) -> str:
    """Lấy phần text response từ dict mà agent trả về.

    Format thực tế phụ thuộc LangGraph — thường là {messages: [...]} với
    message cuối là AIMessage. Ở đây lấy content dạng str để log telemetry.
    """
    try:
        msgs = agent_result.get("messages") if isinstance(agent_result, dict) else None
        if msgs:
            last = msgs[-1]
            # langchain Message object có thuộc tính .content
            content = getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else None)
            if isinstance(content, str):
                return content
    except Exception:
        pass
    return ""


def _enqueue_telemetry(thread_id: str, user_query: str, llm_response: str, response_time_ms: float):
    """Gửi telemetry task vào Celery. Silent-fail nếu broker down."""
    try:
        from app.workers.tasks.telemetry import log_conversation_task
        log_conversation_task.delay(
            session_id=thread_id,
            turn_number=0,  # API chưa track turn; để 0, client UI set sau nếu cần
            user_query=user_query,
            llm_response=llm_response,
            response_time_ms=response_time_ms,
        )
    except Exception as e:
        logger.warning("Could not enqueue telemetry task: %s", e)
