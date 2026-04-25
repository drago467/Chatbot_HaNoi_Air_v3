"""Chat endpoints.

- POST /chat           : đồng bộ, trả JSON khi agent xong
- POST /chat/stream    : SSE stream token-by-token
"""

import time

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from app.api.deps import get_thread_id
from app.api.schemas import ChatRequest, ChatSyncResponse
from app.api.sse import sync_gen_to_sse
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "",
    response_model=ChatSyncResponse,
    summary="Chat đồng bộ",
    description=(
        "Gửi câu hỏi → block đến khi agent trả kết quả đầy đủ. "
        "Dùng cho câu đơn giản hoặc client không cần streaming.\n\n"
        "**Flow**:\n"
        "1. SLM Router (Qwen3-1.4B) phân loại intent.\n"
        "2. Agent (Qwen3-14B thinking) chọn tools theo focused set.\n"
        "3. Gọi tools (DAL query + format).\n"
        "4. Agent tổng hợp response.\n"
        "5. Trả full dict + log telemetry.\n\n"
        "**Latency**: 3-8s tùy câu (thinking mode bật)."
    ),
    responses={
        200: {"description": "Agent trả response thành công"},
        503: {"description": "Backend (Ollama/LLM/DB) không khả dụng"},
    },
)
def chat_sync(req: ChatRequest, thread_id: str = Depends(get_thread_id)):
    """Chat đồng bộ: block đến khi agent trả kết quả. Dùng cho câu đơn giản."""
    from app.agent.agent import run_agent_routed

    t_start = time.time()
    logger.info("chat_sync start (thread_id=%s)", thread_id)
    result = run_agent_routed(req.message, thread_id=thread_id)
    elapsed_ms = (time.time() - t_start) * 1000

    _log_telemetry(
        thread_id=thread_id,
        user_query=req.message,
        llm_response=_extract_text(result),
        response_time_ms=elapsed_ms,
    )

    return ChatSyncResponse(thread_id=thread_id, result=result)


@router.post(
    "/stream",
    summary="Chat streaming (SSE)",
    description=(
        "Gửi câu hỏi → nhận response streaming qua Server-Sent Events. "
        "Ưu tiên dùng cho UI để có TTFT (Time To First Token) thấp.\n\n"
        "**SSE events**:\n"
        "- `event: token` — chunk text assistant\n"
        "- `event: done` — stream kết thúc\n"
        "- `event: error` — lỗi trong agent chain\n\n"
        "**Tại sao SSE thay WebSocket?**\n"
        "- Hướng server → client đơn luồng đã đủ cho stream token.\n"
        "- Qua reverse proxy / CDN dễ hơn, tự động reconnect phía trình duyệt."
    ),
    responses={
        200: {
            "description": "SSE stream thành công",
            "content": {"text/event-stream": {}},
        },
        503: {"description": "Backend không khả dụng"},
    },
)
def chat_stream(req: ChatRequest, thread_id: str = Depends(get_thread_id)):
    """Chat streaming qua SSE."""
    from app.agent.agent import stream_agent_routed

    logger.info("chat_stream start (thread_id=%s)", thread_id)
    sync_gen = stream_agent_routed(req.message, thread_id=thread_id)
    async_gen = sync_gen_to_sse(sync_gen)
    return EventSourceResponse(async_gen, ping=15)


def _extract_text(agent_result: dict) -> str:
    """Lấy phần text response từ dict mà agent trả về.

    Format thực tế phụ thuộc LangGraph — thường là {messages: [...]} với
    message cuối là AIMessage. Ở đây lấy content dạng str để log telemetry.
    """
    try:
        msgs = agent_result.get("messages") if isinstance(agent_result, dict) else None
        if msgs:
            last = msgs[-1]
            content = getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else None)
            if isinstance(content, str):
                return content
    except Exception:
        pass
    return ""


def _log_telemetry(thread_id: str, user_query: str, llm_response: str, response_time_ms: float):
    """Log telemetry đồng bộ vào CSV. Silent-fail nếu logger down."""
    try:
        from app.agent.telemetry import get_evaluation_logger
        logger_instance = get_evaluation_logger()
        logger_instance.log_conversation(
            session_id=thread_id,
            turn_number=0,
            user_query=user_query,
            llm_response=llm_response,
            response_time_ms=response_time_ms,
        )
    except Exception as e:
        logger.warning("Could not log telemetry: %s", e)
