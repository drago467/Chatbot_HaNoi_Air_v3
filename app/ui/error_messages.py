"""Bảng message tiếng Việt cho lỗi thường gặp khi user chat.

Thay stacktrace/raw exception bằng thông điệp thân thiện. Stacktrace đầy đủ
vẫn được log vào logs/app.log qua _logger.exception(trace_id=...).
"""

from __future__ import annotations

from typing import Optional


ERROR_MESSAGES = {
    "API_DOWN": "Không kết nối được backend (FastAPI). Vui lòng kiểm tra server đang chạy.",
    "OLLAMA_DOWN": "Router (Ollama) tạm thời không phản hồi. Vui lòng thử lại sau giây lát.",
    "LLM_API_DOWN": "Dịch vụ AI tạm thời không khả dụng. Vui lòng thử lại sau ít phút.",
    "DB_DOWN": "Không kết nối được cơ sở dữ liệu thời tiết. Vui lòng báo quản trị viên.",
    "LOCATION_NOT_FOUND": "Không tìm thấy địa điểm trong dữ liệu. Vui lòng kiểm tra tên quận/phường.",
    "TIMEOUT": "Yêu cầu xử lý quá lâu. Vui lòng thử lại với câu hỏi ngắn hơn.",
    "RATE_LIMIT": "Đã vượt giới hạn số lượng yêu cầu. Vui lòng đợi ít phút rồi thử lại.",
    "UNKNOWN": "Có lỗi không mong muốn. Mã tham chiếu: {trace_id}. Vui lòng thử lại.",
}


def classify_error(exc: BaseException) -> str:
    """Map exception → error key trong ERROR_MESSAGES."""
    try:
        import requests
    except ImportError:  # defensive
        requests = None  # type: ignore

    msg = str(exc).lower()
    name = type(exc).__name__.lower()

    if requests is not None:
        if isinstance(exc, requests.ConnectionError):
            return "API_DOWN"
        if isinstance(exc, requests.Timeout):
            return "TIMEOUT"
        if isinstance(exc, requests.HTTPError):
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429:
                return "RATE_LIMIT"
            if status and 500 <= status < 600:
                return "API_DOWN"

    # Check specific patterns trước generic "connection refused"
    if "postgres" in msg or "operationalerror" in name or "psycopg" in msg:
        return "DB_DOWN"
    if "ollama" in msg or "router" in msg:
        return "OLLAMA_DOWN"
    if "openai" in msg or "anthropic" in msg or "api error" in msg:
        return "LLM_API_DOWN"
    if "rate limit" in msg or "429" in msg:
        return "RATE_LIMIT"
    if "location" in msg and ("not found" in msg or "không tìm thấy" in msg):
        return "LOCATION_NOT_FOUND"
    if "timeout" in msg or "timeouterror" in name:
        return "TIMEOUT"
    if "connection refused" in msg or "connection aborted" in msg or "max retries" in msg:
        return "API_DOWN"
    return "UNKNOWN"


def friendly_message(exc: BaseException, trace_id: Optional[str] = None) -> str:
    """Get friendly Vietnamese error message for exception."""
    key = classify_error(exc)
    msg = ERROR_MESSAGES[key]
    if "{trace_id}" in msg:
        msg = msg.format(trace_id=trace_id or "unknown")
    return msg
