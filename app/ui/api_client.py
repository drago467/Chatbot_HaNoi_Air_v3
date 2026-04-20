"""HTTP client để Streamlit gọi FastAPI.

Wrapper đơn giản quanh requests. Tất cả network call từ Streamlit UI đi qua
module này, giúp dễ mock khi test và đổi URL bằng env var API_URL.
"""

import os
from typing import Iterator, Optional

import requests

from app.core.logging_config import get_logger

logger = get_logger(__name__)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Timeout mặc định cho request thường
_DEFAULT_TIMEOUT = (5, 30)
# Timeout cho stream: connect 5s, read 5 phút (câu dài + thinking mode)
_STREAM_TIMEOUT = (5, 300)


# ── Chat ──────────────────────────────────────────────────────────────


def chat_stream(message: str, thread_id: str) -> Iterator[str]:
    """Gọi FastAPI /chat/stream, yield từng token text.

    Parse SSE đúng spec: dòng rỗng = ranh giới event, nhiều dòng `data:`
    được nối lại bằng `\\n`. Chỉ strip đúng 1 space sau `data:` (nếu có) để
    KHÔNG mất whitespace thật trong token (ví dụ: leading space giữa từ,
    `\\n` đầu token cho markdown bullet).
    """
    with requests.post(
        f"{API_URL}/chat/stream",
        json={"message": message},
        headers={
            "X-Thread-Id": thread_id,
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
        stream=True,
        timeout=_STREAM_TIMEOUT,
    ) as resp:
        resp.raise_for_status()

        event_type: Optional[str] = None
        data_buf: list[str] = []

        for raw_line in resp.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue

            # Comment/keepalive (ví dụ `: ping`)
            if raw_line.startswith(":"):
                continue

            # Dòng rỗng = ranh giới event → flush event đã accumulate
            if raw_line == "":
                if event_type is None and not data_buf:
                    continue
                payload = "\n".join(data_buf)
                et = event_type
                event_type = None
                data_buf = []
                if et == "done":
                    return
                if et == "error":
                    raise RuntimeError(f"Stream error: {payload}")
                if payload:
                    yield payload
                continue

            if raw_line.startswith("event:"):
                event_type = raw_line[6:].strip()
                continue

            if raw_line.startswith("data:"):
                # Strip đúng 1 space sau ":" theo spec, giữ nguyên phần còn lại
                d = raw_line[5:]
                if d.startswith(" "):
                    d = d[1:]
                data_buf.append(d)


def chat_sync(message: str, thread_id: str) -> dict:
    """Gọi /chat sync, trả full result dict."""
    r = requests.post(
        f"{API_URL}/chat",
        json={"message": message},
        headers={"X-Thread-Id": thread_id, "Content-Type": "application/json"},
        timeout=_STREAM_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def chat_async(message: str, thread_id: str) -> str:
    """Enqueue chat task. Trả task_id."""
    r = requests.post(
        f"{API_URL}/chat/async",
        json={"message": message},
        headers={"X-Thread-Id": thread_id, "Content-Type": "application/json"},
        timeout=_DEFAULT_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["task_id"]


# ── Jobs & task polling ───────────────────────────────────────────────


def enqueue_ingest(include_history: bool = False, history_days: int = 7) -> str:
    """Enqueue data refresh job. Trả task_id."""
    r = requests.post(
        f"{API_URL}/jobs/ingest",
        json={"include_history": include_history, "history_days": history_days},
        timeout=_DEFAULT_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["task_id"]


def poll_task(task_id: str) -> dict:
    """Poll trạng thái task."""
    r = requests.get(f"{API_URL}/tasks/{task_id}", timeout=_DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()


# ── Location & weather ────────────────────────────────────────────────


def get_districts() -> list[str]:
    try:
        r = requests.get(f"{API_URL}/locations/districts", timeout=_DEFAULT_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("get_districts failed: %s", e)
        return []


def get_wards(district: str) -> dict[str, str]:
    try:
        r = requests.get(f"{API_URL}/locations/wards/{district}", timeout=_DEFAULT_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("get_wards failed: %s", e)
        return {}


def get_current_weather(ward_id: str) -> Optional[dict]:
    try:
        r = requests.get(f"{API_URL}/weather/current/{ward_id}", timeout=_DEFAULT_TIMEOUT)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("get_current_weather failed: %s", e)
        return None


def get_hourly_forecast(ward_id: str, hours: int = 24) -> list[dict]:
    try:
        r = requests.get(
            f"{API_URL}/weather/forecast/{ward_id}",
            params={"hours": hours},
            timeout=_DEFAULT_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("get_hourly_forecast failed: %s", e)
        return []


# ── Health ────────────────────────────────────────────────────────────


def get_ready_status() -> dict:
    """Check /ready để biết dependencies còn hoạt động."""
    try:
        r = requests.get(f"{API_URL}/ready", timeout=3.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("get_ready_status failed: %s", e)
        return {"postgres": "error", "redis": "error", "router": "error", "llm": "error"}
