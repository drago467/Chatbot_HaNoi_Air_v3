"""Helper cache đơn giản dùng Redis.

Pattern: cache_get_or_fetch(key, ttl, fetch_fn) — đọc từ Redis, nếu miss thì gọi
fetch_fn() và lưu lại kết quả.

Giá trị được serialize JSON. Không hỗ trợ object phức tạp — chỉ dict/list/str/int/float/bool/None.
"""

import json
from typing import Any, Callable

from app.core.logging_config import get_logger
from app.core.redis_client import get_redis

logger = get_logger(__name__)


def cache_get(key: str) -> Any | None:
    """Lấy giá trị từ cache. Trả None nếu miss hoặc lỗi."""
    try:
        raw = get_redis().get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.warning("cache_get failed for %s: %s", key, e)
        return None


def cache_set(key: str, value: Any, ttl_seconds: int) -> None:
    """Lưu giá trị vào cache với TTL (giây). Bỏ qua nếu lỗi."""
    try:
        get_redis().setex(key, ttl_seconds, json.dumps(value, default=str))
    except Exception as e:
        logger.warning("cache_set failed for %s: %s", key, e)


def cache_get_or_fetch(key: str, ttl_seconds: int, fetch_fn: Callable[[], Any]) -> Any:
    """Đọc cache, nếu miss thì gọi fetch_fn() và cache lại kết quả.

    fetch_fn phải trả về giá trị JSON-serializable. Nếu fetch_fn raise, exception
    được propagate ra ngoài (không cache fail).
    """
    cached = cache_get(key)
    if cached is not None:
        return cached
    value = fetch_fn()
    if value is not None:
        cache_set(key, value, ttl_seconds)
    return value


def cache_delete(key: str) -> None:
    """Xoá 1 key khỏi cache. Dùng khi cần invalidate sau update."""
    try:
        get_redis().delete(key)
    except Exception as e:
        logger.warning("cache_delete failed for %s: %s", key, e)
