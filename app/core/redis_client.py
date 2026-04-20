"""Redis connection pool singleton.

Dùng cho cả cache (app/core/cache.py) và làm session store nếu cần.
Broker Celery dùng URL riêng, không chia sẻ pool này.
"""

import os
import redis

from app.core.logging_config import get_logger

logger = get_logger(__name__)

_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    """Trả về Redis client dùng chung. Lazy init, connection pool mặc định."""
    global _client
    if _client is None:
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _client = redis.Redis.from_url(url, decode_responses=True)
        logger.info("Redis client initialized: %s", url)
    return _client


def ping() -> bool:
    """Check Redis có kết nối được không. Dùng cho /ready endpoint."""
    try:
        return bool(get_redis().ping())
    except Exception as e:
        logger.warning("Redis ping failed: %s", e)
        return False
