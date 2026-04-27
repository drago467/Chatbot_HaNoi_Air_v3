"""SSE helper: wrap sync generator của agent thành async event source.

stream_agent_routed() là generator đồng bộ (chạy LangGraph). Để không block
FastAPI event loop, chạy mỗi lần next() trong threadpool.
"""

import asyncio
import contextvars
from typing import AsyncIterator, Iterator

from app.core.logging_config import get_logger

logger = get_logger(__name__)


async def sync_gen_to_sse(sync_gen: Iterator[str]) -> AsyncIterator[dict]:
    """Chuyển sync generator yield text chunks -> async generator yield SSE events.

    Mỗi chunk được gửi như event "token". Khi generator hết, gửi event "done".
    Nếu có exception, gửi event "error" và kết thúc.
    """
    loop = asyncio.get_event_loop()
    # Capture a single context so every next() call resumes the generator in
    # the same context. Without this, run_in_executor copies the caller context
    # per submission, so ContextVar tokens set in one next() can't be reset in
    # a later one (ValueError: Token was created in a different Context).
    ctx = contextvars.copy_context()

    def _safe_next():
        """Gọi next() trong thread, trả None khi StopIteration."""
        try:
            return ctx.run(next, sync_gen)
        except StopIteration:
            return None

    try:
        while True:
            chunk = await loop.run_in_executor(None, _safe_next)
            if chunk is None:
                yield {"event": "done", "data": ""}
                return
            yield {"event": "token", "data": chunk}
    except Exception as e:
        logger.exception("SSE generator error: %s", e)
        yield {"event": "error", "data": str(e)}
