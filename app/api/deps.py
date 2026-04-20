"""FastAPI dependencies dùng chung cho các route."""

import uuid
from typing import Optional

from fastapi import Header


def get_thread_id(
    x_thread_id: Optional[str] = Header(default=None, alias="X-Thread-Id"),
) -> str:
    """Lấy thread_id từ header. Nếu không có, tạo mới (UUID4).

    Usage trong route:
        def endpoint(thread_id: str = Depends(get_thread_id)): ...
    """
    if x_thread_id and x_thread_id.strip():
        return x_thread_id.strip()
    return str(uuid.uuid4())
