"""Compatibility patch cho langchain-core.

Một số OpenAI-compatible providers trả về None cho các field trong usage_metadata
(ví dụ: prompt_tokens = None) khiến hàm _dict_int_op trong langchain-core crash
khi streaming. Patch này xem None như 0.

Import module này 1 lần ở entry point (app.py, app/api/main.py) TRƯỚC khi dùng agent.
"""

import langchain_core.utils.usage as _usage_mod

_original_dict_int_op = _usage_mod._dict_int_op
_patched = False


def _patched_dict_int_op(left, right, op, *, default=0, depth=0, max_depth=100):
    # Thay None -> 0 để không break phép cộng integer
    cleaned_left = {k: (0 if v is None else v) for k, v in left.items()}
    cleaned_right = {k: (0 if v is None else v) for k, v in right.items()}
    return _original_dict_int_op(
        cleaned_left,
        cleaned_right,
        op,
        default=default,
        depth=depth,
        max_depth=max_depth,
    )


def apply_patch() -> None:
    """Áp dụng monkey-patch. Idempotent — gọi nhiều lần không sao."""
    global _patched
    if _patched:
        return
    _usage_mod._dict_int_op = _patched_dict_int_op
    _patched = True


# Tự áp dụng khi import
apply_patch()
