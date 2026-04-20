"""HanoiWeather — Streamlit UI entry point.

Streamlit chỉ đóng vai trò renderer: toàn bộ logic agent/tools đi qua
FastAPI (app/ui/api_client.py). Layout chia làm 3 vùng cô lập trong
@st.fragment để tương tác với sidebar/info panel KHÔNG cancel chat stream:

    ┌─ Sidebar ─────────┐ ┌─ Chat area ────────────┐ ┌─ Info panel ─┐
    │ (fragment)        │ │ (fragment, streaming)  │ │ (plain)      │
    └───────────────────┘ └────────────────────────┘ └──────────────┘

`st.chat_input` đặt ở top của main body (trước columns) để Streamlit tự
pin ở đáy viewport. Giá trị truyền vào fragment qua session_state.pending_prompt.

Run: streamlit run app.py
"""

from __future__ import annotations

import logging
import time

import streamlit as st

# Langchain monkey-patch cho usage_metadata None — phải import trước agent
from app.core import compat  # noqa: F401
from app.core.logging_config import setup_logging
from app.ui import api_client
from app.ui.components import (
    get_active_conversation,
    init_session_state,
    render_info_panel,
    render_sidebar,
    render_welcome_message,
    should_show_welcome,
)
from app.ui.styles import CUSTOM_CSS

setup_logging()
_logger = logging.getLogger(__name__)


# ── Page config ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chatbot Thời Tiết Hà Nội",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Init session state + sidebar ──────────────────────────────────────

init_session_state()
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

render_sidebar()

# ── Chat input — đặt ở TOP main body để Streamlit tự pin đáy viewport ─
# (Nếu đặt sau st.columns/containers thì behavior pinned-to-bottom mất.)

user_input = st.chat_input("Hỏi về thời tiết Hà Nội...")
if user_input:
    st.session_state.pending_prompt = user_input

# ── Layout: chat area (+ optional info panel) ─────────────────────────

conv = get_active_conversation()

if st.session_state.show_info_panel:
    chat_col, info_col = st.columns([3, 1.2])
else:
    chat_col = st.container()
    info_col = None


# ── Chat handler (stream + persist) ───────────────────────────────────


def _persist_conversation(conv: dict) -> None:
    """Ghi conversation xuống DB. Silent-fail + log nếu DB down."""
    from app.db.conversation_dal import update_conversation

    try:
        update_conversation(
            conv_id=st.session_state.active_id,
            title=conv["title"],
            messages=conv["messages"],
            updated_at=conv["updated_at"],
        )
    except Exception as e:
        _logger.warning("Could not persist conversation: %s", e)


def _handle_new_message(conv: dict, prompt: str) -> None:
    """Thêm tin nhắn user, stream response, lưu DB. Chạy trong chat fragment."""
    from app.dal.timezone_utils import now_ict

    # Ẩn welcome ngay khi có turn đầu tiên
    conv["welcome_dismissed"] = True

    conv["messages"].append({"role": "user", "content": prompt})

    # Auto-title từ câu hỏi đầu tiên của user
    if sum(1 for m in conv["messages"] if m["role"] == "user") == 1:
        conv["title"] = prompt[:30] + ("..." if len(prompt) > 30 else "")

    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream response từ FastAPI SSE — chạy trong fragment nên không bị cancel
    st.session_state.is_streaming = True
    full_response = ""
    start_time = time.time()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            for chunk in api_client.chat_stream(prompt, conv["thread_id"]):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
        except Exception as e:
            _logger.warning("chat_stream error: %s", e)
            full_response = full_response or f"Xin lỗi, đã có lỗi xảy ra: {e}"

        placeholder.markdown(full_response)
        elapsed = time.time() - start_time
        st.caption(f"⏱ {elapsed:.1f}s")

    st.session_state.is_streaming = False

    conv["messages"].append({"role": "assistant", "content": full_response})
    conv["updated_at"] = now_ict()

    _persist_conversation(conv)


def _chat_fragment() -> None:
    """Render chat area. KHÔNG decorate `@st.fragment` — fragment wrapper
    phá CSS pin-to-bottom của `st.chat_input` (Streamlit issue #11502).
    Sidebar vẫn là fragment → widget sidebar chỉ rerun fragment, không cancel
    stream trong main body.
    """
    # 1. Consume pending prompt TRƯỚC → mark welcome_dismissed để check bên dưới
    #    đánh giá đúng (nếu check welcome trước thì turn đầu sẽ flash welcome).
    prompt = st.session_state.pending_suggestion or st.session_state.pending_prompt
    st.session_state.pending_suggestion = None
    st.session_state.pending_prompt = None

    if prompt:
        conv["welcome_dismissed"] = True

    # 2. Welcome screen (chỉ hiện khi conversation mới tinh, chưa có prompt pending)
    if should_show_welcome(conv):
        render_welcome_message()

    # 3. Lịch sử tin nhắn
    for msg in conv["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 4. Xử lý prompt mới (append user msg + stream response)
    if prompt:
        _handle_new_message(conv, prompt)


# ── Render chat + info panel ──────────────────────────────────────────

with chat_col:
    _chat_fragment()

if info_col:
    with info_col:
        render_info_panel()
