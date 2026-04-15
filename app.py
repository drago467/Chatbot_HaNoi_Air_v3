"""
HanoiAir Weather Chatbot — Streamlit UI

Entry point for the chatbot interface. Delegates to app.ui for rendering.
Run with: streamlit run app.py
"""

import operator
import time
from datetime import datetime

import streamlit as st

# ── Patch: langchain-core _dict_int_op doesn't handle None usage values ──
# Some OpenAI-compatible providers return None for usage_metadata fields
# (e.g. prompt_tokens: None), which crashes _dict_int_op during streaming.
# This patch treats None as 0.
import langchain_core.utils.usage as _usage_mod

_original_dict_int_op = _usage_mod._dict_int_op


def _patched_dict_int_op(left, right, op, *, default=0, depth=0, max_depth=100):
    cleaned_left = {k: (0 if v is None else v) for k, v in left.items()}
    cleaned_right = {k: (0 if v is None else v) for k, v in right.items()}
    return _original_dict_int_op(
        cleaned_left, cleaned_right, op,
        default=default, depth=depth, max_depth=max_depth,
    )


_usage_mod._dict_int_op = _patched_dict_int_op
# ── End patch ──────────────────────────────────────────────────────────

from app.core.logging_config import setup_logging
from app.ui.styles import CUSTOM_CSS
from app.ui.components import (
    init_session_state,
    get_active_conversation,
    render_sidebar,
    render_info_panel,
    render_welcome_message,
)

setup_logging()

# ── Page config ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chatbot Thời Tiết Hà Nội",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ──────────────────────────────────────────────────

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Initialize session state ───────────────────────────────────────────

init_session_state()

# ── Left sidebar (conversations + data refresh) ───────────────────────

render_sidebar()

# ── Main area with optional right panel ────────────────────────────────

conv = get_active_conversation()

if st.session_state.show_info_panel:
    chat_col, info_col = st.columns([3, 1.2])
else:
    chat_col = st.container()
    info_col = None

# ── Chat column ────────────────────────────────────────────────────────

with chat_col:
    # Toggle button row
    _, toggle_col = st.columns([10, 1])
    with toggle_col:
        icon = "◀" if st.session_state.show_info_panel else "▶"
        tooltip = "Ẩn thông tin thời tiết" if st.session_state.show_info_panel else "Hiện thông tin thời tiết"
        if st.button(icon, help=tooltip, key="toggle_panel"):
            st.session_state.show_info_panel = not st.session_state.show_info_panel
            st.rerun()

    # Welcome screen when conversation is empty
    if not conv["messages"]:
        render_welcome_message()

    # Display chat history
    for msg in conv["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ── Right info panel ──────────────────────────────────────────────────

if info_col:
    with info_col:
        render_info_panel()

# ── Chat input (page level — spans full width) ────────────────────────

user_input = st.chat_input("Hỏi về thời tiết Hà Nội...")

# Merge: either user typed or clicked a suggestion chip
prompt = st.session_state.pending_suggestion or user_input
if st.session_state.pending_suggestion:
    st.session_state.pending_suggestion = None

if prompt:
    # Add user message
    conv["messages"].append({"role": "user", "content": prompt})

    # Auto-title from first user message
    if sum(1 for m in conv["messages"] if m["role"] == "user") == 1:
        conv["title"] = prompt[:30] + ("..." if len(prompt) > 30 else "")

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream agent response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        start_time = time.time()

        try:
            from app.agent.agent import stream_agent_routed
            for chunk in stream_agent_routed(prompt, thread_id=conv["thread_id"]):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
        except Exception as e:
            full_response = f"Xin lỗi, đã có lỗi xảy ra: {e}"

        # Final render (remove streaming cursor)
        placeholder.markdown(full_response)
        elapsed = time.time() - start_time
        st.caption(f"⏱ {elapsed:.1f}s")

    # Save assistant response
    conv["messages"].append({"role": "assistant", "content": full_response})
    conv["updated_at"] = datetime.now()

    # Telemetry logging
    try:
        from app.agent.telemetry import get_evaluation_logger
        logger = get_evaluation_logger()
        logger.log_conversation(
            session_id=conv["thread_id"],
            turn_number=sum(1 for m in conv["messages"] if m["role"] == "user"),
            user_query=prompt,
            llm_response=full_response,
            response_time_ms=elapsed * 1000,
        )
    except Exception:
        pass
