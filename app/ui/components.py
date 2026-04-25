"""Streamlit UI components — sidebar, right info panel, weather, data refresh.

Gọi FastAPI backend (app/ui/api_client.py). Sau R15 gỡ Celery/Redis: data
refresh chạy sync với st.spinner (block UI 30s-5 phút tùy include_history),
không poll task nữa.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta

import streamlit as st

from app.ui import api_client
from app.ui.utils import (
    get_weather_description_vi,
    get_weather_emoji,
    get_wind_direction,
)

_logger = logging.getLogger(__name__)


# ── Session state management ──────────────────────────────────────────


def init_session_state() -> None:
    """Init tất cả session state keys. Load conversations từ DB qua API."""
    if "conversations" not in st.session_state:
        st.session_state.conversations = _load_conversations_from_api()

    if "active_id" not in st.session_state:
        convs = st.session_state.conversations
        if convs:
            most_recent = max(convs, key=lambda k: convs[k]["updated_at"])
            st.session_state.active_id = most_recent
        else:
            create_new_conversation()

    if "location" not in st.session_state:
        st.session_state.location = None

    if "pending_suggestion" not in st.session_state:
        st.session_state.pending_suggestion = None

    if "show_info_panel" not in st.session_state:
        st.session_state.show_info_panel = True

    if "is_streaming" not in st.session_state:
        st.session_state.is_streaming = False


def _load_conversations_from_api() -> dict:
    """Fetch danh sách hội thoại qua FastAPI. Fallback: empty dict nếu API down."""
    import requests

    try:
        r = requests.get(f"{api_client.API_URL}/conversations", timeout=5)
        r.raise_for_status()
        summaries = r.json()
    except Exception as e:
        _logger.warning("Could not load conversations from API: %s", e)
        return {}

    # Lấy detail cho từng conv (messages) — thesis scope thường <20 conv nên OK
    result: dict = {}
    for s in summaries:
        try:
            d = requests.get(
                f"{api_client.API_URL}/conversations/{s['conv_id']}",
                timeout=5,
            ).json()
            result[s["conv_id"]] = {
                "title": d["title"],
                "messages": d["messages"],
                "thread_id": d["thread_id"],
                "created_at": _parse_iso(d["created_at"]),
                "updated_at": _parse_iso(d["updated_at"]),
                "welcome_dismissed": len(d["messages"]) > 0,
            }
        except Exception as e:
            _logger.warning("Could not load conv %s: %s", s["conv_id"], e)
    return result


def _parse_iso(s: str | datetime) -> datetime:
    """Parse ISO string sang datetime. Tolerate datetime object có sẵn."""
    if isinstance(s, datetime):
        return s
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return datetime.now()


def create_new_conversation() -> str:
    """Tạo hội thoại mới + set active. Persist qua DB."""
    from app.dal.timezone_utils import now_ict

    conv_id = str(uuid.uuid4())
    now = now_ict()
    thread_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {
        "title": "Trò chuyện mới",
        "messages": [],
        "thread_id": thread_id,
        "created_at": now,
        "updated_at": now,
        "welcome_dismissed": False,
    }
    st.session_state.active_id = conv_id

    # Persist trực tiếp (DAL vẫn dùng được — Streamlit container có DB access)
    try:
        from app.db.conversation_dal import save_conversation
        save_conversation(conv_id, thread_id, "Trò chuyện mới", [], now, now)
    except Exception:
        _logger.warning("Could not persist new conversation %s to DB", conv_id)

    return conv_id


def delete_conversation(conv_id: str) -> None:
    """Xoá 1 hội thoại. Nếu đang active, chuyển sang hội thoại mới nhất."""
    convs = st.session_state.conversations
    if conv_id in convs:
        del convs[conv_id]

    try:
        from app.db.conversation_dal import delete_conversation_db
        delete_conversation_db(conv_id)
    except Exception:
        _logger.warning("Could not delete conversation %s from DB", conv_id)

    if st.session_state.active_id == conv_id:
        if convs:
            most_recent = max(convs, key=lambda k: convs[k]["updated_at"])
            st.session_state.active_id = most_recent
        else:
            create_new_conversation()


def get_active_conversation() -> dict:
    """Lấy hội thoại đang active."""
    convs = st.session_state.conversations
    active_id = st.session_state.active_id

    if active_id not in convs:
        active_id = create_new_conversation()

    return convs[active_id]


# ── Left sidebar ──────────────────────────────────────────────────────


def render_sidebar() -> None:
    """Render sidebar. Bọc @st.fragment để toggle/click không rerun toàn app."""
    with st.sidebar:
        _sidebar_fragment()


@st.fragment
def _sidebar_fragment() -> None:
    """Nội dung sidebar. Tất cả tương tác ở đây cô lập trong fragment."""
    _render_header()
    _render_new_chat_button()
    st.divider()
    _render_conversation_list()
    st.divider()
    _render_data_refresh_section()
    st.divider()
    _render_options_section()


def _render_header() -> None:
    """Logo + tên app."""
    st.markdown(
        """
        <div class="sidebar-header">
            <div class="sidebar-header-icon">🌤️</div>
            <div>
                <div class="sidebar-header-text">HanoiWeather</div>
                <div class="sidebar-header-sub">Chatbot Thời Tiết</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_new_chat_button() -> None:
    """Nút tạo hội thoại mới."""
    if st.button("＋  Trò chuyện mới", use_container_width=True, type="primary"):
        create_new_conversation()
        st.rerun(scope="app")


def _render_conversation_list() -> None:
    """List hội thoại theo ngày."""
    convs = st.session_state.conversations
    if not convs:
        return

    today = datetime.now().date()
    yesterday = today - timedelta(days=1)

    groups: dict[str, list[tuple[str, dict]]] = {
        "Hôm nay": [],
        "Hôm qua": [],
        "Trước đó": [],
    }

    sorted_convs = sorted(
        convs.items(),
        key=lambda x: x[1]["updated_at"],
        reverse=True,
    )

    for conv_id, conv in sorted_convs:
        d = conv["created_at"].date() if isinstance(conv["created_at"], datetime) else today
        if d == today:
            groups["Hôm nay"].append((conv_id, conv))
        elif d == yesterday:
            groups["Hôm qua"].append((conv_id, conv))
        else:
            groups["Trước đó"].append((conv_id, conv))

    for group_name, items in groups.items():
        if not items:
            continue

        st.caption(group_name)
        for conv_id, conv in items:
            is_active = conv_id == st.session_state.active_id
            col1, col2 = st.columns([6, 1])

            with col1:
                label = conv["title"]
                if st.button(
                    f"💬 {label}" if is_active else label,
                    key=f"conv_{conv_id}",
                    use_container_width=True,
                    type="secondary",
                ):
                    st.session_state.active_id = conv_id
                    st.rerun(scope="app")

            with col2:
                if st.button("🗑", key=f"del_{conv_id}", help="Xóa hội thoại"):
                    was_active = conv_id == st.session_state.active_id
                    delete_conversation(conv_id)
                    if was_active:
                        st.rerun(scope="app")


# ── Data refresh section ─────────────────────────────────────────────


def _render_data_refresh_section() -> None:
    """Nút cập nhật dữ liệu — chạy sync qua /jobs/ingest (block UI).

    Sau R15 gỡ Celery, ingest chạy đồng bộ. UI hiện `st.spinner` trong lúc
    chờ (~30s-60s cho current/forecast, 2-5 phút cho include_history=True).
    Alternative cho ingest định kỳ: cron external chạy
    `python -m app.scripts.ingest_openweather_async`.
    """
    st.markdown("##### 🔄 Cập nhật dữ liệu")

    include_history = st.checkbox("Bao gồm dữ liệu lịch sử", value=False)

    history_days = 7
    if include_history:
        history_days = st.select_slider(
            "Số ngày lịch sử",
            options=[3, 7, 14],
            value=7,
        )

    spinner_msg = (
        f"Đang ingest current + forecast + history {history_days} ngày... (~2-5 phút)"
        if include_history
        else "Đang ingest current + forecast... (~30-60 giây)"
    )

    if st.button("Cập nhật ngay", use_container_width=True):
        with st.spinner(spinner_msg):
            try:
                api_client.run_ingest(
                    include_history=include_history,
                    history_days=history_days,
                )
                st.success("Cập nhật dữ liệu thành công!")
            except Exception as e:
                st.error(f"Cập nhật thất bại: {e}")


# ── Options section (toggle info panel — thay cho nút floating ▶/◀) ───


def _render_options_section() -> None:
    """Section tuỳ chọn UI. Toggle panel thời tiết.

    Không dùng on_change callback vì Streamlit block st.rerun() trong callback.
    Thay vào đó: check diff trong fragment body rồi st.rerun(scope=\"app\") để
    main body re-evaluate column layout.
    """
    st.markdown("##### ⚙️ Tuỳ chọn")

    # Init widget state 1 lần, sync từ show_info_panel
    if "chk_info_panel" not in st.session_state:
        st.session_state.chk_info_panel = st.session_state.show_info_panel

    # Không truyền value= cùng với key= (sẽ conflict với session_state)
    new_value = st.checkbox(
        "Hiện panel thời tiết",
        key="chk_info_panel",
        help="Ẩn/hiện panel bên phải (địa điểm, thời tiết, biểu đồ)",
    )

    if new_value != st.session_state.show_info_panel:
        st.session_state.show_info_panel = new_value
        # Full-app rerun: main body (ngoài fragment) re-evaluate layout
        st.rerun(scope="app")

    _render_backend_status_badge()


def _render_backend_status_badge() -> None:
    """Badge nhỏ hiển thị trạng thái backend (Postgres/Router/LLM)."""
    if "backend_status" not in st.session_state:
        st.session_state.backend_status = api_client.get_ready_status()

    status = st.session_state.backend_status
    all_ok = all(v in ("ok", "disabled") for v in status.values())

    if all_ok:
        st.caption("🟢 Backend: tất cả sẵn sàng")
    else:
        issues = [k for k, v in status.items() if v not in ("ok", "disabled")]
        st.caption(f"🟡 Backend issues: {', '.join(issues)}")


# ── Right info panel ──────────────────────────────────────────────────


def render_info_panel() -> None:
    """Render right info panel: location selector + weather card + forecast chart.

    Không bọc fragment vì fragment lồng trong st.columns có thể gây lệch layout
    (nội dung nhảy xuống dưới thay vì render đúng cột). Chat stream đã được cô
    lập bằng fragment riêng trong app.py nên info panel không cần cô lập thêm.
    """
    _render_location_selector()
    _render_weather_card()
    _render_temperature_chart()


def _render_location_selector() -> None:
    """Dropdown chọn quận/phường."""
    st.markdown(
        '<div class="info-panel-header">📍 Địa điểm</div>',
        unsafe_allow_html=True,
    )

    district_names = api_client.get_districts()
    if not district_names:
        st.caption("Chưa có dữ liệu quận/huyện")
        return

    selected_district = st.selectbox(
        "Quận/Huyện",
        district_names,
        index=0,
        label_visibility="collapsed",
        key="info_district",
    )

    ward_names = api_client.get_wards(selected_district)
    if ward_names:
        selected_ward = st.selectbox(
            "Phường/Xã",
            list(ward_names.keys()),
            label_visibility="collapsed",
            key="info_ward",
        )
        if selected_ward:
            st.session_state.location = ward_names[selected_ward]


def _render_weather_card() -> None:
    """Card hiển thị thời tiết hiện tại."""
    ward_id = st.session_state.get("location") or "ID_00364"
    weather = api_client.get_current_weather(ward_id)
    if not weather:
        return

    emoji = get_weather_emoji(weather.get("weather_main"))
    desc = get_weather_description_vi(weather.get("weather_main"))
    temp = weather.get("temp", "--")
    humidity = weather.get("humidity", "--")
    wind_speed = weather.get("wind_speed")
    wind_deg = weather.get("wind_deg")
    wind_dir = get_wind_direction(wind_deg) if wind_deg is not None else "--"
    wind_str = f"{wind_speed:.1f} m/s" if wind_speed else "--"

    st.markdown(
        f"""
        <div class="weather-card">
            <div class="weather-card-top">
                <div class="weather-card-temp">{temp}°C</div>
                <div class="weather-card-emoji">{emoji}</div>
            </div>
            <div class="weather-card-desc">{desc}</div>
            <div class="weather-card-grid">
                <div class="weather-card-item">💧 {humidity}%</div>
                <div class="weather-card-item">💨 {wind_str}</div>
                <div class="weather-card-item">🧭 {wind_dir}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_temperature_chart() -> None:
    """Biểu đồ Plotly nhiệt độ 24h."""
    st.markdown(
        '<div class="info-panel-header">📈 Dự báo 24h</div>',
        unsafe_allow_html=True,
    )

    ward_id = st.session_state.get("location") or "ID_00364"
    data = api_client.get_hourly_forecast(ward_id, hours=24)
    if not data:
        st.caption("Chưa có dữ liệu dự báo")
        return

    import plotly.graph_objects as go

    times = [_parse_iso(d["time_local"]) for d in data]
    temps = [d.get("temp") for d in data]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=temps,
            mode="lines+markers",
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=4),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.1)",
            hovertemplate="%{x|%H:%M}<br>%{y:.1f}°C<extra></extra>",
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(tickformat="%H:%M", dtick=3 * 3600 * 1000),
        yaxis=dict(title="°C", title_font_size=12),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key="temp_chart")


# ── Welcome message ───────────────────────────────────────────────────


def render_welcome_message() -> None:
    """Welcome screen — chỉ hiện khi conversation mới tinh, chưa tương tác."""
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-icon">🌤️</div>
            <div class="welcome-title">Xin chào! Tôi là trợ lý thời tiết Hà Nội</div>
            <div class="welcome-subtitle">
                Hỏi tôi bất cứ điều gì về thời tiết — dự báo, so sánh,
                gợi ý trang phục, cảnh báo, và nhiều hơn nữa.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestions = [
        "Thời tiết hôm nay thế nào?",
        "Ngày mai có mưa không?",
        "Nên mặc gì hôm nay?",
        "So sánh Cầu Giấy và Hoàn Kiếm",
    ]

    cols = st.columns(len(suggestions))
    for i, sug in enumerate(suggestions):
        with cols[i]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_suggestion = sug
                # Dismiss welcome ngay trong conversation hiện tại
                conv = get_active_conversation()
                conv["welcome_dismissed"] = True
                st.rerun()


def should_show_welcome(conv: dict) -> bool:
    """Kiểm tra có nên hiển thị welcome screen không.

    Ẩn khi: đã có message, đã dismiss, có pending suggestion, hoặc đang stream.
    """
    if conv.get("welcome_dismissed"):
        return False
    if len(conv.get("messages", [])) > 0:
        return False
    if st.session_state.get("pending_suggestion"):
        return False
    if st.session_state.get("is_streaming"):
        return False
    return True
