"""Streamlit UI components — sidebar, right info panel, weather, data refresh."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time as _time
import uuid
from datetime import datetime, timedelta

import streamlit as st

from app.ui.utils import get_weather_emoji, get_wind_direction, get_weather_description_vi

_logger = logging.getLogger(__name__)


# ── Ollama startup helper ─────────────────────────────────────────────


def ensure_ollama_running() -> None:
    """Check if Ollama is reachable; if not, attempt to start it.

    Only runs once per Streamlit session (result cached in session_state).
    On Windows, starts 'ollama serve' as a background process.
    """
    if st.session_state.get("_ollama_checked"):
        return

    from app.agent.router.config import OLLAMA_BASE_URL, USE_SLM_ROUTER

    if not USE_SLM_ROUTER:
        st.session_state._ollama_checked = True
        return

    if _ollama_is_reachable(OLLAMA_BASE_URL):
        _logger.info("Ollama already running at %s", OLLAMA_BASE_URL)
        st.session_state._ollama_checked = True
        return

    st.toast("Đang khởi động Ollama...", icon="🔄")
    _logger.info("Ollama not reachable — attempting to start")

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except FileNotFoundError:
        st.warning("Không tìm thấy Ollama. Hãy cài đặt từ https://ollama.com")
        st.session_state._ollama_checked = True
        return
    except Exception as e:
        _logger.warning("Could not start Ollama: %s", e)
        st.session_state._ollama_checked = True
        return

    for i in range(30):
        _time.sleep(0.5)
        if _ollama_is_reachable(OLLAMA_BASE_URL):
            _logger.info("Ollama started successfully after %.1fs", (i + 1) * 0.5)
            st.toast("Ollama sẵn sàng!", icon="✅")
            st.session_state._ollama_checked = True
            return

    st.warning("Ollama chưa phản hồi sau 15s. Router sẽ fallback sang full agent.")
    _logger.warning("Ollama did not start within 15s")
    st.session_state._ollama_checked = True


def _ollama_is_reachable(base_url: str) -> bool:
    """Quick HTTP check to see if Ollama is responding."""
    import httpx
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


# ── Database helpers (cached) ──────────────────────────────────────────


@st.cache_data(ttl=3600)
def get_districts() -> list:
    """Get all districts from database (cached 1 hour)."""
    from app.db.dal import query
    try:
        districts = query("""
            SELECT DISTINCT district_name_vi
            FROM dim_ward
            ORDER BY district_name_vi
        """)
        return [d["district_name_vi"] for d in districts if d.get("district_name_vi")]
    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_wards_by_district(district: str) -> dict:
    """Get wards for a specific district (cached 1 hour)."""
    from app.db.dal import query
    try:
        wards = query("""
            SELECT ward_id, ward_name_vi
            FROM dim_ward
            WHERE district_name_vi = %s
            ORDER BY ward_name_vi
        """, (district,))
        return {w["ward_name_vi"]: w["ward_id"] for w in wards}
    except Exception:
        return {}


def get_current_weather_summary(ward_id: str | None = None) -> dict | None:
    """Get latest weather data for display."""
    if ward_id is None:
        ward_id = st.session_state.get("location", "ID_00364")
    return _fetch_weather(ward_id)


@st.cache_data(ttl=300)
def _fetch_weather(ward_id: str) -> dict | None:
    """Cached weather query (5 min TTL). Keyed by ward_id."""
    from app.db.dal import query

    try:
        result = query("""
            SELECT temp, humidity, weather_main, wind_speed, wind_deg
            FROM fact_weather_hourly
            WHERE ward_id = %s
            ORDER BY ts_utc DESC
            LIMIT 1
        """, (ward_id,))
        if result:
            return result[0]
    except Exception:
        pass
    return None


@st.cache_data(ttl=300)
def _fetch_hourly_forecast(ward_id: str) -> list[dict]:
    """Fetch next 24h forecast data for Plotly chart."""
    from app.db.dal import query
    try:
        return query("""
            SELECT ts_utc AT TIME ZONE 'Asia/Ho_Chi_Minh' AS time_local,
                   temp, humidity
            FROM fact_weather_hourly
            WHERE ward_id = %s
              AND data_kind = 'forecast'
              AND ts_utc > NOW()
            ORDER BY ts_utc
            LIMIT 24
        """, (ward_id,))
    except Exception:
        return []


# ── Session state management ──────────────────────────────────────────


def init_session_state() -> None:
    """Initialize all session state keys."""
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}

    if "active_id" not in st.session_state:
        create_new_conversation()

    if "location" not in st.session_state:
        st.session_state.location = None

    if "pending_suggestion" not in st.session_state:
        st.session_state.pending_suggestion = None

    if "show_info_panel" not in st.session_state:
        st.session_state.show_info_panel = True

    ensure_ollama_running()


def create_new_conversation() -> str:
    """Create a new conversation and set it as active. Returns the new ID."""
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {
        "title": "Trò chuyện mới",
        "messages": [],
        "thread_id": str(uuid.uuid4()),
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }
    st.session_state.active_id = conv_id
    return conv_id


def delete_conversation(conv_id: str) -> None:
    """Delete a conversation. If it was active, switch to the most recent one."""
    convs = st.session_state.conversations
    if conv_id in convs:
        del convs[conv_id]

    if st.session_state.active_id == conv_id:
        if convs:
            most_recent = max(convs, key=lambda k: convs[k]["updated_at"])
            st.session_state.active_id = most_recent
        else:
            create_new_conversation()


def get_active_conversation() -> dict:
    """Get the currently active conversation dict."""
    convs = st.session_state.conversations
    active_id = st.session_state.active_id

    if active_id not in convs:
        active_id = create_new_conversation()

    return convs[active_id]


# ── Left sidebar ──────────────────────────────────────────────────────


def render_sidebar() -> None:
    """Render the left sidebar (conversations + data refresh)."""
    with st.sidebar:
        _sidebar_fragment()


@st.fragment
def _sidebar_fragment() -> None:
    """Sidebar content wrapped in @st.fragment for isolated reruns."""
    _render_header()
    _render_new_chat_button()
    st.divider()
    _render_conversation_list()
    st.divider()
    _render_data_refresh_section()


def _render_header() -> None:
    """App logo and title."""
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-header-icon">🌤️</div>
        <div>
            <div class="sidebar-header-text">HanoiAir</div>
            <div class="sidebar-header-sub">Chatbot Thời Tiết</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_new_chat_button() -> None:
    """Button to create a new conversation."""
    if st.button("＋  Trò chuyện mới", use_container_width=True, type="primary"):
        create_new_conversation()
        st.rerun(scope="app")


def _render_conversation_list() -> None:
    """Conversation list grouped by date."""
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
        d = conv["created_at"].date()
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


# ── Data refresh (in left sidebar) ───────────────────────────────────


def _render_data_refresh_section() -> None:
    """Data refresh button with optional history backfill."""
    st.markdown("##### 🔄 Cập nhật dữ liệu")

    include_history = st.checkbox("Bao gồm dữ liệu lịch sử", value=False)

    history_days = 7
    if include_history:
        history_days = st.select_slider(
            "Số ngày lịch sử",
            options=[3, 7, 14],
            value=7,
        )
        st.caption(f"⏱ Ước tính: ~{_estimate_time(history_days)}s")

    if st.button("Cập nhật ngay", use_container_width=True):
        _run_data_refresh(include_history=include_history, history_days=history_days)


def _estimate_time(history_days: int) -> int:
    """Rough time estimate for data refresh."""
    base = 30
    if history_days:
        base += history_days * 10
    return base


def _run_data_refresh(include_history: bool = False, history_days: int = 7) -> None:
    """Execute data ingestion with progress feedback."""
    from app.scripts.ingest_openweather_async import OpenWeatherAsyncIngestor

    ingestor = OpenWeatherAsyncIngestor()
    total_steps = 3 if include_history else 2
    progress = st.progress(0)
    status = st.empty()

    def _run_async(coro):
        """Run an async coroutine safely (handles existing event loop)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)

    try:
        status.info("🌡️ Đang cập nhật thời tiết hiện tại...")
        _run_async(ingestor.run_nowcast())
        progress.progress(1 / total_steps)

        status.info("📊 Đang cập nhật dự báo...")
        _run_async(ingestor.run_forecast())
        progress.progress(2 / total_steps)

        if include_history:
            status.info(f"📜 Đang cập nhật lịch sử ({history_days} ngày)...")
            _run_async(ingestor.run_history_backfill(days=history_days))
            progress.progress(1.0)

        progress.progress(1.0)
        status.success("Cập nhật dữ liệu thành công!")

        _fetch_weather.clear()
        _fetch_hourly_forecast.clear()

    except Exception as e:
        status.warning(f"Lỗi khi cập nhật: {e}")


# ── Right info panel ──────────────────────────────────────────────────


def render_info_panel() -> None:
    """Render the right info panel (location, weather, chart)."""
    _render_location_selector()
    _render_weather_card()
    _render_temperature_chart()


def _render_location_selector() -> None:
    """District and ward selector dropdowns."""
    st.markdown('<div class="info-panel-header">📍 Địa điểm</div>',
                unsafe_allow_html=True)

    district_names = get_districts()
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

    ward_names = get_wards_by_district(selected_district)
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
    """Compact current weather display."""
    weather = get_current_weather_summary()
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

    st.markdown(f"""
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
    """, unsafe_allow_html=True)


def _render_temperature_chart() -> None:
    """Render Plotly temperature chart (24h forecast)."""
    st.markdown('<div class="info-panel-header">📈 Dự báo 24h</div>',
                unsafe_allow_html=True)

    ward_id = st.session_state.get("location", "ID_00364")
    data = _fetch_hourly_forecast(ward_id)
    if not data:
        st.caption("Chưa có dữ liệu dự báo")
        return

    import plotly.graph_objects as go

    times = [d["time_local"] for d in data]
    temps = [d["temp"] for d in data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=temps,
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.1)",
        hovertemplate="%{x|%H:%M}<br>%{y:.1f}°C<extra></extra>",
    ))
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
    """Welcome screen shown when conversation has no messages."""
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">🌤️</div>
        <div class="welcome-title">Xin chào! Tôi là trợ lý thời tiết Hà Nội</div>
        <div class="welcome-subtitle">
            Hỏi tôi bất cứ điều gì về thời tiết — dự báo, so sánh,
            gợi ý trang phục, cảnh báo, và nhiều hơn nữa.
        </div>
    </div>
    """, unsafe_allow_html=True)

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
                st.rerun()
