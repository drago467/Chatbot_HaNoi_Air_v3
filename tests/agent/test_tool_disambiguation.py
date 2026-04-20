"""Test tool docstring disambiguation sau R8.

Mục tiêu: mỗi cặp tool overlap dễ nhầm có "KHÔNG DÙNG KHI" trong docstring,
ngăn LLM "reaching for the hammer".
"""

from __future__ import annotations

import inspect

import pytest

from app.agent.tools import (
    compare as compare_mod,
    core as core_mod,
    forecast as forecast_mod,
    history as history_mod,
    insight as insight_mod,
)


def _docstring(tool_obj) -> str:
    """Lấy docstring của @tool wrapper (đã wrapped bởi langchain tool)."""
    # LangChain @tool wrap fn. Docstring ở fn.description hoặc fn.__doc__
    if hasattr(tool_obj, "description"):
        return tool_obj.description or ""
    if hasattr(tool_obj, "func"):
        return inspect.getdoc(tool_obj.func) or ""
    return inspect.getdoc(tool_obj) or ""


# ── get_current_weather: snapshot-only ──────────────────────────────────────

def test_current_weather_docstring_blocks_future_usage():
    d = _docstring(core_mod.get_current_weather)
    assert "KHÔNG DÙNG" in d, "get_current_weather thiếu 'KHÔNG DÙNG KHI'"
    # Phải mention các khung tương lai cần tránh
    future_markers = ["chiều", "tối", "ngày mai", "sáng mai"]
    mentioned = [w for w in future_markers if w in d.lower()]
    assert len(mentioned) >= 3, (
        f"Docstring thiếu cảnh báo các khung tương lai: mention {mentioned}"
    )


def test_current_weather_docstring_blocks_superlative():
    d = _docstring(core_mod.get_current_weather)
    assert "max" in d.lower() or "đỉnh" in d.lower() or "mạnh nhất" in d.lower(), (
        "Docstring thiếu cảnh báo superlative (max/đỉnh/mạnh nhất cả ngày)"
    )


# ── get_hourly_forecast vs get_daily_forecast ───────────────────────────────

def test_hourly_forecast_blocks_multi_day():
    d = _docstring(forecast_mod.get_hourly_forecast)
    assert "KHÔNG DÙNG" in d


def test_daily_forecast_blocks_snapshot_and_single_day_detail():
    d = _docstring(forecast_mod.get_daily_forecast)
    assert "KHÔNG DÙNG" in d
    # Phân biệt với daily_summary
    assert "get_daily_summary" in d, (
        "daily_forecast docstring phải tham chiếu get_daily_summary cho disambiguation"
    )


# ── get_daily_summary vs daily_forecast + current ──────────────────────────

def test_daily_summary_blocks_snapshot_and_multi_day():
    d = _docstring(history_mod.get_daily_summary)
    assert "KHÔNG DÙNG" in d
    assert "bây giờ" in d.lower() or "tức thời" in d.lower()
    assert "nhiều ngày" in d.lower() or "tuần" in d.lower()


# ── get_weather_history: past-only ──────────────────────────────────────────

def test_weather_history_past_only():
    d = _docstring(history_mod.get_weather_history)
    assert "past" in d.lower() or "đã qua" in d.lower() or "quá khứ" in d.lower()
    assert "KHÔNG DÙNG" in d


# ── compare_with_yesterday past-only, reject future direction ───────────────

def test_compare_with_yesterday_blocks_future_direction():
    d = _docstring(compare_mod.compare_with_yesterday)
    assert "KHÔNG DÙNG" in d
    # Phải mention future direction case + thay bằng cái gì
    assert "ngày mai" in d.lower(), "Phải cảnh báo case 'hôm nay vs ngày mai'"
    assert "get_daily_forecast" in d, (
        "Phải chỉ giải pháp: thay bằng get_daily_forecast"
    )


# ── compare_weather distinct từ other compare ──────────────────────────────

def test_compare_weather_blocks_other_compare_use_cases():
    d = _docstring(compare_mod.compare_weather)
    assert "KHÔNG DÙNG" in d
    # Tham chiếu đến các compare tool khác
    assert "compare_with_yesterday" in d or "seasonal" in d.lower()


# ── get_seasonal_comparison: không dùng cho so 2 thời điểm ─────────────────

def test_seasonal_comparison_blocks_misuse():
    d = _docstring(compare_mod.get_seasonal_comparison)
    assert "KHÔNG DÙNG" in d
    # Phải nói rõ là climatology, không phải so 2 thời điểm
    assert "climatology" in d.lower() or "mùa" in d.lower()


# ── get_rain_timeline: không dùng cho tổng lượng mưa ───────────────────────

def test_rain_timeline_blocks_total_rain_misuse():
    d = _docstring(forecast_mod.get_rain_timeline)
    assert "KHÔNG DÙNG" in d
    # Phân biệt mm/h vs mm/ngày
    assert "mm/h" in d.lower() or "cường độ" in d.lower()
    assert "tổng" in d.lower()  # phải mention "tổng lượng mưa → dùng tool khác"


# ── get_activity_advice: không dùng đơn lẻ khi cần chi tiết ────────────────

def test_activity_advice_requires_combo_for_details():
    d = _docstring(insight_mod.get_activity_advice)
    # Phải nói rõ KHÔNG DÙNG ĐƠN LẺ
    assert "ĐƠN LẺ" in d or "đơn lẻ" in d or "PHẢI gọi kèm" in d.lower() or "combo" in d.lower()
    # Phải reference các tool để gọi kèm
    assert "get_rain_timeline" in d or "get_uv_safe_windows" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
