"""R11 L1: Contract A/B/C metadata tests.

Contract A — temporal-window tools (11): "ngày cover", "trong phạm vi", "nguồn dữ liệu"
Contract B — snapshot/advice tools (12): "áp dụng cho", "⚠ snapshot"
Contract C — historical/aggregate tools (3): "loại dữ liệu", "⚠ không phải hiện tại"

Mọi key là ADDITIVE, không rename R10 keys (giữ backward compat).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.agent.tools.output_builder import (
    # Contract A
    build_hourly_forecast_output,
    build_daily_forecast_output,
    build_rain_timeline_output,
    build_best_time_output,
    build_weather_period_output,
    build_uv_safe_windows_output,
    build_pressure_trend_output,
    build_daily_rhythm_output,
    build_humidity_timeline_output,
    build_sunny_periods_output,
    build_temperature_trend_output,
    # Contract B
    build_current_output,
    build_compare_output,
    build_seasonal_comparison_output,
    build_weather_alerts_output,
    build_detect_phenomena_output,
    build_comfort_index_output,
    build_weather_change_alert_output,
    build_clothing_advice_output,
    build_activity_advice_output,
    build_district_ranking_output,
    build_ward_ranking_output,
    build_district_multi_compare_output,
    # Contract C
    build_weather_history_output,
    build_compare_with_yesterday_output,
    build_daily_summary_output,
)

ICT = ZoneInfo("Asia/Ho_Chi_Minh")


# ═════════════════════════════════════════════════════════════════════
# Contract A — Temporal-window tools (11)
# ═════════════════════════════════════════════════════════════════════

def _make_hourly_forecasts(n=6, start_hour_offset=0):
    """Generate n synthetic hourly forecast entries starting NOW + offset."""
    now = datetime.now(ICT)
    start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=start_hour_offset)
    return [
        {
            "ts_utc": int((start + timedelta(hours=i)).timestamp()),
            "temp": 28, "humidity": 75, "pop": 0.2,
            "wind_speed": 3, "wind_deg": 180, "clouds": 60,
            "weather_main": "Clouds",
        }
        for i in range(n)
    ]


def test_contract_a_hourly_forecast_has_ngay_cover():
    raw = {
        "level": "city", "resolved_location": {"city_name": "Hà Nội"},
        "forecasts": _make_hourly_forecasts(6),
    }
    out = build_hourly_forecast_output(raw)
    assert "ngày cover" in out
    assert isinstance(out["ngày cover"], list) and len(out["ngày cover"]) >= 1
    assert out["trong phạm vi"] is True
    # R10 keys giữ nguyên
    assert "phạm vi thực tế" in out


def test_contract_a_daily_forecast_has_ngay_cover():
    raw = {
        "level": "city", "resolved_location": {"city_name": "Hà Nội"},
        "forecasts": [
            {"date": "2026-04-22", "weather_main": "Clouds", "temp_max": 30, "temp_min": 22},
            {"date": "2026-04-23", "weather_main": "Rain", "temp_max": 28, "temp_min": 21},
        ],
    }
    out = build_daily_forecast_output(raw)
    assert "ngày cover" in out
    assert len(out["ngày cover"]) == 2
    assert "22/04/2026" in out["ngày cover"][0]
    assert "(Thứ " in out["ngày cover"][0]  # weekday format
    assert out["trong phạm vi"] is True


def test_contract_a_weather_period_has_ngay_cover():
    raw = {
        "level": "city", "resolved_location": {"city_name": "Hà Nội"},
        "days": 2,
        "daily_data": [
            {"date": "2026-04-25", "weather_main": "Rain"},
            {"date": "2026-04-26", "weather_main": "Clouds"},
        ],
    }
    out = build_weather_period_output(raw)
    assert "ngày cover" in out
    assert len(out["ngày cover"]) == 2


def test_contract_a_rain_timeline_has_pham_vi():
    raw = {
        "level": "city", "resolved_location": {"city_name": "Hà Nội"},
        "rain_periods": [],
        "forecasts": _make_hourly_forecasts(6),
    }
    out = build_rain_timeline_output(raw)
    assert "ngày cover" in out or "phạm vi thực tế" in out  # Either front-loaded or R10


def test_contract_a_best_time_has_metadata():
    now = datetime.now(ICT)
    raw = {
        "level": "city", "resolved_location": {"city_name": "Hà Nội"},
        "activity": "chay_bo",
        "best_hours": [
            {"ts_utc": int(now.timestamp()), "time_ict": "06:00 Thứ Ba 21/04",
             "score": 95, "temp": 25, "pop": 0.1, "issues": []},
        ],
        "worst_hours": [],
    }
    out = build_best_time_output(raw)
    # Contract A via extracted dates
    assert "ngày cover" in out


def test_contract_a_uv_safe_windows_has_metadata():
    raw = {
        "level": "city", "resolved_location": {"city_name": "Hà Nội"},
        "peak_uvi": 8.5,
        "safe_windows": [],
        "danger_windows": [],
    }
    out = build_uv_safe_windows_output(raw)
    # Uses snapshot metadata (windows-based)
    assert "áp dụng cho" in out


def test_contract_a_pressure_trend_has_metadata():
    raw = {
        "level": "city", "resolved_location": {"city_name": "Hà Nội"},
        "trend": "stable",
        "total_change": 0.5,
    }
    out = build_pressure_trend_output(raw)
    assert "áp dụng cho" in out


def test_contract_a_daily_rhythm_has_metadata():
    raw = {
        "level": "city", "resolved_location": {"city_name": "Hà Nội"},
        "date": "2026-04-21",
        "rhythm": {"morning": {"avg_temp": 25}},
    }
    out = build_daily_rhythm_output(raw)
    assert "ngày cover" in out


def test_contract_a_humidity_timeline_has_metadata():
    now = datetime.now(ICT)
    raw = {
        "level": "city", "resolved_location": {"city_name": "Hà Nội"},
        "timeline": [
            {"ts_utc": int((now + timedelta(hours=i)).timestamp()), "humidity": 80, "dew_point": 23}
            for i in range(6)
        ],
    }
    out = build_humidity_timeline_output(raw)
    assert "ngày cover" in out


def test_contract_a_sunny_periods_has_metadata():
    raw = {
        "level": "city", "resolved_location": {"city_name": "Hà Nội"},
        "sunny_windows": [],
        "cloudy_windows": [],
    }
    out = build_sunny_periods_output(raw)
    assert "áp dụng cho" in out  # snapshot fallback


def test_contract_a_temperature_trend_has_metadata():
    raw = {
        "trend": "warming",
        "slope_per_day": 1.2,
        "hottest_day": {"date": "2026-04-22"},
        "coldest_day": {"date": "2026-04-21"},
        "daily_summary": [
            {"date": "2026-04-21", "min": 22, "max": 30},
            {"date": "2026-04-22", "min": 23, "max": 32},
        ],
    }
    out = build_temperature_trend_output(raw)
    assert "ngày cover" in out


# ═════════════════════════════════════════════════════════════════════
# Contract B — Snapshot/advice tools (12)
# ═════════════════════════════════════════════════════════════════════

def test_contract_b_current_has_snapshot_flag():
    raw = {
        "level": "city", "resolved_location": {"city_name": "Hà Nội"},
        "temp": 28, "humidity": 75, "wind_speed": 3, "wind_deg": 180,
        "weather_main": "Clouds", "time_ict": "11:00",
        "ts_utc": int(datetime.now(ICT).timestamp()),
    }
    out = build_current_output(raw)
    assert "áp dụng cho" in out
    assert out["⚠ snapshot"] is True
    # R10 keys giữ
    assert "thời điểm" in out
    assert "gợi ý dùng output" in out


def test_contract_b_comfort_index_has_snapshot_flag():
    out = build_comfort_index_output({"score": 75, "label": "Thoải mái"})
    assert "áp dụng cho" in out
    assert out["⚠ snapshot"] is True


def test_contract_b_clothing_advice_has_snapshot_flag():
    out = build_clothing_advice_output({"clothing_items": ["áo thoáng"], "notes": []})
    assert "áp dụng cho" in out
    assert out["⚠ snapshot"] is True
    # R10 key giữ
    assert "⚠ KHÔNG suy diễn" in out


def test_contract_b_activity_advice_has_snapshot_flag():
    out = build_activity_advice_output({"advice": "nên", "reason": "thoáng"})
    assert "áp dụng cho" in out
    assert out["⚠ snapshot"] is True
    assert "⚠ KHÔNG suy diễn" in out


def test_contract_b_weather_alerts_has_snapshot():
    out = build_weather_alerts_output({"alerts": []})
    assert "áp dụng cho" in out
    assert out["⚠ snapshot"] is True


def test_contract_b_district_ranking_has_snapshot():
    out = build_district_ranking_output({
        "metric": "temp", "rankings": [],
    })
    assert "áp dụng cho" in out


def test_contract_b_compare_weather_has_snapshot():
    out = build_compare_output({
        "location1": {"name": "A", "weather": {"temp": 28}},
        "location2": {"name": "B", "weather": {"temp": 27}},
        "differences": {"temp_diff": 1},
    })
    assert "áp dụng cho" in out
    assert out["⚠ snapshot"] is True


def test_contract_b_seasonal_comparison_has_snapshot():
    out = build_seasonal_comparison_output({
        "current": {"temp": 28, "humidity": 75},
        "seasonal_avg": {"temp_avg": 27},
        "month_name": "tháng 4",
    })
    assert "áp dụng cho" in out


# ═════════════════════════════════════════════════════════════════════
# Contract C — Historical/aggregate tools (3)
# ═════════════════════════════════════════════════════════════════════

def test_contract_c_weather_history_has_loai_quakhu():
    raw = {
        "level": "city", "resolved_location": {"city_name": "Hà Nội"},
        "date": "2026-04-20", "weather_main": "Clouds",
        "temp": 26, "humidity": 70,
    }
    out = build_weather_history_output(raw)
    assert out["loại dữ liệu"] == "quá khứ"
    assert out["⚠ không phải hiện tại"] is True


def test_contract_c_daily_summary_has_loai_tonghop():
    raw = {
        "level": "city", "resolved_location": {"city_name": "Hà Nội"},
        "date": "2026-04-21", "weather_main": "Clouds",
        "temp_min": 22.8, "temp_max": 30.4, "humidity": 75,
    }
    out = build_daily_summary_output(raw)
    assert out["loại dữ liệu"] == "tổng hợp cả ngày"
    assert out["⚠ không phải hiện tại"] is True
    # R10 keys giữ nguyên (backward compat)
    assert "chênh nhiệt ngày-đêm" in out
    assert "gợi ý dùng output" in out


def test_contract_c_compare_yesterday_has_loai_sosanh():
    raw = {
        "level": "city", "resolved_location": {"city_name": "Hà Nội"},
        "today": {"date": "2026-04-21", "avg_temp": 28, "rain_total": 0},
        "previous": {"date": "2026-04-20", "avg_temp": 27, "rain_total": 0},
        "temp_diff": 1.0,
    }
    out = build_compare_with_yesterday_output(raw)
    assert "loại dữ liệu" in out
    assert "so sánh past" in out["loại dữ liệu"]
    assert out["⚠ không phải hiện tại"] is True


# ═════════════════════════════════════════════════════════════════════
# Regression: R10 keys vẫn còn trong Contract output (backward compat)
# ═════════════════════════════════════════════════════════════════════

def test_r10_pham_vi_thuc_te_still_emitted():
    """R10 `"phạm vi thực tế"` vẫn phải có trong Contract A output (backward compat 6 test R10)."""
    raw = {
        "level": "city", "resolved_location": {"city_name": "Hà Nội"},
        "forecasts": _make_hourly_forecasts(6),
    }
    out = build_hourly_forecast_output(raw)
    assert "phạm vi thực tế" in out


def test_r10_khung_da_qua_still_emitted():
    """R10 past-frame warning giữ nguyên (trigger khi NOW muộn)."""
    from app.agent.tools.output_builder import _detect_forecast_range_gap
    fake_now = datetime(2026, 4, 20, 22, 14, tzinfo=ICT)
    start = datetime(2026, 4, 20, 23, 0, tzinfo=ICT)
    forecasts = [{"ts_utc": int(start.timestamp() + i * 3600)} for i in range(8)]
    out = _detect_forecast_range_gap(forecasts, now=fake_now)
    assert "⚠ lưu ý khung đã qua" in out
    # R11 keys cũng emit
    assert "ngày cover" in out
    assert out["trong phạm vi"] is True


def test_r10_advice_no_hallucinate_still_present():
    """`⚠ KHÔNG suy diễn` cho advice tools giữ nguyên."""
    for fn, raw in [
        (build_clothing_advice_output, {"clothing_items": [], "notes": []}),
        (build_activity_advice_output, {"advice": "", "reason": ""}),
        (build_comfort_index_output, {"score": 50, "label": "OK"}),
    ]:
        out = fn(raw)
        assert "⚠ KHÔNG suy diễn" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
