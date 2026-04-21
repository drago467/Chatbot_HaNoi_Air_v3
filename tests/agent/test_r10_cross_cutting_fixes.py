"""R10: Cross-cutting fixes cho các pattern còn lại sau audit v8.

Tests:
- P0.1: Past-frame detection output-level (hourly/rain_timeline có "⚠ lưu ý khung đã qua")
- P0.2: POLICY 3.7 + 3.8 có trong BASE_PROMPT
- P1.1: "Cuối tuần" hard-route trong TOOL_RULES (best_time/rain_timeline/activity/hourly)
- P1.2: Arithmetic pre-compute ("chênh nhiệt ngày-đêm" trong daily_summary)
- P2: Cấm-list advice tools (clothing/activity/comfort có "⚠ KHÔNG suy diễn")
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.agent.tools.output_builder import (
    _detect_forecast_range_gap,
    build_activity_advice_output,
    build_clothing_advice_output,
    build_comfort_index_output,
    build_daily_summary_output,
    build_hourly_forecast_output,
)

ICT = ZoneInfo("Asia/Ho_Chi_Minh")


# ── P0.1: Past-frame detection output-level ────────────────────────────────

def test_past_frame_trigger_when_eval_runs_evening():
    """Scenario v8 audit B1: NOW=22:14, data bắt đầu 23:00 → khung đã qua
    (sáng+trưa+chiều) phải được cảnh báo explicit."""
    fake_now = datetime(2026, 4, 20, 22, 14, tzinfo=ICT)
    start = datetime(2026, 4, 20, 23, 0, tzinfo=ICT)
    forecasts = [{"ts_utc": int(start.timestamp() + i * 3600)} for i in range(8)]

    out = _detect_forecast_range_gap(forecasts, now=fake_now)

    assert "phạm vi thực tế" in out
    assert "⚠ lưu ý khung đã qua" in out
    warn = out["⚠ lưu ý khung đã qua"]
    # Phải liệt kê 3 khung đã qua: sáng + trưa + chiều
    assert "sáng nay" in warn
    assert "trưa nay" in warn
    assert "chiều nay" in warn
    # Phải có hướng dẫn "KHÔNG dán nhãn khung hôm nay"
    assert "KHÔNG" in warn


def test_past_frame_no_trigger_morning():
    """NOW=10:00 sáng, data bắt đầu 11:00 → chưa có khung nào đã qua."""
    fake_now = datetime(2026, 4, 20, 10, 0, tzinfo=ICT)
    start = datetime(2026, 4, 20, 11, 0, tzinfo=ICT)
    forecasts = [{"ts_utc": int(start.timestamp() + i * 3600)} for i in range(6)]

    out = _detect_forecast_range_gap(forecasts, now=fake_now)

    assert "phạm vi thực tế" in out
    assert "⚠ lưu ý khung đã qua" not in out  # Sáng chưa qua


def test_past_frame_no_trigger_when_data_covers_past():
    """Data bắt đầu TRƯỚC NOW (cover past) → không warn."""
    fake_now = datetime(2026, 4, 20, 16, 0, tzinfo=ICT)
    start = datetime(2026, 4, 20, 8, 0, tzinfo=ICT)  # Data sáng đã có
    forecasts = [{"ts_utc": int(start.timestamp() + i * 3600)} for i in range(12)]

    out = _detect_forecast_range_gap(forecasts, now=fake_now)
    assert "⚠ lưu ý khung đã qua" not in out


def test_past_frame_mid_afternoon_trigger():
    """NOW=15:00, data bắt đầu 16:00 → sáng + trưa đã qua, chiều chưa hoàn toàn qua."""
    fake_now = datetime(2026, 4, 20, 15, 0, tzinfo=ICT)
    start = datetime(2026, 4, 20, 16, 0, tzinfo=ICT)
    forecasts = [{"ts_utc": int(start.timestamp() + i * 3600)} for i in range(8)]

    out = _detect_forecast_range_gap(forecasts, now=fake_now)
    warn = out.get("⚠ lưu ý khung đã qua", "")
    assert "sáng nay" in warn
    assert "trưa nay" in warn
    # "chiều nay" KHÔNG nằm trong list vì chiều kết thúc 18h, NOW=15h chưa qua hết
    assert "chiều nay" not in warn


def test_hourly_forecast_output_includes_past_frame_detection():
    """build_hourly_forecast_output kết hợp _detect_forecast_range_gap."""
    start = int(datetime.now(ICT).timestamp())
    raw = {
        "level": "city",
        "resolved_location": {"city_name": "Hà Nội"},
        "forecasts": [
            {"ts_utc": start + i * 3600, "temp": 25, "humidity": 80,
             "pop": 0.3, "wind_speed": 3, "wind_deg": 180, "clouds": 60,
             "weather_main": "Clouds"}
            for i in range(6)
        ],
    }
    out = build_hourly_forecast_output(raw)
    assert "phạm vi thực tế" in out  # Luôn có khi forecasts ≥ 1


# ── P0.2: BASE_PROMPT POLICY 3.7 + 3.8 ──────────────────────────────────────

def test_base_prompt_has_past_frame_rule():
    from app.agent.agent import BASE_PROMPT_TEMPLATE, _inject_datetime
    p = _inject_datetime(BASE_PROMPT_TEMPLATE)
    assert "3.7 Past-frame detection" in p
    # Phải mention cả 4 khung
    for frame in ("sáng", "trưa", "chiều", "tối"):
        assert frame in p.lower()
    assert "⚠ lưu ý khung đã qua" in p  # Reference to output key


def test_base_prompt_has_weekday_mismatch_rule():
    from app.agent.agent import BASE_PROMPT_TEMPLATE, _inject_datetime
    p = _inject_datetime(BASE_PROMPT_TEMPLATE)
    assert "3.8 Weekday mismatch check" in p


# ── P1.1: "Cuối tuần" hard-route TOOL_RULES ────────────────────────────────

def test_best_time_tool_rule_blocks_cuoi_tuan_48h():
    from app.agent.agent import TOOL_RULES
    rule = TOOL_RULES["get_best_time"]
    assert "cuối tuần" in rule.lower()
    assert "hours=48" in rule.lower() or "48" in rule
    assert "get_weather_period" in rule


def test_rain_timeline_tool_rule_blocks_cuoi_tuan():
    from app.agent.agent import TOOL_RULES
    rule = TOOL_RULES["get_rain_timeline"]
    assert "cuối tuần" in rule.lower() or "tuần này" in rule.lower()
    assert "get_weather_period" in rule


def test_activity_advice_tool_rule_blocks_cuoi_tuan():
    from app.agent.agent import TOOL_RULES
    rule = TOOL_RULES["get_activity_advice"]
    assert "cuối tuần" in rule.lower()
    assert "get_weather_period" in rule


def test_hourly_forecast_tool_rule_blocks_cuoi_tuan():
    from app.agent.agent import TOOL_RULES
    rule = TOOL_RULES["get_hourly_forecast"]
    assert "cuối tuần" in rule.lower() and "get_weather_period" in rule


# ── P1.2: Arithmetic pre-compute (ID 64) ────────────────────────────────────

def test_daily_summary_arithmetic_precomputed():
    """ID 64: bot tính 30.4-22.8=5.4 sai. Builder phải pre-compute đúng 7.6."""
    raw = {
        "level": "city", "resolved_location": {"city_name": "Hà Nội"},
        "date": "2026-04-21", "weather_main": "Clouds",
        "temp_min": 22.8, "temp_max": 30.4, "humidity": 75,
    }
    out = build_daily_summary_output(raw)
    assert "chênh nhiệt ngày-đêm" in out
    assert "7.6" in out["chênh nhiệt ngày-đêm"]  # Không phải 5.4
    assert "°C" in out["chênh nhiệt ngày-đêm"]
    assert "COPY" in out["chênh nhiệt ngày-đêm"]  # Instructs LLM


def test_daily_summary_arithmetic_various_values():
    for tmin, tmax, expected in [
        (20.0, 25.0, "5.0"),
        (15.5, 28.3, "12.8"),
        (18.0, 18.0, "0.0"),  # Edge: equal
    ]:
        raw = {
            "level": "city", "resolved_location": {"city_name": "Hà Nội"},
            "date": "2026-04-21", "weather_main": "Clouds",
            "temp_min": tmin, "temp_max": tmax,
        }
        out = build_daily_summary_output(raw)
        assert expected in out.get("chênh nhiệt ngày-đêm", ""), (
            f"temp_max={tmax} - temp_min={tmin} nên ra {expected}, "
            f"ra: {out.get('chênh nhiệt ngày-đêm')}"
        )


def test_daily_summary_no_arithmetic_when_temp_missing():
    """Khi thiếu min/max → không có key."""
    raw = {
        "level": "city", "resolved_location": {"city_name": "Hà Nội"},
        "date": "2026-04-21", "weather_main": "Clouds",
        "avg_temp": 25,
    }
    out = build_daily_summary_output(raw)
    assert "chênh nhiệt ngày-đêm" not in out


# ── P2: Cấm-list advice tools ──────────────────────────────────────────────

def test_clothing_advice_has_no_hallucinate_warning():
    out = build_clothing_advice_output({"clothing_items": ["áo thoáng"], "notes": ["mang ô"]})
    assert "⚠ KHÔNG suy diễn" in out
    warn = out["⚠ KHÔNG suy diễn"]
    # Phải liệt kê nhãn hiện tượng hay bị bịa
    assert "mưa phùn" in warn
    assert "sương mù" in warn


def test_activity_advice_has_no_hallucinate_warning():
    out = build_activity_advice_output({"advice": "nên", "reason": "trời thoáng"})
    assert "⚠ KHÔNG suy diễn" in out


def test_comfort_index_has_no_hallucinate_warning():
    out = build_comfort_index_output({"score": 75, "label": "Thoải mái"})
    assert "⚠ KHÔNG suy diễn" in out


# ── Regression: advice tools vẫn có field chính ────────────────────────────

def test_advice_tools_still_have_primary_fields():
    """Đảm bảo thêm ⚠ KHÔNG break các field chính (primary output)."""
    # clothing
    out_c = build_clothing_advice_output({"clothing_items": ["áo"], "notes": ["mang ô"]})
    assert out_c["trang phục đề xuất"] == ["áo"]
    assert out_c["ghi chú"] == ["mang ô"]
    # activity
    out_a = build_activity_advice_output({"advice": "nên", "reason": "thoáng",
                                           "recommendations": ["đi sáng"]})
    assert out_a["khuyến nghị"] == "nên"
    assert out_a["lý do"] == "thoáng"
    assert out_a["gợi ý thêm"] == ["đi sáng"]
    # comfort
    out_co = build_comfort_index_output({"score": 75, "label": "Thoải mái",
                                          "recommendation": "ra ngoài OK",
                                          "breakdown": {"temp": "ideal"}})
    assert out_co["điểm thoải mái"] == "75/100"
    assert out_co["mức độ"] == "Thoải mái"
    assert out_co["phân tích"] == {"temp": "ideal"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
