"""Golden snapshot tests for flat VN output builders.

Mục tiêu: verify shaping đúng VN keys + combined values + conditional keys
cho các trường hợp critical đã audit ở v4 (Q16, Q53, Q83, Q94, Q104, Q126...).
"""

from __future__ import annotations

import pytest

from app.agent.tools.output_builder import (
    build_best_time_output,
    build_comfort_index_output,
    build_compare_output,
    build_compare_with_yesterday_output,
    build_current_output,
    build_daily_forecast_output,
    build_district_ranking_output,
    build_error_output,
    build_hourly_forecast_output,
    build_uv_safe_windows_output,
    build_weather_history_output,
)


# ── Group 1: current weather ────────────────────────────────────────────────

def _base_current_ward():
    return {
        "temp": 25.7, "feels_like": 27.2, "humidity": 79, "dew_point": 21.8,
        "pressure": 1005.7, "wind_speed": 5.5, "wind_gust": 9.4, "wind_deg": 180,
        "clouds": 73, "uvi": 0.7, "pop": 0.83, "rain_1h": 0.26,
        "weather_main": "Clouds", "ts_utc": 1745064000,
        "resolved_location": {"ward_name_vi": "Cầu Giấy", "district_name_vi": "Cầu Giấy"},
        "level": "ward",
    }


def test_current_ward_has_all_core_keys():
    out = build_current_output(_base_current_ward())
    # Core VN keys must be present
    for key in ("địa điểm", "thời điểm", "thời tiết chung", "nhiệt độ", "độ ẩm",
                "điểm sương", "xác suất mưa", "cường độ mưa hiện tại", "gió",
                "mây", "UV", "áp suất", "tóm tắt"):
        assert key in out, f"Missing key: {key}"


def test_current_Q16_fix_pop_is_xac_suat_mua_not_luong_mua():
    """Q16 bug: 'pop=0.83' was read as 'lượng mưa 83%'.

    Expected fix: key is 'xác suất mưa' with value 'Cao 83%'.
    There MUST NOT be any key containing 'lượng mưa' at the current-weather level
    (only daily totals have 'tổng lượng mưa').
    """
    out = build_current_output(_base_current_ward())
    assert out["xác suất mưa"] == "Cao 83%"
    # Không có key "lượng mưa %" đánh lừa
    assert "lượng mưa" not in out


def test_current_Q53_Q104_rain_rate_label_correct():
    """Q53/Q104: rain_1h value must be labeled 'Mưa rất nhẹ' with mm/h unit."""
    raw = _base_current_ward()
    raw["rain_1h"] = 0.15
    out = build_current_output(raw)
    assert "Mưa rất nhẹ 0.15 mm/h" in out["cường độ mưa hiện tại"]


def test_current_Q94_wind_text_distinguishes_avg_and_gust():
    """Q94: avg wind vs gust must be labeled separately in gió value."""
    out = build_current_output(_base_current_ward())
    # Must mention both avg (cấp 4, 5.5 m/s) and gust (9.4 m/s)
    assert "5.5 m/s" in out["gió"]
    assert "giật 9.4 m/s" in out["gió"]


def test_current_Q79_wind_direction_vi():
    """Q79: wind_deg=180 must map to 'hướng Nam', not 'Đông Bắc'."""
    out = build_current_output(_base_current_ward())
    assert "hướng Nam" in out["gió"]


def test_current_district_has_no_feels_like():
    """District aggregate doesn't have feels_like → key KHÔNG được xuất hiện."""
    raw = _base_current_ward()
    raw.pop("feels_like")
    raw["level"] = "district"
    raw["resolved_location"] = {"district_name_vi": "Cầu Giấy"}
    out = build_current_output(raw)
    assert "cảm giác" not in out


def test_current_Q126_no_wind_chill_key_at_warm_temp():
    """Q126: bot bịa wind_chill ở 26°C. Key 'cảm giác lạnh' KHÔNG được xuất hiện khi temp>10."""
    out = build_current_output(_base_current_ward())
    assert "cảm giác lạnh" not in out


def test_current_heat_index_triggers_when_hot_humid():
    raw = _base_current_ward()
    raw["temp"] = 32.0
    raw["humidity"] = 75
    out = build_current_output(raw)
    assert "cảm giác nóng" in out


def test_current_visibility_always_shown():
    """B.1: Tầm nhìn key luôn hiện khi có data, để LLM copy thay vì suy diễn.

    Trước R15 chỉ show <5km, dẫn tới v12 ID 12 (sân bay Nội Bài) bot suy diễn
    tầm nhìn từ độ ẩm/mây vì không thấy field. Giờ luôn show với 4 nhãn:
    Kém / Hạn chế / Trung bình / Tốt.
    """
    raw = _base_current_ward()
    raw["visibility"] = 10000
    out = build_current_output(raw)
    assert "tầm nhìn" in out
    assert "Tốt" in out["tầm nhìn"]
    raw["visibility"] = 800
    out = build_current_output(raw)
    assert "tầm nhìn" in out
    assert "Kém" in out["tầm nhìn"]
    raw["visibility"] = 4000
    out = build_current_output(raw)
    assert "Trung bình" in out["tầm nhìn"]


def test_current_no_bịa_fields():
    """Builder KHÔNG tạo key cho fields không có trong raw (temp_spread, wind_chill...)."""
    out = build_current_output(_base_current_ward())
    for bogus in ("temp_spread", "wind_chill", "khu vực ngập"):
        assert bogus not in out


# ── Hourly forecast ─────────────────────────────────────────────────────────

def test_hourly_forecast_structure():
    raw = {
        "level": "ward",
        "resolved_location": {"ward_name_vi": "Cầu Giấy", "district_name_vi": "Cầu Giấy"},
        "data_coverage": "24 giờ tới",
        "forecasts": [
            {"time_ict": "16:00 (ICT)", "temp": 25.9, "humidity": 82, "pop": 0.82,
             "rain_1h": 0.2, "wind_speed": 5.6, "wind_deg": 180, "clouds": 70,
             "weather_main": "Rain"},
        ],
    }
    out = build_hourly_forecast_output(raw)
    assert "địa điểm" in out
    assert "dự báo" in out
    assert isinstance(out["dự báo"], list) and len(out["dự báo"]) == 1
    entry = out["dự báo"][0]
    assert entry["xác suất mưa"] == "Cao 82%"
    assert "Mưa rất nhẹ" in entry["cường độ mưa"]


# ── Daily forecast ──────────────────────────────────────────────────────────

def test_daily_forecast_sub_temps_morn_day_eve_night():
    raw = {
        "level": "city",
        "resolved_location": {"city_name": "Hà Nội"},
        "forecasts": [{
            "date": "2026-04-20", "temp_min": 22.1, "temp_max": 29.4, "temp_avg": 25.8,
            "temp_morn": 23.4, "temp_day": 28.1, "temp_eve": 26.5, "temp_night": 24.0,
            "humidity": 65, "pop": 0.65, "rain_total": 3.2, "uvi": 5.3,
            "wind_speed": 4.2, "wind_gust": 7.1, "wind_deg": 135,
            "weather_main": "Rain", "summary": "Có mưa rào vài nơi",
        }],
    }
    out = build_daily_forecast_output(raw)
    entry = out["dự báo"][0]
    assert "Sáng 23.4°C" in entry["nhiệt độ theo ngày"]
    assert "Trưa 28.1°C" in entry["nhiệt độ theo ngày"]
    assert "Tối 24.0°C" in entry["nhiệt độ theo ngày"]
    assert entry["tóm tắt"] == "Có mưa rào vài nơi"


def test_daily_forecast_Q199_rain_total_label():
    """Q199: 18.73mm daily total should be 'Mưa to' per RAIN_TOTAL_THRESHOLDS."""
    raw = {
        "level": "city",
        "resolved_location": {"city_name": "Hà Nội"},
        "forecasts": [{
            "date": "2026-04-20", "temp_min": 22, "temp_max": 29,
            "humidity": 80, "pop": 0.9, "rain_total": 18.73, "weather_main": "Rain",
        }],
    }
    out = build_daily_forecast_output(raw)
    assert "Mưa to" in out["dự báo"][0]["tổng lượng mưa"]


# ── History (wind_gust only edge case) ──────────────────────────────────────

def test_history_ward_only_gust_not_bịa_avg():
    """Q94 root: history DAL ward-level only has wind_gust, no wind_speed.
    Builder phải label 'Giật X m/s', KHÔNG bịa 'TB X m/s'.
    """
    raw = {
        "level": "ward",
        "resolved_location": {"ward_name_vi": "Cầu Giấy"},
        "temp": 28.5, "humidity": 72,
        "wind_gust": 4.2, "wind_deg": 180, "weather_main": "Clouds",
    }
    out = build_weather_history_output(raw, date_hint="2026-04-18")
    assert "Giật 4.2 m/s" in out["gió"]
    # Không được bịa chữ "TB" hay "trung bình" từ wind_gust
    assert "TB" not in out["gió"]


# ── Compare ─────────────────────────────────────────────────────────────────

def test_compare_with_yesterday_structure():
    raw = {
        "level": "city", "location_name": "Hà Nội",
        "today": {"date": "2026-04-19", "temp_avg": 25.7, "rain_total": 3.6,
                  "humidity": 79, "weather_main": "Rain"},
        "previous": {"date": "2026-04-18", "temp_avg": 28.4, "rain_total": 0.3,
                     "humidity": 65, "weather_main": "Clouds"},
        "changes": ["Nhiệt độ giảm 2.7°C so với hôm qua"],
        "temp_diff": -2.7, "rain_diff": 3.3,
    }
    out = build_compare_with_yesterday_output(raw)
    assert "hôm nay" in out and "hôm qua" in out
    assert out["hôm nay"]["ngày"].startswith("19/04/2026")
    assert out["hôm qua"]["ngày"].startswith("18/04/2026")


# ── Error handling ──────────────────────────────────────────────────────────

def test_error_builder_flat_shape():
    err = {"error": "no_data", "message": "Không có dữ liệu", "suggestion": "Thử ngày khác"}
    out = build_error_output(err)
    assert out == {"lỗi": "Không có dữ liệu", "gợi ý": "Thử ngày khác"}


def test_error_propagates_through_group1_builders():
    """Error dict passed in → builders detect and return flat error."""
    err = {"error": "no_data", "message": "X"}
    assert build_current_output(err) == {"lỗi": "X"}
    assert build_hourly_forecast_output(err) == {"lỗi": "X"}
    assert build_daily_forecast_output(err) == {"lỗi": "X"}


# ── Group 2/3/4 sanity ──────────────────────────────────────────────────────

def test_comfort_index_flat_vn():
    raw = {"score": 75, "label": "Thoải mái", "recommendation": "Phù hợp...",
           "breakdown": {"temp": "Lý tưởng"}}
    out = build_comfort_index_output(raw)
    assert out["điểm thoải mái"] == "75/100"
    assert out["mức độ"] == "Thoải mái"


def test_district_ranking_flat_vn_per_entry():
    # DAL get_district_rankings trả với key "district" + "value" + "unit"
    raw = {
        "metric": "nhiet_do", "unit": "C", "order": "cao_nhat",
        "rankings": [
            {"rank": 1, "district": "Cầu Giấy", "value": 27.5, "unit": "C"},
            {"rank": 2, "district": "Hoàn Kiếm", "value": 26.8, "unit": "C"},
        ],
    }
    out = build_district_ranking_output(raw)
    assert out["chỉ số"] == "nhiệt độ"  # VN label
    assert out["thứ tự"] == "cao nhất"
    assert out["xếp hạng"][0]["hạng"] == 1
    assert out["xếp hạng"][0]["quận"] == "Cầu Giấy"
    assert "27.5°C" in out["xếp hạng"][0]["nhiệt độ"]  # unit format


def test_district_ranking_skips_empty_names():
    """ID 105 bug: DAL trả entry với district rỗng → không xuất hiện trong output."""
    raw = {
        "metric": "nhiet_do", "unit": "C", "order": "cao_nhat",
        "rankings": [
            {"rank": 1, "district": "", "value": 27.5, "unit": "C"},
            {"rank": 2, "district": "Cầu Giấy", "value": 26.8, "unit": "C"},
        ],
    }
    out = build_district_ranking_output(raw)
    assert len(out["xếp hạng"]) == 1
    assert out["xếp hạng"][0]["quận"] == "Cầu Giấy"


def test_compare_weather_humidity_diff_float_no_crash():
    """Bug cũ: humidity_diff=-6.0 (float) + `:+d` format → ValueError."""
    from app.agent.tools.output_builder import build_compare_output
    raw = {
        "location1": {"name": "A", "weather": {"temp": 28.5, "humidity": 72, "weather_main": "Clouds"}},
        "location2": {"name": "B", "weather": {"temp": 26.3, "humidity": 78, "weather_main": "Clouds"}},
        "differences": {"temp_diff": 2.2, "humidity_diff": -6.0},
        "comparison_text": "A nóng hơn B 2.2C",
    }
    out = build_compare_output(raw)
    assert out["chênh lệch"]["độ ẩm"] == "-6%"
    assert out["chênh lệch"]["nhiệt độ"] == "+2.2°C"


def test_seasonal_comparison_comparisons_is_list_of_strings():
    """Bug cũ: build_seasonal_comparison gọi .get() trên str element."""
    from app.agent.tools.output_builder import build_seasonal_comparison_output
    raw = {
        "current": {"temp": 28.1, "humidity": 82},
        "seasonal_avg": {"temp_avg": 27, "humidity": 82, "rain_days": 15},
        "comparisons": ["Nóng hơn bình thường 1.1°C", "Độ ẩm bình thường"],
        "month_name": "Tháng 4",
    }
    out = build_seasonal_comparison_output(raw)
    assert out["nhận xét"] == ["Nóng hơn bình thường 1.1°C", "Độ ẩm bình thường"]


def test_daily_forecast_has_superlatives():
    from app.agent.tools.output_builder import build_daily_forecast_output
    raw = {
        "level": "city",
        "resolved_location": {"city_name": "Hà Nội"},
        "forecasts": [
            {"date": "2026-04-20", "temp_min": 22, "temp_max": 29, "rain_total": 5.2, "pop": 0.9, "weather_main": "Rain"},
            {"date": "2026-04-22", "temp_min": 25, "temp_max": 36.7, "rain_total": 2.1, "pop": 0.6, "weather_main": "Rain"},
            {"date": "2026-04-24", "temp_min": 21, "temp_max": 30, "rain_total": 1.0, "pop": 0.49, "weather_main": "Clouds"},
        ],
    }
    out = build_daily_forecast_output(raw)
    assert "tổng hợp" in out
    sup = out["tổng hợp"]
    # 22/04 có max=36.7 → nóng nhất
    assert "22/04" in sup["ngày nóng nhất"]
    assert "36.7" in sup["ngày nóng nhất"]
    # 24/04 có min=21 → mát nhất
    assert "24/04" in sup["ngày mát nhất"]
    # 20/04 có rain=5.2 → mưa nhiều nhất
    assert "20/04" in sup["ngày mưa nhiều nhất"]


def test_current_has_usage_hint():
    """Gợi ý dùng output phải xuất hiện ở current_weather."""
    out = build_current_output(_base_current_ward())
    assert "gợi ý dùng output" in out
    assert "snapshot" in out["gợi ý dùng output"].lower()


def test_daily_summary_has_usage_hint():
    from app.agent.tools.output_builder import build_daily_summary_output
    raw = {
        "level": "city",
        "resolved_location": {"city_name": "Hà Nội"},
        "date": "2026-04-20",
        "weather_main": "Clouds",
        "avg_temp": 26, "humidity": 75,
    }
    out = build_daily_summary_output(raw)
    assert "gợi ý dùng output" in out
    assert "tổng hợp cả ngày" in out["gợi ý dùng output"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
