"""Tool Mapper — map (intent, scope) -> focused tool list (1-3 tools).

Refactored:
- Tat ca tool now deu support 3 tiers (ward/district/city) nhat quan
- Khong can dedicated get_district_weather/get_city_weather nua (merged vao get_current_weather)
- Khong can get_district_daily_forecast/get_city_daily_forecast (merged vao get_daily_forecast)
- Them 6 insight tools moi: uv_safe, pressure_trend, daily_rhythm, humidity_timeline,
  sunny_periods, district_multi_compare
- Moi intent co tool chinh + tool bo sung (optional insight)
"""

from __future__ import annotations

from app.agent.tools import (
    # Core (3)
    resolve_location,
    get_current_weather,
    get_weather_alerts,
    # Forecast (4)
    get_hourly_forecast,
    get_daily_forecast,
    get_rain_timeline,
    get_best_time,
    # History (3)
    get_weather_history,
    get_daily_summary,
    get_weather_period,
    # Compare (3)
    compare_weather,
    compare_with_yesterday,
    get_seasonal_comparison,
    # Ranking (2)
    get_district_ranking,
    get_ward_ranking_in_district,
    # Insight (6)
    detect_phenomena,
    get_temperature_trend,
    get_comfort_index,
    get_weather_change_alert,
    get_clothing_advice,
    get_activity_advice,
    # Insight New (6)
    get_uv_safe_windows,
    get_pressure_trend,
    get_daily_rhythm,
    get_humidity_timeline,
    get_sunny_periods,
    get_district_multi_compare,
    # Full list
    TOOLS,
)


# ── PRIMARY_TOOL_MAP ──
# Map (intent, scope) -> list of tool functions (1-3 tools)
# Thiet ke: tool chinh + optional insight tool de tang chat luong response
#
# KHAC BIET CHINH so voi phien ban cu:
# 1. Tat ca scope (ward/district/city) dung CUNG tool (vi tool tu dispatch)
# 2. Them insight tools moi de bo sung thong tin
# 3. Khong con cac tool dedicated (get_district_weather, get_city_weather, ...)

PRIMARY_TOOL_MAP: dict[str, dict[str, list]] = {

    # --- CURRENT WEATHER ---
    # "Bay gio troi the nao?", "Nhiet do hien tai?"
    "current_weather": {
        "city":     [get_current_weather, detect_phenomena],
        "district": [get_current_weather, detect_phenomena],
        "ward":     [get_current_weather],
    },

    # --- HOURLY FORECAST ---
    # "Chieu nay mua khong?", "Toi nay may do?"
    "hourly_forecast": {
        "city":     [get_hourly_forecast],
        "district": [get_hourly_forecast],
        "ward":     [get_hourly_forecast],
    },

    # --- DAILY FORECAST ---
    # "Ngay mai the nao?", "Cuoi tuan troi dep khong?"
    "daily_forecast": {
        "city":     [get_daily_forecast, get_temperature_trend],
        "district": [get_daily_forecast, get_temperature_trend],
        "ward":     [get_daily_forecast],
    },

    # --- WEATHER OVERVIEW ---
    # "Tong hop thoi tiet hom nay?", "Overview thoi tiet?"
    "weather_overview": {
        "city":     [get_daily_summary, detect_phenomena, get_daily_rhythm],
        "district": [get_daily_summary, detect_phenomena, get_daily_rhythm],
        "ward":     [get_daily_summary, detect_phenomena],
    },

    # --- RAIN QUERY ---
    # "Luc nao mua?", "Mua den bao gio?", "Co mua khong?"
    "rain_query": {
        "city":     [get_rain_timeline],
        "district": [get_rain_timeline],
        "ward":     [get_rain_timeline],
    },

    # --- TEMPERATURE QUERY ---
    # "Nhiet do?", "Nong khong?", "Lanh bao nhieu?"
    "temperature_query": {
        "city":     [get_current_weather, get_temperature_trend],
        "district": [get_current_weather, get_temperature_trend],
        "ward":     [get_current_weather],
    },

    # --- WIND QUERY ---
    # "Gio manh khong?", "Toc do gio?"
    "wind_query": {
        "city":     [get_current_weather, get_pressure_trend],
        "district": [get_current_weather, get_pressure_trend],
        "ward":     [get_current_weather],
    },

    # --- HUMIDITY / FOG QUERY ---
    # "Do am?", "Co nom am khong?", "Suong mu?"
    "humidity_fog_query": {
        "city":     [get_current_weather, get_humidity_timeline, detect_phenomena],
        "district": [get_current_weather, get_humidity_timeline, detect_phenomena],
        "ward":     [get_current_weather, detect_phenomena],
    },

    # --- HISTORICAL WEATHER ---
    # "Hom qua the nao?", "Ngay 15/3 troi ra sao?"
    "historical_weather": {
        "city":     [get_weather_history],
        "district": [get_weather_history],
        "ward":     [get_weather_history],
    },

    # --- LOCATION COMPARISON ---
    # "Cau Giay vs Dong Da?", "Quan nao nong nhat?"
    "location_comparison": {
        "city":     [get_district_ranking, get_district_multi_compare],
        "district": [compare_weather, get_ward_ranking_in_district],
        "ward":     [compare_weather],
    },

    # --- ACTIVITY WEATHER ---
    # "Di choi duoc khong?", "May gio chay bo tot?"
    "activity_weather": {
        "city":     [get_activity_advice, get_best_time, get_uv_safe_windows],
        "district": [get_activity_advice, get_best_time, get_uv_safe_windows],
        "ward":     [get_activity_advice, get_best_time],
    },

    # --- EXPERT WEATHER PARAM ---
    # "Diem suong?", "Ap suat?", "UV index?"
    "expert_weather_param": {
        "city":     [get_current_weather, get_comfort_index, get_pressure_trend],
        "district": [get_current_weather, get_comfort_index, get_pressure_trend],
        "ward":     [get_current_weather, get_comfort_index],
    },

    # --- WEATHER ALERT ---
    # "Co canh bao gi khong?", "Thoi tiet nguy hiem?"
    "weather_alert": {
        "city":     [get_weather_alerts, get_weather_change_alert, get_pressure_trend],
        "district": [get_weather_alerts, get_weather_change_alert],
        "ward":     [get_weather_alerts, get_weather_change_alert],
    },

    # --- SEASONAL CONTEXT ---
    # "Nong hon binh thuong khong?", "So voi mua nay?"
    "seasonal_context": {
        "city":     [get_seasonal_comparison, compare_with_yesterday],
        "district": [get_seasonal_comparison, compare_with_yesterday],
        "ward":     [get_seasonal_comparison],
    },

    # --- SMALLTALK ---
    # "Xin chao", "Cam on"
    "smalltalk_weather": {
        "city":     [],
        "district": [],
        "ward":     [],
    },
}


def get_focused_tools(intent: str, scope: str) -> list | None:
    """Get focused tool list for (intent, scope) pair.

    Returns:
        List of tool functions if mapping exists, None if should fallback.
        Empty list means no tools needed (e.g. smalltalk).
    """
    scope_map = PRIMARY_TOOL_MAP.get(intent)
    if scope_map is None:
        return None
    tools = scope_map.get(scope)
    if tools is None:
        return None
    return tools


def get_all_tools() -> list:
    """Return the full 31-tool list (for fallback path)."""
    return TOOLS
