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
        "ward":     [get_current_weather, detect_phenomena],
    },

    # --- HOURLY FORECAST ---
    # "Chieu nay mua khong?", "Toi nay may do?", "Luc nao nang?"
    "hourly_forecast": {
        "city":     [get_hourly_forecast, get_sunny_periods],
        "district": [get_hourly_forecast, get_sunny_periods],
        "ward":     [get_hourly_forecast, get_sunny_periods],
    },

    # --- DAILY FORECAST ---
    # "Ngay mai the nao?", "Cuoi tuan troi dep khong?", "Tuan sau?"
    "daily_forecast": {
        "city":     [get_daily_forecast, get_daily_summary, get_weather_period, get_temperature_trend],
        "district": [get_daily_forecast, get_daily_summary, get_weather_period, get_temperature_trend],
        "ward":     [get_daily_forecast, get_daily_summary, get_weather_period, get_temperature_trend],
    },

    # --- WEATHER OVERVIEW ---
    # "Tong hop thoi tiet hom nay?", "Overview thoi tiet?"
    "weather_overview": {
        "city":     [get_daily_summary, detect_phenomena, get_daily_rhythm],
        "district": [get_daily_summary, detect_phenomena, get_daily_rhythm],
        "ward":     [get_daily_summary, detect_phenomena, get_daily_rhythm],
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
        "ward":     [get_current_weather, get_temperature_trend],
    },

    # --- WIND QUERY ---
    # "Gio manh khong?", "Toc do gio?"
    "wind_query": {
        "city":     [get_current_weather, get_pressure_trend],
        "district": [get_current_weather, get_pressure_trend],
        "ward":     [get_current_weather, get_pressure_trend],
    },

    # --- HUMIDITY / FOG QUERY ---
    # "Do am?", "Co nom am khong?", "Suong mu?"
    "humidity_fog_query": {
        "city":     [get_current_weather, get_humidity_timeline, detect_phenomena],
        "district": [get_current_weather, get_humidity_timeline, detect_phenomena],
        "ward":     [get_current_weather, get_humidity_timeline, detect_phenomena],
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
        "district": [compare_weather],
        "ward":     [compare_weather],
    },

    # --- ACTIVITY WEATHER ---
    # "Di choi duoc khong?", "May gio chay bo tot?", "Mac gi hom nay?"
    # get_clothing_advice ADD to primary: "mặc gì hôm nay" vẫn rơi vào
    # activity_weather qua router → cần tool clothing trong focused set.
    "activity_weather": {
        "city":     [get_activity_advice, get_best_time, get_uv_safe_windows, get_clothing_advice],
        "district": [get_activity_advice, get_best_time, get_uv_safe_windows, get_clothing_advice],
        "ward":     [get_activity_advice, get_best_time, get_uv_safe_windows, get_clothing_advice],
    },

    # --- EXPERT WEATHER PARAM ---
    # "Diem suong?", "Ap suat?", "UV index?"
    "expert_weather_param": {
        "city":     [get_current_weather, get_comfort_index, get_pressure_trend],
        "district": [get_current_weather, get_comfort_index, get_pressure_trend],
        "ward":     [get_current_weather, get_comfort_index, get_pressure_trend],
    },

    # --- WEATHER ALERT ---
    # "Co canh bao gi khong?", "Thoi tiet nguy hiem?", "Mua rao/giong khong?"
    # Add get_rain_timeline + get_hourly_forecast: theo intent_disambiguation_rules,
    # "mưa rào/giông" route weather_alert (rule an toàn). Agent cần data mưa cụ thể
    # để answer comprehensive ("có giông + có mưa lúc X" thay vì "không có cảnh báo").
    "weather_alert": {
        "city":     [get_weather_alerts, get_weather_change_alert, get_pressure_trend,
                     get_rain_timeline, get_hourly_forecast],
        "district": [get_weather_alerts, get_weather_change_alert, get_pressure_trend,
                     get_rain_timeline, get_hourly_forecast],
        "ward":     [get_weather_alerts, get_weather_change_alert, get_pressure_trend,
                     get_rain_timeline, get_hourly_forecast],
    },

    # --- SEASONAL CONTEXT ---
    # "Nong hon binh thuong khong?", "So voi mua nay?"
    "seasonal_context": {
        "city":     [get_seasonal_comparison, compare_with_yesterday],
        "district": [get_seasonal_comparison, compare_with_yesterday],
        "ward":     [get_seasonal_comparison, compare_with_yesterday],
    },

    # --- SMALLTALK ---
    # "Xin chao", "Mac gi hom nay?", "Cam on"
    "smalltalk_weather": {
        "city":     [get_current_weather, get_clothing_advice],
        "district": [get_current_weather, get_clothing_advice],
        "ward":     [get_current_weather, get_clothing_advice],
    },
}



# ── EXPANDED_TOOL_MAP ──
# Used when confidence < per-intent threshold (medium confidence zone: 0.45 - threshold).
# Includes PRIMARY tools + 2-3 related tools to cover ambiguous cases.
# Replaces the old 25-tool fallback for medium-confidence routing.

EXPANDED_TOOL_MAP: dict[str, dict[str, list]] = {

    "current_weather": {
        "city":     [get_current_weather, detect_phenomena, get_daily_rhythm, get_comfort_index],
        "district": [get_current_weather, detect_phenomena, get_daily_rhythm, get_comfort_index],
        "ward":     [get_current_weather, detect_phenomena, get_daily_rhythm, get_comfort_index],
    },

    "hourly_forecast": {
        "city":     [get_hourly_forecast, get_sunny_periods, get_rain_timeline, get_daily_rhythm],
        "district": [get_hourly_forecast, get_sunny_periods, get_rain_timeline, get_daily_rhythm],
        "ward":     [get_hourly_forecast, get_sunny_periods, get_rain_timeline, get_daily_rhythm],
    },

    "daily_forecast": {
        "city":     [get_daily_forecast, get_daily_summary, get_weather_period, get_temperature_trend, get_rain_timeline, get_sunny_periods],
        "district": [get_daily_forecast, get_daily_summary, get_weather_period, get_temperature_trend, get_rain_timeline, get_sunny_periods],
        "ward":     [get_daily_forecast, get_daily_summary, get_weather_period, get_temperature_trend, get_rain_timeline],
    },

    "weather_overview": {
        "city":     [get_daily_summary, detect_phenomena, get_daily_rhythm, get_current_weather, get_temperature_trend],
        "district": [get_daily_summary, detect_phenomena, get_daily_rhythm, get_current_weather, get_temperature_trend],
        "ward":     [get_daily_summary, detect_phenomena, get_daily_rhythm, get_current_weather],
    },

    "rain_query": {
        "city":     [get_rain_timeline, get_hourly_forecast, get_daily_forecast, get_weather_alerts, get_weather_change_alert],
        "district": [get_rain_timeline, get_hourly_forecast, get_daily_forecast, get_weather_alerts, get_weather_change_alert],
        "ward":     [get_rain_timeline, get_hourly_forecast, get_weather_alerts],
    },

    "temperature_query": {
        "city":     [get_current_weather, get_temperature_trend, get_hourly_forecast, get_daily_forecast, get_comfort_index],
        "district": [get_current_weather, get_temperature_trend, get_hourly_forecast, get_daily_forecast, get_comfort_index],
        "ward":     [get_current_weather, get_temperature_trend, get_hourly_forecast, get_comfort_index],
    },

    "wind_query": {
        "city":     [get_current_weather, get_pressure_trend, get_hourly_forecast, get_weather_alerts],
        "district": [get_current_weather, get_pressure_trend, get_hourly_forecast, get_weather_alerts],
        "ward":     [get_current_weather, get_pressure_trend, get_hourly_forecast],
    },

    "humidity_fog_query": {
        "city":     [get_current_weather, get_humidity_timeline, detect_phenomena, get_daily_rhythm],
        "district": [get_current_weather, get_humidity_timeline, detect_phenomena, get_daily_rhythm],
        "ward":     [get_current_weather, get_humidity_timeline, detect_phenomena],
    },

    "historical_weather": {
        "city":     [get_weather_history, get_daily_summary, get_weather_period, compare_with_yesterday],
        "district": [get_weather_history, get_daily_summary, get_weather_period, compare_with_yesterday],
        "ward":     [get_weather_history, get_daily_summary, get_weather_period],
    },

    "location_comparison": {
        "city":     [get_district_ranking, get_district_multi_compare, get_current_weather],
        "district": [compare_weather, get_ward_ranking_in_district, get_current_weather],
        "ward":     [compare_weather, get_current_weather],
    },

    "activity_weather": {
        "city":     [get_activity_advice, get_best_time, get_uv_safe_windows, get_current_weather, get_comfort_index, get_clothing_advice],
        "district": [get_activity_advice, get_best_time, get_uv_safe_windows, get_current_weather, get_comfort_index, get_clothing_advice],
        "ward":     [get_activity_advice, get_best_time, get_uv_safe_windows, get_comfort_index],
    },

    "expert_weather_param": {
        "city":     [get_current_weather, get_comfort_index, get_pressure_trend, get_weather_history, get_humidity_timeline],
        "district": [get_current_weather, get_comfort_index, get_pressure_trend, get_weather_history, get_humidity_timeline],
        "ward":     [get_current_weather, get_comfort_index, get_pressure_trend, get_humidity_timeline],
    },

    "weather_alert": {
        "city":     [get_weather_alerts, get_weather_change_alert, get_pressure_trend, get_rain_timeline, get_hourly_forecast],
        "district": [get_weather_alerts, get_weather_change_alert, get_pressure_trend, get_rain_timeline, get_hourly_forecast],
        "ward":     [get_weather_alerts, get_weather_change_alert, get_pressure_trend, get_rain_timeline],
    },

    "seasonal_context": {
        "city":     [get_seasonal_comparison, compare_with_yesterday, get_temperature_trend, get_current_weather],
        "district": [get_seasonal_comparison, compare_with_yesterday, get_temperature_trend, get_current_weather],
        "ward":     [get_seasonal_comparison, compare_with_yesterday, get_temperature_trend],
    },

    "smalltalk_weather": {
        "city":     [get_current_weather, get_clothing_advice, get_comfort_index, get_activity_advice],
        "district": [get_current_weather, get_clothing_advice, get_comfort_index, get_activity_advice],
        "ward":     [get_current_weather, get_clothing_advice, get_comfort_index],
    },
}


def get_focused_tools(
    intent: str,
    scope: str,
    confidence: float = 1.0,
    per_intent_thresholds: dict | None = None,
) -> list | None:
    """Get tool list for (intent, scope) pair, respecting confidence level.

    Routing logic:
    - confidence >= intent_threshold  → PRIMARY_TOOL_MAP (1-3 focused tools)
    - confidence >= 0.45              → EXPANDED_TOOL_MAP (4-6 tools, graceful degradation)
    - confidence < 0.45               → None (caller should fallback)

    Args:
        intent: Classified intent string
        scope: "city" | "district" | "ward"
        confidence: Router confidence score (0.0-1.0)
        per_intent_thresholds: Optional dict of per-intent thresholds.
                               Defaults to global CONFIDENCE_THRESHOLD if None.

    Returns:
        List of tool functions, empty list (smalltalk), or None (should fallback).
    """
    # Determine threshold for this intent
    if per_intent_thresholds is not None:
        from app.agent.router.config import CONFIDENCE_THRESHOLD
        threshold = per_intent_thresholds.get(intent, CONFIDENCE_THRESHOLD)
    else:
        from app.agent.router.config import CONFIDENCE_THRESHOLD
        threshold = CONFIDENCE_THRESHOLD

    # Select map based on confidence
    if confidence >= threshold:
        tool_map = PRIMARY_TOOL_MAP
    elif confidence >= 0.45:
        tool_map = EXPANDED_TOOL_MAP
    else:
        return None  # Very low confidence — caller should fallback or return error

    scope_map = tool_map.get(intent)
    if scope_map is None:
        return None

    tools = scope_map.get(scope)
    if tools is None:
        # Fallback to city-level tools if scope not found
        tools = scope_map.get("city")
    return tools


def get_all_tools() -> list:
    """Return the full 27-tool list (kept for baseline evaluation path)."""
    return TOOLS
