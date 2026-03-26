"""Tool Mapper — map (intent, scope) → focused tool list (1-2 tools)."""

from __future__ import annotations

from app.agent.tools import (
    get_current_weather,
    get_hourly_forecast,
    get_daily_forecast,
    get_weather_history,
    compare_weather,
    compare_with_yesterday,
    get_activity_advice,
    get_weather_alerts,
    detect_phenomena,
    get_seasonal_comparison,
    get_daily_summary,
    get_district_weather,
    get_city_weather,
    get_district_daily_forecast,
    get_city_daily_forecast,
    get_district_ranking,
    get_ward_ranking_in_district,
    get_rain_timeline,
    get_best_time,
    get_temperature_trend,
    get_comfort_index,
    get_weather_change_alert,
    TOOLS,
)

# ── PRIMARY_TOOL_MAP ──
# Map (intent, scope) → list of tool functions (1-2 tools)
# Dựa trên system prompt rules (agent.py:34-59) + evaluation patterns
PRIMARY_TOOL_MAP: dict[str, dict[str, list]] = {
    "current_weather": {
        "city": [get_city_weather],
        "district": [get_district_weather],
        "ward": [get_current_weather],
    },
    "hourly_forecast": {
        "city": [get_hourly_forecast],
        "district": [get_hourly_forecast],
        "ward": [get_hourly_forecast],
    },
    "daily_forecast": {
        "city": [get_city_daily_forecast],
        "district": [get_district_daily_forecast],
        "ward": [get_daily_forecast],
    },
    "weather_overview": {
        "city": [get_city_weather, detect_phenomena],
        "district": [get_district_weather, detect_phenomena],
        "ward": [get_daily_summary],
    },
    "rain_query": {
        "city": [get_rain_timeline],
        "district": [get_rain_timeline],
        "ward": [get_rain_timeline],
    },
    "temperature_query": {
        "city": [get_city_weather],
        "district": [get_district_weather],
        "ward": [get_current_weather],
    },
    "wind_query": {
        "city": [get_city_weather],
        "district": [get_district_weather],
        "ward": [get_current_weather],
    },
    "humidity_fog_query": {
        "city": [get_city_weather, detect_phenomena],
        "district": [get_district_weather, detect_phenomena],
        "ward": [get_current_weather],
    },
    "historical_weather": {
        "city": [get_weather_history],
        "district": [get_weather_history],
        "ward": [get_weather_history],
    },
    "location_comparison": {
        "city": [get_district_ranking],
        "district": [compare_weather],
        "ward": [compare_weather],
    },
    "activity_weather": {
        "city": [get_activity_advice, get_best_time],
        "district": [get_activity_advice, get_best_time],
        "ward": [get_activity_advice],
    },
    "expert_weather_param": {
        "city": [get_city_weather, get_comfort_index],
        "district": [get_district_weather, get_comfort_index],
        "ward": [get_current_weather, get_comfort_index],
    },
    "weather_alert": {
        "city": [get_weather_alerts, get_weather_change_alert],
        "district": [get_weather_alerts, get_weather_change_alert],
        "ward": [get_weather_alerts],
    },
    "seasonal_context": {
        "city": [get_seasonal_comparison, compare_with_yesterday],
        "district": [get_seasonal_comparison, compare_with_yesterday],
        "ward": [get_seasonal_comparison],
    },
    "smalltalk_weather": {
        "city": [],
        "district": [],
        "ward": [],
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
    """Return the full 25-tool list (for fallback path)."""
    return TOOLS
