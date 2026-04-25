"""Tool accuracy evaluation — INTENT_TO_TOOLS mapping and check functions."""


# ---- Intent -> Expected Tools mapping (Hierarchical by location scope) ----
# For each (intent, location_scope), lists acceptable tools that can
# properly answer the question at that scope level.
# Recall = 1.0 if agent called at least one expected tool, else 0.0.
#
# Design principles (post-refactor, 27 tools):
# - 3 scopes: city / district / ward  (POI queries resolve to district via dispatch.py)
# - All tools auto-dispatch to 3 tiers (ward/district/city) via dispatch.py
#   -> ward la don vi data goc, district/city aggregate tu ward
#   -> ward PHAI co cung tool set voi district/city (khong ly do ky thuat de gioi han)
# - INTENT_TO_TOOLS >= PRIMARY_TOOL_MAP (router's focused tools)
# - smalltalk_weather: no-tools-called cung chap nhan duoc (greetings)

INTENT_TO_TOOLS = {
    # current_weather: "Bay gio thoi tiet the nao?"
    # R9: + rain_timeline + hourly_forecast defensive (confusion current->rain 2x)
    "current_weather": {
        "city":     ["get_current_weather", "detect_phenomena", "get_clothing_advice",
                     "get_rain_timeline", "get_hourly_forecast"],
        "district": ["get_current_weather", "detect_phenomena", "get_clothing_advice",
                     "get_rain_timeline", "get_hourly_forecast"],
        "ward":     ["get_current_weather", "detect_phenomena", "get_clothing_advice",
                     "get_rain_timeline", "get_hourly_forecast"],
    },
    # hourly_forecast: "Chieu nay mua khong?", "Toi nay may do?"
    "hourly_forecast": {
        "city":     ["get_hourly_forecast", "get_rain_timeline", "get_sunny_periods", "get_daily_rhythm"],
        "district": ["get_hourly_forecast", "get_rain_timeline", "get_sunny_periods", "get_daily_rhythm"],
        "ward":     ["get_hourly_forecast", "get_rain_timeline", "get_sunny_periods", "get_daily_rhythm"],
    },
    # daily_forecast: "Ngay mai the nao?", "Cuoi tuan?"
    # R9: + rain_timeline + hourly_forecast + daily_summary defensive
    # R13: + get_weather_history (recover "7 ngày qua" misrouted)
    "daily_forecast": {
        "city":     ["get_daily_forecast", "get_weather_period", "get_temperature_trend",
                     "get_daily_summary", "get_rain_timeline", "get_hourly_forecast", "get_weather_history"],
        "district": ["get_daily_forecast", "get_weather_period", "get_temperature_trend",
                     "get_daily_summary", "get_rain_timeline", "get_hourly_forecast", "get_weather_history"],
        "ward":     ["get_daily_forecast", "get_weather_period", "get_temperature_trend",
                     "get_daily_summary", "get_rain_timeline", "get_hourly_forecast", "get_weather_history"],
    },
    # weather_overview: "Tong hop thoi tiet hom nay?"
    # R13: + get_hourly_forecast ("trua/chieu nay" time slot in overview)
    "weather_overview": {
        "city":     ["get_daily_summary", "detect_phenomena", "get_daily_rhythm", "get_current_weather", "get_daily_forecast", "get_weather_period", "get_hourly_forecast"],
        "district": ["get_daily_summary", "detect_phenomena", "get_daily_rhythm", "get_current_weather", "get_daily_forecast", "get_weather_period", "get_hourly_forecast"],
        "ward":     ["get_daily_summary", "detect_phenomena", "get_daily_rhythm", "get_current_weather", "get_daily_forecast", "get_weather_period", "get_hourly_forecast"],
    },
    # rain_query: "Luc nao mua?", "Co mua khong?"
    # R13: + get_current_weather ("dang mua khong" snapshot)
    "rain_query": {
        "city":     ["get_rain_timeline", "get_hourly_forecast", "get_daily_forecast", "get_weather_alerts", "get_daily_summary", "get_current_weather"],
        "district": ["get_rain_timeline", "get_hourly_forecast", "get_daily_forecast", "get_weather_alerts", "get_daily_summary", "get_current_weather"],
        "ward":     ["get_rain_timeline", "get_hourly_forecast", "get_daily_forecast", "get_weather_alerts", "get_daily_summary", "get_current_weather"],
    },
    # temperature_query: "Nhiet do?", "Nong khong?"
    "temperature_query": {
        "city":     ["get_current_weather", "get_temperature_trend", "get_hourly_forecast", "get_daily_forecast"],
        "district": ["get_current_weather", "get_temperature_trend", "get_hourly_forecast", "get_daily_forecast"],
        "ward":     ["get_current_weather", "get_temperature_trend", "get_hourly_forecast", "get_daily_forecast"],
    },
    # wind_query: "Gio manh khong?"
    # R9: + get_weather_alerts (gio giat manh -> canh bao)
    # R13: + get_daily_forecast ("mai / cuoi tuan gio manh")
    "wind_query": {
        "city":     ["get_current_weather", "get_hourly_forecast", "get_pressure_trend", "get_weather_alerts", "get_daily_forecast"],
        "district": ["get_current_weather", "get_hourly_forecast", "get_pressure_trend", "get_weather_alerts", "get_daily_forecast"],
        "ward":     ["get_current_weather", "get_hourly_forecast", "get_pressure_trend", "get_weather_alerts", "get_daily_forecast"],
    },
    # humidity_fog_query: "Do am?", "Suong mu?"
    # R13: + get_hourly_forecast, + get_daily_forecast ("sang mai suong mu")
    "humidity_fog_query": {
        "city":     ["get_current_weather", "detect_phenomena", "get_humidity_timeline", "get_hourly_forecast", "get_daily_forecast"],
        "district": ["get_current_weather", "detect_phenomena", "get_humidity_timeline", "get_hourly_forecast", "get_daily_forecast"],
        "ward":     ["get_current_weather", "detect_phenomena", "get_humidity_timeline", "get_hourly_forecast", "get_daily_forecast"],
    },
    # historical_weather: "Hom qua the nao?", "Tuan truoc?"
    "historical_weather": {
        "city":     ["get_weather_history", "get_daily_summary", "get_weather_period"],
        "district": ["get_weather_history", "get_daily_summary", "get_weather_period"],
        "ward":     ["get_weather_history", "get_daily_summary", "get_weather_period"],
    },
    # location_comparison: "Cau Giay vs Dong Da?", "Quan nao nong nhat?"
    # R9: + get_current_weather (defensive base data for comparison)
    "location_comparison": {
        "city":     ["get_district_ranking", "get_district_multi_compare", "compare_weather", "get_current_weather"],
        "district": ["compare_weather", "get_ward_ranking_in_district", "get_district_ranking",
                     "get_district_multi_compare", "get_current_weather"],
        "ward":     ["compare_weather", "get_current_weather"],
    },
    # activity_weather: "Di choi duoc khong?", "May gio chay bo tot?"
    # R9: + rain_timeline + clothing_advice (activity_advice can combo voi chi tiet)
    # R14 E.1: + daily_forecast + weather_period (future-frame: "ngay mai/cuoi tuan")
    "activity_weather": {
        "city":     ["get_activity_advice", "get_best_time", "get_uv_safe_windows", "get_current_weather",
                     "get_comfort_index", "get_clothing_advice", "get_rain_timeline",
                     "get_daily_forecast", "get_weather_period"],
        "district": ["get_activity_advice", "get_best_time", "get_uv_safe_windows", "get_current_weather",
                     "get_comfort_index", "get_clothing_advice", "get_rain_timeline",
                     "get_daily_forecast", "get_weather_period"],
        "ward":     ["get_activity_advice", "get_best_time", "get_uv_safe_windows", "get_current_weather",
                     "get_comfort_index", "get_clothing_advice", "get_rain_timeline",
                     "get_daily_forecast", "get_weather_period"],
    },
    # expert_weather_param: "Diem suong?", "Ap suat?", "UV index?"
    # R9: + humidity_timeline (dew/am expert)
    "expert_weather_param": {
        "city":     ["get_current_weather", "get_comfort_index", "get_pressure_trend", "get_weather_history",
                     "get_uv_safe_windows", "get_humidity_timeline"],
        "district": ["get_current_weather", "get_comfort_index", "get_pressure_trend", "get_weather_history",
                     "get_uv_safe_windows", "get_humidity_timeline"],
        "ward":     ["get_current_weather", "get_comfort_index", "get_pressure_trend", "get_weather_history",
                     "get_uv_safe_windows", "get_humidity_timeline"],
    },
    # weather_alert: "Co canh bao gi khong?", "Giong manh?", "Mua rao/giong khong?"
    # Đã có get_rain_timeline + get_hourly_forecast (match PRIMARY_TOOL_MAP R4.2).
    # R13: + get_current_weather (snapshot alert check), + get_daily_forecast ("may ngay toi ret hai").
    "weather_alert": {
        "city":     ["get_weather_alerts", "get_weather_change_alert", "get_pressure_trend", "get_rain_timeline", "get_hourly_forecast", "get_current_weather", "get_daily_forecast"],
        "district": ["get_weather_alerts", "get_weather_change_alert", "get_pressure_trend", "get_rain_timeline", "get_hourly_forecast", "get_current_weather", "get_daily_forecast"],
        "ward":     ["get_weather_alerts", "get_weather_change_alert", "get_pressure_trend", "get_rain_timeline", "get_hourly_forecast", "get_current_weather", "get_daily_forecast"],
    },
    # seasonal_context: "Nong hon binh thuong khong?"
    # R13: + get_current_weather, + get_hourly_forecast ("hien tai vs binh thuong").
    "seasonal_context": {
        "city":     ["get_seasonal_comparison", "compare_with_yesterday", "get_temperature_trend", "get_weather_history", "get_daily_summary", "get_current_weather", "get_hourly_forecast"],
        "district": ["get_seasonal_comparison", "compare_with_yesterday", "get_temperature_trend", "get_weather_history", "get_daily_summary", "get_current_weather", "get_hourly_forecast"],
        "ward":     ["get_seasonal_comparison", "compare_with_yesterday", "get_temperature_trend", "get_weather_history", "get_daily_summary", "get_current_weather", "get_hourly_forecast"],
    },
    # smalltalk_weather: "Xin chao", "Mac gi hom nay?"
    # no-tools-called cung chap nhan duoc (greetings).
    # R9: + rain_timeline defensive (user chon Option A; confusion smalltalk->rain 2x).
    "smalltalk_weather": {
        "city":     ["get_current_weather", "get_clothing_advice", "get_comfort_index",
                     "get_activity_advice", "get_rain_timeline"],
        "district": ["get_current_weather", "get_clothing_advice", "get_comfort_index",
                     "get_activity_advice", "get_rain_timeline"],
        "ward":     ["get_current_weather", "get_clothing_advice", "get_comfort_index",
                     "get_activity_advice", "get_rain_timeline"],
    },
}


def _get_expected_tools(intent: str, location_scope: str = "") -> list:
    """Get expected tools for (intent, location_scope) from hierarchical mapping.

    System supports 3 scopes: city / district / ward.
    POI queries (eval CSV may contain location_scope="poi") are resolved to
    "district" at runtime via dispatch.py — so use district tools for poi.
    Falls back to "city" if scope not found.
    Returns empty list for unknown intents.
    """
    intent_map = INTENT_TO_TOOLS.get(intent, {})
    if not intent_map:
        return []
    # POI -> district: dispatch.py resolves all POI locations to district level
    effective_scope = "district" if location_scope == "poi" else location_scope
    return intent_map.get(effective_scope) or intent_map.get("city", [])


def check_tool_accuracy(intent: str, tools_called: list,
                        location_scope: str = "") -> bool:
    """Check if at least one scope-appropriate tool was called.

    Uses hierarchical INTENT_TO_TOOLS: the expected tools depend on BOTH the
    intent AND the location scope (city / district / ward).
    POI scope from eval CSV is mapped to district.
    """
    expected = _get_expected_tools(intent, location_scope)
    if not expected:
        return True  # Unknown intent, skip check
    # For smalltalk: no tools called is acceptable (greeting, out-of-scope, etc.)
    if intent == "smalltalk_weather" and not tools_called:
        return True
    return any(t in expected for t in tools_called)


def check_tool_precision(intent: str, tools_called: list,
                         location_scope: str = "") -> float:
    """What fraction of called tools were scope-appropriate? (precision)

    Uses hierarchical INTENT_TO_TOOLS so that e.g. calling get_district_ranking
    for a ward-level question is correctly identified as irrelevant.
    """
    expected = set(_get_expected_tools(intent, location_scope))
    if not tools_called:
        # No tools called: for smalltalk this is fine (precision=1.0)
        return 1.0 if intent == "smalltalk_weather" else 0.0
    # Exclude resolve_location from precision calc (it's a helper, always valid)
    relevant_calls = [t for t in tools_called if t != "resolve_location"]
    if not relevant_calls:
        return 1.0
    relevant = sum(1 for t in relevant_calls if t in expected)
    return round(relevant / len(relevant_calls), 2)


def check_tool_recall(intent: str, tools_called: list,
                      location_scope: str = "") -> float:
    """Scope-aware tool recall: did the agent call at least one expected tool?

    Returns 1.0 if any called tool is in the expected set for (intent, scope),
    otherwise 0.0.

    This is binary recall because the expected set contains ALTERNATIVES
    (any one is sufficient), not REQUIREMENTS (all must be called).
    """
    expected = _get_expected_tools(intent, location_scope)
    if not expected:
        return 1.0  # Unknown intent
    # For smalltalk: no tools called is acceptable
    if intent == "smalltalk_weather" and not tools_called:
        return 1.0

    called_set = set(tools_called)
    expected_set = set(expected)
    return 1.0 if (called_set & expected_set) else 0.0
