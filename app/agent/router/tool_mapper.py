"""Tool Mapper R9 — map (intent, scope) → focused tool list.

## R9 redesign (2026-04-21)

### Bỏ EXPANDED_TOOL_MAP
- Lý do: confidence hardcode 0.9 ở 53% training samples → confidence không
  reliable để phân biệt high/medium confidence. EXPANDED chỉ kích hoạt ở
  medium zone, nhưng thực tế model predict với confidence ~0.9 cho đa số
  query → EXPANDED gần như không được dùng.
- Hậu quả cũ: khi router nhầm với high confidence → dùng PRIMARY (narrow
  tool set) → bot không có tool để trả lời đúng.

### Defensive tool coverage trong PRIMARY
- Phân tích confusion pairs của Qwen3-4B (training/notebooks/run_02/outputs/
  exp6_summary.json):
  + daily_forecast → rain_query (2 confusions)
  + daily_forecast → hourly_forecast (2)
  + weather_overview → daily_forecast (2)
  + current_weather → rain_query (2)
  + current_weather → temperature_query (2)
  + seasonal_context → historical_weather (2)
  + smalltalk_weather → rain_query (2)
  + activity_weather → smalltalk_weather (2)
- Rule: với confusion pair X→Y count ≥ 2, tool map của X PHẢI chứa tool
  chính của Y. → Defensive coverage.

### Confidence < threshold → fallback full agent (None)
- Nếu router trả confidence thấp, get_focused_tools trả None.
- Caller (stream_slm_agent / run_slm_agent) fallback sang full 27-tool agent.
- Không còn "medium zone EXPANDED".

### Flatten scope structure (trừ location_comparison)
- 14/15 intents có tool set identical giữa city/district/ward.
- Dùng helper `_flat` để tránh lặp lại 3 lần.
- `location_comparison` giữ nested vì city dùng ranking, district/ward dùng
  compare_weather — khác biệt thực sự.
"""

from __future__ import annotations

from app.agent.tools import (
    resolve_location,
    get_current_weather,
    get_weather_alerts,
    get_hourly_forecast,
    get_daily_forecast,
    get_rain_timeline,
    get_best_time,
    get_weather_history,
    get_daily_summary,
    get_weather_period,
    compare_weather,
    compare_with_yesterday,
    get_seasonal_comparison,
    get_district_ranking,
    get_ward_ranking_in_district,
    detect_phenomena,
    get_temperature_trend,
    get_comfort_index,
    get_weather_change_alert,
    get_clothing_advice,
    get_activity_advice,
    get_uv_safe_windows,
    get_pressure_trend,
    get_daily_rhythm,
    get_humidity_timeline,
    get_sunny_periods,
    get_district_multi_compare,
    TOOLS,
)


def _flat(tools: list) -> dict[str, list]:
    """Giúp giảm boilerplate: intent có cùng tool set cho 3 scope."""
    return {"city": tools, "district": tools, "ward": tools}


# ── PRIMARY_TOOL_MAP R9 ──
# Mỗi intent: primary tool + 1-3 defensive tool cover confusion pairs.
# Số tool trung bình ~4 (trong range focused, không vượt 6).

PRIMARY_TOOL_MAP: dict[str, dict[str, list]] = {

    # ══════ SNAPSHOT & NOWCAST ══════

    # "Bây giờ / hiện tại": snapshot + phenomena + defensive rain
    # Acc 77.8% (lowest) — cần defensive nặng.
    # Confusion: →rain (2), →temperature_query (2), →weather_overview (1)
    "current_weather": _flat([
        get_current_weather,       # primary: snapshot
        detect_phenomena,          # insight: nồm/gió mùa/rét đậm
        get_rain_timeline,         # DEFENSIVE: "bây giờ có mưa không"
        get_hourly_forecast,       # DEFENSIVE: user có thể thực hỏi chiều/tối
    ]),

    # ══════ FORECAST (TIME-BASED) ══════

    # "Chiều/tối/vài giờ tới" — 1-48h
    # Acc 84%. Thêm rain để cover edge rain query trong frame giờ.
    "hourly_forecast": _flat([
        get_hourly_forecast,       # primary
        get_sunny_periods,         # "khi nào nắng trong 48h"
        get_rain_timeline,         # DEFENSIVE: rain chi tiết trong 48h
    ]),

    # "Ngày mai / thứ X / 3 ngày tới / tuần" — 1-8 ngày
    # Acc 87.1%. Confusion: →rain (2), →hourly (2).
    "daily_forecast": _flat([
        get_daily_forecast,        # primary
        get_daily_summary,         # sister tool: 1 ngày chi tiết 4 buổi
        get_weather_period,        # range rộng
        get_temperature_trend,     # xu hướng
        get_rain_timeline,         # DEFENSIVE: →rain_query
        get_hourly_forecast,       # DEFENSIVE: →hourly_forecast
    ]),

    # "Tổng hợp hôm nay / overview"
    # Acc 82.6%. Confusion: →daily_forecast (2).
    "weather_overview": _flat([
        get_daily_summary,         # primary: 4 buổi chi tiết
        detect_phenomena,
        get_daily_rhythm,          # nhịp nhiệt trong ngày
        get_daily_forecast,        # DEFENSIVE: →daily_forecast
    ]),

    # ══════ FOCUS BY METRIC ══════

    # "Lúc nào mưa / mấy giờ tạnh / có mưa không"
    # Acc 100% nhưng user thường hỏi combo ("mưa tuần này" → cần daily).
    "rain_query": _flat([
        get_rain_timeline,         # primary: đợt mưa 48h
        get_hourly_forecast,       # giờ-by-giờ chi tiết
        get_daily_forecast,        # "mưa tuần này / ngày mai"
        get_weather_alerts,        # "có cảnh báo mưa to"
    ]),

    # "Nhiệt độ bao nhiêu / nóng không / lạnh"
    # Acc 100%. Thêm hourly/daily cho future frame.
    "temperature_query": _flat([
        get_current_weather,       # primary
        get_temperature_trend,     # "bao giờ ấm/lạnh"
        get_hourly_forecast,       # "tối nay nhiệt"
        get_daily_forecast,        # "ngày mai max/min"
    ]),

    # "Gió mạnh không / tốc độ gió"
    # Acc 100%. Thêm alerts vì gió giật mạnh = cảnh báo.
    "wind_query": _flat([
        get_current_weather,       # primary: có wind_speed + wind_gust
        get_pressure_trend,        # front → gió
        get_hourly_forecast,       # "chiều nay gió bao nhiêu"
        get_weather_alerts,        # "gió giật → cảnh báo"
    ]),

    # "Độ ẩm / sương mù / nồm ẩm"
    # Acc 100%. Giữ nguyên, đã đủ.
    "humidity_fog_query": _flat([
        get_current_weather,       # primary
        get_humidity_timeline,     # timeline ẩm + dew
        detect_phenomena,          # nồm/sương mù đặc trưng HN
    ]),

    # ══════ TIME: PAST ══════

    # "Hôm qua / ngày X đã qua / tuần trước"
    # Acc 100%. Thêm summary cho "chi tiết ngày X".
    "historical_weather": _flat([
        get_weather_history,       # primary
        get_daily_summary,         # DEFENSIVE: 1 ngày chi tiết 4 buổi
    ]),

    # ══════ COMPARISON ══════

    # "So quận / xếp hạng / A vs B"
    # Acc 100%. Scope-DIFFERENT — giữ nested.
    "location_comparison": {
        "city":     [get_district_ranking, get_district_multi_compare, get_current_weather],
        "district": [compare_weather, get_ward_ranking_in_district, get_current_weather],
        "ward":     [compare_weather, get_current_weather],
    },

    # ══════ ACTIVITY & ADVISORY ══════

    # "Đi chơi / chạy bộ / mấy giờ tốt"
    # Acc 94.3%. Confusion: →smalltalk (2).
    # activity_advice generic → cần combo với rain/UV/current.
    "activity_weather": _flat([
        get_activity_advice,       # primary: advise chung
        get_best_time,             # giờ tốt nhất
        get_uv_safe_windows,       # UV window
        get_clothing_advice,       # trang phục — overlap với smalltalk
        get_rain_timeline,         # DEFENSIVE: "chiều đi chơi có mưa không"
        get_current_weather,       # DEFENSIVE: base data + cover →smalltalk
    ]),

    # ══════ EXPERT ══════

    # "Dew point / áp suất / UV index / feels like"
    # Acc 95%.
    "expert_weather_param": _flat([
        get_current_weather,       # primary: có đầy đủ expert fields
        get_comfort_index,         # heat index / wind chill
        get_pressure_trend,        # áp suất
        get_humidity_timeline,     # DEFENSIVE: dew/ẩm expert
    ]),

    # ══════ ALERT & PHENOMENA ══════

    # "Cảnh báo / bão / ngập / rét hại / giông"
    # Acc 93.3%. Rule an toàn: cover cả rain + hourly.
    "weather_alert": _flat([
        get_weather_alerts,        # primary
        get_weather_change_alert,  # đột biến 6-12h
        get_pressure_trend,        # front = cảnh báo
        get_rain_timeline,         # giông/mưa to cụ thể
        get_hourly_forecast,       # "bão khi nào đến"
    ]),

    # ══════ CLIMATOLOGY ══════

    # "Dạo này nóng hơn bình thường / so mùa này"
    # Acc 92.9%. Confusion: →historical_weather (2).
    "seasonal_context": _flat([
        get_seasonal_comparison,   # primary: climatology
        compare_with_yesterday,    # short-term delta
        get_weather_history,       # DEFENSIVE: →historical
        get_temperature_trend,     # xu hướng hỗ trợ "dạo này"
    ]),

    # ══════ SMALLTALK (User Option A: defensive coverage) ══════

    # "Chào / cảm ơn / trời đẹp không / hôm nay nóng nhỉ"
    # Acc 83.3%. Confusion: →rain (2), →weather_overview (1).
    # User chọn Option A: thêm defensive tool.
    "smalltalk_weather": _flat([
        get_current_weather,       # primary: data nhẹ cho "hôm nay thế nào"
        get_clothing_advice,       # "mặc gì"
        get_rain_timeline,         # DEFENSIVE: →rain_query
        get_comfort_index,         # DEFENSIVE: "dễ chịu không"
    ]),
}


def get_focused_tools(
    intent: str,
    scope: str,
    confidence: float = 1.0,
    per_intent_thresholds: dict | None = None,
) -> list | None:
    """Get focused tool list for (intent, scope, confidence).

    R9 logic (bỏ EXPANDED_TOOL_MAP):
    - confidence >= per_intent_threshold → PRIMARY_TOOL_MAP (focused 3-6 tools,
      đã có defensive coverage cho các confusion pair thường gặp).
    - confidence <  per_intent_threshold → None → caller fallback full
      27-tool agent (run_agent path).

    Args:
        intent: Classified intent string.
        scope: "city" | "district" | "ward".
        confidence: Router confidence score (0.0-1.0).
        per_intent_thresholds: Optional per-intent threshold dict. Defaults to
            CONFIDENCE_THRESHOLD nếu None.

    Returns:
        List tool functions, hoặc None nếu confidence quá thấp (caller fallback).
    """
    from app.agent.router.config import CONFIDENCE_THRESHOLD

    threshold = (per_intent_thresholds or {}).get(intent, CONFIDENCE_THRESHOLD)

    if confidence < threshold:
        return None  # Caller fallback sang full agent (27 tools)

    scope_map = PRIMARY_TOOL_MAP.get(intent)
    if scope_map is None:
        return None

    tools = scope_map.get(scope)
    if tools is None:
        # Scope không match (vd unknown scope) → fallback sang city
        tools = scope_map.get("city")
    return tools


def get_all_tools() -> list:
    """Return toàn bộ 27 tools (dùng cho fallback full-agent path)."""
    return TOOLS
