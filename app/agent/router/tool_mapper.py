"""Tool Mapper R9 → R14 — map (intent, scope) → focused tool list.

## R14 E.1 (2026-04-23) — activity_weather future-frame widening

v12 eval (58.8% Đạt) cho thấy 4+ rows (ID 114, 118, 160, 161) fail vì intent
`activity_weather` focused tools thiếu `get_daily_forecast` / `get_weather_period`.
LLM buộc áp `get_activity_advice` (snapshot NOW) cho "sáng mai/ngày mai/cuối tuần".
R14 fix: drop `get_uv_safe_windows` (ít primary, best_time cover giờ tốt), add
`get_daily_forecast` + `get_weather_period` → tool count 6→7 (cap).

## R13 widening (2026-04-22) — defensive safety net layer

### Lý do widen
Audit v11 (data/evaluation/audit_report_v11.md Part C) cho thấy 34/199 (~17%)
queries bị LLM gọi tool ngoài focused_tools → LangGraph reject với error "is not
a valid tool, try one of [...]" → wasted call + fallback `get_current_weather`
→ dán snapshot 08:00 làm "chiều/tối/mai/cuối tuần" (15+ failure IDs).

Root cause: PRIMARY_TOOL_MAP ở R9 optimize cho confusion pair count ≥ 2, nhưng
thiếu baseline coverage cho future-frame queries khi router đưa câu hỏi vào
intent không có forecast tool. Vd `rain_query` focused=[rain_timeline, hourly,
daily, alerts] — thiếu `current_weather` cho "đang mưa không" snapshot check.

### R13 widening rule
- **Baseline future-frame trio**: Mọi intent có khả năng carry time slot
  ("chiều/tối/mai/cuối tuần") phải include {current, hourly, daily}.
- **focused_tools = DEFENSIVE SAFETY NET** (preserve R9 philosophy): wider hơn
  strict intent semantics để cover router misclassification. Không cần
  tool_map == intent definition trong design docs.
- **Avg tools/intent**: 3.7 → ~5.0 (cap 7 giữ nguyên, test_tool_count_reasonable).

### Per-intent widening
| Intent | +Added | Reason |
|---|---|---|
| rain_query | +current_weather | "đang mưa không" snapshot |
| wind_query | +daily_forecast | "mai gió mạnh không" |
| humidity_fog_query | +hourly, +daily | "sáng mai sương mù" |
| weather_overview | +current, +hourly | overview + snapshot + giờ slot |
| weather_alert | +current, +daily | snapshot alert + mấy ngày tới |
| seasonal_context | +current, +hourly | "dạo này so hiện tại" |
| daily_forecast | +weather_history | recover "7 ngày qua" misrouted |

## R9 legacy context (preserved)

### Bỏ EXPANDED_TOOL_MAP
- Lý do: confidence hardcode 0.9 ở 53% training samples → confidence không
  reliable để phân biệt high/medium confidence. EXPANDED chỉ kích hoạt ở
  medium zone, nhưng thực tế model predict với confidence ~0.9 cho đa số
  query → EXPANDED gần như không được dùng.

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
  chính của Y. R13 bổ sung baseline trio theo future-frame.

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
    # R13: +history cho recover "7 ngày qua" misrouted (IDs 147, 166).
    "daily_forecast": _flat([
        get_daily_forecast,        # primary
        get_daily_summary,         # sister tool: 1 ngày chi tiết 4 buổi
        get_weather_period,        # range rộng
        get_temperature_trend,     # xu hướng
        get_rain_timeline,         # DEFENSIVE: →rain_query
        get_hourly_forecast,       # DEFENSIVE: →hourly_forecast
        get_weather_history,       # R13: "7 ngày qua" misrouted recovery
    ]),

    # "Tổng hợp hôm nay / overview"
    # Acc 82.6%. Confusion: →daily_forecast (2).
    # R13: +current, +hourly cho overview có time slot ("trưa nay/chiều nay").
    "weather_overview": _flat([
        get_daily_summary,         # primary: 4 buổi chi tiết
        detect_phenomena,
        get_daily_rhythm,          # nhịp nhiệt trong ngày
        get_daily_forecast,        # DEFENSIVE: →daily_forecast
        get_current_weather,       # R13: "hôm nay thế nào" + snapshot
        get_hourly_forecast,       # R13: "trưa/chiều nay" time slot
    ]),

    # ══════ FOCUS BY METRIC ══════

    # "Lúc nào mưa / mấy giờ tạnh / có mưa không"
    # Acc 100% nhưng user thường hỏi combo ("mưa tuần này" → cần daily).
    # R13: +current cho "đang mưa không / lúc này mưa" snapshot check (ID 3).
    "rain_query": _flat([
        get_rain_timeline,         # primary: đợt mưa 48h
        get_hourly_forecast,       # giờ-by-giờ chi tiết
        get_daily_forecast,        # "mưa tuần này / ngày mai"
        get_weather_alerts,        # "có cảnh báo mưa to"
        get_current_weather,       # R13: "đang mưa / lúc này mưa" snapshot
        get_clothing_advice,       # R15 T2.3: "mưa không và nên mặc gì" (ID 196 fix)
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
    # R13: +daily cho "mai/cuối tuần gió mạnh không" future-frame.
    "wind_query": _flat([
        get_current_weather,       # primary: có wind_speed + wind_gust
        get_pressure_trend,        # front → gió
        get_hourly_forecast,       # "chiều nay gió bao nhiêu"
        get_weather_alerts,        # "gió giật → cảnh báo"
        get_daily_forecast,        # R13: "ngày mai / cuối tuần gió"
    ]),

    # "Độ ẩm / sương mù / nồm ẩm"
    # Acc 100%. R13: +hourly, +daily cho "sáng mai sương mù" future-frame
    # (ID 19 thất bại vì không có hourly/daily để lấy data tương lai).
    "humidity_fog_query": _flat([
        get_current_weather,       # primary
        get_humidity_timeline,     # timeline ẩm + dew
        detect_phenomena,          # nồm/sương mù đặc trưng HN
        get_hourly_forecast,       # R13: "sáng mai / chiều nay sương mù"
        get_daily_forecast,        # R13: "ngày mai / tuần này ẩm thế nào"
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
    # R14 E.1: +daily_forecast + weather_period cho "sáng mai/ngày mai/cuối tuần đi X"
    # (v12 IDs 114, 118, 160, 161: LLM áp activity_advice NOW cho future-frame vì
    # focused tools thiếu forecast). Drop get_uv_safe_windows: best_time đã cover giờ tốt,
    # UV thường là info phụ + full agent vẫn có cho query chuyên UV.
    "activity_weather": _flat([
        get_activity_advice,       # primary: advise chung
        get_best_time,             # giờ tốt nhất (48h)
        get_daily_forecast,        # R14 E.1: "ngày mai/sáng mai tập X"
        get_weather_period,        # R14 E.1: "cuối tuần đi cắm trại"
        get_hourly_forecast,       # R15 T2.2: "sáng CN/chiều mai đi bộ" granular (ID 119 fix)
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
    # R13: +current cho snapshot alert check, +daily cho "mấy ngày tới rét hại".
    "weather_alert": _flat([
        get_weather_alerts,        # primary
        get_weather_change_alert,  # đột biến 6-12h
        get_pressure_trend,        # front = cảnh báo
        get_rain_timeline,         # giông/mưa to cụ thể
        get_hourly_forecast,       # "bão khi nào đến"
        get_current_weather,       # R13: snapshot "đang có cảnh báo gì"
        get_daily_forecast,        # R13: "mấy ngày tới có rét đậm không"
    ]),

    # ══════ CLIMATOLOGY ══════

    # "Dạo này nóng hơn bình thường / so mùa này"
    # Acc 92.9%. Confusion: →historical_weather (2).
    # R13: +current, +hourly cho "hiện tại vs bình thường / hôm nay vs mùa này".
    "seasonal_context": _flat([
        get_seasonal_comparison,   # primary: climatology
        compare_with_yesterday,    # short-term delta
        get_weather_history,       # DEFENSIVE: →historical
        get_temperature_trend,     # xu hướng hỗ trợ "dạo này"
        get_current_weather,       # R13: "hiện tại so bình thường"
        get_hourly_forecast,       # R13: "hôm nay vs mùa này"
        get_daily_forecast,        # R15 T2.1: "hôm nay vs ngày mai" (ID 195 fix)
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
