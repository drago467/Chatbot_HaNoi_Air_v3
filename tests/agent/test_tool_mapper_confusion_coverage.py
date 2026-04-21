"""R9: Test tool map coverage cho confusion pairs của Qwen3-4B.

Ý tưởng: khi router nhầm (confidence cao) predict intent X thay vì Y, tool list
của X phải chứa tool chính của Y → bot vẫn có thể trả lời đúng câu user hỏi.

Dataset: training/notebooks/run_02/outputs/exp6_summary.json (Qwen3-4B-v5).
Threshold: confusion count ≥ 2 → defensive coverage BẮT BUỘC.
"""

from __future__ import annotations

import json
import os

import pytest

from app.agent.router.tool_mapper import (
    PRIMARY_TOOL_MAP,
    get_focused_tools,
    get_all_tools,
)
from app.agent.tools import TOOLS

# Primary tool name cho MỖI intent — dùng để assert coverage.
# Khi confusion X→Y (user thực sự muốn Y), tool này của Y phải có trong X.
INTENT_PRIMARY_TOOL = {
    "current_weather":       "get_current_weather",
    "hourly_forecast":       "get_hourly_forecast",
    "daily_forecast":        "get_daily_forecast",
    "weather_overview":      "get_daily_summary",
    "rain_query":            "get_rain_timeline",
    "temperature_query":     "get_current_weather",
    "wind_query":            "get_current_weather",
    "humidity_fog_query":    "get_humidity_timeline",
    "historical_weather":    "get_weather_history",
    "location_comparison":   "compare_weather",  # district/ward; city dùng ranking
    "activity_weather":      "get_activity_advice",
    "expert_weather_param":  "get_current_weather",
    "weather_alert":         "get_weather_alerts",
    "seasonal_context":      "get_seasonal_comparison",
    "smalltalk_weather":     "get_current_weather",
}


def _load_confusion_pairs(min_count: int = 2) -> list[tuple[str, str, int]]:
    """Load confusion pairs từ Qwen3-4B-v5 eval summary."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "training", "notebooks", "run_02", "outputs", "exp6_summary.json"
    )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    model_key = next(k for k in data["models"] if "4B" in k)
    pairs_raw = data["models"][model_key].get("top_confusion_pairs", [])
    result = []
    for raw in pairs_raw:
        # raw = ["X->Y", count]
        key, count = raw[0], raw[1]
        if count < min_count:
            continue
        x, y = key.split("->")
        result.append((x, y, count))
    return result


def _tool_names_in_primary(intent: str, scope: str = "city") -> set[str]:
    """Lấy set tên tool trong PRIMARY_TOOL_MAP[intent][scope]."""
    scope_map = PRIMARY_TOOL_MAP.get(intent, {})
    tools = scope_map.get(scope) or scope_map.get("city") or []
    return {t.name for t in tools}


# ══ 1. Mọi 15 intent có entry trong PRIMARY_TOOL_MAP ══

KNOWN_INTENTS = list(INTENT_PRIMARY_TOOL.keys())


@pytest.mark.parametrize("intent", KNOWN_INTENTS)
def test_every_intent_has_primary_tool_map_entry(intent):
    assert intent in PRIMARY_TOOL_MAP, f"Missing intent {intent}"
    scope_map = PRIMARY_TOOL_MAP[intent]
    for scope in ("city", "district", "ward"):
        assert scope in scope_map, f"Missing scope {scope} cho intent {intent}"
        assert len(scope_map[scope]) >= 1, f"Empty tool list cho {intent}/{scope}"


@pytest.mark.parametrize("intent", KNOWN_INTENTS)
def test_every_intent_primary_tool_present(intent):
    """Tool chính của intent phải có trong map của intent đó."""
    # location_comparison special: city dùng ranking, district/ward dùng compare_weather
    if intent == "location_comparison":
        assert "get_district_ranking" in _tool_names_in_primary("location_comparison", "city")
        assert "compare_weather" in _tool_names_in_primary("location_comparison", "district")
        assert "compare_weather" in _tool_names_in_primary("location_comparison", "ward")
        return
    primary = INTENT_PRIMARY_TOOL[intent]
    tools = _tool_names_in_primary(intent, "city")
    assert primary in tools, (
        f"Intent {intent} thiếu primary tool {primary} trong city scope. Có: {tools}"
    )


# ══ 2. Confusion coverage (HEART OF R9) ══

CONFUSION_PAIRS = _load_confusion_pairs(min_count=2)


@pytest.mark.parametrize("confusion", CONFUSION_PAIRS)
def test_confusion_pair_coverage(confusion):
    """Khi router nhầm X→Y với count ≥ 2, tool map của X PHẢI chứa primary
    tool của Y (defensive coverage)."""
    wrong_intent, correct_intent, count = confusion
    correct_primary_tool = INTENT_PRIMARY_TOOL.get(correct_intent)
    if correct_primary_tool is None:
        pytest.skip(f"Unknown correct_intent {correct_intent}")
    wrong_intent_tools = _tool_names_in_primary(wrong_intent, "city")
    assert correct_primary_tool in wrong_intent_tools, (
        f"Confusion {wrong_intent}→{correct_intent} (count={count}) KHÔNG được cover: "
        f"tool {correct_primary_tool!r} của {correct_intent} thiếu trong PRIMARY_TOOL_MAP[{wrong_intent}]. "
        f"Có: {sorted(wrong_intent_tools)}"
    )


# ══ 3. Size sanity — tool count không quá lớn ══

@pytest.mark.parametrize("intent", KNOWN_INTENTS)
def test_tool_count_reasonable(intent):
    """Mỗi intent có ≤ 7 tool để focused prompt không quá dài."""
    for scope in ("city", "district", "ward"):
        tools = PRIMARY_TOOL_MAP[intent].get(scope, [])
        assert 1 <= len(tools) <= 7, (
            f"{intent}/{scope} có {len(tools)} tools — nên trong [1, 7]"
        )


# ══ 4. Tất cả tool reference đều là valid tool (không dangling) ══

def test_no_dangling_tool_references():
    valid_tool_names = {t.name for t in TOOLS}
    for intent, scope_map in PRIMARY_TOOL_MAP.items():
        for scope, tools in scope_map.items():
            for t in tools:
                assert t.name in valid_tool_names, (
                    f"Dangling tool {t.name} trong {intent}/{scope}"
                )


# ══ 5. get_focused_tools logic ══

def test_high_confidence_returns_primary():
    """Confidence cao → PRIMARY_TOOL_MAP entry."""
    tools = get_focused_tools("current_weather", "city", confidence=0.95)
    assert tools is not None
    assert len(tools) >= 1
    tool_names = {t.name for t in tools}
    assert "get_current_weather" in tool_names


def test_low_confidence_returns_none_for_fallback():
    """Confidence thấp → None để caller fallback full-agent."""
    # Threshold mặc định ~0.75. Confidence 0.3 chắc chắn < tất cả threshold.
    tools = get_focused_tools("current_weather", "city", confidence=0.3)
    assert tools is None, "Confidence thấp phải trả None để fallback full agent"


def test_unknown_intent_returns_none():
    tools = get_focused_tools("unknown_intent_xyz", "city", confidence=0.95)
    assert tools is None


def test_unknown_scope_fallback_to_city():
    """Scope unknown → fallback city."""
    tools = get_focused_tools("current_weather", "unknown_scope", confidence=0.95)
    assert tools is not None
    expected_city = PRIMARY_TOOL_MAP["current_weather"]["city"]
    assert tools == expected_city


def test_per_intent_thresholds_respected():
    """Nếu per_intent_thresholds chỉ định, dùng giá trị đó."""
    thresholds = {"current_weather": 0.9}
    # Confidence 0.8 < 0.9 → fallback
    assert get_focused_tools("current_weather", "city", 0.8, thresholds) is None
    # Confidence 0.95 > 0.9 → PRIMARY
    assert get_focused_tools("current_weather", "city", 0.95, thresholds) is not None


def test_expanded_tool_map_removed():
    """R9: EXPANDED_TOOL_MAP đã xoá khỏi module."""
    import app.agent.router.tool_mapper as tm
    assert not hasattr(tm, "EXPANDED_TOOL_MAP"), (
        "EXPANDED_TOOL_MAP should be removed in R9"
    )


def test_get_all_tools_returns_27():
    assert len(get_all_tools()) == 27


# ══ 6. Eval INTENT_TO_TOOLS ⊇ PRIMARY_TOOL_MAP (recall invariant) ══

def test_eval_intent_to_tools_is_superset_of_primary():
    """INTENT_TO_TOOLS (eval recall) phải chứa ít nhất mọi tool trong
    PRIMARY_TOOL_MAP — nếu không, recall sẽ bị underestimate."""
    from experiments.evaluation.tool_accuracy import INTENT_TO_TOOLS

    for intent, scope_map in PRIMARY_TOOL_MAP.items():
        if intent not in INTENT_TO_TOOLS:
            pytest.skip(f"Eval không có entry cho intent {intent}")
        for scope, tools in scope_map.items():
            eval_tools = set(INTENT_TO_TOOLS[intent].get(scope, []))
            primary_tool_names = {t.name for t in tools}
            missing = primary_tool_names - eval_tools
            assert not missing, (
                f"INTENT_TO_TOOLS[{intent}][{scope}] thiếu tool từ PRIMARY: {missing}. "
                f"Cần thêm để eval recall nhất quán."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
