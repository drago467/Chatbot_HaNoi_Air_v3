"""R12 L3: 7 shared exemplars validation.

Tests:
- 7 exemplars trong top-level "examples" key
- Mỗi exemplar có đủ fields: title, user, thought, action, observation, response_prefix
- 3 exemplars mới cover F1/F2/F6 (past-frame, superlative, multi-part)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FEW_SHOT_PATH = Path(__file__).resolve().parents[2] / "app" / "config" / "few_shot_examples.json"


@pytest.fixture(scope="module")
def data() -> dict:
    with FEW_SHOT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_examples_key_exists(data):
    assert "examples" in data
    assert isinstance(data["examples"], list)


def test_seven_exemplars(data):
    """R12 L3: 4 R11 exemplars + 3 R12 (past-frame, superlative, multi-part) = 7."""
    assert len(data["examples"]) == 7, (
        f"Expected 7 exemplars (R11 4 + R12 3), got {len(data['examples'])}"
    )


def test_exemplar_fields(data):
    """Mỗi exemplar có đủ 6 fields."""
    required = {"title", "user", "thought", "action", "observation", "response_prefix"}
    for i, ex in enumerate(data["examples"]):
        missing = required - set(ex.keys())
        assert not missing, f"Exemplar #{i} ({ex.get('title','?')}) missing fields: {missing}"


def test_exemplar_5_past_frame(data):
    """Exemplar 5: past-frame refuse with get_weather_history."""
    ex = data["examples"][4]
    assert "past-frame" in ex["title"].lower() or "đã qua" in ex["title"].lower()
    # User asks "chiều nay" — a past frame at 21:00
    assert "chiều nay" in ex["user"].lower()
    # Action must call weather_history (not hourly_forecast)
    assert "get_weather_history" in ex["action"]
    # Response acknowledges "đã qua"
    assert "đã qua" in ex["response_prefix"].lower()


def test_exemplar_6_superlative(data):
    """Exemplar 6: superlative → daily_summary, not current_weather snapshot."""
    ex = data["examples"][5]
    assert "superlative" in ex["title"].lower() or "cả ngày" in ex["title"].lower()
    # User query has superlative word
    assert any(k in ex["user"].lower() for k in ("trung bình", "mạnh nhất", "max", "min"))
    # Action uses daily_summary (not get_current_weather)
    assert "get_daily_summary" in ex["action"]
    # Observation has "tổng hợp" key
    assert "tổng hợp" in ex["observation"]


def test_exemplar_7_multi_part(data):
    """Exemplar 7: multi-part, calls 2 tools in 1 turn."""
    ex = data["examples"][6]
    assert "multi-part" in ex["title"].lower() or "aspects" in ex["title"].lower() or "2" in ex["title"]
    # User query has "và" connector
    assert "và" in ex["user"].lower()
    # Action shows both tools
    assert "get_weather_alerts" in ex["action"]
    assert "get_daily_forecast" in ex["action"]
    # Response numbers the aspects
    assert "(1)" in ex["response_prefix"] and "(2)" in ex["response_prefix"]


def test_exemplar_titles_unique(data):
    """Mỗi exemplar có title unique."""
    titles = [ex["title"] for ex in data["examples"]]
    assert len(titles) == len(set(titles)), f"Duplicate titles: {titles}"


def test_r11_exemplars_retained(data):
    """R11 4 exemplars (in-horizon, compositional, out-of-horizon, empty) vẫn có."""
    titles_lower = [ex["title"].lower() for ex in data["examples"]]
    joined = " | ".join(titles_lower)
    assert "in-horizon" in joined
    assert "compositional" in joined or "verify" in joined
    assert "out-of-horizon" in joined
    assert "empty" in joined


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
