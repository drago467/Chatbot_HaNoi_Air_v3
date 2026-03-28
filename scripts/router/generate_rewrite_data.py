"""Generate multi-task training data for SLM Router (Module 1b).

Extends existing routing data with contextual rewriting examples.
The model learns to:
1. Classify intent + scope (same as before) — from existing data
2. Optionally rewrite ambiguous queries when context is available

Output format (JSONL):
    {"input": "Còn ngày mai?",
     "context": {"location": "Cầu Giấy", "intent": "current_weather", "turn": 1},
     "output": {"intent": "daily_forecast", "scope": "district", "confidence": 0.91,
                "rewritten_query": "Dự báo thời tiết ngày mai ở quận Cầu Giấy?"}}

    {"input": "Thời tiết Cầu Giấy hiện tại?",
     "context": null,
     "output": {"intent": "current_weather", "scope": "district", "confidence": 0.95}}

Usage:
    python scripts/router/generate_rewrite_data.py
    python scripts/router/generate_rewrite_data.py --output data/router/rewrite_train.jsonl
    python scripts/router/generate_rewrite_data.py --use-llm  # Use GPT-4o-mini for variations
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

# ── Seed conversation templates ──
# Each template: (turn0_query, context_snippet, follow_up_queries)
# follow_up_queries: list of (ambiguous_query, rewritten_query, intent, scope)

_SEED_TEMPLATES = [
    # ─────────────── Pattern: Location carry-over ───────────────
    {
        "location": "Cầu Giấy", "location_scope": "district",
        "turn0": "Thời tiết Cầu Giấy hiện tại thế nào?",
        "turn0_intent": "current_weather",
        "followups": [
            ("Còn ngày mai?", "Dự báo thời tiết ngày mai ở quận Cầu Giấy như thế nào?", "daily_forecast", "district"),
            ("Tối mai có mưa không?", "Tối mai ở quận Cầu Giấy có mưa không?", "rain_query", "district"),
            ("Cuối tuần thế nào?", "Dự báo thời tiết cuối tuần ở quận Cầu Giấy?", "daily_forecast", "district"),
            ("Gió mạnh không?", "Gió ở quận Cầu Giấy hiện tại mạnh không?", "wind_query", "district"),
            ("Nhiệt độ bao nhiêu?", "Nhiệt độ hiện tại ở quận Cầu Giấy bao nhiêu độ?", "temperature_query", "district"),
        ],
    },
    {
        "location": "Đống Đa", "location_scope": "district",
        "turn0": "Quận Đống Đa hôm nay có mưa không?",
        "turn0_intent": "rain_query",
        "followups": [
            ("Mưa đến mấy giờ?", "Mưa ở quận Đống Đa đến mấy giờ thì tạnh?", "rain_query", "district"),
            ("Gió thế nào?", "Gió ở quận Đống Đa như thế nào?", "wind_query", "district"),
            ("Ngày mai trời quang không?", "Ngày mai ở quận Đống Đa trời có quang không?", "daily_forecast", "district"),
            ("Thế còn nhiệt độ?", "Nhiệt độ hiện tại ở quận Đống Đa bao nhiêu?", "temperature_query", "district"),
        ],
    },
    {
        "location": "Hoàn Kiếm", "location_scope": "district",
        "turn0": "Hoàn Kiếm nhiệt độ bao nhiêu?",
        "turn0_intent": "temperature_query",
        "followups": [
            ("Tối nay mấy độ?", "Tối nay quận Hoàn Kiếm nhiệt độ bao nhiêu?", "hourly_forecast", "district"),
            ("Ngày mai thế nào?", "Ngày mai ở quận Hoàn Kiếm thời tiết thế nào?", "daily_forecast", "district"),
            ("Dễ chịu không?", "Thời tiết quận Hoàn Kiếm hiện tại có dễ chịu không?", "activity_weather", "district"),
            ("Mặc gì đi?", "Ở quận Hoàn Kiếm hôm nay nên mặc gì khi ra ngoài?", "smalltalk_weather", "district"),
        ],
    },
    {
        "location": "Ba Đình", "location_scope": "district",
        "turn0": "Ba Đình hôm nay tổng quan thời tiết?",
        "turn0_intent": "weather_overview",
        "followups": [
            ("Ngày mai thế nào?", "Thời tiết ngày mai ở quận Ba Đình như thế nào?", "daily_forecast", "district"),
            ("Vậy còn cuối tuần?", "Cuối tuần này ở quận Ba Đình thời tiết ra sao?", "daily_forecast", "district"),
            ("Ở đó có mưa không?", "Quận Ba Đình hôm nay có mưa không?", "rain_query", "district"),
            ("Nơi đó gió thế nào?", "Gió ở quận Ba Đình hiện tại mạnh không?", "wind_query", "district"),
        ],
    },
    {
        "location": "Tây Hồ", "location_scope": "district",
        "turn0": "Chiều nay Tây Hồ mưa không?",
        "turn0_intent": "rain_query",
        "followups": [
            ("Tối nay thế nào?", "Tối nay ở quận Tây Hồ thời tiết ra sao?", "hourly_forecast", "district"),
            ("Ở đó sáng mai thế nào?", "Sáng mai ở quận Tây Hồ trời thế nào?", "hourly_forecast", "district"),
            ("Thế còn gió?", "Gió ở quận Tây Hồ chiều nay mạnh không?", "wind_query", "district"),
            ("Mưa đến mấy giờ tạnh?", "Mưa ở quận Tây Hồ đến mấy giờ thì tạnh?", "rain_query", "district"),
        ],
    },
    {
        "location": "Thanh Xuân", "location_scope": "district",
        "turn0": "Thanh Xuân hiện tại thế nào?",
        "turn0_intent": "current_weather",
        "followups": [
            ("Khu đó UV cao không?", "UV ở quận Thanh Xuân hiện tại có cao không?", "expert_weather_param", "district"),
            ("Thế còn gió mạnh không?", "Gió ở quận Thanh Xuân hiện tại có mạnh không?", "wind_query", "district"),
            ("Có nồm không?", "Quận Thanh Xuân có nồm ẩm không?", "humidity_fog_query", "district"),
            ("Ngày mai dự báo sao?", "Dự báo thời tiết ngày mai ở quận Thanh Xuân?", "daily_forecast", "district"),
        ],
    },
    {
        "location": "Hoàng Mai", "location_scope": "district",
        "turn0": "3 ngày tới Hoàng Mai thế nào?",
        "turn0_intent": "daily_forecast",
        "followups": [
            ("Ngày mai cụ thể?", "Ngày mai ở quận Hoàng Mai thời tiết cụ thể thế nào?", "weather_overview", "district"),
            ("Sáng mai mấy độ?", "Sáng mai ở quận Hoàng Mai nhiệt độ bao nhiêu?", "hourly_forecast", "district"),
            ("Có mưa không?", "Trong 3 ngày tới ở quận Hoàng Mai có mưa không?", "rain_query", "district"),
        ],
    },
    # ─────────────── Pattern: City-level carry-over ───────────────
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Hà Nội hôm nay tổng quan?",
        "turn0_intent": "weather_overview",
        "followups": [
            ("Ngày mai thế nào?", "Hà Nội ngày mai thời tiết như thế nào?", "daily_forecast", "city"),
            ("Mưa không?", "Hà Nội hôm nay có mưa không?", "rain_query", "city"),
            ("Quận nào nóng nhất?", "Hà Nội hôm nay quận nào nóng nhất?", "location_comparison", "city"),
            ("Cuối tuần trời đẹp không?", "Cuối tuần này Hà Nội thời tiết có đẹp không?", "daily_forecast", "city"),
        ],
    },
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Tuần này Hà Nội dự báo thế nào?",
        "turn0_intent": "daily_forecast",
        "followups": [
            ("Nhiệt độ xu hướng?", "Nhiệt độ Hà Nội tuần này có xu hướng như thế nào?", "temperature_query", "city"),
            ("Mưa nhiều không?", "Tuần này Hà Nội mưa nhiều không?", "rain_query", "city"),
            ("Nóng hơn bình thường không?", "Hà Nội tuần này nóng hơn bình thường không?", "seasonal_context", "city"),
        ],
    },
    # ─────────────── Pattern: Time shift ───────────────
    {
        "location": "Cầu Giấy", "location_scope": "district",
        "turn0": "Cầu Giấy bây giờ thế nào?",
        "turn0_intent": "current_weather",
        "followups": [
            ("Tối nay?", "Tối nay ở quận Cầu Giấy thời tiết ra sao?", "hourly_forecast", "district"),
            ("Sáng mai?", "Sáng mai ở quận Cầu Giấy nhiệt độ bao nhiêu?", "hourly_forecast", "district"),
            ("Ngày mai?", "Ngày mai ở quận Cầu Giấy dự báo thế nào?", "daily_forecast", "district"),
            ("Cuối tuần?", "Cuối tuần này ở quận Cầu Giấy thời tiết ra sao?", "daily_forecast", "district"),
        ],
    },
    # ─────────────── Pattern: Ward-level ───────────────
    {
        "location": "Yên Hòa", "location_scope": "ward",
        "turn0": "Phường Yên Hòa thời tiết hiện tại?",
        "turn0_intent": "current_weather",
        "followups": [
            ("UV cao không?", "UV ở phường Yên Hòa hiện tại có cao không?", "expert_weather_param", "ward"),
            ("Tốt nhất đi bộ lúc mấy giờ?", "Ở phường Yên Hòa lúc mấy giờ tốt nhất để đi bộ?", "activity_weather", "ward"),
            ("Ngày mai thế nào?", "Ngày mai ở phường Yên Hòa thời tiết thế nào?", "daily_forecast", "ward"),
        ],
    },
    # ─────────────── Pattern: Activity + context ───────────────
    {
        "location": "Long Biên", "location_scope": "district",
        "turn0": "Long Biên hôm nay thế nào?",
        "turn0_intent": "weather_overview",
        "followups": [
            ("Phù hợp chạy bộ không?", "Ở quận Long Biên hôm nay có phù hợp để đi chạy bộ không?", "activity_weather", "district"),
            ("Lúc mấy giờ tốt nhất?", "Ở quận Long Biên hôm nay lúc mấy giờ là tốt nhất để chạy bộ?", "activity_weather", "district"),
            ("Ở đó ngày mai có mưa không?", "Ngày mai ở quận Long Biên có mưa không?", "rain_query", "district"),
        ],
    },
    # ─────────────── Pattern: Weather alert carry-over ───────────────
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Hôm nay Hà Nội có cảnh báo thời tiết không?",
        "turn0_intent": "weather_alert",
        "followups": [
            ("Mức độ nguy hiểm thế nào?", "Mức độ nguy hiểm của cảnh báo thời tiết Hà Nội hôm nay ra sao?", "weather_alert", "city"),
            ("Ngày mai còn cảnh báo không?", "Ngày mai Hà Nội còn cảnh báo thời tiết không?", "weather_alert", "city"),
            ("Ảnh hưởng đến khu nào?", "Cảnh báo thời tiết hôm nay ảnh hưởng đến khu vực nào ở Hà Nội?", "weather_alert", "city"),
            ("Bao giờ thì qua?", "Cảnh báo thời tiết Hà Nội bao giờ thì kết thúc?", "weather_alert", "city"),
        ],
    },
    {
        "location": "Gia Lâm", "location_scope": "district",
        "turn0": "Quận Gia Lâm có giông bão không?",
        "turn0_intent": "weather_alert",
        "followups": [
            ("Gió mạnh đến mức nào?", "Gió giông ở quận Gia Lâm mạnh đến mức nào?", "weather_alert", "district"),
            ("Có ngập lụt không?", "Quận Gia Lâm có nguy cơ ngập lụt không?", "weather_alert", "district"),
            ("Thế còn ngày mai?", "Ngày mai quận Gia Lâm còn giông không?", "weather_alert", "district"),
        ],
    },
    # ─────────────── Pattern: Historical weather carry-over ───────────────
    {
        "location": "Ba Đình", "location_scope": "district",
        "turn0": "Hôm qua Ba Đình thời tiết thế nào?",
        "turn0_intent": "historical_weather",
        "followups": [
            ("Tuần trước thì sao?", "Tuần trước thời tiết ở quận Ba Đình như thế nào?", "historical_weather", "district"),
            ("Nhiệt độ hôm qua bao nhiêu?", "Nhiệt độ hôm qua ở quận Ba Đình là bao nhiêu độ?", "historical_weather", "district"),
            ("Hôm qua có mưa không?", "Hôm qua ở quận Ba Đình có mưa không?", "historical_weather", "district"),
            ("So với hôm nay thế nào?", "So với hôm qua, thời tiết quận Ba Đình hôm nay có gì khác không?", "seasonal_context", "district"),
        ],
    },
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Tháng trước Hà Nội nóng đến bao nhiêu độ?",
        "turn0_intent": "historical_weather",
        "followups": [
            ("Năm ngoái cùng kỳ thế nào?", "Năm ngoái cùng kỳ này Hà Nội nhiệt độ bao nhiêu?", "historical_weather", "city"),
            ("So với bình thường thế nào?", "Tháng trước Hà Nội nóng hơn bình thường không?", "seasonal_context", "city"),
            ("Ngày nóng nhất là ngày nào?", "Tháng trước Hà Nội ngày nào nóng nhất?", "historical_weather", "city"),
        ],
    },
    # ─────────────── Pattern: Seasonal context carry-over ───────────────
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Năm nay Hà Nội có nóng hơn bình thường không?",
        "turn0_intent": "seasonal_context",
        "followups": [
            ("Bao nhiêu độ hơn?", "Năm nay Hà Nội nóng hơn bình thường bao nhiêu độ?", "seasonal_context", "city"),
            ("Mùa hè năm nay dự báo thế nào?", "Mùa hè năm nay ở Hà Nội được dự báo thế nào?", "seasonal_context", "city"),
            ("Xu hướng tuần tới?", "Xu hướng nhiệt độ Hà Nội tuần tới như thế nào?", "seasonal_context", "city"),
            ("Hôm nay so với trung bình?", "Hôm nay Hà Nội nóng hơn hay mát hơn so với trung bình tháng này?", "seasonal_context", "city"),
        ],
    },
    # ─────────────── Pattern: Location comparison carry-over ───────────────
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Cầu Giấy và Hoàn Kiếm nơi nào mát hơn hôm nay?",
        "turn0_intent": "location_comparison",
        "followups": [
            ("Chênh lệch bao nhiêu độ?", "Nhiệt độ Cầu Giấy và Hoàn Kiếm chênh nhau bao nhiêu độ?", "location_comparison", "district"),
            ("Còn so với Đống Đa?", "So sánh thời tiết Cầu Giấy, Hoàn Kiếm và Đống Đa hôm nay?", "location_comparison", "district"),
            ("Quận nào có gió mạnh nhất?", "Hôm nay quận nào ở Hà Nội có gió mạnh nhất?", "location_comparison", "city"),
            ("Ngày mai đâu mát hơn?", "Ngày mai Cầu Giấy hay Hoàn Kiếm mát hơn?", "location_comparison", "district"),
        ],
    },
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Xếp hạng các quận theo độ ẩm hôm nay?",
        "turn0_intent": "location_comparison",
        "followups": [
            ("Quận nào ẩm nhất?", "Hôm nay quận nào ở Hà Nội có độ ẩm cao nhất?", "location_comparison", "city"),
            ("So sánh ba quận trung tâm?", "So sánh độ ẩm quận Hoàn Kiếm, Ba Đình và Đống Đa hôm nay?", "location_comparison", "district"),
            ("Ngày mai xếp hạng thế nào?", "Ngày mai các quận Hà Nội xếp hạng độ ẩm thế nào?", "location_comparison", "city"),
        ],
    },
    # ─────────────── Pattern: Humidity/fog carry-over ───────────────
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Sáng nay Hà Nội có sương mù không?",
        "turn0_intent": "humidity_fog_query",
        "followups": [
            ("Sương tan lúc mấy giờ?", "Sương mù Hà Nội sáng nay tan lúc mấy giờ?", "humidity_fog_query", "city"),
            ("Có nồm không?", "Hà Nội hôm nay có nồm ẩm không?", "humidity_fog_query", "city"),
            ("Độ ẩm bao nhiêu?", "Độ ẩm Hà Nội sáng nay là bao nhiêu?", "humidity_fog_query", "city"),
            ("Ngày mai còn sương không?", "Sáng mai Hà Nội còn sương mù không?", "humidity_fog_query", "city"),
        ],
    },
    {
        "location": "Đống Đa", "location_scope": "district",
        "turn0": "Quận Đống Đa hôm nay có nồm không?",
        "turn0_intent": "humidity_fog_query",
        "followups": [
            ("Độ ẩm bao nhiêu phần trăm?", "Độ ẩm quận Đống Đa hôm nay là bao nhiêu phần trăm?", "humidity_fog_query", "district"),
            ("Chiều nay còn ẩm không?", "Chiều nay quận Đống Đa còn nồm ẩm không?", "humidity_fog_query", "district"),
            ("Nơi đó có sương sáng sớm không?", "Sáng sớm quận Đống Đa có sương mù không?", "humidity_fog_query", "district"),
        ],
    },
    # ─────────────── Pattern: Expert params carry-over ───────────────
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Chỉ số UV Hà Nội hôm nay bao nhiêu?",
        "turn0_intent": "expert_weather_param",
        "followups": [
            ("Có cao không?", "Chỉ số UV Hà Nội hôm nay có ở mức cao không?", "expert_weather_param", "city"),
            ("Giờ nào UV cao nhất?", "Hôm nay UV ở Hà Nội cao nhất vào lúc mấy giờ?", "expert_weather_param", "city"),
            ("Ngày mai UV thế nào?", "Chỉ số UV Hà Nội ngày mai là bao nhiêu?", "expert_weather_param", "city"),
            ("Áp suất thay đổi không?", "Áp suất khí quyển Hà Nội hôm nay có thay đổi không?", "expert_weather_param", "city"),
        ],
    },
    {
        "location": "Cầu Giấy", "location_scope": "district",
        "turn0": "Áp suất Cầu Giấy đang thay đổi không?",
        "turn0_intent": "expert_weather_param",
        "followups": [
            ("Điểm sương là bao nhiêu?", "Điểm sương ở quận Cầu Giấy hiện tại là bao nhiêu?", "expert_weather_param", "district"),
            ("Tầm nhìn xa không?", "Tầm nhìn xa ở quận Cầu Giấy hiện tại thế nào?", "expert_weather_param", "district"),
            ("Có ảnh hưởng đến sức khỏe không?", "Áp suất thay đổi ở quận Cầu Giấy có ảnh hưởng sức khỏe không?", "expert_weather_param", "district"),
        ],
    },
    # ─────────────── Pattern: Smalltalk/advice carry-over ───────────────
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Hôm nay Hà Nội mặc gì đi làm?",
        "turn0_intent": "smalltalk_weather",
        "followups": [
            ("Có cần mang ô không?", "Hôm nay đi làm ở Hà Nội có cần mang ô không?", "smalltalk_weather", "city"),
            ("Áo khoác có cần không?", "Thời tiết Hà Nội hôm nay có cần mặc áo khoác không?", "smalltalk_weather", "city"),
            ("Ngày mai mặc gì?", "Ngày mai ở Hà Nội nên mặc gì?", "smalltalk_weather", "city"),
            ("Chiều đi về có mưa không?", "Chiều nay đi làm về ở Hà Nội có mưa không?", "rain_query", "city"),
        ],
    },
    {
        "location": "Hoàng Mai", "location_scope": "district",
        "turn0": "Quận Hoàng Mai sáng mai đi học có mưa không?",
        "turn0_intent": "smalltalk_weather",
        "followups": [
            ("Cần mang áo mưa không?", "Sáng mai đi học ở quận Hoàng Mai có cần mang áo mưa không?", "smalltalk_weather", "district"),
            ("Mấy giờ mưa?", "Sáng mai quận Hoàng Mai mấy giờ bắt đầu mưa?", "rain_query", "district"),
            ("Chiều có mưa không?", "Chiều mai quận Hoàng Mai có mưa không?", "rain_query", "district"),
        ],
    },
    # ─────────────── Pattern: Activity weather carry-over ───────────────
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Hôm nay Hà Nội có phù hợp đi picnic không?",
        "turn0_intent": "activity_weather",
        "followups": [
            ("Lúc mấy giờ tốt nhất?", "Hôm nay ở Hà Nội lúc mấy giờ tốt nhất để đi picnic?", "activity_weather", "city"),
            ("Có cần kem chống nắng không?", "Đi picnic ở Hà Nội hôm nay có cần kem chống nắng không?", "activity_weather", "city"),
            ("Ngày mai tốt hơn không?", "Ngày mai ở Hà Nội có điều kiện tốt hơn để đi picnic không?", "activity_weather", "city"),
            ("Có nên đi xe đạp không?", "Hôm nay Hà Nội thời tiết có phù hợp đi xe đạp ngoài trời không?", "activity_weather", "city"),
        ],
    },
    {
        "location": "Tây Hồ", "location_scope": "district",
        "turn0": "Sáng mai chạy bộ quanh Hồ Tây được không?",
        "turn0_intent": "activity_weather",
        "followups": [
            ("Nhiệt độ sáng mai bao nhiêu?", "Sáng mai ở quận Tây Hồ nhiệt độ bao nhiêu?", "temperature_query", "district"),
            ("Mưa không?", "Sáng mai ở quận Tây Hồ có mưa không?", "rain_query", "district"),
            ("UV sáng mai thế nào?", "Chỉ số UV sáng mai ở quận Tây Hồ bao nhiêu?", "expert_weather_param", "district"),
            ("Chiều chạy bộ được không?", "Chiều mai quận Tây Hồ có phù hợp chạy bộ không?", "activity_weather", "district"),
        ],
    },
    # ─────────────── Pattern: current_weather as follow-up (pronoun/short query) ───────────────
    {
        "location": "Cầu Giấy", "location_scope": "district",
        "turn0": "Cầu Giấy tuần này dự báo thế nào?",
        "turn0_intent": "daily_forecast",
        "followups": [
            ("Bây giờ ở đó thế nào?", "Thời tiết hiện tại ở quận Cầu Giấy như thế nào?", "current_weather", "district"),
            ("Còn lúc này trời ra sao?", "Trời ở quận Cầu Giấy lúc này ra sao?", "current_weather", "district"),
            ("Bây giờ nhiệt độ chỗ đó?", "Nhiệt độ hiện tại ở quận Cầu Giấy bao nhiêu?", "temperature_query", "district"),
        ],
    },
    {
        "location": "Hoàng Mai", "location_scope": "district",
        "turn0": "Hoàng Mai hôm nay có mưa không?",
        "turn0_intent": "rain_query",
        "followups": [
            ("Ở đó bây giờ thế nào rồi?", "Thời tiết hiện tại ở quận Hoàng Mai như thế nào?", "current_weather", "district"),
            ("Hiện tại khu đó ra sao?", "Quận Hoàng Mai hiện tại trời ra sao?", "current_weather", "district"),
            ("Còn bây giờ nắng hay mưa?", "Quận Hoàng Mai bây giờ đang nắng hay mưa?", "current_weather", "district"),
        ],
    },
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Hà Nội ngày mai thế nào?",
        "turn0_intent": "daily_forecast",
        "followups": [
            ("Thế còn bây giờ?", "Thời tiết Hà Nội bây giờ như thế nào?", "current_weather", "city"),
            ("Lúc này thành phố ra sao?", "Hà Nội lúc này thời tiết ra sao?", "current_weather", "city"),
            ("Hiện tại trời thế nào?", "Hà Nội hiện tại trời như thế nào?", "current_weather", "city"),
        ],
    },
    {
        "location": "Ba Đình", "location_scope": "district",
        "turn0": "3 ngày tới Ba Đình có mưa không?",
        "turn0_intent": "daily_forecast",
        "followups": [
            ("Ở đó trời hiện giờ thế nào?", "Trời ở quận Ba Đình hiện giờ thế nào?", "current_weather", "district"),
            ("Bây giờ nắng không?", "Quận Ba Đình bây giờ có nắng không?", "current_weather", "district"),
        ],
    },
    {
        "location": "Bắc Từ Liêm", "location_scope": "district",
        "turn0": "Bắc Từ Liêm cuối tuần thế nào?",
        "turn0_intent": "daily_forecast",
        "followups": [
            ("Chỗ đó bây giờ trời ra sao?", "Thời tiết hiện tại ở quận Bắc Từ Liêm ra sao?", "current_weather", "district"),
            ("Lúc này mưa không?", "Quận Bắc Từ Liêm lúc này có mưa không?", "current_weather", "district"),
        ],
    },
    # ─────────────── Pattern: weather_overview as follow-up ───────────────
    {
        "location": "Hoàn Kiếm", "location_scope": "district",
        "turn0": "Hoàn Kiếm chiều nay mưa không?",
        "turn0_intent": "rain_query",
        "followups": [
            ("Tổng quan hôm nay ở đó thế nào?", "Tổng quan thời tiết hôm nay ở quận Hoàn Kiếm như thế nào?", "weather_overview", "district"),
            ("Tóm tắt thời tiết chỗ đó đi?", "Tóm tắt thời tiết hôm nay ở quận Hoàn Kiếm như thế nào?", "weather_overview", "district"),
        ],
    },
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Hà Nội hôm nay nóng không?",
        "turn0_intent": "temperature_query",
        "followups": [
            ("Cho mình tổng quan thời tiết Hà Nội hôm nay đi?", "Tổng quan thời tiết Hà Nội hôm nay như thế nào?", "weather_overview", "city"),
            ("Hôm nay nhìn chung thế nào?", "Nhìn chung thời tiết Hà Nội hôm nay như thế nào?", "weather_overview", "city"),
            ("Tóm lại thời tiết hôm nay ra sao?", "Tóm lại thời tiết Hà Nội hôm nay ra sao?", "weather_overview", "city"),
        ],
    },
    {
        "location": "Long Biên", "location_scope": "district",
        "turn0": "Long Biên ngày mai như thế nào?",
        "turn0_intent": "daily_forecast",
        "followups": [
            ("Hôm nay tổng quan ở đó thế nào?", "Tổng quan thời tiết hôm nay ở quận Long Biên như thế nào?", "weather_overview", "district"),
            ("Nơi đó hôm nay thế nào?", "Thời tiết hôm nay ở quận Long Biên như thế nào?", "weather_overview", "district"),
        ],
    },
    {
        "location": "Hà Đông", "location_scope": "district",
        "turn0": "Hà Đông hôm nay mưa không?",
        "turn0_intent": "rain_query",
        "followups": [
            ("Tổng thể hôm nay ở đó ra sao?", "Tổng thể thời tiết hôm nay ở quận Hà Đông ra sao?", "weather_overview", "district"),
            ("Khu đó hôm nay nhìn chung thế nào?", "Nhìn chung thời tiết quận Hà Đông hôm nay thế nào?", "weather_overview", "district"),
        ],
    },
    # ─────────────── Pattern: Anaphora (pronoun-heavy) ───────────────
    {
        "location": "Hoàn Kiếm", "location_scope": "district",
        "turn0": "Hoàn Kiếm bây giờ mưa không?",
        "turn0_intent": "rain_query",
        "followups": [
            ("Ở đó mưa đến bao giờ?", "Ở quận Hoàn Kiếm mưa đến bao giờ thì tạnh?", "rain_query", "district"),
            ("Rồi sao gió thế nào?", "Gió ở quận Hoàn Kiếm sau khi tạnh mưa thế nào?", "wind_query", "district"),
            ("Chỗ đó tối nay có mưa không?", "Tối nay ở quận Hoàn Kiếm có mưa không?", "hourly_forecast", "district"),
            ("Khu đó UV bao nhiêu?", "UV ở quận Hoàn Kiếm hôm nay bao nhiêu?", "expert_weather_param", "district"),
        ],
    },
    {
        "location": "Đống Đa", "location_scope": "district",
        "turn0": "Đống Đa nhiệt độ bây giờ?",
        "turn0_intent": "temperature_query",
        "followups": [
            ("Chỗ đó có nồm không?", "Quận Đống Đa có nồm ẩm không?", "humidity_fog_query", "district"),
            ("Thế còn áp suất?", "Áp suất ở quận Đống Đa đang thay đổi không?", "expert_weather_param", "district"),
            ("Nơi đó ngày mai thế nào?", "Ngày mai ở quận Đống Đa thời tiết thế nào?", "daily_forecast", "district"),
            ("Vậy còn cuối tuần?", "Cuối tuần này ở quận Đống Đa thời tiết ra sao?", "daily_forecast", "district"),
        ],
    },
    # ─────────────── Pattern: Pronoun-heavy chains ───────────────
    {
        "location": "Nam Từ Liêm", "location_scope": "district",
        "turn0": "Nam Từ Liêm hôm nay ra sao?",
        "turn0_intent": "weather_overview",
        "followups": [
            ("Nơi đó có gió mạnh không?", "Quận Nam Từ Liêm hôm nay có gió mạnh không?", "wind_query", "district"),
            ("Ở đó nóng không?", "Quận Nam Từ Liêm hôm nay có nóng không?", "temperature_query", "district"),
            ("Khu đó chiều có mưa không?", "Chiều nay ở quận Nam Từ Liêm có mưa không?", "rain_query", "district"),
            ("Thế còn sáng sớm mai?", "Sáng sớm mai ở quận Nam Từ Liêm thời tiết thế nào?", "hourly_forecast", "district"),
        ],
    },
    {
        "location": "Gia Lâm", "location_scope": "district",
        "turn0": "Gia Lâm sáng nay thế nào?",
        "turn0_intent": "current_weather",
        "followups": [
            ("Khu đó chiều nay mưa không?", "Chiều nay ở huyện Gia Lâm có mưa không?", "rain_query", "district"),
            ("Nơi đó ngày mai dự báo sao?", "Ngày mai ở huyện Gia Lâm dự báo thế nào?", "daily_forecast", "district"),
            ("Ở đó bây giờ gió thế nào?", "Gió ở huyện Gia Lâm bây giờ như thế nào?", "wind_query", "district"),
            ("Đó có cảnh báo gì không?", "Huyện Gia Lâm hôm nay có cảnh báo thời tiết gì không?", "weather_alert", "district"),
        ],
    },
    {
        "location": "Hai Bà Trưng", "location_scope": "district",
        "turn0": "Hai Bà Trưng hôm nay thời tiết?",
        "turn0_intent": "current_weather",
        "followups": [
            ("Ở đó mặc gì đi làm?", "Ở quận Hai Bà Trưng hôm nay nên mặc gì đi làm?", "smalltalk_weather", "district"),
            ("Bên đó có phù hợp chạy bộ không?", "Quận Hai Bà Trưng hôm nay có phù hợp để chạy bộ không?", "activity_weather", "district"),
            ("Nơi đó chiều nay mưa không?", "Chiều nay ở quận Hai Bà Trưng có mưa không?", "rain_query", "district"),
        ],
    },
    # ─────────────── Pattern: City-level pronoun chains ───────────────
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Hà Nội chiều nay thế nào?",
        "turn0_intent": "current_weather",
        "followups": [
            ("Tối nay ở đây có mưa không?", "Tối nay ở Hà Nội có mưa không?", "rain_query", "city"),
            ("Ở đây ngày mai ra sao?", "Ngày mai ở Hà Nội thời tiết ra sao?", "daily_forecast", "city"),
            ("Thành phố có gió không?", "Hà Nội hôm nay có gió không?", "wind_query", "city"),
            ("Nơi này có cảnh báo không?", "Hà Nội hôm nay có cảnh báo thời tiết không?", "weather_alert", "city"),
        ],
    },
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Hà Nội sáng nay sương mù không?",
        "turn0_intent": "humidity_fog_query",
        "followups": [
            ("Ở đây nồm không?", "Hà Nội hôm nay có nồm ẩm không?", "humidity_fog_query", "city"),
            ("Độ ẩm ở đây bao nhiêu?", "Độ ẩm ở Hà Nội hiện tại bao nhiêu?", "humidity_fog_query", "city"),
            ("Thành phố hôm nay UV cao không?", "Chỉ số UV ở Hà Nội hôm nay có cao không?", "expert_weather_param", "city"),
        ],
    },
    # ─────────────── Pattern: Ward-level context rewrites ───────────────
    {
        "location": "Bồ Đề", "location_scope": "ward",
        "turn0": "Phường Bồ Đề bây giờ thế nào?",
        "turn0_intent": "current_weather",
        "followups": [
            ("Chiều nay ở đó mưa không?", "Chiều nay ở phường Bồ Đề có mưa không?", "rain_query", "ward"),
            ("Ngày mai khu đó thế nào?", "Ngày mai ở phường Bồ Đề thời tiết thế nào?", "daily_forecast", "ward"),
            ("Ở đó có UV cao không?", "UV ở phường Bồ Đề hôm nay có cao không?", "expert_weather_param", "ward"),
        ],
    },
    {
        "location": "Kiến Hưng", "location_scope": "ward",
        "turn0": "Phường Kiến Hưng hôm nay mưa không?",
        "turn0_intent": "rain_query",
        "followups": [
            ("Bên đó gió thế nào?", "Gió ở phường Kiến Hưng hôm nay như thế nào?", "wind_query", "ward"),
            ("Ở đó có nồm không?", "Phường Kiến Hưng hôm nay có nồm ẩm không?", "humidity_fog_query", "ward"),
            ("Ngày mai ở đó thế nào?", "Ngày mai ở phường Kiến Hưng thời tiết thế nào?", "daily_forecast", "ward"),
        ],
    },
    {
        "location": "Xuân Phương", "location_scope": "ward",
        "turn0": "Phường Xuân Phương nhiệt độ bao nhiêu?",
        "turn0_intent": "temperature_query",
        "followups": [
            ("Ở đó trời nắng không?", "Phường Xuân Phương hôm nay có nắng không?", "current_weather", "ward"),
            ("Tổng quan hôm nay ở đó?", "Tổng quan thời tiết hôm nay ở phường Xuân Phương?", "weather_overview", "ward"),
            ("Chỗ đó ngày mai dự báo sao?", "Dự báo thời tiết ngày mai ở phường Xuân Phương?", "daily_forecast", "ward"),
        ],
    },
    # ─────────────── Pattern: 3-turn chains (turn=2 follow-ups) ───────────────
    {
        "location": "Thanh Xuân", "location_scope": "district",
        "turn0": "Thanh Xuân hôm nay thế nào?",
        "turn0_intent": "current_weather",
        "followups": [
            # Turn 2 chain: after rain_query follow-up
            ("Mưa đến tối không?", "Mưa ở quận Thanh Xuân có kéo dài đến tối không?", "rain_query", "district"),
            ("Thế còn ngày mai?", "Ngày mai ở quận Thanh Xuân thời tiết thế nào?", "daily_forecast", "district"),
            ("Bây giờ trời tạnh chưa?", "Quận Thanh Xuân bây giờ trời đã tạnh chưa?", "current_weather", "district"),
            ("Cuối tuần ở đó thế nào?", "Cuối tuần này ở quận Thanh Xuân thời tiết thế nào?", "daily_forecast", "district"),
        ],
    },
    {
        "location": "Đan Phượng", "location_scope": "district",
        "turn0": "Đan Phượng tuần này dự báo sao?",
        "turn0_intent": "daily_forecast",
        "followups": [
            ("Mưa nhiều nhất ngày nào?", "Tuần này huyện Đan Phượng mưa nhiều nhất vào ngày nào?", "rain_query", "district"),
            ("Ngày mai cụ thể ra sao?", "Ngày mai ở huyện Đan Phượng cụ thể thế nào?", "weather_overview", "district"),
            ("Ở đó lúc này thế nào?", "Thời tiết hiện tại ở huyện Đan Phượng như thế nào?", "current_weather", "district"),
        ],
    },
    {
        "location": "Sóc Sơn", "location_scope": "district",
        "turn0": "Sóc Sơn mùa này bất thường không?",
        "turn0_intent": "seasonal_context",
        "followups": [
            ("Hôm nay ở đó thế nào?", "Thời tiết hôm nay ở huyện Sóc Sơn như thế nào?", "current_weather", "district"),
            ("So với hôm qua thế nào?", "Huyện Sóc Sơn hôm nay so với hôm qua thế nào?", "seasonal_context", "district"),
            ("Tuần tới dự báo ra sao?", "Tuần tới ở huyện Sóc Sơn dự báo thế nào?", "daily_forecast", "district"),
            ("Ở đó ngày mai cụ thể?", "Ngày mai ở huyện Sóc Sơn thời tiết cụ thể thế nào?", "weather_overview", "district"),
        ],
    },
    # ─────────────── Pattern: hourly_forecast as turn0 (previously missing) ───────────────
    {
        "location": "Tây Hồ", "location_scope": "district",
        "turn0": "Diễn biến mưa theo giờ hôm nay ở Tây Hồ thế nào?",
        "turn0_intent": "hourly_forecast",
        "followups": [
            ("Thế chiều tối?", "Chiều tối hôm nay ở quận Tây Hồ thời tiết như thế nào?", "hourly_forecast", "district"),
            ("Đêm nay?", "Đêm nay ở quận Tây Hồ dự báo thế nào?", "daily_forecast", "district"),
            ("Gió lúc đó mạnh không?", "Gió ở quận Tây Hồ chiều nay có mạnh không?", "wind_query", "district"),
            ("Ngày mai theo giờ thế nào?", "Ngày mai ở quận Tây Hồ diễn biến theo giờ ra sao?", "hourly_forecast", "district"),
        ],
    },
    {
        "location": "Cầu Giấy", "location_scope": "district",
        "turn0": "Sáng nay Cầu Giấy từng giờ thế nào?",
        "turn0_intent": "hourly_forecast",
        "followups": [
            ("Chiều nay tiếp tục?", "Chiều nay ở quận Cầu Giấy diễn biến thời tiết thế nào?", "hourly_forecast", "district"),
            ("Mưa lúc mấy giờ?", "Sáng nay ở quận Cầu Giấy mưa bắt đầu lúc mấy giờ?", "rain_query", "district"),
            ("UV cao nhất lúc mấy giờ?", "UV ở quận Cầu Giấy hôm nay cao nhất lúc mấy giờ?", "expert_weather_param", "district"),
        ],
    },
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Hà Nội hôm nay theo giờ thời tiết ra sao?",
        "turn0_intent": "hourly_forecast",
        "followups": [
            ("Ở đây tối nay thế nào?", "Tối nay ở Hà Nội thời tiết ra sao?", "hourly_forecast", "city"),
            ("Mưa lúc mấy giờ hết?", "Hôm nay ở Hà Nội mưa đến mấy giờ thì hết?", "rain_query", "city"),
            ("Ngày mai theo giờ?", "Ngày mai ở Hà Nội diễn biến thời tiết theo giờ thế nào?", "hourly_forecast", "city"),
        ],
    },
    # ─────────────── Pattern: wind_query as turn0 (previously missing) ───────────────
    {
        "location": "Gia Lâm", "location_scope": "district",
        "turn0": "Gió ở Gia Lâm hôm nay mạnh không?",
        "turn0_intent": "wind_query",
        "followups": [
            ("Có ảnh hưởng đến việc đi xe không?", "Gió ở huyện Gia Lâm hôm nay có ảnh hưởng đến việc đi xe không?", "activity_weather", "district"),
            ("Thế mưa kèm gió không?", "Hôm nay ở huyện Gia Lâm có mưa kèm gió không?", "rain_query", "district"),
            ("Ngày mai còn gió không?", "Ngày mai ở huyện Gia Lâm còn gió không?", "wind_query", "district"),
            ("Có giông không?", "Huyện Gia Lâm hôm nay có giông bão không?", "weather_alert", "district"),
        ],
    },
    {
        "location": "Hoàng Mai", "location_scope": "district",
        "turn0": "Hoàng Mai sáng nay gió hướng nào?",
        "turn0_intent": "wind_query",
        "followups": [
            ("Tốc độ gió bao nhiêu?", "Tốc độ gió ở quận Hoàng Mai sáng nay bao nhiêu km/h?", "wind_query", "district"),
            ("Chiều nay còn gió không?", "Chiều nay ở quận Hoàng Mai còn gió không?", "wind_query", "district"),
            ("Gió ảnh hưởng đến giao thông không?", "Gió ở quận Hoàng Mai hôm nay có ảnh hưởng giao thông không?", "activity_weather", "district"),
        ],
    },
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Áp suất Hà Nội đang thay đổi thế nào?",
        "turn0_intent": "expert_weather_param",
        "followups": [
            ("Có nguy hiểm không?", "Áp suất thay đổi ở Hà Nội có nguy hiểm không?", "weather_alert", "city"),
            ("Thế độ ẩm?", "Độ ẩm ở Hà Nội hiện tại bao nhiêu?", "humidity_fog_query", "city"),
            ("Điểm sương bao nhiêu?", "Điểm sương ở Hà Nội hiện tại là bao nhiêu?", "expert_weather_param", "city"),
        ],
    },
    # ─────────────── Pattern: More historical_weather chains ───────────────
    {
        "location": "Hoàn Kiếm", "location_scope": "district",
        "turn0": "Hôm qua Hoàn Kiếm nhiệt độ bao nhiêu?",
        "turn0_intent": "historical_weather",
        "followups": [
            ("Ngày kia thì sao?", "Hai ngày trước ở quận Hoàn Kiếm nhiệt độ bao nhiêu?", "historical_weather", "district"),
            ("So với hôm nay thế nào?", "So với hôm qua, hôm nay quận Hoàn Kiếm nóng hơn hay mát hơn?", "seasonal_context", "district"),
            ("Tuần trước như thế nào?", "Tuần trước ở quận Hoàn Kiếm thời tiết như thế nào?", "historical_weather", "district"),
        ],
    },
    {
        "location": "Cầu Giấy", "location_scope": "district",
        "turn0": "Tuần trước Cầu Giấy mưa nhiều không?",
        "turn0_intent": "historical_weather",
        "followups": [
            ("Nhiều hơn tuần này không?", "Tuần trước quận Cầu Giấy mưa nhiều hơn tuần này không?", "seasonal_context", "district"),
            ("Hôm qua thế nào?", "Hôm qua ở quận Cầu Giấy thời tiết thế nào?", "historical_weather", "district"),
            ("Tháng này mưa nhiều không?", "Tháng này quận Cầu Giấy có mưa nhiều không?", "historical_weather", "district"),
        ],
    },
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Hôm qua Hà Nội nóng đến bao nhiêu độ?",
        "turn0_intent": "historical_weather",
        "followups": [
            ("Còn ngày kia?", "Hai hôm trước Hà Nội nhiệt độ cao nhất là bao nhiêu?", "historical_weather", "city"),
            ("Nóng bất thường không?", "Hà Nội những ngày qua nóng bất thường không so với trung bình?", "seasonal_context", "city"),
            ("Hôm nay thế nào?", "Hà Nội hôm nay nhiệt độ thế nào so với hôm qua?", "seasonal_context", "city"),
        ],
    },
    # ─────────────── Pattern: More weather_alert chains ───────────────
    {
        "location": "Đông Anh", "location_scope": "district",
        "turn0": "Huyện Đông Anh hôm nay có cảnh báo không?",
        "turn0_intent": "weather_alert",
        "followups": [
            ("Mức độ thế nào?", "Cảnh báo thời tiết hôm nay ở huyện Đông Anh mức độ ra sao?", "weather_alert", "district"),
            ("Ngày mai còn không?", "Ngày mai huyện Đông Anh còn cảnh báo thời tiết không?", "weather_alert", "district"),
            ("Ảnh hưởng đến đi lại không?", "Cảnh báo thời tiết hôm nay ở huyện Đông Anh có ảnh hưởng đến đi lại không?", "activity_weather", "district"),
        ],
    },
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Hà Nội có bão sắp đến không?",
        "turn0_intent": "weather_alert",
        "followups": [
            ("Bao giờ ảnh hưởng đến thành phố?", "Bão ảnh hưởng đến Hà Nội vào khoảng thời gian nào?", "weather_alert", "city"),
            ("Cần chuẩn bị gì?", "Trước khi bão đến Hà Nội người dân cần chuẩn bị gì?", "weather_alert", "city"),
            ("Mưa lớn đến mức nào?", "Bão đổ bộ Hà Nội sẽ gây mưa lớn đến mức nào?", "weather_alert", "city"),
            ("Ngập lụt nguy cơ không?", "Hà Nội có nguy cơ ngập lụt khi bão đến không?", "weather_alert", "city"),
        ],
    },
    # ─────────────── Pattern: More location_comparison chains ───────────────
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Long Biên và Tây Hồ hôm nay nơi nào mát hơn?",
        "turn0_intent": "location_comparison",
        "followups": [
            ("Còn Cầu Giấy?", "So sánh thời tiết Long Biên, Tây Hồ và Cầu Giấy hôm nay?", "location_comparison", "district"),
            ("Quận ven hồ thường mát hơn không?", "Các quận ven hồ như Tây Hồ thường mát hơn trung tâm không?", "seasonal_context", "city"),
            ("Ngày mai đâu mát hơn?", "Ngày mai Long Biên hay Tây Hồ mát hơn?", "location_comparison", "district"),
        ],
    },
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Quận nào Hà Nội mưa nhiều nhất hôm nay?",
        "turn0_intent": "location_comparison",
        "followups": [
            ("Quận đó mưa từ mấy giờ?", "Quận mưa nhiều nhất hôm nay ở Hà Nội mưa từ mấy giờ?", "rain_query", "district"),
            ("Ngày mai xếp hạng thay đổi không?", "Ngày mai quận nào ở Hà Nội mưa nhiều nhất?", "location_comparison", "city"),
            ("So với tuần trước?", "So với tuần trước, hôm nay quận nào mưa nhiều nhất Hà Nội?", "location_comparison", "city"),
        ],
    },
    # ─────────────── Pattern: More ward-level seeds ───────────────
    {
        "location": "Đông Ngạc", "location_scope": "ward",
        "turn0": "Phường Đông Ngạc hôm nay có mưa không?",
        "turn0_intent": "rain_query",
        "followups": [
            ("Mưa đến mấy giờ?", "Mưa ở phường Đông Ngạc hôm nay đến mấy giờ thì tạnh?", "rain_query", "ward"),
            ("Ngày mai khu đó thế nào?", "Ngày mai ở phường Đông Ngạc thời tiết thế nào?", "daily_forecast", "ward"),
            ("Gió thế nào?", "Gió ở phường Đông Ngạc hôm nay mạnh không?", "wind_query", "ward"),
        ],
    },
    {
        "location": "Tương Mai", "location_scope": "ward",
        "turn0": "Phường Tương Mai nhiệt độ bây giờ?",
        "turn0_intent": "temperature_query",
        "followups": [
            ("Ở đó nóng không?", "Phường Tương Mai hôm nay có nóng không?", "temperature_query", "ward"),
            ("Chiều nay thế nào?", "Chiều nay ở phường Tương Mai thời tiết thế nào?", "hourly_forecast", "ward"),
            ("Tổng quan hôm nay?", "Tổng quan thời tiết hôm nay ở phường Tương Mai?", "weather_overview", "ward"),
        ],
    },
    {
        "location": "Láng", "location_scope": "ward",
        "turn0": "Phường Láng hôm nay tổng quan?",
        "turn0_intent": "weather_overview",
        "followups": [
            ("Chiều nay mưa không?", "Chiều nay ở phường Láng có mưa không?", "rain_query", "ward"),
            ("Có phù hợp chạy bộ sáng mai không?", "Sáng mai ở phường Láng có phù hợp chạy bộ không?", "activity_weather", "ward"),
            ("Ngày mai ở đó thế nào?", "Ngày mai ở phường Láng thời tiết thế nào?", "daily_forecast", "ward"),
        ],
    },
    {
        "location": "Vĩnh Hưng", "location_scope": "ward",
        "turn0": "Phường Vĩnh Hưng cuối tuần thế nào?",
        "turn0_intent": "daily_forecast",
        "followups": [
            ("Thứ bảy hay chủ nhật tốt hơn?", "Thứ bảy hay chủ nhật tuần này ở phường Vĩnh Hưng thời tiết tốt hơn?", "daily_forecast", "ward"),
            ("Bây giờ ở đó thế nào?", "Thời tiết hiện tại ở phường Vĩnh Hưng như thế nào?", "current_weather", "ward"),
            ("Có mưa cuối tuần không?", "Cuối tuần này ở phường Vĩnh Hưng có mưa không?", "rain_query", "ward"),
        ],
    },
    {
        "location": "Thượng Cát", "location_scope": "ward",
        "turn0": "Phường Thượng Cát sáng nay sương mù không?",
        "turn0_intent": "humidity_fog_query",
        "followups": [
            ("Sương tan lúc mấy giờ?", "Sương mù ở phường Thượng Cát sáng nay tan lúc mấy giờ?", "humidity_fog_query", "ward"),
            ("Độ ẩm bao nhiêu?", "Độ ẩm ở phường Thượng Cát sáng nay là bao nhiêu?", "humidity_fog_query", "ward"),
            ("Chiều nay thế nào?", "Chiều nay ở phường Thượng Cát thời tiết thế nào?", "hourly_forecast", "ward"),
        ],
    },
    {
        "location": "Phú Diễn", "location_scope": "ward",
        "turn0": "Phường Phú Diễn có cảnh báo thời tiết không?",
        "turn0_intent": "weather_alert",
        "followups": [
            ("Mức độ nguy hiểm?", "Cảnh báo thời tiết ở phường Phú Diễn hôm nay mức độ ra sao?", "weather_alert", "ward"),
            ("Bao giờ qua?", "Cảnh báo thời tiết ở phường Phú Diễn bao giờ kết thúc?", "weather_alert", "ward"),
            ("Có ảnh hưởng giao thông không?", "Thời tiết ở phường Phú Diễn hôm nay có ảnh hưởng giao thông không?", "activity_weather", "ward"),
        ],
    },
    # ─────────────── Pattern: seasonal_context chains ───────────────
    {
        "location": "Hà Nội", "location_scope": "city",
        "turn0": "Tháng này Hà Nội so với năm ngoái cùng kỳ thế nào?",
        "turn0_intent": "seasonal_context",
        "followups": [
            ("Năm nay nóng hơn hay mát hơn?", "Năm nay Hà Nội cùng thời điểm này nóng hơn hay mát hơn năm ngoái?", "seasonal_context", "city"),
            ("Xu hướng tuần tới?", "Xu hướng thời tiết Hà Nội tuần tới như thế nào?", "seasonal_context", "city"),
            ("Bao giờ hết nóng?", "Hà Nội bao giờ sẽ bớt nóng so với hiện tại?", "seasonal_context", "city"),
        ],
    },
    {
        "location": "Hoàng Mai", "location_scope": "district",
        "turn0": "Hoàng Mai hôm nay so với hôm qua thế nào?",
        "turn0_intent": "seasonal_context",
        "followups": [
            ("Nhiệt độ chênh bao nhiêu?", "Nhiệt độ quận Hoàng Mai hôm nay chênh bao nhiêu so với hôm qua?", "seasonal_context", "district"),
            ("Hôm nay mát hơn không?", "Quận Hoàng Mai hôm nay có mát hơn hôm qua không?", "seasonal_context", "district"),
            ("Ngày mai dự báo thế nào?", "Ngày mai ở quận Hoàng Mai dự báo thế nào?", "daily_forecast", "district"),
        ],
    },
]

# ── Confidence values for synthetic data (realistic distribution) ──
_CONFIDENCE_VALUES = [0.88, 0.91, 0.93, 0.95, 0.87, 0.92, 0.89, 0.94, 0.96, 0.90]

# ── Per-intent confidence baselines (from eval: 91.74% overall, per-intent varies) ──
_INTENT_CONFIDENCE = {
    "current_weather":    0.93,
    "smalltalk_weather":  0.94,
    "weather_overview":   0.91,
    "daily_forecast":     0.92,
    "hourly_forecast":    0.89,
    "rain_query":         0.92,
    "temperature_query":  0.83,  # harder (overlaps with current_weather)
    "wind_query":         0.90,
    "humidity_fog_query": 0.88,
    "historical_weather": 0.91,
    "location_comparison":0.90,
    "activity_weather":   0.89,
    "expert_weather_param":0.87,
    "weather_alert":      0.84,  # harder (overlaps with rain_query)
    "seasonal_context":   0.88,
}


def generate_routing_examples(existing_path: str) -> list[dict]:
    """Load existing routing data and convert to unified multi-task format (context=null).

    Handles two input formats:
    - messages format: {"messages": [{"role":"system",...},{"role":"user","content":"..."},
                                     {"role":"assistant","content":"{\"intent\":...,\"scope\":...}"}]}
    - input/output format: {"input": "...", "output": {"intent": ..., "scope": ...}}

    Output format (unified):
      {"input": "...", "context": null,
       "output": {"intent": "...", "scope": "...", "confidence": 0.92}}
    """
    examples = []
    path = Path(existing_path)
    if not path.exists():
        print(f"Warning: {existing_path} not found, skipping")
        return examples

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)

                # ── messages format (train_clean.jsonl) ──
                if "messages" in rec:
                    msgs = rec["messages"]
                    user_msg = next((m for m in msgs if m["role"] == "user"), None)
                    asst_msg = next((m for m in msgs if m["role"] == "assistant"), None)
                    if not (user_msg and asst_msg):
                        continue
                    out = json.loads(asst_msg["content"])
                    intent = out.get("intent", "current_weather")
                    scope = out.get("scope", "city")
                    # Assign calibrated confidence with small jitter
                    base_conf = _INTENT_CONFIDENCE.get(intent, 0.90)
                    conf = round(min(0.99, max(0.75, base_conf + random.uniform(-0.03, 0.03))), 2)
                    examples.append({
                        "input": user_msg["content"],
                        "context": None,
                        "output": {"intent": intent, "scope": scope, "confidence": conf},
                    })

                # ── input/output format (already converted) ──
                elif "input" in rec and "output" in rec:
                    if "context" not in rec:
                        rec["context"] = None
                    # Backfill confidence if missing
                    if "confidence" not in rec.get("output", {}):
                        intent = rec["output"].get("intent", "current_weather")
                        base_conf = _INTENT_CONFIDENCE.get(intent, 0.90)
                        rec["output"]["confidence"] = round(
                            min(0.99, max(0.75, base_conf + random.uniform(-0.03, 0.03))), 2
                        )
                    examples.append(rec)

            except Exception:
                continue

    print(f"Loaded {len(examples)} routing examples from {existing_path}")
    return examples


def generate_rewrite_examples() -> list[dict]:
    """Generate contextual rewriting examples from seed templates."""
    examples = []

    for template in _SEED_TEMPLATES:
        location = template["location"]
        scope = template["location_scope"]
        turn0_intent = template["turn0_intent"]

        context = {
            "location": location,
            "intent": turn0_intent,
            "turn": 1,
        }

        for ambiguous, rewritten, intent, followup_scope in template["followups"]:
            conf = random.choice(_CONFIDENCE_VALUES)
            examples.append({
                "input": ambiguous,
                "context": context,
                "output": {
                    "intent": intent,
                    "scope": followup_scope,
                    "confidence": conf,
                    "rewritten_query": rewritten,
                },
            })

            # Also add a 3rd-turn example where context intent changes
            context_turn2 = {
                "location": location,
                "intent": intent,
                "turn": 2,
            }
            # Generate a self-referential follow-up for turn 3
            if intent in ("daily_forecast", "rain_query", "temperature_query"):
                if intent == "daily_forecast":
                    turn3_q = "Ngày kia thì sao?"
                    turn3_rw = f"Dự báo thời tiết ngày kia ở {'quận' if 'district' in followup_scope else 'phường'} {location}?"
                    turn3_intent = "daily_forecast"
                elif intent == "rain_query":
                    turn3_q = "Còn ngày mai?"
                    turn3_rw = f"Ngày mai ở {'quận' if 'district' in followup_scope else 'phường'} {location} có mưa không?"
                    turn3_intent = "rain_query"
                else:
                    turn3_q = "Chiều nay mấy độ?"
                    turn3_rw = f"Chiều nay ở {'quận' if 'district' in followup_scope else 'phường'} {location} nhiệt độ bao nhiêu?"
                    turn3_intent = "temperature_query"

                examples.append({
                    "input": turn3_q,
                    "context": context_turn2,
                    "output": {
                        "intent": turn3_intent,
                        "scope": followup_scope,
                        "confidence": random.choice(_CONFIDENCE_VALUES),
                        "rewritten_query": turn3_rw,
                    },
                })

    print(f"Generated {len(examples)} rewrite examples from {len(_SEED_TEMPLATES)} seed templates")
    return examples


def generate_no_rewrite_examples() -> list[dict]:
    """Generate examples where context is available BUT query is already self-contained.

    Model should NOT produce rewritten_query in these cases.
    Covers: explicit location, full query, different location than context, no ambiguity.
    """
    examples = []

    # (query, intent, scope, confidence)
    # Three categories:
    #   A. Explicit NEW location (different from whatever prev context was)
    #   B. Full standalone query (time-explicit, no pronoun/ambiguity)
    #   C. Adversarial: prev context exists but query is clearly self-contained
    standalone_queries = [
        # ── A. current_weather — explicit location ──
        ("Quận Cầu Giấy hiện tại thời tiết thế nào?", "current_weather", "district", 0.94),
        ("Hoàng Mai bây giờ ra sao?", "current_weather", "district", 0.93),
        ("Long Biên lúc này thế nào?", "current_weather", "district", 0.92),
        ("Quận Hà Đông hiện tại trời thế nào?", "current_weather", "district", 0.93),
        ("Hai Bà Trưng bây giờ ra sao?", "current_weather", "district", 0.91),
        ("Phường Yên Hòa thời tiết hiện tại?", "current_weather", "ward", 0.91),
        ("Xã Bát Tràng bây giờ ra sao?", "current_weather", "ward", 0.89),
        ("Phường Phú Thượng hiện tại thế nào?", "current_weather", "ward", 0.90),
        ("Hà Nội lúc này thế nào?", "current_weather", "city", 0.95),
        ("Toàn thành phố Hà Nội hiện tại?", "current_weather", "city", 0.94),
        # ── A. rain_query ──
        ("Đống Đa ngày mai có mưa không?", "rain_query", "district", 0.91),
        ("Tây Hồ chiều nay mưa không?", "rain_query", "district", 0.92),
        ("Bắc Từ Liêm cuối tuần có mưa không?", "rain_query", "district", 0.88),
        ("Quận Hà Đông sáng mai mưa không?", "rain_query", "district", 0.91),
        ("Quận Hai Bà Trưng ngày mai mưa không?", "rain_query", "district", 0.90),
        ("Chiều nay quận Long Biên mưa không?", "rain_query", "district", 0.91),
        ("Phường Vĩnh Tuy tuần này mưa không?", "rain_query", "ward", 0.88),
        ("Xã Kim Anh hôm nay có mưa không?", "rain_query", "ward", 0.87),
        ("Phường Định Công ngày mai mưa không?", "rain_query", "ward", 0.88),
        ("Hà Nội tuần này mưa nhiều không?", "rain_query", "city", 0.94),
        ("Sáng mai toàn Hà Nội mưa không?", "rain_query", "city", 0.93),
        ("Thị xã Sơn Tây ngày mai có mưa không?", "rain_query", "district", 0.87),
        # ── A. daily_forecast ──
        ("Ba Đình tuần này thế nào?", "daily_forecast", "district", 0.89),
        ("Thanh Xuân ngày mai dự báo?", "daily_forecast", "district", 0.89),
        ("Huyện Đông Anh cuối tuần thế nào?", "daily_forecast", "district", 0.88),
        ("Tuần tới quận Đống Đa thế nào?", "daily_forecast", "district", 0.88),
        ("Huyện Ba Vì cuối tuần dự báo?", "daily_forecast", "district", 0.86),
        ("Ngày mai toàn thành phố thế nào?", "daily_forecast", "city", 0.88),
        ("Hà Nội cuối tuần dự báo thế nào?", "daily_forecast", "city", 0.92),
        ("Sáng mai toàn Hà Nội dự báo thế nào?", "daily_forecast", "city", 0.93),
        ("Phường Yên Sở ngày mai dự báo?", "daily_forecast", "ward", 0.87),
        ("Phường Vĩnh Hưng cuối tuần có gì đặc biệt không?", "daily_forecast", "ward", 0.86),
        ("Xã Đan Phượng tuần tới thế nào?", "daily_forecast", "ward", 0.85),
        # ── A. temperature_query ──
        ("Hoàn Kiếm nhiệt độ hiện tại?", "temperature_query", "district", 0.93),
        ("Long Biên bây giờ nhiệt độ?", "temperature_query", "district", 0.91),
        ("Quận Hoàng Mai nhiệt độ ngày mai?", "temperature_query", "district", 0.87),
        ("Tối nay Hà Nội nhiệt độ bao nhiêu?", "temperature_query", "city", 0.90),
        ("Hà Nội ngày mai nhiệt độ bao nhiêu?", "temperature_query", "city", 0.92),
        ("Phường Tây Mỗ hôm nay nóng không?", "temperature_query", "ward", 0.87),
        ("Phường Láng chiều nay mấy độ?", "temperature_query", "ward", 0.88),
        ("Xã Quốc Oai hôm nay nhiệt độ bao nhiêu?", "temperature_query", "ward", 0.86),
        # ── A. wind_query ──
        ("Toàn Hà Nội hôm nay gió thế nào?", "wind_query", "city", 0.91),
        ("Quận Cầu Giấy hôm nay gió mạnh không?", "wind_query", "district", 0.90),
        ("Phường Khương Đình hôm nay gió thế nào?", "wind_query", "ward", 0.88),
        ("Xã Đông Anh hôm nay gió mạnh không?", "wind_query", "ward", 0.86),
        ("Gia Lâm sáng mai gió như thế nào?", "wind_query", "district", 0.89),
        ("Huyện Mê Linh gió hôm nay mạnh không?", "wind_query", "district", 0.87),
        # ── A. weather_overview ──
        ("Quận Cầu Giấy hôm nay tổng quan?", "weather_overview", "district", 0.93),
        ("Huyện Sóc Sơn hôm nay tổng quan?", "weather_overview", "district", 0.90),
        ("Hà Nội hôm nay tổng quan?", "weather_overview", "city", 0.96),
        ("Hà Nội ngày mai nhìn chung thế nào?", "weather_overview", "city", 0.94),
        ("Phường Bạch Mai hôm nay tổng quan?", "weather_overview", "ward", 0.88),
        # ── A. humidity_fog_query ──
        ("Quận Tây Hồ sáng nay có sương mù không?", "humidity_fog_query", "district", 0.91),
        ("Phường Giảng Võ độ ẩm hiện tại?", "humidity_fog_query", "ward", 0.89),
        ("Quận Đống Đa hôm nay độ ẩm bao nhiêu?", "humidity_fog_query", "district", 0.90),
        ("Hà Nội sáng nay có nồm không?", "humidity_fog_query", "city", 0.92),
        ("Phường Thượng Cát sáng nay sương mù không?", "humidity_fog_query", "ward", 0.88),
        ("Huyện Hoài Đức hôm nay có sương không?", "humidity_fog_query", "district", 0.87),
        # ── A. historical_weather ──
        ("Quận Cầu Giấy hôm qua thời tiết thế nào?", "historical_weather", "district", 0.90),
        ("Hôm qua Ba Đình nhiệt độ bao nhiêu?", "historical_weather", "district", 0.89),
        ("Tuần trước Hà Nội mưa nhiều không?", "historical_weather", "city", 0.91),
        ("Huyện Thanh Oai hôm qua thế nào?", "historical_weather", "district", 0.87),
        ("Phường Kim Liên tuần trước ra sao?", "historical_weather", "ward", 0.86),
        ("Hôm qua toàn Hà Nội nhiệt độ cao nhất bao nhiêu?", "historical_weather", "city", 0.90),
        # ── A. location_comparison ──
        ("Đống Đa và Hoàn Kiếm nơi nào mát hơn?", "location_comparison", "district", 0.91),
        ("So sánh Cầu Giấy và Đống Đa hôm nay?", "location_comparison", "district", 0.88),
        ("Hà Nội hôm nay quận nào nóng nhất?", "location_comparison", "city", 0.93),
        ("Long Biên và Gia Lâm hôm nay nơi nào mưa nhiều hơn?", "location_comparison", "district", 0.89),
        ("Quận nào gần Hồ Tây mát nhất hôm nay?", "location_comparison", "city", 0.87),
        # ── A. activity_weather ──
        ("Sáng mai quận Thanh Xuân đi xe máy được không?", "activity_weather", "district", 0.89),
        ("Hà Nội hôm nay có phù hợp đi leo núi không?", "activity_weather", "city", 0.88),
        ("Quận Tây Hồ sáng mai chạy bộ được không?", "activity_weather", "district", 0.90),
        ("Phường Hoàn Kiếm ngày mai picnic được không?", "activity_weather", "ward", 0.87),
        ("Huyện Ba Vì cuối tuần du lịch được không?", "activity_weather", "district", 0.86),
        ("Long Biên hôm nay đi dã ngoại được không?", "activity_weather", "district", 0.88),
        # ── A. expert_weather_param ──
        ("Áp suất khí quyển Hà Nội hiện tại?", "expert_weather_param", "city", 0.87),
        ("UV Hà Nội hôm nay bao nhiêu?", "expert_weather_param", "city", 0.88),
        ("Quận Cầu Giấy điểm sương hiện tại?", "expert_weather_param", "district", 0.86),
        ("Tầm nhìn xa ở sân bay Nội Bài hiện tại?", "expert_weather_param", "district", 0.85),
        ("Phường Nghĩa Đô áp suất hôm nay thế nào?", "expert_weather_param", "ward", 0.84),
        # ── A. weather_alert ──
        ("Hôm nay Hoàn Kiếm có giông không?", "weather_alert", "district", 0.89),
        ("Huyện Thanh Trì có cảnh báo thời tiết không?", "weather_alert", "district", 0.87),
        ("Hà Nội hôm nay có cảnh báo lũ lụt không?", "weather_alert", "city", 0.90),
        ("Quận Gia Lâm có nguy cơ ngập không?", "weather_alert", "district", 0.88),
        ("Huyện Đông Anh hôm nay cảnh báo gì không?", "weather_alert", "district", 0.87),
        ("Phường Phú Diễn hôm nay có nguy hiểm không?", "weather_alert", "ward", 0.86),
        # ── A. seasonal_context ──
        ("Hà Nội hôm nay có gì bất thường không?", "seasonal_context", "city", 0.88),
        ("Quận Đống Đa năm nay mùa đông có lạnh không?", "seasonal_context", "district", 0.86),
        ("Hà Nội tháng này so với trung bình thế nào?", "seasonal_context", "city", 0.89),
        ("Huyện Hoài Đức năm nay mưa nhiều hơn không?", "seasonal_context", "district", 0.85),
        ("Phường Yên Hòa tháng này có gì bất thường?", "seasonal_context", "ward", 0.84),
        # ── A. smalltalk_weather ──
        ("Hà Nội hôm nay mặc gì đi làm?", "smalltalk_weather", "city", 0.94),
        ("Quận Hoàn Kiếm hôm nay đi đâu mặc gì?", "smalltalk_weather", "district", 0.92),
        ("Cảm ơn bạn đã tư vấn!", "smalltalk_weather", "city", 0.96),
        ("Thời tiết Đà Lạt thế nào?", "smalltalk_weather", "city", 0.95),  # ngoài Hà Nội
        ("Bạn có thể giúp tôi không?", "smalltalk_weather", "city", 0.95),
        # ── B. Adversarial: context exists but query fully self-contained ──
        # User explicitly names a new location different from context
        ("Quận Nam Từ Liêm hôm nay ra sao?", "current_weather", "district", 0.93),
        ("Huyện Chương Mỹ ngày mai dự báo?", "daily_forecast", "district", 0.87),
        ("Xã Hương Sơn hôm nay thế nào?", "current_weather", "ward", 0.88),
        ("Phường Hà Đông bây giờ nhiệt độ bao nhiêu?", "temperature_query", "ward", 0.89),
        ("Mỹ Đức hôm nay thế nào?", "current_weather", "district", 0.87),
        ("Quận Bắc Từ Liêm tối nay dự báo?", "hourly_forecast", "district", 0.88),
        ("Huyện Ứng Hòa cuối tuần thế nào?", "daily_forecast", "district", 0.86),
        ("Phường Tùng Thiện hôm nay mưa không?", "rain_query", "ward", 0.87),
        # User switches to city-level from district context
        ("Toàn Hà Nội ngày mai thế nào?", "daily_forecast", "city", 0.93),
        ("Hà Nội tổng quan hôm nay?", "weather_overview", "city", 0.95),
        # User explicitly provides time-specific query (no pronoun ambiguity)
        ("Sáng ngày mai quận Đống Đa thế nào?", "hourly_forecast", "district", 0.89),
        ("Tối nay quận Tây Hồ dự báo ra sao?", "hourly_forecast", "district", 0.88),
        ("Hà Nội lúc 15h hôm nay thế nào?", "hourly_forecast", "city", 0.91),
        ("Sáng mai 6-9h quận Hoàng Mai dự báo?", "hourly_forecast", "district", 0.88),
        ("Huyện Đan Phượng tuần tới từng ngày?", "daily_forecast", "district", 0.86),
        # User asks greetings/out-of-scope (no rewrite ever needed)
        ("Xin chào!", "smalltalk_weather", "city", 0.97),
        ("Thời tiết TP.HCM hôm nay thế nào?", "smalltalk_weather", "city", 0.95),
        ("Bạn có thể dự báo thời tiết Đà Nẵng không?", "smalltalk_weather", "city", 0.94),
        # ── C. More ward-level adversarial ──
        ("Phường Xuân Đỉnh hôm nay nhiệt độ?", "temperature_query", "ward", 0.88),
        ("Xã Mê Linh hôm nay có mưa không?", "rain_query", "ward", 0.87),
        ("Phường Lĩnh Nam gió sáng nay?", "wind_query", "ward", 0.86),
        ("Phường Vĩnh Tuy bây giờ thế nào?", "current_weather", "ward", 0.89),
        ("Xã Yên Bài cuối tuần dự báo?", "daily_forecast", "ward", 0.85),
        ("Phường Tây Hồ hôm nay sương mù không?", "humidity_fog_query", "ward", 0.87),
        ("Xã Hòa Lạc hôm nay thế nào?", "current_weather", "ward", 0.86),
        ("Phường Đại Mỗ ngày mai mưa không?", "rain_query", "ward", 0.86),
        # ── D. Intent switches (user changes topic completely, full info provided) ──
        # After weather query → now asks historical
        ("Hôm qua quận Đống Đa thời tiết thế nào?", "historical_weather", "district", 0.90),
        ("Tuần trước quận Cầu Giấy ra sao?", "historical_weather", "district", 0.88),
        # After daily forecast → now asks expert param
        ("UV ở Hà Nội hôm nay cao không?", "expert_weather_param", "city", 0.88),
        ("Áp suất quận Ba Đình hiện tại bao nhiêu?", "expert_weather_param", "district", 0.86),
        # After rain query → now asks activity
        ("Hà Nội hôm nay đi chạy bộ được không?", "activity_weather", "city", 0.89),
        ("Quận Hoàn Kiếm sáng mai picnic được không?", "activity_weather", "district", 0.88),
        # After current weather → now asks seasonal comparison
        ("Hà Nội năm nay so với năm ngoái?", "seasonal_context", "city", 0.88),
        ("Tháng này Cầu Giấy có gì bất thường không?", "seasonal_context", "district", 0.86),
        # After wind → now asks location comparison
        ("Cầu Giấy hay Đống Đa hôm nay mát hơn?", "location_comparison", "district", 0.90),
        ("Hà Nội hôm nay quận nào ít mưa nhất?", "location_comparison", "city", 0.89),
        # ── E. Multi-turn no-rewrite chains: standalone queries in turns 2,3 ──
        ("Phường Nghĩa Đô hôm nay tổng quan?", "weather_overview", "ward", 0.89),
        ("Quận Bắc Từ Liêm ngày mai thế nào?", "daily_forecast", "district", 0.89),
        ("Huyện Thanh Trì cuối tuần dự báo?", "daily_forecast", "district", 0.87),
        ("Quận Long Biên nhiệt độ sáng mai?", "temperature_query", "district", 0.89),
        ("Hà Nội tuần tới thế nào?", "daily_forecast", "city", 0.93),
        ("Phường Hoàng Liệt hôm nay cảnh báo gì không?", "weather_alert", "ward", 0.86),
        ("Xã Liên Minh cuối tuần có mưa không?", "rain_query", "ward", 0.85),
        ("Quận Nam Từ Liêm gió hôm nay mạnh không?", "wind_query", "district", 0.88),
    ]

    # Diverse prev_contexts — pairs (location, intent, turn) to simulate active conversation
    prev_contexts = [
        {"location": "Cầu Giấy",    "intent": "current_weather",    "turn": 1},
        {"location": "Hoàn Kiếm",   "intent": "rain_query",          "turn": 2},
        {"location": "Hà Nội",      "intent": "weather_overview",    "turn": 1},
        {"location": "Thanh Xuân",  "intent": "daily_forecast",      "turn": 2},
        {"location": "Tây Hồ",      "intent": "temperature_query",   "turn": 1},
        {"location": "Ba Đình",     "intent": "wind_query",          "turn": 2},
        {"location": "Đống Đa",     "intent": "weather_overview",    "turn": 1},
        {"location": "Hoàng Mai",   "intent": "daily_forecast",      "turn": 1},
        {"location": "Long Biên",   "intent": "rain_query",          "turn": 2},
        {"location": "Yên Hòa",     "intent": "current_weather",     "turn": 1},
        {"location": "Hà Nội",      "intent": "daily_forecast",      "turn": 3},
        {"location": "Gia Lâm",     "intent": "weather_alert",       "turn": 1},
        {"location": "Sóc Sơn",     "intent": "seasonal_context",    "turn": 2},
        {"location": "Đống Đa",     "intent": "temperature_query",   "turn": 1},
        {"location": "Hà Nội",      "intent": "location_comparison", "turn": 2},
    ]

    for query, intent, scope, conf in standalone_queries:
        ctx = random.choice(prev_contexts)
        examples.append({
            "input": query,
            "context": ctx,
            "output": {
                "intent": intent,
                "scope": scope,
                "confidence": conf,
                # No rewritten_query — query is already standalone
            },
        })

    print(f"Generated {len(examples)} no-rewrite examples (standalone queries with context)")
    return examples


def augment_with_llm(rewrite_examples: list[dict], no_rewrite_examples: list[dict],
                     n_augments: int = 400) -> list[dict]:
    """Use GPT-4o-mini to generate additional variations.

    Focuses augmentation on rewrite examples (paraphrase ambiguous query only).
    Augments no-rewrite examples separately to preserve the no-rewrite signal.
    """
    try:
        from openai import OpenAI
        client = OpenAI(base_url=os.getenv("API_BASE"), api_key=os.getenv("API_KEY"))
    except Exception as e:
        print(f"LLM augmentation skipped: {e}")
        return []

    REPHRASE_PROMPT = """Bạn là người dùng chatbot thời tiết Hà Nội. Hãy viết lại câu hỏi sau theo cách tự nhiên, ngắn gọn khác (giữ ý nghĩa, giữ độ mơ hồ nếu câu gốc mơ hồ):
Câu gốc: {query}
Viết lại (1 câu, tiếng Việt):"""

    augmented = []

    # ── Augment rewrite examples (80% of budget) ──
    n_rw = int(n_augments * 0.80)
    rw_samples = random.sample(rewrite_examples, min(n_rw, len(rewrite_examples)))

    for ex in rw_samples:
        try:
            resp = client.chat.completions.create(
                model=os.getenv("MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": REPHRASE_PROMPT.format(query=ex["input"])}],
                temperature=0.9,
                max_tokens=80,
            )
            rephrased = resp.choices[0].message.content.strip()
            if rephrased and rephrased != ex["input"] and len(rephrased) > 3:
                new_ex = dict(ex)
                new_ex["input"] = rephrased
                # Keep rewritten_query target unchanged — only the ambiguous input varies
                augmented.append(new_ex)
        except Exception:
            continue

    # ── Augment no-rewrite examples (20% of budget) ──
    n_nr = int(n_augments * 0.20)
    nr_samples = random.sample(no_rewrite_examples, min(n_nr, len(no_rewrite_examples)))

    for ex in nr_samples:
        try:
            resp = client.chat.completions.create(
                model=os.getenv("MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": REPHRASE_PROMPT.format(query=ex["input"])}],
                temperature=0.8,
                max_tokens=80,
            )
            rephrased = resp.choices[0].message.content.strip()
            if rephrased and rephrased != ex["input"] and len(rephrased) > 3:
                new_ex = dict(ex)
                new_ex["input"] = rephrased
                augmented.append(new_ex)
        except Exception:
            continue

    print(f"LLM augmentation: generated {len(augmented)} additional examples "
          f"(~{int(n_augments*0.8)} rewrite + ~{int(n_augments*0.2)} no-rewrite)")
    return augmented


def generate_llm_conversations(n_conversations: int = 60) -> list[dict]:
    """Use GPT-4o-mini to generate brand-new multi-turn rewrite conversations.

    Unlike augment_with_llm (which only paraphrases existing inputs), this creates
    genuinely new (turn0_context, ambiguous_query, rewritten_query) triples from scratch.
    Covers diverse locations × intents × pronoun patterns × ward/district/city scopes.
    """
    try:
        from openai import OpenAI
        client = OpenAI(base_url=os.getenv("API_BASE"), api_key=os.getenv("API_KEY"))
    except Exception as e:
        print(f"LLM conversation generation skipped: {e}")
        return []

    # Seeds: (location, loc_prefix, scope, turn0_intent, followup_intent)
    CONVERSATION_SEEDS = [
        # district seeds — all 15 intents as followup, diverse turn0
        ("Cầu Giấy",    "quận", "district", "daily_forecast",      "current_weather"),
        ("Đống Đa",     "quận", "district", "current_weather",     "rain_query"),
        ("Hoàn Kiếm",   "quận", "district", "weather_overview",    "hourly_forecast"),
        ("Tây Hồ",      "quận", "district", "rain_query",          "wind_query"),
        ("Thanh Xuân",  "quận", "district", "temperature_query",   "daily_forecast"),
        ("Ba Đình",     "quận", "district", "wind_query",          "weather_alert"),
        ("Hoàng Mai",   "quận", "district", "weather_overview",    "activity_weather"),
        ("Long Biên",   "quận", "district", "current_weather",     "expert_weather_param"),
        ("Nam Từ Liêm", "quận", "district", "daily_forecast",      "humidity_fog_query"),
        ("Hà Đông",     "quận", "district", "rain_query",          "smalltalk_weather"),
        ("Gia Lâm",     "huyện","district", "current_weather",     "historical_weather"),
        ("Đông Anh",    "huyện","district", "daily_forecast",      "seasonal_context"),
        ("Sóc Sơn",     "huyện","district", "weather_overview",    "location_comparison"),
        ("Thanh Trì",   "huyện","district", "rain_query",          "temperature_query"),
        ("Mê Linh",     "huyện","district", "current_weather",     "weather_overview"),
        # district seeds with hourly_forecast / wind_query as turn0 (NEW)
        ("Tây Hồ",      "quận", "district", "hourly_forecast",     "rain_query"),
        ("Hoàn Kiếm",   "quận", "district", "hourly_forecast",     "wind_query"),
        ("Gia Lâm",     "huyện","district", "wind_query",          "activity_weather"),
        ("Bắc Từ Từ Liêm","quận","district","wind_query",          "weather_alert"),
        ("Đống Đa",     "quận", "district", "hourly_forecast",     "temperature_query"),
        ("Thanh Trì",   "huyện","district", "wind_query",          "daily_forecast"),
        # city seeds
        ("Hà Nội",      "",     "city",     "weather_overview",    "current_weather"),
        ("Hà Nội",      "",     "city",     "daily_forecast",      "rain_query"),
        ("Hà Nội",      "",     "city",     "current_weather",     "activity_weather"),
        ("Hà Nội",      "",     "city",     "rain_query",          "wind_query"),
        ("Hà Nội",      "",     "city",     "temperature_query",   "seasonal_context"),
        ("Hà Nội",      "",     "city",     "hourly_forecast",     "rain_query"),
        ("Hà Nội",      "",     "city",     "wind_query",          "weather_alert"),
        # ward seeds — expanded from 10 to 20 (NEW)
        ("Yên Hòa",     "phường","ward",    "current_weather",     "daily_forecast"),
        ("Nghĩa Đô",    "phường","ward",    "rain_query",          "humidity_fog_query"),
        ("Khương Đình", "phường","ward",    "weather_overview",    "temperature_query"),
        ("Phú Thượng",  "phường","ward",    "current_weather",     "wind_query"),
        ("Hoàng Liệt",  "phường","ward",    "daily_forecast",      "weather_alert"),
        ("Bồ Đề",       "phường","ward",    "current_weather",     "expert_weather_param"),
        ("Kim Liên",    "phường","ward",    "weather_overview",    "activity_weather"),
        ("Tương Mai",   "phường","ward",    "rain_query",          "current_weather"),
        ("Việt Hưng",   "phường","ward",    "current_weather",     "hourly_forecast"),
        ("Kiến Hưng",   "phường","ward",    "weather_overview",    "smalltalk_weather"),
        ("Đông Ngạc",   "phường","ward",    "current_weather",     "rain_query"),
        ("Tây Mỗ",      "phường","ward",    "daily_forecast",      "temperature_query"),
        ("Láng",        "phường","ward",    "weather_overview",    "activity_weather"),
        ("Vĩnh Hưng",   "phường","ward",    "rain_query",          "daily_forecast"),
        ("Thượng Cát",  "phường","ward",    "humidity_fog_query",  "current_weather"),
        ("Xuân Phương", "phường","ward",    "temperature_query",   "hourly_forecast"),
        ("Phú Diễn",    "phường","ward",    "weather_alert",       "activity_weather"),
        ("Định Công",   "phường","ward",    "daily_forecast",      "wind_query"),
        ("Long Biên",   "phường","ward",    "current_weather",     "seasonal_context"),
        ("Lĩnh Nam",    "phường","ward",    "rain_query",          "historical_weather"),
    ]

    CONVERSATION_PROMPT = """Bạn là người tạo dữ liệu huấn luyện cho chatbot thời tiết Hà Nội.
Tạo 1 cặp hội thoại 2 lượt theo format JSON sau:

Ngữ cảnh:
- Lượt 0: Người dùng hỏi về {turn0_intent} tại {loc_prefix}{location}
- Lượt 1: Người dùng hỏi tiếp câu ngắn/mơ hồ về {followup_intent}, KHÔNG nhắc lại địa điểm (dùng đại từ hoặc bỏ qua địa điểm)
- rewritten_query: Câu lượt 1 được viết lại đầy đủ, tự nhiên, có tên địa điểm rõ ràng

Chú ý:
- Câu lượt 1 phải THỰC SỰ mơ hồ (không có địa điểm, hoặc dùng "ở đó/chỗ đó/đây/đó")
- rewritten_query phải natural, không máy móc
- scope = "{scope}"

Trả về JSON (chỉ JSON, không giải thích):
{{
  "turn0": "...",
  "ambiguous": "...",
  "rewritten": "..."
}}"""

    results = []
    seeds_to_use = random.choices(CONVERSATION_SEEDS, k=n_conversations)

    for location, loc_prefix, scope, turn0_intent, followup_intent in seeds_to_use:
        try:
            prompt = CONVERSATION_PROMPT.format(
                location=location,
                loc_prefix=f"{loc_prefix} " if loc_prefix else "",
                turn0_intent=turn0_intent,
                followup_intent=followup_intent,
                scope=scope,
            )
            resp = client.chat.completions.create(
                model=os.getenv("MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            ambiguous = data.get("ambiguous", "").strip()
            rewritten = data.get("rewritten", "").strip()

            if not ambiguous or not rewritten or ambiguous == rewritten:
                continue

            # Compute realistic confidence
            base_conf = _INTENT_CONFIDENCE.get(followup_intent, 0.90)
            conf = round(min(0.99, max(0.75, base_conf + random.uniform(-0.04, 0.04))), 2)

            results.append({
                "input": ambiguous,
                "context": {
                    "location": location,
                    "intent": turn0_intent,
                    "turn": 1,
                },
                "output": {
                    "intent": followup_intent,
                    "scope": scope,
                    "confidence": conf,
                    "rewritten_query": rewritten,
                },
            })
        except Exception:
            continue

    print(f"LLM conversation generation: {len(results)} new rewrite examples")
    return results


def generate_val_rewrite_examples() -> list[dict]:
    """~55 hand-crafted high-quality rewrite examples for validation set.

    Covers all 15 intents, diverse scopes/pronouns. These are held out from training.
    """
    # (ambiguous_input, context_location, context_intent, context_turn,
    #  output_intent, output_scope, confidence, rewritten_query)
    VAL_REWRITE = [
        # current_weather followups
        ("Ở đó bây giờ thế nào?", "Ba Đình", "daily_forecast", 1, "current_weather", "district", 0.92,
         "Thời tiết hiện tại ở quận Ba Đình như thế nào?"),
        ("Bây giờ nắng không?", "Hà Đông", "rain_query", 1, "current_weather", "district", 0.93,
         "Quận Hà Đông bây giờ có nắng không?"),
        ("Khu đó lúc này ra sao?", "Tây Mỗ", "daily_forecast", 1, "current_weather", "ward", 0.89,
         "Phường Tây Mỗ lúc này thời tiết ra sao?"),
        # rain_query followups
        ("Chiều nay có mưa không?", "Thanh Xuân", "current_weather", 1, "rain_query", "district", 0.91,
         "Chiều nay ở quận Thanh Xuân có mưa không?"),
        ("Mưa đến mấy giờ?", "Long Biên", "rain_query", 2, "rain_query", "district", 0.90,
         "Mưa ở quận Long Biên đến mấy giờ thì tạnh?"),
        ("Tối mai có mưa không?", "Hà Nội", "weather_overview", 1, "rain_query", "city", 0.91,
         "Tối mai ở Hà Nội có mưa không?"),
        ("Ở đó xác suất mưa bao nhiêu?", "Gia Lâm", "daily_forecast", 1, "rain_query", "district", 0.88,
         "Xác suất mưa hôm nay ở huyện Gia Lâm là bao nhiêu?"),
        # daily_forecast followups
        ("Cuối tuần thế nào?", "Cầu Giấy", "current_weather", 1, "daily_forecast", "district", 0.92,
         "Cuối tuần này ở quận Cầu Giấy thời tiết thế nào?"),
        ("Ngày mai ra sao?", "Hà Nội", "daily_forecast", 2, "daily_forecast", "city", 0.91,
         "Ngày mai ở Hà Nội thời tiết ra sao?"),
        ("Thế còn tuần tới?", "Đống Đa", "daily_forecast", 1, "daily_forecast", "district", 0.89,
         "Tuần tới ở quận Đống Đa thời tiết thế nào?"),
        ("Ngày kia nơi đó thế nào?", "Kiến Hưng", "daily_forecast", 1, "daily_forecast", "ward", 0.87,
         "Ngày kia ở phường Kiến Hưng thời tiết thế nào?"),
        # hourly_forecast followups
        ("Sáng mai diễn biến thế nào?", "Hoàn Kiếm", "current_weather", 1, "hourly_forecast", "district", 0.89,
         "Sáng mai ở quận Hoàn Kiếm diễn biến thời tiết thế nào?"),
        ("Tối nay theo giờ?", "Tây Hồ", "rain_query", 1, "hourly_forecast", "district", 0.88,
         "Tối nay ở quận Tây Hồ thời tiết từng giờ thế nào?"),
        # temperature_query followups
        ("Nhiệt độ bao nhiêu?", "Nam Từ Liêm", "current_weather", 1, "temperature_query", "district", 0.90,
         "Nhiệt độ hiện tại ở quận Nam Từ Liêm bao nhiêu độ?"),
        ("Buổi tối mấy độ?", "Hoàn Kiếm", "temperature_query", 1, "temperature_query", "district", 0.88,
         "Tối nay ở quận Hoàn Kiếm nhiệt độ bao nhiêu độ?"),
        ("Nơi đó nóng không?", "Yên Hòa", "weather_overview", 1, "temperature_query", "ward", 0.87,
         "Phường Yên Hòa hôm nay có nóng không?"),
        # wind_query followups
        ("Gió mạnh không?", "Ba Đình", "current_weather", 1, "wind_query", "district", 0.90,
         "Gió ở quận Ba Đình hiện tại có mạnh không?"),
        ("Thế còn gió?", "Sóc Sơn", "weather_alert", 1, "wind_query", "district", 0.88,
         "Gió ở huyện Sóc Sơn hôm nay như thế nào?"),
        ("Hướng gió ra sao?", "Hà Nội", "daily_forecast", 1, "wind_query", "city", 0.87,
         "Gió ở Hà Nội hôm nay hướng nào?"),
        # humidity_fog_query followups
        ("Ở đó có nồm không?", "Đống Đa", "current_weather", 1, "humidity_fog_query", "district", 0.89,
         "Quận Đống Đa hôm nay có nồm ẩm không?"),
        ("Sáng nay sương mù không?", "Hà Nội", "weather_overview", 1, "humidity_fog_query", "city", 0.90,
         "Hà Nội sáng nay có sương mù không?"),
        ("Độ ẩm ở đây bao nhiêu?", "Bồ Đề", "rain_query", 1, "humidity_fog_query", "ward", 0.86,
         "Độ ẩm ở phường Bồ Đề hiện tại là bao nhiêu?"),
        # historical_weather followups
        ("Hôm qua ở đó thế nào?", "Cầu Giấy", "current_weather", 1, "historical_weather", "district", 0.90,
         "Hôm qua ở quận Cầu Giấy thời tiết thế nào?"),
        ("Tuần trước khu đó mưa không?", "Thanh Xuân", "rain_query", 2, "historical_weather", "district", 0.88,
         "Tuần trước ở quận Thanh Xuân có mưa không?"),
        # location_comparison followups
        ("Chênh lệch bao nhiêu?", "Hà Nội", "location_comparison", 1, "location_comparison", "city", 0.88,
         "Nhiệt độ giữa các quận Hà Nội hôm nay chênh lệch bao nhiêu?"),
        ("Còn so với Ba Đình?", "Hà Nội", "location_comparison", 1, "location_comparison", "district", 0.87,
         "So sánh thời tiết Cầu Giấy, Đống Đa và Ba Đình hôm nay?"),
        # activity_weather followups
        ("Có phù hợp chạy bộ không?", "Tây Hồ", "current_weather", 1, "activity_weather", "district", 0.89,
         "Ở quận Tây Hồ hôm nay có phù hợp để chạy bộ không?"),
        ("Sáng mai đi xe đạp được không?", "Hà Nội", "daily_forecast", 1, "activity_weather", "city", 0.88,
         "Sáng mai ở Hà Nội đi xe đạp ngoài trời được không?"),
        ("Ở đó lúc mấy giờ tốt nhất để đi?", "Hoàng Liệt", "activity_weather", 1, "activity_weather", "ward", 0.86,
         "Ở phường Hoàng Liệt hôm nay lúc mấy giờ tốt nhất để ra ngoài?"),
        # expert_weather_param followups
        ("UV cao không?", "Thanh Xuân", "current_weather", 1, "expert_weather_param", "district", 0.87,
         "UV ở quận Thanh Xuân hôm nay có ở mức cao không?"),
        ("Áp suất thay đổi không?", "Hà Nội", "weather_overview", 1, "expert_weather_param", "city", 0.86,
         "Áp suất khí quyển ở Hà Nội hôm nay có thay đổi không?"),
        ("Điểm sương bao nhiêu?", "Cầu Giấy", "expert_weather_param", 1, "expert_weather_param", "district", 0.85,
         "Điểm sương ở quận Cầu Giấy hiện tại là bao nhiêu?"),
        # weather_alert followups
        ("Nguy hiểm không?", "Hà Nội", "weather_alert", 1, "weather_alert", "city", 0.88,
         "Cảnh báo thời tiết hôm nay ở Hà Nội có nguy hiểm không?"),
        ("Bao giờ qua?", "Gia Lâm", "weather_alert", 1, "weather_alert", "district", 0.87,
         "Cảnh báo thời tiết ở huyện Gia Lâm bao giờ thì kết thúc?"),
        ("Có ảnh hưởng đến khu vực mình không?", "Đông Anh", "weather_alert", 1, "weather_alert", "district", 0.86,
         "Cảnh báo thời tiết hôm nay ở huyện Đông Anh có ảnh hưởng đến toàn huyện không?"),
        # seasonal_context followups
        ("So với hôm qua thế nào?", "Hoàng Mai", "current_weather", 1, "seasonal_context", "district", 0.87,
         "Thời tiết quận Hoàng Mai hôm nay so với hôm qua thế nào?"),
        ("Năm nay có bất thường không?", "Hà Nội", "daily_forecast", 1, "seasonal_context", "city", 0.87,
         "Thời tiết Hà Nội năm nay có gì bất thường so với trung bình không?"),
        ("Mùa này thường thế nào?", "Cầu Giấy", "historical_weather", 1, "seasonal_context", "district", 0.86,
         "Mùa này ở quận Cầu Giấy thời tiết thường như thế nào?"),
        # smalltalk_weather followups
        ("Mặc gì đi?", "Hà Nội", "current_weather", 1, "smalltalk_weather", "city", 0.93,
         "Với thời tiết Hà Nội hôm nay nên mặc gì khi ra ngoài?"),
        ("Có cần mang ô không?", "Đống Đa", "rain_query", 1, "smalltalk_weather", "district", 0.91,
         "Đi ra ngoài ở quận Đống Đa hôm nay có cần mang ô không?"),
        # weather_overview followups
        ("Tổng quan hôm nay thế nào?", "Hoàn Kiếm", "rain_query", 1, "weather_overview", "district", 0.91,
         "Tổng quan thời tiết hôm nay ở quận Hoàn Kiếm như thế nào?"),
        ("Nhìn chung khu đó thế nào?", "Bồ Đề", "daily_forecast", 1, "weather_overview", "ward", 0.88,
         "Nhìn chung thời tiết hôm nay ở phường Bồ Đề như thế nào?"),
        # 3-turn examples (turn=3 context)
        ("Thế còn gió?", "Hoàn Kiếm", "temperature_query", 3, "wind_query", "district", 0.88,
         "Gió tại quận Hoàn Kiếm hiện tại thế nào?"),
        ("Ngày kia?", "Cầu Giấy", "rain_query", 3, "daily_forecast", "district", 0.89,
         "Ngày kia ở quận Cầu Giấy thời tiết thế nào?"),
        ("Tạnh chưa?", "Tây Hồ", "rain_query", 3, "current_weather", "district", 0.90,
         "Ở quận Tây Hồ bây giờ mưa đã tạnh chưa?"),
        ("Ở đó sáng mai nữa?", "Khương Đình", "daily_forecast", 3, "hourly_forecast", "ward", 0.86,
         "Sáng mai ở phường Khương Đình thời tiết diễn biến thế nào?"),
    ]

    results = []
    for item in VAL_REWRITE:
        (inp, ctx_loc, ctx_intent, ctx_turn,
         out_intent, out_scope, conf, rewritten) = item
        results.append({
            "input": inp,
            "context": {"location": ctx_loc, "intent": ctx_intent, "turn": ctx_turn},
            "output": {
                "intent": out_intent,
                "scope": out_scope,
                "confidence": conf,
                "rewritten_query": rewritten,
            },
        })
    print(f"Generated {len(results)} val rewrite examples (hand-crafted, all 15 intents)")
    return results


def generate_val_no_rewrite_examples() -> list[dict]:
    """~22 hand-crafted no-rewrite examples for validation set.

    Model must predict intent/scope WITHOUT rewritten_query.
    Covers standalone queries, topic switches, adversarial cases.
    """
    VAL_NO_REWRITE = [
        # (query, ctx_location, ctx_intent, ctx_turn, out_intent, out_scope, conf)
        ("Quận Hoàng Mai ngày mai thế nào?", "Cầu Giấy", "current_weather", 1,
         "daily_forecast", "district", 0.91),
        ("Hà Nội hôm nay tổng quan?", "Đống Đa", "rain_query", 1,
         "weather_overview", "city", 0.95),
        ("Phường Yên Hòa bây giờ ra sao?", "Hoàn Kiếm", "daily_forecast", 1,
         "current_weather", "ward", 0.91),
        ("Toàn Hà Nội ngày mai mưa không?", "Thanh Xuân", "daily_forecast", 2,
         "rain_query", "city", 0.93),
        ("Ba Đình và Đống Đa nơi nào mát hơn?", "Hà Nội", "weather_overview", 1,
         "location_comparison", "district", 0.91),
        ("Hôm qua Tây Hồ thời tiết thế nào?", "Tây Hồ", "current_weather", 1,
         "historical_weather", "district", 0.90),
        ("UV Hà Nội hôm nay cao không?", "Ba Đình", "wind_query", 1,
         "expert_weather_param", "city", 0.87),
        ("Cảm ơn bạn!", "Hà Nội", "current_weather", 1,
         "smalltalk_weather", "city", 0.97),
        ("Thời tiết Huế hôm nay thế nào?", "Cầu Giấy", "rain_query", 2,
         "smalltalk_weather", "city", 0.95),
        ("Quận Bắc Từ Liêm cuối tuần dự báo?", "Hoàng Mai", "daily_forecast", 1,
         "daily_forecast", "district", 0.89),
        ("Hôm nay Hà Nội có bão không?", "Gia Lâm", "daily_forecast", 1,
         "weather_alert", "city", 0.90),
        ("Sáng mai Hà Nội đi xe máy được không?", "Hà Nội", "rain_query", 2,
         "activity_weather", "city", 0.89),
        ("Phường Khương Đình hôm nay có nồm không?", "Cầu Giấy", "current_weather", 1,
         "humidity_fog_query", "ward", 0.88),
        ("Hà Nội năm nay nóng hơn bình thường không?", "Hà Nội", "daily_forecast", 2,
         "seasonal_context", "city", 0.88),
        ("Hôm nay Hoàn Kiếm mặc gì đi làm?", "Ba Đình", "weather_overview", 1,
         "smalltalk_weather", "district", 0.92),
        ("Gió ở Long Biên hôm nay mạnh không?", "Đống Đa", "rain_query", 1,
         "wind_query", "district", 0.90),
        ("Chiều nay quận Hà Đông mấy độ?", "Cầu Giấy", "current_weather", 2,
         "temperature_query", "district", 0.89),
        ("Xã Bát Tràng hôm nay thế nào?", "Hà Nội", "weather_overview", 1,
         "current_weather", "ward", 0.89),
        ("Hà Nội hôm nay dự báo từng giờ?", "Hoàn Kiếm", "current_weather", 2,
         "hourly_forecast", "city", 0.90),
        ("Quận Tây Hồ hôm qua mưa nhiều không?", "Tây Hồ", "rain_query", 2,
         "historical_weather", "district", 0.89),
        ("Áp suất Hà Nội hiện tại bao nhiêu?", "Đống Đa", "temperature_query", 1,
         "expert_weather_param", "city", 0.87),
        ("Huyện Ứng Hòa ngày mai dự báo?", "Hà Nội", "daily_forecast", 1,
         "daily_forecast", "district", 0.86),
    ]

    results = []
    for item in VAL_NO_REWRITE:
        (inp, ctx_loc, ctx_intent, ctx_turn,
         out_intent, out_scope, conf) = item
        results.append({
            "input": inp,
            "context": {"location": ctx_loc, "intent": ctx_intent, "turn": ctx_turn},
            "output": {
                "intent": out_intent,
                "scope": out_scope,
                "confidence": conf,
                # No rewritten_query — self-contained query
            },
        })
    print(f"Generated {len(results)} val no-rewrite examples (hand-crafted)")
    return results


def _subsample_rewrite(rewrite_examples: list[dict], max_per_intent: int = 70) -> list[dict]:
    """Cap over-represented intents in rewrite examples to max_per_intent.

    Preserves under-represented intents fully; caps daily_forecast, rain_query, etc.
    """
    from collections import defaultdict
    by_intent: dict[str, list] = defaultdict(list)
    for ex in rewrite_examples:
        intent = ex.get("output", {}).get("intent", "?")
        by_intent[intent].append(ex)

    result = []
    for intent, exs in by_intent.items():
        if len(exs) > max_per_intent:
            result.extend(random.sample(exs, max_per_intent))
        else:
            result.extend(exs)
    random.shuffle(result)
    return result


def _subsample_routing(routing_examples: list[dict], target: int) -> list[dict]:
    """Subsample routing-only examples to `target` count while preserving intent balance."""
    from collections import defaultdict
    by_intent: dict[str, list] = defaultdict(list)
    for ex in routing_examples:
        intent = ex.get("output", {}).get("intent", "?")
        by_intent[intent].append(ex)

    n_intents = len(by_intent)
    per_intent = target // n_intents
    sampled = []
    for intent, exs in by_intent.items():
        sampled.extend(random.sample(exs, min(per_intent, len(exs))))
    random.shuffle(sampled)
    return sampled


def main(
    existing_train: str = "data/router/train_clean.jsonl",
    existing_val: str = "data/router/val_clean.jsonl",
    output_path: str = "data/router/multitask_train.jsonl",
    val_output_path: str = "data/router/multitask_val.jsonl",
    use_llm: bool = False,
    seed: int = 42,
) -> None:
    random.seed(seed)

    # ── 1. Routing-only examples ──
    routing_examples_full = generate_routing_examples(str(_ROOT / existing_train))

    # ── 2. Contextual rewrite examples ──
    rewrite_examples = generate_rewrite_examples()

    # ── 3. No-rewrite examples ──
    no_rewrite_examples = generate_no_rewrite_examples()

    # ── 4. LLM-generated content (new conversations + paraphrases) ──
    llm_conv_examples: list[dict] = []
    llm_aug_examples: list[dict] = []
    if use_llm:
        llm_conv_examples = generate_llm_conversations(n_conversations=120)
        llm_aug_examples  = augment_with_llm(
            rewrite_examples + llm_conv_examples,
            no_rewrite_examples,
            n_augments=400,
        )

    # ── 5. Balance rewrite intents: cap over-represented intents ──
    # daily_forecast/rain_query/hourly_forecast dominated before; cap each at 70
    all_rewrite_raw = rewrite_examples + llm_conv_examples + [
        x for x in llm_aug_examples if x.get("output", {}).get("rewritten_query")
    ]
    all_rewrite = _subsample_rewrite(all_rewrite_raw, max_per_intent=70)

    n_rewrite = len(all_rewrite)
    n_no_rw = len(no_rewrite_examples) + sum(
        1 for x in llm_aug_examples if not x.get("output", {}).get("rewritten_query")
    )

    # Target ratio: routing ~57%, rewrite ~30%, no-rewrite ~13%
    # Derive routing target from actual rewrite+no-rewrite counts
    contextual_total = n_rewrite + n_no_rw
    routing_target = int(contextual_total / 0.43 * 0.57)   # 57% of total
    # Floor: use minimum of 1200 OR 2× contextual (whichever is smaller),
    # so the ratio stays reasonable even without --use-llm
    routing_floor = min(1200, max(700, contextual_total * 2))
    routing_target = max(routing_target, routing_floor)
    routing_examples = _subsample_routing(routing_examples_full, routing_target)

    # ── 6. Assemble final dataset ──
    all_examples = routing_examples + all_rewrite + no_rewrite_examples
    random.shuffle(all_examples)

    # ── 7. Save train ──
    out_path = _ROOT / output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    total = len(all_examples)
    print(f"\n=== Train dataset summary ===")
    print(f"Total: {total}")
    print(f"  Routing-only  : {len(routing_examples):>4}  ({len(routing_examples)/total*100:4.1f}%)")
    print(f"  With rewrite  : {n_rewrite:>4}  ({n_rewrite/total*100:4.1f}%)")
    print(f"  No-rewrite    : {n_no_rw:>4}  ({n_no_rw/total*100:4.1f}%)")
    print(f"Saved to: {out_path}")

    # ── 8. Val set: routing + rewrite + no-rewrite ──
    val_routing = generate_routing_examples(str(_ROOT / existing_val))
    val_rewrite = generate_val_rewrite_examples()
    val_no_rewrite = generate_val_no_rewrite_examples()
    val_examples = val_routing + val_rewrite + val_no_rewrite
    random.shuffle(val_examples)

    if val_examples:
        val_out_path = _ROOT / val_output_path
        with open(val_out_path, "w", encoding="utf-8") as f:
            for ex in val_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        val_total = len(val_examples)
        n_val_rw = len(val_rewrite)
        n_val_nr = len(val_no_rewrite)
        n_val_rt = len(val_routing)
        print(f"\n=== Val dataset summary ===")
        print(f"Total: {val_total}")
        print(f"  Routing-only  : {n_val_rt:>3}  ({n_val_rt/val_total*100:4.1f}%)")
        print(f"  With rewrite  : {n_val_rw:>3}  ({n_val_rw/val_total*100:4.1f}%)")
        print(f"  No-rewrite    : {n_val_nr:>3}  ({n_val_nr/val_total*100:4.1f}%)")
        print(f"Saved to: {val_out_path}")

    print("\nNext step: Upload multitask_train.jsonl + multitask_val.jsonl to Kaggle dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-task SLM training data")
    parser.add_argument("--train", default="data/router/train_clean.jsonl",
                        help="Existing routing training data (messages format)")
    parser.add_argument("--val", default="data/router/val_clean.jsonl",
                        help="Existing routing val data (messages format)")
    parser.add_argument("--output", default="data/router/multitask_train.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--val-output", default="data/router/multitask_val.jsonl",
                        help="Val output JSONL path")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use GPT-4o-mini to augment with paraphrased examples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        existing_train=args.train,
        existing_val=args.val,
        output_path=args.output,
        val_output_path=args.val_output,
        use_llm=args.use_llm,
        seed=args.seed,
    )
