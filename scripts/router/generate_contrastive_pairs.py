#!/usr/bin/env python3
"""
P0.3 — Generate Contrastive Pairs for Intent Disambiguation
=============================================================
Creates ~200 carefully designed contrastive training samples
for the top confusion pairs identified in v3 evaluation.

Each "pair" consists of two similar queries with DIFFERENT intents,
teaching the model the exact decision boundary.

Usage:
    python scripts/router/generate_contrastive_pairs.py

Output:
    data/router/contrastive_pairs.jsonl
"""

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_FILE = PROJECT_ROOT / "data" / "router" / "contrastive_pairs.jsonl"

# ──────────────────────────────────────────────────────────
# Hanoi Locations for variety
# ──────────────────────────────────────────────────────────

DISTRICTS = [
    "Cầu Giấy", "Hoàng Mai", "Đống Đa", "Ba Đình", "Thanh Xuân",
    "Hoàn Kiếm", "Hai Bà Trưng", "Tây Hồ", "Long Biên", "Nam Từ Liêm",
    "Bắc Từ Liêm", "Hà Đông", "Thanh Trì", "Gia Lâm", "Đông Anh",
    "Sóc Sơn", "Mê Linh", "Thạch Thất", "Hoài Đức", "Sơn Tây",
]

WARDS = [
    "Yên Phụ", "Láng Hạ", "Dịch Vọng Hậu", "Kim Liên", "Bách Khoa",
    "Trung Hòa", "Nhân Chính", "Thượng Đình", "Phú Đô", "Mỹ Đình",
    "Phú Diễn", "Cổ Nhuế", "Xuân La", "Quảng An", "Ngọc Hà",
]

TIME_REFS = ["bây giờ", "hiện tại", "lúc này", "ngay bây giờ"]
TODAY_REFS = ["hôm nay", "ngày hôm nay", "today"]
TOMORROW_REFS = ["ngày mai", "mai", "sáng mai"]

# ──────────────────────────────────────────────────────────
# Contrastive Pair Templates
# ──────────────────────────────────────────────────────────

def generate_current_vs_temperature():
    """
    Confusion pair #1: current_weather ↔ temperature_query
    5 errors in v3 eval
    """
    pairs = []
    for loc in random.sample(DISTRICTS, 12):
        scope = "district"
        # Pair 1: "thời tiết thế nào" vs "nhiệt độ bao nhiêu"
        pairs.append({
            "input": f"Thời tiết ở {loc} {random.choice(TIME_REFS)} thế nào?",
            "context": None,
            "output": {"intent": "current_weather", "scope": scope, "confidence": 0.92}
        })
        pairs.append({
            "input": f"Nhiệt độ ở {loc} {random.choice(TIME_REFS)} bao nhiêu?",
            "context": None,
            "output": {"intent": "temperature_query", "scope": scope, "confidence": 0.93}
        })

    for loc in random.sample(WARDS, 8):
        scope = "ward"
        # Pair 2: "trời thế nào" vs "nóng không"
        pairs.append({
            "input": f"Trời ở phường {loc} {random.choice(TODAY_REFS)} thế nào?",
            "context": None,
            "output": {"intent": "current_weather", "scope": scope, "confidence": 0.91}
        })
        pairs.append({
            "input": f"Phường {loc} {random.choice(TODAY_REFS)} nóng không?",
            "context": None,
            "output": {"intent": "temperature_query", "scope": scope, "confidence": 0.90}
        })

    # City-level
    city_templates_cw = [
        "Hà Nội {time} thời tiết ra sao?",
        "Trời {time} có đẹp không?",
        "Tình hình thời tiết Hà Nội {time}?",
        "{time} Hà Nội thế nào nhỉ?",
        "Hà Nội {time} có nắng không?",
        "Thời tiết {time} ổn không?",
        "Trời Hà Nội {time} sao rồi?",
        "Cho mình hỏi thời tiết Hà Nội {time}",
    ]
    city_templates_tq = [
        "Hà Nội {time} bao nhiêu độ?",
        "{time} Hà Nội nóng lắm không?",
        "Nhiệt độ Hà Nội {time} là mấy?",
        "Hà Nội {time} lạnh không?",
        "{time} nóng bao nhiêu độ?",
        "Nhiệt độ cao nhất {today} bao nhiêu?",
        "Hà Nội {today} nóng cỡ nào?",
        "Bao nhiêu độ C rồi?",
    ]

    for tmpl in city_templates_cw:
        time = random.choice(TIME_REFS)
        today = random.choice(TODAY_REFS)
        pairs.append({
            "input": tmpl.format(time=time, today=today),
            "context": None,
            "output": {"intent": "current_weather", "scope": "city", "confidence": 0.93}
        })

    for tmpl in city_templates_tq:
        time = random.choice(TIME_REFS)
        today = random.choice(TODAY_REFS)
        pairs.append({
            "input": tmpl.format(time=time, today=today),
            "context": None,
            "output": {"intent": "temperature_query", "scope": "city", "confidence": 0.91}
        })

    # Telex variants (common user input without diacritics)
    telex_cw = [
        "thoi tiet Ha Noi bay gio the nao?",
        "bay gio troi co dep khong?",
        "hom nay Ha Noi thoi tiet sao?",
        "tinh hinh thoi tiet bay gio?",
    ]
    telex_tq = [
        "Ha Noi bay gio bao nhieu do?",
        "nhiet do Ha Noi hien tai?",
        "bay gio nong khong?",
        "hom nay nhiet do cao nhat bao nhieu?",
    ]
    for t in telex_cw:
        pairs.append({
            "input": t, "context": None,
            "output": {"intent": "current_weather", "scope": "city", "confidence": 0.90}
        })
    for t in telex_tq:
        pairs.append({
            "input": t, "context": None,
            "output": {"intent": "temperature_query", "scope": "city", "confidence": 0.90}
        })

    return pairs


def generate_rain_vs_alert():
    """
    Confusion pair #2: rain_query ↔ weather_alert
    3 errors in v3 eval
    """
    pairs = []
    for loc in random.sample(DISTRICTS, 10):
        # Simple rain question → rain_query
        pairs.append({
            "input": f"{random.choice(TOMORROW_REFS)} ở {loc} có mưa không?",
            "context": None,
            "output": {"intent": "rain_query", "scope": "district", "confidence": 0.92}
        })
        # Severe weather → weather_alert
        pairs.append({
            "input": f"Có cảnh báo mưa to ở {loc} không?",
            "context": None,
            "output": {"intent": "weather_alert", "scope": "district", "confidence": 0.91}
        })

    # City-level contrastive
    rain_templates = [
        "Hà Nội có mưa không?",
        "Mưa bao giờ tạnh?",
        "{tomorrow} có mưa không nhỉ?",
        "Lúc nào hết mưa?",
        "Cần mang ô không?",
        "Xác suất mưa hôm nay bao nhiêu?",
        "Chiều nay mưa không?",
        "Mưa phùn hay mưa rào?",
    ]
    alert_templates = [
        "Có bão đến Hà Nội không?",
        "Cảnh báo thời tiết hôm nay có gì?",
        "Hà Nội có nguy hiểm gì về thời tiết không?",
        "Có ngập ở đâu không?",
        "Mưa to có nguy hiểm không?",
        "Có giông sét không?",
        "Cảnh báo áp thấp nhiệt đới?",
        "Hà Nội có lũ không?",
        "Thời tiết có gì bất thường không?",
        "Cảnh báo mưa đá?",
    ]

    for tmpl in rain_templates:
        pairs.append({
            "input": tmpl.format(tomorrow=random.choice(TOMORROW_REFS)),
            "context": None,
            "output": {"intent": "rain_query", "scope": "city", "confidence": 0.93}
        })
    for tmpl in alert_templates:
        pairs.append({
            "input": tmpl, "context": None,
            "output": {"intent": "weather_alert", "scope": "city", "confidence": 0.92}
        })

    # Telex
    telex_rain = [
        "co mua khong?", "mua bao gio tanh?", "can mang o khong?",
    ]
    telex_alert = [
        "co bao khong?", "canh bao thoi tiet hom nay?", "co ngap o dau khong?",
    ]
    for t in telex_rain:
        pairs.append({
            "input": t, "context": None,
            "output": {"intent": "rain_query", "scope": "city", "confidence": 0.88}
        })
    for t in telex_alert:
        pairs.append({
            "input": t, "context": None,
            "output": {"intent": "weather_alert", "scope": "city", "confidence": 0.89}
        })

    return pairs


def generate_alert_vs_expert():
    """
    Confusion pair: weather_alert ↔ expert_weather_param
    2 errors in v3 eval
    """
    pairs = []
    expert_templates = [
        "Áp suất khí quyển Hà Nội bao nhiêu?",
        "Chỉ số UV hôm nay bao nhiêu?",
        "Nhiệt độ điểm sương ở Hà Nội?",
        "Chất lượng không khí AQI hôm nay?",
        "Tầm nhìn hiện tại bao nhiêu km?",
        "Áp suất hôm nay bao nhiêu hPa?",
        "Độ cao mây bao nhiêu?",
        "Chỉ số tia cực tím ra sao?",
    ]
    alert_from_param = [
        "Có áp thấp nhiệt đới không?",
        "Áp suất giảm đột ngột — có bão không?",
        "Có cảnh báo nắng nóng gay gắt không?",
        "Rét đậm rét hại Hà Nội?",
        "Cảnh báo UV nguy hiểm?",
    ]

    for tmpl in expert_templates:
        pairs.append({
            "input": tmpl, "context": None,
            "output": {"intent": "expert_weather_param", "scope": "city", "confidence": 0.90}
        })
    for tmpl in alert_from_param:
        pairs.append({
            "input": tmpl, "context": None,
            "output": {"intent": "weather_alert", "scope": "city", "confidence": 0.91}
        })

    return pairs


def generate_overview_vs_current():
    """
    Confusion pair: weather_overview ↔ current_weather
    """
    pairs = []
    overview_templates = [
        "Tóm tắt thời tiết Hà Nội hôm nay",
        "Tổng quan thời tiết ngày mai?",
        "Thời tiết hôm nay nhìn chung sao?",
        "Cho mình cái overview thời tiết hôm nay",
        "Bản tóm tắt thời tiết Hà Nội?",
    ]
    current_templates = [
        "Bây giờ trời thế nào?",
        "Hà Nội đang mưa à?",
        "Hiện tại trời có nắng không?",
        "Ngay lúc này Hà Nội thế nào?",
        "Trời đang ra sao?",
    ]

    for tmpl in overview_templates:
        pairs.append({
            "input": tmpl, "context": None,
            "output": {"intent": "weather_overview", "scope": "city", "confidence": 0.91}
        })
    for tmpl in current_templates:
        pairs.append({
            "input": tmpl, "context": None,
            "output": {"intent": "current_weather", "scope": "city", "confidence": 0.93}
        })

    return pairs


def generate_hourly_vs_daily():
    """
    Confusion pair: hourly_forecast ↔ daily_forecast
    """
    pairs = []
    for loc in random.sample(DISTRICTS, 6):
        # Hourly
        pairs.append({
            "input": f"2-3 tiếng nữa {loc} thế nào?",
            "context": None,
            "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.91}
        })
        pairs.append({
            "input": f"Chiều nay ở {loc} có mưa không?",
            "context": None,
            "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.90}
        })
        # Daily
        pairs.append({
            "input": f"Ngày mai {loc} thế nào?",
            "context": None,
            "output": {"intent": "daily_forecast", "scope": "district", "confidence": 0.92}
        })
        pairs.append({
            "input": f"Cuối tuần {loc} có mưa không?",
            "context": None,
            "output": {"intent": "daily_forecast", "scope": "district", "confidence": 0.91}
        })

    return pairs


def generate_wind_fixes():
    """
    Fix: wind queries incorrectly labeled as expert_weather_param
    """
    pairs = []
    wind_templates = [
        "Gió ở {loc} mạnh không?",
        "Tốc độ gió {loc} bao nhiêu km/h?",
        "Hướng gió ở {loc} hiện tại?",
        "Gió cấp mấy?",
        "{loc} hôm nay có gió to không?",
        "Gió bao nhiêu km/h rồi?",
    ]
    for loc in random.sample(DISTRICTS, 6):
        for tmpl in random.sample(wind_templates, 3):
            pairs.append({
                "input": tmpl.format(loc=loc),
                "context": None,
                "output": {"intent": "wind_query", "scope": "district", "confidence": 0.91}
            })

    return pairs


def generate_activity_fixes():
    """
    Fix: activity queries labeled as smalltalk_weather
    """
    pairs = []
    activity_templates = [
        "Hôm nay ra ngoài được không?",
        "Sáng mai đi chạy bộ được không?",
        "Cuối tuần đi picnic có ổn không?",
        "Hôm nay đi tưới cây được không?",
        "Phơi đồ hôm nay được không?",
        "Đi dạo buổi tối Hà Nội ổn không?",
        "Nên mặc gì hôm nay?",
        "{loc} hôm nay đi dạo được không?",
    ]
    for tmpl in activity_templates:
        loc = random.choice(DISTRICTS)
        pairs.append({
            "input": tmpl.format(loc=loc),
            "context": None,
            "output": {
                "intent": "activity_weather",
                "scope": "city" if "{loc}" not in tmpl else "district",
                "confidence": 0.90
            }
        })

    return pairs


def main():
    random.seed(42)

    all_pairs = []
    all_pairs.extend(generate_current_vs_temperature())
    all_pairs.extend(generate_rain_vs_alert())
    all_pairs.extend(generate_alert_vs_expert())
    all_pairs.extend(generate_overview_vs_current())
    all_pairs.extend(generate_hourly_vs_daily())
    all_pairs.extend(generate_wind_fixes())
    all_pairs.extend(generate_activity_fixes())

    # Shuffle
    random.shuffle(all_pairs)

    # Write
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for rec in all_pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Stats
    from collections import Counter
    intents = Counter(r["output"]["intent"] for r in all_pairs)

    print("=" * 60)
    print("  P0.3 — Contrastive Pairs Generated")
    print("=" * 60)
    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  Output: {OUT_FILE}")
    print(f"\n  Intent distribution:")
    for intent, count in intents.most_common():
        print(f"    {intent:30s} {count:3d}")
    print(f"\n  Categories:")
    print(f"    current_weather ↔ temperature_query:  ~56 pairs")
    print(f"    rain_query ↔ weather_alert:           ~46 pairs")
    print(f"    weather_alert ↔ expert_weather_param: ~13 pairs")
    print(f"    weather_overview ↔ current_weather:   ~10 pairs")
    print(f"    hourly_forecast ↔ daily_forecast:     ~24 pairs")
    print(f"    wind_query fixes:                     ~18 pairs")
    print(f"    activity_weather fixes:               ~8 pairs")


if __name__ == "__main__":
    main()
