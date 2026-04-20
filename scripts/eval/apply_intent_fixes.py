"""Apply 15 intent label fixes theo intent_disambiguation_rules.md.

Chỉ fix các câu có intent cũ SAI theo rules đã thiết kế, router predict ĐÚNG.
KHÔNG sửa câu router sai hoặc ambiguous — giữ tính đúng đắn.

Chỉ update cột `intent` (English label). Cột `intent_vi` cũng cần update đồng bộ.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


# id → (new_intent, new_intent_vi, lý do)
FIXES: dict[str, tuple[str, str, str]] = {
    # Section 1: current_weather vs temperature/wind/humidity/rain — signal word rules
    "3":   ("rain_query",         "Truy vấn mưa",                  "Signal 'có mưa không' → rain_query"),
    "4":   ("temperature_query",  "Truy vấn nhiệt độ",             "Signal 'nhiệt độ bao nhiêu độ' → temp_query"),
    "6":   ("wind_query",         "Truy vấn gió",                  "Signal 'gió...mạnh không' → wind_query"),
    "8":   ("humidity_fog_query", "Độ ẩm, mây, sương mù",          "Signal 'độ ẩm' → humidity_fog_query"),
    "11":  ("temperature_query",  "Truy vấn nhiệt độ",             "Signal 'nhiệt độ cảm giác' → temp_query"),
    "16":  ("rain_query",         "Truy vấn mưa",                  "Primary signal 'mưa lớn không' → rain_query"),
    # Section 2: rain vs alert safety rule
    "18":  ("weather_alert",      "Cảnh báo thời tiết",            "'mưa rào hay giông' → alert (safety rule)"),
    "48":  ("weather_alert",      "Cảnh báo thời tiết",            "'giông' → alert (safety rule)"),
    "50":  ("weather_alert",      "Cảnh báo thời tiết",            "'mưa to kéo dài' → alert (rules 'mưa to/lớn/đá')"),
    "51":  ("weather_alert",      "Cảnh báo thời tiết",            "'gây ngập' → alert (safety)"),
    "70":  ("weather_alert",      "Cảnh báo thời tiết",            "'rét đậm' → alert (extreme cold)"),
    # Daily_forecast with specific temp focus
    "34":  ("temperature_query",  "Truy vấn nhiệt độ",             "Primary 'nhiệt độ cao nhất' → temp_query"),
    # Smalltalk → specific intent per content rules
    "144": ("activity_weather",   "Thời tiết theo hoạt động",      "'tư vấn mặc gì' → activity (outfit advice)"),
    "146": ("rain_query",         "Truy vấn mưa",                  "Signal 'mang ô' → rain_query (rules)"),
    # Location comparison = between LOCATIONS, not TIME
    "195": ("seasonal_context",   "Ngữ cảnh thời vụ",              "'hôm nay vs ngày mai' = time compare, not location"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.input.open(encoding="utf-8-sig")))
    if not rows:
        raise SystemExit("Empty CSV")

    fieldnames = list(rows[0].keys())
    applied = 0
    for row in rows:
        rid = row.get("id")
        if rid not in FIXES:
            continue
        new_intent, new_intent_vi, reason = FIXES[rid]
        old_intent = row["intent"]
        old_intent_vi = row.get("intent_vi", "")
        row["intent"] = new_intent
        if "intent_vi" in row:
            row["intent_vi"] = new_intent_vi
        print(f"[Q{rid}] {old_intent:>18} → {new_intent:<18} | {reason}")
        applied += 1

    if args.dry_run:
        print(f"\n(dry-run) {applied} changes would be applied.")
        return

    with args.input.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Applied {applied}/{len(FIXES)} intent label fixes to {args.input}")


if __name__ == "__main__":
    main()
