"""Apply in-place fixes cho 14 rows trong eval CSV có location_name sai vs dim_ward DB.

Fix 2 loại:
1. Thêm "Phường " prefix (10 câu).
2. Thay ward bị sáp nhập bằng ward mới (4 câu — user đã duyệt).

Cập nhật cả `location_name` cột VÀ `question` text (nếu chứa tên phường cũ).

Usage:
    python -m scripts.eval.apply_location_fixes \\
        --input data/evaluation/hanoi_weather_chatbot_eval_questions.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


# id → (new_ward_name_in_loc_name, old_ward_name_in_question, new_ward_name_in_question)
# Format: loc_name kept "(District)" unchanged; only phường name changes.
FIXES: dict[str, dict[str, str]] = {
    # Loại 1 — chỉ thêm "Phường " prefix (question text không đổi)
    "5":   {"new_ward": "Phường Phú Diễn",   "old_q_ward": None, "new_q_ward": None},
    "18":  {"new_ward": "Phường Xuân Đỉnh",  "old_q_ward": None, "new_q_ward": None},
    "172": {"new_ward": "Phường Bạch Mai",   "old_q_ward": None, "new_q_ward": None},
    "173": {"new_ward": "Phường Lĩnh Nam",   "old_q_ward": None, "new_q_ward": None},
    "174": {"new_ward": "Phường Vĩnh Tuy",   "old_q_ward": None, "new_q_ward": None},
    "175": {"new_ward": "Phường Nghĩa Đô",   "old_q_ward": None, "new_q_ward": None},
    "176": {"new_ward": "Phường Tương Mai",  "old_q_ward": None, "new_q_ward": None},
    "177": {"new_ward": "Phường Giảng Võ",   "old_q_ward": None, "new_q_ward": None},
    "178": {"new_ward": "Phường Kiến Hưng",  "old_q_ward": None, "new_q_ward": None},
    "179": {"new_ward": "Phường Xuân Phương","old_q_ward": None, "new_q_ward": None},

    # Loại 2 — ward cũ đã bị sáp nhập, thay bằng ward mới trong district
    # (cả location_name VÀ question text phải đổi)
    "3":  {"new_ward": "Phường Cầu Giấy",    "old_q_ward": "Dịch Vọng",    "new_q_ward": "Cầu Giấy"},
    "14": {"new_ward": "Phường Khương Đình", "old_q_ward": "Khương Trung", "new_q_ward": "Khương Đình"},
    "30": {"new_ward": "Phường Đống Đa",     "old_q_ward": "Quang Trung",  "new_q_ward": "Đống Đa"},
    "48": {"new_ward": "Phường Tây Hồ",      "old_q_ward": "Nhật Tân",     "new_q_ward": "Tây Hồ"},
}


def apply_ward_to_location_name(old: str, new_ward: str) -> str:
    """Thay ward name trong format 'WardName (DistrictName)' bằng new_ward."""
    # Parse '(District)' suffix
    idx = old.rfind("(")
    if idx < 0:
        return new_ward  # no district suffix, just replace
    suffix = old[idx:].strip()  # "(Bắc Từ Liêm)" etc
    return f"{new_ward} {suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    args = ap.parse_args()

    # Đọc (utf-8-sig strip BOM)
    rows = list(csv.DictReader(args.input.open(encoding="utf-8-sig")))
    if not rows:
        raise SystemExit("Empty CSV")

    fieldnames = list(rows[0].keys())
    applied = 0
    for row in rows:
        rid = row.get("id")
        if rid not in FIXES:
            continue
        fix = FIXES[rid]

        old_loc = row["location_name"]
        row["location_name"] = apply_ward_to_location_name(old_loc, fix["new_ward"])

        # Cập nhật question text nếu Loại 2 (ward cũ bị đổi)
        if fix["old_q_ward"] and fix["new_q_ward"]:
            row["question"] = row["question"].replace(
                fix["old_q_ward"], fix["new_q_ward"]
            )

        print(f"[Q{rid}] {old_loc!r} → {row['location_name']!r}")
        applied += 1

    # Ghi lại (không BOM để Python tools đọc cleaner; Excel vẫn mở tốt với UTF-8)
    with args.input.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Applied {applied} fixes to {args.input}")


if __name__ == "__main__":
    main()
