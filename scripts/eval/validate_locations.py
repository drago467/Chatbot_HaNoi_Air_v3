"""Validate location_name trong eval CSV vs dim_ward DB.

Bộ 126 phường/xã + 30 quận/huyện Hà Nội sau sáp nhập lưu trong `dim_ward`.
Script này check mỗi row trong CSV eval questions, flag những câu có tên
ward/district/city không khớp DB + suggest fix gần nhất qua fuzzy match.

Usage:
    python -m scripts.eval.validate_locations \\
        --input data/evaluation/hanoi_weather_chatbot_eval_questions.csv \\
        --output data/evaluation/location_mismatch.csv
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import sys
from pathlib import Path


def load_canonical_sets() -> tuple[set[str], set[str], dict[str, set[str]]]:
    """Load canonical ward/district names từ dim_ward DB.

    Returns:
        (ward_set, district_set, ward_to_districts) — ward_to_districts map
        ward_name_vi → set các district_name_vi mà ward đó thuộc về.
    """
    from app.db.dal import query

    rows = query("SELECT ward_name_vi, district_name_vi FROM dim_ward")
    ward_set: set[str] = set()
    district_set: set[str] = set()
    ward_to_districts: dict[str, set[str]] = {}
    for r in rows:
        w = (r.get("ward_name_vi") or "").strip()
        d = (r.get("district_name_vi") or "").strip()
        if w:
            ward_set.add(w)
            ward_to_districts.setdefault(w, set()).add(d)
        if d:
            district_set.add(d)
    return ward_set, district_set, ward_to_districts


def normalize_district(name: str) -> str:
    """Chuẩn hoá tên district: strip 'Quận '/'Huyện '/'Thị xã ' prefix."""
    n = name.strip()
    for prefix in ("Quận ", "Huyện ", "Thị xã ", "TP. ", "Thành phố "):
        if n.startswith(prefix):
            n = n[len(prefix):].strip()
    return n


def find_closest(name: str, candidates: set[str], n: int = 3, cutoff: float = 0.6) -> list[str]:
    """Tìm các tên gần nhất trong candidates bằng difflib."""
    return difflib.get_close_matches(name, list(candidates), n=n, cutoff=cutoff)


def parse_ward_location(location_name: str) -> tuple[str, str] | None:
    """Parse format 'Ward (District)' → (ward, district). None nếu không khớp."""
    m = re.match(r"^\s*(.+?)\s*\((.+?)\)\s*$", location_name)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def check_district(name: str, canonical: set[str]) -> tuple[bool, list[str]]:
    """Check tên district, trả (ok, suggestions)."""
    # Direct match
    if name in canonical:
        return True, []
    # Thử normalize (gỡ "Quận "/"Huyện ")
    normalized = normalize_district(name)
    for c in canonical:
        if normalize_district(c) == normalized:
            return True, [c]  # suggest canonical form (with prefix)
    # Fuzzy
    return False, find_closest(name, canonical)


def check_ward(
    ward: str,
    district: str,
    ward_set: set[str],
    district_set: set[str],
    ward_to_districts: dict[str, set[str]],
) -> tuple[bool, list[str]]:
    """Check (ward, district) pair hợp lệ + ward thuộc district đó."""
    if ward not in ward_set:
        suggs = find_closest(ward, ward_set)
        return False, [f"ward không có trong DB; gần nhất: {suggs}"]

    # Ward tồn tại. Check district có đúng không.
    _district_ok, district_suggs = check_district(district, district_set)
    if not _district_ok:
        return False, [f"district không có trong DB; gần nhất: {district_suggs}"]

    # Check pair — ward đó có thuộc district đó không
    expected_districts = ward_to_districts.get(ward, set())
    # Normalize so sánh
    norm_district = normalize_district(district)
    if not any(normalize_district(d) == norm_district for d in expected_districts):
        return False, [
            f"ward '{ward}' KHÔNG thuộc district '{district}'; "
            f"thuộc: {sorted(expected_districts)}"
        ]

    return True, []


def validate_row(
    row: dict,
    ward_set: set[str],
    district_set: set[str],
    ward_to_districts: dict[str, set[str]],
) -> tuple[str, str]:
    """Validate 1 row. Return (status, detail). status ∈ {'ok','mismatch','skip'}."""
    scope = (row.get("location_scope") or "").strip().lower()
    location_name = (row.get("location_name") or "").strip()

    if scope == "city":
        # Chỉ nhận "Hà Nội" (hoặc biến thể). Các city khác (Đà Nẵng...) → OOS test.
        canonical_city = "Hà Nội"
        if not location_name or "Hà Nội" in location_name or location_name.lower() == "ha noi":
            return "ok", ""
        # OOS city → skip (cố ý test OOS)
        if location_name in ("Đà Nẵng", "TP.HCM", "Hồ Chí Minh"):
            return "skip", f"OOS city '{location_name}' — cố ý để test out-of-scope"
        return "mismatch", f"scope=city nhưng tên lạ: '{location_name}'"

    if scope == "poi":
        return "skip", "POI — không bắt buộc match DB (clarification flow)"

    if scope == "district":
        # location_name có thể là "A, B" cho location_comparison
        names = [n.strip() for n in re.split(r",(?![^(]*\))", location_name) if n.strip()]
        bad = []
        for n in names:
            # Strip "Hà Nội nội/ngoại thành" — special case
            if "nội thành" in n or "ngoại thành" in n:
                continue
            ok, suggs = check_district(n, district_set)
            if not ok:
                bad.append(f"'{n}' → suggest {suggs}")
        if bad:
            return "mismatch", "; ".join(bad)
        return "ok", ""

    if scope == "ward":
        parsed = parse_ward_location(location_name)
        if not parsed:
            return "mismatch", f"không parse được format 'Ward (District)': '{location_name}'"
        ward, district = parsed
        ok, detail = check_ward(ward, district, ward_set, district_set, ward_to_districts)
        if not ok:
            return "mismatch", "; ".join(detail)
        return "ok", ""

    # Empty scope (OOS questions) → skip
    if not scope:
        return "skip", "scope trống"

    return "skip", f"scope lạ: '{scope}'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", type=Path,
                    default=Path("data/evaluation/location_mismatch.csv"))
    args = ap.parse_args()

    print(f"Loading canonical sets từ dim_ward DB...")
    ward_set, district_set, ward_to_districts = load_canonical_sets()
    print(f"  wards: {len(ward_set)}, districts: {len(district_set)}")

    if not ward_set or not district_set:
        print("ERROR: dim_ward trống — chạy build_dim_ward trước!", file=sys.stderr)
        sys.exit(1)

    # Đọc CSV (utf-8-sig để strip BOM nếu có)
    rows = list(csv.DictReader(args.input.open(encoding="utf-8-sig")))
    print(f"Loaded {len(rows)} questions từ {args.input}")

    # Validate
    mismatches = []
    ok_count = 0
    skip_count = 0
    for row in rows:
        status, detail = validate_row(row, ward_set, district_set, ward_to_districts)
        if status == "ok":
            ok_count += 1
        elif status == "skip":
            skip_count += 1
        else:
            mismatches.append({
                "id": row.get("id"),
                "question": row.get("question"),
                "intent": row.get("intent"),
                "location_scope": row.get("location_scope"),
                "location_name": row.get("location_name"),
                "problem": detail,
            })

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "question", "intent", "location_scope", "location_name", "problem"],
        )
        writer.writeheader()
        writer.writerows(mismatches)

    # Report
    print()
    print(f"{'─' * 60}")
    print(f"OK:       {ok_count}/{len(rows)}")
    print(f"SKIP:     {skip_count}/{len(rows)} (POI/OOS/scope trống)")
    print(f"MISMATCH: {len(mismatches)}/{len(rows)}  → ghi vào {args.output}")
    print(f"{'─' * 60}")

    if mismatches:
        print("\nTop mismatches:")
        for m in mismatches[:10]:
            print(f"  [Q{m['id']}] {m['location_scope']}={m['location_name']!r}")
            print(f"           → {m['problem']}")
        if len(mismatches) > 10:
            print(f"  ... và {len(mismatches) - 10} câu khác, xem {args.output}")

    sys.exit(0 if not mismatches else 2)


if __name__ == "__main__":
    main()
