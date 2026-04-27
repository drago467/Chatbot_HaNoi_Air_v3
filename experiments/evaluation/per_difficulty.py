"""Per-difficulty breakdown for thesis Chapter 4.

Aggregates tool-call accuracy and manual audit verdict per difficulty group
(easy / medium / hard) from full_run_v12.jsonl + audit_report_thesis.md.

Tool Acc uses experiments.evaluation.tool_accuracy.check_tool_accuracy.
Audit verdict comes from a hardcoded set of 20 "Chưa đạt" IDs documented in
audit_report_thesis.md PHẦN A. Everything else in the 199-case audited subset
is counted as pass (Đạt or Đạt một phần).

Output:
  - Console: formatted breakdown table
  - CSV: data/evaluation/breakdowns/per_difficulty_v12.csv
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tool_accuracy import check_tool_accuracy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRACE_PATH = ROOT / "data" / "evaluation" / "traces" / "full_run_v12.jsonl"
OUTPUT_DIR = ROOT / "data" / "evaluation" / "breakdowns"
OUTPUT_CSV = OUTPUT_DIR / "per_difficulty_v12.csv"

# 20 "Chưa đạt" IDs from audit_report_thesis.md PHẦN A line 2005.
# The source list contained 21 IDs; ID 124 also appears in "Đạt một phần"
# (its conclusion reads "Trung thực nhưng miss data"), so it is treated as
# partial pass. The remaining 20 form the canonical Chưa đạt set.
CHUA_DAT_IDS = frozenset({
    12, 35, 37, 58, 61, 74, 105, 106, 111, 113,
    119, 125, 137, 143, 147, 160, 168, 179, 192, 193,
})


def load_traces(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def per_difficulty(traces: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in traces:
        diff = t.get("expected", {}).get("difficulty", "unknown")
        groups[diff].append(t)

    rows: list[dict] = []
    for diff, items in groups.items():
        total = len(items)
        tool_ok = 0
        chua_dat = 0
        for t in items:
            intent = t["expected"]["intent"]
            scope = t["expected"]["scope"]
            tools_called = [tc["name"] for tc in t.get("tool_calls", [])]
            if check_tool_accuracy(intent, tools_called, scope):
                tool_ok += 1
            if t["id"] in CHUA_DAT_IDS:
                chua_dat += 1
        pass_count = total - chua_dat
        rows.append({
            "difficulty": diff,
            "count": total,
            "tool_ok": tool_ok,
            "tool_acc_pct": round(100 * tool_ok / total, 1),
            "chua_dat": chua_dat,
            "chua_dat_pct": round(100 * chua_dat / total, 1),
            "pass": pass_count,
            "pass_pct": round(100 * pass_count / total, 1),
        })
    return rows


def main() -> None:
    traces = load_traces(TRACE_PATH)
    rows = per_difficulty(traces)

    label_vi = {"easy": "Đơn giản", "medium": "Trung bình", "hard": "Đa bước"}
    order = ["easy", "medium", "hard"]
    rows.sort(key=lambda r: order.index(r["difficulty"]) if r["difficulty"] in order else 99)

    print(f"Loaded {len(traces)} traces from {TRACE_PATH.name}")
    total_chua = sum(r["chua_dat"] for r in rows)
    total_tool_ok = sum(r["tool_ok"] for r in rows)
    print(f"Total Chưa đạt mapped: {total_chua}/{len(CHUA_DAT_IDS)} expected")
    print(f"Overall Tool Acc: {round(100 * total_tool_ok / len(traces), 1)}%")
    print()
    print(f"{'Độ khó':<14}{'Số câu':>8}{'Tool Acc':>12}{'Đạt+Một phần':>16}{'Chưa đạt':>12}")
    print("-" * 64)
    for r in rows:
        print(
            f"{label_vi.get(r['difficulty'], r['difficulty']):<14}"
            f"{r['count']:>8}"
            f"{r['tool_acc_pct']:>11.1f}%"
            f"{r['pass_pct']:>15.1f}%"
            f"{r['chua_dat_pct']:>11.1f}%"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV: {OUTPUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
