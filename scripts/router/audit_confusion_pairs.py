#!/usr/bin/env python3
"""
P0.2 — Audit & Relabel Confusion Pairs in Training Data
=========================================================
Based on v3 eval: current_weather 75%, weather_alert 82.2%
Root cause: inconsistent labels in training data for confusing intent pairs.

Usage:
    python scripts/router/audit_confusion_pairs.py

Outputs:
    data/router/audit_report.json     — flagged samples with suggested relabels
    data/router/multitask_train_v3.jsonl — relabeled training data
    data/router/multitask_val_v3.jsonl   — relabeled validation data
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "router"

TRAIN_FILE = DATA_DIR / "multitask_train_v2.jsonl"
VAL_FILE = DATA_DIR / "multitask_val_v2.jsonl"
OUT_TRAIN = DATA_DIR / "multitask_train_v3.jsonl"
OUT_VAL = DATA_DIR / "multitask_val_v3.jsonl"
REPORT_FILE = DATA_DIR / "audit_report.json"

# ──────────────────────────────────────────────────────────
# Disambiguation Rules (from docs/intent_disambiguation_rules.md)
# ──────────────────────────────────────────────────────────

# Signal words that STRONGLY indicate a specific intent
# Each tuple: (regex_pattern, correct_intent, override_from_intents)
RELABEL_RULES = [
    # ── current_weather vs temperature_query ──
    # "nhiệt độ / nhiet do / bao nhiêu độ" → temperature_query
    (r"nhi[eệ]t\s*[đd][oộ]", "temperature_query", {"current_weather"}),
    (r"nhiet\s*do", "temperature_query", {"current_weather"}),
    (r"bao\s*nhi[eê]u\s*[đd][oộ]", "temperature_query", {"current_weather"}),
    (r"m[aấ]y\s*[đd][oộ]", "temperature_query", {"current_weather"}),
    # "nóng không / lạnh không" → temperature_query
    (r"\bn[oó]ng\s*(kh[oô]ng|l[aắ]m|qu[aá]|c[oỡ])", "temperature_query", {"current_weather"}),
    (r"\bl[aạ]nh\s*(kh[oô]ng|l[aắ]m|qu[aá]|c[oỡ])", "temperature_query", {"current_weather"}),

    # "thời tiết / thoi tiet" + general question → current_weather
    # Only if currently mislabeled as temperature_query
    (r"th[oờ]i\s*ti[eế]t.*(th[eế]\s*n[aà]o|sao|ra\s*sao)", "current_weather", {"temperature_query"}),
    (r"thoi\s*tiet.*(the\s*nao|sao|ra\s*sao)", "current_weather", {"temperature_query"}),

    # ── rain_query vs weather_alert ──
    # "bão" → weather_alert, BUT exclude "dự báo" (forecast)
    # Use negative lookbehind to avoid matching "dự báo" / "du bao"
    (r"(?<!d[uự]\s)(?<!du\s)\bb[aã]o\b(?!\s*nhi[eê]u)", "weather_alert", {"rain_query"}),
    (r"\bl[uũ]\b", "weather_alert", {"rain_query"}),
    (r"\bng[aậ]p\b", "weather_alert", {"rain_query"}),
    (r"\bgi[oô]ng\b", "weather_alert", {"rain_query"}),
    (r"\bs[eé]t\b", "weather_alert", {"rain_query"}),
    (r"\bl[oố]c\b", "weather_alert", {"rain_query"}),
    (r"c[aả]nh\s*b[aá]o", "weather_alert", {"rain_query", "weather_overview"}),
    (r"nguy\s*hi[eể]m", "weather_alert", {"rain_query"}),
    # "mưa to / mưa lớn / mưa đá" + severity → weather_alert
    (r"m[uư]a\s*(to|l[oớ]n|[đd][aá])", "weather_alert", {"rain_query"}),

    # Simple rain questions → rain_query
    (r"c[oó]\s*m[uư]a\s*kh[oô]ng", "rain_query", {"weather_alert"}),
    (r"m[uư]a\s*bao\s*gi[oờ]\s*t[aạ]nh", "rain_query", {"weather_alert"}),

    # ── wind_query fixes ──
    # "hướng gió / tốc độ gió / gió mạnh" → wind_query (not expert_weather_param)
    (r"h[uư][oớ]ng\s*gi[oó]", "wind_query", {"expert_weather_param", "current_weather"}),
    (r"t[oố]c\s*[đd][oộ]\s*gi[oó]", "wind_query", {"expert_weather_param", "current_weather"}),
    (r"gi[oó]\s*m[aạ]nh", "wind_query", {"expert_weather_param", "current_weather"}),
    (r"c[aấ]p\s*gi[oó]", "wind_query", {"expert_weather_param"}),
    (r"gi[oó]\s*bao\s*nhi[eê]u", "wind_query", {"expert_weather_param"}),

    # ── activity_weather vs smalltalk_weather ──
    # "ra ngoài / đi dạo / chạy bộ / picnic" → activity_weather
    (r"ra\s*ngo[aà]i", "activity_weather", {"smalltalk_weather"}),
    (r"[đd]i\s*d[aạ]o", "activity_weather", {"smalltalk_weather"}),
    (r"ch[aạ]y\s*b[oộ]", "activity_weather", {"smalltalk_weather"}),
    (r"t[uư][oớ]i\s*c[aâ]y", "activity_weather", {"smalltalk_weather"}),
    (r"gi[aặ]t\s*[đd][oồ]", "activity_weather", {"smalltalk_weather"}),
    # "mang / cầm ô" → rain_query (practical rain question)
    (r"(mang|c[aầ]m)\s*[oô]", "rain_query", {"smalltalk_weather"}),
    (r"(mang|c[aầ]m)\s*[aá]o\s*m[uư]a", "rain_query", {"smalltalk_weather"}),
    # "mặc gì" → activity_weather (clothing advice)
    (r"m[aặ]c\s*(g[iì]|sao|th[eế]\s*n[aà]o)", "activity_weather", {"smalltalk_weather"}),
]


def normalize_for_matching(text: str) -> str:
    """Lowercase + collapse whitespace for regex matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


def apply_rules(record: dict) -> dict | None:
    """
    Apply disambiguation rules to a single record.
    Returns a dict with relabel info if a rule fired, or None.
    """
    inp = record.get("input", "")
    out = record.get("output", {})
    if isinstance(out, str):
        out = json.loads(out)
    current_intent = out.get("intent", "")

    normalized = normalize_for_matching(inp)

    for pattern, correct_intent, override_from in RELABEL_RULES:
        if current_intent in override_from:
            if re.search(pattern, normalized):
                return {
                    "input": inp,
                    "old_intent": current_intent,
                    "new_intent": correct_intent,
                    "rule_pattern": pattern,
                    "reason": f"Signal matched: /{pattern}/ → {correct_intent}",
                }
    return None


def process_file(input_path: Path, output_path: Path) -> list[dict]:
    """Process a JSONL file, apply relabeling rules, write output."""
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    changes = []
    for i, rec in enumerate(records):
        result = apply_rules(rec)
        if result:
            result["line_number"] = i + 1
            result["file"] = input_path.name
            changes.append(result)
            # Apply the relabel
            out = rec["output"]
            if isinstance(out, str):
                out = json.loads(out)
                rec["output"] = out
            out["intent"] = result["new_intent"]

    # Write relabeled file
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return changes


def print_summary(changes: list[dict], total: int, filename: str):
    """Print a human-readable summary of changes."""
    print(f"\n{'='*60}")
    print(f"  {filename}: {len(changes)} relabeled / {total} total")
    print(f"{'='*60}")

    if not changes:
        print("  No changes needed!")
        return

    # Group by transition
    transitions = Counter()
    for c in changes:
        transitions[(c["old_intent"], c["new_intent"])] += 1

    print(f"\n  Relabel transitions:")
    for (old, new), count in transitions.most_common():
        print(f"    {old:30s} → {new:30s}  ×{count}")

    print(f"\n  Sample relabels:")
    for c in changes[:10]:
        inp_short = c["input"][:60] + ("..." if len(c["input"]) > 60 else "")
        print(f"    [{c['line_number']:4d}] {inp_short}")
        print(f"           {c['old_intent']} → {c['new_intent']}")
        print(f"           Rule: {c['rule_pattern']}")


def main():
    print("=" * 60)
    print("  P0.2 — Audit & Relabel Confusion Pairs")
    print("=" * 60)

    if not TRAIN_FILE.exists():
        print(f"ERROR: {TRAIN_FILE} not found")
        sys.exit(1)

    # Count original records
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_total = sum(1 for line in f if line.strip())
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        val_total = sum(1 for line in f if line.strip())

    # Process train
    train_changes = process_file(TRAIN_FILE, OUT_TRAIN)
    print_summary(train_changes, train_total, "multitask_train_v2 → v3")

    # Process val
    val_changes = process_file(VAL_FILE, OUT_VAL)
    print_summary(val_changes, val_total, "multitask_val_v2 → v3")

    # Save audit report
    report = {
        "total_train": train_total,
        "total_val": val_total,
        "train_relabeled": len(train_changes),
        "val_relabeled": len(val_changes),
        "train_changes": train_changes,
        "val_changes": val_changes,
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  Audit report: {REPORT_FILE}")
    print(f"  Relabeled train: {OUT_TRAIN}")
    print(f"  Relabeled val:   {OUT_VAL}")

    # Verify intent distribution
    print(f"\n{'='*60}")
    print("  Post-relabel intent distribution (train)")
    print(f"{'='*60}")
    with open(OUT_TRAIN, "r", encoding="utf-8") as f:
        intents = []
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out = rec["output"]
                if isinstance(out, str):
                    out = json.loads(out)
                intents.append(out["intent"])
    for intent, count in sorted(Counter(intents).items(), key=lambda x: -x[1]):
        pct = count / len(intents) * 100
        bar = "█" * int(pct / 2)
        print(f"    {intent:30s} {count:4d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
