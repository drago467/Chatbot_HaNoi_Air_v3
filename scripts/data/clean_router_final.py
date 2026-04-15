"""
scripts/data/clean_router_final.py
Final data cleaning — produces v4-clean train + val for Exp6 fine-tuning.

Three actions:
  A. Remove 94 leaked val samples (72 exact-normalized + 22 fuzzy ≥ 0.85)
  B. Fix 5 mislabeled "mùa này" train samples: smalltalk_weather → seasonal_context
  C. Add 25 hand-crafted val samples for 7 weak intents

Design references used:
  - docs/Weather Intent Design.md (intent definitions, Section 1-14)
  - docs/intent_disambiguation_rules.md (confusion pairs, signal words)
  - app/agent/router/config.py (ROUTER_SYSTEM_PROMPT, VALID_INTENTS)
  - app/agent/router/tool_mapper.py (tool mapping per intent)

Input:
  data/router/multitask_train_v3_clean.jsonl  (3302)
  data/router/multitask_val_v3_clean.jsonl    (434)

Output:
  data/router/multitask_train_v4_clean.jsonl  (3302 — same count, 5 relabeled)
  data/router/multitask_val_v4_clean.jsonl    (340 - 94 + 25 = 365)
  data/router/expert_audit_report.json        (updated with final stats)
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_IN = ROOT / "data/router/multitask_train_v3_clean.jsonl"
VAL_IN = ROOT / "data/router/multitask_val_v3_clean.jsonl"
TRAIN_OUT = ROOT / "data/router/multitask_train_v4_clean.jsonl"
VAL_OUT = ROOT / "data/router/multitask_val_v4_clean.jsonl"
REPORT = ROOT / "data/router/expert_audit_report.json"


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def save(data, p):
    with open(p, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def strip_diacritics(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def norm_key(q: str) -> str:
    s = strip_diacritics(q).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def char_ngrams(s: str, n: int = 4) -> set:
    s = norm_key(s)
    return {s[i : i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


# ═══════════════════════════════════════════════════════════════════
# B. TRAIN MISLABEL FIXES
# ═══════════════════════════════════════════════════════════════════
# 5 samples labeled smalltalk_weather but contain "mùa này" / "mua nay"
# which clearly matches seasonal_context definition:
#   "SO SÁNH với hôm qua/tuần trước, xu hướng, bất thường theo MÙA"
#   — ROUTER_SYSTEM_PROMPT (config.py line 71)
#
# Compare with correctly-labeled samples in same dataset:
#   [seasonal_context] "Thời tiết mùa này ra sao?"
#   [seasonal_context] "Mùa này Hà Nội thường mưa nhiều không?"
#   [seasonal_context] "Mùa này thời tiết ra sao?"
#
# The 5 below follow the exact same pattern but were mislabeled.
# ═══════════════════════════════════════════════════════════════════

MISLABEL_FIXES = [
    # query (exact match) → new intent
    # Kept original scope + confidence — only intent changes.
    ("Phường Chương Mỹ mùa này thế nào nhỉ?", "seasonal_context"),
    ("Phường Suối Hai mùa này thế nào nhỉ?", "seasonal_context"),
    ("phuong chuong my mua nay the nao nhi?", "seasonal_context"),
    ("Phường Tùng Thiện mùa này thế nào nhỉ?", "seasonal_context"),
    ("Mùa này Hà Nội thời tiết hay nhỉ?", "seasonal_context"),
]


# ═══════════════════════════════════════════════════════════════════
# C. NEW VAL SAMPLES (25 hand-crafted)
# ═══════════════════════════════════════════════════════════════════
# After removing 94 leaked val samples (434 → 340), 7 intents fall
# below the 20-sample threshold needed for stable per-intent F1.
#
# Each sample below includes:
#   - RATIONALE: why this intent (citing design doc section)
#   - Scope reasoning
#   - Diverse Vietnamese register (formal/informal, diacritic/ASCII)
# ═══════════════════════════════════════════════════════════════════

NEW_VAL_SAMPLES = [
    # ── expert_weather_param: 19 → 20 (need 1) ──────────────────
    # Design Section 12: "áp suất, UV, điểm sương, tầm nhìn"
    # Disambiguation Rule 5: "áp suất / khí áp" → expert_weather_param
    {
        "input": "Chỉ số tia UV ở Ba Đình trưa nay khoảng mức nào?",
        "context": None,
        "output": {"intent": "expert_weather_param", "scope": "district", "confidence": 0.92},
        "_rationale": "UV index = technical param (Section 12). Ba Đình = district.",
    },

    # ── historical_weather: 14 → 20 (need 6) ────────────────────
    # Design Section 9: "thời tiết NGÀY/TUẦN TRƯỚC, dữ liệu QUÁ KHỨ"
    {
        "input": "Hôm qua ở Thanh Xuân có mưa to không?",
        "context": None,
        "output": {"intent": "historical_weather", "scope": "district", "confidence": 0.91},
        "_rationale": "'Hôm qua' = past data (Section 9). Thanh Xuân = district.",
    },
    {
        "input": "Tuần trước nhiệt độ trung bình ở Hà Nội là bao nhiêu?",
        "context": None,
        "output": {"intent": "historical_weather", "scope": "city", "confidence": 0.90},
        "_rationale": "'Tuần trước' = past data. No specific location → city scope.",
    },
    {
        "input": "Ba ngày trước ở Long Biên trời có nắng không?",
        "context": None,
        "output": {"intent": "historical_weather", "scope": "district", "confidence": 0.89},
        "_rationale": "'Ba ngày trước' = past data. Long Biên = district.",
    },
    {
        "input": "nhiet do ha noi hom qua la bao nhieu?",
        "context": None,
        "output": {"intent": "historical_weather", "scope": "city", "confidence": 0.90},
        "_rationale": "'hom qua' = past data (no diacritics variant). city scope.",
    },
    {
        "input": "Phường Láng Hạ tuần trước thời tiết thế nào?",
        "context": None,
        "output": {"intent": "historical_weather", "scope": "ward", "confidence": 0.88},
        "_rationale": "'Tuần trước' = historical. 'Phường Láng Hạ' = ward scope.",
    },
    {
        "input": "Cách đây 5 ngày ở Đống Đa có mưa không?",
        "context": None,
        "output": {"intent": "historical_weather", "scope": "district", "confidence": 0.87},
        "_rationale": "'Cách đây 5 ngày' = past data. Đống Đa = district.",
    },

    # ── hourly_forecast: 19 → 20 (need 1) ───────────────────────
    # Design Section 2: "dự báo chi tiết theo giờ trong 24-48h"
    # Disambiguation Rule 4: "chiều nay" = slot today → hourly_forecast
    {
        "input": "Từ 2 giờ đến 5 giờ chiều nay ở Hoàn Kiếm trời có nắng không?",
        "context": None,
        "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.91},
        "_rationale": "'Từ 2-5 giờ chiều nay' = specific time slot within today (Rule 4). Hoàn Kiếm = district.",
    },

    # ── humidity_fog_query: 15 → 20 (need 5) ────────────────────
    # Design Section 8: "độ ẩm, mây, sương mù, tầm nhìn xa"
    # ROUTER_SYSTEM_PROMPT: "hỏi về ĐỘ ẨM, SƯƠNG MÙ, sương muối"
    {
        "input": "Bây giờ độ ẩm ở Cầu Giấy khoảng bao nhiêu phần trăm?",
        "context": None,
        "output": {"intent": "humidity_fog_query", "scope": "district", "confidence": 0.92},
        "_rationale": "'Độ ẩm' + 'bao nhiêu phần trăm' = explicit humidity question (Section 8). Cầu Giấy = district.",
    },
    {
        "input": "Sáng mai ở Sóc Sơn có sương mù dày không?",
        "context": None,
        "output": {"intent": "humidity_fog_query", "scope": "district", "confidence": 0.90},
        "_rationale": "'Sương mù' = fog (Section 8). Sóc Sơn = district.",
    },
    {
        "input": "do am o ha noi hom nay co cao khong?",
        "context": None,
        "output": {"intent": "humidity_fog_query", "scope": "city", "confidence": 0.89},
        "_rationale": "'do am' (no diacritics) = humidity. 'ha noi' = city scope.",
    },
    {
        "input": "Phường Yên Phụ sáng sớm có sương muối không?",
        "context": None,
        "output": {"intent": "humidity_fog_query", "scope": "ward", "confidence": 0.88},
        "_rationale": "'Sương muối' = frost/dew → humidity_fog_query (ROUTER_SYSTEM_PROMPT). 'Phường Yên Phụ' = ward.",
    },
    {
        "input": "Đêm nay ở Tây Hồ có sương mù không nhỉ?",
        "context": None,
        "output": {"intent": "humidity_fog_query", "scope": "district", "confidence": 0.91},
        "_rationale": "'Sương mù' = fog. 'Tây Hồ' = district.",
    },

    # ── location_comparison: 19 → 20 (need 1) ───────────────────
    # Design Section 10: "So sánh thời tiết giữa 2-3 quận/phường"
    {
        "input": "So sánh thời tiết hôm nay giữa Đống Đa và Hai Bà Trưng?",
        "context": None,
        "output": {"intent": "location_comparison", "scope": "district", "confidence": 0.93},
        "_rationale": "'So sánh' + 2 districts = explicit comparison (Section 10).",
    },

    # ── rain_query: 19 → 20 (need 1) ────────────────────────────
    # Design Section 5: "hỏi CÓ MƯA KHÔNG, xác suất mưa"
    # Disambiguation Rule 2: "có mưa không" = simple yes/no → rain_query
    {
        "input": "Sáng mai ở Hà Đông có khả năng mưa không?",
        "context": None,
        "output": {"intent": "rain_query", "scope": "district", "confidence": 0.91},
        "_rationale": "'Có khả năng mưa không' = rain probability (Rule 2). Hà Đông = district.",
    },

    # ── smalltalk_weather: 10 → 20 (need 10) ────────────────────
    # Design Section 14: "chào hỏi, ngoài phạm vi, câu hỏi không liên quan thời tiết"
    # tool_mapper: smalltalk_weather → [get_current_weather, get_clothing_advice]
    # Disambiguation Rule 6: greeting, bot identity, thanks → smalltalk

    # Greeting (Rule 6: "chào / hello / xin chào")
    {
        "input": "Xin chào, bạn có thể giúp gì cho mình?",
        "context": None,
        "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.95},
        "_rationale": "Greeting + bot capability question (Rule 6, Section 14). No location → city.",
    },
    # Gratitude (Rule 6: "cảm ơn / thanks")
    {
        "input": "Cảm ơn bạn nhiều nhé!",
        "context": None,
        "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.96},
        "_rationale": "Gratitude/farewell (Rule 6). No weather content.",
    },
    # Bot identity, no diacritics (Rule 6: "bạn là ai / bạn tên gì")
    {
        "input": "ban la ai vay?",
        "context": None,
        "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.94},
        "_rationale": "'ban la ai' = bot identity, no diacritics (Rule 6).",
    },
    # Non-Hanoi city → out of scope (ROUTER_SYSTEM_PROMPT: Hà Nội only)
    {
        "input": "Thời tiết ở Nha Trang hôm nay ra sao?",
        "context": None,
        "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.90},
        "_rationale": "Nha Trang = non-Hanoi → out-of-scope → smalltalk (Section 14: 'ngoài phạm vi').",
    },
    # Completely off-topic (Section 14: "câu hỏi không liên quan thời tiết")
    {
        "input": "Giá vé xe buýt ở Hà Nội bao nhiêu?",
        "context": None,
        "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.97},
        "_rationale": "Bus fare = unrelated to weather → off-topic → smalltalk (Section 14).",
    },
    # Casual weather comment (Section 14: "than phiền, hỏi mơ hồ về thời tiết")
    {
        "input": "hom nay troi dep qua nhi?",
        "context": None,
        "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.85},
        "_rationale": "Casual weather comment, no diacritics. No specific question → smalltalk (Section 14).",
    },
    # Bot capability question
    {
        "input": "Chatbot này có thể làm được những gì?",
        "context": None,
        "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.93},
        "_rationale": "Bot capability question = not weather-related → smalltalk.",
    },
    # Farewell, no diacritics (Rule 6: greeting/farewell)
    {
        "input": "tam biet nhe, hen gap lai!",
        "context": None,
        "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.95},
        "_rationale": "'tam biet' = farewell (Rule 6). No weather content.",
    },
    # Clothing advice (Section 14: "nên mặc gì"; tool_mapper → smalltalk)
    {
        "input": "Hôm nay ra ngoài nên mặc đồ gì nhỉ?",
        "context": None,
        "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.87},
        "_rationale": "'Mặc đồ gì' = clothing advice → smalltalk (Section 14 + tool_mapper: get_clothing_advice).",
    },
    # Non-Hanoi city #2 → out of scope
    {
        "input": "Thời tiết ở Quảng Ninh cuối tuần này thế nào?",
        "context": None,
        "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.91},
        "_rationale": "Quảng Ninh = non-Hanoi → out-of-scope → smalltalk.",
    },
]


def main():
    train = load(TRAIN_IN)
    val = load(VAL_IN)

    print("=" * 65)
    print("FINAL DATA CLEANING — v3 → v4")
    print("=" * 65)

    # ── B. Fix train mislabels ──────────────────────────────────
    fix_map = {q: new_intent for q, new_intent in MISLABEL_FIXES}
    fixed_count = 0
    for x in train:
        q = x["input"]
        if q in fix_map:
            old = x["output"]["intent"]
            new = fix_map[q]
            print(f"  TRAIN FIX: [{old}] → [{new}]  '{q[:60]}'")
            x["output"]["intent"] = new
            fixed_count += 1

    print(f"\n  → Fixed {fixed_count} train mislabels (expected 5)")
    assert fixed_count == 5, f"Expected 5 fixes, got {fixed_count}"

    # ── A. Remove leaked val samples ────────────────────────────
    train_nk = set(norm_key(x["input"]) for x in train)
    train_ngs = [(x["input"], char_ngrams(x["input"])) for x in train]

    leaked_indices = set()
    leak_exact = 0
    leak_fuzzy = 0

    for vi, v in enumerate(val):
        vk = norm_key(v["input"])
        if vk in train_nk:
            leaked_indices.add(vi)
            leak_exact += 1
            continue
        v_ng = char_ngrams(v["input"])
        for tq, t_ng in train_ngs:
            if jaccard(v_ng, t_ng) >= 0.85:
                leaked_indices.add(vi)
                leak_fuzzy += 1
                break

    val_clean = [v for i, v in enumerate(val) if i not in leaked_indices]
    print(f"\n  VAL LEAK REMOVAL: {len(leaked_indices)} removed "
          f"({leak_exact} exact-normalized + {leak_fuzzy} fuzzy)")
    print(f"  Val: {len(val)} → {len(val_clean)}")

    # ── C. Add new val samples ──────────────────────────────────
    # Strip _rationale field before saving (it's for documentation only)
    new_samples = []
    collisions = 0
    for s in NEW_VAL_SAMPLES:
        clean = {k: v for k, v in s.items() if k != "_rationale"}
        nk = norm_key(clean["input"])
        # Verify no collision with train
        if nk in train_nk:
            print(f"  ⚠️ COLLISION with train: '{clean['input'][:50]}' — SKIPPED")
            collisions += 1
            continue
        # Verify no collision with existing val_clean
        if any(norm_key(v["input"]) == nk for v in val_clean):
            print(f"  ⚠️ COLLISION with val: '{clean['input'][:50]}' — SKIPPED")
            collisions += 1
            continue
        new_samples.append(clean)

    val_final = val_clean + new_samples
    print(f"\n  NEW VAL SAMPLES: {len(new_samples)} added ({collisions} collisions skipped)")
    print(f"  Val final: {len(val_clean)} + {len(new_samples)} = {len(val_final)}")

    # ── Verify: all intents ≥ 20 in val ─────────────────────────
    val_intents = Counter(x["output"]["intent"] for x in val_final)
    print(f"\n  VAL PER-INTENT COUNTS (final):")
    weak = []
    for intent in sorted(val_intents):
        c = val_intents[intent]
        flag = " ⚠ <20" if c < 20 else " ✓"
        print(f"    {intent:<22} {c:>4}{flag}")
        if c < 20:
            weak.append(intent)

    if weak:
        print(f"\n  ⚠️ {len(weak)} intents still <20: {weak}")
    else:
        print(f"\n  ✅ All 15 intents have ≥20 val samples")

    # ── Final leakage verification ──────────────────────────────
    train_nk_final = set(norm_key(x["input"]) for x in train)
    train_ngs_final = [(x["input"], char_ngrams(x["input"])) for x in train]
    remaining_leaks = 0
    for v in val_final:
        vk = norm_key(v["input"])
        if vk in train_nk_final:
            remaining_leaks += 1
            continue
        v_ng = char_ngrams(v["input"])
        for _, t_ng in train_ngs_final:
            if jaccard(v_ng, t_ng) >= 0.85:
                remaining_leaks += 1
                break

    print(f"\n  FINAL LEAK CHECK: {remaining_leaks} (must be 0)")
    assert remaining_leaks == 0, f"Still {remaining_leaks} leaks!"

    # ── Save outputs ────────────────────────────────────────────
    save(train, TRAIN_OUT)
    save(val_final, VAL_OUT)

    # ── Summary report ──────────────────────────────────────────
    train_intents = Counter(x["output"]["intent"] for x in train)
    train_scopes = Counter(x["output"]["scope"] for x in train)
    val_scopes = Counter(x["output"]["scope"] for x in val_final)

    report = {
        "version": "v4-clean",
        "actions": {
            "train_mislabels_fixed": fixed_count,
            "val_leaked_removed": len(leaked_indices),
            "val_leaked_exact_normalized": leak_exact,
            "val_leaked_fuzzy": leak_fuzzy,
            "val_new_samples_added": len(new_samples),
            "collisions_skipped": collisions,
        },
        "final_counts": {
            "train": len(train),
            "val": len(val_final),
        },
        "train_distribution": {
            "intents": dict(sorted(train_intents.items(), key=lambda kv: -kv[1])),
            "scopes": dict(sorted(train_scopes.items(), key=lambda kv: -kv[1])),
        },
        "val_distribution": {
            "intents": dict(sorted(val_intents.items(), key=lambda kv: -kv[1])),
            "scopes": dict(sorted(val_scopes.items(), key=lambda kv: -kv[1])),
        },
        "val_all_intents_gte_20": len(weak) == 0,
        "final_leak_count": remaining_leaks,
    }

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 65}")
    print(f"OUTPUTS:")
    print(f"  {TRAIN_OUT}  ({len(train)} samples)")
    print(f"  {VAL_OUT}  ({len(val_final)} samples)")
    print(f"  {REPORT}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
