"""
scripts/data/clean_router_v5.py
Final expert-level cleaning — v4 → v5.

Four actions:
  1. Relabel ~24 umbrella queries: smalltalk_weather → rain_query
     (disambiguation rules: "mang ô / cần mang áo mưa" → rain_query)
  2. Relabel 3 "mặc gì" queries: activity_weather → smalltalk_weather
     (design doc Section 14: generic clothing advice = smalltalk)
  3. Relabel 3 weather_alert mislabels: hourly_forecast/temperature_query → weather_alert
     ("bão", "giông", "rét đậm" are alert triggers per anti-confusion rules)
  4. Augment smalltalk_weather with 55 genuine samples
     (greetings, farewell, identity, OOS cities, off-topic, casual comments)
  5. Add 5 val samples for smalltalk + full leak verification

References:
  - docs/Weather Intent Design.md (Section 14: smalltalk definition)
  - docs/intent_disambiguation_rules.md (Rule 2: umbrella→rain, Rule 6: smalltalk)
  - app/agent/router/config.py (anti-confusion: bão/giông→alert, bây giờ→current)
  - app/agent/router/tool_mapper.py (smalltalk tools include get_clothing_advice)

Input:
  data/router/multitask_train_v4_clean.jsonl  (3302)
  data/router/multitask_val_v4_clean.jsonl    (365)

Output:
  data/router/multitask_train_v5_clean.jsonl  (~3357)
  data/router/multitask_val_v5_clean.jsonl    (~370)
  data/router/expert_audit_report.json        (final stats)
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_IN = ROOT / "data/router/multitask_train_v4_clean.jsonl"
VAL_IN = ROOT / "data/router/multitask_val_v4_clean.jsonl"
TRAIN_OUT = ROOT / "data/router/multitask_train_v5_clean.jsonl"
VAL_OUT = ROOT / "data/router/multitask_val_v5_clean.jsonl"
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
# 1. UMBRELLA QUERIES: smalltalk_weather → rain_query
# ═══════════════════════════════════════════════════════════════════
# Pattern: "mang ô", "cần ô", "phòng mưa", "đem ô", "thủ ô", "chuẩn bị ô"
# Disambiguation rules: "mang ô / cần mang áo mưa" → rain_query
# Data evidence: 11/30 already labeled rain_query (majority vote)
UMBRELLA_PATTERNS = [
    # Vietnamese diacritic — match "ô" as whole word (umbrella)
    r"(?:mang|cần|đem|thủ|chuẩn bị)\b.*\bô\b",
    r"\bô\b\s+(?:không|khong)",
    r"phòng mưa",
    # Non-diacritic — match "o" as standalone word ONLY (not part of "ao khoac")
    r"(?:mang|can|dem|thu|chuan bi)\b.*\b(?<![a-z])o\b(?![a-z])",
    r"\bphong mua\b",
]

# Clothing keywords — EXCLUDE these from umbrella matching
_CLOTHING_EXCLUDE = [
    "áo khoác", "ao khoac", "áo phao", "ao phao", "áo mỏng", "ao mong",
    "áo dày", "ao day", "mặc", "mac ", "trang phục", "trang phuc",
]


def is_umbrella_query(q: str) -> bool:
    q_lower = q.lower()
    # Exclude clothing queries — these have "áo khoác", "mặc gì", etc.
    if any(kw in q_lower for kw in _CLOTHING_EXCLUDE):
        return False
    for pat in UMBRELLA_PATTERNS:
        if re.search(pat, q_lower):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════
# 2. "MẶC GÌ" QUERIES: activity_weather → smalltalk_weather
# ═══════════════════════════════════════════════════════════════════
# These 3 are generic clothing queries without specific outdoor activity.
# Design doc Section 14: "nên mặc gì" = smalltalk_weather
# tool_mapper: smalltalk_weather includes get_clothing_advice
CLOTHING_RELABEL = [
    "ban co the tu van hom nay nen mac gi khong?",         # L2206
    "hom nay mac gi di lam?",                               # L2276
    "o dai hoc bach khoa hom nay mac gi cho phu hop?",     # L2732
]


# ═══════════════════════════════════════════════════════════════════
# 3. WEATHER ALERT MISLABELS
# ═══════════════════════════════════════════════════════════════════
# Anti-confusion rules: bão/giông/rét đậm → weather_alert
# Safety rule: "miss an alert >> false positive alert"
ALERT_RELABEL = {
    # query → (old_intent, new_intent)
    "Hôm nay ở phường Xuân Đỉnh, thời tiết từng giờ có mưa hay bão gì không?":
        ("hourly_forecast", "weather_alert"),   # L1224: "bão" trigger
    "Trong 3 giờ tới ở phường Xuân Đỉnh có mưa rào hay giông không?":
        ("hourly_forecast", "weather_alert"),   # L2261: "giông" trigger
    "Hà Nội mấy ngày tới có rét đậm không, nhiệt độ khoảng bao nhiêu?":
        ("temperature_query", "weather_alert"), # L3257: "rét đậm" trigger
}


# ═══════════════════════════════════════════════════════════════════
# 4. SMALLTALK AUGMENTATION (55 hand-crafted samples)
# ═══════════════════════════════════════════════════════════════════
# After moving ~24 umbrella out, smalltalk drops to ~85.
# Adding 55 genuine samples → ~140 total.
# Categories: greeting(10), farewell(10), identity(10), OOS(10), off-topic(10), casual(5)

SMALLTALK_AUGMENT = [
    # ── Greetings (10) ─────────────────────────────────────────
    # Rule 6: "chào / hello / xin chào"
    {"input": "Chào buổi sáng!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.96}},
    {"input": "Hello chatbot!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.95}},
    {"input": "Bạn ơi, chào nhé!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.94}},
    {"input": "chao buoi chieu!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.93}},
    {"input": "Chào bạn, mình cần hỏi thời tiết!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.92}},
    {"input": "Hey, có ai không?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.91}},
    {"input": "Chào buổi tối nhé!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.95}},
    {"input": "hi ban", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.94}},
    {"input": "Ê, có đó không?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.90}},
    {"input": "Chào bạn thời tiết!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.93}},

    # ── Farewell/Thanks (10) ───────────────────────────────────
    # Rule 6: "tạm biệt / cảm ơn / thanks"
    {"input": "Cảm ơn nhé, mình biết rồi!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.96}},
    {"input": "ok cam on ban nhieu!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.95}},
    {"input": "Hẹn gặp lại nhé bạn!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.94}},
    {"input": "Bye bye, cảm ơn thông tin!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.95}},
    {"input": "Thanks nha!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.93}},
    {"input": "Tạm biệt, mai mình hỏi tiếp!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.94}},
    {"input": "ok roi, cam on ban!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.93}},
    {"input": "Cảm ơn bạn tư vấn thời tiết!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.96}},
    {"input": "Mình hiểu rồi, cảm ơn!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.95}},
    {"input": "thoi tam biet, hen gap lai!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.94}},

    # ── Bot Identity/Capability (10) ──────────────────────────
    # Rule 6: "bạn là ai / bạn tên gì / giúp gì"
    {"input": "Bạn biết dự báo được mấy ngày?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.91}},
    {"input": "Ai tạo ra bạn vậy?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.93}},
    {"input": "ban co thong minh khong?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.90}},
    {"input": "Bạn có thể dự báo chính xác không?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.91}},
    {"input": "Chatbot thời tiết này hoạt động thế nào?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.92}},
    {"input": "ban ten gi?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.94}},
    {"input": "Bạn lấy dữ liệu thời tiết từ đâu?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.89}},
    {"input": "Mình hỏi bạn về giao thông được không?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.93}},
    {"input": "bot nay biet nhung gi?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.91}},
    {"input": "Bạn chỉ biết thời tiết Hà Nội thôi hả?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.92}},

    # ── Out-of-scope cities (10) ─────────────────────────────
    # Section 14: "ngoài phạm vi" → smalltalk
    {"input": "Thời tiết ở Sapa hôm nay thế nào?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.91}},
    {"input": "Huế có mưa không bạn?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.90}},
    {"input": "Phú Quốc cuối tuần này nắng không?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.89}},
    {"input": "thoi tiet can tho hom nay?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.90}},
    {"input": "Vinh có lạnh không bạn ơi?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.88}},
    {"input": "Dự báo thời tiết Quy Nhơn ngày mai?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.91}},
    {"input": "Buôn Ma Thuột có nóng lắm không?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.87}},
    {"input": "nhiet do o hue bao nhieu do?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.90}},
    {"input": "Mưa ở Cần Thơ có to không?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.89}},
    {"input": "Thời tiết ở Hạ Long hôm nay ra sao?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.90}},

    # ── Off-topic (10) ───────────────────────────────────────
    # Section 14: "câu hỏi không liên quan thời tiết"
    {"input": "Giá xăng hôm nay bao nhiêu?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.97}},
    {"input": "Tỉ giá USD/VND hôm nay thế nào?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.97}},
    {"input": "co quan an ngon nao o hoan kiem khong?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.96}},
    {"input": "Lịch thi đấu bóng đá hôm nay?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.97}},
    {"input": "Đường đến sân bay Nội Bài đi thế nào?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.96}},
    {"input": "gia vang hom nay la bao nhieu?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.97}},
    {"input": "Cho mình hỏi số điện thoại cứu hộ?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.95}},
    {"input": "Bạn có biết nhà hàng nào ngon ở Tây Hồ không?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.96}},
    {"input": "xe bus so 32 di qua nhung dau?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.96}},
    {"input": "Hôm nay có kẹt xe ở đường nào không?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.95}},

    # ── Casual weather comments (5) ──────────────────────────
    {"input": "Ôi trời nóng kinh khủng!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.86}},
    {"input": "Gió mát quá đi!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.85}},
    {"input": "mua hoai mua suot a?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.84}},
    {"input": "Sao dạo này trời cứ âm u thế nhỉ?", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.85}},
    {"input": "Trời hôm nay tuyệt vời ghê!", "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.86}},
]


# ═══════════════════════════════════════════════════════════════════
# 5. NEW VAL SAMPLES (5 for smalltalk_weather)
# ═══════════════════════════════════════════════════════════════════
NEW_VAL_SAMPLES = [
    {"input": "Chào buổi chiều, bạn giúp mình được không?", "context": None,
     "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.94}},
    {"input": "Thời tiết Đà Lạt có lạnh lắm không?", "context": None,
     "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.90}},
    {"input": "hom nay gia xang tang hay giam?", "context": None,
     "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.97}},
    {"input": "Bạn có thể dự báo thời tiết Sapa không?", "context": None,
     "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.91}},
    {"input": "ok minh hieu roi, cam on ban nhieu!", "context": None,
     "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.95}},
]


def main():
    train = load(TRAIN_IN)
    val = load(VAL_IN)

    print("=" * 65)
    print("FINAL DATA CLEANING — v4 → v5")
    print("=" * 65)

    # ── 1. Umbrella relabels ────────────────────────────────────
    umbrella_fixed = 0
    for x in train:
        if x["output"]["intent"] == "smalltalk_weather" and is_umbrella_query(x["input"]):
            print(f"  UMBRELLA → rain_query: '{x['input'][:65]}'")
            x["output"]["intent"] = "rain_query"
            umbrella_fixed += 1

    print(f"\n  → {umbrella_fixed} umbrella queries: smalltalk → rain_query")

    # ── 2. Clothing relabels ────────────────────────────────────
    clothing_fixed = 0
    clothing_set = set(CLOTHING_RELABEL)
    for x in train:
        if x["input"] in clothing_set and x["output"]["intent"] == "activity_weather":
            print(f"  CLOTHING → smalltalk: '{x['input'][:65]}'")
            x["output"]["intent"] = "smalltalk_weather"
            clothing_fixed += 1

    print(f"\n  → {clothing_fixed} clothing queries: activity_weather → smalltalk")
    assert clothing_fixed == 3, f"Expected 3, got {clothing_fixed}"

    # ── 3. Alert relabels ───────────────────────────────────────
    alert_fixed = 0
    for x in train:
        q = x["input"]
        if q in ALERT_RELABEL:
            old_intent, new_intent = ALERT_RELABEL[q]
            assert x["output"]["intent"] == old_intent, \
                f"Expected {old_intent}, got {x['output']['intent']} for '{q[:50]}'"
            print(f"  ALERT FIX: [{old_intent}] → [{new_intent}]: '{q[:55]}'")
            x["output"]["intent"] = new_intent
            alert_fixed += 1

    print(f"\n  → {alert_fixed} alert mislabels fixed")
    assert alert_fixed == 3, f"Expected 3, got {alert_fixed}"

    # ── 4. Smalltalk augmentation ───────────────────────────────
    train_nk = set(norm_key(x["input"]) for x in train)
    augmented = 0
    collisions = 0
    for s in SMALLTALK_AUGMENT:
        sample = {"input": s["input"], "context": None, "output": s["output"]}
        nk = norm_key(sample["input"])
        if nk in train_nk:
            print(f"  ⚠️ COLLISION: '{sample['input'][:50]}' — SKIPPED")
            collisions += 1
            continue
        train.append(sample)
        train_nk.add(nk)
        augmented += 1

    print(f"\n  → {augmented} smalltalk samples added ({collisions} collisions)")

    # ── 5. Val augmentation ─────────────────────────────────────
    val_collisions = 0
    val_added = 0
    for s in NEW_VAL_SAMPLES:
        nk = norm_key(s["input"])
        if nk in train_nk:
            print(f"  ⚠️ VAL COLLISION with train: '{s['input'][:50]}' — SKIPPED")
            val_collisions += 1
            continue
        if any(norm_key(v["input"]) == nk for v in val):
            print(f"  ⚠️ VAL COLLISION with val: '{s['input'][:50]}' — SKIPPED")
            val_collisions += 1
            continue
        val.append(s)
        val_added += 1

    print(f"\n  → {val_added} val samples added ({val_collisions} collisions)")

    # ── Final verification ──────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("VERIFICATION")
    print(f"{'=' * 65}")

    # Zero duplicates in train
    train_inputs = [x["input"] for x in train]
    dupes = len(train_inputs) - len(set(train_inputs))
    print(f"\n  Train exact duplicates: {dupes}")
    assert dupes == 0, f"{dupes} duplicates found!"

    # Zero leakage
    final_train_nk = set(norm_key(x["input"]) for x in train)
    final_train_ngs = [(x["input"], char_ngrams(x["input"])) for x in train]

    exact_leaks = 0
    fuzzy_leaks = 0
    for v in val:
        vk = norm_key(v["input"])
        if vk in final_train_nk:
            exact_leaks += 1
            print(f"  ⚠️ EXACT LEAK: '{v['input'][:60]}'")
            continue
        v_ng = char_ngrams(v["input"])
        for _, t_ng in final_train_ngs:
            if jaccard(v_ng, t_ng) >= 0.85:
                fuzzy_leaks += 1
                break

    print(f"  Exact leaks: {exact_leaks}")
    print(f"  Fuzzy leaks (jaccard≥0.85): {fuzzy_leaks}")
    assert exact_leaks == 0 and fuzzy_leaks == 0, "Leakage detected!"

    # Per-intent counts
    train_intents = Counter(x["output"]["intent"] for x in train)
    val_intents = Counter(x["output"]["intent"] for x in val)

    print(f"\n  TRAIN PER-INTENT ({len(train)} total):")
    for intent in sorted(train_intents, key=lambda x: -train_intents[x]):
        c = train_intents[intent]
        ratio = c / len(train) * 100
        print(f"    {intent:<22} {c:>5} ({ratio:>5.1f}%)")

    print(f"\n  VAL PER-INTENT ({len(val)} total):")
    weak_val = []
    for intent in sorted(val_intents):
        c = val_intents[intent]
        flag = " ⚠ <20" if c < 20 else " ✓"
        print(f"    {intent:<22} {c:>4}{flag}")
        if c < 20:
            weak_val.append(intent)

    if weak_val:
        print(f"\n  ⚠️ {len(weak_val)} val intents <20: {weak_val}")
    else:
        print(f"\n  ✅ All 15 val intents ≥20")

    # Scope distribution
    train_scopes = Counter(x["output"]["scope"] for x in train)
    val_scopes = Counter(x["output"]["scope"] for x in val)
    print(f"\n  Train scopes: {dict(train_scopes)}")
    print(f"  Val scopes:   {dict(val_scopes)}")

    # Max/min ratio
    max_intent = max(train_intents.values())
    min_intent = min(train_intents.values())
    ratio = max_intent / min_intent
    print(f"\n  Train max/min intent ratio: {max_intent}/{min_intent} = {ratio:.1f}x", end="")
    if ratio <= 2.5:
        print(" ✓")
    else:
        print(f" ⚠ >2.5x — consider augmenting {min(train_intents, key=train_intents.get)}")

    # ── Save outputs ────────────────────────────────────────────
    save(train, TRAIN_OUT)
    save(val, VAL_OUT)

    # ── Report ──────────────────────────────────────────────────
    report = {
        "version": "v5-clean-final",
        "parent": "v4-clean",
        "actions": {
            "umbrella_relabeled": umbrella_fixed,
            "clothing_relabeled": clothing_fixed,
            "alert_relabeled": alert_fixed,
            "smalltalk_augmented": augmented,
            "val_added": val_added,
            "collisions_skipped": collisions + val_collisions,
        },
        "final_counts": {
            "train": len(train),
            "val": len(val),
        },
        "train_distribution": {
            "intents": dict(sorted(train_intents.items(), key=lambda kv: -kv[1])),
            "scopes": dict(sorted(train_scopes.items(), key=lambda kv: -kv[1])),
        },
        "val_distribution": {
            "intents": dict(sorted(val_intents.items(), key=lambda kv: -kv[1])),
            "scopes": dict(sorted(val_scopes.items(), key=lambda kv: -kv[1])),
        },
        "quality_checks": {
            "train_duplicates": dupes,
            "exact_leaks": exact_leaks,
            "fuzzy_leaks": fuzzy_leaks,
            "val_all_intents_gte_20": len(weak_val) == 0,
            "train_max_min_ratio": round(ratio, 2),
        },
        "known_limitations": [
            "47.79% of confidence values are exactly 0.90 (synthetic artifact)",
            "13.96% context.intent != output.intent (expected: multi-turn intent shifts)",
            "Weather comment boundary: some 'trời đẹp không?' in smalltalk could be current_weather",
        ],
    }

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 65}")
    print("OUTPUTS:")
    print(f"  {TRAIN_OUT}  ({len(train)} samples)")
    print(f"  {VAL_OUT}  ({len(val)} samples)")
    print(f"  {REPORT}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
