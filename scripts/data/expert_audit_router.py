"""
scripts/data/expert_audit_router.py
Expert-level data science audit on cleaned router train/val data.

Goes beyond basic dedupe/leakage to check:
  1. Near-duplicate fuzzy detection (char n-gram Jaccard)
  2. Intent class balance + minority check
  3. Scope × Intent co-distribution
  4. Query length distribution per intent
  5. Diacritic-vs-ASCII split (style diversity)
  6. Confidence field: synthetic pattern check
  7. Context field: multi-turn coverage
  8. Train↔val leakage: exact + fuzzy
  9. Per-intent val coverage (can we evaluate every class?)
 10. Heuristic mislabel flagging (rule-based, not semantic)
"""

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/router/multitask_train_v3_clean.jsonl"
VAL = ROOT / "data/router/multitask_val_v3_clean.jsonl"
OUT = ROOT / "data/router/expert_audit_report.json"


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def strip_diacritics(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def has_diacritics(s: str) -> bool:
    return strip_diacritics(s) != s


def norm_key(q: str) -> str:
    s = strip_diacritics(q).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def char_ngrams(s: str, n: int = 4) -> set:
    s = norm_key(s)
    if len(s) < n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ─── 1. Basic stats ─────────────────────────────────────────────
def basic_stats(name, data):
    intents = Counter(x["output"]["intent"] for x in data)
    scopes = Counter(x["output"]["scope"] for x in data)
    n = len(data)
    return {
        "name": name,
        "n": n,
        "intents": dict(sorted(intents.items(), key=lambda kv: -kv[1])),
        "intent_min_pct": round(min(intents.values()) / n * 100, 2),
        "intent_max_pct": round(max(intents.values()) / n * 100, 2),
        "intent_imbalance_ratio": round(max(intents.values()) / min(intents.values()), 2),
        "scopes": dict(scopes),
        "scope_min_pct": round(min(scopes.values()) / n * 100, 2),
    }


# ─── 2. Length profile ──────────────────────────────────────────
def length_profile(data):
    lens = defaultdict(list)
    for x in data:
        lens[x["output"]["intent"]].append(len(x["input"]))
    return {
        k: {
            "mean": round(sum(v) / len(v), 1),
            "min": min(v),
            "max": max(v),
            "n": len(v),
        }
        for k, v in sorted(lens.items())
    }


# ─── 3. Diacritic split ─────────────────────────────────────────
def diacritic_split(data):
    dia = sum(1 for x in data if has_diacritics(x["input"]))
    return {
        "with_diacritics": dia,
        "ascii_only": len(data) - dia,
        "ascii_pct": round((len(data) - dia) / len(data) * 100, 1),
    }


# ─── 4. Confidence field ────────────────────────────────────────
def confidence_profile(data):
    confs = [x["output"].get("confidence", None) for x in data]
    confs = [c for c in confs if c is not None]
    if not confs:
        return {"note": "no confidence field"}
    dist = Counter(round(c, 2) for c in confs)
    return {
        "n": len(confs),
        "unique_values": len(dist),
        "min": min(confs),
        "max": max(confs),
        "mean": round(sum(confs) / len(confs), 3),
        "mode_val": dist.most_common(1)[0][0],
        "mode_pct": round(dist.most_common(1)[0][1] / len(confs) * 100, 1),
        "top5": dist.most_common(5),
    }


# ─── 5. Context field ───────────────────────────────────────────
def context_profile(data):
    has_ctx = sum(1 for x in data if x.get("context"))
    rewritten = sum(1 for x in data if x["output"].get("rewritten_query"))
    return {
        "with_context": has_ctx,
        "with_rewritten": rewritten,
        "context_pct": round(has_ctx / len(data) * 100, 1),
        "rewritten_pct": round(rewritten / len(data) * 100, 1),
    }


# ─── 6. Fuzzy train↔val leakage (char n-gram jaccard ≥ 0.85) ────
def fuzzy_leakage(train, val, threshold=0.85):
    # Precompute val norm sets (smaller)
    val_norm = [(i, v["input"], char_ngrams(v["input"])) for i, v in enumerate(val)]
    # For train, precompute on the fly but short-circuit via exact key
    train_keys = {norm_key(t["input"]) for t in train}

    near_dupes = []
    seen_exact = 0
    for vi, vq, v_ng in val_norm:
        if norm_key(vq) in train_keys:
            seen_exact += 1
            continue
        # Scan train for near match — cap at first 500 to avoid O(N*M) blow-up
        # but we want completeness, so do full scan
        best_j = 0.0
        best_q = None
        for t in train:
            j = jaccard(v_ng, char_ngrams(t["input"]))
            if j > best_j:
                best_j = j
                best_q = t["input"]
                if j >= 0.99:
                    break
        if best_j >= threshold:
            near_dupes.append(
                {"val": vq, "train": best_q, "jaccard": round(best_j, 3)}
            )
    return {
        "exact_overlap": seen_exact,
        "near_duplicate_count": len(near_dupes),
        "threshold": threshold,
        "sample": near_dupes[:15],
    }


# ─── 7. Internal near-dupes (train only) ────────────────────────
def internal_near_dupes(data, threshold=0.90, sample_limit=20):
    # Bucket by intent to reduce compares
    by_intent = defaultdict(list)
    for x in data:
        by_intent[x["output"]["intent"]].append(x["input"])

    dupes = []
    for intent, queries in by_intent.items():
        ngs = [char_ngrams(q) for q in queries]
        n = len(queries)
        for i in range(n):
            for j in range(i + 1, n):
                jj = jaccard(ngs[i], ngs[j])
                if jj >= threshold:
                    dupes.append(
                        {
                            "intent": intent,
                            "q1": queries[i],
                            "q2": queries[j],
                            "jaccard": round(jj, 3),
                        }
                    )
                    if len(dupes) >= 500:
                        break
            if len(dupes) >= 500:
                break
        if len(dupes) >= 500:
            break
    return {
        "threshold": threshold,
        "count": len(dupes),
        "sample": dupes[:sample_limit],
    }


# ─── 8. Heuristic mislabel flagging ─────────────────────────────
MISLABEL_RULES = [
    # (rule_name, regex, expected_intent, description)
    (
        "out_of_hanoi_city",
        re.compile(
            r"\b(đà nẵng|da nang|hải phòng|hai phong|sài gòn|sai gon|hcm|hồ chí minh|đà lạt|da lat|huế|hue|tokyo|bangkok|seoul)\b",
            re.I,
        ),
        None,
        "Non-Hanoi city → should be out-of-scope / rejected",
    ),
    (
        "off_topic_time",
        re.compile(r"\b(thứ mấy|mấy giờ|thu may|may gio)\b", re.I),
        None,
        "Off-topic (time/day question, not weather)",
    ),
    (
        "clothing_advice",
        re.compile(
            r"\b(mang theo ô|can mang o|nên mặc|nen mac|mặc gì|mac gi|mang áo khoác|ao khoac co can|cần áo khoác|can ao khoac|cần ô|can o|chuan bi ao)\b",
            re.I,
        ),
        "activity_weather",
        "Clothing / umbrella advice → activity_weather",
    ),
    (
        "seasonal_mua_nay",
        re.compile(r"\b(mùa này|mua nay)\b", re.I),
        "seasonal_context",
        "‘mùa này’ phrase → seasonal_context",
    ),
    (
        "explicit_wind_speed",
        re.compile(
            r"\b(tốc độ gió|toc do gio|gió mạnh|gio manh|km/h)\b",
            re.I,
        ),
        "wind_query",
        "Wind speed / strength → wind_query",
    ),
    (
        "explicit_humidity",
        re.compile(r"\b(độ ẩm|do am|sương mù|suong mu)\b", re.I),
        "humidity_fog_query",
        "Humidity / fog → humidity_fog_query",
    ),
    (
        "explicit_uv_dewpoint",
        re.compile(
            r"\b(tia uv|uv|áp suất|ap suat|điểm sương|diem suong|tầm nhìn|tam nhin)\b",
            re.I,
        ),
        "expert_weather_param",
        "Technical parameter → expert_weather_param",
    ),
]


def heuristic_mislabels(data):
    findings = defaultdict(list)
    for x in data:
        q = x["input"]
        lbl = x["output"]["intent"]
        for rule_name, pattern, expected, desc in MISLABEL_RULES:
            if pattern.search(q):
                if expected is None:
                    findings[rule_name].append(
                        {"query": q, "label": lbl, "issue": desc}
                    )
                elif lbl != expected and lbl != "smalltalk_weather":
                    # Skip if already correct
                    pass
                elif lbl != expected:
                    findings[rule_name].append(
                        {
                            "query": q,
                            "label": lbl,
                            "expected": expected,
                            "issue": desc,
                        }
                    )
    return {
        k: {"count": len(v), "sample": v[:10]} for k, v in findings.items()
    }


# ─── 9. Per-intent val coverage ─────────────────────────────────
def val_coverage(train, val):
    train_intents = Counter(x["output"]["intent"] for x in train)
    val_intents = Counter(x["output"]["intent"] for x in val)
    all_intents = set(train_intents) | set(val_intents)
    rows = []
    for i in sorted(all_intents):
        t = train_intents.get(i, 0)
        v = val_intents.get(i, 0)
        rows.append(
            {
                "intent": i,
                "train": t,
                "val": v,
                "ratio": round(t / v, 1) if v else None,
                "val_pct": round(v / sum(val_intents.values()) * 100, 2),
                "adequate_for_f1": v >= 20,  # heuristic: <20 val samples = noisy F1
            }
        )
    return rows


# ─── 10. Input field integrity ──────────────────────────────────
def integrity_check(data):
    issues = {
        "empty_input": 0,
        "whitespace_only": 0,
        "very_short_lt3": 0,
        "very_long_gt200": 0,
        "missing_intent": 0,
        "missing_scope": 0,
        "invalid_scope": 0,
        "null_confidence": 0,
    }
    valid_scopes = {"city", "district", "ward"}
    for x in data:
        q = x.get("input", "")
        if not q:
            issues["empty_input"] += 1
        elif not q.strip():
            issues["whitespace_only"] += 1
        elif len(q.strip()) < 3:
            issues["very_short_lt3"] += 1
        elif len(q) > 200:
            issues["very_long_gt200"] += 1
        out = x.get("output", {})
        if not out.get("intent"):
            issues["missing_intent"] += 1
        if not out.get("scope"):
            issues["missing_scope"] += 1
        elif out["scope"] not in valid_scopes:
            issues["invalid_scope"] += 1
        if out.get("confidence") is None:
            issues["null_confidence"] += 1
    return issues


def main():
    train = load(TRAIN)
    val = load(VAL)

    print("Running expert audit ...")
    report = {
        "files": {"train": str(TRAIN.name), "val": str(VAL.name)},
        "basic_stats": {
            "train": basic_stats("train", train),
            "val": basic_stats("val", val),
        },
        "integrity": {
            "train": integrity_check(train),
            "val": integrity_check(val),
        },
        "length_profile_train": length_profile(train),
        "length_profile_val": length_profile(val),
        "diacritic_split": {
            "train": diacritic_split(train),
            "val": diacritic_split(val),
        },
        "confidence": {
            "train": confidence_profile(train),
            "val": confidence_profile(val),
        },
        "context": {
            "train": context_profile(train),
            "val": context_profile(val),
        },
        "val_coverage": val_coverage(train, val),
    }

    print("  → fuzzy leakage check (slow) ...")
    report["fuzzy_leakage"] = fuzzy_leakage(train, val, threshold=0.85)

    print("  → internal near-dupes (train) ...")
    report["internal_near_dupes_train"] = internal_near_dupes(train, threshold=0.90)

    print("  → heuristic mislabels ...")
    report["heuristic_mislabels_train"] = heuristic_mislabels(train)
    report["heuristic_mislabels_val"] = heuristic_mislabels(val)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ─── PRINT SUMMARY ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERT AUDIT — SUMMARY")
    print("=" * 70)

    bs_t = report["basic_stats"]["train"]
    bs_v = report["basic_stats"]["val"]
    print(f"\nTrain: {bs_t['n']} samples | Val: {bs_v['n']} samples")
    print(
        f"Intent imbalance ratio — train: {bs_t['intent_imbalance_ratio']}x "
        f"| val: {bs_v['intent_imbalance_ratio']}x"
    )
    print(
        f"Intent min/max pct — train: {bs_t['intent_min_pct']}% / {bs_t['intent_max_pct']}% "
        f"| val: {bs_v['intent_min_pct']}% / {bs_v['intent_max_pct']}%"
    )

    print("\n── Integrity ──")
    for k, v in report["integrity"]["train"].items():
        if v > 0:
            print(f"  TRAIN {k}: {v}")
    for k, v in report["integrity"]["val"].items():
        if v > 0:
            print(f"  VAL   {k}: {v}")

    print("\n── Diacritic split ──")
    ds_t = report["diacritic_split"]["train"]
    ds_v = report["diacritic_split"]["val"]
    print(f"  Train: {ds_t['ascii_pct']}% ASCII-only ({ds_t['ascii_only']}/{bs_t['n']})")
    print(f"  Val:   {ds_v['ascii_pct']}% ASCII-only ({ds_v['ascii_only']}/{bs_v['n']})")

    print("\n── Fuzzy leakage (train ← val, jaccard ≥ 0.85) ──")
    fl = report["fuzzy_leakage"]
    print(f"  exact overlap:    {fl['exact_overlap']}")
    print(f"  near-duplicates:  {fl['near_duplicate_count']}")
    for s in fl["sample"][:5]:
        print(f"    [{s['jaccard']}] val: '{s['val'][:55]}'")
        print(f"              train: '{s['train'][:55]}'")

    print("\n── Internal near-dupes (train, jaccard ≥ 0.90) ──")
    ind = report["internal_near_dupes_train"]
    print(f"  count: {ind['count']}")
    for s in ind["sample"][:5]:
        print(f"    [{s['intent']}] {s['q1'][:50]}  ↔  {s['q2'][:50]}")

    print("\n── Heuristic mislabels (TRAIN) ──")
    for rule, info in report["heuristic_mislabels_train"].items():
        if info["count"]:
            print(f"  {rule}: {info['count']}")
            for s in info["sample"][:3]:
                print(f"    [{s.get('label', '?')}] {s['query'][:60]}")

    print("\n── Heuristic mislabels (VAL) ──")
    for rule, info in report["heuristic_mislabels_val"].items():
        if info["count"]:
            print(f"  {rule}: {info['count']}")
            for s in info["sample"][:3]:
                print(f"    [{s.get('label', '?')}] {s['query'][:60]}")

    print("\n── Val coverage per intent (adequate_for_f1: val ≥ 20) ──")
    weak = [r for r in report["val_coverage"] if not r["adequate_for_f1"]]
    if weak:
        print(f"  ⚠️  {len(weak)} intents have <20 val samples — noisy F1:")
        for r in weak:
            print(f"    {r['intent']}: val={r['val']}, train={r['train']}")
    else:
        print("  ✅ All intents have ≥20 val samples")

    print("\n── Confidence distribution (train) ──")
    c = report["confidence"]["train"]
    print(
        f"  {c['n']} samples, mean={c['mean']}, "
        f"mode={c['mode_val']} ({c['mode_pct']}%), unique={c['unique_values']}"
    )
    print(f"  top5: {c['top5']}")

    print("\n── Context coverage ──")
    ct = report["context"]["train"]
    cv = report["context"]["val"]
    print(f"  Train: {ct['context_pct']}% context, {ct['rewritten_pct']}% rewritten")
    print(f"  Val:   {cv['context_pct']}% context, {cv['rewritten_pct']}% rewritten")

    print(f"\nFull report: {OUT}")


if __name__ == "__main__":
    main()
