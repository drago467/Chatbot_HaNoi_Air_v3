"""
scripts/data/audit_router_data.py
Router data audit for Exp6 — dedupe train, verify val disjoint.

Input:
  data/router/multitask_train_v3.jsonl         (4005)
  data/router/multitask_val_v3.jsonl           (672, 35.4% leakage)
  data/router/multitask_val_v3_clean.jsonl     (434, already disjoint)

Output:
  data/router/multitask_train_v3_clean.jsonl   (deduped train)
  data/router/audit_report.json                (metrics + findings)
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data/router/multitask_train_v3.jsonl"
VAL_ORIG_PATH = ROOT / "data/router/multitask_val_v3.jsonl"
VAL_CLEAN_PATH = ROOT / "data/router/multitask_val_v3_clean.jsonl"
TRAIN_CLEAN_PATH = ROOT / "data/router/multitask_train_v3_clean.jsonl"
REPORT_PATH = ROOT / "data/router/audit_report.json"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def norm(q: str) -> str:
    return q.strip().lower()


def dedupe_train(train: list[dict]) -> tuple[list[dict], dict]:
    """Dedupe train set.

    Rules:
      - Group by normalized query.
      - If a group has 1 intent label (and possibly repeated scopes/confidence):
          → keep first occurrence, drop rest. Harmless dedupe.
      - If a group has >1 intent labels (contradictory):
          → drop ALL occurrences of that query. Contradictory signal corrupts training.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for ex in train:
        groups[norm(ex["input"])].append(ex)

    kept: list[dict] = []
    stats = {
        "total_in": len(train),
        "unique_queries": len(groups),
        "dupes_same_label_dropped": 0,
        "conflicting_queries_dropped_entirely": 0,
        "conflicting_rows_dropped": 0,
        "conflicting_examples": [],
    }

    for q, rows in groups.items():
        intents = {r["output"]["intent"] for r in rows}
        if len(intents) > 1:
            stats["conflicting_queries_dropped_entirely"] += 1
            stats["conflicting_rows_dropped"] += len(rows)
            if len(stats["conflicting_examples"]) < 10:
                stats["conflicting_examples"].append(
                    {"query": rows[0]["input"], "intents": sorted(intents)}
                )
            continue
        kept.append(rows[0])
        stats["dupes_same_label_dropped"] += len(rows) - 1

    stats["total_out"] = len(kept)
    return kept, stats


def compute_dist(samples: list[dict]) -> dict:
    intents = Counter(x["output"]["intent"] for x in samples)
    scopes = Counter(x["output"]["scope"] for x in samples)
    return {
        "intent": dict(sorted(intents.items(), key=lambda kv: -kv[1])),
        "scope": dict(sorted(scopes.items(), key=lambda kv: -kv[1])),
    }


def check_leakage(train_queries: set[str], val: list[dict]) -> dict:
    overlap = [v for v in val if norm(v["input"]) in train_queries]
    return {
        "val_total": len(val),
        "overlap_count": len(overlap),
        "overlap_pct": round(len(overlap) / len(val) * 100, 2),
        "sample_overlaps": [v["input"] for v in overlap[:5]],
    }


def main() -> None:
    train = load_jsonl(TRAIN_PATH)
    val_orig = load_jsonl(VAL_ORIG_PATH)
    val_clean = load_jsonl(VAL_CLEAN_PATH)

    # Dedupe train
    train_clean, dedupe_stats = dedupe_train(train)
    train_clean_queries = {norm(ex["input"]) for ex in train_clean}

    # Leakage check vs val
    leak_orig = check_leakage(train_clean_queries, val_orig)
    leak_clean = check_leakage(train_clean_queries, val_clean)

    # Distributions
    dist_train_in = compute_dist(train)
    dist_train_out = compute_dist(train_clean)
    dist_val_clean = compute_dist(val_clean)

    # Write clean train
    with open(TRAIN_CLEAN_PATH, "w", encoding="utf-8") as f:
        for ex in train_clean:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Report
    report = {
        "train_dedupe": dedupe_stats,
        "leakage_check": {
            "train_clean_vs_val_original": leak_orig,
            "train_clean_vs_val_clean": leak_clean,
        },
        "distribution": {
            "train_input": dist_train_in,
            "train_output": dist_train_out,
            "val_clean": dist_val_clean,
        },
        "output_files": {
            "train_clean": str(TRAIN_CLEAN_PATH.relative_to(ROOT)),
            "val_clean": str(VAL_CLEAN_PATH.relative_to(ROOT)),
        },
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print summary
    print("=" * 60)
    print("ROUTER DATA AUDIT — SUMMARY")
    print("=" * 60)
    print(f"Train IN:  {dedupe_stats['total_in']} samples, {dedupe_stats['unique_queries']} unique queries")
    print(f"  - dupes (same label) dropped: {dedupe_stats['dupes_same_label_dropped']}")
    print(f"  - conflicting queries dropped ENTIRELY: {dedupe_stats['conflicting_queries_dropped_entirely']}")
    print(f"  - conflicting rows dropped: {dedupe_stats['conflicting_rows_dropped']}")
    print(f"Train OUT: {dedupe_stats['total_out']} samples")
    print()
    print(f"Val clean: {len(val_clean)} samples, {leak_clean['overlap_count']} overlap ({leak_clean['overlap_pct']}%)")
    print()
    print("Conflicting query samples (dropped):")
    for ex in dedupe_stats["conflicting_examples"][:5]:
        print(f"  '{ex['query'][:60]}' → {ex['intents']}")
    print()
    print(f"Outputs written:")
    print(f"  {TRAIN_CLEAN_PATH}")
    print(f"  {REPORT_PATH}")


if __name__ == "__main__":
    main()
