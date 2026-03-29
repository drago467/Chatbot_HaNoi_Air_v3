"""
Phase 2: Consolidate training data from multiple sources into a single,
balanced, deduplicated multitask dataset for SLM Router v2.

Sources:
  1. data/router/multitask_train.jsonl  (1,983 records — multitask format)
  2. data/router/raw/augmented_v2.jsonl (2,330 records — chat format)
  3. data/router/train_clean.jsonl      (2,022 records — chat format)

Output:
  data/router/multitask_train_v2.jsonl  (≥4,500 balanced, deduplicated)
  data/router/multitask_val_v2.jsonl    (≥700 balanced)

Usage:
  python scripts/router/consolidate_dataset.py
  python scripts/router/consolidate_dataset.py --dry-run   # stats only
"""

import copy
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "router"

# ── Canonical label sets (from app/agent/router/config.py) ──
VALID_INTENTS = [
    "current_weather", "hourly_forecast", "daily_forecast", "weather_overview",
    "rain_query", "temperature_query", "wind_query", "humidity_fog_query",
    "historical_weather", "location_comparison", "activity_weather",
    "expert_weather_param", "weather_alert", "seasonal_context", "smalltalk_weather",
]
VALID_SCOPES = ["city", "district", "ward"]


# ── System prompt (v2 — full 15 intents, from app/agent/router/config.py) ──
SYSTEM_PROMPT_V2 = """Phân loại intent và scope cho câu hỏi thời tiết Hà Nội. Trả về JSON.

## Intents:
- current_weather: thời tiết NGAY LÚC NÀY (nhiệt độ, trời nắng/mưa, chung chung)
- hourly_forecast: diễn biến CHI TIẾT THEO GIỜ trong ngày (không chỉ hỏi mưa)
- daily_forecast: dự báo NHIỀU NGÀY tới (3 ngày, tuần tới, cuối tuần)
- weather_overview: TỔNG QUAN, tóm tắt thời tiết hôm nay/ngày mai (không hỏi thông số cụ thể)
- rain_query: hỏi CÓ MƯA KHÔNG, xác suất mưa, mưa bao lâu/lúc nào tạnh — KHÔNG phải cảnh báo nguy hiểm
- temperature_query: hỏi CỤ THỂ VỀ NHIỆT ĐỘ (bao nhiêu độ, nóng/lạnh thế nào) — KỂ CẢ nhiệt độ ngày mai/cuối tuần
- wind_query: hỏi CỤ THỂ VỀ GIÓ (gió mạnh không, hướng gió, tốc độ gió)
- humidity_fog_query: hỏi về ĐỘ ẨM, SƯƠNG MÙ, sương muối
- historical_weather: thời tiết NGÀY/TUẦN TRƯỚC, dữ liệu QUÁ KHỨ
- location_comparison: SO SÁNH thời tiết giữa các quận/phường/địa điểm
- activity_weather: thời tiết có PHÙ HỢP ĐỂ LÀM hoạt động X không (ra ngoài, thoải mái/dễ chịu không, chạy bộ, picnic)
- expert_weather_param: thông số KỸ THUẬT chuyên sâu (áp suất, tia UV, điểm sương, tầm nhìn)
- weather_alert: CẢNH BÁO nguy hiểm: bão/áp thấp, ngập lụt, GIÔNG/LỐC mạnh, mưa DÔNG, rét hại, nắng nóng cực đoan, thay đổi thời tiết đột ngột
- seasonal_context: SO SÁNH với hôm qua/tuần trước, xu hướng, bất thường theo MÙA
- smalltalk_weather: chào hỏi, ngoài phạm vi (không phải Hà Nội), câu hỏi không liên quan thời tiết

## Scopes:
- city: toàn Hà Nội, hoặc KHÔNG NÓI RÕ địa điểm
- district: nhắc tên QUẬN/HUYỆN (Ba Đình, Cầu Giấy...) hoặc ĐỊA ĐIỂM NỔI TIẾNG thuộc quận (Hồ Gươm→Hoàn Kiếm, Lăng Bác→Ba Đình, Nội Bài→Sóc Sơn...)
- ward: nhắc tên PHƯỜNG/XÃ (Phường Dịch Vọng Hậu, Xã Tiên Dương...)

## Multi-task output (khi có context từ lượt trước):
Nếu được cung cấp context (location/intent từ lượt trước) VÀ câu hỏi hiện tại thiếu địa điểm hoặc dùng đại từ (ở đó, thế còn, còn ..., vậy ...), hãy điền thêm trường "rewritten_query" với câu hỏi đầy đủ ngữ cảnh.

## Output format:
Khi không có context hoặc câu hỏi đầy đủ:
{"intent": "...", "scope": "...", "confidence": 0.92}

Khi có context và cần rewrite:
{"intent": "...", "scope": "...", "confidence": 0.91, "rewritten_query": "Dự báo thời tiết ngày mai ở quận Cầu Giấy như thế nào?"}"""


# ═══════════════════════════════════════════════════════════════════════════
# Loaders — each source has a different format, normalize to multitask
# ═══════════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL, skip blank lines."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_multitask(path: Path) -> list[dict]:
    """
    Load multitask_train.jsonl (native format).
    Schema: {"input", "context", "output": {"intent", "scope", "confidence", "rewritten_query?"}}
    """
    records = load_jsonl(path)
    valid = []
    for r in records:
        out = r.get("output", {})
        if isinstance(out, str):
            out = json.loads(out)
        intent = out.get("intent", "")
        scope = out.get("scope", "")
        if intent in VALID_INTENTS and scope in VALID_SCOPES:
            r["output"] = out
            r["_source"] = "multitask_train"
            valid.append(r)
    return valid


def load_chat_format(path: Path, source_tag: str) -> list[dict]:
    """
    Load chat-format JSONL (augmented_v2.jsonl, train_clean.jsonl).
    Schema: {"messages": [{"role": ..., "content": ...}], "metadata"?: {...}}

    Converts to multitask format:
      {"input", "context": null, "output": {"intent", "scope", "confidence"}}
    """
    records = load_jsonl(path)
    converted = []
    for r in records:
        msgs = r.get("messages", [])
        user_msg = ""
        asst_msg = ""
        for m in msgs:
            if m["role"] == "user":
                user_msg = m["content"]
            elif m["role"] == "assistant":
                asst_msg = m["content"]

        if not user_msg or not asst_msg:
            continue

        try:
            out = json.loads(asst_msg)
        except json.JSONDecodeError:
            continue

        intent = out.get("intent", "")
        scope = out.get("scope", "")
        if intent not in VALID_INTENTS or scope not in VALID_SCOPES:
            continue

        record = {
            "input": user_msg,
            "context": None,
            "output": {
                "intent": intent,
                "scope": scope,
                "confidence": float(out.get("confidence", 0.90)),
            },
            "_source": source_tag,
        }
        # Preserve metadata source type if available
        meta = r.get("metadata", {})
        if meta.get("source"):
            record["_data_source"] = meta["source"]

        converted.append(record)

    return converted


# ═══════════════════════════════════════════════════════════════════════════
# Deduplication
# ═══════════════════════════════════════════════════════════════════════════

def dedup_exact(records: list[dict]) -> list[dict]:
    """
    Exact dedup by (normalized_input, intent, scope).
    Keep first occurrence (priority: multitask_train > augmented_v2 > train_clean).
    """
    seen = set()
    unique = []
    for r in records:
        key_text = r["input"].strip().lower()
        intent = r["output"]["intent"]
        scope = r["output"]["scope"]
        key = hashlib.md5(f"{key_text}|{intent}|{scope}".encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


# ═══════════════════════════════════════════════════════════════════════════
# Telex augmentation (no-diacritic variants)
# ═══════════════════════════════════════════════════════════════════════════

def _strip_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics → ASCII-like Telex."""
    import unicodedata
    nfd = unicodedata.normalize("NFD", text)
    ascii_text = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return ascii_text.replace("đ", "d").replace("Đ", "D").lower()


def augment_telex_variants(
    records: list[dict],
    ratio: float = 0.15,
    seed: int = 42,
) -> list[dict]:
    """
    For a random subset of records, create a Telex (no-diacritic) variant.
    This teaches the model to handle non-diacritic input.

    Args:
        records: list of multitask records
        ratio: fraction of records to augment (default 15%)
        seed: random seed for reproducibility

    Returns:
        List of new Telex-variant records to append
    """
    rng = random.Random(seed)
    # Only augment records that have diacritics
    candidates = [
        r for r in records
        if _strip_diacritics(r["input"]) != r["input"].lower()
    ]
    n = int(len(candidates) * ratio)
    selected = rng.sample(candidates, min(n, len(candidates)))

    variants = []
    for r in selected:
        variant = copy.deepcopy(r)
        variant["input"] = _strip_diacritics(r["input"])
        variant["_source"] = "telex_augment"
        variants.append(variant)

    return variants


# ═══════════════════════════════════════════════════════════════════════════
# Balancing
# ═══════════════════════════════════════════════════════════════════════════

def balance_intents(
    records: list[dict],
    target_per_class: int = 350,
    seed: int = 42,
) -> list[dict]:
    """
    Balance intent distribution via undersampling majority + oversampling minority.
    Target: no intent < target_per_class * 0.6 or > target_per_class * 1.4

    Args:
        records: list of multitask records
        target_per_class: target count per intent
        seed: random seed

    Returns:
        Balanced record list
    """
    rng = random.Random(seed)
    by_intent = defaultdict(list)
    for r in records:
        by_intent[r["output"]["intent"]].append(r)

    balanced = []
    for intent in VALID_INTENTS:
        items = by_intent.get(intent, [])
        if not items:
            print(f"  ⚠️  No samples for intent '{intent}'")
            continue

        if len(items) >= target_per_class:
            # Undersample: take target_per_class random samples
            balanced.extend(rng.sample(items, target_per_class))
        else:
            # Oversample: take all + sample extra from existing
            balanced.extend(items)
            extra_needed = target_per_class - len(items)
            extra = rng.choices(items, k=extra_needed)
            balanced.extend(extra)

    rng.shuffle(balanced)
    return balanced


# ═══════════════════════════════════════════════════════════════════════════
# Train / Val split
# ═══════════════════════════════════════════════════════════════════════════

def stratified_split(
    records: list[dict],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Stratified split by (intent, scope).
    Ensures each combo has at least 1 sample in val.
    """
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in records:
        key = (r["output"]["intent"], r["output"]["scope"])
        groups[key].append(r)

    train, val = [], []
    for key, items in sorted(groups.items()):
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ═══════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════

def save_jsonl(records: list[dict], path: Path):
    """Save as JSONL, stripping internal _source / _data_source tags."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            clean = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
    print(f"  ✅ Saved {len(records):,} records → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def print_distribution(records: list[dict], label: str):
    """Print intent and scope distributions."""
    intents = Counter(r["output"]["intent"] for r in records)
    scopes = Counter(r["output"]["scope"] for r in records)
    ctx_count = sum(1 for r in records if r.get("context"))
    rw_count = sum(1 for r in records if r.get("output", {}).get("rewritten_query"))

    print(f"\n{'='*60}")
    print(f"  {label} — {len(records):,} total")
    print(f"  Context records: {ctx_count} ({ctx_count/len(records)*100:.1f}%)")
    print(f"  Rewrite records: {rw_count} ({rw_count/len(records)*100:.1f}%)")
    print(f"{'='*60}")

    print(f"\n  {'Intent':<25} {'Count':>6} {'%':>7}")
    print(f"  {'-'*40}")
    for intent, count in sorted(intents.items(), key=lambda x: -x[1]):
        print(f"  {intent:<25} {count:>6} {count/len(records)*100:>6.1f}%")

    print(f"\n  {'Scope':<25} {'Count':>6} {'%':>7}")
    print(f"  {'-'*40}")
    for scope, count in sorted(scopes.items(), key=lambda x: -x[1]):
        print(f"  {scope:<25} {count:>6} {count/len(records)*100:>6.1f}%")

    if records:
        sources = Counter(r.get("_source", "unknown") for r in records)
        print(f"\n  {'Source':<25} {'Count':>6}")
        print(f"  {'-'*35}")
        for src, count in sorted(sources.items(), key=lambda x: -x[1]):
            print(f"  {src:<25} {count:>6}")


def main():
    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print("  SLM Router v2 — Dataset Consolidation Pipeline")
    print("=" * 60)

    # ── Step 1: Load all sources ──
    print("\n📥 Step 1: Loading data sources...")

    multitask_path = DATA_DIR / "multitask_train.jsonl"
    augmented_path = DATA_DIR / "raw" / "augmented_v2.jsonl"
    train_clean_path = DATA_DIR / "train_clean.jsonl"

    records_mt = load_multitask(multitask_path) if multitask_path.exists() else []
    print(f"  multitask_train.jsonl  : {len(records_mt):,} records")

    records_aug = load_chat_format(augmented_path, "augmented_v2") if augmented_path.exists() else []
    print(f"  augmented_v2.jsonl     : {len(records_aug):,} records")

    records_clean = load_chat_format(train_clean_path, "train_clean") if train_clean_path.exists() else []
    print(f"  train_clean.jsonl      : {len(records_clean):,} records")

    # Priority order: multitask (has context/rewrite) > augmented > train_clean
    all_records = records_mt + records_aug + records_clean
    print(f"\n  Total raw: {len(all_records):,}")

    # ── Step 2: Dedup ──
    print("\n🔍 Step 2: Deduplication...")
    deduped = dedup_exact(all_records)
    removed = len(all_records) - len(deduped)
    print(f"  Removed {removed:,} exact duplicates → {len(deduped):,} unique")

    # ── Step 3: Telex augmentation ──
    print("\n🔤 Step 3: Telex augmentation (15% no-diacritic variants)...")
    telex_variants = augment_telex_variants(deduped, ratio=0.15)
    print(f"  Generated {len(telex_variants):,} Telex variants")
    deduped.extend(telex_variants)
    print(f"  Total after augmentation: {len(deduped):,}")

    # ── Step 4: Balance intents ──
    print("\n⚖️  Step 4: Balancing intent distribution...")
    # Calculate target: aim for total ~5,000
    n_intents = len(VALID_INTENTS)
    target_per_class = max(300, len(deduped) // n_intents)
    print(f"  Target per intent: {target_per_class}")
    balanced = balance_intents(deduped, target_per_class=target_per_class)
    print(f"  Total after balancing: {len(balanced):,}")

    # ── Step 5: Stratified split ──
    print("\n✂️  Step 5: Stratified train/val split (85/15)...")
    train_set, val_set = stratified_split(balanced, val_ratio=0.15)
    print(f"  Train: {len(train_set):,}  |  Val: {len(val_set):,}")

    # ── Step 6: Print stats ──
    print_distribution(train_set, "TRAIN SET")
    print_distribution(val_set, "VAL SET")

    # ── Step 7: Save ──
    if dry_run:
        print("\n⏭️  Dry run — skipping save.")
    else:
        print("\n💾 Step 7: Saving...")
        save_jsonl(train_set, DATA_DIR / "multitask_train_v2.jsonl")
        save_jsonl(val_set, DATA_DIR / "multitask_val_v2.jsonl")

        # Also save existing val as test (keep original test_clean for reference)
        val_orig = DATA_DIR / "multitask_val.jsonl"
        if val_orig.exists():
            test_records = load_multitask(val_orig)
            save_jsonl(test_records, DATA_DIR / "multitask_test_v2.jsonl")
            print(f"  ✅ Copied original val ({len(test_records):,}) → multitask_test_v2.jsonl (test set)")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
