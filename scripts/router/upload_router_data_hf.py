"""
scripts/router/upload_router_data_hf.py
One-off uploader: push clean router train/val + training system prompt to HuggingFace.

Exp6 notebooks on Colab pull data from this repo via hf_hub_download (no Google Drive).

Usage:
    export HF_TOKEN=hf_...
    python scripts/router/upload_router_data_hf.py
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
HF_REPO_ID = "daredevil467/hanoi-weather-router-data"

FILES_TO_UPLOAD = {
    # v5 data (size ablation baseline)
    "multitask_train_v5_clean.jsonl": ROOT / "data/router/multitask_train_v5_clean.jsonl",
    "multitask_val_v5_clean.jsonl":   ROOT / "data/router/multitask_val_v5_clean.jsonl",
    # v6 data (targeted augmentation for weak intents)
    "multitask_train_v6_clean.jsonl": ROOT / "data/router/multitask_train_v6_clean.jsonl",
    "multitask_val_v6_clean.jsonl":   ROOT / "data/router/multitask_val_v6_clean.jsonl",
    # audit reports
    "audit_report.json":               ROOT / "data/router/audit_report.json",
    "expert_audit_report.json":        ROOT / "data/router/expert_audit_report.json",
}

README_MD = """---
language:
- vi
license: cc-by-4.0
task_categories:
- text-classification
tags:
- intent-classification
- weather
- vietnamese
- hanoi
- routing
- slm
- qwen3
---

# Hanoi Weather Router — Training Data (v5 + v6)

Multi-task intent + scope classification data for Hanoi weather chatbot router SLM.
Used to fine-tune Qwen2.5/Qwen3 base models (0.5B → 8B) in router ablation studies.

## Files

| File | Rows | Purpose |
|------|------|---------|
| `multitask_train_v5_clean.jsonl` | 3366 | v5 training — size ablation baseline |
| `multitask_val_v5_clean.jsonl`   | 373  | v5 validation — zero leakage |
| `multitask_train_v6_clean.jsonl` | 3471 | v6 training — targeted augmentation (+105 samples) |
| `multitask_val_v6_clean.jsonl`   | 385  | v6 validation — zero leakage (+12 samples) |
| `audit_report.json`              | 1    | v3 cleaning provenance + distribution stats |
| `expert_audit_report.json`       | 1    | v5 expert audit: final stats + actions taken |

## v6 Changes (Data Quality Ablation)

v6 adds 117 manually crafted samples targeting 3 weak intents identified in Exp6:
- `hourly_forecast`: +40 train, +5 val (added clear hourly signal words)
- `smalltalk_weather`: +15 train, +2 val (genuine smalltalk, no activity overlap)
- `activity_weather`: +15 hard negatives (clear named activities)
- `daily_forecast` + `rain_query`: +20 train, +3 val (disambiguation samples)
- Multi-turn context: +15 train (follow-up turns)

## Preprocessing

**v3 cleaning** (from `multitask_train_v3.jsonl` 4005 + `multitask_val_v3.jsonl` 672):

1. **Val leakage fix**: removed 238/672 (35.4%) val samples that appeared in train → 434 clean
2. **Train dedupe**: removed 645 exact-duplicate rows (same query + same label)
3. **Train conflict removal**: dropped 17 queries (58 rows) with contradictory intent labels

**v4 expert audit** (from v3-clean train 3302 + val 434):

4. **Train mislabel fix**: relabeled 5 "mùa này" samples from smalltalk_weather → seasonal_context
5. **Val normalized+fuzzy leak removal**: removed 94 val samples (72 exact-after-normalization + 22 fuzzy jaccard≥0.85) → 340
6. **Val augmentation**: added 25 hand-crafted samples for 7 weak intents → all 15 intents ≥20 val samples → 365 final
7. **Final leak verification**: 0 leaks (exact + fuzzy) between train and val

**v5 final cleaning** (from v4-clean train 3302 + val 365):

8. **Umbrella relabel**: 24 umbrella queries: smalltalk_weather → rain_query (per disambiguation rules)
9. **Clothing relabel**: 3 generic clothing queries: activity_weather → smalltalk_weather (per design doc)
10. **Alert relabel**: 3 samples with alert keywords (bão/giông/rét đậm): hourly_forecast/temperature_query → weather_alert
11. **Smalltalk augmentation**: +55 genuine samples (10 greetings, 10 farewell, 10 identity, 10 OOS cities, 10 off-topic, 5 casual)
12. **Val augmentation**: +5 smalltalk val samples + 3 multi-turn context smalltalk → 373 final
13. **Multi-turn smalltalk**: +10 train + 3 val context samples (greeting/farewell/OOS mid-conversation)
14. **Final verification**: 0 duplicates, 0 leaks (exact + fuzzy), all 15 intents ≥20 val, max/min ratio 2.0x

## Schema

Each line is a JSON object:
```json
{
  "input": "Gió ở Hòa Phú hiện tại mạnh không?",
  "context": null,
  "output": {
    "intent": "wind_query",
    "scope": "ward",
    "confidence": 0.9
  }
}
```

**Optional output field**: `rewritten_query` — present when `context` is given and the
query needs rewriting for standalone retrieval.

## Intents (15 classes)

current_weather, hourly_forecast, daily_forecast, weather_overview, rain_query,
temperature_query, wind_query, humidity_fog_query, historical_weather,
location_comparison, activity_weather, expert_weather_param, weather_alert,
seasonal_context, smalltalk_weather

## Scopes (3 classes)

city, district, ward

## Training Prompt

See `system_prompt.txt` — the exact prompt used during SFT. Non-diacritic ASCII Vietnamese
(consistency with v4 training recipe). Production inference uses diacritic version
(the model generalizes across both forms).
"""

# SYSTEM_PROMPT — diacritic Vietnamese, identical to:
#   app/agent/router/config.py::ROUTER_SYSTEM_PROMPT
#   model/router/Modelfile.v4::SYSTEM
# Must stay in sync. If you change one, change all three.
SYSTEM_PROMPT = (
    "Phân loại intent và scope cho câu hỏi thời tiết Hà Nội. Trả về JSON.\n"
    "\n"
    "## Intents:\n"
    "- current_weather: thời tiết NGAY LÚC NÀY (nhiệt độ, trời nắng/mưa, chung chung)\n"
    "- hourly_forecast: diễn biến CHI TIẾT THEO GIỜ trong ngày (không chỉ hỏi mưa)\n"
    "- daily_forecast: dự báo NHIỀU NGÀY tới (3 ngày, tuần tới, cuối tuần)\n"
    "- weather_overview: TỔNG QUAN, tóm tắt thời tiết hôm nay/ngày mai (không hỏi thông số cụ thể)\n"
    "- rain_query: hỏi CÓ MƯA KHÔNG, xác suất mưa, mưa bao lâu/lúc nào tạnh\n"
    "- temperature_query: hỏi CỤ THỂ VỀ NHIỆT ĐỘ (bao nhiêu độ, nóng/lạnh)\n"
    "- wind_query: hỏi CỤ THỂ VỀ GIÓ (gió mạnh không, hướng gió, tốc độ gió)\n"
    "- humidity_fog_query: hỏi về ĐỘ ẨM, SƯƠNG MÙ, sương muối\n"
    "- historical_weather: thời tiết NGÀY/TUẦN TRƯỚC, dữ liệu QUÁ KHỨ\n"
    "- location_comparison: SO SÁNH thời tiết giữa các quận/phường/địa điểm\n"
    "- activity_weather: thời tiết PHÙ HỢP ĐỂ LÀM hoạt động X không (chạy bộ, picnic)\n"
    "- expert_weather_param: thông số KỸ THUẬT chuyên sâu (áp suất, UV, điểm sương, tầm nhìn)\n"
    "- weather_alert: CẢNH BÁO nguy hiểm: bão/áp thấp, ngập, giông/lốc, rét hại, nắng nóng cực đoan\n"
    "- seasonal_context: SO SÁNH với hôm qua/tuần trước, xu hướng, bất thường theo MÙA\n"
    "- smalltalk_weather: chào hỏi, ngoài phạm vi, câu hỏi không liên quan thời tiết\n"
    "\n"
    "## Anti-confusion rules:\n"
    "- bây giờ/bay gio = thời điểm hiện tại -> current_weather (KHÔNG phải wind_query)\n"
    "- gió/gió mạnh/tốc độ gió = yếu tố gió -> wind_query\n"
    "- bão/lũ/cảnh báo -> weather_alert (KHÔNG phải rain_query)\n"
    "\n"
    "## Scopes:\n"
    "- city: toàn Hà Nội hoặc không nói rõ địa điểm\n"
    "- district: quận/huyện hoặc địa điểm nổi tiếng (Hồ Gươm->Hoàn Kiếm, Lăng Bác->Ba Đình, Nội Bài->Sóc Sơn...)\n"
    "- ward: phường/xã\n"
    "\n"
    "## Output:\n"
    '{"intent":"...","scope":"...","confidence":0.9}\n'
    "Thêm rewritten_query nếu có context và câu thiếu địa điểm."
)


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN env var not set.", file=sys.stderr)
        print("  export HF_TOKEN=hf_...", file=sys.stderr)
        sys.exit(1)

    for name, path in FILES_TO_UPLOAD.items():
        if not path.exists():
            print(f"ERROR: Missing file {path}", file=sys.stderr)
            sys.exit(1)

    api = HfApi(token=token)

    print(f"Creating/ensuring repo: {HF_REPO_ID} (dataset)")
    api.create_repo(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )

    # Upload README
    print("Uploading README.md ...")
    api.upload_file(
        path_or_fileobj=README_MD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )

    # Upload system prompt
    print("Uploading system_prompt.txt ...")
    api.upload_file(
        path_or_fileobj=SYSTEM_PROMPT.encode("utf-8"),
        path_in_repo="system_prompt.txt",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )

    # Upload data files
    for name, path in FILES_TO_UPLOAD.items():
        print(f"Uploading {name} ({path.stat().st_size} bytes) ...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )

    print()
    print("=" * 60)
    print(f"Dataset live at: https://huggingface.co/datasets/{HF_REPO_ID}")
    print("=" * 60)
    print()
    print("Use in Colab notebook:")
    print(f'  from huggingface_hub import hf_hub_download')
    print(f'  train_file = hf_hub_download(repo_id="{HF_REPO_ID}", filename="multitask_train_v5_clean.jsonl", repo_type="dataset")')


if __name__ == "__main__":
    main()
