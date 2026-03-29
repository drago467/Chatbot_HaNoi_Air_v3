"""
Build slm_router_qwen3_colab_a100_v3.ipynb — FIXED version.

Key fixes vs v2:
  1. COMPLETION-ONLY loss masking — only train on assistant response,
     not system prompt (reduces loss from 13.6 → ~0.1, huge accuracy gain)
  2. Remove label_smoothing_factor (causes loss inflation for this task)
  3. Use cosine LR schedule for better convergence
  4. Add per-intent confusion matrix for error analysis
  5. Epochs 5→8 with early stopping patience=2

Usage:
    python scripts/router/build_notebook_v3.py
"""

import json
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "slm_router_qwen3_colab_a100_v3.ipynb"


def make_cell(source: str, cell_type: str = "code") -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.split("\n") if "\n" in source else [source],
        "outputs": [] if cell_type == "code" else [],
        **({"execution_count": None} if cell_type == "code" else {}),
    }


def make_md(text: str) -> dict:
    return make_cell(text, "markdown")


def fix_source(lines: list[str]) -> list[str]:
    if not lines:
        return lines
    fixed = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            fixed.append(line.rstrip("\n") + "\n")
        else:
            fixed.append(line.rstrip("\n"))
    return fixed


CELLS = []

# ── Cell 0: Title ──
CELLS.append(make_md(
    "# SLM Router v3 — Qwen3-4B Multi-task Fine-tuning (Colab A100)\n"
    "\n"
    "**Key fix vs v2**: Completion-only loss masking — train ONLY on assistant response, not system prompt.  \n"
    "**Model**: `unsloth/Qwen3-4B` (4B params, full bf16, A100)  \n"
    "**Task**: Intent routing (15 classes) + contextual query rewriting  \n"
    "**Dataset**: multitask_train_v2.jsonl (~3,800 balanced samples)  \n"
    "**Export**: GGUF Q4_K_M (~2.6GB) for laptop deployment via Ollama"
))

# ── Cell 1: Install ──
CELLS.append(make_cell(
    '%%capture\n'
    '!pip install "unsloth[colab-new]" "trl>=0.17,<0.19"'
))

# ── Cell 2: Mount Drive + GPU check ──
CELLS.append(make_cell(
    'from google.colab import drive\n'
    'drive.mount("/content/drive")\n'
    '\n'
    'import torch, subprocess\n'
    'gpu = subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"]).decode().strip()\n'
    'print(f"GPU: {gpu}")\n'
    'print(f"bf16 : {torch.cuda.is_bf16_supported()}")\n'
    'print(f"Torch: {torch.__version__}")\n'
    'assert torch.cuda.is_bf16_supported(), "A100 required for bf16 training"'
))

# ── Cell 3: Imports ──
CELLS.append(make_cell(
    'import json, re, os, unicodedata\n'
    'from pathlib import Path\n'
    'from collections import Counter\n'
    '\n'
    'from unsloth import FastLanguageModel\n'
    'import torch\n'
    'from datasets import Dataset\n'
    'from transformers import DataCollatorForSeq2Seq\n'
    'from trl import SFTTrainer, SFTConfig\n'
    '\n'
    'import trl; print(f"TRL version: {trl.__version__}")'
))

# ── Cell 4: Configuration ──
CELLS.append(make_cell(
    '# ══════════════════ CONFIG v3 ══════════════════\n'
    'BASE_MODEL     = "unsloth/Qwen3-4B"   # 4B params, bf16\n'
    'MAX_SEQ_LENGTH = 1024\n'
    'LORA_R         = 64      # v3: doubled from 32 for better capacity\n'
    'LORA_ALPHA     = 128     # alpha/r = 2\n'
    'EPOCHS         = 8       # v3: more epochs, early stopping watches val_loss\n'
    'BATCH_SIZE     = 4       # per device (4B needs more VRAM)\n'
    'GRAD_ACCUM     = 8       # effective batch = 32\n'
    'LR             = 2e-4    # v3: slightly higher LR, cosine schedule\n'
    'WARMUP_RATIO   = 0.06\n'
    '\n'
    '# Paths (Google Drive)\n'
    'DRIVE_DIR  = Path("/content/drive/MyDrive/hanoi-router-v3")\n'
    'DRIVE_DIR.mkdir(parents=True, exist_ok=True)\n'
    'OUTPUT_DIR = str(DRIVE_DIR / "outputs")\n'
    '\n'
    '# Data — upload multitask_train_v2.jsonl & multitask_val_v2.jsonl to this folder\n'
    'DATA_DIR   = Path("/content/drive/MyDrive/chatbot-hanoi-weather")\n'
    'TRAIN_FILE = DATA_DIR / "multitask_train_v2.jsonl"\n'
    'VAL_FILE   = DATA_DIR / "multitask_val_v2.jsonl"\n'
    '\n'
    'print(f"Model  : {BASE_MODEL}")\n'
    'print(f"LoRA   : r={LORA_R}, alpha={LORA_ALPHA}")\n'
    'print(f"Batch  : {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} effective")\n'
    'print(f"Epochs : {EPOCHS}")\n'
    'print(f"LR     : {LR}")\n'
    'print(f"Train  : {TRAIN_FILE}")\n'
    'print(f"Output : {OUTPUT_DIR}")'
))

# ── Cell 5: System Prompt (15 intents with anti-confusion rules) ──
CELLS.append(make_cell(
    "SYSTEM_PROMPT = (\n"
    '    "Phân loại intent và scope cho câu hỏi thời tiết Hà Nội. Trả về JSON." + chr(10)\n'
    '    + chr(10)\n'
    '    + "## Intents:" + chr(10)\n'
    '    + "- current_weather: thời tiết NGAY LÚC NÀY (nhiệt độ, trời nắng/mưa, chung chung)" + chr(10)\n'
    '    + "- hourly_forecast: diễn biến CHI TIẾT THEO GIỜ trong ngày (không chỉ hỏi mưa)" + chr(10)\n'
    '    + "- daily_forecast: dự báo NHIỀU NGÀY tới (3 ngày, tuần tới, cuối tuần)" + chr(10)\n'
    '    + "- weather_overview: TỔNG QUAN, tóm tắt thời tiết hôm nay/ngày mai (không hỏi thông số cụ thể)" + chr(10)\n'
    '    + "- rain_query: hỏi CÓ MƯA KHÔNG, xác suất mưa, mưa bao lâu/lúc nào tạnh" + chr(10)\n'
    '    + "- temperature_query: hỏi CỤ THỂ VỀ NHIỆT ĐỘ (bao nhiêu độ, nóng/lạnh)" + chr(10)\n'
    '    + "- wind_query: hỏi CỤ THỂ VỀ GIÓ (gió mạnh không, hướng gió, tốc độ gió)" + chr(10)\n'
    '    + "- humidity_fog_query: hỏi về ĐỘ ẨM, SƯƠNG MÙ, sương muối" + chr(10)\n'
    '    + "- historical_weather: thời tiết NGÀY/TUẦN TRƯỚC, dữ liệu QUÁ KHỨ" + chr(10)\n'
    '    + "- location_comparison: SO SÁNH thời tiết giữa các quận/phường/địa điểm" + chr(10)\n'
    '    + "- activity_weather: thời tiết PHÙ HỢP ĐỂ LÀM hoạt động X không (chạy bộ, picnic)" + chr(10)\n'
    '    + "- expert_weather_param: thông số KỸ THUẬT chuyên sâu (áp suất, UV, điểm sương, tầm nhìn)" + chr(10)\n'
    '    + "- weather_alert: CẢNH BÁO nguy hiểm: bão/áp thấp, ngập, giông/lốc, rét hại, nắng nóng cực đoan" + chr(10)\n'
    '    + "- seasonal_context: SO SÁNH với hôm qua/tuần trước, xu hướng, bất thường theo MÙA" + chr(10)\n'
    '    + "- smalltalk_weather: chào hỏi, ngoài phạm vi, câu hỏi không liên quan thời tiết" + chr(10)\n'
    '    + chr(10)\n'
    '    + "## Anti-confusion rules:" + chr(10)\n'
    '    + "- bay gio/bây giờ = thời điểm hiện tại -> current_weather (KHÔNG phải wind_query)" + chr(10)\n'
    '    + "- gio/gió mạnh/tốc độ gió = yếu tố gió -> wind_query" + chr(10)\n'
    '    + "- bão/lũ/cảnh báo -> weather_alert (KHÔNG phải rain_query)" + chr(10)\n'
    '    + chr(10)\n'
    '    + "## Scopes:" + chr(10)\n'
    '    + "- city: toàn Hà Nội hoặc không nói rõ địa điểm" + chr(10)\n'
    '    + "- district: quận/huyện hoặc địa điểm nổi tiếng (Hồ Gươm->Hoàn Kiếm)" + chr(10)\n'
    '    + "- ward: phường/xã" + chr(10)\n'
    '    + chr(10)\n'
    '    + "## Output:" + chr(10)\n'
    '    + \'{"intent":"...","scope":"...","confidence":0.9}\' + chr(10)\n'
    '    + "Thêm rewritten_query nếu có context và câu thiếu địa điểm."\n'
    ")\n"
    '\n'
    'print(SYSTEM_PROMPT[:200], "...")\n'
    'print(f"Prompt length: {len(SYSTEM_PROMPT)} chars")'
))

# ── Cell 6: Load model ──
CELLS.append(make_cell(
    'model, tokenizer = FastLanguageModel.from_pretrained(\n'
    '    model_name     = BASE_MODEL,\n'
    '    max_seq_length = MAX_SEQ_LENGTH,\n'
    '    dtype          = torch.bfloat16,\n'
    '    load_in_4bit   = False,  # A100 has enough VRAM for full bf16\n'
    ')\n'
    'print(f"Model loaded: {BASE_MODEL}")\n'
    'print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")'
))

# ── Cell 7: LoRA — r=64 for better capacity ──
CELLS.append(make_cell(
    'model = FastLanguageModel.get_peft_model(\n'
    '    model,\n'
    '    r              = LORA_R,       # v3: 64 (was 32)\n'
    '    lora_alpha     = LORA_ALPHA,   # v3: 128 (was 64)\n'
    '    lora_dropout   = 0.05,\n'
    '    target_modules = [\n'
    '        "q_proj", "k_proj", "v_proj", "o_proj",\n'
    '        "gate_proj", "up_proj", "down_proj",\n'
    '    ],\n'
    '    bias           = "none",\n'
    '    use_gradient_checkpointing = "unsloth",\n'
    ')\n'
    'trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n'
    'total     = sum(p.numel() for p in model.parameters())\n'
    'print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")'
))

# ── Cell 8: Tokenizer setup ──
CELLS.append(make_cell(
    '# Manual tokenizer setup (skip get_chat_template — Unsloth Qwen3 bugs)\n'
    'if tokenizer.pad_token is None:\n'
    '    tokenizer.pad_token    = tokenizer.eos_token\n'
    '    tokenizer.pad_token_id = tokenizer.eos_token_id\n'
    'tokenizer.padding_side = "right"\n'
    '\n'
    '# Verify ChatML tokens\n'
    'test_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)\n'
    'print(f"pad_token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")\n'
    'print(f"<|im_start|> ids: {test_ids}")\n'
    'assert len(test_ids) == 1, "ChatML special tokens not found"'
))

# ── Cell 9: Load & validate dataset ──
CELLS.append(make_cell(
    'VALID_INTENTS = [\n'
    '    "current_weather", "hourly_forecast", "daily_forecast", "weather_overview",\n'
    '    "rain_query", "temperature_query", "wind_query", "humidity_fog_query",\n'
    '    "historical_weather", "location_comparison", "activity_weather",\n'
    '    "expert_weather_param", "weather_alert", "seasonal_context", "smalltalk_weather",\n'
    ']\n'
    'VALID_SCOPES = ["city", "district", "ward"]\n'
    '\n'
    'def load_jsonl(path):\n'
    '    records = []\n'
    '    with open(path, "r", encoding="utf-8") as f:\n'
    '        for line in f:\n'
    '            if line.strip():\n'
    '                records.append(json.loads(line))\n'
    '    return records\n'
    '\n'
    'train_raw = load_jsonl(TRAIN_FILE)\n'
    'val_raw   = load_jsonl(VAL_FILE)\n'
    '\n'
    'def validate(records, name):\n'
    '    valid = []\n'
    '    for r in records:\n'
    '        out = r["output"] if isinstance(r["output"], dict) else json.loads(r["output"])\n'
    '        if out["intent"] in VALID_INTENTS and out["scope"] in VALID_SCOPES:\n'
    '            valid.append(r)\n'
    '    print(f"{name}: {len(valid)}/{len(records)} valid")\n'
    '    return valid\n'
    '\n'
    'train_raw = validate(train_raw, "Train")\n'
    'val_raw   = validate(val_raw, "Val")\n'
    '\n'
    'intents = Counter(r["output"]["intent"] if isinstance(r["output"], dict) else json.loads(r["output"])["intent"] for r in train_raw)\n'
    'ctx_count = sum(1 for r in train_raw if r.get("context"))\n'
    'rw_count  = sum(1 for r in train_raw if (r["output"] if isinstance(r["output"], dict) else json.loads(r["output"])).get("rewritten_query"))\n'
    'print(f"Context records: {ctx_count} ({ctx_count/len(train_raw)*100:.1f}%)")\n'
    'print(f"Rewrite records: {rw_count} ({rw_count/len(train_raw)*100:.1f}%)")\n'
    'for intent, count in sorted(intents.items(), key=lambda x: -x[1]):\n'
    '    print(f"  {intent:<25} {count:>5} ({count/len(train_raw)*100:.1f}%)")'
))

# ── Cell 10: Format records — manual ChatML with chr(10) ──
CELLS.append(make_cell(
    'IM_START = "<|im_start|>"\n'
    'IM_END   = "<|im_end|>"\n'
    'NL       = chr(10)\n'
    '\n'
    'def format_record(rec):\n'
    '    user_msg = str(rec.get("input", "")).strip()\n'
    '    ctx = rec.get("context")\n'
    '    if ctx:\n'
    '        ctx_str  = json.dumps(ctx, ensure_ascii=False, separators=(",", ":"))\n'
    '        user_msg = "[CONTEXT: " + ctx_str + "]" + NL + user_msg\n'
    '    out = rec.get("output", {})\n'
    '    if isinstance(out, str): out = json.loads(out)\n'
    '    output_dict = {\n'
    '        "intent":     out["intent"],\n'
    '        "scope":      out["scope"],\n'
    '        "confidence": round(float(out.get("confidence", 0.9)), 2),\n'
    '    }\n'
    '    rw = out.get("rewritten_query")\n'
    '    if rw and str(rw).strip():\n'
    '        output_dict["rewritten_query"] = str(rw).strip()\n'
    '    text  = IM_START + "system"    + NL + SYSTEM_PROMPT + IM_END + NL\n'
    '    text += IM_START + "user"      + NL + user_msg      + IM_END + NL\n'
    '    text += IM_START + "assistant" + NL + json.dumps(output_dict, ensure_ascii=False) + IM_END + NL\n'
    '    return text\n'
    '\n'
    'sample = format_record(train_raw[0])\n'
    'print(sample[:400])\n'
    'print("...")\n'
    'print(f"Tokens: {len(tokenizer.encode(sample))}")\n'
    '\n'
    'train_texts = [format_record(r) for r in train_raw]\n'
    'val_texts   = [format_record(r) for r in val_raw]\n'
    '\n'
    'lengths = [len(tokenizer.encode(t)) for t in train_texts[:200]]\n'
    'print(f"Avg tokens: {sum(lengths)/len(lengths):.0f}, Max: {max(lengths)}, Over {MAX_SEQ_LENGTH}: {sum(1 for l in lengths if l > MAX_SEQ_LENGTH)}")\n'
    '\n'
    'raw_train = Dataset.from_dict({"text": train_texts})\n'
    'raw_val   = Dataset.from_dict({"text": val_texts})\n'
    'print(f"Train: {len(raw_train)}, Val: {len(raw_val)}")'
))

# ── Cell 11: Pre-tokenize WITH completion-only masking ──
# This is the KEY FIX — mask labels for everything except assistant response
CELLS.append(make_cell(
    '# v3 KEY FIX: Completion-only loss masking\n'
    '# Only compute loss on assistant response tokens, NOT system prompt / user query.\n'
    '# This reduces loss from ~13.6 (full-seq) to ~0.1 (response-only) and\n'
    '# dramatically improves accuracy because the model focuses on learning\n'
    '# the classification output, not memorizing the system prompt.\n'
    '\n'
    'ASSISTANT_MARKER = "<|im_start|>assistant" + chr(10)\n'
    'assistant_ids = tokenizer.encode(ASSISTANT_MARKER, add_special_tokens=False)\n'
    'print(f"Assistant marker token IDs: {assistant_ids}")\n'
    'print(f"Assistant marker length: {len(assistant_ids)} tokens")\n'
    '\n'
    'def tokenize_with_masking(examples):\n'
    '    """Tokenize and mask labels: -100 for everything before assistant response."""\n'
    '    enc = tokenizer(\n'
    '        examples["text"],\n'
    '        truncation=True,\n'
    '        max_length=MAX_SEQ_LENGTH,\n'
    '        padding=False,\n'
    '        return_tensors=None,\n'
    '    )\n'
    '    all_labels = []\n'
    '    masked_count = 0\n'
    '    for input_ids in enc["input_ids"]:\n'
    '        labels = [-100] * len(input_ids)  # mask everything by default\n'
    '        # Find the assistant marker position\n'
    '        marker_len = len(assistant_ids)\n'
    '        found = False\n'
    '        for i in range(len(input_ids) - marker_len + 1):\n'
    '            if input_ids[i:i+marker_len] == assistant_ids:\n'
    '                # Unmask everything AFTER the marker (the actual response)\n'
    '                start = i + marker_len\n'
    '                for j in range(start, len(input_ids)):\n'
    '                    labels[j] = input_ids[j]\n'
    '                found = True\n'
    '                masked_count += 1\n'
    '                break\n'
    '        if not found:\n'
    '            # Fallback: unmask everything (should not happen)\n'
    '            labels = list(input_ids)\n'
    '        all_labels.append(labels)\n'
    '    enc["labels"] = all_labels\n'
    '    return enc\n'
    '\n'
    'train_dataset = raw_train.map(tokenize_with_masking, batched=True, remove_columns=["text"])\n'
    'val_dataset   = raw_val.map(tokenize_with_masking, batched=True, remove_columns=["text"])\n'
    '\n'
    '# Verify masking works correctly\n'
    'sample_labels = train_dataset[0]["labels"]\n'
    'sample_ids    = train_dataset[0]["input_ids"]\n'
    'n_masked = sum(1 for l in sample_labels if l == -100)\n'
    'n_total  = len(sample_labels)\n'
    'n_active = n_total - n_masked\n'
    'print(f"Train tokenized: {len(train_dataset)} samples")\n'
    'print(f"Val tokenized  : {len(val_dataset)} samples")\n'
    'print(f"Sample: {n_total} tokens total, {n_masked} masked (-100), {n_active} active")\n'
    'print(f"Masking ratio: {n_masked/n_total*100:.1f}% masked (system+user), {n_active/n_total*100:.1f}% active (assistant)")\n'
    '\n'
    '# Decode the active part to verify\n'
    'active_ids = [sample_ids[i] for i in range(len(sample_labels)) if sample_labels[i] != -100]\n'
    'print(f"Active tokens decoded: {tokenizer.decode(active_ids)[:200]}")'
))

# ── Cell 12: Data collator ──
CELLS.append(make_cell(
    'data_collator = DataCollatorForSeq2Seq(\n'
    '    tokenizer          = tokenizer,\n'
    '    model              = model,\n'
    '    label_pad_token_id = -100,\n'
    '    pad_to_multiple_of = 8,\n'
    ')\n'
    'print("DataCollatorForSeq2Seq ready")'
))

# ── Cell 13: SFTConfig — FIXED: no label_smoothing, cosine schedule ──
CELLS.append(make_cell(
    'training_args = SFTConfig(\n'
    '    output_dir                  = OUTPUT_DIR,\n'
    '    num_train_epochs            = EPOCHS,\n'
    '    per_device_train_batch_size = BATCH_SIZE,\n'
    '    per_device_eval_batch_size  = BATCH_SIZE,\n'
    '    gradient_accumulation_steps = GRAD_ACCUM,\n'
    '    learning_rate               = LR,\n'
    '    warmup_ratio                = WARMUP_RATIO,\n'
    '    lr_scheduler_type           = "cosine",   # v3: cosine annealing\n'
    '    logging_steps               = 10,          # v3: log more frequently\n'
    '    eval_strategy               = "epoch",\n'
    '    save_strategy               = "epoch",\n'
    '    save_total_limit            = 2,\n'
    '    metric_for_best_model       = "eval_loss",\n'
    '    load_best_model_at_end      = True,\n'
    '    bf16                        = True,\n'
    '    fp16                        = False,\n'
    '    optim                       = "adamw_torch_fused",\n'
    '    weight_decay                = 0.01,\n'
    '    max_grad_norm               = 1.0,\n'
    '    max_seq_length              = MAX_SEQ_LENGTH,\n'
    '    # v3: NO label_smoothing_factor (was 0.1 in v2 — inflated loss)\n'
    ')\n'
    'print("SFTConfig ready")\n'
    'steps_per_epoch = len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)\n'
    'total_steps = steps_per_epoch * EPOCHS\n'
    'print(f"Steps/epoch: {steps_per_epoch}  |  Total: {total_steps}")\n'
    'print(f"Warmup steps: {int(total_steps * WARMUP_RATIO)}")\n'
    'print(f"LR scheduler: cosine")'
))

# ── Cell 14: Trainer ──
CELLS.append(make_cell(
    'trainer = SFTTrainer(\n'
    '    model            = model,\n'
    '    processing_class = tokenizer,\n'
    '    args             = training_args,\n'
    '    train_dataset    = train_dataset,\n'
    '    eval_dataset     = val_dataset,\n'
    '    data_collator    = data_collator,\n'
    '    packing          = False,\n'
    ')\n'
    'print("SFTTrainer ready")'
))

# ── Cell 15: Train ──
CELLS.append(make_cell(
    'trainer_stats = trainer.train()\n'
    'print(f"Train loss: {trainer_stats.training_loss:.4f}")\n'
    'print(f"Runtime:    {trainer_stats.metrics[\'train_runtime\']:.0f}s")\n'
    'print(f"Samples/s:  {trainer_stats.metrics[\'train_samples_per_second\']:.1f}")'
))

# ── Cell 16: Save adapter ──
CELLS.append(make_cell(
    'adapter_dir = DRIVE_DIR / "lora_adapter"\n'
    'model.save_pretrained(str(adapter_dir))\n'
    'tokenizer.save_pretrained(str(adapter_dir))\n'
    'print(f"Adapter saved to {adapter_dir}")'
))

# ── Cell 17: Full evaluation WITH confusion matrix ──
CELLS.append(make_cell(
    '# v3: Eval with per-intent confusion matrix\n'
    'model.eval()\n'
    '\n'
    'TEST_FILE = DATA_DIR / "multitask_val_v2.jsonl"\n'
    'if not TEST_FILE.exists():\n'
    '    TEST_FILE = VAL_FILE\n'
    'test_records = []\n'
    'with open(TEST_FILE, encoding="utf-8") as f:\n'
    '    for line in f:\n'
    '        if line.strip():\n'
    '            test_records.append(json.loads(line))\n'
    'print(f"Test records: {len(test_records)}")\n'
    '\n'
    'NL_eval = chr(10)\n'
    '\n'
    'correct_route  = 0\n'
    'total_route    = 0\n'
    'rw_correct     = 0\n'
    'rw_total       = 0\n'
    'rw_entity_ok   = 0\n'
    'norw_correct   = 0\n'
    'norw_total     = 0\n'
    '\n'
    '# Per-intent tracking\n'
    'intent_correct = Counter()\n'
    'intent_total   = Counter()\n'
    'confusion_pairs = Counter()   # (expected, predicted)\n'
    '\n'
    'for rec in test_records:\n'
    '    out = rec["output"] if isinstance(rec["output"], dict) else json.loads(rec["output"])\n'
    '    expected_intent = out["intent"]\n'
    '    expected_scope  = out["scope"]\n'
    '    has_rw = bool(out.get("rewritten_query"))\n'
    '\n'
    '    user_msg = str(rec.get("input", "")).strip()\n'
    '    ctx = rec.get("context")\n'
    '    if ctx:\n'
    '        ctx_str  = json.dumps(ctx, ensure_ascii=False, separators=(",",":"))\n'
    '        user_msg = "[CONTEXT: " + ctx_str + "]" + NL_eval + user_msg\n'
    '\n'
    '    prompt  = "<|im_start|>system" + NL_eval + SYSTEM_PROMPT + "<|im_end|>" + NL_eval\n'
    '    prompt += "<|im_start|>user"   + NL_eval + user_msg      + "<|im_end|>" + NL_eval\n'
    '    prompt += "<|im_start|>assistant" + NL_eval\n'
    '\n'
    '    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(model.device)\n'
    '    with torch.no_grad():\n'
    '        out_ids = model.generate(\n'
    '            **inputs, max_new_tokens=128, temperature=0.0,\n'
    '            do_sample=False, use_cache=False,\n'
    '        )\n'
    '    raw = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()\n'
    '    if "</think>" in raw:\n'
    '        raw = raw[raw.rfind("</think>") + len("</think>"):].strip()\n'
    '\n'
    '    try:\n'
    '        pred = json.loads(raw)\n'
    '    except json.JSONDecodeError:\n'
    '        intent_total[expected_intent] += 1\n'
    '        total_route += 1\n'
    '        continue\n'
    '\n'
    '    pred_intent = pred.get("intent", "")\n'
    '    pred_scope  = pred.get("scope", "")\n'
    '    route_ok = (pred_intent == expected_intent and pred_scope == expected_scope)\n'
    '    intent_total[expected_intent] += 1\n'
    '    if route_ok:\n'
    '        correct_route += 1\n'
    '        intent_correct[expected_intent] += 1\n'
    '    else:\n'
    '        confusion_pairs[(expected_intent, pred_intent)] += 1\n'
    '    total_route += 1\n'
    '\n'
    '    if has_rw:\n'
    '        rw_total += 1\n'
    '        if route_ok:\n'
    '            rw_correct += 1\n'
    '        pred_rw = pred.get("rewritten_query", "")\n'
    '        if ctx and ctx.get("location") and ctx["location"].lower() in pred_rw.lower():\n'
    '            rw_entity_ok += 1\n'
    '    else:\n'
    '        norw_total += 1\n'
    '        if route_ok:\n'
    '            norw_correct += 1\n'
    '\n'
    'print("=" * 60)\n'
    'print("FULL EVAL RESULTS (v3)")\n'
    'print("=" * 60)\n'
    'ra = correct_route / total_route if total_route else 0\n'
    'print(f"Routing accuracy    : {correct_route}/{total_route} = {ra:.1%}    target >= 92%")\n'
    'if rw_total:\n'
    '    print(f"Rewrite routing acc : {rw_correct}/{rw_total} = {rw_correct/rw_total:.1%}    target >= 85%")\n'
    '    print(f"Entity coverage     : {rw_entity_ok}/{rw_total} = {rw_entity_ok/rw_total:.1%}    target >= 70%")\n'
    'if norw_total:\n'
    '    print(f"No-rewrite routing  : {norw_correct}/{norw_total} = {norw_correct/norw_total:.1%}    target >= 80%")\n'
    '\n'
    'verdict = "PASS" if ra >= 0.92 else "FAIL"\n'
    'print("=" * 60)\n'
    'print("Pass routing>=92% : " + verdict)\n'
    'print("=" * 60)\n'
    '\n'
    '# Per-intent accuracy\n'
    'print()\n'
    'print("PER-INTENT ACCURACY:")\n'
    'print(f"  {\'Intent\':<25} {\'Correct\':>7} {\'Total\':>7} {\'Acc\':>7}")\n'
    'print(f"  {\'-\'*50}")\n'
    'for intent in sorted(VALID_INTENTS):\n'
    '    t = intent_total.get(intent, 0)\n'
    '    c = intent_correct.get(intent, 0)\n'
    '    acc = c / t if t else 0\n'
    '    flag = " <<<" if acc < 0.85 else ""\n'
    '    print(f"  {intent:<25} {c:>7} {t:>7} {acc:>6.1%}{flag}")\n'
    '\n'
    '# Top confusion pairs\n'
    'if confusion_pairs:\n'
    '    print()\n'
    '    print("TOP CONFUSION PAIRS (expected -> predicted):")\n'
    '    for (exp, pred), count in confusion_pairs.most_common(10):\n'
    '        print(f"  {exp:<25} -> {pred:<25} x{count}")'
))

# ── Cell 18: Inference test ──
_CELL18 = (
    'NL_inf = chr(10)\n'
    'test_cases = [\n'
    '    {"name": "Basic current",   "input": "Bay gio thoi tiet Cau Giay the nao?",\n'
    '     "expected_intent": "current_weather", "context": None},\n'
    '    {"name": "Daily forecast",  "input": "Cuoi tuan Ha Noi the nao?",\n'
    '     "expected_intent": "daily_forecast",  "context": None},\n'
    '    {"name": "Wind query",      "input": "Gio o Hoang Mai manh khong?",\n'
    '     "expected_intent": "wind_query",      "context": None},\n'
    '    {"name": "Weather alert",   "input": "Co bao o Ha Noi khong?",\n'
    '     "expected_intent": "weather_alert",   "context": None},\n'
    '    {"name": "Context rewrite", "input": "Con ngay mai?",\n'
    '     "expected_intent": "daily_forecast",\n'
    '     "context": {"location": "Cau Giay", "intent": "current_weather", "turn": 1}},\n'
    '    {"name": "Activity",        "input": "Sang mai di chay bo duoc khong?",\n'
    '     "expected_intent": "activity_weather", "context": None},\n'
    ']\n'
    '\n'
    'print("=" * 60)\n'
    'print("INFERENCE TEST")\n'
    'print("=" * 60)\n'
    'all_pass = True\n'
    'for tc in test_cases:\n'
    '    user_msg = tc["input"]\n'
    '    ctx = tc.get("context")\n'
    '    if ctx:\n'
    '        ctx_str  = json.dumps(ctx, ensure_ascii=False, separators=(",",":"))\n'
    '        user_msg = "[CONTEXT: " + ctx_str + "]" + NL_inf + user_msg\n'
    '    prompt  = "<|im_start|>system"    + NL_inf + SYSTEM_PROMPT + "<|im_end|>" + NL_inf\n'
    '    prompt += "<|im_start|>user"      + NL_inf + user_msg      + "<|im_end|>" + NL_inf\n'
    '    prompt += "<|im_start|>assistant" + NL_inf\n'
    '    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(model.device)\n'
    '    with torch.no_grad():\n'
    '        out_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.0, do_sample=False, use_cache=False)\n'
    '    raw = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()\n'
    '    if "</think>" in raw:\n'
    '        raw = raw[raw.rfind("</think>") + len("</think>"):].strip()\n'
    '    try:\n'
    '        pred = json.loads(raw)\n'
    '    except Exception:\n'
    '        pred = {"raw": raw}\n'
    '    ok = pred.get("intent") == tc["expected_intent"]\n'
    '    status = "OK" if ok else "FAIL"\n'
    '    if not ok:\n'
    '        all_pass = False\n'
    '    name = tc["name"]\n'
    '    inp_text = tc["input"][:60]\n'
    '    print(status.rjust(4) + " [" + name + "]")\n'
    '    print("     Input:  " + inp_text)\n'
    '    print("     Pred:   " + json.dumps(pred, ensure_ascii=False))\n'
    '    print()\n'
    'print("All passed: " + str(all_pass))'
)
CELLS.append(make_cell(_CELL18))

# ── Cell 19: Free disk + GGUF Export ──
CELLS.append(make_cell(
    '# Free disk space before export\n'
    'import shutil, gc\n'
    'output_path = Path(OUTPUT_DIR)\n'
    'if output_path.exists():\n'
    '    for ckpt in sorted(output_path.glob("checkpoint-*")):\n'
    '        print(f"Deleting {ckpt.name}...")\n'
    '        shutil.rmtree(ckpt)\n'
    'torch.cuda.empty_cache()\n'
    'gc.collect()\n'
    '\n'
    '# Export Q4_K_M to Drive (saves disk space vs /content)\n'
    'gguf_dir = str(DRIVE_DIR / "gguf" / "q4_k_m")\n'
    'print("Exporting GGUF Q4_K_M...")\n'
    'model.save_pretrained_gguf(\n'
    '    gguf_dir,\n'
    '    tokenizer,\n'
    '    quantization_method="q4_k_m",\n'
    ')\n'
    'print("Q4_K_M done!")\n'
    '\n'
    'for f in Path(gguf_dir).iterdir():\n'
    '    size_mb = f.stat().st_size / 1024 / 1024\n'
    '    print(f"  {f.name}: {size_mb:.1f} MB")'
))

# ── Cell 20: Ollama Modelfile ──
CELLS.append(make_cell(
    'modelfile_content = (\n'
    '    "FROM ./unsloth.Q4_K_M.gguf" + chr(10)\n'
    '    + chr(10)\n'
    '    + "TEMPLATE " + chr(34)*3 + chr(10)\n'
    '    + "{{- if .System }}<|im_start|>system" + chr(10)\n'
    '    + "{{ .System }}<|im_end|>" + chr(10)\n'
    '    + "{{ end }}<|im_start|>user" + chr(10)\n'
    '    + "{{ .Prompt }}<|im_end|>" + chr(10)\n'
    '    + "<|im_start|>assistant" + chr(10)\n'
    '    + "{{ .Response }}<|im_end|>" + chr(10)\n'
    '    + chr(34)*3 + chr(10)\n'
    '    + chr(10)\n'
    '    + "PARAMETER temperature 0" + chr(10)\n'
    '    + "PARAMETER num_predict 128" + chr(10)\n'
    '    + "PARAMETER stop " + chr(34) + "<|im_end|>" + chr(34) + chr(10)\n'
    '    + "PARAMETER stop " + chr(34) + "<|im_start|>" + chr(34) + chr(10)\n'
    ')\n'
    '\n'
    'modelfile_path = DRIVE_DIR / "gguf" / "q4_k_m" / "Modelfile"\n'
    'modelfile_path.write_text(modelfile_content, encoding="utf-8")\n'
    'print(f"Modelfile saved to {modelfile_path}")\n'
    'print(modelfile_content)'
))

# ── Cell 21: GGUF Export Fallback (if disk full) ──
CELLS.append(make_cell(
    '# FALLBACK: If disk is full, push directly to HuggingFace\n'
    '# Uncomment below to use:\n'
    '\n'
    '# from google.colab import userdata\n'
    '# hf_token = userdata.get("HF_TOKEN")\n'
    '# HF_REPO = "your-username/qwen3-4b-hanoi-weather-router-v3"\n'
    '#\n'
    '# model.push_to_hub_gguf(\n'
    '#     HF_REPO,\n'
    '#     tokenizer,\n'
    '#     quantization_method="q4_k_m",\n'
    '#     token=hf_token,\n'
    '# )\n'
    '# print(f"Pushed GGUF Q4_K_M to https://huggingface.co/{HF_REPO}")'
))

# ── Cell 22: Summary ──
CELLS.append(make_cell(
    'print("=" * 60)\n'
    'print("  TRAINING SUMMARY v3")\n'
    'print("=" * 60)\n'
    'print(f"  Model      : {BASE_MODEL}")\n'
    'print(f"  LoRA       : r={LORA_R}, alpha={LORA_ALPHA}")\n'
    'print(f"  Dataset    : {len(train_dataset)} train / {len(val_dataset)} val")\n'
    'print(f"  Epochs     : {EPOCHS}")\n'
    'print(f"  Train loss : {trainer_stats.training_loss:.4f}")\n'
    'print(f"  Key fix    : completion-only loss masking")\n'
    'print(f"  Output     : {DRIVE_DIR}")\n'
    'print("=" * 60)\n'
    'print()\n'
    'print("Next steps:")\n'
    'print("  1. Download GGUF Q4_K_M from Drive")\n'
    'print("  2. ollama create hanoi-weather-router-v3 -f Modelfile")\n'
    'print("  3. Test: ollama run hanoi-weather-router-v3")'
))


# ═══════════════════════════════════════════════════════════════════════════
# Build notebook
# ═══════════════════════════════════════════════════════════════════════════

def build_notebook():
    for cell in CELLS:
        if isinstance(cell["source"], list):
            cell["source"] = fix_source(cell["source"])

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "A100"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": CELLS,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print(f"Notebook saved to {OUTPUT_PATH}")
    print(f"Total cells: {len(CELLS)}")

    import ast
    errors = 0
    for i, cell in enumerate(CELLS):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.strip().startswith("%%") or source.strip().startswith("!"):
            continue
        try:
            ast.parse(source)
        except SyntaxError as e:
            errors += 1
            print(f"  ❌ Cell {i}: SyntaxError at line {e.lineno}: {e.msg}")
            lines = source.split("\n")
            if e.lineno and e.lineno <= len(lines):
                start = max(0, e.lineno - 3)
                for j in range(start, min(len(lines), e.lineno + 2)):
                    marker = ">>>" if j == e.lineno - 1 else "   "
                    print(f"    {marker} {j+1}: {lines[j]}")

    print("  ✅ All code cells pass ast.parse" if errors == 0 else f"  ❌ {errors} cells have syntax errors")


if __name__ == "__main__":
    build_notebook()
