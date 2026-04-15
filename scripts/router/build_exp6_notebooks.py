"""
scripts/router/build_exp6_notebooks.py
Generate Exp6 router ablation notebooks (5 training + 1 eval).

Each training notebook is a self-contained Colab A100 fine-tune recipe:
 - Download clean data from HF dataset (daredevil467/hanoi-weather-router-data)
 - Fine-tune base model with identical LoRA recipe (r=32, alpha=64)
 - Push merged checkpoint to HF model repo (size-specific name)
 - No Google Drive (user explicitly opted out — insufficient storage)

Usage:
    python scripts/router/build_exp6_notebooks.py

Output:
    training/notebooks/exp6_router_ablation/01_train_qwen25_05b.ipynb
    training/notebooks/exp6_router_ablation/02_train_qwen25_15b.ipynb
    training/notebooks/exp6_router_ablation/03_train_qwen3_17b.ipynb
    training/notebooks/exp6_router_ablation/04_train_qwen3_4b_v5.ipynb
    training/notebooks/exp6_router_ablation/05_train_qwen3_8b.ipynb
    training/notebooks/exp6_router_ablation/06_eval_all_sizes.ipynb
    training/notebooks/exp6_router_ablation/README.md
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "training/notebooks/exp6_router_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HF_DATA_REPO = "daredevil467/hanoi-weather-router-data"

# Size-specific training configs. Effective batch size = 32 for all sizes.
SIZE_CONFIGS = [
    {
        "index": "01",
        "label": "qwen25_05b",
        "display_name": "Qwen2.5-0.5B-Instruct",
        "base_model": "unsloth/Qwen2.5-0.5B-Instruct",
        "hf_repo": "daredevil467/hanoi-router-qwen25-05b",
        "batch_size": 16,
        "grad_accum": 2,
    },
    {
        "index": "02",
        "label": "qwen25_15b",
        "display_name": "Qwen2.5-1.5B-Instruct",
        "base_model": "unsloth/Qwen2.5-1.5B-Instruct",
        "hf_repo": "daredevil467/hanoi-router-qwen25-15b",
        "batch_size": 8,
        "grad_accum": 4,
    },
    {
        "index": "03",
        "label": "qwen3_17b",
        "display_name": "Qwen3-1.7B",
        "base_model": "unsloth/Qwen3-1.7B",
        "hf_repo": "daredevil467/hanoi-router-qwen3-17b",
        "batch_size": 8,
        "grad_accum": 4,
    },
    {
        "index": "04",
        "label": "qwen3_4b_v5",
        "display_name": "Qwen3-4B (v5 retrain on clean data)",
        "base_model": "unsloth/Qwen3-4B",
        "hf_repo": "daredevil467/hanoi-router-qwen3-4b-v5",
        "batch_size": 4,
        "grad_accum": 8,
    },
    {
        "index": "05",
        "label": "qwen3_8b",
        "display_name": "Qwen3-8B",
        "base_model": "unsloth/Qwen3-8B",
        "hf_repo": "daredevil467/hanoi-router-qwen3-8b",
        "batch_size": 2,
        "grad_accum": 16,
    },
]


def _code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": src.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


def _md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def _nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"gpuClass": "premium"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ── Training notebook template ─────────────────────────────────────────────

def build_train_notebook(cfg: dict) -> dict:
    cells: list[dict] = []

    cells.append(_md(
        f"# Exp6 Router Ablation — Train {cfg['display_name']}\n"
        f"\n"
        f"**Size**: {cfg['label']}  \n"
        f"**Base model**: `{cfg['base_model']}`  \n"
        f"**Output repo**: `{cfg['hf_repo']}`  \n"
        f"**Data**: [`{HF_DATA_REPO}`](https://huggingface.co/datasets/{HF_DATA_REPO}) (train 3366, val 373)  \n"
        f"**Effective batch**: {cfg['batch_size']} × {cfg['grad_accum']} = {cfg['batch_size'] * cfg['grad_accum']}\n"
        f"\n"
        f"Run on Colab A100 (premium GPU). Before running, set `HF_TOKEN` in Colab Secrets (🔑 panel)."
    ))

    cells.append(_code(
        "%%capture\n"
        "!pip install -U \"unsloth[colab-new]\" \"trl>=0.17,<0.19\" \"huggingface_hub>=0.26\" datasets\n"
    ))

    cells.append(_code(
        "import os, torch, subprocess\n"
        "from google.colab import userdata\n"
        "from huggingface_hub import login\n"
        "\n"
        "# HF login from Colab Secrets (🔑 icon in left panel)\n"
        "HF_TOKEN = userdata.get('HF_TOKEN')\n"
        "assert HF_TOKEN, 'Set HF_TOKEN in Colab Secrets before running'\n"
        "login(token=HF_TOKEN)\n"
        "\n"
        "gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader']).decode().strip()\n"
        "print(f'GPU: {gpu}')\n"
        "print(f'bf16 supported: {torch.cuda.is_bf16_supported()}')\n"
        "assert torch.cuda.is_bf16_supported(), 'A100 (or better) required for bf16 training'\n"
    ))

    cells.append(_code(
        "# ═══════════════════════ CONFIG (Exp6 shared recipe) ═══════════════════════\n"
        f"BASE_MODEL     = {cfg['base_model']!r}\n"
        f"HF_REPO        = {cfg['hf_repo']!r}\n"
        f"HF_DATA_REPO   = {HF_DATA_REPO!r}\n"
        "\n"
        "MAX_SEQ_LENGTH = 1024\n"
        "LORA_R         = 32          # Identical to v4 recipe\n"
        "LORA_ALPHA     = 64\n"
        "LORA_DROPOUT   = 0.1\n"
        "EPOCHS         = 10\n"
        f"BATCH_SIZE     = {cfg['batch_size']}\n"
        f"GRAD_ACCUM     = {cfg['grad_accum']}\n"
        "LR             = 2e-4\n"
        "WARMUP_RATIO   = 0.06\n"
        "EVAL_STEPS     = 50\n"
        "OUTPUT_DIR     = '/content/outputs'\n"
        "\n"
        "print(f'Model      : {BASE_MODEL}')\n"
        "print(f'LoRA       : r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}')\n"
        "print(f'Batch      : {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} effective')\n"
        "print(f'Epochs     : {EPOCHS}')\n"
        "print(f'LR         : {LR}')\n"
        "print(f'HF target  : {HF_REPO}')\n"
    ))

    cells.append(_code(
        "# ═══════════════════════ DATA DOWNLOAD from HF ═══════════════════════\n"
        "from huggingface_hub import hf_hub_download\n"
        "import json as _json\n"
        "\n"
        "train_file = hf_hub_download(repo_id=HF_DATA_REPO, filename='multitask_train_v5_clean.jsonl', repo_type='dataset')\n"
        "val_file   = hf_hub_download(repo_id=HF_DATA_REPO, filename='multitask_val_v5_clean.jsonl',   repo_type='dataset')\n"
        "prompt_file= hf_hub_download(repo_id=HF_DATA_REPO, filename='system_prompt.txt',              repo_type='dataset')\n"
        "\n"
        "with open(train_file, encoding='utf-8') as f:\n"
        "    train_raw = [_json.loads(l) for l in f]\n"
        "with open(val_file, encoding='utf-8') as f:\n"
        "    val_raw = [_json.loads(l) for l in f]\n"
        "with open(prompt_file, encoding='utf-8') as f:\n"
        "    SYSTEM_PROMPT = f.read().strip()\n"
        "\n"
        "print(f'Train: {len(train_raw)} samples')\n"
        "print(f'Val:   {len(val_raw)} samples')\n"
        "print(f'System prompt: {len(SYSTEM_PROMPT)} chars')\n"
        "assert len(train_raw) == 3366, f'Expected 3366 train rows, got {len(train_raw)}'\n"
        "assert len(val_raw)   == 373,  f'Expected 373 val rows, got {len(val_raw)}'\n"
    ))

    cells.append(_code(
        "# ═══════════════════════ LOAD BASE MODEL ═══════════════════════\n"
        "from unsloth import FastLanguageModel\n"
        "\n"
        "model, tokenizer = FastLanguageModel.from_pretrained(\n"
        "    model_name     = BASE_MODEL,\n"
        "    max_seq_length = MAX_SEQ_LENGTH,\n"
        "    dtype          = torch.bfloat16,\n"
        "    load_in_4bit   = False,\n"
        ")\n"
        "print(f'Model loaded: {BASE_MODEL}')\n"
        "print(f'Params: {sum(p.numel() for p in model.parameters()):,}')\n"
    ))

    cells.append(_code(
        "# ═══════════════════════ APPLY LoRA ═══════════════════════\n"
        "model = FastLanguageModel.get_peft_model(\n"
        "    model,\n"
        "    r              = LORA_R,\n"
        "    lora_alpha     = LORA_ALPHA,\n"
        "    lora_dropout   = LORA_DROPOUT,\n"
        "    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',\n"
        "                      'gate_proj', 'up_proj', 'down_proj'],\n"
        "    bias           = 'none',\n"
        "    use_gradient_checkpointing = 'unsloth',\n"
        ")\n"
        "trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
        "total     = sum(p.numel() for p in model.parameters())\n"
        "print(f'Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)')\n"
    ))

    cells.append(_code(
        "# ═══════════════════════ TOKENIZER SETUP (manual ChatML) ═══════════════════════\n"
        "if tokenizer.pad_token is None:\n"
        "    tokenizer.pad_token    = tokenizer.eos_token\n"
        "    tokenizer.pad_token_id = tokenizer.eos_token_id\n"
        "tokenizer.padding_side = 'right'\n"
        "\n"
        "# Verify ChatML special tokens exist (Qwen2/3 family)\n"
        "im_start_ids = tokenizer.encode('<|im_start|>', add_special_tokens=False)\n"
        "im_end_ids   = tokenizer.encode('<|im_end|>',   add_special_tokens=False)\n"
        "print(f'<|im_start|> token ids: {im_start_ids}')\n"
        "print(f'<|im_end|> token ids:   {im_end_ids}')\n"
        "assert len(im_start_ids) == 1 and len(im_end_ids) == 1, 'ChatML special tokens missing'\n"
    ))

    cells.append(_code(
        "# ═══════════════════════ FORMAT RECORDS (ChatML, identical to v4) ═══════════════════════\n"
        "import json\n"
        "from datasets import Dataset\n"
        "\n"
        "IM_START = '<|im_start|>'\n"
        "IM_END   = '<|im_end|>'\n"
        "NL       = chr(10)\n"
        "\n"
        "def format_record(rec):\n"
        "    user_msg = str(rec.get('input', '')).strip()\n"
        "    ctx = rec.get('context')\n"
        "    if ctx:\n"
        "        ctx_str  = json.dumps(ctx, ensure_ascii=False, separators=(',', ':'))\n"
        "        user_msg = '[CONTEXT: ' + ctx_str + ']' + NL + user_msg\n"
        "    out = rec.get('output', {})\n"
        "    if isinstance(out, str):\n"
        "        out = json.loads(out)\n"
        "    output_dict = {\n"
        "        'intent':     out['intent'],\n"
        "        'scope':      out['scope'],\n"
        "        'confidence': round(float(out.get('confidence', 0.9)), 2),\n"
        "    }\n"
        "    rw = out.get('rewritten_query')\n"
        "    if rw and str(rw).strip():\n"
        "        output_dict['rewritten_query'] = str(rw).strip()\n"
        "    text  = IM_START + 'system'    + NL + SYSTEM_PROMPT + IM_END + NL\n"
        "    text += IM_START + 'user'      + NL + user_msg      + IM_END + NL\n"
        "    text += IM_START + 'assistant' + NL + json.dumps(output_dict, ensure_ascii=False) + IM_END + NL\n"
        "    return text\n"
        "\n"
        "train_texts = [format_record(r) for r in train_raw]\n"
        "val_texts   = [format_record(r) for r in val_raw]\n"
        "\n"
        "lengths = [len(tokenizer.encode(t)) for t in train_texts[:200]]\n"
        "print(f'Avg tokens: {sum(lengths)/len(lengths):.0f}, Max: {max(lengths)}, Over {MAX_SEQ_LENGTH}: {sum(1 for l in lengths if l > MAX_SEQ_LENGTH)}')\n"
        "\n"
        "raw_train = Dataset.from_dict({'text': train_texts})\n"
        "raw_val   = Dataset.from_dict({'text': val_texts})\n"
        "print(f'Train: {len(raw_train)}, Val: {len(raw_val)}')\n"
    ))

    cells.append(_code(
        "# ═══════════════════════ COMPLETION-ONLY MASKING (v3 key fix) ═══════════════════════\n"
        "# Mask everything up to '<|im_start|>assistant\\n'. Loss only on the JSON response.\n"
        "ASSISTANT_MARKER = '<|im_start|>assistant' + chr(10)\n"
        "assistant_ids = tokenizer.encode(ASSISTANT_MARKER, add_special_tokens=False)\n"
        "print(f'Assistant marker ids: {assistant_ids}')\n"
        "\n"
        "def tokenize_with_masking(examples):\n"
        "    enc = tokenizer(\n"
        "        examples['text'],\n"
        "        truncation=True,\n"
        "        max_length=MAX_SEQ_LENGTH,\n"
        "        padding=False,\n"
        "        return_tensors=None,\n"
        "    )\n"
        "    all_labels = []\n"
        "    for input_ids in enc['input_ids']:\n"
        "        labels = [-100] * len(input_ids)\n"
        "        marker_len = len(assistant_ids)\n"
        "        for i in range(len(input_ids) - marker_len + 1):\n"
        "            if input_ids[i:i+marker_len] == assistant_ids:\n"
        "                start = i + marker_len\n"
        "                for j in range(start, len(input_ids)):\n"
        "                    labels[j] = input_ids[j]\n"
        "                break\n"
        "        else:\n"
        "            labels = list(input_ids)  # fallback (should not happen)\n"
        "        all_labels.append(labels)\n"
        "    enc['labels'] = all_labels\n"
        "    return enc\n"
        "\n"
        "train_dataset = raw_train.map(tokenize_with_masking, batched=True, remove_columns=['text'])\n"
        "val_dataset   = raw_val.map(tokenize_with_masking, batched=True, remove_columns=['text'])\n"
        "\n"
        "# Sanity check\n"
        "sl = train_dataset[0]['labels']\n"
        "masked = sum(1 for x in sl if x == -100)\n"
        "print(f'Sample: {len(sl)} total, {masked} masked ({masked/len(sl)*100:.1f}%), {len(sl)-masked} active')\n"
    ))

    cells.append(_code(
        "# ═══════════════════════ TRAINER SETUP ═══════════════════════\n"
        "from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback\n"
        "from trl import SFTConfig, SFTTrainer\n"
        "\n"
        "data_collator = DataCollatorForSeq2Seq(\n"
        "    tokenizer          = tokenizer,\n"
        "    model              = model,\n"
        "    label_pad_token_id = -100,\n"
        "    pad_to_multiple_of = 8,\n"
        ")\n"
        "\n"
        "training_args = SFTConfig(\n"
        "    output_dir                  = OUTPUT_DIR,\n"
        "    num_train_epochs            = EPOCHS,\n"
        "    per_device_train_batch_size = BATCH_SIZE,\n"
        "    per_device_eval_batch_size  = BATCH_SIZE,\n"
        "    gradient_accumulation_steps = GRAD_ACCUM,\n"
        "    learning_rate               = LR,\n"
        "    warmup_ratio                = WARMUP_RATIO,\n"
        "    lr_scheduler_type           = 'cosine',\n"
        "    logging_steps               = 10,\n"
        "    eval_strategy               = 'steps',\n"
        "    eval_steps                  = EVAL_STEPS,\n"
        "    save_strategy               = 'steps',\n"
        "    save_steps                  = EVAL_STEPS,\n"
        "    save_total_limit            = 2,\n"
        "    metric_for_best_model       = 'eval_loss',\n"
        "    load_best_model_at_end      = True,\n"
        "    bf16                        = True,\n"
        "    fp16                        = False,\n"
        "    optim                       = 'adamw_torch_fused',\n"
        "    weight_decay                = 0.01,\n"
        "    max_grad_norm               = 1.0,\n"
        "    max_seq_length              = MAX_SEQ_LENGTH,\n"
        "    report_to                   = 'none',\n"
        ")\n"
        "\n"
        "early_stop = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)\n"
        "\n"
        "trainer = SFTTrainer(\n"
        "    model            = model,\n"
        "    processing_class = tokenizer,\n"
        "    args             = training_args,\n"
        "    train_dataset    = train_dataset,\n"
        "    eval_dataset     = val_dataset,\n"
        "    data_collator    = data_collator,\n"
        "    packing          = False,\n"
        "    callbacks        = [early_stop],\n"
        ")\n"
        "print('Trainer ready')\n"
    ))

    cells.append(_code(
        "# ═══════════════════════ TRAIN ═══════════════════════\n"
        "trainer_stats = trainer.train()\n"
        "print(trainer_stats)\n"
    ))

    cells.append(_code(
        "# ═══════════════════════ SAVE ADAPTER + MERGE + PUSH to HF ═══════════════════════\n"
        "# 1. Save LoRA adapter locally\n"
        "adapter_dir = '/content/outputs/lora_adapter'\n"
        "model.save_pretrained(adapter_dir)\n"
        "tokenizer.save_pretrained(adapter_dir)\n"
        "print(f'Adapter saved to {adapter_dir}')\n"
        "\n"
        "# 2. Free disk (checkpoints) before merge\n"
        "import shutil, gc\n"
        "from pathlib import Path\n"
        "out = Path(OUTPUT_DIR)\n"
        "for ckpt in sorted(out.glob('checkpoint-*')):\n"
        "    print(f'Deleting {ckpt.name}...')\n"
        "    shutil.rmtree(ckpt)\n"
        "torch.cuda.empty_cache()\n"
        "gc.collect()\n"
        "\n"
        "# 3. Push merged model (16-bit) to HF Hub — used by eval notebook 06\n"
        "print(f'Pushing merged model to HF: {HF_REPO}')\n"
        "model.push_to_hub_merged(\n"
        "    HF_REPO,\n"
        "    tokenizer,\n"
        "    save_method='merged_16bit',\n"
        "    token=HF_TOKEN,\n"
        ")\n"
        "print(f'Pushed: https://huggingface.co/{HF_REPO}')\n"
    ))

    cells.append(_code(
        "# ═══════════════════════ (OPTIONAL) Push GGUF Q4_K_M for Ollama production ═══════════════════════\n"
        "# Run this cell only for the size you want to deploy via Ollama locally.\n"
        "# GGUF export takes ~5-15 min depending on model size; safe to skip for ablation-only runs.\n"
        "\n"
        "GGUF_HF_REPO = HF_REPO + '-gguf'\n"
        "print(f'Pushing GGUF Q4_K_M to {GGUF_HF_REPO}...')\n"
        "model.push_to_hub_gguf(\n"
        "    GGUF_HF_REPO,\n"
        "    tokenizer,\n"
        "    quantization_method='q4_k_m',\n"
        "    token=HF_TOKEN,\n"
        ")\n"
        "print(f'Pushed GGUF: https://huggingface.co/{GGUF_HF_REPO}')\n"
    ))

    return _nb(cells)


# ── Eval notebook (06) ─────────────────────────────────────────────────────

def build_eval_notebook() -> dict:
    cells: list[dict] = []

    cells.append(_md(
        "# Exp6 — Eval All Router Sizes (Colab A100)\n"
        "\n"
        "Runs FP16 inference on 373 clean val samples for 5 merged router models.\n"
        "\n"
        "**Metrics** (aligned with v4 notebook evaluation methodology):\n"
        "- **Routing accuracy** = intent + scope both correct (primary metric, same as v4)\n"
        "- Intent accuracy + Wilson 95% CI\n"
        "- Scope accuracy\n"
        "- Macro-F1 + per-intent F1\n"
        "- Per-intent accuracy + top confusion pairs (same as v4)\n"
        "- Rewrite routing accuracy + entity coverage (multi-turn samples)\n"
        "- Latency P50/P90/P95 (GPU FP16, not production Ollama CPU)\n"
        "\n"
        "**Output**: `exp6_summary.json` + plots + per-sample CSV.\n"
        "\n"
        "⚠️ **Quantization note**: This notebook uses **FP16** (merged_16bit on HF). Production deployment\n"
        "uses **Q4_K_M GGUF** via Ollama — expected delta ≤1pp, negligible for ranking.\n"
    ))

    cells.append(_code(
        "%%capture\n"
        "!pip install -U transformers accelerate huggingface_hub scikit-learn\n"
    ))

    cells.append(_code(
        "import os, json, time, re, gc, math\n"
        "import torch\n"
        "from pathlib import Path\n"
        "from collections import Counter, defaultdict\n"
        "from google.colab import userdata\n"
        "from huggingface_hub import login, hf_hub_download\n"
        "\n"
        "HF_TOKEN = userdata.get('HF_TOKEN')\n"
        "assert HF_TOKEN, 'Set HF_TOKEN in Colab Secrets'\n"
        "login(token=HF_TOKEN)\n"
        "\n"
        "import subprocess\n"
        "gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader']).decode().strip()\n"
        "print(f'GPU: {gpu}')\n"
    ))

    cells.append(_code(
        "# ═══════════════════════ CONFIGS ═══════════════════════\n"
        "HF_DATA_REPO = 'daredevil467/hanoi-weather-router-data'\n"
        "\n"
        "MODELS = [\n"
        "    ('Qwen2.5-0.5B', 'daredevil467/hanoi-router-qwen25-05b',   0.5),\n"
        "    ('Qwen2.5-1.5B', 'daredevil467/hanoi-router-qwen25-15b',   1.5),\n"
        "    ('Qwen3-1.7B',   'daredevil467/hanoi-router-qwen3-17b',    1.7),\n"
        "    ('Qwen3-4B-v5',  'daredevil467/hanoi-router-qwen3-4b-v5',  4.0),\n"
        "    ('Qwen3-8B',     'daredevil467/hanoi-router-qwen3-8b',     8.0),\n"
        "]\n"
        "OUTPUT_DIR = Path('/content/outputs')\n"
        "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    ))

    cells.append(_code(
        "# ═══════════════════════ LOAD VAL DATA + SYSTEM PROMPT ═══════════════════════\n"
        "val_file    = hf_hub_download(repo_id=HF_DATA_REPO, filename='multitask_val_v5_clean.jsonl', repo_type='dataset')\n"
        "prompt_file = hf_hub_download(repo_id=HF_DATA_REPO, filename='system_prompt.txt',            repo_type='dataset')\n"
        "\n"
        "with open(val_file, encoding='utf-8') as f:\n"
        "    val_samples = [json.loads(l) for l in f]\n"
        "with open(prompt_file, encoding='utf-8') as f:\n"
        "    SYSTEM_PROMPT = f.read().strip()\n"
        "\n"
        "print(f'Val samples: {len(val_samples)}')\n"
        "print(f'System prompt: {len(SYSTEM_PROMPT)} chars')\n"
        "n_with_ctx = sum(1 for s in val_samples if s.get('context'))\n"
        "n_with_rw  = sum(1 for s in val_samples if s['output'].get('rewritten_query'))\n"
        "print(f'With context: {n_with_ctx}  |  With rewrite: {n_with_rw}')\n"
    ))

    cells.append(_code(
        "# ═══════════════════════ HELPERS ═══════════════════════\n"
        "\n"
        "def wilson_ci(k: int, n: int, z: float = 1.96):\n"
        "    if n == 0:\n"
        "        return (0.0, 0.0)\n"
        "    p = k / n\n"
        "    denom = 1 + z*z/n\n"
        "    center = (p + z*z/(2*n)) / denom\n"
        "    margin = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom\n"
        "    return (round((center - margin)*100, 2), round((center + margin)*100, 2))\n"
        "\n"
        "JSON_RE = re.compile(r'\\{[^{}]*\\}', re.DOTALL)\n"
        "\n"
        "def parse_output(text: str):\n"
        "    \"\"\"Parse JSON from model output, stripping </think> tags (Qwen3).\"\"\"\n"
        "    # Strip thinking tags if present (Qwen3 family)\n"
        "    if '</think>' in text:\n"
        "        text = text[text.rfind('</think>') + len('</think>'):].strip()\n"
        "    m = JSON_RE.search(text)\n"
        "    if not m:\n"
        "        return None\n"
        "    try:\n"
        "        return json.loads(m.group(0))\n"
        "    except Exception:\n"
        "        return None\n"
        "\n"
        "def build_prompt(sample: dict) -> str:\n"
        "    \"\"\"Build ChatML prompt, prepending [CONTEXT: ...] when context exists (same as v4).\"\"\"\n"
        "    user_msg = str(sample.get('input', '')).strip()\n"
        "    ctx = sample.get('context')\n"
        "    if ctx:\n"
        "        ctx_str = json.dumps(ctx, ensure_ascii=False, separators=(',', ':'))\n"
        "        user_msg = '[CONTEXT: ' + ctx_str + ']\\n' + user_msg\n"
        "    return (\n"
        "        '<|im_start|>system\\n' + SYSTEM_PROMPT + '<|im_end|>\\n'\n"
        "        '<|im_start|>user\\n'   + user_msg      + '<|im_end|>\\n'\n"
        "        '<|im_start|>assistant\\n'\n"
        "    )\n"
    ))

    cells.append(_code(
        "# ═══════════════════════ EVAL LOOP — 5 models ═══════════════════════\n"
        "# Metrics aligned with slm_router_qwen3_colab_a100_v4.ipynb (Cell 19)\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "\n"
        "results = {}\n"
        "per_sample_rows = []\n"
        "\n"
        "for name, repo, params_b in MODELS:\n"
        "    print(f'\\n{\"=\"*60}')\n"
        "    print(f'Evaluating {name} ({repo})')\n"
        "    print(f'{\"=\"*60}')\n"
        "\n"
        "    tokenizer = AutoTokenizer.from_pretrained(repo, token=HF_TOKEN)\n"
        "    model = AutoModelForCausalLM.from_pretrained(\n"
        "        repo, token=HF_TOKEN, torch_dtype=torch.float16, device_map='auto',\n"
        "    )\n"
        "    model.eval()\n"
        "\n"
        "    if tokenizer.pad_token is None:\n"
        "        tokenizer.pad_token = tokenizer.eos_token\n"
        "\n"
        "    # ── Counters (same structure as v4) ──\n"
        "    correct_route  = 0   # intent + scope both correct (primary metric)\n"
        "    total_route    = 0\n"
        "    correct_intent = 0\n"
        "    correct_scope  = 0\n"
        "    parse_failures = 0\n"
        "    # Rewrite tracking (v4 parity)\n"
        "    rw_correct     = 0   # routing correct on rewrite samples\n"
        "    rw_total       = 0   # samples that have expected rewritten_query\n"
        "    rw_entity_ok   = 0   # rewrite contains context location entity\n"
        "    norw_correct   = 0   # routing correct on non-rewrite samples\n"
        "    norw_total     = 0\n"
        "    # Per-intent tracking\n"
        "    intent_correct = Counter()\n"
        "    intent_total   = Counter()\n"
        "    confusion_pairs = Counter()  # (expected, predicted)\n"
        "    intent_true, intent_pred = [], []\n"
        "    latencies = []\n"
        "\n"
        "    for idx, sample in enumerate(val_samples):\n"
        "        prompt = build_prompt(sample)\n"
        "        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to('cuda')\n"
        "        t0 = time.time()\n"
        "        with torch.no_grad():\n"
        "            out = model.generate(\n"
        "                **inputs,\n"
        "                max_new_tokens=128,\n"
        "                do_sample=False,\n"
        "                temperature=None,\n"
        "                top_p=None,\n"
        "                pad_token_id=tokenizer.eos_token_id,\n"
        "            )\n"
        "        latency_ms = (time.time() - t0) * 1000\n"
        "        latencies.append(latency_ms)\n"
        "\n"
        "        gen_ids = out[0][inputs['input_ids'].shape[1]:]\n"
        "        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()\n"
        "        pred = parse_output(gen_text)\n"
        "\n"
        "        gt = sample['output']\n"
        "        expected_intent = gt['intent']\n"
        "        expected_scope  = gt['scope']\n"
        "        has_rw = bool(gt.get('rewritten_query'))\n"
        "        intent_true.append(expected_intent)\n"
        "        intent_total[expected_intent] += 1\n"
        "        total_route += 1\n"
        "\n"
        "        if pred is None:\n"
        "            parse_failures += 1\n"
        "            intent_pred.append('<parse_error>')\n"
        "            confusion_pairs[(expected_intent, '<parse_error>')] += 1\n"
        "            if has_rw:\n"
        "                rw_total += 1\n"
        "            else:\n"
        "                norw_total += 1\n"
        "        else:\n"
        "            pi = pred.get('intent', '')\n"
        "            ps = pred.get('scope',  '')\n"
        "            intent_pred.append(pi)\n"
        "            route_ok = (pi == expected_intent and ps == expected_scope)\n"
        "\n"
        "            if pi == expected_intent:\n"
        "                correct_intent += 1\n"
        "                intent_correct[expected_intent] += 1\n"
        "            else:\n"
        "                confusion_pairs[(expected_intent, pi)] += 1\n"
        "            if ps == expected_scope:\n"
        "                correct_scope += 1\n"
        "            if route_ok:\n"
        "                correct_route += 1\n"
        "\n"
        "            # Rewrite tracking (v4 parity)\n"
        "            if has_rw:\n"
        "                rw_total += 1\n"
        "                if route_ok:\n"
        "                    rw_correct += 1\n"
        "                pred_rw = pred.get('rewritten_query', '')\n"
        "                ctx = sample.get('context')\n"
        "                if ctx and ctx.get('location') and ctx['location'].lower() in pred_rw.lower():\n"
        "                    rw_entity_ok += 1\n"
        "            else:\n"
        "                norw_total += 1\n"
        "                if route_ok:\n"
        "                    norw_correct += 1\n"
        "\n"
        "        per_sample_rows.append({\n"
        "            'model': name, 'query': sample['input'],\n"
        "            'has_context': bool(sample.get('context')),\n"
        "            'gt_intent': expected_intent, 'gt_scope': expected_scope,\n"
        "            'pred_raw': gen_text,\n"
        "            'pred_intent': intent_pred[-1],\n"
        "            'pred_scope': pred.get('scope', '') if pred else '',\n"
        "            'route_ok': pred is not None and (pred.get('intent') == expected_intent and pred.get('scope') == expected_scope),\n"
        "            'latency_ms': round(latency_ms, 1),\n"
        "        })\n"
        "\n"
        "        if (idx + 1) % 50 == 0:\n"
        "            print(f'  [{idx+1}/{len(val_samples)}] routing_acc={correct_route/(idx+1)*100:.1f}%')\n"
        "\n"
        "    n = len(val_samples)\n"
        "    routing_acc = correct_route / n\n"
        "    intent_acc  = correct_intent / n\n"
        "    scope_acc   = correct_scope  / n\n"
        "    ci_low, ci_high = wilson_ci(correct_route, n)  # CI on routing accuracy (primary)\n"
        "    ci_intent_low, ci_intent_high = wilson_ci(correct_intent, n)\n"
        "\n"
        "    # Macro F1 + per-intent F1\n"
        "    from sklearn.metrics import f1_score, classification_report\n"
        "    labels = sorted(set(intent_true))\n"
        "    macro_f1 = f1_score(intent_true, intent_pred, labels=labels, average='macro', zero_division=0)\n"
        "    per_intent_report = classification_report(intent_true, intent_pred, labels=labels, zero_division=0, output_dict=True)\n"
        "\n"
        "    sorted_lats = sorted(latencies)\n"
        "    def pctl(p):\n"
        "        return sorted_lats[int(len(sorted_lats)*p)] if sorted_lats else 0\n"
        "\n"
        "    results[name] = {\n"
        "        'repo': repo,\n"
        "        'params_billions': params_b,\n"
        "        # Primary metric (v4 parity): intent+scope both correct\n"
        "        'routing_accuracy_pct':  round(routing_acc*100, 2),\n"
        "        'routing_ci95':          [ci_low, ci_high],\n"
        "        # Breakdown\n"
        "        'intent_accuracy_pct':   round(intent_acc*100, 2),\n"
        "        'intent_ci95':           [ci_intent_low, ci_intent_high],\n"
        "        'scope_accuracy_pct':    round(scope_acc*100, 2),\n"
        "        'macro_f1':              round(macro_f1, 4),\n"
        "        'parse_failures':        parse_failures,\n"
        "        # Rewrite tracking (v4 parity)\n"
        "        'rewrite_routing_acc':   round(rw_correct/rw_total*100, 2) if rw_total else None,\n"
        "        'rewrite_entity_cov':    round(rw_entity_ok/rw_total*100, 2) if rw_total else None,\n"
        "        'rewrite_n':             rw_total,\n"
        "        'no_rewrite_routing_acc': round(norw_correct/norw_total*100, 2) if norw_total else None,\n"
        "        # Latency\n"
        "        'latency_p50_ms':        round(pctl(0.50), 1),\n"
        "        'latency_p90_ms':        round(pctl(0.90), 1),\n"
        "        'latency_p95_ms':        round(pctl(0.95), 1),\n"
        "        # Per-intent detail\n"
        "        'per_intent_f1':         {k: round(v['f1-score'], 3) for k, v in per_intent_report.items() if isinstance(v, dict) and 'f1-score' in v},\n"
        "        'per_intent_accuracy':   {i: round(intent_correct[i]/intent_total[i]*100, 1) if intent_total[i] else 0 for i in sorted(intent_total)},\n"
        "        'top_confusion_pairs':   [(f'{e}->{p}', c) for (e, p), c in confusion_pairs.most_common(10)],\n"
        "    }\n"
        "\n"
        "    # Print summary (v4-style)\n"
        "    print(f'  Routing acc:  {correct_route}/{n} = {routing_acc:.1%}  CI=[{ci_low}, {ci_high}]')\n"
        "    print(f'  Intent acc:   {correct_intent}/{n} = {intent_acc:.1%}  Macro-F1={macro_f1:.4f}')\n"
        "    print(f'  Scope acc:    {correct_scope}/{n} = {scope_acc:.1%}')\n"
        "    if rw_total:\n"
        "        print(f'  Rewrite routing: {rw_correct}/{rw_total} = {rw_correct/rw_total:.1%}  Entity cov: {rw_entity_ok}/{rw_total} = {rw_entity_ok/rw_total:.1%}')\n"
        "    print(f'  P50={pctl(0.50):.0f}ms  ParseFail={parse_failures}')\n"
        "\n"
        "    # Per-intent accuracy table\n"
        "    print(f'  {\"Intent\":<25} {\"Correct\":>7} {\"Total\":>7} {\"Acc\":>7}')\n"
        "    for intent in sorted(intent_total):\n"
        "        t = intent_total[intent]\n"
        "        c = intent_correct.get(intent, 0)\n"
        "        acc = c/t if t else 0\n"
        "        flag = ' <<<' if acc < 0.85 else ''\n"
        "        print(f'  {intent:<25} {c:>7} {t:>7} {acc:>6.1%}{flag}')\n"
        "\n"
        "    if confusion_pairs:\n"
        "        print(f'  TOP CONFUSION:')\n"
        "        for (exp, pred), cnt in confusion_pairs.most_common(5):\n"
        "            print(f'    {exp:<25} -> {pred:<25} x{cnt}')\n"
        "\n"
        "    # Free GPU\n"
        "    del model, tokenizer\n"
        "    torch.cuda.empty_cache()\n"
        "    gc.collect()\n"
    ))

    cells.append(_code(
        "# ═══════════════════════ SAVE RESULTS ═══════════════════════\n"
        "summary = {\n"
        "    'experiment': 'Exp6_Router_Size_Ablation',\n"
        "    'rq': 'RQ5: Is Qwen3-4B the sweet spot for router size?',\n"
        "    'dataset': 'multitask_val_v5_clean.jsonl (373 samples, 0% leakage)',\n"
        "    'inference_precision': 'fp16 (merged_16bit)',\n"
        "    'primary_metric': 'routing_accuracy_pct (intent+scope both correct, same as v4)',\n"
        "    'models': results,\n"
        "}\n"
        "\n"
        "with open(OUTPUT_DIR / 'exp6_summary.json', 'w', encoding='utf-8') as f:\n"
        "    json.dump(summary, f, ensure_ascii=False, indent=2)\n"
        "\n"
        "# Per-sample CSV\n"
        "import csv\n"
        "with open(OUTPUT_DIR / 'exp6_per_sample.csv', 'w', encoding='utf-8', newline='') as f:\n"
        "    w = csv.DictWriter(f, fieldnames=per_sample_rows[0].keys())\n"
        "    w.writeheader()\n"
        "    w.writerows(per_sample_rows)\n"
        "\n"
        "print('Saved:')\n"
        "print(f'  {OUTPUT_DIR}/exp6_summary.json')\n"
        "print(f'  {OUTPUT_DIR}/exp6_per_sample.csv')\n"
    ))

    cells.append(_code(
        "# ═══════════════════════ PLOTS ═══════════════════════\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "sizes = [results[name]['params_billions'] for name, _, _ in MODELS]\n"
        "raccs = [results[name]['routing_accuracy_pct'] for name, _, _ in MODELS]\n"
        "iaccs = [results[name]['intent_accuracy_pct'] for name, _, _ in MODELS]\n"
        "lats  = [results[name]['latency_p50_ms'] for name, _, _ in MODELS]\n"
        "names = [name for name, _, _ in MODELS]\n"
        "\n"
        "# Figure 1: Size vs Routing Accuracy (primary, v4-aligned)\n"
        "fig, ax = plt.subplots(figsize=(8, 5))\n"
        "ax.plot(sizes, raccs, 'o-', linewidth=2, markersize=10, label='Routing (intent+scope)')\n"
        "ax.plot(sizes, iaccs, 's--', linewidth=1.5, markersize=8, alpha=0.6, label='Intent only')\n"
        "for s, a, n in zip(sizes, raccs, names):\n"
        "    ax.annotate(f'{n}\\n{a:.1f}%', (s, a), textcoords='offset points', xytext=(8, -4), fontsize=8)\n"
        "ax.set_xscale('log')\n"
        "ax.set_xlabel('Model size (billion params, log scale)')\n"
        "ax.set_ylabel('Accuracy (%)')\n"
        "ax.set_title('Exp6: Router Size vs Routing Accuracy (373 clean val)')\n"
        "ax.legend()\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.savefig(OUTPUT_DIR / 'exp6_size_vs_accuracy.png', dpi=150)\n"
        "plt.show()\n"
        "\n"
        "# Figure 2: Size vs Latency\n"
        "fig, ax = plt.subplots(figsize=(8, 5))\n"
        "ax.plot(sizes, lats, 's-', color='orange', linewidth=2, markersize=10)\n"
        "for s, l, n in zip(sizes, lats, names):\n"
        "    ax.annotate(n, (s, l), textcoords='offset points', xytext=(8, -4), fontsize=9)\n"
        "ax.set_xscale('log')\n"
        "ax.set_xlabel('Model size (billion params, log scale)')\n"
        "ax.set_ylabel('Latency P50 (ms, GPU FP16)')\n"
        "ax.set_title('Exp6: Router Size vs Inference Latency')\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.savefig(OUTPUT_DIR / 'exp6_size_vs_latency.png', dpi=150)\n"
        "plt.show()\n"
        "\n"
        "# Figure 3: Per-intent F1 heatmap\n"
        "import numpy as np\n"
        "intents = sorted(results[MODELS[0][0]]['per_intent_f1'].keys())\n"
        "f1_matrix = []\n"
        "for name, _, _ in MODELS:\n"
        "    f1_matrix.append([results[name]['per_intent_f1'].get(i, 0) for i in intents])\n"
        "f1_matrix = np.array(f1_matrix)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(14, 5))\n"
        "im = ax.imshow(f1_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')\n"
        "ax.set_xticks(range(len(intents)))\n"
        "ax.set_xticklabels(intents, rotation=45, ha='right', fontsize=8)\n"
        "ax.set_yticks(range(len(MODELS)))\n"
        "ax.set_yticklabels(names)\n"
        "for i in range(len(MODELS)):\n"
        "    for j in range(len(intents)):\n"
        "        ax.text(j, i, f'{f1_matrix[i,j]:.2f}', ha='center', va='center', fontsize=7)\n"
        "plt.colorbar(im, ax=ax, label='F1')\n"
        "ax.set_title('Exp6: Per-Intent F1 by Router Size')\n"
        "plt.tight_layout()\n"
        "plt.savefig(OUTPUT_DIR / 'exp6_per_intent_f1_heatmap.png', dpi=150)\n"
        "plt.show()\n"
        "\n"
        "print('\\nDownload all files from /content/outputs/ (left panel > Files)')\n"
    ))

    return _nb(cells)


# ── README ─────────────────────────────────────────────────────────────────

README = """# Exp6 — Router Size Ablation Notebooks

Fine-tunes 5 router models of increasing size on identical clean data to answer RQ5:
**Is Qwen3-4B the sweet spot for Hanoi weather router?**

## Run order

1. **Upload clean data to HuggingFace** (run once, locally):
   ```bash
   export HF_TOKEN=hf_...
   python scripts/router/upload_router_data_hf.py
   ```
   Creates dataset: `daredevil467/hanoi-weather-router-data`

2. **Colab Secrets**: Add `HF_TOKEN` in Colab left panel 🔑 (same token).

3. **Run training notebooks** (one at a time on A100 Colab):
   - `01_train_qwen25_05b.ipynb`  (~30 min)
   - `02_train_qwen25_15b.ipynb`  (~1h)
   - `03_train_qwen3_17b.ipynb`   (~1h)
   - `04_train_qwen3_4b_v5.ipynb` (~2h)
   - `05_train_qwen3_8b.ipynb`    (~3h)

   Each notebook pushes merged_16bit checkpoint to HF (size-specific repo).

4. **Run eval notebook** (single run, evaluates all 5 models):
   - `06_eval_all_sizes.ipynb` (~30 min total)

   Downloads `exp6_summary.json` + `exp6_per_sample.csv` + plots from `/content/outputs/`.

5. **Copy results to local** `data/evaluation/thesis_final/exp6_router_size/`.

## Recipe (identical across sizes — fair comparison)

- LoRA: r=32, alpha=64, dropout=0.1
- Target modules: q/k/v/o + gate/up/down proj
- Epochs: 10 with EarlyStoppingCallback (patience=5)
- Effective batch: 32 (varied per-device batch + grad_accum based on size)
- LR: 2e-4 cosine, warmup 6%
- Completion-only loss masking (mask up to `<|im_start|>assistant\\n`)
- bf16 training on A100

## HuggingFace repos produced

| Size | Base | Output |
|------|------|--------|
| 0.5B | unsloth/Qwen2.5-0.5B-Instruct | daredevil467/hanoi-router-qwen25-05b |
| 1.5B | unsloth/Qwen2.5-1.5B-Instruct | daredevil467/hanoi-router-qwen25-15b |
| 1.7B | unsloth/Qwen3-1.7B            | daredevil467/hanoi-router-qwen3-17b  |
| 4B   | unsloth/Qwen3-4B              | daredevil467/hanoi-router-qwen3-4b-v5 |
| 8B   | unsloth/Qwen3-8B              | daredevil467/hanoi-router-qwen3-8b   |

## Notes

- **No Google Drive**: all checkpoints push directly to HF (GDrive free tier insufficient).
- **GGUF export**: only needed for production Ollama deployment; GGUF cell at end of each
  training notebook is optional and off by default. Run it only for the sweet-spot model
  you'll deploy locally (likely `qwen3-4b-v5`).
- **Inference precision mismatch note**: eval notebook uses FP16, production uses Q4_K_M GGUF.
  Delta expected ≤1pp, documented in thesis Limitations section.
"""


def main() -> None:
    for cfg in SIZE_CONFIGS:
        path = OUT_DIR / f"{cfg['index']}_train_{cfg['label']}.ipynb"
        nb = build_train_notebook(cfg)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"Wrote {path}")

    eval_path = OUT_DIR / "06_eval_all_sizes.ipynb"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(build_eval_notebook(), f, ensure_ascii=False, indent=1)
    print(f"Wrote {eval_path}")

    readme_path = OUT_DIR / "README.md"
    readme_path.write_text(README, encoding="utf-8")
    print(f"Wrote {readme_path}")


if __name__ == "__main__":
    main()
