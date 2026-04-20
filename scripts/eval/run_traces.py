"""Trace runner: chạy từng câu hỏi qua agent + log full chain vào JSONL.

Gọi `run_agent_routed()` trực tiếp (Python process trên host, không qua HTTP).
Agent sẽ dùng config từ `.env`:
  - OLLAMA_BASE_URL=<cloudflared tunnel Colab>  → SLM router
  - AGENT_API_BASE=<provider>                    → LLM agent

Output schema (1 dòng JSON per câu hỏi):
{
  "id": int, "question": str,
  "expected": {"intent","scope","location","difficulty","notes"},
  "router": {"path","intent","scope","confidence","latency_ms","fallback_reason","rewritten_query","focused_tools"},
  "tool_calls": [{"seq","name","args","output","tool_call_id"}],
  "final_answer": str,
  "total_latency_ms": float,
  "completeness": {"tool_count_msgs","tool_count_ai","mismatch"},
  "error": str | null,
  "timestamp": str
}

Usage:
    python -m scripts.eval.run_traces \\
        --input data/evaluation/hanoi_weather_chatbot_eval_questions.csv \\
        --output data/evaluation/traces/run1.jsonl
    # Test 5 câu đầu:
    python -m scripts.eval.run_traces --input ... --output ... --limit 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path


# ── Tool extraction ─────────────────────────────────────────────────


def _extract_tool_calls(messages: list) -> tuple[list[dict], dict]:
    """Duyệt messages, pair AIMessage.tool_calls với ToolMessage response.

    Trả về (calls, completeness) trong đó completeness = {
        "tool_count_msgs": số ToolMessage,
        "tool_count_ai": số tool_call declare trong AIMessage,
        "mismatch": True nếu 2 số không bằng nhau,
    }
    """
    # Map tool_call_id → ToolMessage content
    tool_outputs: dict[str, str] = {}
    tool_msg_count = 0
    for msg in messages:
        if getattr(msg, "type", None) == "tool":
            tc_id = getattr(msg, "tool_call_id", "") or ""
            content = getattr(msg, "content", "")
            if tc_id:
                tool_outputs[tc_id] = str(content)
            tool_msg_count += 1

    calls = []
    seq = 0
    ai_call_count = 0
    for msg in messages:
        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
            continue
        for tc in msg.tool_calls:
            ai_call_count += 1
            if isinstance(tc, dict):
                name = tc.get("name")
                args = tc.get("args", {})
                tc_id = tc.get("id", "")
            else:
                name = getattr(tc, "name", None)
                args = getattr(tc, "args", {})
                tc_id = getattr(tc, "id", "")
            if not name:
                continue
            seq += 1
            calls.append({
                "seq": seq,
                "name": name,
                "args": args if isinstance(args, dict) else str(args),
                "output": tool_outputs.get(tc_id, ""),
                "tool_call_id": tc_id,
            })

    completeness = {
        "tool_count_msgs": tool_msg_count,
        "tool_count_ai": ai_call_count,
        "mismatch": tool_msg_count != ai_call_count,
    }
    return calls, completeness


def _extract_final_answer(messages: list) -> str:
    """Lấy content của AIMessage cuối cùng không có tool_calls (= final response)."""
    for msg in reversed(messages):
        msg_type = getattr(msg, "type", None)
        if msg_type != "ai":
            continue
        # Skip AI messages that are just tool_call announcements
        has_tool_calls = getattr(msg, "tool_calls", None)
        content = getattr(msg, "content", "")
        if not has_tool_calls and content:
            return str(content)
    return ""


# ── Trace one question ────────────────────────────────────────────────


def trace_one(row: dict, run_agent_fn) -> dict:
    """Chạy 1 câu hỏi, trả về trace dict."""
    qid = row.get("id", "").strip()
    question = row.get("question", "").strip()

    trace = {
        "id": int(qid) if qid.isdigit() else qid,
        "question": question,
        "expected": {
            "intent": row.get("intent", ""),
            "scope": row.get("location_scope", ""),
            "location": row.get("location_name", ""),
            "time": row.get("time_expression", ""),
            "weather_param": row.get("weather_param", ""),
            "difficulty": row.get("difficulty", ""),
            "notes": row.get("notes", ""),
        },
        "router": None,
        "tool_calls": [],
        "final_answer": "",
        "total_latency_ms": 0.0,
        "completeness": {},
        "error": None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    # Thread_id riêng cho mỗi câu → tránh bleed state giữa câu không liên quan
    thread_id = f"trace-{qid}-{uuid.uuid4().hex[:8]}"

    t_start = time.time()
    try:
        result = run_agent_fn(question, thread_id=thread_id)
    except Exception as e:
        trace["error"] = f"{type(e).__name__}: {e}"
        trace["error_traceback"] = traceback.format_exc()
        trace["total_latency_ms"] = (time.time() - t_start) * 1000
        return trace

    trace["total_latency_ms"] = (time.time() - t_start) * 1000

    # Router meta
    if isinstance(result, dict) and "_router" in result:
        trace["router"] = result["_router"]

    # Tool calls + final answer
    messages = result.get("messages", []) if isinstance(result, dict) else []
    calls, completeness = _extract_tool_calls(messages)
    trace["tool_calls"] = calls
    trace["completeness"] = completeness
    trace["final_answer"] = _extract_final_answer(messages)

    return trace


# ── Main ─────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Run eval questions through agent, log full traces to JSONL")
    ap.add_argument("--input", required=True, type=Path, help="Eval CSV path")
    ap.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N rows (for testing)")
    ap.add_argument("--offset", type=int, default=0, help="Skip first N rows")
    ap.add_argument("--append", action="store_true", help="Append to output file (default: overwrite)")
    args = ap.parse_args()

    # Lazy import để .env load trước
    import dotenv
    dotenv.load_dotenv()

    # Patch langchain compat (None usage metadata)
    from app.core import compat  # noqa: F401

    from app.agent.agent import run_agent_routed

    # Đọc CSV (utf-8-sig strip BOM)
    rows = list(csv.DictReader(args.input.open(encoding="utf-8-sig")))
    if args.offset:
        rows = rows[args.offset:]
    if args.limit:
        rows = rows[:args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    print(f"Running {len(rows)} questions → {args.output} (mode={mode})")
    print(f"{'─' * 60}")

    n_ok = 0
    n_err = 0
    n_mismatch = 0
    t_total = time.time()

    with args.output.open(mode, encoding="utf-8") as f:
        for i, row in enumerate(rows):
            qid = row.get("id", f"row{i}")
            question = row.get("question", "")[:60]
            print(f"[{i + 1}/{len(rows)}] Q{qid}: {question}...", end=" ", flush=True)

            trace = trace_one(row, run_agent_routed)

            # Ghi ngay (crash-resilient)
            f.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
            f.flush()

            if trace["error"]:
                n_err += 1
                print(f"❌ ERROR ({trace['total_latency_ms']:.0f}ms): {trace['error'][:80]}")
            else:
                n_ok += 1
                router = trace.get("router") or {}
                n_tools = len(trace["tool_calls"])
                mismatch_flag = ""
                if trace["completeness"].get("mismatch"):
                    n_mismatch += 1
                    mismatch_flag = " ⚠MISMATCH"
                print(
                    f"✓ {router.get('intent','?')}/{router.get('scope','?')}"
                    f" conf={router.get('confidence',0):.2f} "
                    f"tools={n_tools} {trace['total_latency_ms']:.0f}ms{mismatch_flag}"
                )

    elapsed = time.time() - t_total
    print(f"{'─' * 60}")
    print(f"Done: {n_ok} ok, {n_err} errors, {n_mismatch} tool-logging mismatches in {elapsed:.1f}s")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
