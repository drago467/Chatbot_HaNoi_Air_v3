"""Verify whether the Qwen3-14B API provider accepts invoke (non-streaming) with
thinking enabled. Probes both explicit `enable_thinking=True` and provider default.

Run once in R11a day 1 to decide:
- If invoke+thinking works → proceed with simple unified thinking path.
- If invoke+thinking rejected → need stream-collect wrapper fallback.

Exit code 0 = invoke+thinking works, 1 = need wrapper.
"""

from __future__ import annotations

import os
import sys
import traceback

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

API_BASE = os.getenv("AGENT_API_BASE")
API_KEY = os.getenv("AGENT_API_KEY")
MODEL = os.getenv("AGENT_MODEL", "qwen3-14b")

if not API_BASE or not API_KEY:
    print("ERROR: AGENT_API_BASE / AGENT_API_KEY missing in .env")
    sys.exit(2)

print(f"Provider: {API_BASE}")
print(f"Model:    {MODEL}")
print("=" * 60)

# Hard prompt that should trigger thinking when enabled
HARD_PROMPT = [
    SystemMessage(content="Bạn là trợ lý. Hôm nay là Thứ Ba 21/04/2026."),
    HumanMessage(content="Cuối tuần này rơi vào ngày mấy tháng mấy? Trả lời ngắn gọn."),
]

SIMPLE_PROMPT = [
    SystemMessage(content="You are a helpful assistant. Reply very briefly."),
    HumanMessage(content="Hello in Vietnamese, in one short sentence."),
]


def _probe_invoke(label: str, extra_body: dict | None, temperature: float = 0, prompt=SIMPLE_PROMPT):
    print(f"\n[{label}]")
    try:
        kwargs = {"model": MODEL, "temperature": temperature, "base_url": API_BASE, "api_key": API_KEY}
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        llm = ChatOpenAI(**kwargs)
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        has_think = "<think>" in content or "</think>" in content
        print(f"  ✓ OK — len={len(content)}, <think>_in_content={has_think}")
        print(f"  preview: {content[:200]!r}")
        # Check additional_kwargs for reasoning_content (some providers put it there)
        add_kwargs = getattr(resp, "additional_kwargs", {}) or {}
        if "reasoning_content" in add_kwargs:
            rc = add_kwargs["reasoning_content"]
            print(f"  reasoning_content_len={len(rc) if rc else 0}")
        return True, None
    except Exception as e:
        print(f"  ✗ FAIL — {type(e).__name__}: {e}")
        return False, e


def _probe_stream(label: str, extra_body: dict | None, temperature: float, prompt=SIMPLE_PROMPT):
    print(f"\n[{label}]")
    try:
        kwargs = {"model": MODEL, "temperature": temperature, "base_url": API_BASE, "api_key": API_KEY}
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        llm = ChatOpenAI(**kwargs)
        chunks = []
        for chunk in llm.stream(prompt):
            if hasattr(chunk, "content") and chunk.content:
                chunks.append(chunk.content)
        full = "".join(chunks)
        print(f"  ✓ OK — stream chunks joined, len={len(full)}")
        print(f"  preview: {full[:200]!r}")
        return True, None
    except Exception as e:
        print(f"  ✗ FAIL — {type(e).__name__}: {e}")
        return False, e


results = {}

# === Invoke probes ===
results["invoke_default"] = _probe_invoke("probe_1_invoke_default", None)
results["invoke_thinking_true_simple"] = _probe_invoke("probe_2_invoke_thinking_true_simple", {"enable_thinking": True})
results["invoke_thinking_true_hard"] = _probe_invoke(
    "probe_2b_invoke_thinking_true_hard_prompt", {"enable_thinking": True}, prompt=HARD_PROMPT
)
results["invoke_thinking_false"] = _probe_invoke("probe_3_invoke_thinking_false", {"enable_thinking": False})

# === Stream probes ===
results["stream_thinking_true_temp0"] = _probe_stream("probe_4a_stream_thinking_true_temp0", {"enable_thinking": True}, temperature=0)
results["stream_thinking_true_temp06"] = _probe_stream("probe_4b_stream_thinking_true_temp0.6", {"enable_thinking": True}, temperature=0.6)
results["stream_thinking_false_temp0"] = _probe_stream("probe_4c_stream_thinking_false_temp0", {"enable_thinking": False}, temperature=0)
results["stream_default_temp0"] = _probe_stream("probe_4d_stream_default_temp0", None, temperature=0)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name, (ok, err) in results.items():
    mark = "✓" if ok else "✗"
    print(f"  {mark} {name}")

invoke_ok = results["invoke_thinking_true_simple"][0] and results["invoke_thinking_true_hard"][0]
stream_ok = results["stream_thinking_true_temp0"][0] or results["stream_thinking_true_temp06"][0]

print()
print("DECISIONS:")
print(f"  invoke+thinking works?  {invoke_ok}")
print(f"  stream+thinking works?  {stream_ok}")
if invoke_ok and stream_ok:
    print("  → Both paths OK, unified thinking possible.")
    sys.exit(0)
elif invoke_ok and not stream_ok:
    print("  → Invoke OK, stream blocked. Keep production stream non-thinking,")
    print("     eval invoke thinking ON.")
    sys.exit(0)
elif not invoke_ok and stream_ok:
    print("  → Invoke blocked, need stream-collect wrapper for eval.")
    sys.exit(1)
else:
    print("  → Both paths fail. Investigate manually.")
    sys.exit(2)
