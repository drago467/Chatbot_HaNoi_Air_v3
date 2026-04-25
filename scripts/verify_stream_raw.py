"""Probe streaming with raw openai client to isolate langchain_openai 0.2.0 parsing bug.

Also tests LangGraph agent.stream() path (actual production flow) to see if it masks
the bug or reproduces it.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_BASE = os.getenv("AGENT_API_BASE")
API_KEY = os.getenv("AGENT_API_KEY")
MODEL = os.getenv("AGENT_MODEL", "qwen3-14b")

client = OpenAI(base_url=API_BASE, api_key=API_KEY)

print(f"Provider: {API_BASE}, Model: {MODEL}")
print("=" * 60)

# === Probe A: raw openai client, stream, enable_thinking=False ===
print("\n[A] raw openai.stream, enable_thinking=False, temp=0")
try:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello in Vietnamese"},
        ],
        temperature=0,
        stream=True,
        extra_body={"enable_thinking": False},
    )
    chunks_content = []
    chunk_count = 0
    for chunk in resp:
        chunk_count += 1
        if chunk.choices and chunk.choices[0].delta.content:
            chunks_content.append(chunk.choices[0].delta.content)
    print(f"  ✓ OK — total chunks={chunk_count}, content_len={len(''.join(chunks_content))}")
    print(f"  preview: {''.join(chunks_content)[:120]!r}")
except Exception as e:
    print(f"  ✗ FAIL — {type(e).__name__}: {e}")

# === Probe B: raw openai client, stream, enable_thinking=True ===
print("\n[B] raw openai.stream, enable_thinking=True, temp=0.6")
try:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Bạn là trợ lý. Hôm nay là Thứ Ba 21/04/2026."},
            {"role": "user", "content": "Cuối tuần này rơi vào ngày mấy?"},
        ],
        temperature=0.6,
        stream=True,
        extra_body={"enable_thinking": True},
    )
    chunks_content = []
    chunks_reasoning = []
    chunk_count = 0
    has_thinking = False
    for chunk in resp:
        chunk_count += 1
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                chunks_content.append(delta.content)
            # Some providers use "reasoning_content" attribute for thinking
            reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if reasoning:
                has_thinking = True
                chunks_reasoning.append(reasoning)
    full_content = "".join(chunks_content)
    full_reasoning = "".join(chunks_reasoning)
    print(f"  ✓ OK — chunks={chunk_count}, content_len={len(full_content)}, reasoning_len={len(full_reasoning)}, has_reasoning={has_thinking}")
    print(f"  content preview: {full_content[:200]!r}")
    if full_reasoning:
        print(f"  reasoning preview: {full_reasoning[:200]!r}")
except Exception as e:
    print(f"  ✗ FAIL — {type(e).__name__}: {e}")

# === Probe C: LangGraph agent.stream() actual production path ===
print("\n[C] LangGraph agent.stream (production path) with existing agent setup")
try:
    from app.agent.agent import get_agent
    agent = get_agent()
    msg = "Xin chào!"
    chunks = []
    errors = []
    for event in agent.stream({"messages": [{"role": "user", "content": msg}]}, {"configurable": {"thread_id": "probe"}}, stream_mode="messages"):
        try:
            if event and len(event) >= 2:
                msg_chunk, metadata = event
                if hasattr(msg_chunk, "content") and msg_chunk.content:
                    if isinstance(msg_chunk.content, str):
                        chunks.append(msg_chunk.content)
        except Exception as inner:
            errors.append(str(inner))
    full = "".join(chunks)
    print(f"  ✓ OK — chunks={len(chunks)}, full_len={len(full)}, inner_errors={len(errors)}")
    print(f"  preview: {full[:200]!r}")
except Exception as e:
    import traceback
    print(f"  ✗ FAIL — {type(e).__name__}: {e}")
    traceback.print_exc(limit=6)

print("\n" + "=" * 60)
