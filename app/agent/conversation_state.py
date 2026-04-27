"""ConversationState — thread-safe in-memory store for multi-turn router context.

Schema mirrors training data (multitask_train_v6_clean.jsonl):
    {"location": str, "intent": str, "turn": int}

Anything beyond these three fields was never seen during fine-tuning and biases
the router's rewrite output (the model latches onto stray strings like
"Hà Nội" inside richer context dicts). Keep this surface minimal.

TTL = 30 minutes (configurable via CONVERSATION_TTL_SECONDS env var).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field


_TTL_SECONDS = int(os.getenv("CONVERSATION_TTL_SECONDS", "1800"))

# Tools whose location_hint argument should seed last location when no
# resolve_location call was made.
_WEATHER_TOOLS = frozenset({
    "get_current_weather",
    "get_daily_forecast",
    "get_hourly_forecast",
    "get_rain_timeline",
    "get_weather_history",
    "get_daily_summary",
    "get_weather_period",
})


@dataclass
class ConversationState:
    """Per-thread router context. Three fields, matched to training schema."""
    location: str | None = None
    intent: str | None = None
    turn_count: int = 0
    updated_at: float = field(default_factory=time.time)

    def to_context_json(self) -> dict:
        """Serialize for [CONTEXT: ...] injection into the router prompt."""
        return {
            "location": self.location,
            "intent": self.intent,
            "turn": self.turn_count,
        }


class ConversationStateStore:
    """Thread-safe in-memory store with TTL. Not persisted across restarts."""

    def __init__(self, ttl_seconds: int = _TTL_SECONDS):
        self._store: dict[str, ConversationState] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def get(self, thread_id: str) -> ConversationState | None:
        with self._lock:
            state = self._store.get(thread_id)
            if state is None:
                return None
            if time.time() - state.updated_at > self._ttl:
                del self._store[thread_id]
                return None
            return state

    def update(
        self, thread_id: str, tool_call_logs: list[dict], intent: str
    ) -> ConversationState:
        """Advance state for one completed turn.

        tool_call_logs: list of {tool_name, tool_input(JSON str or dict), tool_output(JSON str)}.
        Both streaming and non-streaming paths produce this shape (the latter
        via messages_to_tool_call_logs).
        """
        new_loc = _extract_location(tool_call_logs)
        with self._lock:
            state = self._store.get(thread_id) or ConversationState()
            if new_loc is not None:
                state.location = new_loc
            state.intent = intent
            state.turn_count += 1
            state.updated_at = time.time()
            self._store[thread_id] = state
            return state

    def evict_expired(self) -> int:
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._store.items() if now - v.updated_at > self._ttl]
            for k in expired:
                del self._store[k]
        return len(expired)


def _parse_json(value) -> dict | None:
    if isinstance(value, dict):
        return value
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _extract_location(tool_call_logs: list[dict]) -> str | None:
    """Pick a location name from this turn's tool calls.

    Priority: resolve_location output (canonical) > location_hint arg (fallback).
    """
    resolve_name: str | None = None
    hint_name: str | None = None

    for tc in tool_call_logs:
        name = tc.get("tool_name", "")

        if name == "resolve_location":
            out = _parse_json(tc.get("tool_output", ""))
            if out and out.get("status") == "ok":
                data = out.get("data") or {}
                resolve_name = (
                    data.get("ward_name")
                    or data.get("district_name")
                    or "Hà Nội"
                )
        elif name in _WEATHER_TOOLS and hint_name is None:
            args = _parse_json(tc.get("tool_input", "")) or {}
            lh = args.get("location_hint")
            if lh:
                hint_name = str(lh)

    return resolve_name or hint_name


def messages_to_tool_call_logs(messages) -> list[dict]:
    """Convert a LangGraph result['messages'] list into the tool_call_logs shape.

    Pairs AIMessage.tool_calls with their matching ToolMessage outputs by
    tool_call_id.
    """
    pending: dict[str, dict] = {}
    logs: list[dict] = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            if isinstance(tc, dict):
                tc_id = tc.get("id", "") or ""
                tc_name = tc.get("name", "") or ""
                tc_args = tc.get("args", {}) or {}
            else:
                tc_id = getattr(tc, "id", "") or ""
                tc_name = getattr(tc, "name", "") or ""
                tc_args = getattr(tc, "args", {}) or {}
            args_str = (
                json.dumps(tc_args, ensure_ascii=False)
                if isinstance(tc_args, dict)
                else str(tc_args)
            )
            if tc_id:
                pending[tc_id] = {"tool_name": tc_name, "tool_input": args_str}

        if getattr(msg, "type", None) == "tool":
            tc_id = getattr(msg, "tool_call_id", "") or ""
            entry = pending.pop(tc_id, None)
            logs.append({
                "tool_name": entry["tool_name"] if entry else getattr(msg, "name", "unknown"),
                "tool_input": entry["tool_input"] if entry else "",
                "tool_output": str(getattr(msg, "content", "") or ""),
                "success": True,
            })
    return logs


_store: ConversationStateStore | None = None
_store_lock = threading.Lock()


def get_conversation_store() -> ConversationStateStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ConversationStateStore()
    return _store
