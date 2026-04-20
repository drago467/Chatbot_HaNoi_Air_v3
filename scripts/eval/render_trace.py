"""Render JSONL trace thành text block dễ đọc cho manual review.

Format mỗi câu:
─── Q1 [easy, city, current_weather] ──────────────────────────────
Question: "Bây giờ thời tiết ở Hà Nội như thế nào?"
Expected: current_weather / city / Hà Nội
Router  : ✓ current_weather/city conf=0.92 (420ms)
          rewritten_query: null
          focused_tools: [resolve_location, get_current_weather_city]
Tools:
  1. resolve_location(query="Hà Nội")
     → {"ward_id": null, "district_id": null, "city": true}
     [35ms]
  2. get_current_weather_city()
     → temp=28.5, humidity=78, weather_main=Clouds ...
     [120ms]
Answer: "Hiện tại Hà Nội có nhiệt độ 28.5°C, độ ẩm 78%..."
Total: 2340ms

Usage:
    python -m scripts.eval.render_trace --trace <file.jsonl>
    python -m scripts.eval.render_trace --trace <file.jsonl> --start 1 --end 20
    python -m scripts.eval.render_trace --trace <file.jsonl> --ids 1,3,5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ── Formatting helpers ────────────────────────────────────────────────


def _truncate(s: str, n: int = 200) -> str:
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + f"...(+{len(s)-n} chars)"


def _format_args(args) -> str:
    """Format tool args as 'key=value, key=value'."""
    if not args:
        return ""
    if isinstance(args, dict):
        parts = []
        for k, v in args.items():
            v_str = repr(v) if isinstance(v, str) else str(v)
            parts.append(f"{k}={v_str}")
        return ", ".join(parts)
    return str(args)


def _router_verdict(router: dict | None, expected_intent: str, expected_scope: str) -> str:
    """Icon ✓/✗ dựa trên match giữa router vs expected."""
    if not router:
        return "? (no router meta)"
    r_intent = (router.get("intent") or "").strip()
    r_scope = (router.get("scope") or "").strip()
    e_intent = (expected_intent or "").strip()
    e_scope = (expected_scope or "").strip()
    intent_ok = (r_intent == e_intent) if e_intent else True
    scope_ok = (r_scope == e_scope) if e_scope else True
    icons = []
    if not intent_ok:
        icons.append(f"✗ intent mismatch (got {r_intent!r})")
    if not scope_ok:
        icons.append(f"✗ scope mismatch (got {r_scope!r})")
    if not icons:
        return "✓"
    return "  ".join(icons)


def render_trace(t: dict, width: int = 70) -> str:
    """Render 1 trace thành multi-line string."""
    qid = t.get("id")
    exp = t.get("expected") or {}
    diff = exp.get("difficulty", "?")
    scope = exp.get("scope", "?")
    intent = exp.get("intent", "?")

    lines = []
    lines.append(f"─── Q{qid} [{diff}, {scope}, {intent}] {'─' * max(0, width - 20 - len(str(qid)) - len(diff) - len(scope) - len(intent))}")
    lines.append(f'Question: "{t.get("question", "")}"')

    # Expected block
    exp_loc = exp.get("location", "")
    exp_time = exp.get("time", "")
    exp_param = exp.get("weather_param", "")
    exp_line = f"Expected: {intent} / {scope} / {exp_loc}"
    if exp_time:
        exp_line += f" / time={exp_time}"
    if exp_param and exp_param != "general":
        exp_line += f" / param={exp_param}"
    lines.append(exp_line)
    notes = exp.get("notes", "")
    if notes:
        lines.append(f"  Note: {notes}")

    # Router block
    router = t.get("router")
    if router:
        verdict = _router_verdict(router, intent, scope)
        path = router.get("path", "?")
        conf = router.get("confidence", 0) or 0
        lat = router.get("latency_ms", 0) or 0
        r_intent = router.get("intent", "?")
        r_scope = router.get("scope", "?")
        lines.append(
            f"Router  : {verdict} [{path}] {r_intent}/{r_scope} "
            f"conf={conf:.2f} ({lat:.0f}ms)"
        )
        rewritten = router.get("rewritten_query")
        if rewritten:
            lines.append(f"          rewritten: {rewritten!r}")
        fallback_reason = router.get("fallback_reason")
        if fallback_reason:
            lines.append(f"          fallback_reason: {fallback_reason}")
        focused = router.get("focused_tools") or []
        if focused:
            lines.append(f"          focused_tools: {focused}")
    else:
        lines.append("Router  : (no metadata)")

    # Tool calls
    calls = t.get("tool_calls") or []
    if calls:
        lines.append(f"Tools ({len(calls)}):")
        for c in calls:
            args_str = _format_args(c.get("args"))
            lines.append(f"  {c.get('seq')}. {c.get('name')}({args_str})")
            output = _truncate(c.get("output", ""), 300)
            if output:
                lines.append(f"     → {output}")
    else:
        lines.append("Tools   : (không gọi tool)")

    # Completeness warning
    comp = t.get("completeness") or {}
    if comp.get("mismatch"):
        lines.append(
            f"⚠ TOOL LOG MISMATCH: AI declared {comp.get('tool_count_ai')} "
            f"tool_calls nhưng chỉ {comp.get('tool_count_msgs')} ToolMessage"
        )

    # Final answer
    answer = t.get("final_answer", "")
    if answer:
        lines.append(f"Answer  : {_truncate(answer, 600)}")
    else:
        lines.append("Answer  : (trống)")

    # Metadata
    lat_total = t.get("total_latency_ms", 0)
    lines.append(f"Total   : {lat_total:.0f}ms")

    err = t.get("error")
    if err:
        lines.append(f"❌ ERROR: {err}")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, type=Path, help="JSONL trace file")
    ap.add_argument("--start", type=int, default=None, help="Chỉ render từ id >= start")
    ap.add_argument("--end", type=int, default=None, help="Chỉ render đến id <= end")
    ap.add_argument("--ids", type=str, default=None, help="Comma-separated ids, e.g. '1,3,5'")
    ap.add_argument("--width", type=int, default=70, help="Render width")
    args = ap.parse_args()

    target_ids: set | None = None
    if args.ids:
        target_ids = {int(x.strip()) for x in args.ids.split(",") if x.strip().isdigit()}

    with args.trace.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            tid = t.get("id")
            try:
                tid_int = int(tid)
            except (ValueError, TypeError):
                tid_int = None

            # Filter
            if target_ids is not None:
                if tid_int not in target_ids:
                    continue
            else:
                if args.start is not None and (tid_int is None or tid_int < args.start):
                    continue
                if args.end is not None and (tid_int is None or tid_int > args.end):
                    continue

            print(render_trace(t, width=args.width))
            print()  # blank line between traces


if __name__ == "__main__":
    main()
