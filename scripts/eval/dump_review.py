"""Dump full trace chain (Q → router → tool name + args + FULL output → full answer)
thành markdown dễ đọc để review thủ công.

Khác với render_trace.py (truncate output 300 char để in terminal), script này:
- KHÔNG truncate tool output
- KHÔNG truncate final_answer
- Format markdown với code block cho tool output JSON
- Group theo cluster (critical / high / med) dựa trên audit_auto_v1.csv

Usage:
    python -m scripts.eval.dump_review \\
        --trace data/evaluation/traces/full_run_v1.jsonl \\
        --audit data/evaluation/audit_auto_v1.csv \\
        --output data/evaluation/trace_review_detail.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_traces(path: Path) -> dict:
    traces = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line)
                traces[t["id"]] = t
    return traces


def load_audit(path: Path) -> dict:
    audits = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                audits[int(row["id"])] = row
            except (ValueError, KeyError):
                pass
    return audits


def format_output(output_str: str) -> str:
    """Format tool output — try parse JSON for pretty print, fallback str."""
    if not output_str:
        return "(empty)"
    s = output_str.strip()
    # Try parse as Python dict string
    try:
        # Quick hack: try eval if looks like dict repr
        if s.startswith("{") and s.endswith("}"):
            obj = json.loads(s)
            return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: return as-is (already a string representation)
    return s


def render_full_trace(t: dict, audit: dict | None = None) -> str:
    """Render 1 trace with FULL detail, no truncation."""
    exp = t.get("expected") or {}
    router = t.get("router") or {}
    qid = t.get("id")

    lines = []
    lines.append(f"## Q{qid} — {exp.get('intent','?')}/{exp.get('scope','?')} [{exp.get('difficulty','?')}]")
    lines.append("")
    lines.append(f"**Question**: {t.get('question','')}")
    lines.append("")

    # Expected
    lines.append(f"**Expected**: intent=`{exp.get('intent','')}` scope=`{exp.get('scope','')}` location=`{exp.get('location','')}` time=`{exp.get('time','')}` param=`{exp.get('weather_param','')}`")
    note = exp.get("notes", "")
    if note:
        lines.append(f"**Note (from CSV)**: {note}")
    lines.append("")

    # Audit verdict icons
    if audit:
        intent_ok = audit.get("intent_match") == "True"
        scope_ok = audit.get("scope_match") == "True"
        tool_ok = audit.get("tool_recall") == "True"
        loc_ok = audit.get("location_in_args") == "True"
        all_pass = audit.get("all_pass") == "True"
        def icon(v): return "✓" if v else "✗"
        lines.append(f"**Auto-check**: intent={icon(intent_ok)} scope={icon(scope_ok)} tool_recall={icon(tool_ok)} location={icon(loc_ok)} → **{'PASS' if all_pass else 'FAIL'}**")
        lines.append("")

    # Router
    lines.append(f"**Router**: path=`{router.get('path','?')}` intent=`{router.get('intent','?')}` scope=`{router.get('scope','?')}` confidence={router.get('confidence',0):.3f} latency={router.get('latency_ms',0):.0f}ms")
    if router.get("fallback_reason"):
        lines.append(f"  - fallback_reason: `{router.get('fallback_reason')}`")
    if router.get("rewritten_query"):
        lines.append(f"  - rewritten_query: `{router.get('rewritten_query')}`")
    if router.get("focused_tools"):
        lines.append(f"  - focused_tools: `{router.get('focused_tools')}`")
    lines.append("")

    # Tool calls — FULL output
    calls = t.get("tool_calls") or []
    if calls:
        lines.append(f"**Tool calls** ({len(calls)}):")
        for c in calls:
            lines.append("")
            lines.append(f"### Tool {c.get('seq','?')}: `{c.get('name','?')}`")
            lines.append("")
            lines.append("**Args**:")
            lines.append("```json")
            lines.append(json.dumps(c.get("args", {}), ensure_ascii=False, indent=2, default=str))
            lines.append("```")
            lines.append("")
            lines.append("**Output** (FULL):")
            lines.append("```json")
            lines.append(format_output(c.get("output", "")))
            lines.append("```")
    else:
        lines.append("**Tool calls**: (none)")
    lines.append("")

    # Final answer — FULL
    lines.append("**Final answer** (FULL):")
    lines.append("")
    answer = t.get("final_answer", "")
    if answer:
        # Quote each line with > for markdown blockquote
        for ln in answer.split("\n"):
            lines.append(f"> {ln}")
    else:
        lines.append("> (empty)")
    lines.append("")

    # Meta
    lines.append(f"**Latency**: {t.get('total_latency_ms',0):.0f}ms")
    err = t.get("error")
    if err:
        lines.append(f"**❌ ERROR**: {err}")

    # Space for manual review
    lines.append("")
    lines.append("**My review** (faithfulness / completeness / correctness / actionability):")
    lines.append("")
    lines.append("- [ ] TODO — fill in manual evaluation")
    lines.append("")
    lines.append("---")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--audit", type=Path, help="audit_auto_v1.csv (optional, for verdict icons)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--ids", type=str, help="Comma-separated ids, e.g. '48,56,144,186'")
    ap.add_argument("--failed-only", action="store_true", help="Only dump failed auto-check traces")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()

    traces = load_traces(args.trace)
    audits = load_audit(args.audit) if args.audit else {}

    # Determine which IDs to dump
    if args.ids:
        target_ids = {int(x.strip()) for x in args.ids.split(",") if x.strip().isdigit()}
    else:
        target_ids = set(traces.keys())

    if args.failed_only and audits:
        target_ids &= {qid for qid, a in audits.items() if a.get("all_pass") != "True"}

    if args.start is not None:
        target_ids &= {qid for qid in target_ids if isinstance(qid, int) and qid >= args.start}
    if args.end is not None:
        target_ids &= {qid for qid in target_ids if isinstance(qid, int) and qid <= args.end}

    sorted_ids = sorted(target_ids, key=lambda x: int(x) if isinstance(x, int) or str(x).isdigit() else 999999)

    # Write header
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# Trace review — full chain dump")
    lines.append(f"")
    lines.append(f"Trace file: `{args.trace.name}` | Total dumped: {len(sorted_ids)}/{len(traces)}")
    if args.failed_only:
        lines.append(f"Filter: **failed auto-check only**")
    lines.append("")
    lines.append("Format: mỗi câu có full tool output + full answer để bạn so sánh số liệu faithfulness.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for qid in sorted_ids:
        t = traces.get(qid)
        if not t:
            continue
        audit = audits.get(qid)
        lines.append(render_full_trace(t, audit))
        lines.append("")

    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output} — {len(sorted_ids)} traces")


if __name__ == "__main__":
    main()
