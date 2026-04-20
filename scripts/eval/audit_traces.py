"""Programmatic audit của trace JSONL — check 4/5 checkpoints tự động.

Checkpoints evaluate:
1. router_intent_match: router intent == expected?
2. router_scope_match: router scope == expected?
3. tool_recall_ok: ít nhất 1 tool gọi ∈ INTENT_TO_TOOLS[expected.intent][expected.scope]?
4. location_in_args: tên location xuất hiện trong tool_calls[].args?
5. (answer quality) — để AI review thủ công, không auto-check.

Output:
- `audit_auto_v1.csv` — 1 row per question + all flags
- `audit_summary_v1.md` — aggregate stats + top failure clusters

Usage:
    python -m scripts.eval.audit_traces --trace data/evaluation/traces/full_run_v1.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_intent_to_tools():
    """Load INTENT_TO_TOOLS mapping từ experiments.evaluation.tool_accuracy."""
    from experiments.evaluation.tool_accuracy import INTENT_TO_TOOLS, _get_expected_tools
    return INTENT_TO_TOOLS, _get_expected_tools


def extract_location_parts(location_name: str) -> list[str]:
    """Parse location_name ra các keyword có thể xuất hiện trong tool args.

    "Phường Cầu Giấy (Cầu Giấy)" → ["Phường Cầu Giấy", "Cầu Giấy"]
    "Cầu Giấy" → ["Cầu Giấy"]
    "Cầu Giấy, Hoàn Kiếm" → ["Cầu Giấy", "Hoàn Kiếm"]
    "Hà Nội" → ["Hà Nội"]
    "" → []
    """
    if not location_name:
        return []
    # Split multi-location (comma-separated)
    parts = []
    for p in location_name.split(","):
        p = p.strip()
        if not p:
            continue
        # Extract from "Ward (District)"
        if "(" in p and ")" in p:
            main = p[:p.index("(")].strip()
            district = p[p.index("(")+1:p.index(")")].strip()
            if main:
                parts.append(main)
            if district:
                parts.append(district)
        else:
            parts.append(p)
    return parts


def location_in_tool_args(trace: dict) -> bool:
    """Check if any expected location keyword xuất hiện trong tool args.

    Trả True nếu:
    - Không có expected location (city=Hà Nội mặc định), hoặc
    - Có ít nhất 1 tool call có args chứa 1 keyword location.
    """
    loc_name = trace.get("expected", {}).get("location", "")
    parts = extract_location_parts(loc_name)
    if not parts:
        return True  # không có location → skip check

    for call in trace.get("tool_calls", []):
        args_str = json.dumps(call.get("args", {}), ensure_ascii=False)
        for part in parts:
            if part.lower() in args_str.lower():
                return True
    return False


def audit_trace(trace: dict, get_expected_tools) -> dict:
    """Evaluate 1 trace, return dict các flag."""
    exp = trace.get("expected", {})
    router = trace.get("router") or {}

    e_intent = (exp.get("intent") or "").strip()
    e_scope = (exp.get("scope") or "").strip()
    r_intent = (router.get("intent") or "").strip()
    r_scope = (router.get("scope") or "").strip()
    r_path = router.get("path") or ""
    r_conf = router.get("confidence") or 0.0
    r_fallback_reason = router.get("fallback_reason") or ""

    # Checkpoint 1-2
    intent_match = (r_intent == e_intent) if e_intent else None
    # POI scope: dispatch.py resolve POI → district level. Router predict "district"
    # cho POI là ĐÚNG design (xem [app/agent/dispatch.py]). Skip check khi e_scope=poi.
    if e_scope == "poi":
        scope_match = True if r_scope in ("district", "city") else False
    else:
        scope_match = (r_scope == e_scope) if e_scope else None

    # Checkpoint 3: tool recall
    tools_called = [c.get("name") for c in trace.get("tool_calls", []) if c.get("name")]
    expected_tools = get_expected_tools(e_intent, e_scope)
    tool_recall = bool(set(tools_called) & set(expected_tools)) if expected_tools else None

    # Checkpoint 4: location in args
    loc_ok = location_in_tool_args(trace)

    # Compute PASS (all of 1-4, if not None, must be True)
    all_checks = [intent_match, scope_match, tool_recall, loc_ok]
    checks_actionable = [c for c in all_checks if c is not None]
    all_pass = all(checks_actionable) if checks_actionable else True

    # Special: smalltalk + 0 tools = OK
    if e_intent == "smalltalk_weather" and len(tools_called) == 0:
        tool_recall = True  # override
        all_pass = intent_match if intent_match is not None else True

    return {
        "id": trace.get("id"),
        "question": trace.get("question", "")[:80],
        "expected_intent": e_intent,
        "router_intent": r_intent,
        "intent_match": intent_match,
        "expected_scope": e_scope,
        "router_scope": r_scope,
        "scope_match": scope_match,
        "router_path": r_path,
        "router_conf": r_conf,
        "fallback_reason": r_fallback_reason,
        "tools_called": ",".join(tools_called),
        "expected_tools": ",".join(expected_tools),
        "tool_recall": tool_recall,
        "location_in_args": loc_ok,
        "all_pass": all_pass,
        "difficulty": exp.get("difficulty", ""),
        "notes": exp.get("notes", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--out-csv", type=Path,
                    default=Path("data/evaluation/audit_auto_v1.csv"))
    ap.add_argument("--out-md", type=Path,
                    default=Path("data/evaluation/audit_summary_v1.md"))
    args = ap.parse_args()

    _, get_expected_tools = load_intent_to_tools()

    # Load traces
    traces = []
    with args.trace.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    print(f"Loaded {len(traces)} traces")

    # Audit each
    results = [audit_trace(t, get_expected_tools) for t in traces]

    # Write CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {args.out_csv}")

    # Aggregate stats
    n = len(results)
    n_pass = sum(1 for r in results if r["all_pass"])
    n_intent_fail = sum(1 for r in results if r["intent_match"] is False)
    n_scope_fail = sum(1 for r in results if r["scope_match"] is False)
    n_tool_fail = sum(1 for r in results if r["tool_recall"] is False)
    n_loc_fail = sum(1 for r in results if not r["location_in_args"])
    n_fallback = sum(1 for r in results if r["router_path"] == "fallback")
    n_no_tool = sum(1 for r in results if not r["tools_called"])

    # Confusion pairs
    confusion = Counter()
    for r in results:
        if r["intent_match"] is False:
            confusion[(r["expected_intent"], r["router_intent"])] += 1

    # Scope confusion
    scope_conf = Counter()
    for r in results:
        if r["scope_match"] is False:
            scope_conf[(r["expected_scope"], r["router_scope"])] += 1

    # Failure patterns by intent
    fails_by_intent = defaultdict(list)
    for r in results:
        if not r["all_pass"]:
            fails_by_intent[r["expected_intent"]].append(r["id"])

    # Write markdown summary
    md_lines = []
    md_lines.append("# AI Audit — full_run_v1 — auto analysis\n")
    md_lines.append(f"Trace file: `{args.trace.name}` — {n} questions\n")
    md_lines.append("## Executive summary\n")
    md_lines.append(f"- **PASS all auto-checks**: {n_pass}/{n} ({100*n_pass/n:.1f}%)")
    md_lines.append(f"- Router intent mismatch: {n_intent_fail}/{n} ({100*n_intent_fail/n:.1f}%)")
    md_lines.append(f"- Router scope mismatch: {n_scope_fail}/{n} ({100*n_scope_fail/n:.1f}%)")
    md_lines.append(f"- Tool recall fail (không gọi tool expected): {n_tool_fail}/{n}")
    md_lines.append(f"- Location không có trong tool args: {n_loc_fail}/{n}")
    md_lines.append(f"- Router fallback: {n_fallback}/{n} ({100*n_fallback/n:.1f}%)")
    md_lines.append(f"- No tool call: {n_no_tool}/{n}\n")

    md_lines.append("## Top intent confusion (expected → router)\n")
    md_lines.append("| Expected | Router predicted | Count |")
    md_lines.append("|---|---|---|")
    for (e, r), c in confusion.most_common(15):
        md_lines.append(f"| {e} | {r} | {c} |")
    md_lines.append("")

    md_lines.append("## Scope confusion\n")
    md_lines.append("| Expected | Router predicted | Count |")
    md_lines.append("|---|---|---|")
    for (e, r), c in scope_conf.most_common(10):
        md_lines.append(f"| {e} | {r} | {c} |")
    md_lines.append("")

    md_lines.append("## Failures by intent\n")
    md_lines.append("| Intent | Fail count | Fail ids |")
    md_lines.append("|---|---|---|")
    for intent, ids in sorted(fails_by_intent.items(), key=lambda x: -len(x[1])):
        md_lines.append(f"| {intent} | {len(ids)} | {ids[:20]}{'...' if len(ids)>20 else ''} |")
    md_lines.append("")

    md_lines.append("## Auto-check fail details\n")
    md_lines.append("Các câu không pass auto-check (cần AI review answer quality):\n")
    md_lines.append("| ID | Q (truncated) | Expected intent/scope | Router | Tool recall | Loc in args |")
    md_lines.append("|---|---|---|---|---|---|")
    fail_rows = [r for r in results if not r["all_pass"]]
    for r in fail_rows:
        ir = r["intent_match"]
        sr = r["scope_match"]
        tr = r["tool_recall"]
        lr = r["location_in_args"]
        icon = lambda v: "✓" if v is True else ("✗" if v is False else "—")
        md_lines.append(
            f"| Q{r['id']} | {r['question'][:50]} | {r['expected_intent']}/{r['expected_scope']} | "
            f"{icon(ir)}{r['router_intent']}/{icon(sr)}{r['router_scope']} "
            f"| {icon(tr)} | {icon(lr)} |"
        )

    args.out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {args.out_md}")

    # Console summary
    print("\n" + "─" * 60)
    print(f"PASS: {n_pass}/{n} ({100*n_pass/n:.1f}%)")
    print(f"Intent fail: {n_intent_fail}, Scope fail: {n_scope_fail}, "
          f"Tool fail: {n_tool_fail}, Loc fail: {n_loc_fail}, Fallback: {n_fallback}")
    print("─" * 60)


if __name__ == "__main__":
    main()
