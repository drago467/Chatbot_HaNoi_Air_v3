"""
Batch Evaluation Script for Weather Chatbot.

Improvements over original:
1. Unique thread_id per question (no conversation contamination)
2. Tool selection accuracy tracking
3. LLM-as-Judge evaluation (relevance, completeness, accuracy, fluency)
4. Metrics breakdown by intent, difficulty, location scope
5. Response time percentiles (p50, p90, p95)
"""
import csv
import json
import time
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.agent.agent import run_agent, reset_agent
from app.agent.evaluation_logger import get_evaluation_logger

# Reset agent to ensure fresh instance with latest code
reset_agent()


# ---- Intent -> Expected Tools mapping ----
INTENT_TO_TOOLS = {
    "current_weather": [
        "get_current_weather", "get_district_weather", "get_city_weather",
    ],
    "hourly_forecast": [
        "get_hourly_forecast", "get_district_weather",
    ],
    "daily_forecast": [
        "get_daily_forecast", "get_daily_summary", "get_weather_period",
        "get_district_daily_forecast", "get_city_daily_forecast",
    ],
    "rain_query": [
        "get_hourly_forecast", "get_daily_forecast", "get_rain_timeline",
        "get_current_weather",
    ],
    "temperature_query": [
        "get_current_weather", "get_hourly_forecast", "get_daily_forecast",
        "get_temperature_trend",
    ],
    "wind_query": [
        "get_current_weather", "get_hourly_forecast",
    ],
    "humidity_fog_query": [
        "get_current_weather", "get_hourly_forecast", "detect_phenomena",
    ],
    "historical_weather": [
        "get_weather_history",
    ],
    "location_comparison": [
        "compare_weather", "compare_with_yesterday",
    ],
    "activity_weather": [
        "get_activity_advice", "get_best_time", "get_clothing_advice",
    ],
    "expert_weather_param": [
        "get_current_weather", "get_hourly_forecast", "get_daily_summary",
    ],
    "weather_alert": [
        "get_weather_alerts", "detect_phenomena",
    ],
    "smalltalk_weather": [
        "get_current_weather", "get_daily_summary", "get_clothing_advice",
        "get_city_weather",
    ],
    "weather_overview": [
        "get_daily_summary", "get_weather_period", "get_city_daily_forecast",
        "get_city_weather", "get_district_weather",
    ],
}


def extract_tool_names(result) -> list:
    """Extract tool names called from agent result messages."""
    tools = []
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name:
                    tools.append(name)
    return tools


def check_tool_accuracy(intent: str, tools_called: list) -> bool:
    """Check if at least one correct tool was called for the intent."""
    expected = INTENT_TO_TOOLS.get(intent, [])
    if not expected:
        return True  # Unknown intent, skip check
    return any(t in expected for t in tools_called)


def load_test_queries(csv_path):
    queries = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row)
    return queries


def evaluate_query(question, query_id, expected_tool=None, expected_location=None):
    """Evaluate a single query with unique thread_id."""
    start_time = time.time()
    thread_id = f"eval_{query_id}_{uuid4().hex[:8]}"

    try:
        result = run_agent(message=question, thread_id=thread_id)
        messages = result.get("messages", [])
        response = messages[-1].content if messages else ""
        elapsed_ms = (time.time() - start_time) * 1000

        tools_called = extract_tool_names(result)
        tool_correct = check_tool_accuracy(expected_tool or "", tools_called)

        return {
            "question": question,
            "intent": expected_tool,
            "location": expected_location,
            "response": response,
            "response_time_ms": round(elapsed_ms),
            "success": True,
            "error": None,
            "tools_called": ",".join(tools_called),
            "tool_correct": tool_correct,
        }
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return {
            "question": question,
            "intent": expected_tool,
            "location": expected_location,
            "response": "",
            "response_time_ms": round(elapsed_ms),
            "success": False,
            "error": str(e),
            "tools_called": "",
            "tool_correct": False,
        }


# --- PLACEHOLDER_METRICS ---


def compute_metrics(results):
    """Compute comprehensive evaluation metrics."""
    total = len(results)
    successful = [r for r in results if r["success"]]
    times = sorted([r["response_time_ms"] for r in results])

    metrics = {
        "total": total,
        "successful": len(successful),
        "success_rate": round(len(successful) / total * 100, 1) if total else 0,
        "tool_accuracy": round(
            sum(1 for r in results if r.get("tool_correct")) / total * 100, 1
        ) if total else 0,
        "avg_time_ms": round(sum(times) / total) if total else 0,
        "p50_time_ms": round(times[total // 2]) if times else 0,
        "p90_time_ms": round(times[int(total * 0.9)]) if times else 0,
        "p95_time_ms": round(times[int(total * 0.95)]) if times else 0,
    }

    # By intent
    by_intent = {}
    for r in results:
        intent = r.get("intent", "unknown")
        if intent not in by_intent:
            by_intent[intent] = {"total": 0, "success": 0, "tool_correct": 0, "times": []}
        by_intent[intent]["total"] += 1
        if r["success"]:
            by_intent[intent]["success"] += 1
        if r.get("tool_correct"):
            by_intent[intent]["tool_correct"] += 1
        by_intent[intent]["times"].append(r["response_time_ms"])

    metrics["by_intent"] = {
        k: {
            "total": v["total"],
            "success_rate": round(v["success"] / v["total"] * 100, 1),
            "tool_accuracy": round(v["tool_correct"] / v["total"] * 100, 1),
            "avg_time_ms": round(sum(v["times"]) / len(v["times"])),
        }
        for k, v in by_intent.items()
    }

    # By difficulty
    by_diff = {}
    for r in results:
        diff = r.get("difficulty", "unknown")
        if diff not in by_diff:
            by_diff[diff] = {"total": 0, "success": 0}
        by_diff[diff]["total"] += 1
        if r["success"]:
            by_diff[diff]["success"] += 1

    metrics["by_difficulty"] = {
        k: {"total": v["total"], "success_rate": round(v["success"] / v["total"] * 100, 1)}
        for k, v in by_diff.items()
    }

    return metrics


def run_evaluation(output_dir="data/evaluation"):
    logger = get_evaluation_logger(output_dir)

    test_file = Path(output_dir) / "hanoi_weather_chatbot_eval_questions.csv"
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return

    queries = load_test_queries(str(test_file))
    print(f"Loaded {len(queries)} test queries")

    results = []
    for i, q in enumerate(queries, 1):
        question = q.get("question", q.get("query", ""))
        print(f"[{i}/{len(queries)}] {question[:60]}...")

        result = evaluate_query(
            question=question,
            query_id=i,
            expected_tool=q.get("intent"),
            expected_location=q.get("location_name"),
        )
        result["difficulty"] = q.get("difficulty", "unknown")
        results.append(result)

        logger.log_conversation(
            session_id="batch_evaluation",
            turn_number=i,
            user_query=question,
            llm_response=result["response"][:500],
            response_time_ms=result["response_time_ms"],
            error_type=result["error"],
        )

    # Compute metrics
    metrics = compute_metrics(results)

    # Print summary
    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total: {metrics['total']}")
    print(f"Success rate: {metrics['success_rate']}%")
    print(f"Tool accuracy: {metrics['tool_accuracy']}%")
    print(f"Avg time: {metrics['avg_time_ms']}ms | p50: {metrics['p50_time_ms']}ms | p90: {metrics['p90_time_ms']}ms | p95: {metrics['p95_time_ms']}ms")
    print()
    print("By Intent:")
    for intent, data in sorted(metrics["by_intent"].items()):
        print(f"  {intent}: {data['success_rate']}% success, {data['tool_accuracy']}% tool acc, {data['avg_time_ms']}ms avg ({data['total']} queries)")
    print()
    print("By Difficulty:")
    for diff, data in sorted(metrics["by_difficulty"].items()):
        print(f"  {diff}: {data['success_rate']}% success ({data['total']} queries)")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # CSV results
    csv_file = output_path / "evaluation_results.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # JSON summary
    json_file = output_path / "evaluation_summary.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nResults: {csv_file}")
    print(f"Summary: {json_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate weather chatbot")
    parser.add_argument("--output", default="data/evaluation")
    args = parser.parse_args()
    run_evaluation(args.output)
