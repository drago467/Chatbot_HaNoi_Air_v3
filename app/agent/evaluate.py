"""
Batch Evaluation Script for Weather Chatbot.

Features:
1. Unique thread_id per question (no conversation contamination)
2. Tool selection accuracy tracking (intent -> expected tools)
3. LLM-as-Judge evaluation (relevance, completeness, fluency, faithfulness)
   - Based on G-Eval (NeurIPS 2023), RAGAS framework best practices
   - Scale 1-5 for highest human-LLM alignment
   - Chain-of-thought before scoring
   - Separate faithfulness judge (reference-based, needs tool output)
4. Metrics breakdown by intent, difficulty
5. Response time percentiles (p50, p90, p95)
"""
import csv
import json
import math
import time
import os
from pathlib import Path
from uuid import uuid4
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field

from app.agent.agent import run_agent, reset_agent
from app.agent.evaluation_logger import get_evaluation_logger

# Reset agent to ensure fresh instance with latest code
reset_agent()


# ---- Statistical Helpers ----

def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple:
    """Wilson score confidence interval for binomial proportion.

    More accurate than normal approximation for small samples (n < 30).
    Reference: Agresti & Coull (1998), Wilson (1927).

    Args:
        successes: Number of successes (e.g., correct tool selections)
        total: Total number of trials
        z: Z-score for confidence level (1.96 = 95% CI)

    Returns:
        (lower_bound, upper_bound) as percentages (0-100)
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    lower = round(max(0.0, center - margin) * 100, 1)
    upper = round(min(1.0, center + margin) * 100, 1)
    return (lower, upper)


# ---- Intent -> Expected Tools mapping (Hierarchical by location scope) ----
# For each (intent, location_scope), lists the acceptable tools that can
# properly answer the question at that scope level.
# Recall = 1.0 if agent called at least one expected tool, else 0.0.
# This is stricter than a flat list: the agent must pick the RIGHT tool
# for the RIGHT location level.
#
# Design rationale per intent (verified against all 163 test questions):
#
# current_weather: city→aggregated city tool, district→district/ward tool,
#   ward/poi→ward-level tool. Agent MUST match the location granularity.
#
# hourly_forecast: get_hourly_forecast works at all scopes (auto-resolves
#   location). get_rain_timeline is a valid alternative for rain-related
#   hourly questions.
#
# daily_forecast: city→city aggregated daily, district→district aggregated
#   or ward-level daily, ward/poi→ward daily. get_weather_period is a
#   valid alternative for multi-day queries at any scope.
#
# weather_overview: summary tools (get_daily_summary, get_weather_period)
#   plus scope-appropriate data tools. detect_phenomena adds context.
#
# rain_query: spans hourly+daily time scales. Includes rain-specific tools
#   (get_rain_timeline) plus scope-appropriate forecast tools.
#
# temperature_query: scope-appropriate current/forecast tools plus
#   get_temperature_trend for trend questions.
#
# wind_query / humidity_fog_query: scope-appropriate current/forecast tools
#   plus domain-specific tools (detect_phenomena, get_weather_alerts).
#
# historical_weather: only get_weather_history at all scopes.
#
# location_comparison: needs data from multiple locations. compare_weather
#   is the primary tool. Individual data tools also valid (called twice).
#   city scope = comparing districts within the city.
#
# activity_weather: activity-specific tools (get_activity_advice, get_best_time,
#   get_comfort_index, get_clothing_advice) plus scope-appropriate data tools.
#
# expert_weather_param: scope-appropriate data tools. get_weather_history
#   valid for "hôm qua" expert queries.
#
# weather_alert: alert-specific tools plus supplementary data tools.
#
# smalltalk_weather: very diverse category (greetings, clothing advice,
#   seasonal questions, anomaly detection). Broad expected set. Special case:
#   no tools called is acceptable (greetings, out-of-scope).

INTENT_TO_TOOLS = {
    "current_weather": {
        "city": ["get_city_weather"],
        "district": ["get_district_weather", "get_current_weather"],
        "ward": ["get_current_weather"],
        "poi": ["get_current_weather"],
    },
    "hourly_forecast": {
        "city": ["get_hourly_forecast", "get_rain_timeline"],
        "district": ["get_hourly_forecast", "get_rain_timeline"],
        "ward": ["get_hourly_forecast", "get_rain_timeline"],
        "poi": ["get_hourly_forecast", "get_rain_timeline"],
    },
    "daily_forecast": {
        "city": ["get_city_daily_forecast", "get_weather_period", "get_daily_summary"],
        "district": ["get_district_daily_forecast", "get_daily_forecast", "get_weather_period"],
        "ward": ["get_daily_forecast", "get_weather_period"],
        "poi": ["get_daily_forecast", "get_weather_period"],
    },
    "weather_overview": {
        "city": ["get_daily_summary", "get_weather_period", "get_city_daily_forecast",
                 "get_city_weather", "detect_phenomena"],
        "district": ["get_daily_summary", "get_weather_period", "get_district_daily_forecast",
                     "get_district_weather", "detect_phenomena"],
        "ward": ["get_daily_summary", "get_weather_period", "get_daily_forecast"],
        "poi": ["get_daily_summary", "get_weather_period", "get_daily_forecast"],
    },
    "rain_query": {
        "city": ["get_rain_timeline", "get_hourly_forecast", "get_weather_period",
                 "get_city_daily_forecast", "detect_phenomena"],
        "district": ["get_rain_timeline", "get_hourly_forecast", "get_district_weather",
                     "get_weather_period", "get_district_daily_forecast"],
        "ward": ["get_rain_timeline", "get_hourly_forecast", "get_current_weather",
                 "get_daily_forecast"],
        "poi": ["get_rain_timeline", "get_hourly_forecast", "get_current_weather"],
    },
    "temperature_query": {
        "city": ["get_city_weather", "get_hourly_forecast", "get_weather_period",
                 "get_temperature_trend", "get_city_daily_forecast"],
        "district": ["get_district_weather", "get_current_weather", "get_hourly_forecast",
                     "get_district_daily_forecast", "get_weather_period"],
        "ward": ["get_current_weather", "get_hourly_forecast", "get_daily_forecast"],
        "poi": ["get_current_weather", "get_hourly_forecast"],
    },
    "wind_query": {
        "city": ["get_city_weather", "get_hourly_forecast", "get_weather_alerts",
                 "get_weather_period", "detect_phenomena"],
        "district": ["get_district_weather", "get_current_weather", "get_hourly_forecast"],
        "ward": ["get_current_weather", "get_hourly_forecast"],
        "poi": ["get_current_weather", "get_hourly_forecast"],
    },
    "humidity_fog_query": {
        "city": ["get_city_weather", "get_hourly_forecast", "detect_phenomena",
                 "get_weather_period"],
        "district": ["get_district_weather", "get_current_weather", "get_hourly_forecast",
                     "detect_phenomena"],
        "ward": ["get_current_weather", "get_hourly_forecast", "detect_phenomena"],
        "poi": ["get_current_weather", "get_hourly_forecast"],
    },
    "historical_weather": {
        "city": ["get_weather_history"],
        "district": ["get_weather_history"],
        "ward": ["get_weather_history"],
        "poi": ["get_weather_history"],
    },
    "location_comparison": {
        "city": ["get_district_weather", "get_district_ranking", "get_weather_period",
                 "get_city_daily_forecast"],
        "district": ["compare_weather", "get_district_weather", "get_current_weather",
                     "get_rain_timeline", "get_hourly_forecast", "get_district_daily_forecast"],
        "ward": ["compare_weather", "get_current_weather", "get_hourly_forecast"],
        "poi": ["compare_weather", "get_current_weather", "get_hourly_forecast"],
    },
    "activity_weather": {
        "city": ["get_activity_advice", "get_best_time", "get_comfort_index",
                 "get_clothing_advice", "get_city_weather", "get_hourly_forecast"],
        "district": ["get_activity_advice", "get_best_time", "get_comfort_index",
                     "get_clothing_advice", "get_district_weather", "get_hourly_forecast",
                     "get_daily_forecast", "get_district_daily_forecast"],
        "ward": ["get_activity_advice", "get_best_time", "get_comfort_index",
                 "get_hourly_forecast", "get_daily_forecast"],
        "poi": ["get_activity_advice", "get_best_time", "get_comfort_index",
                "get_clothing_advice", "get_hourly_forecast", "get_weather_period"],
    },
    "expert_weather_param": {
        "city": ["get_city_weather", "get_hourly_forecast", "get_daily_summary",
                 "get_weather_history"],
        "district": ["get_district_weather", "get_current_weather", "get_hourly_forecast",
                     "get_weather_history"],
        "ward": ["get_current_weather", "get_hourly_forecast", "get_daily_summary"],
        "poi": ["get_current_weather", "get_hourly_forecast"],
    },
    "weather_alert": {
        "city": ["get_weather_alerts", "get_weather_change_alert", "detect_phenomena",
                 "get_hourly_forecast", "get_temperature_trend", "get_weather_period"],
        "district": ["get_weather_alerts", "get_weather_change_alert", "detect_phenomena",
                     "get_hourly_forecast", "get_temperature_trend"],
        "ward": ["get_weather_alerts", "get_weather_change_alert", "get_hourly_forecast"],
        "poi": ["get_weather_alerts", "get_weather_change_alert"],
    },
    "smalltalk_weather": {
        # Smalltalk is very diverse: greetings, clothing advice, seasonal,
        # anomaly detection, rain check, stargazing, etc.
        # Broad expected set; no-tools-called is also acceptable (greetings).
        "city": [
            "get_city_weather", "get_daily_summary", "get_clothing_advice",
            "get_comfort_index", "get_seasonal_comparison", "get_weather_change_alert",
            "get_hourly_forecast", "get_current_weather", "get_daily_forecast",
            "get_city_daily_forecast", "detect_phenomena", "get_weather_period",
        ],
    },
}


# ---- Pydantic Models for Judge Responses ----

class QualityScore(BaseModel):
    """LLM judge response for quality evaluation."""
    reasoning: str = Field(description="Phân tích ngắn gọn 2-3 câu")
    relevance: int = Field(ge=1, le=5, description="Mức độ liên quan 1-5")
    completeness: int = Field(ge=1, le=5, description="Mức độ đầy đủ 1-5")
    fluency: int = Field(ge=1, le=5, description="Mức độ tự nhiên 1-5")
    actionability: int = Field(ge=1, le=5, description="Tính hữu dụng thực tế 1-5")


class FaithfulnessScore(BaseModel):
    """LLM judge response for faithfulness evaluation."""
    reasoning: str = Field(description="Giải thích ngắn")
    faithfulness: int = Field(ge=1, le=5, description="Độ trung thực 1-5")


# ---- LLM-as-Judge Prompts ----
# Based on G-Eval (NeurIPS 2023) chain-of-thought approach
# Scale 1-5 for highest human-LLM alignment (arxiv 2601.03444)

JUDGE_PROMPT_QUALITY = """Bạn là chuyên gia đánh giá chatbot thời tiết Hà Nội. Hãy đánh giá câu trả lời dưới đây.
Đây là chatbot thời tiết chuyên về Hà Nội với các thuật ngữ chuyên ngành như "nồm ẩm", "gió Lào", "rét đậm", "sương mù".

## Câu hỏi của người dùng:
{question}

## Câu trả lời của chatbot:
{response}

## Hướng dẫn đánh giá:
Hãy suy nghĩ từng bước trước khi cho điểm.

**RELEVANCE (Mức độ liên quan):**
- 5: Trả lời chính xác, đúng trọng tâm câu hỏi
- 4: Trả lời đúng nhưng có thông tin thừa nhỏ
- 3: Trả lời một phần, bỏ sót điểm quan trọng
- 2: Trả lời lạc đề hoặc sai hướng
- 1: Hoàn toàn không liên quan hoặc từ chối trả lời

**COMPLETENESS (Đầy đủ):**
- 5: Đầy đủ tất cả thông tin quan trọng (nhiệt độ, độ ẩm, gió, mưa, khuyến nghị nếu cần)
- 4: Đầy đủ, thiếu chi tiết nhỏ không quan trọng
- 3: Thiếu một số thông tin quan trọng
- 2: Thiếu nhiều thông tin cần thiết
- 1: Gần như không có thông tin hữu ích

**FLUENCY (Tự nhiên):**
- 5: Rất tự nhiên, chuyên nghiệp, dễ đọc
- 4: Tự nhiên, có lỗi nhỏ không đáng kể
- 3: Chấp nhận được, có vài chỗ gượng
- 2: Khó đọc, nhiều lỗi ngữ pháp/từ vựng
- 1: Không thể đọc được

**ACTIONABILITY (Tính hữu dụng):**
- 5: Có khuyến nghị cụ thể, thực tế (mang ô, mặc áo khoác, tránh ra ngoài 10-14h, uống nhiều nước)
- 4: Có khuyến nghị nhưng chung chung (nên cẩn thận, chú ý thời tiết)
- 3: Ít khuyến nghị, chủ yếu liệt kê số liệu
- 2: Không có khuyến nghị dù câu hỏi cần (ví dụ: hỏi có nên ra ngoài không mà chỉ trả lời nhiệt độ)
- 1: Thông tin không dùng được, không giúp người dùng ra quyết định"""

JUDGE_PROMPT_FAITHFULNESS = """Bạn là chuyên gia kiểm tra tính chính xác của chatbot thời tiết Hà Nội.

## Câu hỏi của người dùng:
{question}

## Dữ liệu thời tiết thực tế (từ database):
{tool_output}

## Câu trả lời của chatbot:
{response}

## Nhiệm vụ:
Kiểm tra xem câu trả lời có chứa thông tin SAI hoặc BỊA ĐẶT không có trong dữ liệu thực tế không.
Lưu ý: chatbot có thể làm tròn số hoặc diễn giải dữ liệu, điều đó là chấp nhận được.

**FAITHFULNESS (Độ trung thực):**
- 5: Tất cả thông tin đều chính xác, không có gì bịa đặt
- 4: Hầu hết chính xác, có 1 chi tiết nhỏ không chính xác
- 3: Có 1-2 thông tin sai hoặc không có trong dữ liệu
- 2: Có nhiều thông tin sai hoặc bịa đặt
- 1: Phần lớn thông tin sai hoặc không có cơ sở"""


# ---- Helper Functions ----

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


def extract_tool_outputs(result) -> str:
    """Extract tool outputs from agent result for faithfulness check."""
    outputs = []
    for msg in result.get("messages", []):
        msg_type = getattr(msg, "type", None)
        if msg_type == "tool":
            content = getattr(msg, "content", str(msg))
            if content:
                outputs.append(str(content)[:1000])
    return "\n---\n".join(outputs) if outputs else ""


def _get_expected_tools(intent: str, location_scope: str = "") -> list:
    """Get expected tools for (intent, location_scope) from hierarchical mapping.

    Falls back to "city" scope if the specific scope is not defined for an intent.
    Returns empty list for unknown intents.
    """
    intent_map = INTENT_TO_TOOLS.get(intent, {})
    if not intent_map:
        return []
    # Look up by scope; fallback to "city" if scope not found in this intent
    tools = intent_map.get(location_scope) or intent_map.get("city", [])
    return tools


def check_tool_accuracy(intent: str, tools_called: list,
                        location_scope: str = "") -> bool:
    """Check if at least one scope-appropriate tool was called.

    Uses hierarchical INTENT_TO_TOOLS: the expected tools depend on BOTH the
    intent AND the location scope (city/district/ward/poi).
    """
    expected = _get_expected_tools(intent, location_scope)
    if not expected:
        return True  # Unknown intent, skip check
    # For smalltalk: no tools called is acceptable (greeting, out-of-scope, etc.)
    if intent == "smalltalk_weather" and not tools_called:
        return True
    return any(t in expected for t in tools_called)


def check_tool_precision(intent: str, tools_called: list,
                         location_scope: str = "") -> float:
    """What fraction of called tools were scope-appropriate? (precision)

    Uses hierarchical INTENT_TO_TOOLS so that e.g. calling get_city_weather
    for a ward-level question is correctly identified as irrelevant.
    """
    expected = set(_get_expected_tools(intent, location_scope))
    if not tools_called:
        # No tools called: for smalltalk this is fine (precision=1.0)
        return 1.0 if intent == "smalltalk_weather" else 0.0
    # Exclude resolve_location from precision calc (it's a helper, always valid)
    relevant_calls = [t for t in tools_called if t != "resolve_location"]
    if not relevant_calls:
        return 1.0
    relevant = sum(1 for t in relevant_calls if t in expected)
    return round(relevant / len(relevant_calls), 2)


def check_tool_recall(intent: str, tools_called: list,
                      location_scope: str = "") -> float:
    """Scope-aware tool recall: did the agent call at least one expected tool?

    Returns 1.0 if any called tool is in the expected set for (intent, scope),
    otherwise 0.0.

    This is binary recall because the expected set contains ALTERNATIVES
    (any one is sufficient), not REQUIREMENTS (all must be called).
    Using fractional recall (|called ∩ expected| / |expected|) would penalize
    the agent for not calling ALL alternatives, which is incorrect.

    Example:
    - intent=current_weather, scope=city
    - expected = ["get_city_weather"]
    - tools_called = ["get_city_weather"] → recall = 1.0
    - tools_called = ["get_current_weather"] → recall = 0.0 (wrong scope)
    """
    expected = _get_expected_tools(intent, location_scope)
    if not expected:
        return 1.0  # Unknown intent
    # For smalltalk: no tools called is acceptable
    if intent == "smalltalk_weather" and not tools_called:
        return 1.0

    called_set = set(tools_called)
    expected_set = set(expected)
    return 1.0 if (called_set & expected_set) else 0.0


def categorize_error(error_str: str) -> str:
    """Categorize error type for analysis."""
    err = error_str.lower()
    if "location" in err or "not_found" in err or "ambiguous" in err:
        return "location_resolution"
    elif "no_data" in err or "database" in err or "không có dữ liệu" in err:
        return "data_unavailable"
    elif "timeout" in err or "connection" in err or "refused" in err:
        return "network"
    elif "openai" in err or "api" in err or "rate_limit" in err:
        return "llm_api"
    return "unknown"


def call_judge_quality(client, prompt, model=None) -> Optional[QualityScore]:
    """Call LLM judge for quality scoring with structured output via response_format."""
    if model is None:
        model = os.getenv("JUDGE_MODEL", os.getenv("MODEL", "gpt-4o-mini"))
    try:
        resp = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400,
            response_format=QualityScore,
        )
        return resp.choices[0].message.parsed
    except Exception:
        # Fallback: try without structured output (for APIs that don't support it)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return QualityScore(**data)
        except Exception:
            return None


def call_judge_faithfulness(client, prompt, model=None) -> Optional[FaithfulnessScore]:
    """Call LLM judge for faithfulness scoring with structured output via response_format."""
    if model is None:
        model = os.getenv("JUDGE_MODEL", os.getenv("MODEL", "gpt-4o-mini"))
    try:
        resp = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400,
            response_format=FaithfulnessScore,
        )
        return resp.choices[0].message.parsed
    except Exception:
        # Fallback: try without structured output
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return FaithfulnessScore(**data)
        except Exception:
            return None


def llm_judge(question, response, tool_output=None, client=None) -> dict:
    """Run LLM-as-Judge evaluation. Returns dict with validated scores 1-5."""
    if client is None:
        from openai import OpenAI
        client = OpenAI(
            base_url=os.getenv("API_BASE"),
            api_key=os.getenv("API_KEY"),
        )

    # Quality judge (always run)
    quality = call_judge_quality(client, JUDGE_PROMPT_QUALITY.format(
        question=question, response=response,
    ))

    # Faithfulness judge (only if tool output available)
    faith = None
    if tool_output and len(tool_output.strip()) > 10:
        faith = call_judge_faithfulness(client, JUDGE_PROMPT_FAITHFULNESS.format(
            question=question, response=response,
            tool_output=tool_output[:2000],
        ))

    return {
        "relevance": quality.relevance if quality else None,
        "completeness": quality.completeness if quality else None,
        "fluency": quality.fluency if quality else None,
        "actionability": quality.actionability if quality else None,
        "faithfulness": faith.faithfulness if faith else None,
        "judge_reasoning": quality.reasoning if quality else "",
        "faith_reasoning": faith.reasoning if faith else "",
    }


def load_test_queries(csv_path):
    queries = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row)
    return queries


def evaluate_query(question, query_id, expected_tool=None, expected_location=None,
                   location_scope="", judge_client=None, skip_judge=False):
    """Evaluate a single query with unique thread_id and optional LLM judge."""
    start_time = time.time()
    thread_id = f"eval_{query_id}_{uuid4().hex[:8]}"

    try:
        result = run_agent(message=question, thread_id=thread_id)
        messages = result.get("messages", [])
        response = messages[-1].content if messages else ""
        if response is None:
            response = ""
        elapsed_ms = (time.time() - start_time) * 1000

        tools_called = extract_tool_names(result)
        intent = expected_tool or ""
        tool_correct = check_tool_accuracy(intent, tools_called, location_scope)
        tool_output = extract_tool_outputs(result)

        eval_result = {
            "question": question,
            "intent": expected_tool,
            "location": expected_location,
            "location_scope": location_scope,
            "response": response,
            "response_time_ms": round(elapsed_ms),
            "success": True,
            "error": None,
            "error_category": None,
            "tools_called": ",".join(tools_called),
            "tool_correct": tool_correct,
            "tool_precision": check_tool_precision(intent, tools_called, location_scope),
            "tool_recall": check_tool_recall(intent, tools_called, location_scope),
            "tool_output_raw": tool_output[:500],
        }

        # LLM-as-Judge
        if not skip_judge and response:
            judge_scores = llm_judge(question, response, tool_output, judge_client)
            eval_result.update({
                "judge_relevance": judge_scores.get("relevance"),
                "judge_completeness": judge_scores.get("completeness"),
                "judge_fluency": judge_scores.get("fluency"),
                "judge_actionability": judge_scores.get("actionability"),
                "judge_faithfulness": judge_scores.get("faithfulness"),
                "judge_reasoning": judge_scores.get("judge_reasoning", ""),
                "faith_reasoning": judge_scores.get("faith_reasoning", ""),
            })
        else:
            eval_result.update({
                "judge_relevance": None,
                "judge_completeness": None,
                "judge_fluency": None,
                "judge_actionability": None,
                "judge_faithfulness": None,
                "judge_reasoning": "",
                "faith_reasoning": "",
            })

        return eval_result

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        error_str = str(e)
        return {
            "question": question,
            "intent": expected_tool,
            "location": expected_location,
            "location_scope": location_scope,
            "response": "",
            "response_time_ms": round(elapsed_ms),
            "success": False,
            "error": error_str,
            "error_category": categorize_error(error_str),
            "tools_called": "",
            "tool_correct": False,
            "tool_precision": 0.0,
            "tool_recall": 0.0,
            "tool_output_raw": "",
            "judge_relevance": None,
            "judge_completeness": None,
            "judge_fluency": None,
            "judge_actionability": None,
            "judge_faithfulness": None,
            "judge_reasoning": "",
            "faith_reasoning": "",
        }


def compute_metrics(results):
    """Compute comprehensive evaluation metrics including judge scores."""
    total = len(results)
    successful = [r for r in results if r["success"]]
    times = sorted([r["response_time_ms"] for r in results])

    # Core counts for CI calculation
    tool_correct_count = sum(1 for r in results if r.get("tool_correct"))
    recall_correct_count = sum(1 for r in results if r.get("tool_recall", 0) >= 1.0)

    metrics = {
        "total": total,
        "successful": len(successful),
        "success_rate": round(len(successful) / total * 100, 1) if total else 0,
        "tool_accuracy": round(tool_correct_count / total * 100, 1) if total else 0,
        "tool_accuracy_ci95": wilson_ci(tool_correct_count, total),
        "tool_precision_avg": round(
            sum(r.get("tool_precision", 0) for r in results) / total, 2
        ) if total else 0,
        "tool_recall_avg": round(
            sum(r.get("tool_recall", 0) for r in results) / total, 2
        ) if total else 0,
        "tool_recall_ci95": wilson_ci(recall_correct_count, total),
        "avg_time_ms": round(sum(times) / total) if total else 0,
        "p50_time_ms": round(times[total // 2]) if times else 0,
        "p90_time_ms": round(times[int(total * 0.9)]) if times else 0,
        "p95_time_ms": round(times[int(total * 0.95)]) if times else 0,
    }

    # Error category breakdown
    error_cats = {}
    for r in results:
        cat = r.get("error_category")
        if cat:
            error_cats[cat] = error_cats.get(cat, 0) + 1
    if error_cats:
        metrics["error_categories"] = error_cats

    # Judge score averages
    judge_dims = ["judge_relevance", "judge_completeness", "judge_fluency", "judge_actionability", "judge_faithfulness"]
    for dim in judge_dims:
        vals = [r[dim] for r in results if r.get(dim) is not None]
        metrics[dim + "_avg"] = round(sum(vals) / len(vals), 2) if vals else None
        metrics[dim + "_count"] = len(vals)

    # By intent (with judge scores)
    by_intent = {}
    for r in results:
        intent = r.get("intent", "unknown")
        if intent not in by_intent:
            by_intent[intent] = {
                "total": 0, "success": 0, "tool_correct": 0, "times": [],
                "judge_scores": {d: [] for d in judge_dims},
            }
        by_intent[intent]["total"] += 1
        if r["success"]:
            by_intent[intent]["success"] += 1
        if r.get("tool_correct"):
            by_intent[intent]["tool_correct"] += 1
        by_intent[intent]["times"].append(r["response_time_ms"])
        for d in judge_dims:
            if r.get(d) is not None:
                by_intent[intent]["judge_scores"][d].append(r[d])

    metrics["by_intent"] = {}
    for k, v in by_intent.items():
        entry = {
            "total": v["total"],
            "success_rate": round(v["success"] / v["total"] * 100, 1),
            "tool_accuracy": round(v["tool_correct"] / v["total"] * 100, 1),
            "tool_accuracy_ci95": wilson_ci(v["tool_correct"], v["total"]),
            "avg_time_ms": round(sum(v["times"]) / len(v["times"])),
        }
        for d in judge_dims:
            vals = v["judge_scores"][d]
            entry[d + "_avg"] = round(sum(vals) / len(vals), 2) if vals else None
        metrics["by_intent"][k] = entry

    # By difficulty (with judge scores)
    by_diff = {}
    for r in results:
        diff = r.get("difficulty", "unknown")
        if diff not in by_diff:
            by_diff[diff] = {
                "total": 0, "success": 0, "tool_correct": 0,
                "judge_scores": {d: [] for d in judge_dims},
            }
        by_diff[diff]["total"] += 1
        if r["success"]:
            by_diff[diff]["success"] += 1
        if r.get("tool_correct"):
            by_diff[diff]["tool_correct"] += 1
        for d in judge_dims:
            if r.get(d) is not None:
                by_diff[diff]["judge_scores"][d].append(r[d])

    metrics["by_difficulty"] = {}
    for k, v in by_diff.items():
        entry = {
            "total": v["total"],
            "success_rate": round(v["success"] / v["total"] * 100, 1),
            "tool_accuracy": round(v["tool_correct"] / v["total"] * 100, 1),
            "tool_accuracy_ci95": wilson_ci(v["tool_correct"], v["total"]),
        }
        for d in judge_dims:
            vals = v["judge_scores"][d]
            entry[d + "_avg"] = round(sum(vals) / len(vals), 2) if vals else None
        metrics["by_difficulty"][k] = entry

    return metrics


def run_evaluation(output_dir="data/evaluation", skip_judge=False):
    """Run full evaluation pipeline."""
    logger = get_evaluation_logger(output_dir)

    test_file = Path(output_dir) / "hanoi_weather_chatbot_eval_questions.csv"
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return

    queries = load_test_queries(str(test_file))
    print(f"Loaded {len(queries)} test queries")

    # Initialize judge client once (reuse connection)
    judge_client = None
    if not skip_judge:
        try:
            from openai import OpenAI
            judge_client = OpenAI(
                base_url=os.getenv("API_BASE"),
                api_key=os.getenv("API_KEY"),
            )
            print("LLM-as-Judge: ENABLED")
        except Exception as e:
            print(f"LLM-as-Judge: DISABLED ({e})")
            skip_judge = True
    else:
        print("LLM-as-Judge: SKIPPED (--skip-judge)")

    results = []
    for i, q in enumerate(queries, 1):
        question = q.get("question", q.get("query", ""))
        print(f"[{i}/{len(queries)}] {question[:60]}...")

        result = evaluate_query(
            question=question,
            query_id=i,
            expected_tool=q.get("intent"),
            expected_location=q.get("location_name"),
            location_scope=q.get("location_scope", ""),
            judge_client=judge_client,
            skip_judge=skip_judge,
        )
        result["difficulty"] = q.get("difficulty", "unknown")
        results.append(result)

        # Print judge scores inline
        if not skip_judge and result.get("judge_relevance") is not None:
            r, c, fl, a, fa = (
                result.get("judge_relevance", "-"),
                result.get("judge_completeness", "-"),
                result.get("judge_fluency", "-"),
                result.get("judge_actionability", "-"),
                result.get("judge_faithfulness", "-"),
            )
            print(f"  -> Judge: R={r} C={c} F={fl} A={a} Faith={fa}")

        logger.log_conversation(
            session_id=f"eval_{i}",
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
    ci = metrics['tool_accuracy_ci95']
    print(f"Tool accuracy: {metrics['tool_accuracy']}% [95% CI: {ci[0]}-{ci[1]}%]")
    rci = metrics['tool_recall_ci95']
    print(f"Tool precision: {metrics['tool_precision_avg']} | Tool recall: {metrics['tool_recall_avg']} [95% CI: {rci[0]}-{rci[1]}%]")
    print(f"Avg time: {metrics['avg_time_ms']}ms | p50: {metrics['p50_time_ms']}ms | p90: {metrics['p90_time_ms']}ms | p95: {metrics['p95_time_ms']}ms")

    # Judge scores
    if not skip_judge:
        print()
        print("LLM-as-Judge Scores (1-5):")
        for dim in ["judge_relevance", "judge_completeness", "judge_fluency",
                     "judge_actionability", "judge_faithfulness"]:
            avg = metrics.get(dim + "_avg")
            cnt = metrics.get(dim + "_count", 0)
            label = dim.replace("judge_", "").capitalize()
            print(f"  {label}: {avg}/5 ({cnt} rated)")

    print()
    print("By Intent:")
    for intent, data in sorted(metrics["by_intent"].items()):
        ci = data.get("tool_accuracy_ci95", (0, 0))
        judge_str = ""
        if not skip_judge:
            r_avg = data.get("judge_relevance_avg", "-")
            f_avg = data.get("judge_faithfulness_avg", "-")
            judge_str = f", rel={r_avg}, faith={f_avg}"
        print(f"  {intent}: {data['tool_accuracy']}% [CI: {ci[0]}-{ci[1]}%], {data['avg_time_ms']}ms{judge_str} ({data['total']}q)")

    print()
    print("By Difficulty:")
    for diff, data in sorted(metrics["by_difficulty"].items()):
        ci = data.get("tool_accuracy_ci95", (0, 0))
        judge_str = ""
        if not skip_judge:
            r_avg = data.get("judge_relevance_avg", "-")
            f_avg = data.get("judge_faithfulness_avg", "-")
            judge_str = f", rel={r_avg}, faith={f_avg}"
        print(f"  {diff}: acc={data['tool_accuracy']}% [CI: {ci[0]}-{ci[1]}%]{judge_str} ({data['total']}q)")

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
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM-as-Judge evaluation (faster, no judge scores)")
    args = parser.parse_args()
    run_evaluation(args.output, skip_judge=args.skip_judge)
