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

from app.agent.agent import run_agent, run_agent_routed, reset_agent
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
# For each (intent, location_scope), lists acceptable tools that can
# properly answer the question at that scope level.
# Recall = 1.0 if agent called at least one expected tool, else 0.0.
#
# Design principles (post-refactor, 27 tools):
# - 3 scopes: city / district / ward  (POI queries resolve to district via dispatch.py)
# - All tools auto-dispatch to 3 tiers (ward/district/city) via dispatch.py
#   → ward là đơn vị data gốc, district/city aggregate từ ward
#   → ward PHẢI có cùng tool set với district/city (không lý do kỹ thuật để giới hạn)
# - INTENT_TO_TOOLS ⊇ PRIMARY_TOOL_MAP (router's focused tools)
# - smalltalk_weather: no-tools-called cũng chấp nhận được (greetings)

INTENT_TO_TOOLS = {
    # current_weather: "Bây giờ thời tiết thế nào?"
    # current chính; phenomena bổ sung ngữ cảnh; clothing_advice gợi ý thực tế
    "current_weather": {
        "city":     ["get_current_weather", "detect_phenomena", "get_clothing_advice"],
        "district": ["get_current_weather", "detect_phenomena", "get_clothing_advice"],
        "ward":     ["get_current_weather", "detect_phenomena", "get_clothing_advice"],
    },
    # hourly_forecast: "Chiều nay mưa không?", "Tối nay mấy độ?", "Lúc nào nắng?"
    # rain_timeline alternative cho mưa; sunny_periods cho câu hỏi nắng theo giờ
    # daily_rhythm hợp lệ cho câu hỏi theo buổi ("chiều nay", "sáng nay") — trả data tổng hợp per-period
    "hourly_forecast": {
        "city":     ["get_hourly_forecast", "get_rain_timeline", "get_sunny_periods", "get_daily_rhythm"],
        "district": ["get_hourly_forecast", "get_rain_timeline", "get_sunny_periods", "get_daily_rhythm"],
        "ward":     ["get_hourly_forecast", "get_rain_timeline", "get_sunny_periods", "get_daily_rhythm"],
    },
    # daily_forecast: "Ngày mai thế nào?", "Cuối tuần?"
    # weather_period cho multi-day range; temperature_trend bổ sung xu hướng
    "daily_forecast": {
        "city":     ["get_daily_forecast", "get_weather_period", "get_temperature_trend"],
        "district": ["get_daily_forecast", "get_weather_period", "get_temperature_trend"],
        "ward":     ["get_daily_forecast", "get_weather_period", "get_temperature_trend"],
    },
    # weather_overview: "Tổng hợp thời tiết hôm nay?"
    # daily_summary chính; phenomena + daily_rhythm bổ sung bức tranh toàn cảnh
    # current_weather cũng chấp nhận (subset hợp lệ khi router classify gần intent)
    # daily_forecast hợp lệ khi agent dùng forecast data để tổng hợp overview
    "weather_overview": {
        "city":     ["get_daily_summary", "detect_phenomena", "get_daily_rhythm", "get_current_weather", "get_daily_forecast"],
        "district": ["get_daily_summary", "detect_phenomena", "get_daily_rhythm", "get_current_weather", "get_daily_forecast"],
        "ward":     ["get_daily_summary", "detect_phenomena", "get_daily_rhythm", "get_current_weather", "get_daily_forecast"],
    },
    # rain_query: "Lúc nào mưa?", "Có mưa không?", "Mưa lớn gây ngập?"
    # rain_timeline chính; hourly/daily forecast là alternatives; weather_alerts cho câu hỏi ngập/lũ
    "rain_query": {
        "city":     ["get_rain_timeline", "get_hourly_forecast", "get_daily_forecast", "get_weather_alerts"],
        "district": ["get_rain_timeline", "get_hourly_forecast", "get_daily_forecast", "get_weather_alerts"],
        "ward":     ["get_rain_timeline", "get_hourly_forecast", "get_daily_forecast", "get_weather_alerts"],
    },
    # temperature_query: "Nhiệt độ?", "Nóng không?", "Nhiệt độ ngày mai?"
    # current cho hiện tại; trend cho xu hướng; hourly/daily cho dự báo (cross-intent overlap)
    "temperature_query": {
        "city":     ["get_current_weather", "get_temperature_trend", "get_hourly_forecast", "get_daily_forecast"],
        "district": ["get_current_weather", "get_temperature_trend", "get_hourly_forecast", "get_daily_forecast"],
        "ward":     ["get_current_weather", "get_temperature_trend", "get_hourly_forecast", "get_daily_forecast"],
    },
    # wind_query: "Gió mạnh không?", "Tốc độ gió?"
    # pressure_trend: áp suất giảm → front lạnh/bão → gió mạnh
    "wind_query": {
        "city":     ["get_current_weather", "get_hourly_forecast", "get_pressure_trend"],
        "district": ["get_current_weather", "get_hourly_forecast", "get_pressure_trend"],
        "ward":     ["get_current_weather", "get_hourly_forecast", "get_pressure_trend"],
    },
    # humidity_fog_query: "Độ ẩm?", "Sương mù?"
    # phenomena phát hiện nồm/sương; humidity_timeline chi tiết theo giờ
    "humidity_fog_query": {
        "city":     ["get_current_weather", "detect_phenomena", "get_humidity_timeline"],
        "district": ["get_current_weather", "detect_phenomena", "get_humidity_timeline"],
        "ward":     ["get_current_weather", "detect_phenomena", "get_humidity_timeline"],
    },
    # historical_weather: "Hôm qua thế nào?", "Tuần trước?", "Ngày 15/3?"
    # weather_history cho 1 ngày; daily_summary cho tóm tắt; weather_period cho khoảng nhiều ngày
    "historical_weather": {
        "city":     ["get_weather_history", "get_daily_summary", "get_weather_period"],
        "district": ["get_weather_history", "get_daily_summary", "get_weather_period"],
        "ward":     ["get_weather_history", "get_daily_summary", "get_weather_period"],
    },
    # location_comparison: "Cầu Giấy vs Đống Đa?", "Quận nào nóng nhất?"
    # city: ranking toàn TP + multi_compare; district: so sánh 2 nơi + ranking ward + multi_compare; ward: so sánh trực tiếp
    "location_comparison": {
        "city":     ["get_district_ranking", "get_district_multi_compare"],
        "district": ["compare_weather", "get_ward_ranking_in_district", "get_district_ranking", "get_district_multi_compare"],
        "ward":     ["compare_weather"],
    },
    # activity_weather: "Đi chơi được không?", "Mấy giờ chạy bộ tốt?", "Ra ngoài có ổn không?"
    # activity_advice chính; best_time tìm giờ tối ưu; uv_safe_windows an toàn UV
    # current_weather + comfort_index hợp lệ cho câu hỏi đơn giản (cross-intent overlap)
    "activity_weather": {
        "city":     ["get_activity_advice", "get_best_time", "get_uv_safe_windows", "get_current_weather", "get_comfort_index"],
        "district": ["get_activity_advice", "get_best_time", "get_uv_safe_windows", "get_current_weather", "get_comfort_index"],
        "ward":     ["get_activity_advice", "get_best_time", "get_uv_safe_windows", "get_current_weather", "get_comfort_index"],
    },
    # expert_weather_param: "Điểm sương?", "Áp suất?", "UV index?", "Lượng mưa hôm qua?"
    # comfort_index cho cảm giác thực tế; pressure_trend cho áp suất chi tiết
    # weather_history hợp lệ cho thông số kỹ thuật quá khứ (cross-intent overlap)
    # uv_safe_windows hợp lệ cho câu hỏi UV ("UV cao không?", "mấy giờ UV an toàn?")
    "expert_weather_param": {
        "city":     ["get_current_weather", "get_comfort_index", "get_pressure_trend", "get_weather_history", "get_uv_safe_windows"],
        "district": ["get_current_weather", "get_comfort_index", "get_pressure_trend", "get_weather_history", "get_uv_safe_windows"],
        "ward":     ["get_current_weather", "get_comfort_index", "get_pressure_trend", "get_weather_history", "get_uv_safe_windows"],
    },
    # weather_alert: "Có cảnh báo gì không?", "Giông mạnh?", "Mưa dông?"
    # alerts chính; change_alert phát hiện biến đổi; pressure_trend front/bão
    # rain_timeline + hourly_forecast hợp lệ cho alert về mưa/giông (cross-intent overlap)
    "weather_alert": {
        "city":     ["get_weather_alerts", "get_weather_change_alert", "get_pressure_trend", "get_rain_timeline", "get_hourly_forecast"],
        "district": ["get_weather_alerts", "get_weather_change_alert", "get_pressure_trend", "get_rain_timeline", "get_hourly_forecast"],
        "ward":     ["get_weather_alerts", "get_weather_change_alert", "get_pressure_trend", "get_rain_timeline", "get_hourly_forecast"],
    },
    # seasonal_context: "Nóng hơn bình thường không?", "So với mùa này?"
    # seasonal_comparison chính; compare_with_yesterday + temperature_trend bổ sung
    "seasonal_context": {
        "city":     ["get_seasonal_comparison", "compare_with_yesterday", "get_temperature_trend"],
        "district": ["get_seasonal_comparison", "compare_with_yesterday", "get_temperature_trend"],
        "ward":     ["get_seasonal_comparison", "compare_with_yesterday", "get_temperature_trend"],
    },
    # smalltalk_weather: "Xin chào", "Mặc gì hôm nay?", "Ngắm sao đẹp không?"
    # Broad set; no-tools-called cũng chấp nhận được (greetings)
    # activity_advice hợp lệ cho câu hỏi casual về hoạt động (cross-intent overlap)
    "smalltalk_weather": {
        "city":     ["get_current_weather", "get_clothing_advice", "get_comfort_index", "get_activity_advice"],
        "district": ["get_current_weather", "get_clothing_advice", "get_comfort_index", "get_activity_advice"],
        "ward":     ["get_current_weather", "get_clothing_advice", "get_comfort_index", "get_activity_advice"],
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

## Quy tắc chấm điểm:
CÁC TRƯỜNG HỢP CHẤP NHẬN ĐƯỢC (KHÔNG trừ điểm):
- Làm tròn số: 28.14°C → 28.1°C, 86.5% → 87% (sai lệch ≤ 0.5 đơn vị)
- Diễn giải weather_main sang tiếng Việt: "Clouds" → "nhiều mây", "Rain" → "mưa"
- Diễn giải mức gió: "2.5 m/s" → "gió nhẹ" (theo Beaufort)
- Đánh giá chung dựa trên số liệu: "trời mát" khi temp 22-26°C
- Trích xuất giá trị đúng từ forecast array cho đúng ngày/giờ được hỏi

CÁC TRƯỜNG HỢP SAI (PHẢI trừ điểm):
- Số liệu khác hẳn: tool nói 2.1 m/s nhưng chatbot nói 4.5 m/s (>100% sai lệch)
- Bịa ngày không có trong data: tool chỉ có 3 ngày nhưng chatbot trình bày 5 ngày
- Nhầm lẫn giá trị: dùng wind_gust thay cho wind_speed, hay avg_temp thay cho temp_max
- Tạo thông tin không có cơ sở: UV index, áp suất khi tool không trả về

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
                outputs.append(str(content)[:4000])
    return "\n---\n".join(outputs) if outputs else ""


def extract_detailed_tool_calls(result) -> list:
    """Extract detailed tool calls with name, input, output from agent result.

    Pairs AIMessage.tool_calls with ToolMessage responses via tool_call_id.
    Returns list of dicts: [{name, input, output}, ...]
    """
    messages = result.get("messages", [])

    # Map tool_call_id → output content
    tool_outputs = {}
    for msg in messages:
        if getattr(msg, "type", None) == "tool":
            tc_id = getattr(msg, "tool_call_id", "")
            content = getattr(msg, "content", "")
            if tc_id:
                tool_outputs[tc_id] = str(content)[:500]

    calls = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if isinstance(tc, dict):
                    name = tc.get("name")
                    args = tc.get("args", {})
                    tc_id = tc.get("id", "")
                else:
                    name = getattr(tc, "name", None)
                    args = getattr(tc, "args", {})
                    tc_id = getattr(tc, "id", "")
                if name:
                    calls.append({
                        "name": name,
                        "input": str(args)[:200],
                        "output": tool_outputs.get(tc_id, ""),
                    })
    return calls


def _get_expected_tools(intent: str, location_scope: str = "") -> list:
    """Get expected tools for (intent, location_scope) from hierarchical mapping.

    System supports 3 scopes: city / district / ward.
    POI queries (eval CSV may contain location_scope="poi") are resolved to
    "district" at runtime via dispatch.py — so use district tools for poi.
    Falls back to "city" if scope not found.
    Returns empty list for unknown intents.
    """
    intent_map = INTENT_TO_TOOLS.get(intent, {})
    if not intent_map:
        return []
    # POI → district: dispatch.py resolves all POI locations to district level
    effective_scope = "district" if location_scope == "poi" else location_scope
    return intent_map.get(effective_scope) or intent_map.get("city", [])


def check_tool_accuracy(intent: str, tools_called: list,
                        location_scope: str = "") -> bool:
    """Check if at least one scope-appropriate tool was called.

    Uses hierarchical INTENT_TO_TOOLS: the expected tools depend on BOTH the
    intent AND the location scope (city / district / ward).
    POI scope from eval CSV is mapped to district.
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

    Uses hierarchical INTENT_TO_TOOLS so that e.g. calling get_district_ranking
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
    - expected = ["get_current_weather", "detect_phenomena"]
    - tools_called = ["get_current_weather"] → recall = 1.0
    - tools_called = ["get_weather_history"] → recall = 0.0 (wrong tool for intent)
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
        # JUDGE_* takes priority; fallback to legacy API_* for backward compat
        client = OpenAI(
            base_url=os.getenv("JUDGE_API_BASE") or os.getenv("API_BASE"),
            api_key=os.getenv("JUDGE_API_KEY") or os.getenv("API_KEY"),
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
            tool_output=tool_output[:8000],
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
                   location_scope="", judge_client=None, skip_judge=False,
                   mode="baseline"):
    """Evaluate a single query with unique thread_id and optional LLM judge.

    Args:
        mode: "baseline" (27 tools), "routed" (SLM, no fallback), "hybrid" (SLM + fallback)
    """
    start_time = time.time()
    thread_id = f"eval_{query_id}_{uuid4().hex[:8]}"

    # Router metadata defaults
    _router_defaults = {
        "router_path": "", "router_intent": "", "router_scope": "",
        "router_confidence": None, "router_latency_ms": None,
        "router_focused_tools": "", "router_fallback_reason": "",
    }

    try:
        # Choose run function based on mode
        if mode == "routed":
            result = run_agent_routed(message=question, thread_id=thread_id,
                                      no_fallback=True)
        elif mode == "hybrid":
            result = run_agent_routed(message=question, thread_id=thread_id,
                                      no_fallback=False)
        else:  # baseline
            result = run_agent(message=question, thread_id=thread_id)

        # Extract router metadata if present
        router_info = result.pop("_router", None)

        messages = result.get("messages", [])
        response = messages[-1].content if messages else ""
        if response is None:
            response = ""
        elapsed_ms = (time.time() - start_time) * 1000

        tools_called = extract_tool_names(result)
        detailed_tool_calls = extract_detailed_tool_calls(result)
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
            "tool_output_raw": tool_output[:2000],
        }

        # Router metadata
        if router_info:
            eval_result["router_path"] = router_info.get("path", "")
            eval_result["router_intent"] = router_info.get("intent", "")
            eval_result["router_scope"] = router_info.get("scope", "")
            eval_result["router_confidence"] = router_info.get("confidence", 0)
            eval_result["router_latency_ms"] = round(router_info.get("latency_ms", 0), 1)
            eval_result["router_focused_tools"] = ",".join(router_info.get("focused_tools", []))
            eval_result["router_fallback_reason"] = router_info.get("fallback_reason") or ""
        else:
            eval_result.update(_router_defaults)

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

        # Attach detailed tool calls for logging (popped before CSV write)
        eval_result["_detailed_tool_calls"] = detailed_tool_calls
        return eval_result

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        error_str = str(e)
        err_result = {
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
        err_result.update(_router_defaults)
        return err_result


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

    # Router metrics (if available)
    routed_results = [r for r in results if r.get("router_path") == "routed"]
    fallback_results = [r for r in results if r.get("router_path") == "fallback"]
    has_router = bool(routed_results or fallback_results)

    if has_router:
        total_with_router = len(routed_results) + len(fallback_results)
        metrics["router_coverage"] = round(len(routed_results) / total_with_router * 100, 1)
        metrics["router_fallback_rate"] = round(len(fallback_results) / total_with_router * 100, 1)

        router_latencies = [r["router_latency_ms"] for r in results
                            if r.get("router_latency_ms") is not None]
        if router_latencies:
            metrics["router_avg_latency_ms"] = round(sum(router_latencies) / len(router_latencies), 1)

        # Router intent accuracy vs expected intent from CSV
        intent_correct = sum(1 for r in results
                             if r.get("router_intent") and r.get("router_intent") == r.get("intent"))
        intent_total = sum(1 for r in results if r.get("router_intent"))
        if intent_total:
            metrics["router_intent_accuracy"] = round(intent_correct / intent_total * 100, 1)
            metrics["router_intent_accuracy_ci95"] = wilson_ci(intent_correct, intent_total)

        # Routed-only tool accuracy (how accurate when using focused tools)
        routed_correct = sum(1 for r in routed_results if r.get("tool_correct"))
        if routed_results:
            metrics["routed_tool_accuracy"] = round(routed_correct / len(routed_results) * 100, 1)

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


def run_evaluation(output_dir="data/evaluation", skip_judge=False, mode="baseline"):
    """Run full evaluation pipeline.

    Args:
        output_dir: Base directory (contains eval questions CSV)
        skip_judge: Skip LLM-as-Judge evaluation
        mode: "baseline" (27 tools), "routed" (SLM, no fallback), "hybrid" (SLM + fallback)
    """
    # Results go to mode-specific subdirectory
    results_dir = Path(output_dir) / mode
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = get_evaluation_logger(str(results_dir))

    test_file = Path(output_dir) / "hanoi_weather_chatbot_eval_questions.csv"
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return

    queries = load_test_queries(str(test_file))
    print(f"Loaded {len(queries)} test queries")
    print(f"Mode: {mode}")

    # Initialize judge client once (reuse connection)
    judge_client = None
    if not skip_judge:
        try:
            from openai import OpenAI
            # JUDGE_* takes priority; fallback to legacy API_* for backward compat
            judge_client = OpenAI(
                base_url=os.getenv("JUDGE_API_BASE") or os.getenv("API_BASE"),
                api_key=os.getenv("JUDGE_API_KEY") or os.getenv("API_KEY"),
            )
            judge_model = os.getenv("JUDGE_MODEL") or os.getenv("MODEL", "gpt-4o-mini")
            print(f"LLM-as-Judge: ENABLED (model={judge_model})")
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
            mode=mode,
        )
        result["difficulty"] = q.get("difficulty", "unknown")
        results.append(result)

        # Print router info inline
        if mode != "baseline" and result.get("router_path"):
            rp = result["router_path"]
            ri = result.get("router_intent", "?")
            rs = result.get("router_scope", "?")
            rc = result.get("router_confidence", "?")
            rl = result.get("router_latency_ms", "?")
            print(f"  -> Router: {rp} ({ri}/{rs}, conf={rc}, {rl}ms)")

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

        # Log conversation (with tool names)
        tools_called_list = result.get("tools_called", "").split(",") if result.get("tools_called") else None
        logger.log_conversation(
            session_id=f"eval_{i}",
            turn_number=i,
            user_query=question,
            llm_response=result["response"][:500],
            response_time_ms=result["response_time_ms"],
            tool_calls=tools_called_list,
            error_type=result["error"],
        )

        # Log individual tool calls
        for tc in result.get("_detailed_tool_calls", []):
            logger.log_tool_call(
                session_id=f"eval_{i}",
                turn_number=i,
                tool_name=tc["name"],
                tool_input=tc["input"],
                tool_output=tc["output"],
                success=True,
            )

    # Remove internal key before saving to CSV
    for r in results:
        r.pop("_detailed_tool_calls", None)

    # Compute metrics
    metrics = compute_metrics(results)
    metrics["mode"] = mode

    # Print summary
    print()
    print("=" * 60)
    print(f"EVALUATION RESULTS (mode={mode})")
    print("=" * 60)
    print(f"Total: {metrics['total']}")
    print(f"Success rate: {metrics['success_rate']}%")
    ci = metrics['tool_accuracy_ci95']
    print(f"Tool accuracy: {metrics['tool_accuracy']}% [95% CI: {ci[0]}-{ci[1]}%]")
    rci = metrics['tool_recall_ci95']
    print(f"Tool precision: {metrics['tool_precision_avg']} | Tool recall: {metrics['tool_recall_avg']} [95% CI: {rci[0]}-{rci[1]}%]")
    print(f"Avg time: {metrics['avg_time_ms']}ms | p50: {metrics['p50_time_ms']}ms | p90: {metrics['p90_time_ms']}ms | p95: {metrics['p95_time_ms']}ms")

    # Router metrics
    if mode != "baseline" and metrics.get("router_coverage") is not None:
        print()
        print("Router Metrics:")
        print(f"  Coverage (routed): {metrics.get('router_coverage')}%")
        print(f"  Fallback rate: {metrics.get('router_fallback_rate')}%")
        print(f"  Avg router latency: {metrics.get('router_avg_latency_ms')}ms")
        ria = metrics.get('router_intent_accuracy')
        if ria is not None:
            ria_ci = metrics.get('router_intent_accuracy_ci95', (0, 0))
            print(f"  Router intent accuracy: {ria}% [95% CI: {ria_ci[0]}-{ria_ci[1]}%]")
        rta = metrics.get('routed_tool_accuracy')
        if rta is not None:
            print(f"  Routed-only tool accuracy: {rta}%")

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
    csv_file = results_dir / "evaluation_results.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    json_file = results_dir / "evaluation_summary.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nResults: {csv_file}")
    print(f"Summary: {json_file}")


def load_jsonl(path: str) -> list:
    """Load a JSONL file, one JSON object per line."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _evaluate_turn_mt(result: dict, turn_spec: dict, turn_index: int) -> dict:
    """Evaluate a single turn in a multi-turn conversation.

    Returns a dict with per-turn metrics.
    """
    tools_called = extract_tool_names(result)
    messages = result.get("messages", [])
    response = messages[-1].content if messages else ""
    if response is None:
        response = ""

    intent = turn_spec.get("expected_intent", "")
    scope = turn_spec.get("expected_scope", "city")
    expected_location = turn_spec.get("expected_location")
    requires_context = turn_spec.get("requires_context_from_turn") is not None

    tool_correct = check_tool_accuracy(intent, tools_called, scope)
    tool_precision = check_tool_precision(intent, tools_called, scope)

    # Entity Resolution Accuracy: for context-dependent turns,
    # check if expected location appears in response (proxy for correct entity resolution)
    entity_resolved = True
    if requires_context and expected_location:
        # Normalize: check if location name (or close variant) appears in response
        loc_lower = expected_location.lower()
        resp_lower = response.lower()
        entity_resolved = loc_lower in resp_lower

    return {
        "turn": turn_index,
        "user": turn_spec.get("user", ""),
        "expected_intent": intent,
        "expected_scope": scope,
        "expected_location": expected_location,
        "requires_context": requires_context,
        "tools_called": ",".join(tools_called),
        "tool_correct": tool_correct,
        "tool_precision": tool_precision,
        "entity_resolved": entity_resolved,
        "response_snippet": response[:200],
    }


def evaluate_multi_turn(
    scenarios_path: str = "data/evaluation/multi_turn_scenarios.jsonl",
    output_dir: str = "data/evaluation/multi_turn",
    mode: str = "routed",
    skip_judge: bool = True,
) -> dict:
    """Multi-turn evaluation framework.

    Metrics computed:
    - ERA  (Entity Resolution Accuracy): % context-dependent turns where entity resolved correctly
    - CSR  (Conversation Success Rate): % conversations where ALL turns had correct tool selection
    - CRR  (Context Retention Rate): avg consecutive correct turns before first context failure
    - Turn-level Tool Accuracy: weighted by turn index (later turns harder)

    Args:
        scenarios_path: Path to multi_turn_scenarios.jsonl
        output_dir: Output directory for results
        mode: Agent mode ("routed" recommended; "baseline" for comparison)
        skip_judge: Skip LLM-as-Judge (expensive for multi-turn)
    """
    from app.agent.agent import run_agent_routed, run_agent, reset_agent

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not Path(scenarios_path).exists():
        print(f"Scenarios file not found: {scenarios_path}")
        return {}

    conversations = load_jsonl(scenarios_path)
    print(f"Loaded {len(conversations)} conversations from {scenarios_path}")
    print(f"Mode: {mode}")

    # Per-conversation results
    conv_results = []
    all_turn_results = []

    for conv_idx, conv in enumerate(conversations, 1):
        thread_id = f"mt_eval_{conv['conversation_id']}_{uuid4().hex[:8]}"
        pattern = conv.get("pattern", "unknown")
        difficulty = conv.get("difficulty", "unknown")
        turns = conv.get("turns", [])

        print(f"\n[{conv_idx}/{len(conversations)}] {conv['conversation_id']} ({pattern}, {difficulty})")

        turn_metrics = []
        conv_success = True

        for turn_spec in turns:
            user_msg = turn_spec.get("user", "")
            turn_idx = turn_spec.get("turn", 0)
            print(f"  Turn {turn_idx}: {user_msg[:60]}...")

            try:
                if mode == "routed":
                    result = run_agent_routed(user_msg, thread_id, no_fallback=False)
                else:
                    result = run_agent(user_msg, thread_id)

                tm = _evaluate_turn_mt(result, turn_spec, turn_idx)
                tm["success"] = True
                tm["error"] = None

            except Exception as e:
                tm = {
                    "turn": turn_idx,
                    "user": user_msg,
                    "expected_intent": turn_spec.get("expected_intent", ""),
                    "expected_scope": turn_spec.get("expected_scope", "city"),
                    "expected_location": turn_spec.get("expected_location"),
                    "requires_context": turn_spec.get("requires_context_from_turn") is not None,
                    "tools_called": "",
                    "tool_correct": False,
                    "tool_precision": 0.0,
                    "entity_resolved": False,
                    "response_snippet": "",
                    "success": False,
                    "error": str(e)[:200],
                }

            if not tm["tool_correct"]:
                conv_success = False

            turn_metrics.append(tm)
            all_turn_results.append({
                **tm,
                "conversation_id": conv["conversation_id"],
                "pattern": pattern,
                "difficulty": difficulty,
            })

            # Print turn result
            tc = tm.get("tool_correct", False)
            er = tm.get("entity_resolved", True)
            tools = tm.get("tools_called", "")[:50]
            print(f"    tool_correct={tc}, entity_resolved={er}, tools=[{tools}]")

        # Compute Context Retention Rate for this conversation:
        # How many consecutive correct turns (from turn 0) before first failure?
        crr = 0
        for tm in turn_metrics:
            if tm["tool_correct"]:
                crr += 1
            else:
                break

        conv_results.append({
            "conversation_id": conv["conversation_id"],
            "pattern": pattern,
            "difficulty": difficulty,
            "num_turns": len(turns),
            "conversation_success": conv_success,
            "context_retention_rate": crr,
            "turn_tool_accuracy": round(
                sum(1 for t in turn_metrics if t["tool_correct"]) / len(turn_metrics) * 100, 1
            ) if turn_metrics else 0.0,
            "era": round(
                sum(1 for t in turn_metrics if t.get("requires_context") and t.get("entity_resolved", False))
                / max(1, sum(1 for t in turn_metrics if t.get("requires_context"))) * 100, 1
            ) if any(t.get("requires_context") for t in turn_metrics) else None,
        })

    # ── Aggregate Metrics ──
    total_convs = len(conv_results)
    total_turns = len(all_turn_results)

    csr = sum(1 for c in conv_results if c["conversation_success"]) / total_convs * 100 if total_convs else 0
    avg_crr = sum(c["context_retention_rate"] for c in conv_results) / total_convs if total_convs else 0

    context_turns = [t for t in all_turn_results if t.get("requires_context")]
    era = (
        sum(1 for t in context_turns if t.get("entity_resolved", False)) / len(context_turns) * 100
        if context_turns else 0
    )
    overall_tool_acc = (
        sum(1 for t in all_turn_results if t.get("tool_correct")) / total_turns * 100
        if total_turns else 0
    )

    # By pattern
    by_pattern: dict = {}
    for c in conv_results:
        p = c["pattern"]
        if p not in by_pattern:
            by_pattern[p] = {"total": 0, "success": 0}
        by_pattern[p]["total"] += 1
        if c["conversation_success"]:
            by_pattern[p]["success"] += 1
    for p in by_pattern:
        d = by_pattern[p]
        d["csr"] = round(d["success"] / d["total"] * 100, 1)

    metrics = {
        "mode": mode,
        "total_conversations": total_convs,
        "total_turns": total_turns,
        "CSR": round(csr, 1),           # Conversation Success Rate (%)
        "ERA": round(era, 1),            # Entity Resolution Accuracy (%)
        "avg_CRR": round(avg_crr, 2),    # Avg Context Retention Rate (turns)
        "overall_tool_accuracy": round(overall_tool_acc, 1),
        "context_dependent_turns": len(context_turns),
        "by_pattern": by_pattern,
    }

    # ── Print Summary ──
    print("\n" + "=" * 60)
    print(f"MULTI-TURN EVALUATION RESULTS (mode={mode})")
    print("=" * 60)
    print(f"Total conversations: {total_convs} | Total turns: {total_turns}")
    print(f"CSR  (Conversation Success Rate):   {metrics['CSR']}%")
    print(f"ERA  (Entity Resolution Accuracy):  {metrics['ERA']}%")
    print(f"avg_CRR (Avg context retention):    {metrics['avg_CRR']} turns")
    print(f"Overall turn-level tool accuracy:   {metrics['overall_tool_accuracy']}%")
    print()
    print("By Pattern:")
    for p, d in sorted(by_pattern.items()):
        print(f"  {p}: CSR={d['csr']}% ({d['success']}/{d['total']} convs)")

    # ── Save Results ──
    conv_csv = output_path / f"conv_results_{mode}.csv"
    with open(conv_csv, "w", newline="", encoding="utf-8") as f:
        if conv_results:
            writer = csv.DictWriter(f, fieldnames=conv_results[0].keys())
            writer.writeheader()
            writer.writerows(conv_results)

    turn_csv = output_path / f"turn_results_{mode}.csv"
    with open(turn_csv, "w", newline="", encoding="utf-8") as f:
        if all_turn_results:
            writer = csv.DictWriter(f, fieldnames=all_turn_results[0].keys())
            writer.writeheader()
            writer.writerows(all_turn_results)

    summary_json = output_path / f"summary_{mode}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate weather chatbot")
    parser.add_argument("--output", default="data/evaluation")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM-as-Judge evaluation (faster, no judge scores)")
    parser.add_argument("--mode", choices=["baseline", "routed", "hybrid"],
                        default="baseline",
                        help="baseline: 27 tools, no router | routed: SLM router, no fallback | hybrid: SLM router with fallback")
    parser.add_argument("--multi-turn", action="store_true",
                        help="Run multi-turn evaluation instead of single-turn")
    parser.add_argument("--mt-scenarios", default="data/evaluation/multi_turn_scenarios.jsonl",
                        help="Path to multi-turn scenarios JSONL")
    args = parser.parse_args()

    if args.multi_turn:
        evaluate_multi_turn(
            scenarios_path=args.mt_scenarios,
            output_dir=args.output + "/multi_turn",
            mode=args.mode,
            skip_judge=args.skip_judge,
        )
    else:
        run_evaluation(args.output, skip_judge=args.skip_judge, mode=args.mode)
