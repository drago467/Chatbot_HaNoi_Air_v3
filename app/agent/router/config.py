"""Router configuration — paths, thresholds, constants."""

import os
from pathlib import Path

# Project root
_ROOT = Path(__file__).resolve().parents[3]

# ── Model ──
GGUF_PATH = os.getenv(
    "SLM_MODEL_PATH",
    str(_ROOT / "model" / "router" / "Qwen2.5-1.5B-Instruct.Q8_0.gguf"),
)

# ── Ollama (primary inference backend) ──
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "hanoi-weather-router")

# ── Thresholds ──
CONFIDENCE_THRESHOLD = float(os.getenv("SLM_CONFIDENCE_THRESHOLD", "0.75"))

# ── Per-intent adaptive thresholds (Module 4: Calibrated Confidence Routing)
# Updated 2026-03-29 based on v3 confusion analysis:
# - weather_alert lowered: miss a storm warning is MORE dangerous than false alert
# - activity_weather lowered: no longer confused with smalltalk after data relabeling
# - current_weather vs temperature_query: both medium, let smart routing handle edge cases
PER_INTENT_THRESHOLDS: dict[str, float] = {
    "current_weather":      0.65,   # common intent, relax threshold
    "weather_overview":     0.65,   # typically easy to classify
    "smalltalk_weather":    0.60,   # highly distinctive
    "hourly_forecast":      0.70,   # good accuracy
    "daily_forecast":       0.70,   # good accuracy
    "rain_query":           0.70,   # improved after data relabeling
    "historical_weather":   0.72,   # moderate accuracy
    "humidity_fog_query":   0.75,   # default
    "seasonal_context":     0.75,   # default
    "wind_query":           0.70,   # improved: wind queries now properly labeled
    "location_comparison":  0.78,   # can confuse with current_weather
    "activity_weather":     0.70,   # improved: no longer confused with smalltalk
    "expert_weather_param": 0.78,   # slightly relaxed from 0.80
    "temperature_query":    0.75,   # relaxed from 0.80: smart routing handles overlap
    "weather_alert":        0.35,   # SAFETY: miss alert >> false positive
}

# ── Confidence calibration (Module 4: Temperature Scaling)
# Post-hoc temperature scaling: calibrated_conf = softmax(logits / T)
# T=1.0 means no calibration (identity). Update via scripts/router/calibrate.py
CALIBRATION_TEMPERATURE = float(os.getenv("SLM_CALIBRATION_TEMPERATURE", "1.0"))

# ── Calibration file path ──
CALIBRATION_FILE = _ROOT / "model" / "router" / "calibration.json"

# ── Feature flag ──
USE_SLM_ROUTER = os.getenv("USE_SLM_ROUTER", "false").lower() in ("true", "1", "yes")

# ── System prompt — Multi-task: routing + optional contextual rewriting ──
# When context is provided (turn > 0), model also outputs rewritten_query if needed.
# Training data format: {"input": ..., "context": null|{...}, "output": {...}}
ROUTER_SYSTEM_PROMPT = """Phân loại intent và scope cho câu hỏi thời tiết Hà Nội. Trả về JSON.

## Intents:
- current_weather: thời tiết NGAY LÚC NÀY (nhiệt độ, trời nắng/mưa, chung chung)
- hourly_forecast: diễn biến CHI TIẾT THEO GIỜ trong ngày (không chỉ hỏi mưa)
- daily_forecast: dự báo NHIỀU NGÀY tới (3 ngày, tuần tới, cuối tuần)
- weather_overview: TỔNG QUAN, tóm tắt thời tiết hôm nay/ngày mai (không hỏi thông số cụ thể)
- rain_query: hỏi CÓ MƯA KHÔNG, mưa bao lâu, xác suất mưa, mưa lúc nào tạnh
- temperature_query: hỏi CỤ THỂ VỀ NHIỆT ĐỘ (bao nhiêu độ, nóng/lạnh thế nào)
- wind_query: hỏi CỤ THỂ VỀ GIÓ (gió mạnh không, hướng gió, tốc độ gió)
- humidity_fog_query: hỏi về ĐỘ ẨM, SƯƠNG MÙ, sương muối
- historical_weather: thời tiết NGÀY/TUẦN TRƯỚC, dữ liệu QUÁ KHỨ
- location_comparison: SO SÁNH thời tiết giữa các quận/phường/địa điểm
- activity_weather: thời tiết có PHÙ HỢP ĐỂ LÀM hoạt động X không (chạy bộ, picnic, du lịch)
- expert_weather_param: thông số KỸ THUẬT chuyên sâu (áp suất, tia UV, điểm sương, tầm nhìn)
- weather_alert: CẢNH BÁO, nguy hiểm, bão, ngập, thay đổi đột ngột
- seasonal_context: SO SÁNH với hôm qua/tuần trước, xu hướng, bất thường theo MÙA
- smalltalk_weather: chào hỏi, ngoài phạm vi (không phải Hà Nội), câu hỏi không liên quan thời tiết

## Scopes:
- city: toàn Hà Nội, hoặc KHÔNG NÓI RÕ địa điểm
- district: nhắc tên QUẬN/HUYỆN (Ba Đình, Cầu Giấy...) hoặc ĐỊA ĐIỂM NỔI TIẾNG thuộc quận (Hồ Gươm→Hoàn Kiếm, Lăng Bác→Ba Đình, Nội Bài→Sóc Sơn...)
- ward: nhắc tên PHƯỜNG/XÃ (Phường Dịch Vọng Hậu, Xã Tiên Dương...)

## Multi-task output (khi có context từ lượt trước):
Nếu được cung cấp context (location/intent từ lượt trước) VÀ câu hỏi hiện tại thiếu địa điểm hoặc dùng đại từ (ở đó, thế còn, còn ..., vậy ...), hãy điền thêm trường "rewritten_query" với câu hỏi đầy đủ ngữ cảnh.

## Output format:
Khi không có context hoặc câu hỏi đầy đủ:
{"intent": "...", "scope": "...", "confidence": 0.92}

Khi có context và cần rewrite:
{"intent": "...", "scope": "...", "confidence": 0.91, "rewritten_query": "Dự báo thời tiết ngày mai ở quận Cầu Giấy như thế nào?"}"""

# ── Intents + Scopes (must match training data) ──
VALID_INTENTS = [
    "current_weather", "hourly_forecast", "daily_forecast", "weather_overview",
    "rain_query", "temperature_query", "wind_query", "humidity_fog_query",
    "historical_weather", "location_comparison", "activity_weather",
    "expert_weather_param", "weather_alert", "seasonal_context", "smalltalk_weather",
]
VALID_SCOPES = ["city", "district", "ward"]

# ── Convenience dict ──
ROUTER_CONFIG = {
    "gguf_path": GGUF_PATH,
    "ollama_base_url": OLLAMA_BASE_URL,
    "ollama_model": OLLAMA_MODEL,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "use_slm_router": USE_SLM_ROUTER,
}
