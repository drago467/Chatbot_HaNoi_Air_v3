"""LangGraph Agent for Weather Chatbot."""

import logging
import os
import threading
from dotenv import load_dotenv

load_dotenv()

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_openai import ChatOpenAI
import psycopg

from app.agent.tools import TOOLS
from app.dal.timezone_utils import now_ict

logger = logging.getLogger(__name__)

# Vietnamese weekday names
_WEEKDAYS_VI = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]

# ═══════════════════════════════════════════════════════════════
# System Prompt — Modular Architecture
# BASE_PROMPT: luôn có (~65 dòng) — context chung cho mọi agent
# TOOL_RULES: per-tool rules — chỉ gửi khi tool đó được focused
# SYSTEM_PROMPT_TEMPLATE: full prompt cho fallback agent (25 tools)
# ═══════════════════════════════════════════════════════════════

BASE_PROMPT_TEMPLATE = """Bạn là trợ lý thời tiết chuyên về Hà Nội. CHỈ trả lời về thời tiết khu vực Hà Nội.
Phong cách: thân thiện, chuyên nghiệp, ngắn gọn, dùng tiếng Việt tự nhiên có dấu.

## Thời gian hiện tại
Hôm nay là: {today_weekday}, ngày {today_date} | Giờ hiện tại: {today_time} ICT (UTC+7)
→ LUÔN dùng thông tin này khi tính "hôm qua", "tuần này", "cuối tuần", "3 ngày tới", v.v.
→ KHÔNG BAO GIỜ tự suy đoán ngày tháng. Chỉ dùng ngày ở trên.

## 30 quận/huyện Hà Nội (TẤT CẢ đều thuộc Hà Nội)
Nội thành: Ba Đình, Hoàn Kiếm, Hai Bà Trưng, Đống Đa, Tây Hồ, Cầu Giấy, Thanh Xuân, Hoàng Mai, Long Biên, Bắc Từ Liêm, Nam Từ Liêm, Hà Đông
Ngoại thành: Sóc Sơn, Đông Anh, Gia Lâm, Thanh Trì, Mê Linh, Sơn Tây, Ba Vì, Phúc Thọ, Đan Phượng, Hoài Đức, Quốc Oai, Thạch Thất, Chương Mỹ, Thanh Oai, Thường Tín, Phú Xuyên, Ứng Hòa, Mỹ Đức
→ Khi user hỏi về BẤT KỲ quận/huyện nào ở trên → ĐÂY LÀ HÀ NỘI, PHẢI gọi tool.

## Quy ước thời gian (ICT = UTC+7)
- "sáng" = 6h-11h, "trưa" = 11h-13h, "chiều" = 13h-18h, "tối" = 18h-22h, "đêm" = 22h-6h
- "cuối tuần" = Thứ 7 + Chủ nhật tuần này (hoặc tuần tới nếu đã qua)
- "tuần này" = từ hôm nay đến Chủ nhật

## Địa điểm nổi tiếng (POI)
Hỗ trợ: Hồ Gươm, Mỹ Đình, Hồ Tây, Sân bay Nội Bài, Times City, Văn Miếu, Lăng Bác, Royal City, Keangnam, Cầu Long Biên, Phố cổ... Hệ thống tự động map về quận/huyện.

## QUY TẮC VÀNG — TUYỆT ĐỐI KHÔNG BỊA DỮ LIỆU
- CHỈ báo cáo số liệu CHÍNH XÁC từ tool trả về. KHÔNG tự tính, nội suy, ước lượng.
- Nếu tool trả data cho 2 ngày mà user hỏi 5 ngày → NÓI RÕ "Hiện chỉ có dữ liệu cho ngày X-Y", CHỈ trình bày data có sẵn. TUYỆT ĐỐI KHÔNG bịa thêm ngày.
- KHÔNG tự tạo min/max, trung bình, xu hướng nếu tool không trả về rõ ràng.
- Nếu tool không trả UV/áp suất → KHÔNG đề cập thông số đó.
- Khi trình bày dự báo nhiều ngày: TỪNG NGÀY phải lấy đúng số liệu từ forecasts array tương ứng.
  KHÔNG dùng số liệu ngày A cho ngày B. KHÔNG bịa thêm ngày không có trong data.
- Nếu response có con số, con số đó PHẢI tồn tại trong tool output. Làm tròn 1 chữ số thập phân là OK.
- LUÔN ghi rõ phạm vi: "Theo dự báo 24h tới...", "Dữ liệu ngày 27/03..."
- Chú ý field "data_coverage" trong kết quả tool — dùng nó để giới thiệu phạm vi dữ liệu.
- **PHÂN BIỆT RÕ wind_speed và wind_gust**: wind_speed là tốc độ gió trung bình, wind_gust là gió giật (cao hơn nhiều). KHÔNG nhầm lẫn hoặc trộn 2 giá trị này.
- Khi tool trả cả current + forecast: BÁO CÁO ĐÚNG section. Ví dụ hỏi "hiện tại" → dùng current, hỏi "tối nay" → dùng forecast giờ tương ứng. KHÔNG trộn lẫn.

## Lưu ý về dữ liệu
- Dữ liệu HIỆN TẠI không có pop → check weather_main + gọi thêm get_hourly_forecast
- rain_1h chỉ có khi đang mưa. wind_gust có thể NULL khi gió nhẹ.
- Dự báo giờ: tối đa 48h. Dự báo ngày: tối đa 8 ngày. Lịch sử: 14 ngày gần nhất.
- Dữ liệu thiếu/lỗi: thông báo rõ ràng, gợi ý khu vực/thời gian khác.
- Khi user hỏi giờ cụ thể (VD "7h sáng mai") mà tool không có data cho giờ đó → NÓI "không có dữ liệu cho giờ đó", KHÔNG đoán.

## Hiện tượng đặc biệt Hà Nội
- Nồm ẩm: Tháng 2-4, độ ẩm > 85%, điểm sương - nhiệt <= 2°C
- Gió mùa Đông Bắc: Tháng 10-3, gió Bắc/Đông Bắc
- Rét đậm: Tháng 11-3, nhiệt < 15°C, mây > 70%

## Định dạng số liệu
- Nhiệt: 1 decimal °C | Gió: 1 decimal m/s | Ẩm: % nguyên | Áp suất: hPa nguyên | UV: 1 decimal
- Luôn kèm đơn vị, hướng gió tiếng Việt (Đông Bắc, Tây Nam...)

## Định dạng trả lời
- Cho quận/thành phố: tổng quan + nổi bật + hiện tượng đặc biệt
- Cho phường: chi tiết đầy đủ các thông số
- Luôn kèm khuyến nghị thực tế (mang ô, áo khoác, tránh giờ nào...)
- Dùng bullet points khi nhiều thông tin

## Hội thoại nhiều lượt
- "ở đó thế nào?" → dùng địa điểm lượt trước
- "còn ngày mai?" → giữ địa điểm, đổi thời gian
- User hỏi chung chung không rõ địa điểm → mặc định Hà Nội (city-level)
- Nếu context không rõ và cần chính xác → có thể hỏi lại khu vực cụ thể

## Xử lý lỗi
- Tool trả error → KHÔNG retry cùng tool với cùng tham số. Giải thích rõ + gợi ý thay thế.
- KHÔNG gọi cùng 1 tool quá 3 lần. Error 2 lần → dừng, thông báo user.
- Không tìm thấy địa điểm → gợi ý: "quận Cầu Giấy, phường Dịch Vọng"
- Không có dữ liệu → nêu rõ giới hạn, gợi ý thời gian/khu vực khác

## Khi KHÔNG gọi tool
- Lời chào → trả lời thân thiện, giới thiệu bản thân là trợ lý thời tiết Hà Nội
- Câu hỏi về chatbot → trả lời trực tiếp
- Cảm ơn, tạm biệt → đáp lại lịch sự
- Thời tiết NGOÀI Hà Nội → "Mình chỉ hỗ trợ khu vực Hà Nội"
- LƯU Ý: Nhắc đến Hà Nội/quận/huyện → PHẢI gọi tool, KHÔNG từ chối
"""

# ── Tool-specific rules: chỉ gửi cho focused agent khi tool đó được chọn ──
TOOL_RULES = {
    "get_current_weather": """- "bây giờ", "hiện tại", "đang" → get_current_weather
- Hỗ trợ mọi cấp: phường, quận, toàn Hà Nội (tự dispatch theo location)
- Cho quận: tổng quan + nổi bật + hiện tượng đặc biệt
- Cho toàn Hà Nội: dữ liệu aggregated từ tất cả quận""",

    "get_hourly_forecast": """- "chiều nay", "tối nay", "3 giờ nữa", "sáng mai" → dự báo theo giờ
- Tối đa 48h. Xa hơn → thông báo giới hạn, gợi ý dùng dự báo theo ngày""",

    "get_daily_forecast": """- "ngày mai", "hôm nay" cả ngày, "tuần này", "3 ngày tới" → dự báo theo ngày
- Hỗ trợ mọi cấp: phường, quận, toàn Hà Nội
- Tối đa 8 ngày. Xa hơn → thông báo giới hạn, cung cấp data có sẵn""",

    "get_daily_summary": """- Tổng hợp 1 ngày: min/max/avg các thông số
- Hỗ trợ mọi cấp: phường (chi tiết nhất), quận, toàn Hà Nội""",

    "get_weather_history": """- "hôm qua", "tuần trước" → lịch sử thời tiết
- Giới hạn: chỉ có 14 ngày gần nhất. Xa hơn → thông báo giới hạn.
- Hỗ trợ: phường, quận, toàn Hà Nội""",

    "get_rain_timeline": """- "mưa đến bao giờ", "mấy giờ tạnh", "khi nào mưa" → timeline mưa
- Trả về: rain_periods (start/end/max_pop), next_rain, next_clear
- Hỗ trợ: phường, quận, toàn Hà Nội""",

    "get_best_time": """- "mấy giờ tốt nhất", "lúc nào nên đi" → thời điểm tốt nhất cho hoạt động
- Hỗ trợ: phường, quận, toàn Hà Nội""",

    "get_clothing_advice": """- "mặc gì", "cần áo khoác không", "mang ô không" → tư vấn trang phục
- Hỗ trợ mọi cấp: phường, quận, toàn Hà Nội""",

    "get_temperature_trend": """- "ấm lên khi nào", "xu hướng nhiệt", "bao giờ hết rét" → xu hướng nhiệt độ
- Hỗ trợ mọi cấp: phường, quận, toàn Hà Nội""",

    "get_seasonal_comparison": """- "nóng hơn bình thường không", "dạo này", "mùa này" → so sánh với trung bình mùa
- LUÔN gọi tool ngay cả khi câu hỏi mang tính chuyện phiếm ("nhỉ?", "quá!", "thật không?")
  Ví dụ: "Trời dạo này khó chịu quá nhỉ?" → GỌI get_seasonal_comparison, trả lời dựa trên data
- Nếu error "no_weather_data" → thông báo, gợi ý hỏi thời tiết hiện tại""",

    "get_activity_advice": """- "đi chơi được không", "chạy bộ được không" → tư vấn hoạt động
- 15 loại: chay_bo, picnic, bike, chup_anh, du_lich, cam_trai, cau_ca, lam_vuon, boi_loi, leo_nui, di_dao, su_kien, dua_dieu, tap_the_duc, phoi_do""",

    "get_comfort_index": """- "thoải mái không", "dễ chịu không", "ra ngoài được không" → chỉ số thoải mái
- Cũng phục vụ: wind chill, heat index, chỉ số cảm giác lạnh/nóng""",

    "get_weather_change_alert": """- "trời có thay đổi không", "có chuyển mưa không" → cảnh báo thay đổi thời tiết""",

    "get_weather_alerts": """- "cảnh báo", "nguy hiểm", "giông lốc", "bão", "lũ", "ngập", "rét hại", "nắng nóng gay gắt" → cảnh báo thời tiết
- Câu hỏi về ngập lụt, tầm nhìn, gió giật → ĐÂY LÀ thời tiết""",

    "detect_phenomena": """- "nồm ẩm", "gió mùa", "sương mù", "hiện tượng đặc biệt" → phát hiện hiện tượng
- Hỗ trợ mọi level: phường/quận/thành phố""",

    "compare_weather": """- "A và B nơi nào nóng/lạnh/ẩm hơn?" → compare_weather(location_hint1="A", location_hint2="B")
- BẮT BUỘC dùng compare_weather. KHÔNG gọi get_current_weather 2 lần riêng lẻ.""",

    "compare_with_yesterday": """- "hôm nay so với hôm qua", "thay đổi so với hôm qua", "mấy hôm nay hay thay đổi" → so sánh với hôm qua
- LUÔN gọi tool khi user nhận xét về thay đổi thời tiết gần đây
- Hỗ trợ: phường, quận, toàn Hà Nội
- Nếu error "not_enough_data" → thông báo, gợi ý xem thời tiết hiện tại""",

    "get_district_ranking": """- "quận nào nóng nhất", "top", "xếp hạng" → xếp hạng quận
- Metrics: nhiet_do, do_am, gio, mua, uvi, ap_suat, diem_suong, may""",

    "get_ward_ranking_in_district": """- "phường nào trong quận X nóng nhất" → xếp hạng phường trong quận""",

    "get_weather_period": """- "tuần này", "3 ngày tới", "cuối tuần" → thời tiết theo khoảng thời gian""",

    # ── 6 NEW insight tools ──
    "get_uv_safe_windows": """- "lúc nào ra ngoài an toàn?", "UV thấp lúc mấy giờ?", "giờ nào nên đi bộ?"
- Trả về: khung giờ UV < ngưỡng, peak_uv_time, summary
- Hỗ trợ: phường, quận, toàn Hà Nội""",

    "get_pressure_trend": """- "áp suất thay đổi?", "có front thời tiết?", "áp suất giảm mạnh?"
- Phát hiện: front lạnh (áp suất giảm >3 hPa/3h), khí áp thấp
- Hỗ trợ: phường, quận, toàn Hà Nội""",

    "get_daily_rhythm": """- "sáng nay mấy độ?", "chiều nay nóng không?", "tối mát chưa?"
- Chia ngày thành 4 khung: sáng (6-10h), trưa (10-14h), chiều (14-18h), tối (18-22h)
- Hỗ trợ: phường, quận, toàn Hà Nội""",

    "get_humidity_timeline": """- "độ ẩm thay đổi?", "khi nào hết nồm?", "điểm sương?"
- Phát hiện: nom ẩm (humidity ≥85% AND temp-dew_point ≤2°C)
- Hỗ trợ: phường, quận, toàn Hà Nội""",

    "get_sunny_periods": """- "khi nào có nắng?", "lúc nào trời quang?", "có nắng phơi đồ?"
- Nắng = mây <40%, pop <30%, không mưa
- Hỗ trợ: phường, quận, toàn Hà Nội""",

    "get_district_multi_compare": """- "tổng hợp thời tiết các quận", "quận nào thoải mái nhất?"
- So sánh nhiều chỉ số cùng lúc: nhiệt độ, độ ẩm, UV, gió, mưa, áp suất""",
}

# ── Full prompt cho fallback agent (25 tools) — giữ tool selection rules ──
SYSTEM_PROMPT_TEMPLATE = BASE_PROMPT_TEMPLATE + """
## Quy tắc chọn tool (tất cả tool đều hỗ trợ 3 cấp: phường/quận/toàn Hà Nội)
- "bây giờ", "hiện tại", "đang" → get_current_weather (tự dispatch theo cấp)
- "chiều nay", "tối nay", "3 giờ nữa", "sáng mai" → get_hourly_forecast
- "sáng nay mấy độ", "chiều nóng không" → get_daily_rhythm
- "ngày mai", "hôm nay" (cả ngày) → get_daily_summary
- "tuần này", "3 ngày tới", "cuối tuần" → get_weather_period
- "hôm qua", "tuần trước" → get_weather_history
- "quận nào nóng nhất", "top", "xếp hạng" → get_district_ranking
- "phường nào trong quận X" → get_ward_ranking_in_district
- "so sánh tổng hợp các quận" → get_district_multi_compare
- "mưa đến bao giờ", "mấy giờ tạnh", "khi nào mưa" → get_rain_timeline
- "khi nào có nắng", "trời quang lúc nào" → get_sunny_periods
- "mấy giờ tốt nhất", "lúc nào nên" → get_best_time
- "UV an toàn lúc nào", "giờ nào ra ngoài" → get_uv_safe_windows
- "mặc gì", "cần áo khoác không", "mang ô không" → get_clothing_advice
- "ấm lên khi nào", "xu hướng nhiệt", "bao giờ hết rét" → get_temperature_trend
- "áp suất thay đổi?", "có front thời tiết?" → get_pressure_trend
- "khi nào hết nồm", "độ ẩm thay đổi" → get_humidity_timeline
- "nóng hơn bình thường không" → get_seasonal_comparison
- "đi chơi được không", "chạy bộ được không" → get_activity_advice
- "thoải mái không", "dễ chịu không", "ra ngoài được không" → get_comfort_index
- "trời có thay đổi không", "có chuyển mưa không" → get_weather_change_alert

### So sánh hai địa điểm → BẮT BUỘC dùng compare_weather
- "A và B nơi nào nóng/lạnh/ẩm hơn?" → compare_weather(location_hint1="A", location_hint2="B")
- KHÔNG gọi get_current_weather 2 lần riêng lẻ khi so sánh. PHẢI dùng compare_weather.

### Cảnh báo thời tiết → get_weather_alerts + detect_phenomena + get_pressure_trend
- "cảnh báo", "nguy hiểm", "giông lốc", "bão", "lũ", "ngập" → get_weather_alerts
- "nồm ẩm", "gió mùa", "sương mù" → detect_phenomena + get_humidity_timeline
- "trời có thay đổi gì", "sắp mưa" → get_weather_change_alert + get_pressure_trend

### Insight tools mới (hỗ trợ 3 cấp: phường/quận/TP)
- "UV an toàn lúc nào", "giờ nào ra ngoài an toàn" → get_uv_safe_windows
- "áp suất thay đổi", "có front thời tiết" → get_pressure_trend
- "sáng nay mấy độ", "chiều nóng không", "nhịp nhiệt trong ngày" → get_daily_rhythm
- "khi nào hết nồm", "độ ẩm thay đổi thế nào", "điểm sương" → get_humidity_timeline
- "khi nào có nắng", "trời quang lúc nào" → get_sunny_periods
- "tổng hợp so sánh các quận", "quận nào thoải mái nhất" → get_district_multi_compare

## Khi cần gọi nhiều tool
- "Thời tiết Hà Nội hôm nay" → get_current_weather + get_district_ranking(nhiet_do)
- "Có nên đi chơi không" → get_best_time + get_clothing_advice + get_uv_safe_windows
- "Quận Cầu Giấy thời tiết thế nào" → get_current_weather + get_ward_ranking_in_district
- "Ra ngoài có ổn không" → get_comfort_index + get_clothing_advice + get_uv_safe_windows
- "Có nồm ẩm không" → detect_phenomena + get_humidity_timeline

## Ví dụ câu trả lời tốt

Câu hỏi: "Bây giờ thời tiết Cầu Giấy thế nào?"
→ Gọi get_current_weather, trả lời:
"Quận Cầu Giấy hiện tại: 28.5°C (cảm giác 31°C), trời có mây, độ ẩm 75%.
Gió Đông Nam 2.3 m/s, UV 5.2 (trung bình).
💡 Trời oi bức, nên mang nước khi ra ngoài."

Câu hỏi: "Chiều nay có mưa không?"
→ Gọi get_rain_timeline, trả lời:
"Theo dự báo, chiều nay (13h-18h) xác suất mưa 65%, cao nhất lúc 15h (80%).
Mưa có thể kéo dài 2-3 tiếng. Nên mang ô khi ra ngoài."
"""


def _inject_datetime(template: str) -> str:
    """Inject current date/time into a prompt template."""
    now = now_ict()
    return template.format(
        today_weekday=_WEEKDAYS_VI[now.weekday()],
        today_date=now.strftime("%d/%m/%Y"),
        today_time=now.strftime("%H:%M"),
    )


def get_system_prompt() -> str:
    """Build full system prompt (25 tools) with current date/time injected."""
    return _inject_datetime(SYSTEM_PROMPT_TEMPLATE)


def get_focused_system_prompt(tool_names: list) -> str:
    """Build focused prompt: BASE + only rules for given tools.

    Used by focused agents (1-2 tools) after SLM routing.
    Significantly shorter than full prompt — reduces confusion and tokens.
    """
    base = _inject_datetime(BASE_PROMPT_TEMPLATE)

    # Collect tool-specific rules
    rules = []
    for name in tool_names:
        rule = TOOL_RULES.get(name)
        if rule:
            rules.append(rule.strip())

    if rules:
        return base + "\n## Hướng dẫn sử dụng công cụ\n" + "\n".join(rules)
    return base


def _prompt_with_datetime(state) -> list:
    """LangGraph prompt callable: full prompt for 25-tool agent."""
    from langchain_core.messages import SystemMessage
    system_msg = SystemMessage(content=get_system_prompt())
    return [system_msg] + state["messages"]


def _focused_prompt_callable(tool_names: list):
    """Return a state_modifier callable for focused agent with dynamic prompt."""
    def modifier(state) -> list:
        from langchain_core.messages import SystemMessage
        prompt = get_focused_system_prompt(tool_names)
        return [SystemMessage(content=prompt)] + state["messages"]
    return modifier

# Thread-safe agent cache
_agent = None
_agent_lock = threading.Lock()
_db_connection = None
_model = None         # Shared ChatOpenAI instance (reused by focused agents)
_checkpointer = None  # Shared PostgresSaver (reused by focused agents)


def get_agent():
    """Get or create the weather agent (thread-safe)."""
    global _agent
    if _agent is None:
        with _agent_lock:
            # Double-check after acquiring lock
            if _agent is None:
                _agent = create_weather_agent()
    return _agent


def reset_agent():
    """Reset the cached agent to force recreation with fresh connections."""
    global _agent, _db_connection, _model, _checkpointer
    with _agent_lock:
        if _db_connection is not None:
            try:
                _db_connection.close()
            except:
                pass
            _db_connection = None
        _agent = None
        _model = None
        _checkpointer = None


def create_weather_agent():
    global _model, _checkpointer, _db_connection

    API_BASE = os.getenv("API_BASE")
    API_KEY = os.getenv("API_KEY")
    MODEL_NAME = os.getenv("MODEL", "gpt-4o-2024-11-20")

    if not API_BASE or not API_KEY:
        raise ValueError("API_BASE and API_KEY must be set in .env")

    _model = ChatOpenAI(model=MODEL_NAME, temperature=0, base_url=API_BASE, api_key=API_KEY)

    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL must be set in .env")

    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    _checkpointer = PostgresSaver(conn)
    _checkpointer.setup()
    _db_connection = conn

    agent = create_react_agent(
        model=_model, tools=TOOLS,
        state_modifier=_prompt_with_datetime, checkpointer=_checkpointer,
    )
    return agent

def run_agent(message: str, thread_id: str = "default") -> dict:
    """Run agent synchronously (blocking).
    
    Also logs tool calls to evaluation_logger.
    Includes automatic retry on connection errors.
    """
        
    # Get logger
    try:
        from app.agent.evaluation_logger import get_evaluation_logger
        logger = get_evaluation_logger()
    except Exception:
        logger = None
    
    agent = get_agent()
    config = {"configurable": {"thread_id": thread_id}}
    
    # Wrap tools to log calls (if logger available)
    if logger:
        # We'll log after getting results
        pass
    
    # Retry logic for stale connections
    max_retries = 2
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config)
            break  # Success, exit retry loop
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # Reset agent to get fresh connection
                reset_agent()
                agent = get_agent()
            else:
                raise last_error
    
    # Extract and log tool calls from result
    if logger:
        try:
            messages = result.get("messages", [])
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        logger.log_tool_call(
                            session_id=thread_id,
                            turn_number=0,
                            tool_name=tc.get("name", "unknown"),
                            tool_input=str(tc.get("args", {}))[:200],
                            tool_output="",
                            success=True,
                            execution_time_ms=0
                        )
        except Exception as e:
            pass  # Don't break on logging errors
    
    return result


def stream_agent(message: str, thread_id: str = "default"):
    """Stream agent response token by token.
    
    Yields chunks of the response for real-time display.
    Only yields LLM text (AIMessageChunk from node "agent").
    
    Includes automatic retry on connection errors.
    
    Args:
        message: User message
        thread_id: Conversation thread ID
        
    Yields:
        Text chunks from the agent's response
    """
    from langchain_core.messages import ToolMessage, AIMessageChunk
    
    max_retries = 2
    last_error = None
    
    for attempt in range(max_retries):
        try:
            agent = get_agent()
            config = {"configurable": {"thread_id": thread_id}}
            
            # Stream with "messages" mode to get token-by-token updates
            for event in agent.stream(
                {"messages": [{"role": "user", "content": message}]},
                config,
                stream_mode="messages"
            ):
                # event is a tuple of (message_chunk, metadata)
                if event and len(event) >= 2:
                    msg_chunk, metadata = event
                    
                    # Skip tool messages (they contain raw JSON from DAL)
                    if isinstance(msg_chunk, ToolMessage):
                        continue
                    
                    # Skip messages with tool_calls (function calling JSON)
                    if hasattr(msg_chunk, "tool_calls") and msg_chunk.tool_calls:
                        continue
                    
                    # Only yield content from agent node, not tools node
                    if metadata.get("langgraph_node") == "agent":
                        if hasattr(msg_chunk, "content") and msg_chunk.content:
                            yield msg_chunk.content
            return  # Success, exit function
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # Reset agent to get fresh connection
                reset_agent()
            else:
                raise last_error


def stream_agent_with_updates(message: str, thread_id: str = "default"):
    """Stream agent response with both messages and tool updates.
    
    Yields dict with 'type' and 'content' keys:
    - type='message': text chunk from LLM
    - type='tool': tool call start/update/end
    
    Also logs tool calls to evaluation_logger.
    
    Args:
        message: User message
        thread_id: Conversation thread ID
        
    Yields:
        Dict with type and content
    """
    from langchain_core.messages import ToolMessage, AIMessageChunk
        
    # Retry logic for stale connections
    max_retries = 2
    last_error = None
    
    for attempt in range(max_retries):
        try:
            agent = get_agent()
            config = {"configurable": {"thread_id": thread_id}}
            
            # Get logger
            try:
                from app.agent.evaluation_logger import get_evaluation_logger
                logger = get_evaluation_logger()
            except Exception:
                logger = None
            
            # Stream with both messages and updates
            for event in agent.stream(
                {"messages": [{"role": "user", "content": message}]},
                config,
                stream_mode=["messages", "updates"]
            ):
                # Handle different event formats from LangGraph
                # When stream_mode is a list, events come as (stream_name, event_data)
                if isinstance(event, tuple) and len(event) == 2:
                    stream_name, event_data = event
                    
                    if stream_name == "messages":
                        # event_data is (chunk, metadata)
                        if isinstance(event_data, tuple) and len(event_data) == 2:
                            msg_chunk, metadata = event_data
                            
                            # Skip tool messages (raw JSON from DAL)
                            if isinstance(msg_chunk, ToolMessage):
                                continue
                            
                            # Skip messages with tool_calls (function calling JSON)
                            if hasattr(msg_chunk, "tool_calls") and msg_chunk.tool_calls:
                                continue
                            
                            # Message chunk from agent node
                            if metadata.get("langgraph_node") == "agent":
                                if hasattr(msg_chunk, "content") and msg_chunk.content:
                                    yield {"type": "message", "content": msg_chunk.content}
                            
                            # Tool updates (from tools node)
                            if metadata.get("langgraph_node") == "tools":
                                yield {"type": "tool", "content": msg_chunk if isinstance(msg_chunk, str) else str(msg_chunk)}
                    
                    elif stream_name == "updates":
                        # event_data is dict with tool outputs
                        yield {"type": "tool", "content": event_data}

            return  # Success, exit function

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # Reset agent to get fresh connection
                reset_agent()
            else:
                raise last_error


# ═══════════════════════════════════════════════════════════════
# SLM Router — Focused ReAct Agent (1-2 tools instead of 25)
# ═══════════════════════════════════════════════════════════════


def _create_focused_agent(tools: list):
    """Create a ReAct agent with focused tool set and dynamic prompt.

    Uses focused system prompt (BASE + only relevant tool rules)
    instead of full 25-tool prompt — reduces confusion and tokens.
    """
    get_agent()  # ensure _model and _checkpointer are initialized
    tool_names = [t.name for t in tools]
    return create_react_agent(
        model=_model,
        tools=tools,
        state_modifier=_focused_prompt_callable(tool_names),
        checkpointer=_checkpointer,
    )


def stream_agent_routed(message: str, thread_id: str = "default"):
    """Stream agent response with SLM routing.

    Pipeline:
    1. SLM Router classifies intent + scope (~50-100ms)
    2. If confident → create focused ReAct agent with 1-2 tools
    3. If not confident → fallback to full 25-tool agent

    Yields text chunks (same interface as stream_agent).
    """
    from langchain_core.messages import ToolMessage

    from app.agent.router.config import USE_SLM_ROUTER
    from app.agent.router.slm_router import get_router
    from app.agent.router.tool_mapper import get_focused_tools

    # If router disabled, use standard path
    if not USE_SLM_ROUTER:
        yield from stream_agent(message, thread_id)
        return

    # Step 1: Classify
    router = get_router()
    result = router.classify(message)
    logger.info("SLM Router: %s", result)

    # Step 2: Decide path
    if result.should_fallback:
        logger.info("SLM Router → fallback (%s)", result.fallback_reason)
        yield from stream_agent(message, thread_id)
        return

    # Step 3: Get focused tools
    focused_tools = get_focused_tools(result.intent, result.scope)

    # smalltalk or unknown mapping → fallback
    if focused_tools is None or (not focused_tools and result.intent != "smalltalk_weather"):
        logger.info("SLM Router → fallback (no tool mapping for %s/%s)", result.intent, result.scope)
        yield from stream_agent(message, thread_id)
        return

    # smalltalk_weather → no tools, let LLM respond directly
    if not focused_tools:
        focused_tools = []

    logger.info(
        "SLM Router → fast path: %s/%s, %d tools: %s",
        result.intent,
        result.scope,
        len(focused_tools),
        [t.name for t in focused_tools],
    )

    # Step 4: Create focused agent and stream
    max_retries = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            focused_agent = _create_focused_agent(focused_tools)
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 10,  # Focused agent: max 10 steps (prevents tool loops)
            }

            for event in focused_agent.stream(
                {"messages": [{"role": "user", "content": message}]},
                config,
                stream_mode="messages",
            ):
                if event and len(event) >= 2:
                    msg_chunk, metadata = event
                    if isinstance(msg_chunk, ToolMessage):
                        continue
                    if hasattr(msg_chunk, "tool_calls") and msg_chunk.tool_calls:
                        continue
                    if metadata.get("langgraph_node") == "agent":
                        if hasattr(msg_chunk, "content") and msg_chunk.content:
                            yield msg_chunk.content
            return
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                reset_agent()
            else:
                raise last_error


def run_agent_routed(message: str, thread_id: str = "default", *,
                     no_fallback: bool = False) -> dict:
    """Run agent with SLM routing (blocking).

    Always attempts SLM routing (ignores USE_SLM_ROUTER flag).
    Use run_agent() for the baseline (no routing) path.

    Args:
        message: User message
        thread_id: Conversation thread ID
        no_fallback: If True, force routing even for low confidence
                     (structural failures like model_error still fall back)

    Returns:
        Agent result dict with '_router' metadata key.
    """
    from app.agent.router.slm_router import get_router
    from app.agent.router.tool_mapper import get_focused_tools

    # Step 1: Classify
    router = get_router()
    rr = router.classify(message)
    logger.info("SLM Router: %s", rr)

    def _router_meta(path, **extra):
        meta = {
            "path": path,
            "intent": rr.intent,
            "scope": rr.scope,
            "confidence": rr.confidence,
            "latency_ms": rr.latency_ms,
            "fallback_reason": rr.fallback_reason,
        }
        meta.update(extra)
        return meta

    # Step 2: Fallback decision
    if rr.should_fallback:
        # no_fallback only overrides low_confidence; structural failures always fall back
        can_force = (no_fallback and rr.fallback_reason
                     and rr.fallback_reason.startswith("low_confidence"))
        if not can_force:
            logger.info("SLM Router → fallback (%s)", rr.fallback_reason)
            result = run_agent(message, thread_id)
            result["_router"] = _router_meta("fallback")
            return result

    # Step 3: Get focused tools
    focused_tools = get_focused_tools(rr.intent, rr.scope)

    if focused_tools is None:
        logger.info("SLM Router → fallback (no mapping for %s/%s)", rr.intent, rr.scope)
        result = run_agent(message, thread_id)
        result["_router"] = _router_meta("fallback",
                                          fallback_reason=f"no_mapping:{rr.intent}/{rr.scope}")
        return result

    if not focused_tools and rr.intent != "smalltalk_weather":
        logger.info("SLM Router → fallback (empty tools for %s/%s)", rr.intent, rr.scope)
        result = run_agent(message, thread_id)
        result["_router"] = _router_meta("fallback",
                                          fallback_reason=f"empty_tools:{rr.intent}/{rr.scope}")
        return result

    tool_names = [t.name for t in focused_tools]
    logger.info("SLM Router → routed: %s/%s, tools=%s", rr.intent, rr.scope, tool_names)

    # Step 4: Run focused agent
    max_retries = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            focused_agent = _create_focused_agent(focused_tools)
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 10,  # Focused agent: max 10 steps (prevents tool loops)
            }
            result = focused_agent.invoke(
                {"messages": [{"role": "user", "content": message}]}, config
            )
            result["_router"] = _router_meta("routed", focused_tools=tool_names)
            return result
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                reset_agent()
            else:
                raise last_error
