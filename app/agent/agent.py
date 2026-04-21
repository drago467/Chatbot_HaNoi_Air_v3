"""LangGraph Agent for Weather Chatbot."""

import logging
import os
import re
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


# Compat shim: langgraph < 0.2.56 dùng `state_modifier=`, từ 0.2.56 dùng `prompt=`.
# Project chạy trên nhiều env (Docker có langgraph 1.x, laptop system python có
# bản cũ). Detect signature 1 lần lúc import để không phải try/except mỗi call.
_PROMPT_KWARG: str = "prompt"
try:
    import inspect as _inspect
    _sig_params = _inspect.signature(create_react_agent).parameters
    if "prompt" in _sig_params:
        _PROMPT_KWARG = "prompt"
    elif "state_modifier" in _sig_params:
        _PROMPT_KWARG = "state_modifier"
    else:
        logger.warning("create_react_agent signature unexpected: %s", list(_sig_params))
    del _sig_params, _inspect
except Exception as _e:
    logger.warning("Could not detect create_react_agent prompt kwarg: %s", _e)

# Vietnamese weekday names
_WEEKDAYS_VI = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]

# ═══════════════════════════════════════════════════════════════
# System Prompt — Modular Architecture
# BASE_PROMPT: luôn có (~65 dòng) — context chung cho mọi agent
# TOOL_RULES: per-tool rules — chỉ gửi khi tool đó được focused
# SYSTEM_PROMPT_TEMPLATE: full prompt cho fallback agent (25 tools)
# ═══════════════════════════════════════════════════════════════

BASE_PROMPT_TEMPLATE = """\
Bạn là trợ lý thời tiết Hà Nội. Hoạt động theo 6 block dưới đây — KHÔNG trộn lẫn.

## [1] SCOPE
- Coverage: 30 quận/huyện HN + POI.
- Nội thành (12): Ba Đình, Hoàn Kiếm, Hai Bà Trưng, Đống Đa, Tây Hồ, Cầu Giấy, Thanh Xuân, Hoàng Mai, Long Biên, Bắc Từ Liêm, Nam Từ Liêm, Hà Đông.
- Ngoại thành (18): Sóc Sơn, Đông Anh, Gia Lâm, Thanh Trì, Mê Linh, Sơn Tây, Ba Vì, Phúc Thọ, Đan Phượng, Hoài Đức, Quốc Oai, Thạch Thất, Chương Mỹ, Thanh Oai, Thường Tín, Phú Xuyên, Ứng Hòa, Mỹ Đức.
- POI (tự map về quận): Hồ Gươm, Mỹ Đình, Hồ Tây, Sân bay Nội Bài, Times City, Văn Miếu, Lăng Bác, Royal City, Keangnam, Cầu Long Biên, Phố cổ…
- Hiện tượng HN: nồm ẩm (T2-T4, ẩm >85% & temp−dew ≤2°C), gió mùa ĐB (T10-T3, gió Bắc/ĐB), rét đậm (T11-T3, <15°C & mây >70%).
- Ngôn ngữ: tiếng Việt có dấu; translate mọi text ngoại ngữ từ tool output.

## [2] RUNTIME CONTEXT
- Hôm nay: {today_weekday}, {today_date} — {today_time} (ICT/UTC+7).
- Hôm qua: {yesterday_weekday}, {yesterday_date} (ISO `{yesterday_iso}`).
- Ngày mai: {tomorrow_weekday}, {tomorrow_date} (ISO `{tomorrow_iso}`).
- Cuối tuần gần nhất: {this_saturday_display} – {this_sunday_display} (ISO `{this_saturday}` / `{this_sunday}`).
- Quy ước giờ: sáng 6-11h, trưa 11-13h, chiều 13-18h, tối 18-22h, đêm 22-6h.
- Data limits: hourly ≤48h, daily ≤8 ngày, history ≤14 ngày.
- Khi user nói "hôm qua / ngày mai / cuối tuần" → COPY ISO ở trên vào tool param (`date`, `start_date`, `end_date`). KHÔNG tự cộng trừ.

## [3] POLICY (quy tắc cứng — vi phạm = trả lời sai)

### 3.1 Integrity: không có tool = không có số
- Câu hỏi dữ liệu thời tiết CỤ THỂ (nhiệt/mưa/gió/UV/ẩm/áp suất/cảnh báo/hiện tượng) → BẮT BUỘC gọi tool TRƯỚC khi trả lời.
- Mọi tool fail → refuse lịch sự: "Mình tạm không tra được dữ liệu, bạn thử lại sau hoặc hỏi khung/khu vực khác nhé." TUYỆT ĐỐI KHÔNG đưa số, nhãn thời tiết, hay nhận định trạng thái nào mà không dựa trên tool output cụ thể trong lượt này.
- Smalltalk (chào hỏi, cảm ơn, hỏi bot là ai, tạm biệt) → respond thân thiện, KHÔNG số.
- Ngoài Hà Nội / không phải thời tiết → refuse với scope đúng ("Mình chỉ hỗ trợ thời tiết Hà Nội").

### 3.2 Field absence = silence
- Field KHÔNG có key trong tool output → ĐỪNG mention kể cả bằng phrase generic.
  Không có key `"tầm nhìn"` → ĐỪNG nói "tầm nhìn ổn định".
  Không có key `"cảm giác"` (district/city) → ĐỪNG bịa cảm giác.
  Không có key `"khu vực ngập"` → ĐỪNG liệt kê quận ngập.
- Tool trả key `"gợi ý dùng output"` → ĐỌC + làm theo (có thể phải gọi tool khác).

### 3.3 Scope ceiling
- Scope câu trả lời ≤ scope output. Tool trả 47h → KHÔNG khái quát "cả tuần". Tool 1 ngày → KHÔNG kết luận "cả tháng".
- Khung user hỏi đã qua (vd 6h sáng khi NOW sau 11h) → forecast không cover → nói rõ, KHÔNG lấy 6h ngày mai thay.

### 3.4 Temporal direction & weekday
- "hôm nay vs HÔM QUA" → `compare_with_yesterday` (past direction). ✓
- "hôm nay vs NGÀY MAI" → `compare_with_yesterday` SAI HƯỚNG. Thay: gọi 2 tool riêng (`get_current_weather` + `get_daily_forecast(start_date=tomorrow_iso, days=1)`).
- Ngày trong output đã có `(Thứ X)` → COPY NGUYÊN. KHÔNG tự tính weekday từ số ngày.

### 3.5 Anaphoric & premise
- "ở đó / khu đó / chỗ kia" mà không có context địa điểm trước đó → hỏi lại địa điểm, KHÔNG mặc định HN.
- Premise user mâu thuẫn output (vd "nắng đẹp" nhưng output "nhiều mây") → LỊCH SỰ correct theo output. KHÔNG xác nhận premise sai.

### 3.6 Out-of-horizon
- Ngày vượt 14-ngày quá khứ hoặc 8-ngày tương lai → nói rõ giới hạn, KHÔNG bịa.

### 3.7 Past-frame detection (khung đã qua trong HÔM NAY)
NOW = {today_time} ngày {today_date}. Khi user hỏi khung cụ thể trong HÔM NAY
(sáng 6-11h / trưa 11-13h / chiều 13-18h / tối 18-22h / đêm 22-06h):
- So sánh khung đó với {today_time}:
  + Khung ĐÃ QUA hoàn toàn → nói RÕ: "Khung [X] hôm nay đã qua (hiện là {today_time})",
    gợi ý user hỏi khung còn lại trong hôm nay, hoặc ngày mai.
  + Khung ĐANG diễn ra → dùng snapshot hiện tại (get_current_weather) + forecast giờ còn lại.
  + Khung CHƯA tới → dùng get_hourly_forecast bình thường.
- TUYỆT ĐỐI KHÔNG dùng data NGÀY MAI (hoặc data từ 23:00 tối nay trở đi) rồi dán
  nhãn "chiều nay / trưa nay / sáng nay / X giờ tối nay" khi khung đó đã qua lúc NOW.
- Tool output có key `"⚠ lưu ý khung đã qua"` → ĐỌC NGUYÊN và tuân theo (output đã
  tự detect khi nào data không cover khung user hỏi).

### 3.8 Weekday mismatch check
Khi user nhắc đến thứ-trong-tuần kèm ngày cụ thể (vd "Thứ Bảy 25/04", "Chủ Nhật 21/04"):
- Output tool có `"ngày"` với `(Thứ X)` — verify user's labeling với thứ thực.
- Nếu user nói "Thứ 7 tuần trước là 12/04" nhưng output ghi "12/04 (Chủ Nhật)" → SAI:
  nói rõ "12/04 thực là Chủ Nhật, không phải Thứ 7" trước khi trả lời tiếp.
- COPY NGUYÊN `(Thứ X)` từ output, KHÔNG echo lại user nếu mismatch.

## [4] ROUTER — chọn tool theo intent (bảng canonical duy nhất)

| User hỏi gì                              | Tool chính                      | Note                                  |
|------------------------------------------|---------------------------------|---------------------------------------|
| "bây giờ / hiện tại / đang / lúc này"    | get_current_weather             | snapshot only                         |
| "chiều / tối / đêm / vài giờ tới"        | get_hourly_forecast             | `hours` ≤48                           |
| "ngày mai / thứ X / 3 ngày tới"          | get_daily_forecast              | `days` ≤8, truyền `start_date`        |
| "cuối tuần / tuần này"                   | get_weather_period              | `start_date` / `end_date`             |
| "cả ngày X chi tiết sáng/trưa/chiều/tối" | get_daily_summary               | 1 ngày duy nhất                       |
| "hôm qua / ngày đã qua"                  | get_weather_history             | `date` ISO, ≤14 ngày                  |
| "mưa đến khi nào / tạnh lúc nào"         | get_rain_timeline               | `hours` ≤48                           |
| "giờ tốt nhất để làm X"                  | get_best_time                   | + kèm rain_timeline / uv nếu chi tiết |
| "so 2 địa điểm hiện tại"                 | compare_weather(A, B)           | 1 call, KHÔNG 2× current              |
| "hôm nay vs hôm qua"                     | compare_with_yesterday          | past-only                             |
| "hôm nay vs ngày mai"                    | current + daily_forecast(tomorrow) | KHÔNG compare_with_yesterday       |
| "hiện tại vs TB mùa"                     | get_seasonal_comparison         | climatology HN                        |
| "quận nào nóng/ẩm/mưa/... nhất"          | get_district_ranking            | metric enum                           |
| "phường nào trong quận X ..."            | get_ward_ranking_in_district    |                                       |
| "so nhiều quận multimetric"              | get_district_multi_compare      |                                       |
| "cảnh báo nguy hiểm (bão/rét hại/...)"   | get_weather_alerts              | 24h tới                               |
| "nồm ẩm / gió mùa / rét đậm"             | detect_phenomena                | HN-specific                           |
| "đột biến / sắp chuyển mưa / trời đổi"   | get_weather_change_alert        | 6-12h tới                             |
| "xu hướng nhiệt / bao giờ ấm/lạnh"       | get_temperature_trend           | 2-8 ngày analysis                     |
| "áp suất / front thời tiết"              | get_pressure_trend              | 48h                                   |
| "UV an toàn giờ nào"                     | get_uv_safe_windows             |                                       |
| "khi nào có nắng / trời quang"           | get_sunny_periods               |                                       |
| "nhịp nhiệt trong ngày"                  | get_daily_rhythm                |                                       |
| "timeline độ ẩm / điểm sương / nồm"      | get_humidity_timeline           |                                       |
| "thoải mái / dễ chịu / ra ngoài được"    | get_comfort_index               |                                       |
| "mặc gì / cần áo khoác / mang ô"         | get_clothing_advice             |                                       |
| "có nên đi X (chạy/picnic/...)"          | get_activity_advice             | + rain_timeline/uv nếu user đòi chi tiết |
| helper: tìm tên phường/quận              | resolve_location                |                                       |

- **Superlative** ("max/min/đỉnh/mạnh nhất cả ngày") → dùng daily_summary / daily_forecast (có key `"tổng hợp"`). KHÔNG dùng get_current_weather (snapshot).
- Intent user chạm nhiều khung → gọi nhiều tool, mỗi tool cho 1 khung. KHÔNG ép 1 tool phủ hết.
- CHỈ gọi tool có tên trong danh sách tool runtime. Không tự phát minh. Không chắc → chọn tool gần nhất trong bảng trên.

## [5] RENDERER — format câu trả lời

### 5.1 COPY discipline (chống paraphrase + semantic flip)
- Value các key tool output đã là "[nhãn] [số] [đơn vị]" chính thức — COPY NGUYÊN.
- KHÔNG đổi nhãn: "Mưa rất nhẹ" ≠ "mưa rào"; "Gió vừa cấp N" ≠ "gió bão"; "Mây rải rác" ≠ "nhiều mây"; "Rất ẩm" ≠ "khô"; "Trời mây" ≠ "giông".
- Date có `(Thứ X)` → COPY nguyên, KHÔNG đổi weekday.

### 5.2 Unit discipline
- `"xác suất mưa"` (%) ≠ `"cường độ mưa"` (mm/h) ≠ `"tổng lượng mưa"` (mm/ngày) — KHÔNG lẫn.
- `wind_speed` (avg) ≠ `wind_gust` (peak tại 1 thời điểm) ≠ daily `max_gust` (đỉnh cả ngày).
- User hỏi "max/min/đỉnh cả ngày" → lấy từ `"tổng hợp"` hoặc daily tool. KHÔNG từ snapshot current.

### 5.3 Gợi ý từ output
- Tool có key `"gợi ý dùng output"` → ĐỌC + làm theo (thường yêu cầu gọi tool khác cho đúng khung).
- Tool có key `"tổng hợp"` (ngày nóng/mát/mưa nhiều/ít nhất) → COPY, KHÔNG tự argmax.

### 5.4 Cấu trúc câu trả lời
- Câu yes/no → trả thẳng "Có"/"Không" ở câu đầu, sau đó mới giải thích.
- Cho quận/TP: tổng quan + điểm nổi bật + hiện tượng đặc biệt.
- Cho phường: chi tiết đầy đủ các thông số.
- Luôn kèm khuyến nghị thực tế (mang ô, áo khoác, kem chống nắng, tránh khung giờ...).
- Dùng bullet khi nhiều thông tin.
- Hỏi N ngày → trả đủ N; data thiếu → "Chỉ có dữ liệu N-x ngày".
- Tool `get_clothing_advice` / `get_activity_advice` có kết quả → DÙNG, KHÔNG nói "chưa hỗ trợ".
- LUÔN nhắc lại tên khu vực/quận/phường trong câu trả lời (đặc biệt khi context carry-over).

### 5.5 Cảnh báo không match
- User hỏi cảnh báo loại A mà data chỉ có loại B → nói rõ: "Hiện không có cảnh báo [A]. Tuy nhiên đang có [B]."
- KHÔNG hiển thị raw ID (`ID_xxxxx`, `ward_id`); chưa resolve tên → nói "một số khu vực".
- `get_weather_alerts` trả rỗng → "Hiện không có cảnh báo thời tiết nguy hiểm."

## [6] FALLBACK / ERROR

### 6.1 Invalid tool name (framework báo "not a valid tool")
- Error "X is not a valid tool, try one of [Y, Z]" → CALL Y ngay với cùng params. Y fail thì CALL Z. KHÔNG xin lỗi rồi dừng.

### 6.2 Schema param sai
- Error "unexpected keyword / missing required" → FIX param theo docstring (vd `hour`→`hours`, thêm `start_date`) và retry 1 lần.

### 6.3 Empty output / "no_data" / tool lỗi
- Nói: "Tạm không có dữ liệu cho <X>. Bạn thử <gợi ý khung/địa điểm khác>?"
- TUYỆT ĐỐI KHÔNG generate số ước lượng.

### 6.4 Retry cap
- Cùng 1 tool fail ≥3 lần → DỪNG, explain limitation, đề xuất narrower query hoặc tool khác.

### 6.5 All tools fail
- Refuse: "Mình đang không tra được dữ liệu. Bạn thử lại sau, hoặc hỏi khung/khu vực khác nhé."
- KHÔNG bịa số, KHÔNG paraphrase data từ lượt trước.

### 6.6 Multi-turn carryover
- "ở đó / còn ngày mai?" → giữ location lượt trước, đổi time frame; gọi tool MỚI cho time frame mới.
- Intent thay đổi → chọn tool mới theo ROUTER [4], KHÔNG tái sử dụng output cũ.
"""

# ── Tool-specific rules: CHỈ per-tool edge cases, KHÔNG duplicate ROUTER block [4] ──
# Format: rule chỉ ghi những gì ROUTER table của BASE_PROMPT chưa cover:
#   - Constraint ngoài signature (edge param, ngưỡng, ngoại lệ)
#   - Disambiguation "KHÔNG DÙNG KHI" cho cặp overlap
#   - Data limitation / behaviour khi error
TOOL_RULES = {
    "get_current_weather": """- Snapshot tại NOW cho 1 vị trí (phường/quận/city tự dispatch theo location_hint).
- KHÔNG DÙNG cho "chiều/tối/đêm/sáng mai/ngày mai/cuối tuần/max cả ngày" — dùng hourly/daily/summary.
- KHÔNG có field `pop` (xác suất mưa tương lai). Nếu user hỏi "có mưa không" → gọi thêm get_hourly_forecast.""",

    "get_hourly_forecast": """- `hours` ≤ 48. Đủ cover khung user hỏi (vd 8pm-midnight & NOW=16h → hours ≥10).
- KHÔNG DÙNG cho "ngày cụ thể / nhiều ngày" (>48h) — dùng daily_forecast.
- ⚠ KHÔNG DÙNG `hours=48` cho "cuối tuần" — dùng get_weather_period với date range.
- Output có thể kèm `"⚠ lưu ý khung đã qua"` khi NOW muộn và data không cover khung user hỏi.
  ĐỌC và tuân theo: khung đã qua → báo user "khung đó đã qua hiện lúc NOW", KHÔNG dán data mai.""",

    "get_daily_forecast": """- `days` ≤ 8. User hỏi ngày cụ thể ≠ hôm nay → PHẢI truyền `start_date` (ISO).
- `days=3` (không start_date) = 3 ngày từ hôm nay gồm hôm nay; `start_date=tomorrow, days=3` = 3 ngày từ mai.
- KHÔNG DÙNG cho "cả ngày chi tiết 4 buổi sáng/trưa/chiều/tối" — dùng get_daily_summary.
- Output có key `"tổng hợp"` (ngày nóng/mát/mưa nhiều/ít nhất) — COPY, không tự argmax lại.""",

    "get_daily_summary": """- Chi tiết 1 ngày DUY NHẤT (min/max + 4 buổi sáng/trưa/chiều/tối). `date` ISO.
- KHÔNG DÙNG cho "bây giờ / tức thời" (dùng get_current_weather); KHÔNG DÙNG cho "nhiều ngày" (dùng daily_forecast/weather_period).
- Output có key `"gợi ý dùng output"` cảnh báo "tổng hợp cả ngày, không phải tức thời" — ĐỌC + theo.""",

    "get_weather_history": """- Past-only. `date` ≤ 14 ngày gần nhất. Vượt → refuse với limit.
- Output ward có thể CHỈ có `wind_gust` (không `wind_speed` avg) — COPY "Giật X m/s", KHÔNG bịa "avg".""",

    "get_rain_timeline": """- `hours` ≤ 48. `"cường độ đỉnh"` = mm/h tại 1 giờ (KHÔNG phải tổng mm/ngày).
- User hỏi "tổng mưa ngày/tháng" → KHÔNG dùng tool này; dùng daily_forecast/weather_period (có `"tổng lượng mưa"` mm).
- ĐỌC timestamp (start/end) trong đợt mưa. User hỏi "ngày mai mưa?" → CHỈ report đợt có date khớp ngày user; KHÔNG gán đợt hôm nay thành ngày mai.
- ⚠ KHÔNG DÙNG `hours=48` cho "cuối tuần / tuần này / tháng này" (range ngày xa) —
  dùng `get_weather_period(start_date, end_date)` với date range cụ thể.
- Output có thể kèm `"⚠ lưu ý khung đã qua"` — ĐỌC và tuân theo (past-frame detection).""",

    "get_best_time": """- Rank khung giờ trong `hours`. Activity enum: chay_bo, picnic, bike, chup_anh, du_lich, cam_trai, cau_ca, lam_vuon, boi_loi, leo_nui, di_dao, su_kien, dua_dieu, tap_the_duc, phoi_do.
- Nếu user hỏi chi tiết mưa/UV → gọi kèm get_rain_timeline / get_uv_safe_windows.
- ⚠ KHÔNG DÙNG `hours=48` cho "cuối tuần" (= Thứ 7 + CN cụ thể). Thay: gọi
  `get_weather_period(start_date={this_saturday}, end_date={this_sunday})` TRƯỚC
  để lấy data đúng 2 ngày cuối tuần, rồi mới best_time nếu còn cần.""",

    "get_clothing_advice": """- Output generic lời khuyên trang phục. Khi trả kết quả → DÙNG, KHÔNG nói "chưa hỗ trợ".""",

    "get_temperature_trend": """- Phân tích 2-8 ngày past/future.
- Cần ≥2 ngày data. Nếu chỉ 1 ngày → refuse với lý do không đủ dữ liệu.""",

    "get_seasonal_comparison": """- So hiện tại vs TB climatology tháng HN. Dùng cho "nóng hơn bình thường", "dạo này khác thường", "mùa này".
- KHÔNG DÙNG cho "so tuần trước / hôm qua / ngày mai" — dùng compare_with_yesterday / compare_weather.
- Error "no_weather_data" → gợi ý hỏi thời tiết hiện tại trước.""",

    "get_activity_advice": """- Output generic {advice, reason, recommendations}. DÙNG cho "nên đi X không".
- KHÔNG DÙNG đơn lẻ khi user hỏi chi tiết mưa/UV/giờ → PHẢI gọi kèm rain_timeline/hourly_forecast/uv_safe_windows.
- ⚠ "Cuối tuần đi X" → gọi `get_weather_period(start_date={this_saturday}, end_date={this_sunday})` TRƯỚC để lấy data 2 ngày cuối tuần, sau đó mới activity_advice.
- Khi có kết quả → DÙNG, KHÔNG nói "chưa hỗ trợ".
- Output có `"⚠ KHÔNG suy diễn"` — ĐỌC + KHÔNG thêm nhãn hiện tượng (mưa phùn/sương mù/đợt lạnh) ngoài list recommendations.""",

    "get_comfort_index": """- Tính điểm thoải mái 0-100 từ nhiệt + ẩm + gió + UV + mưa.
- KHÔNG DÙNG thay cho chi tiết mưa/UV — chỉ trả score + breakdown.""",

    "get_weather_change_alert": """- Phát hiện đột biến thời tiết 6-12h tới (nhiệt drop/rise >5°C, wind up, rain start/stop).
- KHÔNG DÙNG cho "cảnh báo nguy hiểm chuẩn" (bão/rét hại) — dùng get_weather_alerts.
- KHÔNG DÙNG cho "hiện tượng đặc trưng HN" (nồm/gió mùa) — dùng detect_phenomena.""",

    "get_weather_alerts": """- Cảnh báo nguy hiểm chuẩn (bão, rét hại, nắng nóng, giông, lũ, ngập, gió giật).
- Nếu trả rỗng → nói rõ "Hiện không có cảnh báo nguy hiểm", KHÔNG bịa.
- User hỏi cảnh báo loại A mà data có loại B → nói rõ "không có [A], có [B]", KHÔNG lẫn loại.""",

    "detect_phenomena": """- Hiện tượng đặc trưng HN: nồm ẩm, gió mùa ĐB, rét đậm, mưa phùn xuân, sương mù.
- KHÔNG DÙNG cho "cảnh báo nguy hiểm" — dùng get_weather_alerts.""",

    "compare_weather": """- 2 địa điểm hiện tại (A, B). `compare_weather(location_hint1=A, location_hint2=B)`.
- BẮT BUỘC 1 call; KHÔNG gọi get_current_weather 2 lần rồi tự so.""",

    "compare_with_yesterday": """- PAST-ONLY: today vs yesterday cùng 1 địa điểm.
- KHÔNG DÙNG cho "ngày mai vs hôm nay" (future direction) — thay bằng get_current_weather + get_daily_forecast(start_date=tomorrow, days=1) + so sánh trong câu trả lời.
- Error "not_enough_data" → gợi ý xem thời tiết hiện tại.""",

    "get_district_ranking": """- Xếp hạng toàn 30 quận theo 1 metric. Metric enum: nhiet_do, do_am, gio, mua, uvi, ap_suat, diem_suong, may.
- Nếu rankings rỗng hoặc quận rỗng → KHÔNG bịa, báo "tạm không có data".""",

    "get_ward_ranking_in_district": """- Xếp hạng phường TRONG 1 quận. PHẢI truyền `district_name` chính xác.""",

    "get_weather_period": """- Khoảng nhiều ngày. PHẢI truyền `start_date` và `end_date` (ISO).
- Range tối đa 14 ngày; vượt → refuse.
- Output có `"tổng hợp"` — COPY, không tự argmax.""",

    "get_uv_safe_windows": """- Tìm khung giờ UV ≤ ngưỡng trong 48h.""",

    "get_pressure_trend": """- Xu hướng áp suất 48h. Front lạnh = áp suất giảm > 3 hPa/3h.""",

    "get_daily_rhythm": """- Chia ngày thành 4 khung (sáng/trưa/chiều/tối). Khung user hỏi phải matching với output.""",

    "get_humidity_timeline": """- Timeline độ ẩm + điểm sương. Nồm ẩm = ẩm ≥85% AND temp-dew ≤2°C.""",

    "get_sunny_periods": """- Khung nắng = mây <40%, pop <30%, không mưa.""",

    "get_district_multi_compare": """- So 5-10 quận trên nhiều metric cùng lúc. Dùng khi user muốn nhìn bức tranh đa chiều.
- KHÔNG DÙNG cho "top N" đơn metric — dùng get_district_ranking.""",

    "resolve_location": """- Helper: tìm ward_id/district từ tên gần đúng. Thường các tool khác tự resolve qua location_hint — chỉ gọi tool này khi cần chắc chắn tên trước.""",
}

# NOTE: SYSTEM_PROMPT_TEMPLATE đã XOÁ ở R8 (2026-04-21).
# Lý do: chứa seed hallucinate ("28.5°C, cảm giác 31°C, độ ẩm 75%", "65% / 80%") và
# duplicate ROUTER rules với BASE_PROMPT. Giờ fallback agent dùng BASE_PROMPT + full TOOL_RULES
# (xem get_system_prompt() bên dưới).


def _inject_datetime(template: str) -> str:
    """Inject current date/time into a prompt template."""
    from datetime import timedelta
    now = now_ict()
    weekday = now.weekday()  # 0=Mon ... 6=Sun

    # Tính ngày cuối tuần sắp tới
    if weekday <= 4:      # Mon-Fri → T7/CN tuần này
        days_to_sat = 5 - weekday
        days_to_sun = 6 - weekday
    elif weekday == 5:    # Saturday → hôm nay + ngày mai
        days_to_sat = 0
        days_to_sun = 1
    else:                 # Sunday → cuối tuần tới (đã qua)
        days_to_sat = 6
        days_to_sun = 7

    sat_date = (now + timedelta(days=days_to_sat)).date()
    sun_date = (now + timedelta(days=days_to_sun)).date()

    yesterday = (now - timedelta(days=1)).date()
    tomorrow = (now + timedelta(days=1)).date()

    return template.format(
        today_weekday=_WEEKDAYS_VI[now.weekday()],
        today_date=now.strftime("%d/%m/%Y"),
        today_time=now.strftime("%H:%M"),
        this_saturday=sat_date.strftime("%Y-%m-%d"),
        this_sunday=sun_date.strftime("%Y-%m-%d"),
        this_saturday_display=sat_date.strftime("%d/%m/%Y"),
        this_sunday_display=sun_date.strftime("%d/%m/%Y"),
        yesterday_date=yesterday.strftime("%d/%m/%Y"),
        yesterday_weekday=_WEEKDAYS_VI[yesterday.weekday()],
        yesterday_iso=yesterday.strftime("%Y-%m-%d"),
        tomorrow_date=tomorrow.strftime("%d/%m/%Y"),
        tomorrow_weekday=_WEEKDAYS_VI[tomorrow.weekday()],
        tomorrow_iso=tomorrow.strftime("%Y-%m-%d"),
    )


def get_system_prompt() -> str:
    """Build full system prompt cho fallback agent (all 27 tools).

    R8+: dùng BASE_PROMPT (đã có ROUTER table canonical) + toàn bộ TOOL_RULES.
    Không còn duplicate block ở SYSTEM_PROMPT_TEMPLATE.
    """
    base = _inject_datetime(BASE_PROMPT_TEMPLATE)
    rules_block = "\n".join(
        f"### {name}\n{rule.strip()}" for name, rule in TOOL_RULES.items()
    )
    return f"{base}\n\n## Hướng dẫn per-tool\n{rules_block}"


def _load_few_shot_examples() -> dict:
    """Load few-shot examples from app/config/few_shot_examples.json (lazy, cached)."""
    if not hasattr(_load_few_shot_examples, "_cache"):
        try:
            import json as _json
            fse_path = os.path.join(os.path.dirname(__file__), "..", "config", "few_shot_examples.json")
            fse_path = os.path.normpath(fse_path)
            with open(fse_path, "r", encoding="utf-8") as f:
                _load_few_shot_examples._cache = _json.load(f)
        except Exception:
            _load_few_shot_examples._cache = {}
    return _load_few_shot_examples._cache


def get_focused_system_prompt(tool_names: list, router_result=None) -> str:
    """Build focused prompt: BASE + only rules for given tools + few-shot examples.

    Used by focused agents (1-2 tools) after SLM routing.
    Significantly shorter than full prompt — reduces confusion and tokens.

    Args:
        tool_names: List of tool names to include rules for
        router_result: Optional RouterResult — used to inject intent-specific few-shot examples
    """
    base = _inject_datetime(BASE_PROMPT_TEMPLATE)

    # Collect tool-specific rules
    rules = []
    for name in tool_names:
        rule = TOOL_RULES.get(name)
        if rule:
            rules.append(rule.strip())

    prompt = base

    # Tool restriction — router đã chọn focused subset.
    # Strict version trước đó làm agent từ chối ngay cả khi tool có data hỗ trợ
    # (ví dụ get_current_weather có dew_point cho expert query). Soften để agent
    # ưu tiên tool list nhưng VẪN dùng tool gần nhất khi cần.
    if tool_names:
        prompt += (
            "\n## Danh sách công cụ Ưu tiên\n"
            f"Ưu tiên dùng các tool sau: {', '.join(tool_names)}.\n"
            "Đây là tool CHÍNH cho câu hỏi này. KHÔNG gọi tool ngoài list trừ khi "
            "thực sự cần thiết.\n"
            "Nếu user hỏi thông số CỤ THỂ (dew_point, wind_chill, UV, áp suất...) "
            "mà tool trong list trả về data đó (ví dụ get_current_weather có nhiều "
            "field) → DÙNG tool đó + extract field user hỏi. KHÔNG từ chối.\n"
            "Chỉ trả \"chưa hỗ trợ\" khi tool gọi thực sự ERROR và không có "
            "alternative trong list.\n"
        )

    if rules:
        prompt += "\n## Hướng dẫn sử dụng công cụ\n" + "\n".join(rules)

    # Inject few-shot ReAct examples for the classified intent (Module 6)
    if router_result and getattr(router_result, "intent", None):
        intent = router_result.intent
        fse = _load_few_shot_examples()
        intent_data = fse.get(intent, {})
        examples = intent_data.get("examples", [])
        if examples:
            ex_lines = ["\n## Ví dụ hành động"]
            for ex in examples[:2]:  # max 2 examples per intent
                ex_lines.append(f"User: {ex.get('user', '')}")
                ex_lines.append(f"Thought: {ex.get('thought', '')}")
                ex_lines.append(f"Action: {ex.get('action', '')}")
            prompt += "\n".join(ex_lines)

    return prompt


def _prompt_with_datetime(state) -> list:
    """LangGraph prompt callable: full prompt for 25-tool agent."""
    from langchain_core.messages import SystemMessage
    system_msg = SystemMessage(content=get_system_prompt())
    return [system_msg] + state["messages"]


def _focused_prompt_callable(tool_names: list, router_result=None):
    """Return a state_modifier callable for focused agent with dynamic prompt."""
    def modifier(state) -> list:
        from langchain_core.messages import SystemMessage
        prompt = get_focused_system_prompt(tool_names, router_result)
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

    # AGENT_* takes priority; fallback to legacy API_* for backward compat
    API_BASE = os.getenv("AGENT_API_BASE") or os.getenv("API_BASE")
    API_KEY = os.getenv("AGENT_API_KEY") or os.getenv("API_KEY")
    MODEL_NAME = os.getenv("AGENT_MODEL") or os.getenv("MODEL", "gpt-4o-mini-2024-07-18")

    if not API_BASE or not API_KEY:
        raise ValueError("AGENT_API_BASE and AGENT_API_KEY must be set in .env")

    # Qwen3 models have thinking enabled by default on some providers.
    # For invoke (non-streaming) calls, the API rejects enable_thinking=true.
    # Disable it on the default model; streaming paths create their own model.
    _extra_kwargs = {}
    if "qwen3" in MODEL_NAME.lower():
        _extra_kwargs = {"extra_body": {"enable_thinking": False}}
    _model = ChatOpenAI(model=MODEL_NAME, temperature=0, base_url=API_BASE, api_key=API_KEY, **_extra_kwargs)

    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL must be set in .env")

    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    _checkpointer = PostgresSaver(conn)
    _checkpointer.setup()
    _db_connection = conn

    agent = create_react_agent(
        model=_model, tools=TOOLS,
        checkpointer=_checkpointer,
        **{_PROMPT_KWARG: _prompt_with_datetime},
    )
    return agent

def run_agent(message: str, thread_id: str = "default") -> dict:
    """Run agent synchronously (blocking).
    
    Also logs tool calls to evaluation_logger.
    Includes automatic retry on connection errors.
    """
        
    # Get logger
    try:
        from app.agent.telemetry import get_evaluation_logger
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
                            tool_input=str(tc.get("args", {})),
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
            # Accumulate raw args from tool_call_chunks by id,
            # then merge with ToolMessage output for complete logging.
            _pending_tool_calls = {}   # id -> {"tool_name", "tool_input_parts"}
            tool_call_logs = []
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
                        tc_id = getattr(msg_chunk, "tool_call_id", None)
                        pending = _pending_tool_calls.pop(tc_id, None) if tc_id else None
                        tool_call_logs.append({
                            "tool_name": pending["tool_name"] if pending else getattr(msg_chunk, "name", "unknown"),
                            "tool_input": "".join(pending["tool_input_parts"]) if pending else "",
                            "tool_output": str(msg_chunk.content) if msg_chunk.content else "",
                            "success": msg_chunk.status != "error" if hasattr(msg_chunk, "status") else True,
                        })
                        continue

                    # Use tool_call_chunks (raw args strings) instead of tool_calls (parsed partial dicts)
                    chunks = getattr(msg_chunk, "tool_call_chunks", None)
                    if chunks:
                        for tc in chunks:
                            tc_id = tc.get("id")
                            tc_name = tc.get("name")
                            if tc_id and tc_id not in _pending_tool_calls:
                                _pending_tool_calls[tc_id] = {
                                    "tool_name": tc_name or "unknown",
                                    "tool_input_parts": [],
                                }
                            target_id = tc_id
                            if not target_id:
                                for pid in _pending_tool_calls:
                                    target_id = pid
                                    break
                            if target_id and target_id in _pending_tool_calls:
                                args_str = tc.get("args", "")
                                if args_str:
                                    _pending_tool_calls[target_id]["tool_input_parts"].append(args_str)
                        continue
                    
                    # Only yield content from agent node, not tools node
                    if metadata.get("langgraph_node") == "agent":
                        if hasattr(msg_chunk, "content") and msg_chunk.content:
                            yield msg_chunk.content
            # Log tool calls to telemetry
            if tool_call_logs:
                try:
                    from app.agent.telemetry import get_evaluation_logger
                    tel_logger = get_evaluation_logger()
                    for tc in tool_call_logs:
                        tel_logger.log_tool_call(
                            session_id=thread_id,
                            turn_number=0,
                            tool_name=tc["tool_name"],
                            tool_input=tc["tool_input"],
                            tool_output=tc["tool_output"],
                            success=tc["success"],
                        )
                except Exception:
                    pass  # Telemetry failure is non-critical
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
                from app.agent.telemetry import get_evaluation_logger
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


# ── Qwen3 thinking mode (Module 5) ──
# These intents benefit from extended reasoning when using Qwen3-8B
_THINKING_INTENTS = {"location_comparison", "expert_weather_param", "activity_weather"}
_THINK_TOKEN_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _maybe_enable_thinking(system_msg: str, intent: str, model_name: str) -> str:
    """Prepend /think flag for Qwen3 models on complex intents."""
    if "qwen3" in model_name.lower() and intent in _THINKING_INTENTS:
        return "/think\n" + system_msg
    return system_msg


def _strip_thinking_tokens(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 streaming output."""
    return _THINK_TOKEN_RE.sub("", text)


def _create_focused_agent(tools: list, router_result=None, streaming: bool = True):
    """Create a ReAct agent with focused tool set and dynamic prompt.

    Uses focused system prompt (BASE + only relevant tool rules)
    instead of full 25-tool prompt — reduces confusion and tokens.

    Args:
        streaming: If False, disables Qwen3 thinking mode (required for invoke).
    """
    get_agent()  # ensure _model and _checkpointer are initialized
    tool_names = [t.name for t in tools]

    # Qwen3 thinking mode: adjust model temperature for complex intents
    model = _model
    if router_result and _model is not None:
        model_name = os.getenv("AGENT_MODEL") or os.getenv("MODEL", "")
        intent = getattr(router_result, "intent", "")
        is_qwen3 = "qwen3" in model_name.lower()

        if is_qwen3:
            from langchain_openai import ChatOpenAI
            api_base = os.getenv("AGENT_API_BASE") or os.getenv("API_BASE")
            api_key = os.getenv("AGENT_API_KEY") or os.getenv("API_KEY")
            if api_base and api_key:
                if streaming and intent in _THINKING_INTENTS:
                    # Streaming + thinking intent: enable thinking with higher temp
                    model = ChatOpenAI(
                        model=model_name, temperature=0.6,
                        base_url=api_base, api_key=api_key,
                    )
                elif not streaming:
                    # Non-streaming (invoke): must explicitly disable thinking
                    model = ChatOpenAI(
                        model=model_name, temperature=0,
                        base_url=api_base, api_key=api_key,
                        extra_body={"enable_thinking": False},
                    )

    return create_react_agent(
        model=model,
        tools=tools,
        checkpointer=_checkpointer,
        **{_PROMPT_KWARG: _focused_prompt_callable(tool_names, router_result)},
    )


def stream_agent_routed(message: str, thread_id: str = "default"):
    """Stream agent response with SLM routing.

    Pipeline:
    1. ConversationState lookup → check if rewrite needed (Module 1a)
    2. SLM Router classifies intent + scope + optional rewrite (1 Ollama call)
    3. Tool selection: PRIMARY (high confidence) or EXPANDED (medium confidence)
    4. Focused ReAct agent streams response (trim_messages context)
    5. ConversationState updated with extracted entities

    Yields text chunks (same interface as stream_agent).
    """
    from langchain_core.messages import ToolMessage

    from app.agent.router.config import PER_INTENT_THRESHOLDS, USE_SLM_ROUTER
    from app.agent.router.slm_router import get_router
    from app.agent.router.tool_mapper import get_focused_tools
    from app.agent.conversation_state import get_conversation_store

    # If router disabled, use standard path
    if not USE_SLM_ROUTER:
        yield from stream_agent(message, thread_id)
        return

    # Step 1: Get conversation context
    store = get_conversation_store()
    context = store.get(thread_id)

    # Step 2: Classify (with context for multi-task rewriting)
    router = get_router()
    rr = router.classify(message, context=context)
    logger.info("SLM Router: %s", rr)

    # Step 3: Decide path
    if rr.should_fallback:
        logger.info("SLM Router → fallback (%s)", rr.fallback_reason)
        yield from stream_agent(message, thread_id)
        return

    # Use rewritten query if model produced one
    effective_message = rr.rewritten_query if rr.rewritten_query else message
    if rr.rewritten_query:
        logger.info("SLM Router rewrote query: %r → %r", message[:50], rr.rewritten_query[:60])

    # Step 4: Get focused tools (confidence-aware selection)
    focused_tools = get_focused_tools(
        rr.intent, rr.scope, rr.confidence, PER_INTENT_THRESHOLDS
    )

    if focused_tools is None or (not focused_tools and rr.intent != "smalltalk_weather"):
        logger.info("SLM Router → fallback (no tool mapping for %s/%s)", rr.intent, rr.scope)
        yield from stream_agent(message, thread_id)
        return

    focused_tools = focused_tools or []

    logger.info(
        "SLM Router → fast path: %s/%s (conf=%.2f), %d tools: %s",
        rr.intent, rr.scope, rr.confidence,
        len(focused_tools), [t.name for t in focused_tools],
    )

    # Step 5: Create focused agent and stream (with scope enforcement)
    from app.agent.dispatch import router_scope_var
    scope_token = router_scope_var.set(rr.scope)

    max_retries = 2
    last_error = None

    try:
        for attempt in range(max_retries):
            try:
                focused_agent = _create_focused_agent(focused_tools, router_result=rr)
                config = {
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": 15,
                }

                # Collect tool call info for telemetry logging
                # AIMessageChunks split tool_calls across multiple chunks during streaming.
                # tool_call_chunks contains raw args strings; tool_calls contains parsed (incomplete) dicts.
                # We accumulate raw args from tool_call_chunks by id, then merge with ToolMessage output.
                _pending_tool_calls = {}   # id -> {"tool_name", "tool_input_parts"}
                tool_call_logs = []
                for event in focused_agent.stream(
                    {"messages": [{"role": "user", "content": effective_message}]},
                    config,
                    stream_mode="messages",
                ):
                    if event and len(event) >= 2:
                        msg_chunk, metadata = event
                        if isinstance(msg_chunk, ToolMessage):
                            tc_id = getattr(msg_chunk, "tool_call_id", None)
                            pending = _pending_tool_calls.pop(tc_id, None) if tc_id else None
                            tool_call_logs.append({
                                "tool_name": pending["tool_name"] if pending else getattr(msg_chunk, "name", "unknown"),
                                "tool_input": "".join(pending["tool_input_parts"]) if pending else "",
                                "tool_output": str(msg_chunk.content) if msg_chunk.content else "",
                                "success": msg_chunk.status != "error" if hasattr(msg_chunk, "status") else True,
                            })
                            continue
                        # Use tool_call_chunks (raw args strings) instead of tool_calls (parsed partial dicts)
                        chunks = getattr(msg_chunk, "tool_call_chunks", None)
                        if chunks:
                            for tc in chunks:
                                tc_id = tc.get("id")
                                tc_name = tc.get("name")
                                if tc_id and tc_id not in _pending_tool_calls:
                                    _pending_tool_calls[tc_id] = {
                                        "tool_name": tc_name or "unknown",
                                        "tool_input_parts": [],
                                    }
                                target_id = tc_id
                                if not target_id:
                                    # Continuation chunks may have no id; match by index
                                    idx = tc.get("index", 0)
                                    for pid, pval in _pending_tool_calls.items():
                                        target_id = pid
                                        break
                                if target_id and target_id in _pending_tool_calls:
                                    args_str = tc.get("args", "")
                                    if args_str:
                                        _pending_tool_calls[target_id]["tool_input_parts"].append(args_str)
                            continue
                        if metadata.get("langgraph_node") == "agent":
                            if hasattr(msg_chunk, "content") and msg_chunk.content:
                                content = msg_chunk.content
                                # Strip Qwen3 thinking tokens before streaming
                                content = _strip_thinking_tokens(content)
                                if content:
                                    yield content

                # Update ConversationState (intent + turn count) after successful streaming turn.
                # Full entity extraction (location) is only available via invoke; for streaming
                # we update what we know from the router result.
                try:
                    from app.agent.conversation_state import ConversationState
                    import time as _time
                    state = store.get(thread_id) or ConversationState()
                    state.last_intent = rr.intent
                    state.turn_count = (state.turn_count or 0) + 1
                    state.updated_at = _time.time()
                    with store._lock:
                        store._store[thread_id] = state
                except Exception:
                    pass  # State update failure is non-critical

                # Log tool calls to telemetry
                if tool_call_logs:
                    try:
                        from app.agent.telemetry import get_evaluation_logger
                        tel_logger = get_evaluation_logger()
                        turn = (context.turn_count or 0) + 1 if context else 1
                        for tc in tool_call_logs:
                            tel_logger.log_tool_call(
                                session_id=thread_id,
                                turn_number=turn,
                                tool_name=tc["tool_name"],
                                tool_input=tc["tool_input"],
                                tool_output=tc["tool_output"],
                                success=tc["success"],
                            )
                    except Exception:
                        pass  # Telemetry failure is non-critical

                return
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    reset_agent()
                else:
                    raise last_error
    finally:
        router_scope_var.reset(scope_token)


def run_agent_routed(message: str, thread_id: str = "default", *,
                     no_fallback: bool = False,
                     use_rewrite: bool = True) -> dict:
    """Run agent with SLM routing (blocking).

    Always attempts SLM routing (ignores USE_SLM_ROUTER flag).
    Use run_agent() for the baseline (no routing) path.

    Args:
        message: User message
        thread_id: Conversation thread ID
        no_fallback: If True, force routing even for low confidence
                     (structural failures like model_error still fall back)
        use_rewrite: If False, ignore SLM rewritten query and use original
                     message. Used for MT-Context ablation (context injection
                     without query rewriting).

    Returns:
        Agent result dict with '_router' metadata key.
    """
    from app.agent.router.config import PER_INTENT_THRESHOLDS
    from app.agent.router.slm_router import get_router
    from app.agent.router.tool_mapper import get_focused_tools
    from app.agent.conversation_state import get_conversation_store

    # Step 1: Get conversation context
    store = get_conversation_store()
    context = store.get(thread_id)

    # Step 2: Classify (with context for multi-task rewriting)
    router = get_router()
    rr = router.classify(message, context=context)
    logger.info("SLM Router: %s", rr)

    def _router_meta(path, **extra):
        meta = {
            "path": path,
            "intent": rr.intent,
            "scope": rr.scope,
            "confidence": rr.confidence,
            "latency_ms": rr.latency_ms,
            "fallback_reason": rr.fallback_reason,
            "rewritten_query": rr.rewritten_query,
        }
        meta.update(extra)
        return meta

    # Use rewritten query if available (and rewriting not disabled for ablation)
    if use_rewrite and rr.rewritten_query:
        effective_message = rr.rewritten_query
        logger.info("Router rewrite: %r → %r", message[:50], rr.rewritten_query[:60])
    else:
        effective_message = message

    # Step 3: Fallback decision
    if rr.should_fallback:
        can_force = (no_fallback and rr.fallback_reason
                     and rr.fallback_reason.startswith("low_confidence"))
        if not can_force:
            logger.info("SLM Router → fallback (%s)", rr.fallback_reason)
            result = run_agent(message, thread_id)
            result["_router"] = _router_meta("fallback")
            return result

    # Step 4: Get focused tools (confidence-aware)
    focused_tools = get_focused_tools(
        rr.intent, rr.scope, rr.confidence, PER_INTENT_THRESHOLDS
    )

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
    logger.info("SLM Router → routed: %s/%s (conf=%.2f), tools=%s",
                rr.intent, rr.scope, rr.confidence, tool_names)

    # Step 5: Run focused agent (with scope enforcement)
    from app.agent.dispatch import router_scope_var
    scope_token = router_scope_var.set(rr.scope)

    max_retries = 2
    last_error = None

    try:
        for attempt in range(max_retries):
            try:
                focused_agent = _create_focused_agent(focused_tools, router_result=rr, streaming=False)
                config = {
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": 15,
                }
                result = focused_agent.invoke(
                    {"messages": [{"role": "user", "content": effective_message}]}, config
                )
                result["_router"] = _router_meta("routed", focused_tools=tool_names)

                # Step 6: Update ConversationState with extracted entities
                try:
                    store.update_from_result(thread_id, result, rr.intent)
                except Exception as e:
                    logger.debug("ConversationState update failed (non-critical): %s", e)

                return result
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    reset_agent()
                else:
                    raise last_error
    finally:
        router_scope_var.reset(scope_token)
