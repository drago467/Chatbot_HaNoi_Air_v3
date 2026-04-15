"""
clean_router_v6.py — Build v6 router dataset with targeted augmentation.

Changes vs v5:
  - +40 hourly_forecast train  (fix confusion with daily/temperature)
  - +15 smalltalk_weather train (pure casual / greeting / OOS)
  - +15 activity_weather train  (hard negatives — clear specific activities)
  - +20 daily_forecast/rain_query disambiguation train
  - +15 multi-turn context train
  - +12 val samples across weak intents

All location names ONLY from data/processed/dim_ward.csv.
Disambiguation rules from docs/intent_disambiguation_rules.md.

Usage:
  python scripts/data/clean_router_v6.py              # create v6 files
  python scripts/data/clean_router_v6.py --check-only # just print stats
"""

import json
import sys
import unicodedata
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_V5 = ROOT / "data/router/multitask_train_v5_clean.jsonl"
VAL_V5   = ROOT / "data/router/multitask_val_v5_clean.jsonl"
TRAIN_V6 = ROOT / "data/router/multitask_train_v6_clean.jsonl"
VAL_V6   = ROOT / "data/router/multitask_val_v6_clean.jsonl"

CHECK_ONLY = "--check-only" in sys.argv

# ---------------------------------------------------------------------------
# NEW TRAIN SAMPLES — 105 total (manually crafted, verified against rules)
# ---------------------------------------------------------------------------

# ===========================================================================
# A. hourly_forecast — +40 train
#    Signal words: chiều nay, tối nay, sáng nay/mai, đêm nay,
#                  X tiếng nữa, từ Xh đến Yh, mấy giờ, buổi sáng/chiều/tối,
#                  theo từng giờ, chi tiết theo giờ
#    Hard negatives avoided: ngày mai/tuần tới (→daily), bao nhiêu độ (→temperature),
#                            bây giờ thời tiết (→current)
# ===========================================================================
NEW_HOURLY_TRAIN = [
    # --- city scope (10) ---
    {"input": "3 tiếng nữa Hà Nội thời tiết sẽ ra sao?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "city", "confidence": 0.93}},
    {"input": "Từ giờ đến 18h tối nay Hà Nội có mưa không nhỉ?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "city", "confidence": 0.94}},
    {"input": "Hà Nội lúc 20 giờ tối nay thời tiết thế nào?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "city", "confidence": 0.95}},
    {"input": "Tối nay từ 20h đến 22h ở Hà Nội có mưa không?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "city", "confidence": 0.95}},
    {"input": "Hà Nội buổi chiều nay thay đổi theo từng giờ thế nào?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "city", "confidence": 0.92}},
    {"input": "Mấy tiếng tới Hà Nội thời tiết thay đổi như thế nào?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "city", "confidence": 0.93}},
    {"input": "Đêm nay từ 22h trở đi Hà Nội có lạnh không?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "city", "confidence": 0.91}},
    {"input": "Hà Nội sáng nay từ 6h đến 9h dự báo thế nào?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "city", "confidence": 0.94}},
    {"input": "Khoảng 3 tiếng nữa ở Hà Nội thời tiết ra sao?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "city", "confidence": 0.93}},
    {"input": "ha noi sang nay tu 6h den 9h du bao the nao?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "city", "confidence": 0.94}},

    # --- district scope (15) ---
    {"input": "Sáng nay từ 7h đến 10h ở quận Đống Đa thời tiết thế nào?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.95}},
    {"input": "2 tiếng nữa ở Hoàng Mai trời thế nào nhỉ?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.93}},
    {"input": "Từ 14h đến 17h chiều nay quận Cầu Giấy có mưa không?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.95}},
    {"input": "Tối nay từ 19h đến 21h ở Bắc Từ Liêm thời tiết ra sao?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.94}},
    {"input": "Chiều nay mấy giờ sẽ có mưa ở quận Hai Bà Trưng?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.94}},
    {"input": "Đêm nay ở Thanh Xuân từng giờ thế nào?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.93}},
    {"input": "4 tiếng nữa ở Tây Hồ trời có trong không?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.91}},
    {"input": "Lúc 16h chiều nay ở Hà Đông thời tiết thế nào?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.95}},
    {"input": "Sáng sớm mai từ 5h đến 8h quận Long Biên có sương mù không?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.93}},
    {"input": "Từ giờ đến tối ở Hoàn Kiếm thời tiết thay đổi ra sao?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.92}},
    {"input": "Chiều nay khoảng 3-4 giờ ở quận Nam Từ Liêm có nắng không?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.93}},
    {"input": "Tối nay mấy giờ ở Đống Đa bắt đầu lạnh?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.93}},
    {"input": "Lúc 9 giờ sáng nay ở quận Thanh Trì trời thế nào?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.95}},
    {"input": "Từ trưa đến chiều huyện Đông Anh thời tiết biến đổi ra sao?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.93}},
    {"input": "Tối nay từ 20h quận Ba Đình dự báo thế nào?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.94}},

    # --- ward scope (15) ---
    {"input": "Từ 14h đến 16h chiều nay ở phường Bạch Mai, Hai Bà Trưng có mưa không?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.95}},
    {"input": "Sáng mai từ 7h đến 9h ở phường Định Công, Hoàng Mai trời thế nào?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.94}},
    {"input": "Chiều nay mấy giờ mưa ở phường Hoàng Liệt?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.94}},
    {"input": "Từ giờ đến đêm ở phường Yên Hòa, Cầu Giấy thế nào?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.92}},
    {"input": "Sáng nay 6-9h dự báo theo giờ ở phường Láng, Đống Đa?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.95}},
    {"input": "3 tiếng tới ở phường Vĩnh Tuy, Hai Bà Trưng thời tiết ra sao?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.93}},
    {"input": "Đêm nay từ 22h ở phường Khương Đình, Thanh Xuân thế nào?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.92}},
    {"input": "Lúc 15h chiều nay ở xã Bát Tràng, Gia Lâm trời thế nào?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.94}},
    {"input": "Chiều nay từ 13h đến 17h ở phường Ngọc Hà, Ba Đình có nắng không?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.94}},
    {"input": "Tối nay ở phường Tương Mai, Hoàng Mai từng giờ thế nào?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.93}},
    {"input": "Từ 8h đến 12h sáng nay ở phường Đông Ngạc, Bắc Từ Liêm thế nào?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.94}},
    {"input": "Mấy giờ sáng mai ở phường Nghĩa Đô, Cầu Giấy bắt đầu mưa?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.93}},
    {"input": "Chiều tối nay 17-20h ở phường Yên Sở, Hoàng Mai có mưa không?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.94}},
    {"input": "Sáng mai từ 7h ở xã Sóc Sơn, Sóc Sơn dự báo theo giờ thế nào?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.93}},
    {"input": "chieu nay tu 13h den 17h o phuong ngoc ha, ba dinh co nang khong?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.94}},
]

# ===========================================================================
# B. smalltalk_weather — +15 pure casual/greeting/OOS/vague
#    Design: greetings, bot identity, vague weather comments, OOS topics
#    MUST NOT contain: chạy bộ, picnic, tưới cây, giặt đồ, đạp xe, leo núi
# ===========================================================================
NEW_SMALLTALK_TRAIN = [
    # Greetings & farewells (5)
    {"input": "Xin chào bạn ơi, mình mới dùng ứng dụng này lần đầu!",
     "context": None, "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.96}},
    {"input": "Chào buổi sáng chatbot nhé!",
     "context": None, "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.97}},
    {"input": "Hẹn gặp lại bạn vào lần sau nhé!",
     "context": None, "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.97}},
    {"input": "OK được rồi, mình đi đây. Cảm ơn nhé!",
     "context": None, "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.96}},
    {"input": "Bye bạn, thanks nhiều lắm!",
     "context": None, "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.97}},
    # Bot identity & capability (5)
    {"input": "Bạn dự báo được thời tiết mấy ngày tới?",
     "context": None, "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.92}},
    {"input": "Ứng dụng này tên là gì vậy bạn?",
     "context": None, "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.96}},
    {"input": "Mình hỏi bạn về thời tiết vùng biển được không?",
     "context": None, "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.91}},
    {"input": "Bạn có biết chỉ số độ ẩm bao nhiêu thì tốt không?",
     "context": None, "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.89}},
    {"input": "ban tu van thoi tiet hay la robot nhi?",
     "context": None, "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.94}},
    # Casual vague weather comments (5)
    {"input": "Hôm nay thời tiết mát mẻ ghê, dễ chịu quá!",
     "context": None, "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.94}},
    {"input": "Sao dạo này Hà Nội cứ mưa hoài vậy nhỉ?",
     "context": None, "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.93}},
    {"input": "Thời tiết hôm nay thích hợp để ở nhà quá!",
     "context": None, "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.92}},
    {"input": "Trời nắng đẹp thế này mà phải đi làm, tiếc quá!",
     "context": None, "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.93}},
    {"input": "Không khí hôm nay trong lành quá bạn ơi!",
     "context": None, "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.94}},
]

# ===========================================================================
# C. activity_weather — +15 hard negatives (clear specific outdoor activities)
#    Per design: MUST have named activity (chạy bộ, picnic, giặt đồ, tưới cây...)
#    Scope: city/district/ward based on location in query
# ===========================================================================
NEW_ACTIVITY_TRAIN = [
    # Chạy bộ / Thể dục ngoài trời (5)
    {"input": "Sáng nay có thể đi chạy bộ ở phường Đống Đa không?",
     "context": None, "output": {"intent": "activity_weather", "scope": "ward", "confidence": 0.93}},
    {"input": "Hôm nay thích hợp để tập gym ngoài trời ở quận Cầu Giấy không?",
     "context": None, "output": {"intent": "activity_weather", "scope": "district", "confidence": 0.91}},
    {"input": "Chiều nay chạy bộ ở Hồ Tây có ổn không?",
     "context": None, "output": {"intent": "activity_weather", "scope": "district", "confidence": 0.92}},
    {"input": "Sáng mai có thể đi tập thể dục ngoài trời ở Hà Nội không?",
     "context": None, "output": {"intent": "activity_weather", "scope": "city", "confidence": 0.93}},
    {"input": "Cuối tuần này đạp xe quanh Hồ Hoàn Kiếm thời tiết có ổn không?",
     "context": None, "output": {"intent": "activity_weather", "scope": "district", "confidence": 0.91}},
    # Picnic / Dã ngoại (5)
    {"input": "Thứ 7 này đi picnic ở phường Phú Thượng, Tây Hồ có phù hợp không?",
     "context": None, "output": {"intent": "activity_weather", "scope": "ward", "confidence": 0.92}},
    {"input": "Cuối tuần này tổ chức BBQ ngoài trời ở Long Biên thời tiết sao?",
     "context": None, "output": {"intent": "activity_weather", "scope": "district", "confidence": 0.91}},
    {"input": "Ngày mai đi chơi công viên ở Ba Đình thời tiết có thích hợp không?",
     "context": None, "output": {"intent": "activity_weather", "scope": "district", "confidence": 0.92}},
    {"input": "Hôm nay leo núi ở khu vực Ba Vì có mưa không?",
     "context": None, "output": {"intent": "activity_weather", "scope": "district", "confidence": 0.90}},
    {"input": "Chiều nay đi câu cá ở Hồ Tây có ổn không?",
     "context": None, "output": {"intent": "activity_weather", "scope": "district", "confidence": 0.91}},
    # Giặt đồ / Phơi đồ / Làm vườn (5)
    {"input": "Hôm nay có thể phơi chăn màn ở sân thượng quận Thanh Xuân không?",
     "context": None, "output": {"intent": "activity_weather", "scope": "district", "confidence": 0.93}},
    {"input": "Sáng mai nên giặt đồ sớm không, trời có nắng không?",
     "context": None, "output": {"intent": "activity_weather", "scope": "city", "confidence": 0.92}},
    {"input": "Chiều nay phơi đồ được không, Hà Nội có mưa không?",
     "context": None, "output": {"intent": "activity_weather", "scope": "city", "confidence": 0.93}},
    {"input": "Hôm nay tưới cây ở ban công được không, trời có mưa không?",
     "context": None, "output": {"intent": "activity_weather", "scope": "city", "confidence": 0.91}},
    {"input": "Cuối tuần này rửa xe ngoài trời ở quận Hà Đông có bị mưa không?",
     "context": None, "output": {"intent": "activity_weather", "scope": "district", "confidence": 0.91}},
]

# ===========================================================================
# D. daily_forecast & rain_query disambiguation — +20 train
#    daily: hỏi tổng quan nhiều ngày (có thể mention mưa nhưng focus multi-day)
#    rain:  hỏi cụ thể về mưa (timing, probability, duration)
# ===========================================================================
NEW_DAILY_TRAIN = [
    # daily_forecast — multi-day overview, may mention rain (10)
    {"input": "3 ngày tới Hà Nội thời tiết thế nào, có hay bị mưa không?",
     "context": None, "output": {"intent": "daily_forecast", "scope": "city", "confidence": 0.88}},
    {"input": "Cuối tuần này thời tiết quận Thanh Xuân thế nào?",
     "context": None, "output": {"intent": "daily_forecast", "scope": "district", "confidence": 0.92}},
    {"input": "Từ thứ Hai đến thứ Sáu tuần này ở Hà Nội thế nào?",
     "context": None, "output": {"intent": "daily_forecast", "scope": "city", "confidence": 0.91}},
    {"input": "Tuần tới thời tiết Hà Nội có ổn không, hay mưa nhiều?",
     "context": None, "output": {"intent": "daily_forecast", "scope": "city", "confidence": 0.88}},
    {"input": "Ngày mai và ngày kia ở quận Hoàng Mai thế nào?",
     "context": None, "output": {"intent": "daily_forecast", "scope": "district", "confidence": 0.91}},
    {"input": "Thứ 7 và chủ nhật này Hà Nội thời tiết ra sao?",
     "context": None, "output": {"intent": "daily_forecast", "scope": "city", "confidence": 0.92}},
    {"input": "Thời tiết Hà Nội cả tuần tới thế nào?",
     "context": None, "output": {"intent": "daily_forecast", "scope": "city", "confidence": 0.92}},
    {"input": "Từ thứ Tư đến thứ Bảy ở quận Cầu Giấy dự báo thế nào?",
     "context": None, "output": {"intent": "daily_forecast", "scope": "district", "confidence": 0.91}},
    {"input": "7 ngày tới ở phường Yên Hòa, Cầu Giấy thời tiết thay đổi ra sao?",
     "context": None, "output": {"intent": "daily_forecast", "scope": "ward", "confidence": 0.91}},
    {"input": "du bao thoi tiet ha noi ca tuan toi the nao?",
     "context": None, "output": {"intent": "daily_forecast", "scope": "city", "confidence": 0.92}},
]

NEW_RAIN_TRAIN = [
    # rain_query — specific focus on rain (10)
    {"input": "Ngày mai mấy giờ có mưa ở quận Đống Đa?",
     "context": None, "output": {"intent": "rain_query", "scope": "district", "confidence": 0.94}},
    {"input": "Sáng mai Hà Nội có mưa không, có cần mang ô không?",
     "context": None, "output": {"intent": "rain_query", "scope": "city", "confidence": 0.93}},
    {"input": "Chiều mai ở phường Hoàng Liệt, Hoàng Mai có mưa rào không?",
     "context": None, "output": {"intent": "rain_query", "scope": "ward", "confidence": 0.94}},
    {"input": "Tỉ lệ mưa ở Hà Nội ngày mai bao nhiêu phần trăm?",
     "context": None, "output": {"intent": "rain_query", "scope": "city", "confidence": 0.93}},
    {"input": "Mưa tại quận Cầu Giấy hôm nay sẽ kéo dài đến mấy giờ?",
     "context": None, "output": {"intent": "rain_query", "scope": "district", "confidence": 0.93}},
    {"input": "Hôm nay ở phường Thanh Xuân, Thanh Xuân có mưa lúc nào không?",
     "context": None, "output": {"intent": "rain_query", "scope": "ward", "confidence": 0.93}},
    {"input": "Ngày mai có phải mang ô đi học không?",
     "context": None, "output": {"intent": "rain_query", "scope": "city", "confidence": 0.92}},
    {"input": "Khả năng mưa ở quận Long Biên tối nay bao nhiêu phần trăm?",
     "context": None, "output": {"intent": "rain_query", "scope": "district", "confidence": 0.92}},
    {"input": "Mưa ngày mai ở Hà Nội kéo dài từ mấy giờ đến mấy giờ?",
     "context": None, "output": {"intent": "rain_query", "scope": "city", "confidence": 0.93}},
    {"input": "Chiều tối nay ở phường Lĩnh Nam, Hoàng Mai có mưa không?",
     "context": None, "output": {"intent": "rain_query", "scope": "ward", "confidence": 0.93}},
]

# ===========================================================================
# E. Multi-turn context samples — +15 train
#    Context format: {"location": "...", "intent": "...", "turn": 1}
#    Output includes rewritten_query when user drops location
# ===========================================================================
NEW_MULTITURN_TRAIN = [
    # hourly follow-up — slot changes (6)
    {
        "input": "Tối nay thì ra sao?",
        "context": {"location": "Cầu Giấy", "intent": "hourly_forecast", "turn": 1},
        "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.92,
                   "rewritten_query": "Tối nay quận Cầu Giấy thời tiết thế nào?"}
    },
    {
        "input": "Buổi chiều thì sao?",
        "context": {"location": "Bạch Mai, Hai Bà Trưng", "intent": "hourly_forecast", "turn": 1},
        "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.92,
                   "rewritten_query": "Chiều nay phường Bạch Mai, Hai Bà Trưng thời tiết thế nào?"}
    },
    {
        "input": "Lúc 20h thế nào?",
        "context": {"location": "Hà Nội", "intent": "hourly_forecast", "turn": 1},
        "output": {"intent": "hourly_forecast", "scope": "city", "confidence": 0.93,
                   "rewritten_query": "Tối nay lúc 20h Hà Nội thời tiết thế nào?"}
    },
    {
        "input": "Từ 22h thì sao?",
        "context": {"location": "Hoàng Mai", "intent": "hourly_forecast", "turn": 1},
        "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.92,
                   "rewritten_query": "Từ 22h tối nay quận Hoàng Mai thời tiết ra sao?"}
    },
    {
        "input": "Khoảng 2-3 giờ?",
        "context": {"location": "Láng, Đống Đa", "intent": "hourly_forecast", "turn": 1},
        "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.93,
                   "rewritten_query": "Khoảng 14-15h chiều nay ở phường Láng, Đống Đa thời tiết thế nào?"}
    },
    {
        "input": "Còn buổi trưa?",
        "context": {"location": "Tây Hồ", "intent": "hourly_forecast", "turn": 1},
        "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.92,
                   "rewritten_query": "Trưa nay quận Tây Hồ thời tiết thế nào?"}
    },
    # daily follow-up — day changes (5)
    {
        "input": "Riêng thứ Tư thì sao?",
        "context": {"location": "Hà Nội", "intent": "daily_forecast", "turn": 1},
        "output": {"intent": "daily_forecast", "scope": "city", "confidence": 0.92,
                   "rewritten_query": "Thứ Tư tuần tới Hà Nội thời tiết thế nào?"}
    },
    {
        "input": "Ngày cuối tuần?",
        "context": {"location": "Thanh Xuân", "intent": "daily_forecast", "turn": 1},
        "output": {"intent": "daily_forecast", "scope": "district", "confidence": 0.91,
                   "rewritten_query": "Cuối tuần này quận Thanh Xuân thời tiết thế nào?"}
    },
    {
        "input": "Chủ nhật thì ra sao?",
        "context": {"location": "Hà Nội", "intent": "daily_forecast", "turn": 1},
        "output": {"intent": "daily_forecast", "scope": "city", "confidence": 0.92,
                   "rewritten_query": "Chủ nhật này Hà Nội thời tiết thế nào?"}
    },
    {
        "input": "Còn ngày mai?",
        "context": {"location": "Yên Hòa, Cầu Giấy", "intent": "daily_forecast", "turn": 1},
        "output": {"intent": "daily_forecast", "scope": "ward", "confidence": 0.93,
                   "rewritten_query": "Ngày mai phường Yên Hòa, Cầu Giấy thời tiết thế nào?"}
    },
    {
        "input": "Ngày mai trước đi",
        "context": {"location": "Hoàng Mai", "intent": "daily_forecast", "turn": 1},
        "output": {"intent": "daily_forecast", "scope": "district", "confidence": 0.93,
                   "rewritten_query": "Ngày mai quận Hoàng Mai thời tiết thế nào?"}
    },
    # smalltalk → weather follow-up (4)
    {
        "input": "Hôm nay có mưa không?",
        "context": {"location": "Hà Nội", "intent": "smalltalk_weather", "turn": 1},
        "output": {"intent": "rain_query", "scope": "city", "confidence": 0.92,
                   "rewritten_query": "Hôm nay Hà Nội có mưa không?"}
    },
    {
        "input": "Vậy cho mình biết 3 ngày tới nào",
        "context": {"location": "Hà Nội", "intent": "smalltalk_weather", "turn": 1},
        "output": {"intent": "daily_forecast", "scope": "city", "confidence": 0.91,
                   "rewritten_query": "Dự báo thời tiết Hà Nội 3 ngày tới thế nào?"}
    },
    {
        "input": "Quận Cầu Giấy thì sao?",
        "context": {"location": "Hà Nội", "intent": "smalltalk_weather", "turn": 1},
        "output": {"intent": "current_weather", "scope": "district", "confidence": 0.92,
                   "rewritten_query": "Quận Cầu Giấy hôm nay thời tiết thế nào?"}
    },
    {
        "input": "Sáng nay Hà Nội có nắng không?",
        "context": {"location": "Hà Nội", "intent": "smalltalk_weather", "turn": 1},
        "output": {"intent": "current_weather", "scope": "city", "confidence": 0.92,
                   "rewritten_query": "Sáng nay Hà Nội có nắng không?"}
    },
]

# ---------------------------------------------------------------------------
# NEW VAL SAMPLES — 12 total
# ---------------------------------------------------------------------------

NEW_HOURLY_VAL = [
    {"input": "Chiều nay khoảng 2-3 giờ Hà Nội có mưa không?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "city", "confidence": 0.93}},
    {"input": "Sáng mai từ 6h đến 9h quận Hoàn Kiếm trời thế nào?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.95}},
    {"input": "Tối nay mấy giờ ở Cầu Giấy bắt đầu se lạnh?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "district", "confidence": 0.93}},
    {"input": "Từ 15h đến 18h chiều nay ở phường Phương Liệt, Thanh Xuân có nắng không?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.94}},
    {"input": "3 tiếng nữa ở phường Ô Chợ Dừa, Đống Đa thời tiết thế nào?",
     "context": None, "output": {"intent": "hourly_forecast", "scope": "ward", "confidence": 0.93}},
]

NEW_SMALLTALK_VAL = [
    # smalltalk hard case — casual, no specific activity
    {"input": "Ứng dụng này lấy dữ liệu thời tiết từ đâu vậy?",
     "context": None, "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.92}},
    {"input": "Hôm nay không khí mát quá, thích thật!",
     "context": None, "output": {"intent": "smalltalk_weather", "scope": "city", "confidence": 0.94}},
]

NEW_ACTIVITY_VAL = [
    # activity hard case — specific activity
    {"input": "Sáng mai đi chạy bộ ở quận Tây Hồ được không?",
     "context": None, "output": {"intent": "activity_weather", "scope": "district", "confidence": 0.93}},
    {"input": "Cuối tuần này phơi đồ ngoài trời ở phường Xuân Đỉnh, Bắc Từ Liêm có ổn không?",
     "context": None, "output": {"intent": "activity_weather", "scope": "ward", "confidence": 0.92}},
]

NEW_DAILY_VAL = [
    {"input": "Tuần này Hà Nội thời tiết ra sao, có nhiều mưa không?",
     "context": None, "output": {"intent": "daily_forecast", "scope": "city", "confidence": 0.88}},
    {"input": "5 ngày tới ở phường Phúc Lợi, Long Biên thời tiết thay đổi thế nào?",
     "context": None, "output": {"intent": "daily_forecast", "scope": "ward", "confidence": 0.90}},
]

NEW_RAIN_VAL = [
    {"input": "Sáng mai có mưa ở quận Đống Đa không, cần mang ô không?",
     "context": None, "output": {"intent": "rain_query", "scope": "district", "confidence": 0.93}},
]

# ---------------------------------------------------------------------------
# Combine all new samples
# ---------------------------------------------------------------------------
ALL_NEW_TRAIN = (
    NEW_HOURLY_TRAIN
    + NEW_SMALLTALK_TRAIN
    + NEW_ACTIVITY_TRAIN
    + NEW_DAILY_TRAIN
    + NEW_RAIN_TRAIN
    + NEW_MULTITURN_TRAIN
)

ALL_NEW_VAL = (
    NEW_HOURLY_VAL
    + NEW_SMALLTALK_VAL
    + NEW_ACTIVITY_VAL
    + NEW_DAILY_VAL
    + NEW_RAIN_VAL
)

# ---------------------------------------------------------------------------
# Quality check utilities
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase + remove extra whitespace + normalize unicode."""
    text = unicodedata.normalize("NFC", text.strip().lower())
    return re.sub(r"\s+", " ", text)


def char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    """Char n-gram Jaccard similarity between two strings."""
    def ngrams(s, n):
        return set(s[i:i+n] for i in range(len(s) - n + 1))
    na, nb = ngrams(a, n), ngrams(b, n)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


def check_leakage(train_inputs, val_inputs, threshold=0.85):
    """Check for exact and near-duplicate leakage between train and val."""
    exact = 0
    fuzzy = []
    train_norm = [normalize(t) for t in train_inputs]
    val_norm   = [normalize(v) for v in val_inputs]
    train_set  = set(train_norm)

    for vi, vn in enumerate(val_norm):
        if vn in train_set:
            exact += 1
            print(f"  [EXACT LEAK] val[{vi}]: {val_inputs[vi][:60]}")
        else:
            for ti, tn in enumerate(train_norm):
                sim = char_ngram_jaccard(vn, tn)
                if sim >= threshold:
                    fuzzy.append((vi, ti, sim))
                    print(f"  [FUZZY LEAK {sim:.2f}] val: {val_inputs[vi][:50]} | train: {train_inputs[ti][:50]}")
    return exact, len(fuzzy)


def check_internal_dupes(inputs, label=""):
    """Check for exact duplicates within a single list."""
    norm = [normalize(t) for t in inputs]
    counts = Counter(norm)
    dupes = {k: v for k, v in counts.items() if v > 1}
    if dupes:
        for k, v in dupes.items():
            print(f"  [DUPE x{v} in {label}]: {k[:60]}")
    return len(dupes)


def print_distribution(samples, label):
    intent_counts = Counter(s["output"]["intent"] for s in samples)
    scope_counts  = Counter(s["output"]["scope"]  for s in samples)
    total = len(samples)
    print(f"\n  {label} ({total} samples):")
    print("  Intent distribution:")
    for k, v in sorted(intent_counts.items(), key=lambda x: x[1]):
        print(f"    {k}: {v}")
    print("  Scope distribution:")
    for k, v in sorted(scope_counts.items()):
        pct = 100 * v / total if total else 0
        print(f"    {k}: {v} ({pct:.1f}%)")
    # Check all intents have val ≥ 20
    if label.startswith("VAL"):
        under = {k: v for k, v in intent_counts.items() if v < 20}
        if under:
            print(f"  [WARN] Intents with val < 20: {under}")
        else:
            print("  [OK] All intents have val >= 20")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_jsonl(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_jsonl(samples, path):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main():
    print("=" * 60)
    print("Router Dataset v6 — Quality Check")
    print("=" * 60)

    train_v5 = load_jsonl(TRAIN_V5)
    val_v5   = load_jsonl(VAL_V5)

    train_v6 = train_v5 + ALL_NEW_TRAIN
    val_v6   = val_v5   + ALL_NEW_VAL

    train_inputs = [s["input"] for s in train_v6]
    val_inputs   = [s["input"] for s in val_v6]

    print(f"\n[COUNTS]")
    print(f"  Train v5: {len(train_v5)} | New: {len(ALL_NEW_TRAIN)} | Total v6: {len(train_v6)}")
    print(f"  Val   v5: {len(val_v5)}  | New: {len(ALL_NEW_VAL)}  | Total v6: {len(val_v6)}")

    print(f"\n[NEW TRAIN SAMPLE BREAKDOWN]")
    new_counts = Counter(s["output"]["intent"] for s in ALL_NEW_TRAIN)
    for k, v in sorted(new_counts.items()):
        print(f"  {k}: +{v}")

    print(f"\n[INTERNAL DUPLICATE CHECK — new train samples]")
    nd = check_internal_dupes([s["input"] for s in ALL_NEW_TRAIN], "new_train")
    if nd == 0:
        print("  [OK] No internal duplicates")

    print(f"\n[LEAKAGE CHECK — new val vs all train (threshold=0.85)]")
    new_val_inputs = [s["input"] for s in ALL_NEW_VAL]
    exact, fuzzy = check_leakage(train_inputs, new_val_inputs)
    if exact == 0 and fuzzy == 0:
        print("  [OK] Zero leakage (exact + fuzzy)")
    else:
        print(f"  [WARN] Exact: {exact}, Fuzzy: {fuzzy}")

    print(f"\n[FULL LEAKAGE CHECK — val_v6 vs train_v6 (sample of 50 val)]")
    import random
    random.seed(42)
    sample_val = random.sample(val_inputs, min(50, len(val_inputs)))
    exact2, fuzzy2 = check_leakage(train_inputs, sample_val, threshold=0.85)
    if exact2 == 0 and fuzzy2 == 0:
        print("  [OK] Full leakage check passed")

    print_distribution(train_v6, "TRAIN v6")
    print_distribution(val_v6, "VAL v6")

    if not CHECK_ONLY:
        save_jsonl(train_v6, TRAIN_V6)
        save_jsonl(val_v6, VAL_V6)
        print(f"\n[SAVED]")
        print(f"  {TRAIN_V6} ({len(train_v6)} samples)")
        print(f"  {VAL_V6} ({len(val_v6)} samples)")
    else:
        print(f"\n[CHECK ONLY — no files written]")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
