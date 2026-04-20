"""Forecast tools — hourly, daily, rain_timeline, best_time.

Tất cả đều hỗ trợ 3 tier (ward/district/city) nhất quán thông qua dispatch_forecast.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============== Tool: get_hourly_forecast ==============

class GetHourlyForecastInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID (ví dụ: ID_00169)")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")
    hours: int = Field(default=24, description="Số giờ dự báo (1-48)")


@tool(args_schema=GetHourlyForecastInput)
def get_hourly_forecast(ward_id: str = None, location_hint: str = None, hours: int = 24) -> dict:
    """DỰ BÁO THEO GIỜ cho 1-48 giờ tới. MAX `hours` = 48.

    DÙNG KHI: user hỏi khung giờ CỤ THỂ trong 48h:
        "chiều/tối/đêm nay", "sáng mai", "X giờ tối nay", "vài giờ tới", "6-9h sáng mai".
        Set `hours` ĐỦ phủ khung user hỏi (vd NOW=16h & "8pm-midnight" → `hours` ≥ 10;
        NOW=11h & "6-9h sáng mai" → `hours` ≥ 24).

    KHÔNG DÙNG KHI:
        - "bây giờ / hiện tại" → get_current_weather (snapshot).
        - "ngày cụ thể / nhiều ngày" (>48h) → get_daily_forecast.
        - "hôm qua / ngày đã qua" → get_weather_history.
        - Khoảng vượt 48h → refuse hoặc chuyển daily_forecast.

    Returns: Flat VN dict, key `"dự báo"` = list per-hour flat VN dict
    (`"thời điểm"`, `"thời tiết"`, `"nhiệt độ"`, `"độ ẩm"`, `"xác suất mưa"`,
    `"cường độ mưa"` khi mưa, `"gió"`, `"mây"`) + `"tóm tắt tổng"` + `"ghi chú dữ liệu"`.

    DÙNG KHI: user hỏi về chiều nay, tối nay, sáng mai, vài giờ tới,
    mưa lúc mấy giờ, nhiệt độ tối nay, gió đêm nay, khoảng thời gian cụ thể.
    Hỗ trợ: phường/xã, quận/huyện, toàn Hà Nội (tự động dispatch).

    ⚠ QUAN TRỌNG về param `hours`:
    - `hours` là KHOẢNG THỜI GIAN (range từ NOW đến NOW+hours), KHÔNG phải 1 giờ cụ thể.
    - hours=3 → trả 3 record đầu tiên từ NOW (KHÔNG phải "3 giờ user hỏi").
    - Nếu user hỏi khung giờ tương lai xa (ví dụ "6h-9h SÁNG MAI"):
      → phải set hours đủ lớn để cover khung đó (thường hours=24-30).
      → Sau đó đọc forecasts array, PICK entries có `time_ict` match khung user hỏi.
      → KHÔNG báo cáo data của 3 giờ đầu tiên khi user hỏi sáng mai.

    VÍ DỤ:
    - "chiều nay có mưa không" (13-18h TODAY, NOW=8am) → hours=10-11 (đủ cover đến 18h)
    - "9h tối nay Cầu Giấy nhiệt độ" (21h TODAY) → hours=13+, pick entry có time_ict='21:00'
    - "6h-9h sáng mai Long Biên sương mù" (NOW=10am) → hours=24, pick entries 06-09:00 ngày mai
    - "3 giờ tới có giông không" → hours=3 (đúng nghĩa 3 giờ từ NOW)

    KHÔNG DÙNG KHI: hỏi cả ngày mai/tuần này (dùng get_daily_forecast),
    hỏi hiện tại (dùng get_current_weather), hỏi mưa đến bao giờ (dùng get_rain_timeline).
    """
    from app.agent.dispatch import dispatch_forecast
    from app.dal.weather_dal import get_hourly_forecast as dal_ward
    from app.dal.weather_aggregate_dal import (
        get_district_hourly_forecast as dal_district,
        get_city_hourly_forecast as dal_city,
    )

    from app.agent.tools.output_builder import build_hourly_forecast_output

    hours = max(1, min(hours, 48))
    raw = dispatch_forecast(
        ward_id=ward_id,
        location_hint=location_hint,
        ward_fn=dal_ward,
        district_fn=dal_district,
        city_fn=dal_city,
        ward_args={"hours": hours},
        district_args={"hours": hours},
        city_args={"hours": hours},
        forecast_type="hourly",
        default_scope="city",
    )
    return build_hourly_forecast_output(raw)


# ============== Tool: get_daily_forecast ==============

class GetDailyForecastInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID (ví dụ: ID_00169)")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")
    days: int = Field(default=7, description="Số ngày dự báo (1-8)")
    start_date: Optional[str] = Field(default=None, description="Ngày bắt đầu dự báo (YYYY-MM-DD). Mặc định: hôm nay. Dùng để offset khi user hỏi 'ngày mai', '3 ngày nữa', v.v.")


@tool(args_schema=GetDailyForecastInput)
def get_daily_forecast(ward_id: str = None, location_hint: str = None, days: int = 7, start_date: str = None) -> dict:
    """DỰ BÁO THEO NGÀY cho khoảng 1-8 ngày (overview per-day). MAX `days` = 8.

    DÙNG KHI: user hỏi nguyên ngày hoặc nhiều ngày tương lai:
        "ngày mai", "ngày kia", "thứ X", "3 ngày tới", "cuối tuần này",
        "tuần tới" (phần trong 8 ngày).

    KHÔNG DÙNG KHI:
        - "cả ngày X chi tiết sáng/trưa/chiều/tối" 1 ngày duy nhất → get_daily_summary.
        - "chiều/tối/X giờ" trong 48h → get_hourly_forecast.
        - "hôm qua" → get_weather_history.
        - "bây giờ" → get_current_weather.
        - Khoảng vượt 8 ngày → refuse nói rõ "tối đa 8 ngày", KHÔNG bịa.

    LOGIC param:
        - `start_date` (ISO YYYY-MM-DD): mặc định hôm nay. User hỏi "ngày mai" → PHẢI
          truyền `start_date=tomorrow_iso`, `days=1`. KHÔNG gọi days=1 thiếu start_date.
        - `days`: số ngày bao gồm start_date. 1 ≤ days ≤ 8.

    Returns: Flat VN dict:
        - `"dự báo"`: list per-day flat VN dict (`"ngày"` DD/MM/YYYY `(Thứ X)`,
          `"thời tiết"`, `"nhiệt độ"` Thấp-Cao, `"nhiệt độ theo ngày"` Sáng/Trưa/Chiều/Tối
          overview, `"xác suất mưa"`, `"tổng lượng mưa"` mm, `"gió"`, `"UV"`, `"mọc-lặn"`).
        - `"tổng hợp"`: pre-computed superlatives (ngày nóng/mát/mưa nhiều/ít nhất).
          COPY thẳng, KHÔNG tự argmax qua list.
    """
    from app.agent.dispatch import dispatch_forecast
    from app.dal.weather_dal import get_daily_forecast as dal_ward
    from app.dal.weather_aggregate_dal import (
        get_district_daily_forecast as dal_district,
        get_city_daily_forecast as dal_city,
    )

    from app.agent.tools.output_builder import build_daily_forecast_output

    days = max(1, min(days, 8))
    extra_args = {"days": days}
    if start_date:
        extra_args["start_date"] = start_date
    raw = dispatch_forecast(
        ward_id=ward_id,
        location_hint=location_hint,
        ward_fn=dal_ward,
        district_fn=dal_district,
        city_fn=dal_city,
        ward_args=extra_args,
        district_args=extra_args,
        city_args=extra_args,
        forecast_type="daily",
        default_scope="city",
    )
    return build_daily_forecast_output(raw)


# ============== Tool: get_rain_timeline ==============

class GetRainTimelineInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")
    hours: int = Field(default=24, description="Số giờ scan (1-48)")


@tool(args_schema=GetRainTimelineInput)
def get_rain_timeline(ward_id: str = None, location_hint: str = None, hours: int = 24) -> dict:
    """TIMELINE MƯA: đợt mưa bắt đầu/kết thúc/cường độ đỉnh trong 1-48h. MAX `hours` = 48.

    DÙNG KHI: user hỏi MỐC mưa trong 48h:
        "lúc nào mưa?", "mưa đến bao giờ?", "trời tạnh lúc nào?", "chiều có mưa không?".
        Phải đọc timestamp start/end trong output — gán vào đúng NGÀY user hỏi.

    KHÔNG DÙNG KHI:
        - "tổng lượng mưa ngày / tháng" → dùng get_daily_forecast / get_weather_period
          (có `"tổng lượng mưa"` mm/ngày). Key `"cường độ đỉnh"` ở đây là mm/h TẠI 1 GIỜ,
          KHÔNG phải tổng.
        - "ngày mai có mưa không" nếu data chỉ 48h và hỏi ngày > 48h tới → refuse với limit.

    Returns: Flat VN dict với `"đợt mưa"` = list `[{"bắt đầu", "kết thúc", "xác suất cao nhất",
    "cường độ đỉnh" (mm/h)}]`, `"tổng số đợt"`, `"tóm tắt"` (next_rain / next_clear).
    """
    from app.agent.dispatch import dispatch_forecast
    from app.dal.weather_dal import get_hourly_forecast as dal_ward_hourly
    from app.dal.weather_aggregate_dal import (
        get_district_hourly_forecast as dal_district_hourly,
        get_city_hourly_forecast as dal_city_hourly,
    )
    from app.dal.weather_dal import analyze_rain_from_forecasts

    hours = max(1, min(hours, 48))

    # We need raw forecasts first, then analyze
    result = dispatch_forecast(
        ward_id=ward_id,
        location_hint=location_hint,
        ward_fn=dal_ward_hourly,
        district_fn=dal_district_hourly,
        city_fn=dal_city_hourly,
        ward_args={"hours": hours},
        district_args={"hours": hours},
        city_args={"hours": hours},
        forecast_type="hourly",
        default_scope="city",
    )

    from app.agent.tools.output_builder import build_rain_timeline_output, build_error_output

    if result.get("error"):
        return build_error_output(result)

    forecasts = result.get("forecasts", [])
    rain_analysis = analyze_rain_from_forecasts(forecasts, hours)
    rain_analysis["resolved_location"] = result.get("resolved_location", {})
    rain_analysis["level"] = result.get("level", "city")
    rain_analysis["data_coverage"] = result.get("data_coverage")
    return build_rain_timeline_output(rain_analysis)


# ============== Tool: get_best_time ==============

class GetBestTimeInput(BaseModel):
    activity: str = Field(description="Hoạt động: chay_bo, picnic, bike, chup_anh, du_lich, cam_trai, ...")
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")
    hours: int = Field(default=24, description="Số giờ quét (1-48)")


@tool(args_schema=GetBestTimeInput)
def get_best_time(activity: str, ward_id: str = None, location_hint: str = None, hours: int = 24) -> dict:
    """Tìm KHUNG GIỜ TỐT NHẤT để thực hiện hoạt động ngoài trời. MAX hours=48.

    DÙNG KHI: "mấy giờ chạy bộ tốt?", "lúc nào đi chơi đẹp nhất?".

    Returns: Flat VN dict với `"giờ tốt nhất": [top 5 slots]`, `"giờ kém nhất": [bottom 3]`,
    mỗi slot có `"thời điểm", "điểm" (N/100), "nhiệt độ", "xác suất mưa", "ghi chú"`.
    """
    from app.agent.dispatch import dispatch_forecast
    from app.dal.weather_dal import get_hourly_forecast as dal_ward_hourly
    from app.dal.weather_aggregate_dal import (
        get_district_hourly_forecast as dal_district_hourly,
        get_city_hourly_forecast as dal_city_hourly,
    )
    from app.dal.activity_dal import get_best_time_for_activity

    hours = max(1, min(hours, 48))

    # Get forecasts first
    result = dispatch_forecast(
        ward_id=ward_id,
        location_hint=location_hint,
        ward_fn=dal_ward_hourly,
        district_fn=dal_district_hourly,
        city_fn=dal_city_hourly,
        ward_args={"hours": hours},
        district_args={"hours": hours},
        city_args={"hours": hours},
        forecast_type="hourly",
        default_scope="city",
    )

    from app.agent.tools.output_builder import build_best_time_output, build_error_output

    if result.get("error"):
        return build_error_output(result)

    forecasts = result.get("forecasts", [])
    best = get_best_time_for_activity(activity, forecasts=forecasts, hours=hours)
    best["resolved_location"] = result.get("resolved_location", {})
    best["level"] = result.get("level", "city")
    return build_best_time_output(best)
