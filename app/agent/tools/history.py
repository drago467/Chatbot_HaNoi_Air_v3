"""History tools — weather_history, daily_summary, weather_period.

Tất cả đều hỗ trợ 3 tier nhất quán.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============== Tool: get_weather_history ==============

class GetWeatherHistoryInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")
    date: str = Field(description="Ngày (YYYY-MM-DD)")


@tool(args_schema=GetWeatherHistoryInput)
def get_weather_history(ward_id: str = None, location_hint: str = None, date: str = None) -> dict:
    """LỊCH SỬ thời tiết 1 ngày đã qua (≤ 14 ngày gần nhất). PAST-only.

    DÙNG KHI: user hỏi ngày ĐÃ QUA:
        "hôm qua" (truyền `date=<yesterday_iso>`), "hôm kia", "ngày 15/3", "tuần trước".

    KHÔNG DÙNG KHI:
        - "hôm nay / bây giờ" → get_current_weather hoặc get_daily_summary.
        - "ngày mai / tương lai" → get_daily_forecast.
        - So today vs yesterday → get_compare_with_yesterday (1 tool thay vì 2).
        - Ngày vượt 14-ngày past → refuse với limit.

    Returns: Flat VN dict (`"ngày"` DD/MM/YYYY `(Thứ X)`, `"thời tiết chung"`, `"nhiệt độ"`,
    `"cảm giác"` CHỈ ward, `"nhiệt độ min-max"`, `"độ ẩm"`, `"điểm sương"`,
    `"tổng lượng mưa"`, `"gió"`, `"UV"`).

    ⚠ Ward-level history CHỈ có `wind_gust` (không có `wind_speed` avg). Value `"gió"` sẽ là
    "Giật X m/s" — KHÔNG diễn giải thành "TB X m/s" hay "wind avg".
    """
    from app.agent.dispatch import resolve_and_dispatch
    from app.dal.weather_dal import (
        get_weather_history as dal_ward,
        get_district_weather_history as dal_district,
        get_city_weather_history as dal_city,
    )

    from app.agent.tools.output_builder import build_weather_history_output
    raw = resolve_and_dispatch(
        ward_id=ward_id,
        location_hint=location_hint,
        default_scope="city",
        ward_fn=dal_ward,
        district_fn=lambda district_name: dal_district(district_name, date),
        city_fn=lambda: dal_city(date),
        ward_args={"date": date},
        label="lịch sử thời tiết",
    )
    return build_weather_history_output(raw, date_hint=date)


# ============== Tool: get_daily_summary ==============

class GetDailySummaryInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")
    date: Optional[str] = Field(default=None, description="Ngày (YYYY-MM-DD), mặc định hôm nay")


@tool(args_schema=GetDailySummaryInput)
def get_daily_summary(ward_id: str = None, location_hint: str = None, date: str = None) -> dict:
    """TỔNG HỢP CHI TIẾT 1 NGÀY DUY NHẤT với 4 khung sáng/trưa/chiều/tối (ward only).

    DÙNG KHI: user cần nhìn CẢ NGÀY chi tiết:
        "hôm nay/ngày mai/ngày X có gì?", "tổng hợp ngày", "cả ngày mấy độ?",
        "ngày X sáng/trưa/chiều/tối mấy độ", "nhiệt độ max/min ngày X".

    KHÔNG DÙNG KHI:
        - "bây giờ / tức thời" → get_current_weather (snapshot tại NOW).
        - "nhiều ngày / tuần / cuối tuần" → get_daily_forecast / get_weather_period
          (tool này CHỈ trả 1 ngày duy nhất).
        - "X giờ tối nay" (khung giờ cụ thể <1h granularity) → get_hourly_forecast.

    Returns: Flat VN dict: `"ngày"` (DD/MM `(Thứ X)`), `"thời tiết chung"`, `"nhiệt độ"`
    (min/max/TB), `"nhiệt độ theo ngày"` (Sáng/Trưa/Chiều/Tối — CHỈ ward), `"độ ẩm"`,
    `"xác suất mưa"`, `"tổng lượng mưa"` mm, `"gió"` (có thể kèm max_gust daily), `"UV"`,
    `"thời gian nắng"`, `"mọc-lặn"`.
    `"gợi ý dùng output"`: cảnh báo đây là TỔNG HỢP CẢ NGÀY, không phải tức thời.
    """
    from app.dal.timezone_utils import now_ict
    from datetime import date as date_type

    if date is None:
        query_date = now_ict().date()
    else:
        try:
            from datetime import datetime
            query_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            return {"error": "invalid_date", "message": f"Ngày không hợp lệ: {date}. Dùng format YYYY-MM-DD"}

    # Ward level: rich daily summary with temp_progression
    from app.dal.weather_dal import get_daily_summary_data
    from app.dal.weather_aggregate_dal import (
        get_district_daily_summary_data,
        get_city_daily_summary_data,
    )

    # District level: daily aggregate data
    def _district_summary(district_name):
        row = get_district_daily_summary_data(district_name, query_date)
        if not row:
            return {"error": "no_data", "message": f"Không có dữ liệu ngày {query_date} cho {district_name}"}
        from app.agent.dispatch import normalize_agg_keys
        row = normalize_agg_keys(row)
        row["level"] = "district"
        return row

    # City level: daily aggregate data
    def _city_summary():
        row = get_city_daily_summary_data(query_date)
        if not row:
            return {"error": "no_data", "message": f"Không có dữ liệu ngày {query_date} cho Hà Nội"}
        from app.agent.dispatch import normalize_agg_keys
        row = normalize_agg_keys(row)
        row["level"] = "city"
        return row

    from app.agent.dispatch import resolve_and_dispatch
    from app.agent.tools.output_builder import build_daily_summary_output
    raw = resolve_and_dispatch(
        ward_id=ward_id,
        location_hint=location_hint,
        default_scope="city",
        ward_fn=lambda ward_id: get_daily_summary_data(ward_id, query_date),
        district_fn=_district_summary,
        city_fn=_city_summary,
        label="tổng hợp ngày",
    )
    return build_daily_summary_output(raw)


# ============== Tool: get_weather_period ==============

class GetWeatherPeriodInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")
    start_date: str = Field(description="Ngày bắt đầu (YYYY-MM-DD)")
    end_date: str = Field(description="Ngày kết thúc (YYYY-MM-DD)")


@tool(args_schema=GetWeatherPeriodInput)
def get_weather_period(ward_id: str = None, location_hint: str = None,
                       start_date: str = None, end_date: str = None) -> dict:
    """THỜI TIẾT KHOẢNG NHIỀU NGÀY (past 14 ngày + forecast 8 ngày, tối đa 14 ngày range).

    DÙNG KHI: user hỏi RANGE ngày:
        "tuần này", "cuối tuần", "3 ngày tới", "từ ngày A đến ngày B",
        "tuần trước", "cả tháng" (giới hạn 14 ngày mỗi call).
        PREFER tool này cho range rộng thay nhiều call daily_summary/forecast.

    KHÔNG DÙNG KHI:
        - 1 ngày duy nhất chi tiết 4 buổi → get_daily_summary.
        - Khung giờ trong 48h → get_hourly_forecast.
        - Range > 14 ngày hoặc vượt 8-ngày forecast — refuse với limit.

    Returns: Flat VN dict: `"phạm vi"` (range), `"ngày"` = list per-day flat VN dict,
    `"tổng hợp"` (ngày nóng/mát/mưa nhiều/ít nhất — pre-computed, COPY thẳng),
    `"thống kê tổng"` (nhiệt độ TB/thấp/cao, tổng mưa, số ngày có mưa).
    """
    from datetime import datetime

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return {"error": "invalid_date", "message": "Ngày không hợp lệ. Dùng format YYYY-MM-DD"}

    if (end_dt - start_dt).days > 14:
        return {"error": "range_too_large", "message": "Khoảng thời gian tối đa 14 ngày"}

    # Ward: use weather_period_data
    from app.dal.weather_dal import get_weather_period_data
    from app.dal.weather_aggregate_dal import (
        get_district_weather_period_data,
        get_city_weather_period_data,
    )

    def _ward_period(ward_id):
        rows = get_weather_period_data(ward_id, start_date, end_date)
        if not rows:
            return {"error": "no_data", "message": f"Không có dữ liệu từ {start_date} đến {end_date}"}
        return _summarize_period(rows, "ward")

    # District: use DAL
    def _district_period(district_name):
        rows = get_district_weather_period_data(district_name, start_date, end_date)
        if not rows:
            return {"error": "no_data", "message": f"Không có dữ liệu từ {start_date} đến {end_date}"}
        from app.agent.dispatch import normalize_rows
        rows = normalize_rows(rows)
        return _summarize_period(rows, "district")

    # City: use DAL
    def _city_period():
        rows = get_city_weather_period_data(start_date, end_date)
        if not rows:
            return {"error": "no_data", "message": f"Không có dữ liệu từ {start_date} đến {end_date}"}
        from app.agent.dispatch import normalize_rows
        rows = normalize_rows(rows)
        return _summarize_period(rows, "city")

    from app.agent.dispatch import resolve_and_dispatch
    from app.agent.tools.output_builder import build_weather_period_output
    raw = resolve_and_dispatch(
        ward_id=ward_id,
        location_hint=location_hint,
        default_scope="city",
        ward_fn=_ward_period,
        district_fn=_district_period,
        city_fn=_city_period,
        label="thời tiết nhiều ngày",
    )
    return build_weather_period_output(raw)


def _summarize_period(rows: list, level: str) -> dict:
    """Tổng hợp thống kê từ nhiều ngày."""
    temps = [r.get("temp_avg") or r.get("temp") for r in rows if r.get("temp_avg") or r.get("temp")]
    temp_mins = [r.get("temp_min") for r in rows if r.get("temp_min") is not None]
    temp_maxs = [r.get("temp_max") for r in rows if r.get("temp_max") is not None]
    rains = [r.get("rain_total") or r.get("total_rain") or 0 for r in rows]

    summary = {
        "days": len(rows),
        "daily_data": rows,
        "statistics": {
            "avg_temp": round(sum(temps) / len(temps), 1) if temps else None,
            "min_temp": min(temp_mins) if temp_mins else None,
            "max_temp": max(temp_maxs) if temp_maxs else None,
            "total_rain": round(sum(rains), 1),
            "rain_days": sum(1 for r in rains if r > 0.5),
        },
        "level": level,
    }
    return summary
