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
    """Lấy thời tiết của một NGÀY trong QUÁ KHỨ.

    DÙNG KHI: user hỏi "hôm qua", "tuần trước", "ngày 15/3".
    Hỗ trợ: phường/xã, quận/huyện, toàn Hà Nội.
    Lưu ý: dữ liệu lịch sử chỉ có 14 ngày gần nhất.
    """
    from app.agent.dispatch import resolve_and_dispatch
    from app.dal.weather_dal import (
        get_weather_history as dal_ward,
        get_district_weather_history as dal_district,
        get_city_weather_history as dal_city,
    )

    return resolve_and_dispatch(
        ward_id=ward_id,
        location_hint=location_hint,
        default_scope="city",
        ward_fn=dal_ward,
        district_fn=lambda district_name: dal_district(district_name, date),
        city_fn=lambda: dal_city(date),
        ward_args={"date": date},
        label="lịch sử thời tiết",
    )


# ============== Tool: get_daily_summary ==============

class GetDailySummaryInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")
    date: Optional[str] = Field(default=None, description="Ngày (YYYY-MM-DD), mặc định hôm nay")


@tool(args_schema=GetDailySummaryInput)
def get_daily_summary(ward_id: str = None, location_hint: str = None, date: str = None) -> dict:
    """Tổng hợp thời tiết CẢ NGÀY: nhiệt độ min/max/trưa/tối, mưa, UV, gió, mặt trời.

    DÙNG KHI: user hỏi "hôm nay thời tiết thế nào?", "tổng hợp ngày", "ngày mai có gì?".
    Hỗ trợ: phường/xã (chi tiết nhất với temp_progression sáng/trưa/chiều/tối),
    quận/huyện và toàn Hà Nội (daily aggregate).
    Trả về: temp_range, temp_progression, rain_assessment, uv_level, daylight_hours, wind.
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
    return resolve_and_dispatch(
        ward_id=ward_id,
        location_hint=location_hint,
        default_scope="city",
        ward_fn=lambda ward_id: get_daily_summary_data(ward_id, query_date),
        district_fn=_district_summary,
        city_fn=_city_summary,
        label="tổng hợp ngày",
    )


# ============== Tool: get_weather_period ==============

class GetWeatherPeriodInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")
    start_date: str = Field(description="Ngày bắt đầu (YYYY-MM-DD)")
    end_date: str = Field(description="Ngày kết thúc (YYYY-MM-DD)")


@tool(args_schema=GetWeatherPeriodInput)
def get_weather_period(ward_id: str = None, location_hint: str = None,
                       start_date: str = None, end_date: str = None) -> dict:
    """Lấy thời tiết NHIỀU NGÀY trong khoảng thời gian.

    DÙNG KHI: user hỏi "tuần này", "3 ngày tới", "từ ngày A đến ngày B".
    Hỗ trợ: phường/xã (chi tiết), quận/huyện và toàn Hà Nội (aggregate).
    Trả về: daily data + thống kê tổng hợp (avg/min/max temp, total rain, ...).
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
    return resolve_and_dispatch(
        ward_id=ward_id,
        location_hint=location_hint,
        default_scope="city",
        ward_fn=_ward_period,
        district_fn=_district_period,
        city_fn=_city_period,
        label="thời tiết nhiều ngày",
    )


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
