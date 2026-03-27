"""History tools — weather_history, daily_summary, weather_period.

Tat ca deu ho tro 3 tier nhat quan.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============== Tool: get_weather_history ==============

class GetWeatherHistoryInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Ten phuong/xa hoac quan/huyen")
    date: str = Field(description="Ngay (YYYY-MM-DD)")


@tool(args_schema=GetWeatherHistoryInput)
def get_weather_history(ward_id: str = None, location_hint: str = None, date: str = None) -> dict:
    """Lay thoi tiet cua mot NGAY trong QUA KHU.

    DUNG KHI: user hoi "hom qua", "tuan truoc", "ngay 15/3".
    Ho tro: phuong/xa, quan/huyen, toan Ha Noi.
    Luu y: du lieu lich su chi co 14 ngay gan nhat.
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
        label="lich su thoi tiet",
    )


# ============== Tool: get_daily_summary ==============

class GetDailySummaryInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Ten phuong/xa hoac quan/huyen")
    date: Optional[str] = Field(default=None, description="Ngay (YYYY-MM-DD), mac dinh hom nay")


@tool(args_schema=GetDailySummaryInput)
def get_daily_summary(ward_id: str = None, location_hint: str = None, date: str = None) -> dict:
    """Tong hop thoi tiet CA NGAY: nhiet do min/max/trua/toi, mua, UV, gio, mat troi.

    DUNG KHI: user hoi "hom nay thoi tiet the nao?", "tong hop ngay", "ngay mai co gi?".
    Ho tro: phuong/xa (chi tiet nhat voi temp_progression sang/trua/chieu/toi),
    quan/huyen va toan Ha Noi (daily aggregate).
    Tra ve: temp_range, temp_progression, rain_assessment, uv_level, daylight_hours, wind.
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
            return {"error": "invalid_date", "message": f"Ngay khong hop le: {date}. Dung format YYYY-MM-DD"}

    # Ward level: rich daily summary with temp_progression
    from app.dal.weather_dal import get_daily_summary_data

    # District level: daily aggregate data
    def _district_summary(district_name):
        from app.db.dal import query_one
        row = query_one("""
            SELECT district_name_vi, date, avg_temp, temp_min, temp_max,
                   avg_humidity, avg_pop, total_rain, weather_main,
                   avg_dew_point, avg_pressure, avg_clouds, max_uvi,
                   avg_wind_deg, max_wind_gust, avg_wind_speed, ward_count
            FROM fact_weather_district_daily
            WHERE district_name_vi = %s AND date = %s::date
        """, (district_name, str(query_date)))
        if not row:
            return {"error": "no_data", "message": f"Khong co du lieu ngay {query_date} cho {district_name}"}
        from app.agent.dispatch import normalize_agg_keys
        row = normalize_agg_keys(row)
        row["level"] = "district"
        return row

    # City level: daily aggregate data
    def _city_summary():
        from app.db.dal import query_one
        row = query_one("""
            SELECT date, avg_temp, temp_min, temp_max,
                   avg_humidity, avg_pop, total_rain, weather_main,
                   avg_dew_point, avg_pressure, avg_clouds, max_uvi,
                   avg_wind_deg, max_wind_gust, avg_wind_speed, ward_count
            FROM fact_weather_city_daily
            WHERE date = %s::date
        """, (str(query_date),))
        if not row:
            return {"error": "no_data", "message": f"Khong co du lieu ngay {query_date} cho Ha Noi"}
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
        label="tong hop ngay",
    )


# ============== Tool: get_weather_period ==============

class GetWeatherPeriodInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Ten phuong/xa hoac quan/huyen")
    start_date: str = Field(description="Ngay bat dau (YYYY-MM-DD)")
    end_date: str = Field(description="Ngay ket thuc (YYYY-MM-DD)")


@tool(args_schema=GetWeatherPeriodInput)
def get_weather_period(ward_id: str = None, location_hint: str = None,
                       start_date: str = None, end_date: str = None) -> dict:
    """Lay thoi tiet NHIEU NGAY trong khoang thoi gian.

    DUNG KHI: user hoi "tuan nay", "3 ngay toi", "tu ngay A den ngay B".
    Ho tro: phuong/xa (chi tiet), quan/huyen va toan Ha Noi (aggregate).
    Tra ve: daily data + thong ke tong hop (avg/min/max temp, total rain, ...).
    """
    from datetime import datetime

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return {"error": "invalid_date", "message": "Ngay khong hop le. Dung format YYYY-MM-DD"}

    if (end_dt - start_dt).days > 14:
        return {"error": "range_too_large", "message": "Khoang thoi gian toi da 14 ngay"}

    # Ward: use weather_period_data
    from app.dal.weather_dal import get_weather_period_data

    def _ward_period(ward_id):
        rows = get_weather_period_data(ward_id, start_date, end_date)
        if not rows:
            return {"error": "no_data", "message": f"Khong co du lieu tu {start_date} den {end_date}"}
        return _summarize_period(rows, "ward")

    # District: query district daily
    def _district_period(district_name):
        from app.db.dal import query
        rows = query("""
            SELECT date, avg_temp, temp_min, temp_max, avg_humidity, avg_pop, total_rain,
                   weather_main, avg_wind_speed, avg_wind_deg, max_uvi
            FROM fact_weather_district_daily
            WHERE district_name_vi = %s AND date BETWEEN %s AND %s
            ORDER BY date
        """, (district_name, start_date, end_date))
        if not rows:
            return {"error": "no_data", "message": f"Khong co du lieu tu {start_date} den {end_date}"}
        from app.agent.dispatch import normalize_rows
        rows = normalize_rows(rows)
        return _summarize_period(rows, "district")

    # City: query city daily
    def _city_period():
        from app.db.dal import query
        rows = query("""
            SELECT date, avg_temp, temp_min, temp_max, avg_humidity, avg_pop, total_rain,
                   weather_main, avg_wind_speed, avg_wind_deg, max_uvi
            FROM fact_weather_city_daily
            WHERE date BETWEEN %s AND %s
            ORDER BY date
        """, (start_date, end_date))
        if not rows:
            return {"error": "no_data", "message": f"Khong co du lieu tu {start_date} den {end_date}"}
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
        label="thoi tiet nhieu ngay",
    )


def _summarize_period(rows: list, level: str) -> dict:
    """Tong hop thong ke tu nhieu ngay."""
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
