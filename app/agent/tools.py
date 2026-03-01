"""LangGraph Agent Tools for Weather Chatbot."""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============== Tool 1: resolve_location ==============

class ResolveLocationInput(BaseModel):
    location_hint: str = Field(
        description="Ten phuong/xa hoac quan/huyen tai Ha Noi. Vi du: Cau Giay, Dong Da"
    )


@tool(args_schema=ResolveLocationInput)
def resolve_location(location_hint: str) -> dict:
    """Giai quyet dia diem am thanh."""
    from app.dal.location_dal import resolve_location as dal_resolve
    return dal_resolve(location_hint)


# ============== Tool 2: get_current_weather ==============

class GetCurrentWeatherInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="ward_id (vi du: ID_00169)")
    location_hint: Optional[str] = Field(default=None, description="Ten phuong/xa hoac quan/huyen")


@tool(args_schema=GetCurrentWeatherInput)
def get_current_weather(ward_id: str = None, location_hint: str = None) -> dict:
    """Lay thoi tiet Hien tai (real-time) + enrich (heat_index, wind_chill, seasonal)."""
    from app.agent.utils import auto_resolve_location, enrich_weather_response
    from app.dal import get_current_weather as dal_get_current_weather

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved["status"] != "ok":
        return {"error": resolved["status"], "message": resolved.get("message", "")}

    weather = dal_get_current_weather(resolved["ward_id"])
    weather = enrich_weather_response(weather)
    weather["resolved_location"] = resolved["data"]
    return weather


# ============== Tool 3: get_hourly_forecast ==============

class GetHourlyForecastInput(BaseModel):
    ward_id: Optional[str] = Field(default=None)
    location_hint: Optional[str] = Field(default=None)
    hours: int = Field(default=24, description="So gio du bao (1-48)")


@tool(args_schema=GetHourlyForecastInput)
def get_hourly_forecast(ward_id: str = None, location_hint: str = None, hours: int = 24) -> dict:
    """Lay du bao thoi tiet THEO GIO."""
    from app.agent.utils import auto_resolve_location
    from app.dal import get_hourly_forecast as dal_get_hourly_forecast

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved["status"] != "ok":
        return {"error": resolved["status"]}

    forecast = dal_get_hourly_forecast(resolved["ward_id"], hours)
    forecast["resolved_location"] = resolved["data"]
    return forecast


# ============== Tool 4: get_daily_forecast ==============

class GetDailyForecastInput(BaseModel):
    ward_id: Optional[str] = Field(default=None)
    location_hint: Optional[str] = Field(default=None)
    days: int = Field(default=7, description="So ngay du bao (1-8)")


@tool(args_schema=GetDailyForecastInput)
def get_daily_forecast(ward_id: str = None, location_hint: str = None, days: int = 7) -> dict:
    """Lay du bao thoi tiet THEO NGAY."""
    from app.agent.utils import auto_resolve_location
    from app.dal import get_daily_forecast as dal_get_daily_forecast

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved["status"] != "ok":
        return {"error": resolved["status"]}

    forecast = dal_get_daily_forecast(resolved["ward_id"], days)
    forecast["resolved_location"] = resolved["data"]
    return forecast


# ============== Tool 5: get_weather_history ==============

class GetWeatherHistoryInput(BaseModel):
    ward_id: Optional[str] = Field(default=None)
    location_hint: Optional[str] = Field(default=None)
    date: str = Field(description="Ngay (YYYY-MM-DD)")


@tool(args_schema=GetWeatherHistoryInput)
def get_weather_history(ward_id: str = None, location_hint: str = None, date: str = None) -> dict:
    """Lay thoi tiet cua mot NGAY trong QUA KHU."""
    from app.agent.utils import auto_resolve_location
    from app.dal import get_weather_history as dal_get_weather_history

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved["status"] != "ok":
        return {"error": resolved["status"]}

    history = dal_get_weather_history(resolved["ward_id"], date)
    history["resolved_location"] = resolved["data"]
    return history


# ============== Tool 6: compare_weather ==============

class CompareWeatherInput(BaseModel):
    ward_id1: Optional[str] = Field(default=None)
    location_hint1: Optional[str] = Field(default=None)
    ward_id2: Optional[str] = Field(default=None)
    location_hint2: Optional[str] = Field(default=None)


@tool(args_schema=CompareWeatherInput)
def compare_weather(ward_id1: str = None, location_hint1: str = None, ward_id2: str = None, location_hint2: str = None) -> dict:
    """So sanh thoi tiet giua HAI dia diem."""
    from app.agent.utils import auto_resolve_location
    from app.dal import compare_weather as dal_compare_weather

    r1 = auto_resolve_location(ward_id=ward_id1, location_hint=location_hint1)
    r2 = auto_resolve_location(ward_id=ward_id2, location_hint=location_hint2)

    if r1["status"] != "ok" or r2["status"] != "ok":
        return {"error": "location"}

    result = dal_compare_weather(r1["ward_id"], r2["ward_id"])
    result["location1"] = r1["data"]
    result["location2"] = r2["data"]
    return result


# ============== Tool 7: compare_with_yesterday ==============

class CompareWithYesterdayInput(BaseModel):
    ward_id: Optional[str] = Field(default=None)
    location_hint: Optional[str] = Field(default=None)


@tool(args_schema=CompareWithYesterdayInput)
def compare_with_yesterday(ward_id: str = None, location_hint: str = None) -> dict:
    """So sanh thoi tiet HOM NAY voi HOM QUA."""
    from app.agent.utils import auto_resolve_location
    from app.dal import compare_with_yesterday as dal_compare_with_yesterday

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved["status"] != "ok":
        return {"error": resolved["status"]}

    result = dal_compare_with_yesterday(resolved["ward_id"])
    result["resolved_location"] = resolved["data"]
    return result


# ============== Tool 8: get_activity_advice ==============

class GetActivityAdviceInput(BaseModel):
    activity: str = Field(description="Hoat dong: chay bo, dap xe, dao choi, photo, picnic")
    ward_id: Optional[str] = Field(default=None)
    location_hint: Optional[str] = Field(default=None)


@tool(args_schema=GetActivityAdviceInput)
def get_activity_advice(activity: str, ward_id: str = None, location_hint: str = None) -> dict:
    """Khuyen cao co NEN thuc hien hoat dong ngoai troi khong."""
    from app.agent.utils import auto_resolve_location
    from app.dal import get_activity_advice as dal_get_activity_advice

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved["status"] != "ok":
        return {"error": resolved["status"]}

    result = dal_get_activity_advice(activity, resolved["ward_id"])
    result["resolved_location"] = resolved["data"]
    return result


# ============== Tool 9: get_weather_alerts ==============

class GetWeatherAlertsInput(BaseModel):
    ward_id: str = Field(default="all", description="ward_id (mac dinh all)")


@tool(args_schema=GetWeatherAlertsInput)
def get_weather_alerts(ward_id: str = "all") -> dict:
    """Lay CANH BAO thoi tiet nguy hiem."""
    from app.dal import get_weather_alerts as dal_get_weather_alerts
    return dal_get_weather_alerts(ward_id)


# ============== Tool 10: detect_phenomena ==============

class DetectPhenomenaInput(BaseModel):
    ward_id: Optional[str] = Field(default=None)
    location_hint: Optional[str] = Field(default=None)


@tool(args_schema=DetectPhenomenaInput)
def detect_phenomena(ward_id: str = None, location_hint: str = None) -> dict:
    """Phat hien cac HIEN TUONG THOI TIET DAC BIET tai Ha Noi."""
    from app.agent.utils import auto_resolve_location
    from app.dal import get_current_weather as dal_get_current_weather
    from app.dal.weather_knowledge_dal import detect_hanoi_weather_phenomena

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved["status"] != "ok":
        return {"error": resolved["status"]}

    weather = dal_get_current_weather(resolved["ward_id"])
    phenomena = detect_hanoi_weather_phenomena(weather)

    return {"phenomena": phenomena.get("phenomena", []), "resolved_location": resolved["data"]}


# ============== Tool 11: get_seasonal_comparison ==============

class GetSeasonalComparisonInput(BaseModel):
    ward_id: Optional[str] = Field(default=None)
    location_hint: Optional[str] = Field(default=None)


@tool(args_schema=GetSeasonalComparisonInput)
def get_seasonal_comparison(ward_id: str = None, location_hint: str = None) -> dict:
    """So sanh thoi tiet hien tai voi trung binh mua."""
    from app.agent.utils import auto_resolve_location
    from app.dal import get_current_weather as dal_get_current_weather
    from app.dal.weather_knowledge_dal import compare_with_seasonal

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved["status"] != "ok":
        return {"error": resolved["status"]}

    weather = dal_get_current_weather(resolved["ward_id"])

    if weather.get("error"):
        return {"error": weather.get("error"), "message": weather.get("message", "")}

    seasonal = compare_with_seasonal(weather)

    return {
        "current": weather,
        "seasonal_avg": seasonal["seasonal_avg"],
        "comparisons": seasonal["comparisons"],
        "month_name": seasonal["month_name"],
        "resolved_location": resolved["data"]
    }


# ============== Tool 12: get_daily_summary ==============

class GetDailySummaryInput(BaseModel):
    ward_id: Optional[str] = Field(default=None)
    location_hint: Optional[str] = Field(default=None)
    date: str = Field(default="today", description="YYYY-MM-DD hoac 'today'")


@tool(args_schema=GetDailySummaryInput)
def get_daily_summary(ward_id: str = None, location_hint: str = None, date: str = "today") -> dict:
    """Tong hop thoi tiet 1 NGAY."""
    from app.agent.utils import auto_resolve_location
    from app.db.dal import query_one
    from datetime import datetime
    from app.dal.weather_helpers import wind_deg_to_vietnamese
    from app.dal.weather_knowledge_dal import compare_with_seasonal

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved["status"] != "ok":
        return {"error": resolved["status"]}

    query_date = datetime.now().date() if date == "today" else datetime.strptime(date, "%Y-%m-%d").date()

    row = query_one(
        "SELECT * FROM fact_weather_daily WHERE ward_id = %s AND date = %s",
        (resolved["ward_id"], query_date)
    )

    if not row:
        return {"error": "no_data", "message": f"Khong co du lieu ngay {date}"}

    rain_total = row.get("rain_total") or 0
    rain_assessment = "Khong mua" if rain_total == 0 else f"Mua {rain_total:.1f}mm"

    uvi = row.get("uvi") or 0
    uv_level = "Cuc cao" if uvi >= 11 else "Rat cao" if uvi >= 8 else "Cao" if uvi >= 6 else "Trung binh" if uvi >= 3 else "Thap"

    seasonal = compare_with_seasonal({"temp": row.get("temp_avg"), "humidity": row.get("humidity")})

    return {
        "date": str(query_date),
        "resolved_location": resolved["data"],
        "temp_range": {"min": row.get("temp_min"), "max": row.get("temp_max")},
        "temp_progression": {"sang": row.get("temp_morn"), "trua": row.get("temp_day"), "chieu": row.get("temp_eve"), "toi": row.get("temp_night")},
        "humidity": row.get("humidity"),
        "rain_assessment": rain_assessment,
        "uv_level": uv_level,
        "weather_main": row.get("weather_main"),
        "seasonal_comparison": seasonal.get("comparisons", [])
    }


# ============== Tool 13: get_weather_period ==============

class GetWeatherPeriodInput(BaseModel):
    ward_id: Optional[str] = Field(default=None)
    location_hint: Optional[str] = Field(default=None)
    start_date: str = Field(description="YYYY-MM-DD")
    end_date: str = Field(description="YYYY-MM-DD")


@tool(args_schema=GetWeatherPeriodInput)
def get_weather_period(ward_id: str = None, location_hint: str = None, start_date: str = None, end_date: str = None) -> dict:
    """Tong hop thoi tiet nhieu NGAY."""
    from app.agent.utils import auto_resolve_location
    from app.db.dal import query

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved["status"] != "ok":
        return {"error": resolved["status"]}

    rows = query(
        "SELECT date, temp_min, temp_max, temp_avg, pop, rain_total, uvi FROM fact_weather_daily WHERE ward_id = %s AND date BETWEEN %s AND %s ORDER BY date",
        (resolved["ward_id"], start_date, end_date)
    )

    if not rows:
        return {"error": "no_data"}

    temps = [r["temp_avg"] for r in rows if r.get("temp_avg")]
    rainy_days = sum(1 for r in rows if (r.get("pop") or 0) > 0.5)
    total_rain = sum(r.get("rain_total") or 0 for r in rows)

    return {
        "period": f"{start_date} den {end_date}",
        "days_count": len(rows),
        "resolved_location": resolved["data"],
        "temp_range": f"{min(temps):.0f} - {max(temps):.0f}C" if temps else None,
        "rainy_days": rainy_days,
        "total_rain": total_rain,
        "days": [{"date": str(r["date"]), "temp_avg": r.get("temp_avg")} for r in rows]
    }


# Export all tools
TOOLS = [
    resolve_location,
    get_current_weather,
    get_hourly_forecast,
    get_daily_forecast,
    get_weather_history,
    compare_weather,
    compare_with_yesterday,
    get_activity_advice,
    get_weather_alerts,
    detect_phenomena,
    get_seasonal_comparison,
    get_daily_summary,
    get_weather_period,
]
