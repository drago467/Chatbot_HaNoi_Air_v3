"""LangGraph Agent Tools for Weather Chatbot."""

import functools
from typing import Optional
from pydantic import BaseModel, Field
from app.dal.timezone_utils import now_ict
from langchain_core.tools import tool


# ============== Tool 1: resolve_location ==============

class ResolveLocationInput(BaseModel):
    location_hint: str = Field(
        description="Ten phuong/xa hoac quan/huyen tai Ha Noi. Vi du: Cau Giay, Dong Da"
    )


@tool(args_schema=ResolveLocationInput)
def resolve_location(location_hint: str) -> dict:
    """Giai quyet dia diem mo ho (tim phuong/xa tu ten)."""
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
    
    # Guard: if weather has error, return early
    if "error" in weather:
        return weather
    
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

    data = dal_get_hourly_forecast(resolved["ward_id"], hours)
    
    # Guard: if data has error, return early
    if isinstance(data, dict) and "error" in data:
        return data
    
    return {"forecasts": data, "count": len(data), "resolved_location": resolved["data"]}


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

    data = dal_get_daily_forecast(resolved["ward_id"], days)
    
    # Guard: if data has error, return early
    if isinstance(data, dict) and "error" in data:
        return data
    
    return {"forecasts": data, "count": len(data), "resolved_location": resolved["data"]}


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
    
    # Guard: if history has error, return early
    if "error" in history:
        return history
    
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
    result["location1_info"] = r1["data"]
    result["location2_info"] = r2["data"]
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
    # Convert 'all' to None for DAL
    actual_id = None if ward_id == "all" else ward_id
    alerts = dal_get_weather_alerts(actual_id)
    return {"alerts": alerts, "count": len(alerts)}


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
    """Tong hop thoi tiet 1 NGAY: temp_range, feels_like_gap, daylight, phenomena."""
    from app.agent.utils import auto_resolve_location
    from app.db.dal import query_one
    from datetime import datetime
    from app.dal.weather_helpers import wind_deg_to_vietnamese
    from app.dal.weather_knowledge_dal import compare_with_seasonal, detect_hanoi_weather_phenomena

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved["status"] != "ok":
        return {"error": resolved["status"]}

    query_date = now_ict().date() if date == "today" else datetime.strptime(date, "%Y-%m-%d").date()

    row = query_one(
        "SELECT * FROM fact_weather_daily WHERE ward_id = %s AND date = %s",
        (resolved["ward_id"], query_date)
    )

    if not row:
        return {"error": "no_data", "message": f"Khong co du lieu ngay {date}"}

    # Temp range + bien do nhiet
    temp_min = row.get("temp_min")
    temp_max = row.get("temp_max")
    temp_range = temp_max - temp_min if temp_min is not None and temp_max is not None else 0
    bien_do_nhiet = f"Bien do nhiet {temp_range:.0f}C" if temp_range > 0 else ""
    if temp_range > 10:
        bien_do_nhiet += " - Sang lanh, trua nong, nen mac ao khoac"

    # Feels like gap
    feels_like_day = row.get("feels_like_day") or 0
    temp_day = row.get("temp_day") or 0
    feels_like_gap = feels_like_day - temp_day

    # Rain assessment
    rain_total = row.get("rain_total") or 0
    if rain_total == 0:
        rain_assessment = "Khong mua"
    elif rain_total < 10:
        rain_assessment = f"Mua nhe {rain_total:.1f}mm"
    elif rain_total < 25:
        rain_assessment = f"Mua vua {rain_total:.1f}mm"
    else:
        rain_assessment = f"Mua to {rain_total:.1f}mm - Nen mang o"

    # UV level
    uvi = row.get("uvi") or 0
    if uvi >= 11:
        uv_level = "Cuc cao - Nguy hiem"
    elif uvi >= 8:
        uv_level = "Rat cao - Han che ra ngoai 10h-14h"
    elif uvi >= 6:
        uv_level = "Cao - Can che nang"
    elif uvi >= 3:
        uv_level = "Trung binh"
    else:
        uv_level = "Thap"

    # Daylight hours
    daylight_hours = None
    if row.get("sunrise") and row.get("sunset"):
        try:
            sunrise = row["sunrise"]
            sunset = row["sunset"]
            if hasattr(sunrise, "replace"):
                sunrise = sunrise.replace(tzinfo=None)
                sunset = sunset.replace(tzinfo=None)
            daylight = sunset - sunrise
            daylight_hours = round((sunset - sunrise).total_seconds() / 3600, 1)
        except (TypeError, ValueError, AttributeError):
            pass

    # Wind direction
    wind_dir = wind_deg_to_vietnamese(row.get("wind_deg")) if row.get("wind_deg") else None

    # Seasonal comparison
    seasonal = compare_with_seasonal({"temp": row.get("temp_avg"), "humidity": row.get("humidity")})

    # Phenomena detection
    phenomena = detect_hanoi_weather_phenomena(row)

    return {
        "date": str(query_date),
        "resolved_location": resolved["data"],
        "temp_range": {"min": temp_min, "max": temp_max, "bien_do": temp_range},
        "temp_progression": {"sang": row.get("temp_morn"), "trua": row.get("temp_day"), "chieu": row.get("temp_eve"), "toi": row.get("temp_night")},
        "feels_like_gap": feels_like_gap,
        "humidity": row.get("humidity"),
        "dew_point": row.get("dew_point"),
        "pressure": row.get("pressure"),
        "rain_assessment": rain_assessment,
        "pop": row.get("pop"),
        "uvi": uvi,
        "uv_level": uv_level,
        "daylight_hours": daylight_hours,
        "wind": {"speed": row.get("wind_speed"), "direction": wind_dir},
        "weather_main": row.get("weather_main"),
        "weather_description": row.get("weather_description"),
        "seasonal_comparison": seasonal.get("comparisons", []),
        "phenomena": phenomena.get("phenomena", []),
        "note": bien_do_nhiet
    }


# ============== Tool 13: get_weather_period ==============

class GetWeatherPeriodInput(BaseModel):
    ward_id: Optional[str] = Field(default=None)
    location_hint: Optional[str] = Field(default=None)
    start_date: str = Field(description="YYYY-MM-DD")
    end_date: str = Field(description="YYYY-MM-DD")


@tool(args_schema=GetWeatherPeriodInput)
def get_weather_period(ward_id: str = None, location_hint: str = None, start_date: str = None, end_date: str = None) -> dict:
    """Tong hop thoi tiet nhieu NGAY: trend, best/worst day, extremes."""
    from app.agent.utils import auto_resolve_location
    from app.db.dal import query
    from app.dal.weather_knowledge_dal import get_seasonal_average
    from datetime import datetime

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved["status"] != "ok":
        return {"error": resolved["status"]}

    # Query with more columns
    rows = query(
        "SELECT date, temp_min, temp_max, temp_avg, humidity, pop, rain_total, uvi, wind_speed, weather_main FROM fact_weather_daily WHERE ward_id = %s AND date BETWEEN %s AND %s ORDER BY date",
        (resolved["ward_id"], start_date, end_date)
    )

    if not rows:
        return {"error": "no_data"}

    # Aggregation
    temps = [r["temp_avg"] for r in rows if r.get("temp_avg")]
    temp_min = min(temps) if temps and all(t is not None for t in temps) else None
    temp_max = max(temps) if temps and all(t is not None for t in temps) else None
    temp_avg = sum(temps) / len(temps) if temps and all(t is not None for t in temps) else None
    
    rainy_days = sum(1 for r in rows if (r.get("pop") or 0) > 0.5 or (r.get("rain_total") or 0) > 0)
    total_rain = sum(r.get("rain_total") or 0 for r in rows)
    avg_humidity = sum(r.get("humidity") or 0 for r in rows) / len(rows) if rows else 0
    max_uvi = max((r.get("uvi") or 0) for r in rows) if rows else 0

    # Trend detection
    trend = "stable"
    if len(temps) >= 3:
        first_half = sum(temps[:len(temps)//2]) / (len(temps)//2)
        second_half = sum(temps[len(temps)//2:]) / (len(temps) - len(temps)//2)
        diff = second_half - first_half
        if diff > 2:
            trend = "warming"
        elif diff < -2:
            trend = "cooling"

    # Best/worst day scoring
    def score_day(r):
        score = 0
        score += 50 if r.get("rain_total") is None or r["rain_total"] < 5 else -20
        score += 30 if (r.get("uvi") or 0) < 6 else -10
        score += 20 if 20 <= (r.get("temp_avg") or 30) <= 30 else -10
        return score

    scored = [(r, score_day(r)) for r in rows]
    best_day = max(scored, key=lambda x: x[1])[0] if scored else None
    worst_day = min(scored, key=lambda x: x[1])[0] if scored else None

    # Days list
    days = [
        {
            "date": str(r["date"]),
            "temp_avg": r.get("temp_avg"),
            "temp_range": f"{r.get('temp_min')} - {r.get('temp_max')}",
            "humidity": r.get("humidity"),
            "pop": r.get("pop"),
            "rain_total": r.get("rain_total"),
            "weather_main": r.get("weather_main")
        }
        for r in rows
    ]

    # Seasonal comparison
    month = now_ict().month
    seasonal = get_seasonal_average(month)
    seasonal_diff = temp_avg - seasonal["temp_avg"] if temp_avg else 0
    seasonal_comp = f"Nong hon {seasonal_diff:.1f}C" if seasonal_diff > 2 else f"Lanh hon {abs(seasonal_diff):.1f}C" if seasonal_diff < -2 else "Binh thuong"

    return {
        "period": f"{start_date} den {end_date}",
        "days_count": len(rows),
        "resolved_location": resolved["data"],
        "aggregation": {
            "temp_range": f"{temp_min:.0f} - {temp_max:.0f}C" if temp_min and temp_max else None,
            "temp_avg": round(temp_avg, 1) if temp_avg else None,
            "total_rain": round(total_rain, 1),
            "rainy_days": rainy_days,
            "avg_humidity": round(avg_humidity, 0),
            "max_uvi": max_uvi,
            "trend": trend
        },
        "best_day": {"date": str(best_day["date"]), "temp": best_day.get("temp_avg")} if best_day else None,
        "worst_day": {"date": str(worst_day["date"]), "temp": worst_day.get("temp_avg")} if worst_day else None,
        "days": days,
        "seasonal_comparison": seasonal_comp
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
