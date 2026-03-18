"""LangGraph Agent Tools for Weather Chatbot."""

from typing import Optional
from pydantic import BaseModel, Field
from app.dal.timezone_utils import now_ict
from langchain_core.tools import tool


# ============== Tool 1: resolve_location ==============

class ResolveLocationInput(BaseModel):
    location_hint: str = Field(
        description="Tên phường/xã hoặc quận/huyện tại Hà Nội. Ví dụ: Cầu Giấy, Đống Đa"
    )


@tool(args_schema=ResolveLocationInput)
def resolve_location(location_hint: str) -> dict:
    """Giải quyết địa điểm mơ hồ (tìm phường/xã từ tên)."""
    from app.dal.location_dal import resolve_location as dal_resolve
    return dal_resolve(location_hint)


# ============== Tool 2: get_current_weather ==============

class GetCurrentWeatherInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="ward_id (vi du: ID_00169)")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")


@tool(args_schema=GetCurrentWeatherInput)
def get_current_weather(ward_id: str = None, location_hint: str = None) -> dict:
    """Lay thoi tiet HIEN TAI (real-time) cho mot phuong/xa.

    DUNG KHI: user hoi "bay gio", "hien tai", "dang", "luc nay".
    KHONG DUNG KHI: hoi ve tuong lai (dung get_hourly_forecast),
    hoi ca ngay (dung get_daily_summary), hoi ve quan/TP (dung get_district_weather/get_city_weather).
    Luu y: du lieu hien tai KHONG co pop (xac suat mua). Neu user hoi "co mua khong?",
    check weather_main + goi them get_hourly_forecast 1-2h.
    """
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
    hours: int = Field(default=24, description="Số giờ dự báo (1-48)")


@tool(args_schema=GetHourlyForecastInput)
def get_hourly_forecast(ward_id: str = None, location_hint: str = None, hours: int = 24) -> dict:
    """Lay du bao thoi tiet THEO GIO (1-48 gio toi).

    DUNG KHI: user hoi ve chieu nay, toi nay, sang mai, vai gio toi,
    mua luc may gio, nhiet do toi nay, gio dem nay, khoang thoi gian cu the.
    KHONG DUNG KHI: hoi ca ngay mai/tuan nay (dung get_daily_summary hoac get_weather_period),
    hoi hien tai (dung get_current_weather), hoi mua den bao gio (dung get_rain_timeline).
    """
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
    days: int = Field(default=7, description="Số ngày dự báo (1-8)")


@tool(args_schema=GetDailyForecastInput)
def get_daily_forecast(ward_id: str = None, location_hint: str = None, days: int = 7) -> dict:
    """Lay du bao thoi tiet THEO NGAY (1-8 ngay toi) cho mot phuong/xa.

    DUNG KHI: user hoi "ngay mai", "cuoi tuan", "3 ngay toi" cho mot phuong cu the.
    KHONG DUNG KHI: hoi ve quan/TP (dung get_district_daily_forecast/get_city_daily_forecast),
    hoi theo gio (dung get_hourly_forecast).
    """
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
    """Lay thoi tiet cua mot NGAY trong QUA KHU.

    DUNG KHI: user hoi "hom qua", "tuan truoc", "ngay 15/3".
    Luu y: du lieu lich su THIEU visibility va UV - khong hua tra cac thong so nay.
    """
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
    """So sánh thời tiết giữa HAI địa điểm."""
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
    """So sánh thời tiết HÔM NAY với HÔM QUA."""
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
    activity: str = Field(description="Hoạt động: chạy bộ, đạp xe, dạo chơi, photo, picnic")
    ward_id: Optional[str] = Field(default=None)
    location_hint: Optional[str] = Field(default=None)


@tool(args_schema=GetActivityAdviceInput)
def get_activity_advice(activity: str, ward_id: str = None, location_hint: str = None) -> dict:
    """Khuyến cáo có NÊN thực hiện hoạt động ngoài trời không."""
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
    """Lấy CẢNH BÁO thời tiết nguy hiểm."""
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
    """Phát hiện các HIỆN TƯỢNG THỜI TIẾT ĐẶC BIỆT tại Hà Nội."""
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
    """So sánh thời tiết hiện tại với trung bình mùa."""
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
    """Tổng hợp thời tiết 1 NGÀY: temp_range, feels_like_gap, daylight, hiện tượng."""
    from app.agent.utils import auto_resolve_location
    from app.dal.weather_dal import get_daily_summary_data
    from app.dal.weather_knowledge_dal import compare_with_seasonal, detect_hanoi_weather_phenomena
    from datetime import datetime
    from app.dal.timezone_utils import now_ict

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved["status"] != "ok":
        return {"error": resolved["status"]}

    query_date = now_ict().date() if date == "today" else datetime.strptime(date, "%Y-%m-%d").date()

    # Get daily data from DAL
    summary = get_daily_summary_data(resolved["ward_id"], query_date)
    if "error" in summary:
        return summary

    # Add seasonal comparison
    seasonal = compare_with_seasonal({"temp": summary["temp_range"].get("min"), "humidity": summary.get("humidity")})
    summary["seasonal_comparison"] = seasonal.get("comparisons", [])

    # Add phenomena detection
    phenomena_data = {
        "temp": summary["temp_progression"].get("trua"),
        "humidity": summary.get("humidity"),
        "dew_point": summary.get("dew_point"),
        "wind_deg": summary["wind"].get("direction"),
        "wind_speed": summary["wind"].get("speed"),
        "clouds": summary.get("clouds"),
        "weather_main": summary.get("weather_main"),
        "visibility": 10000,
    }
    phenomena = detect_hanoi_weather_phenomena(phenomena_data)
    summary["phenomena"] = phenomena.get("phenomena", [])

    summary["resolved_location"] = resolved["data"]
    return summary


# ============== Tool 13: get_weather_period ==============

class GetWeatherPeriodInput(BaseModel):
    ward_id: Optional[str] = Field(default=None)
    location_hint: Optional[str] = Field(default=None)
    start_date: str = Field(description="YYYY-MM-DD")
    end_date: str = Field(description="YYYY-MM-DD")


@tool(args_schema=GetWeatherPeriodInput)
def get_weather_period(ward_id: str = None, location_hint: str = None, start_date: str = None, end_date: str = None) -> dict:
    """Tổng hợp thời tiết nhiều NGÀY: trend, best/worst day, extremes."""
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
    temps = [r["temp_avg"] for r in rows if r.get("temp_avg") is not None]
    temp_min = min(temps) if temps else None
    temp_max = max(temps) if temps else None
    temp_avg = sum(temps) / len(temps) if temps else None
    
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
        # Don't reward missing data - only give points for actual low rain
        rain = r.get("rain_total")
        if rain is not None and rain < 5:
            score += 50
        elif rain is not None:
            score -= 20
        
        # UV scoring
        uvi = r.get("uvi")
        if uvi is not None and uvi < 6:
            score += 30
        elif uvi is not None:
            score -= 10
        
        # Temperature scoring
        temp = r.get("temp_avg")
        if temp is not None and 20 <= temp <= 30:
            score += 20
        elif temp is not None:
            score -= 10
        
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
    seasonal_diff = temp_avg - seasonal.get("temp_avg", temp_avg) if temp_avg else 0
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


# ============== Tool 14: get_district_weather ==============

class GetDistrictWeatherInput(BaseModel):
    district_name: str = Field(
        description="Ten quan/huyen tai Ha Noi. Vi du: 'Quan Cau Giay', 'Huyen Ba Vi', 'Dong Da'"
    )
    hours: int = Field(default=24, description="So gio du bao (1-48)")


@tool(args_schema=GetDistrictWeatherInput)
def get_district_weather(district_name: str, hours: int = 24) -> dict:
    """Lay thoi tiet hien tai va du bao theo gio cho mot quan/huyen.

    DUNG KHI: user hoi ve thoi tiet mot quan/huyen cu the.
    Du lieu tong hop tu tat ca phuong/xa, bao gom: nhiet do, do am, gio, ap suat,
    diem suong, UV, may, tam nhin, xac suat mua, huong gio.
    Co enrichment: heat_index, wind_chill, seasonal_comparison, phenomena, temp_spread.
    """
    from app.dal.weather_aggregate_dal import (
        get_district_current_weather,
        get_district_hourly_forecast
    )
    from app.agent.utils import auto_resolve_location

    # Resolve location to get correct district name (e.g., "Cầu Giấy" -> "Quận Cầu Giấy")
    resolved = auto_resolve_location(location_hint=district_name)
    if resolved.get("level") == "district":
        district_name = resolved["district_name"]
    elif resolved.get("status") == "not_found":
        return {"error": "not_found", "message": resolved.get("message", f"Không tìm thấy quận/huyện: {district_name}")}

    current = get_district_current_weather(district_name)
    if "error" in current:
        return current

    from app.agent.utils import enrich_district_response
    current = enrich_district_response(current)

    forecasts = get_district_hourly_forecast(district_name, hours)

    return {
        "current": current,
        "forecasts": forecasts[:hours],
        "count": len(forecasts[:hours]),
        "source": "aggregated"
    }


# ============== Tool 15: get_city_weather ==============

class GetCityWeatherInput(BaseModel):
    hours: int = Field(default=24, description="Số giờ dự báo (1-48)")


@tool(args_schema=GetCityWeatherInput)
def get_city_weather(hours: int = 24) -> dict:
    """Lay thoi tiet hien tai va du bao cho toan TP Ha Noi.

    DUNG KHI: user hoi "thoi tiet Ha Noi", "Ha Noi hom nay the nao".
    Du lieu tong hop tu 126 phuong/xa, co enrichment day du.
    Nen ket hop voi get_district_ranking de cho biet quan nao nong/lanh nhat.
    """
    from app.dal.weather_aggregate_dal import (
        get_city_current_weather,
        get_city_hourly_forecast
    )

    current = get_city_current_weather()
    if "error" in current:
        return current

    from app.agent.utils import enrich_city_response
    current = enrich_city_response(current)

    forecasts = get_city_hourly_forecast(hours)

    return {
        "current": current,
        "forecasts": forecasts[:hours],
        "count": len(forecasts[:hours]),
        "source": "aggregated"
    }


# ============== Tool 16: get_district_daily_forecast ==============

class GetDistrictDailyForecastInput(BaseModel):
    district_name: str = Field(
        description="Tên quận/huyện tại Hà Nội. Ví dụ: 'Quận Cầu Giấy', 'Huyện Ba Vì'"
    )
    days: int = Field(default=7, description="Số ngày dự báo (1-8)")


@tool(args_schema=GetDistrictDailyForecastInput)
def get_district_daily_forecast(district_name: str, days: int = 7) -> dict:
    """Lấy dự báo thời tiết theo NGÀY cho một quận/huyện."""
    from app.dal.weather_aggregate_dal import (
        get_district_current_weather,
        get_district_daily_forecast
    )
    from app.agent.utils import auto_resolve_location

    # Resolve location to get correct district name
    resolved = auto_resolve_location(location_hint=district_name)
    if resolved.get("level") == "district":
        district_name = resolved["district_name"]
    elif resolved.get("status") == "not_found":
        return {"error": "not_found", "message": resolved.get("message", f"Không tìm thấy quận/huyện: {district_name}")}

    current = get_district_current_weather(district_name)
    if "error" in current:
        return current

    from app.agent.utils import enrich_district_response
    current = enrich_district_response(current)

    forecasts = get_district_daily_forecast(district_name, days)

    return {
        "current": current,
        "forecasts": forecasts[:days],
        "count": len(forecasts[:days]),
        "source": "aggregated"
    }


# ============== Tool 17: get_city_daily_forecast ==============

class GetCityDailyForecastInput(BaseModel):
    days: int = Field(default=7, description="Số ngày dự báo (1-8)")


@tool(args_schema=GetCityDailyForecastInput)
def get_city_daily_forecast(days: int = 7) -> dict:
    """Lấy dự báo thời tiết theo NGÀY cho toàn TP Hà Nội."""
    from app.dal.weather_aggregate_dal import (
        get_city_current_weather,
        get_city_daily_forecast
    )

    current = get_city_current_weather()
    if "error" in current:
        return current

    from app.agent.utils import enrich_city_response
    current = enrich_city_response(current)

    forecasts = get_city_daily_forecast(days)

    return {
        "current": current,
        "forecasts": forecasts[:days],
        "count": len(forecasts[:days]),
        "source": "aggregated"
    }


# ============== Tool 18: get_district_ranking ==============

class GetDistrictRankingInput(BaseModel):
    metric: str = Field(
        default="nhiet_do",
        description="Metric: nhiet_do, do_am, gio, mua, uvi, ap_suat, diem_suong, may"
    )
    order: str = Field(default="cao_nhat", description="cao_nhat hoac thap_nhat")
    limit: int = Field(default=5, description="So luong ket qua (1-30)")


@tool(args_schema=GetDistrictRankingInput)
def get_district_ranking(metric: str = "nhiet_do", order: str = "cao_nhat", limit: int = 5) -> dict:
    """Xep hang cac quan/huyen theo chi so thoi tiet.

    DUNG KHI: user hoi "quan nao nong nhat?", "top 5 quan am nhat?",
    "noi nao gio manh nhat Ha Noi?", "quan nao mua nhieu nhat?".
    Metric: nhiet_do, do_am, gio, mua, uvi, ap_suat, diem_suong, may.
    """
    from app.dal.weather_aggregate_dal import get_district_rankings
    return get_district_rankings(metric, order, limit)


# ============== Tool 19: get_ward_ranking_in_district ==============

class GetWardRankingInput(BaseModel):
    district_name: str = Field(description="Ten quan/huyen. Vi du: 'Quan Cau Giay'")
    metric: str = Field(default="nhiet_do", description="Metric: nhiet_do, do_am, gio, uvi")
    order: str = Field(default="cao_nhat", description="cao_nhat hoac thap_nhat")
    limit: int = Field(default=10, description="So luong ket qua")


@tool(args_schema=GetWardRankingInput)
def get_ward_ranking_in_district(
    district_name: str, metric: str = "nhiet_do", order: str = "cao_nhat", limit: int = 10
) -> dict:
    """Xep hang cac phuong/xa trong mot quan/huyen theo chi so thoi tiet.

    DUNG KHI: user hoi "phuong nao o Cau Giay nong nhat?",
    "top phuong mua nhieu nhat quan Dong Da?".
    """
    from app.dal.weather_aggregate_dal import get_ward_rankings_in_district
    from app.agent.utils import auto_resolve_location

    resolved = auto_resolve_location(location_hint=district_name)
    if resolved.get("level") == "district":
        district_name = resolved["district_name"]

    return get_ward_rankings_in_district(district_name, metric, order, limit)


# ============== Tool 20: get_rain_timeline ==============

class GetRainTimelineInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Ten dia diem")
    hours: int = Field(default=24, description="So gio scan (1-48)")


@tool(args_schema=GetRainTimelineInput)
def get_rain_timeline(
    ward_id: Optional[str] = None, location_hint: Optional[str] = None, hours: int = 24
) -> dict:
    """Phan tich timeline mua/tanh tu du bao theo gio.

    DUNG KHI: user hoi "mua den bao gio?", "may gio tanh?",
    "khi nao mua?", "ngay mai mua vao khoang may gio?".
    Tra ve: cac khoang thoi gian mua, thoi diem mua tiep theo, thoi diem tanh.
    """
    from app.agent.utils import auto_resolve_location
    from app.dal.weather_dal import get_rain_timeline as dal_rain_timeline

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved.get("status") != "ok":
        return {"error": "location_not_found", "message": resolved.get("message", "Khong tim thay dia diem")}

    if resolved["level"] == "ward":
        return dal_rain_timeline(resolved["ward_id"], hours)
    else:
        return {"error": "need_ward", "message": "Can chi dinh phuong/xa cu the de xem timeline mua"}


# ============== Tool 21: get_best_time ==============

class GetBestTimeInput(BaseModel):
    activity: str = Field(
        description="Hoat dong: chay_bo, dua_dieu, picnic, bike, chup_anh, tap_the_duc, phoi_do"
    )
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Ten dia diem")
    hours: int = Field(default=24, description="So gio scan (1-48)")


@tool(args_schema=GetBestTimeInput)
def get_best_time(
    activity: str, ward_id: Optional[str] = None,
    location_hint: Optional[str] = None, hours: int = 24
) -> dict:
    """Tim thoi diem tot nhat trong ngay cho mot hoat dong.

    DUNG KHI: user hoi "may gio chay bo tot nhat?", "luc nao chup anh dep nhat?",
    "khi nao phoi do tot?", "gio nao nen di picnic?".
    Tra ve: top 5 gio tot nhat va 3 gio xau nhat voi diem so.
    """
    from app.agent.utils import auto_resolve_location
    from app.dal.activity_dal import get_best_time_for_activity

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved.get("status") != "ok":
        return {"error": "location_not_found", "message": resolved.get("message", "Khong tim thay dia diem")}

    wid = resolved.get("ward_id")
    if not wid and resolved["level"] == "district":
        # Use first ward in district
        from app.dal.location_dal import get_wards_in_district
        wards = get_wards_in_district(resolved["district_name"])
        wid = wards[0]["ward_id"] if wards else None

    if not wid:
        return {"error": "need_ward", "message": "Khong xac dinh duoc phuong/xa"}

    return get_best_time_for_activity(activity, wid, hours)


# ============== Tool 22: get_clothing_advice ==============

class GetClothingAdviceInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Ten dia diem")
    hours_ahead: int = Field(default=0, description="So gio phia truoc (0=hien tai)")


@tool(args_schema=GetClothingAdviceInput)
def get_clothing_advice(
    ward_id: Optional[str] = None, location_hint: Optional[str] = None, hours_ahead: int = 0
) -> dict:
    """Tu van trang phuc dua tren thoi tiet.

    DUNG KHI: user hoi "hom nay mac gi?", "can ao khoac khong?",
    "can mang o khong?", "mac gi di lam?".
    Tra ve: danh sach quan ao, ghi chu, thong tin thoi tiet.
    """
    from app.agent.utils import auto_resolve_location
    from app.dal.activity_dal import get_clothing_advice as dal_clothing

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved.get("status") != "ok":
        return {"error": "location_not_found", "message": resolved.get("message", "Khong tim thay dia diem")}

    wid = resolved.get("ward_id")
    if not wid and resolved["level"] == "district":
        from app.dal.location_dal import get_wards_in_district
        wards = get_wards_in_district(resolved["district_name"])
        wid = wards[0]["ward_id"] if wards else None

    if not wid:
        return {"error": "need_ward", "message": "Khong xac dinh duoc phuong/xa"}

    return dal_clothing(wid, hours_ahead)


# ============== Tool 23: get_temperature_trend ==============

class GetTemperatureTrendInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Ten dia diem")
    days: int = Field(default=7, description="So ngay phan tich (2-8)")


@tool(args_schema=GetTemperatureTrendInput)
def get_temperature_trend(
    ward_id: Optional[str] = None, location_hint: Optional[str] = None, days: int = 7
) -> dict:
    """Phan tich xu huong nhiet do trong vai ngay toi.

    DUNG KHI: user hoi "khi nao am len?", "may ngay toi co lanh hon khong?",
    "xu huong nhiet do tuan nay?", "bao gio het ret?".
    Tra ve: xu huong (warming/cooling/stable), ngay nong/lanh nhat, diem ngoat.
    """
    from app.agent.utils import auto_resolve_location
    from app.dal.weather_dal import get_temperature_trend as dal_temp_trend

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved.get("status") != "ok":
        return {"error": "location_not_found", "message": resolved.get("message", "Khong tim thay dia diem")}

    wid = resolved.get("ward_id")
    if not wid and resolved["level"] == "district":
        from app.dal.location_dal import get_wards_in_district
        wards = get_wards_in_district(resolved["district_name"])
        wid = wards[0]["ward_id"] if wards else None

    if not wid:
        return {"error": "need_ward", "message": "Khong xac dinh duoc phuong/xa"}

    return dal_temp_trend(wid, days)


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
    get_district_weather,
    get_city_weather,
    get_district_daily_forecast,
    get_city_daily_forecast,
    get_district_ranking,
    get_ward_ranking_in_district,
    get_rain_timeline,
    get_best_time,
    get_clothing_advice,
    get_temperature_trend,
]
