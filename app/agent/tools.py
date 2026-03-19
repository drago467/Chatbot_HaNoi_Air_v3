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
    ward_id: Optional[str] = Field(default=None, description="ward_id (ví dụ: ID_00169)")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")


@tool(args_schema=GetCurrentWeatherInput)
def get_current_weather(ward_id: str = None, location_hint: str = None) -> dict:
    """Lấy thời tiết HIỆN TẠI (real-time) cho một phường/xã.

    DÙNG KHI: user hỏi "bây giờ", "hiện tại", "đang", "lúc này".
    KHÔNG DÙNG KHI: hỏi về tương lai (dùng get_hourly_forecast),
    hỏi cả ngày (dùng get_daily_summary), hỏi về quận/TP (dùng get_district_weather/get_city_weather).
    Lưu ý: dữ liệu hiện tại KHÔNG có pop (xác suất mưa). Nếu user hỏi "có mưa không?",
    check weather_main + gọi thêm get_hourly_forecast 1-2h.
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
    """Lấy dự báo thời tiết THEO GIỜ (1-48 giờ tới).

    DÙNG KHI: user hỏi về chiều nay, tối nay, sáng mai, vài giờ tới,
    mưa lúc mấy giờ, nhiệt độ tối nay, gió đêm nay, khoảng thời gian cụ thể.
    KHÔNG DÙNG KHI: hỏi cả ngày mai/tuần này (dùng get_daily_summary hoặc get_weather_period),
    hỏi hiện tại (dùng get_current_weather), hỏi mưa đến bao giờ (dùng get_rain_timeline).
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
    """Lấy dự báo thời tiết THEO NGÀY (1-8 ngày tới) cho một phường/xã.

    DÙNG KHI: user hỏi "ngày mai", "cuối tuần", "3 ngày tới" cho một phường cụ thể.
    KHÔNG DÙNG KHI: hỏi về quận/TP (dùng get_district_daily_forecast/get_city_daily_forecast),
    hỏi theo giờ (dùng get_hourly_forecast).
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
    """Lấy thời tiết của một NGÀY trong QUÁ KHỨ.

    DÙNG KHI: user hỏi "hôm qua", "tuần trước", "ngày 15/3".
    Lưu ý: dữ liệu lịch sử THIẾU visibility và UV - không hứa trả các thông số này.
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
    """So sánh thời tiết HIỆN TẠI giữa HAI địa điểm.

    DÙNG KHI: "A và B nơi nào nóng/lạnh/ẩm hơn?", "so sánh thời tiết A với B",
    "Cầu Giấy hay Hoàn Kiếm mát hơn?".
    KHÔNG DÙNG KHI: so sánh hôm nay vs hôm qua (dùng compare_with_yesterday),
    so sánh với trung bình mùa (dùng get_seasonal_comparison).
    Trả về: thời tiết hiện tại của cả 2 nơi để so sánh.
    """
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
    """So sánh thời tiết HÔM NAY với HÔM QUA cho một địa điểm.

    DÙNG KHI: "hôm nay nóng hơn hôm qua không?", "so với hôm qua thế nào?".
    KHÔNG DÙNG KHI: so sánh 2 địa điểm (dùng compare_weather).
    """
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
    activity: str = Field(description="Hoạt động: chay_bo, dua_dieu, picnic, bike, chup_anh, tap_the_duc, phoi_do, du_lich, cam_trai, cau_ca, lam_vuon, boi_loi, leo_nui, di_dao, su_kien")
    ward_id: Optional[str] = Field(default=None)
    location_hint: Optional[str] = Field(default=None)


@tool(args_schema=GetActivityAdviceInput)
def get_activity_advice(activity: str, ward_id: str = None, location_hint: str = None) -> dict:
    """Khuyến cáo có NÊN thực hiện hoạt động ngoài trời không.

    DÙNG KHI: "đi chơi được không?", "chạy bộ có ổn không?", "có nên picnic không?",
    "thời tiết có phù hợp để [hoạt động] không?".
    KHÔNG DÙNG KHI: hỏi mấy giờ tốt nhất (dùng get_best_time),
    hỏi mặc gì (dùng get_clothing_advice).
    Trả về: mức khuyến cáo (nen/co_the/han_che/khong_nen), lý do, khuyến nghị.
    Hoạt động hỗ trợ: chay_bo, dua_dieu, picnic, bike, chup_anh, tap_the_duc,
    phoi_do, du_lich, cam_trai, cau_ca, lam_vuon, boi_loi, leo_nui, di_dao, su_kien.
    """
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
    ward_id: str = Field(default="all", description="ward_id (mặc định all)")


@tool(args_schema=GetWeatherAlertsInput)
def get_weather_alerts(ward_id: str = "all") -> dict:
    """Lấy CẢNH BÁO thời tiết nguy hiểm trong 24h tới.

    DÙNG KHI: "có cảnh báo gì không?", "thời tiết có nguy hiểm không?",
    "có giông bão không?", "có rét hại không?".
    Trả về: danh sách cảnh báo (gió giật > 20m/s, rét hại < 13°C, nắng nóng > 39°C, giông).
    """
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
    """Phát hiện các HIỆN TƯỢNG THỜI TIẾT ĐẶC BIỆT tại Hà Nội.

    DÙNG KHI: "có hiện tượng gì đặc biệt không?", "có nồm ẩm không?",
    "có gió mùa đông bắc không?", "có rét đậm không?".
    Trả về: danh sách hiện tượng (nồm ẩm, gió Lào, gió mùa ĐB, rét đậm, sương mù, mưa dông).
    """
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
    """So sánh thời tiết hiện tại với trung bình mùa (climatology Hà Nội).

    DÙNG KHI: "nóng hơn bình thường không?", "thời tiết có bất thường không?",
    "so với mùa này thế nào?".
    Trả về: nhiệt độ/độ ẩm hiện tại vs trung bình tháng, nhận xét chênh lệch.
    """
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
    from app.dal.weather_dal import get_weather_period_data
    from app.dal.weather_knowledge_dal import get_seasonal_average
    from datetime import datetime

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved["status"] != "ok":
        return {"error": resolved["status"]}

    rows = get_weather_period_data(resolved["ward_id"], start_date, end_date)

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
        description="Tên quận/huyện tại Hà Nội. Ví dụ: 'Quận Cầu Giấy', 'Huyện Ba Vì', 'Đống Đa'"
    )
    hours: int = Field(default=24, description="Số giờ dự báo (1-48)")


@tool(args_schema=GetDistrictWeatherInput)
def get_district_weather(district_name: str, hours: int = 24) -> dict:
    """Lấy thời tiết hiện tại và dự báo theo giờ cho một quận/huyện.

    DÙNG KHI: user hỏi về thời tiết một quận/huyện cụ thể.
    Dữ liệu tổng hợp từ tất cả phường/xã, bao gồm: nhiệt độ, độ ẩm, gió, áp suất,
    điểm sương, UV, mây, tầm nhìn, xác suất mưa, hướng gió.
    Có enrichment: heat_index, wind_chill, seasonal_comparison, phenomena, temp_spread.
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
    """Lấy thời tiết hiện tại và dự báo cho toàn TP Hà Nội.

    DÙNG KHI: user hỏi "thời tiết Hà Nội", "Hà Nội hôm nay thế nào".
    Dữ liệu tổng hợp từ 126 phường/xã, có enrichment đầy đủ.
    Nên kết hợp với get_district_ranking để cho biết quận nào nóng/lạnh nhất.
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
        description="Chỉ số: nhiet_do, do_am, gio, mua, uvi, ap_suat, diem_suong, may"
    )
    order: str = Field(default="cao_nhat", description="cao_nhat hoặc thap_nhat")
    limit: int = Field(default=5, description="Số lượng kết quả (1-30)")


@tool(args_schema=GetDistrictRankingInput)
def get_district_ranking(metric: str = "nhiet_do", order: str = "cao_nhat", limit: int = 5) -> dict:
    """Xếp hạng các quận/huyện theo chỉ số thời tiết.

    DÙNG KHI: user hỏi "quận nào nóng nhất?", "top 5 quận ẩm nhất?",
    "nơi nào gió mạnh nhất Hà Nội?", "quận nào mưa nhiều nhất?".
    Chỉ số: nhiet_do, do_am, gio, mua, uvi, ap_suat, diem_suong, may.
    """
    from app.dal.weather_aggregate_dal import get_district_rankings
    return get_district_rankings(metric, order, limit)


# ============== Tool 19: get_ward_ranking_in_district ==============

class GetWardRankingInput(BaseModel):
    district_name: str = Field(description="Tên quận/huyện. Ví dụ: 'Quận Cầu Giấy'")
    metric: str = Field(default="nhiet_do", description="Chỉ số: nhiet_do, do_am, gio, uvi")
    order: str = Field(default="cao_nhat", description="cao_nhat hoặc thap_nhat")
    limit: int = Field(default=10, description="Số lượng kết quả")


@tool(args_schema=GetWardRankingInput)
def get_ward_ranking_in_district(
    district_name: str, metric: str = "nhiet_do", order: str = "cao_nhat", limit: int = 10
) -> dict:
    """Xếp hạng các phường/xã trong một quận/huyện theo chỉ số thời tiết.

    DÙNG KHI: user hỏi "phường nào ở Cầu Giấy nóng nhất?",
    "top phường mưa nhiều nhất quận Đống Đa?".
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
    location_hint: Optional[str] = Field(default=None, description="Tên địa điểm")
    hours: int = Field(default=24, description="Số giờ scan (1-48)")


@tool(args_schema=GetRainTimelineInput)
def get_rain_timeline(
    ward_id: Optional[str] = None, location_hint: Optional[str] = None, hours: int = 24
) -> dict:
    """Phân tích timeline mưa/tạnh từ dự báo theo giờ.

    DÙNG KHI: user hỏi "mưa đến bao giờ?", "mấy giờ tạnh?",
    "khi nào mưa?", "ngày mai mưa vào khoảng mấy giờ?".
    Trả về: các khoảng thời gian mưa, thời điểm mưa tiếp theo, thời điểm tạnh.
    """
    from app.agent.utils import auto_resolve_location
    from app.dal.weather_dal import get_rain_timeline as dal_rain_timeline

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved.get("status") != "ok":
        return {"error": "location_not_found", "message": resolved.get("message", "Không tìm thấy địa điểm")}

    wid = resolved.get("ward_id")
    if not wid and resolved["level"] == "district":
        # Fallback: dùng phường đại diện trong quận
        from app.dal.location_dal import get_wards_in_district
        wards = get_wards_in_district(resolved["district_name"])
        wid = wards[0]["ward_id"] if wards else None
        if wid:
            result = dal_rain_timeline(wid, hours)
            result["note"] = f"Dữ liệu đại diện từ {wards[0].get('ward_name_vi', '')} trong {resolved['district_name']}"
            return result

    if resolved["level"] == "ward":
        return dal_rain_timeline(resolved["ward_id"], hours)
    else:
        return {"error": "need_ward", "message": "Cần chỉ định phường/xã cụ thể để xem timeline mưa"}


# ============== Tool 21: get_best_time ==============

class GetBestTimeInput(BaseModel):
    activity: str = Field(
        description="Hoạt động: chay_bo, dua_dieu, picnic, bike, chup_anh, tap_the_duc, phoi_do, du_lich, cam_trai, cau_ca, lam_vuon, boi_loi, leo_nui, di_dao, su_kien"
    )
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên địa điểm")
    hours: int = Field(default=24, description="Số giờ scan (1-48)")


@tool(args_schema=GetBestTimeInput)
def get_best_time(
    activity: str, ward_id: Optional[str] = None,
    location_hint: Optional[str] = None, hours: int = 24
) -> dict:
    """Tìm thời điểm tốt nhất trong ngày cho một hoạt động.

    DÙNG KHI: user hỏi "mấy giờ chạy bộ tốt nhất?", "lúc nào chụp ảnh đẹp nhất?",
    "khi nào phơi đồ tốt?", "giờ nào nên đi picnic?".
    Trả về: top 5 giờ tốt nhất và 3 giờ xấu nhất với điểm số.
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
    location_hint: Optional[str] = Field(default=None, description="Tên địa điểm")
    hours_ahead: int = Field(default=0, description="Số giờ phía trước (0=hiện tại)")


@tool(args_schema=GetClothingAdviceInput)
def get_clothing_advice(
    ward_id: Optional[str] = None, location_hint: Optional[str] = None, hours_ahead: int = 0
) -> dict:
    """Tư vấn trang phục dựa trên thời tiết.

    DÙNG KHI: user hỏi "hôm nay mặc gì?", "cần áo khoác không?",
    "cần mang ô không?", "mặc gì đi làm?".
    Trả về: danh sách quần áo, ghi chú, thông tin thời tiết.
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
    location_hint: Optional[str] = Field(default=None, description="Tên địa điểm")
    days: int = Field(default=7, description="Số ngày phân tích (2-8)")


@tool(args_schema=GetTemperatureTrendInput)
def get_temperature_trend(
    ward_id: Optional[str] = None, location_hint: Optional[str] = None, days: int = 7
) -> dict:
    """Phân tích xu hướng nhiệt độ trong vài ngày tới.

    DÙNG KHI: user hỏi "khi nào ấm lên?", "mấy ngày tới có lạnh hơn không?",
    "xu hướng nhiệt độ tuần này?", "bao giờ hết rét?".
    Trả về: xu hướng (warming/cooling/stable), ngày nóng/lạnh nhất, điểm ngoặt.
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


# ============== Tool 24: get_comfort_index ==============

class GetComfortIndexInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên địa điểm")
    hours_ahead: int = Field(default=0, description="Số giờ phía trước (0=hiện tại)")


@tool(args_schema=GetComfortIndexInput)
def get_comfort_index(
    ward_id: Optional[str] = None, location_hint: Optional[str] = None, hours_ahead: int = 0
) -> dict:
    """Tính chỉ số thoải mái tổng hợp (0-100) dựa trên nhiệt độ, độ ẩm, gió, UV, mưa.

    DÙNG KHI: user hỏi "ra ngoài có thoải mái không?", "thời tiết dễ chịu không?",
    "có nên ra ngoài không?", "thời tiết thế nào cho hoạt động ngoài trời?".
    KHÔNG DÙNG KHI: hỏi hoạt động cụ thể (dùng get_activity_advice),
    hỏi mặc gì (dùng get_clothing_advice).
    Trả về: điểm 0-100, nhãn (Rất thoải mái/Thoải mái/Chấp nhận được/Khó chịu/Rất khó chịu),
    phân tích từng yếu tố, khuyến nghị.
    """
    from app.agent.utils import auto_resolve_location
    from app.dal.weather_helpers import compute_comfort_index
    from app.dal.activity_dal import _get_weather_for_activity

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved.get("status") != "ok":
        return {"error": "location_not_found", "message": resolved.get("message", "Không tìm thấy địa điểm")}

    wid = resolved.get("ward_id")
    if not wid and resolved["level"] == "district":
        from app.dal.location_dal import get_wards_in_district
        wards = get_wards_in_district(resolved["district_name"])
        wid = wards[0]["ward_id"] if wards else None

    if not wid:
        return {"error": "need_ward", "message": "Không xác định được phường/xã"}

    weather = _get_weather_for_activity(wid, hours_ahead)
    if "error" in weather:
        return weather

    result = compute_comfort_index(
        temp=weather.get("temp"),
        humidity=weather.get("humidity"),
        wind_speed=weather.get("wind_speed"),
        uvi=weather.get("uvi") or 0,
        pop=weather.get("pop") or 0,
    )

    if result is None:
        return {"error": "no_data", "message": "Không đủ dữ liệu để tính chỉ số thoải mái"}

    result["weather_summary"] = {
        "temp": weather.get("temp"),
        "humidity": weather.get("humidity"),
        "wind_speed": weather.get("wind_speed"),
        "uvi": weather.get("uvi"),
        "pop": weather.get("pop"),
        "weather_main": weather.get("weather_main"),
    }
    result["resolved_location"] = resolved["data"]
    return result


# ============== Tool 25: get_weather_change_alert ==============

class GetWeatherChangeAlertInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên địa điểm")
    hours: int = Field(default=6, description="Số giờ scan phía trước (1-12)")


@tool(args_schema=GetWeatherChangeAlertInput)
def get_weather_change_alert(
    ward_id: Optional[str] = None, location_hint: Optional[str] = None, hours: int = 6
) -> dict:
    """Phát hiện thay đổi thời tiết đáng kể trong vài giờ tới.

    DÙNG KHI: user hỏi "thời tiết có thay đổi gì không?", "trời có chuyển mưa không?",
    "có gì bất thường không?", "thời tiết sắp tới thế nào?".
    KHÔNG DÙNG KHI: hỏi dự báo chi tiết (dùng get_hourly_forecast),
    hỏi cảnh báo nguy hiểm (dùng get_weather_alerts).
    Trả về: danh sách thay đổi đáng kể (nhiệt độ, mưa, gió) + thời điểm xảy ra.
    """
    from app.agent.utils import auto_resolve_location
    from app.dal.weather_dal import detect_weather_changes

    resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
    if resolved.get("status") != "ok":
        return {"error": "location_not_found", "message": resolved.get("message", "Không tìm thấy địa điểm")}

    wid = resolved.get("ward_id")
    if not wid and resolved["level"] == "district":
        from app.dal.location_dal import get_wards_in_district
        wards = get_wards_in_district(resolved["district_name"])
        wid = wards[0]["ward_id"] if wards else None

    if not wid:
        return {"error": "need_ward", "message": "Không xác định được phường/xã"}

    result = detect_weather_changes(wid, min(hours, 12))
    result["resolved_location"] = resolved["data"]
    return result


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
    get_comfort_index,
    get_weather_change_alert,
]
