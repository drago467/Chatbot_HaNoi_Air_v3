"""Activity Advice DAL - Activity-specific weather recommendations."""

from typing import Dict, Any, List
from app.dal.weather_dal import get_current_weather, get_hourly_forecast
from app.dal.weather_knowledge_dal import detect_hanoi_weather_phenomena
from app.config.thresholds import KTTV_THRESHOLDS, THRESHOLDS


def _get_weather_for_activity(ward_id: str, hours_ahead: int = 3) -> Dict[str, Any]:
    """Get weather data for activity advice.

    Uses forecast data for future activities, falls back to current weather.
    Also supplements current weather with forecast pop (rain probability)
    since current data doesn't include pop from OpenWeather API.

    Args:
        ward_id: Ward ID
        hours_ahead: Hours ahead to check forecast (default 3)

    Returns:
        Weather data dictionary
    """
    current = get_current_weather(ward_id)

    if hours_ahead <= 0:
        # User asking about right now - use current but supplement with forecast pop
        if "error" not in current:
            forecasts = get_hourly_forecast(ward_id, hours=2)
            if forecasts:
                # Current data has pop=NULL, get from nearest forecast
                current["pop"] = forecasts[0].get("pop", 0)
                if current.get("rain_1h") is None:
                    current["rain_1h"] = forecasts[0].get("rain_1h", 0)
        return current

    # For future activities, use forecast data
    forecasts = get_hourly_forecast(ward_id, hours=hours_ahead)

    if not forecasts:
        # Fallback to current weather if no forecast available
        return current

    # Average the forecast hours for a representative picture
    temps = [f["temp"] for f in forecasts if f.get("temp") is not None]
    humidities = [f["humidity"] for f in forecasts if f.get("humidity") is not None]
    pops = [f["pop"] for f in forecasts if f.get("pop") is not None]
    winds = [f["wind_speed"] for f in forecasts if f.get("wind_speed") is not None]
    uvis = [f["uvi"] for f in forecasts if f.get("uvi") is not None]
    rains = [f["rain_1h"] for f in forecasts if f.get("rain_1h") is not None]

    # Build representative weather from forecast
    weather = {
        "temp": round(sum(temps) / len(temps), 1) if temps else None,
        "humidity": round(sum(humidities) / len(humidities)) if humidities else None,
        "pop": max(pops) if pops else 0,  # Use max pop (worst case for rain)
        "rain_1h": max(rains) if rains else 0,
        "uvi": max(uvis) if uvis else 0,  # Use max UV
        "wind_speed": max(winds) if winds else 0,  # Use max wind
        "wind_deg": forecasts[0].get("wind_deg"),
        "dew_point": forecasts[0].get("dew_point"),
        "clouds": forecasts[0].get("clouds"),
        "weather_main": forecasts[0].get("weather_main", ""),
        "weather_description": forecasts[0].get("weather_description", ""),
        "data_source": "forecast",
        "forecast_hours": len(forecasts),
    }

    return weather


def get_activity_advice(activity: str, ward_id: str, hours_ahead: int = 3) -> Dict[str, Any]:
    """Get weather-based activity advice for a location.

    Args:
        activity: Activity type (e.g., 'chay_bo', 'dua_dieu', 'picnic')
        ward_id: Ward ID
        hours_ahead: Hours ahead to check (0=now, 3=default for near future)

    Returns:
        Dictionary with advice and recommendations
    """
    weather = _get_weather_for_activity(ward_id, hours_ahead)

    if "error" in weather:
        return {
            "advice": "unknown",
            "reason": weather.get("message", "Khong co du lieu thoi tiet"),
            "activity": activity
        }

    issues = []
    recommendations = []

    # Get weather values - use None as default to detect missing data
    temp = weather.get("temp")
    humidity = weather.get("humidity")
    pop = weather.get("pop") or 0
    rain_1h = weather.get("rain_1h") or 0
    uvi = weather.get("uvi") or 0
    wind_speed = weather.get("wind_speed") or 0
    weather_main = weather.get("weather_main", "")

    # Check for missing data
    if temp is None:
        issues.append("Thiếu dữ liệu nhiệt độ")
        recommendations.append("Không thể đánh giá - dữ liệu thời tiết không có sẵn")

    if humidity is None:
        issues.append("Thiếu dữ liệu độ ẩm")
        recommendations.append("Không thể đánh giá - dữ liệu thời tiết không có sẵn")

    # Rain checks (using pop + weather_main since rain_1h is sparse)
    if pop > THRESHOLDS.get("POP_VERY_LIKELY", 0.8):
        issues.append(f"Khả năng mưa cao ({pop*100:.0f}%)")
        recommendations.append("Nên mang áo mưa hoặc ô, cân nhắc hoãn hoạt động ngoài trời")
    elif pop > THRESHOLDS.get("POP_LIKELY", 0.5):
        issues.append(f"Có thể mưa ({pop*100:.0f}%)")
        recommendations.append("Nên mang ô phòng mưa")
    elif weather_main in ("Rain", "Drizzle", "Thunderstorm"):
        issues.append(f"Đang có {weather_main}")
        recommendations.append("Nên đợi tạnh mưa hoặc mang áo mưa")

    # Temperature checks (only if temp is available)
    if temp is not None:
        if temp > KTTV_THRESHOLDS["NANG_NONG"]:
            issues.append(f"Nhiệt độ cao ({temp}°C)")
            recommendations.append("Nên chọn buổi sáng sớm (6-9h) hoặc chiều muộn (17h trở đi)")
        elif temp < KTTV_THRESHOLDS["RET_DAM"]:
            issues.append(f"Nhiệt độ thấp ({temp}°C)")
            recommendations.append("Mặc ấm, hạn chế ra ngoài vào ban đêm")

    # UV checks
    if uvi >= THRESHOLDS.get("UV_VERY_HIGH", 10):
        issues.append(f"UV cực cao ({uvi})")
        recommendations.append("Hạn chế ra ngoài 10h-14h, dùng kem chống nắng SPF50+")
    elif uvi >= THRESHOLDS.get("UV_HIGH", 7):
        issues.append(f"UV cao ({uvi})")
        recommendations.append("Đội mũ, dùng kem chống nắng")

    # Wind checks
    if wind_speed > THRESHOLDS.get("WIND_DANGEROUS", 20):
        issues.append(f"Gió rất mạnh ({wind_speed} m/s)")
        recommendations.append("NGUY HIỂM - Không nên ra ngoài")
    elif wind_speed > THRESHOLDS.get("WIND_STRONG", 10):
        issues.append(f"Gió mạnh ({wind_speed} m/s)")
        recommendations.append("Cẩn thận khi đi xe máy, tránh khu vực có cây cao")

    # Humidity checks (only if humidity is available)
    if humidity is not None:
        if humidity >= KTTV_THRESHOLDS["NOM_AM_HUMIDITY"]:
            issues.append(f"Độ ẩm rất cao ({humidity}%)")
            recommendations.append("Mang quần áo thay đổi, tránh hoạt động mạnh")

    # Hanoi-specific phenomena
    phenomena = detect_hanoi_weather_phenomena(weather)
    for p in phenomena["phenomena"]:
        issues.append(p["name"])
        recommendations.append(p["description"])

    # Determine overall advice
    if len(issues) == 0:
        advice = "nen"
        reason = "Thoi tiet thuan loi cho hoat dong ngoai troi"
    elif len(issues) == 1:
        advice = "co_the"
        reason = f"Can luu y: {issues[0]}"
    elif any("NGUY HIỂM" in r for r in recommendations):
        advice = "khong_nen"
        reason = f"Thoi tiet nguy hiem: {', '.join(issues)}"
    else:
        advice = "han_che"
        reason = f"Nhieu yeu to bat loi: {', '.join(issues)}"

    return {
        "advice": advice,
        "reason": reason,
        "recommendations": recommendations,
        "weather_summary": weather.get("weather_description", ""),
        "temp": temp,
        "humidity": humidity,
        "pop": pop,
        "uvi": uvi,
        "wind_speed": wind_speed,
        "phenomena": phenomena["phenomena"],
        "activity": activity,
        "data_source": weather.get("data_source", "current"),
    }


# Activity-specific advice templates
ACTIVITY_TEMPLATES = {
    "chay_bo": {
        "name": "Chay bo",
        "good_conditions": "Nhiet do 15-25°C, do am <80%, khong mua",
        "bad_conditions": "Nang nong, mua, gio manh"
    },
    "dua_dieu": {
        "name": "Dua dien/Cho con choi",
        "good_conditions": "Nhiet do 20-30°C, troi quang hoac may nhe",
        "bad_conditions": "UV cao, mua, gio manh"
    },
    "picnic": {
        "name": "Picnic/Du lich ngoai troi",
        "good_conditions": "Nhiet do 22-28°C, troi quang",
        "bad_conditions": "Mua, gio manh, nang nong"
    },
    "bike": {
        "name": "Di xe dap",
        "good_conditions": "Nhiet do 18-28°C, gio nhe (<5 m/s)",
        "bad_conditions": "Gio manh, mua, troi tuot"
    },
    "chup_anh": {
        "name": "Chup anh ngoai troi",
        "good_conditions": "Sang som hoac chieu muon, may nhe",
        "bad_conditions": "Nang gay, mua, suong mu"
    },
    "tap_the_duc": {
        "name": "Tap the duc ngoai troi",
        "good_conditions": "Nhiet do 18-25°C, do am thap",
        "bad_conditions": "Nang nong, nom am, mua"
    },
}


def get_activity_advice_detailed(activity: str, ward_id: str, hours_ahead: int = 3) -> Dict[str, Any]:
    """Get detailed activity advice with activity-specific recommendations.

    Args:
        activity: Activity type key
        ward_id: Ward ID
        hours_ahead: Hours ahead to check

    Returns:
        Detailed advice dictionary
    """
    advice = get_activity_advice(activity, ward_id, hours_ahead)

    activity_info = ACTIVITY_TEMPLATES.get(activity, {
        "name": activity,
        "good_conditions": "Thoi tiet thuan loi",
        "bad_conditions": "Thoi tiet bat loi"
    })

    advice["activity_name"] = activity_info["name"]
    advice["good_conditions"] = activity_info["good_conditions"]
    advice["bad_conditions"] = activity_info["bad_conditions"]

    return advice
