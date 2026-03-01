"""Activity Advice DAL - Activity-specific weather recommendations."""

from typing import Dict, Any, List
from app.dal.weather_dal import get_current_weather
from app.dal.weather_knowledge_dal import detect_hanoi_weather_phenomena
from app.config.thresholds import KTTV_THRESHOLDS, THRESHOLDS


def get_activity_advice(activity: str, ward_id: str) -> Dict[str, Any]:
    """Get weather-based activity advice for a location.
    
    Args:
        activity: Activity type (e.g., 'chay_bo', 'dua_dieu', 'picnic')
        ward_id: Ward ID
        
    Returns:
        Dictionary with advice and recommendations
    """
    weather = get_current_weather(ward_id)
    
    if "error" in weather:
        return {
            "advice": "unknown",
            "reason": weather.get("message", "Khong co du lieu thoi tiet"),
            "activity": activity
        }
    
    issues = []
    recommendations = []
    
    temp = weather.get("temp", 20)
    humidity = weather.get("humidity", 50)
    pop = weather.get("pop", 0)
    rain_1h = weather.get("rain_1h", 0)
    uvi = weather.get("uvi", 0)
    wind_speed = weather.get("wind_speed", 0)
    
    # Temperature checks
    if temp > KTTV_THRESHOLDS["NANG_NONG"]:
        issues.append(f"Nhiet do cao ({temp}°C)")
        recommendations.append("Nen chon buoi sang som (6-9h) hoac chieu muon (17h tro di)")
    elif temp < KTTV_THRESHOLDS["RET_DAM"]:
        issues.append(f"Nhiet do thap ({temp}°C)")
        recommendations.append("Mac am, han che ra ngoai vao ban dem")
    
    # Rain checks
    if pop > THRESHOLDS["POP_LIKELY"] or rain_1h > 0:
        issues.append(f"Kha nang mua {pop*100:.0f}%")
        recommendations.append("Mang theo ao mua hoac hoan lai hoat dong")
    
    # UV checks
    if uvi > THRESHOLDS["UV_HIGH"]:
        issues.append(f"UV cao ({uvi})")
        recommendations.append("Doi mu, kem chong nang SPF 30+")
    
    # Wind checks
    if wind_speed > THRESHOLDS["WIND_STRONG"]:
        issues.append(f"Gio manh ({wind_speed} m/s)")
        recommendations.append("Can than khi ra ngoai, tranh cay cao")
    
    # Humidity checks (Nom am)
    if humidity >= KTTV_THRESHOLDS["NOM_AM_HUMIDITY"]:
        issues.append(f"Do am rat cao ({humidity}%)")
        recommendations.append("Mang quan ao thay dong, tranh hoat dong manh")
    
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
        "activity": activity
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
        "bad_Conditions": "Gio manh, mua, troi tuot"
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


def get_activity_advice_detailed(activity: str, ward_id: str) -> Dict[str, Any]:
    """Get detailed activity advice with activity-specific recommendations.
    
    Args:
        activity: Activity type key
        ward_id: Ward ID
        
    Returns:
        Detailed advice dictionary
    """
    # Get basic advice
    advice = get_activity_advice(activity, ward_id)
    
    # Add activity-specific context
    activity_info = ACTIVITY_TEMPLATES.get(activity, {
        "name": activity,
        "good_conditions": "Thoi tiet thuan loi",
        "bad_conditions": "Thoi tiet bat loi"
    })
    
    advice["activity_name"] = activity_info["name"]
    advice["good_conditions"] = activity_info["good_conditions"]
    advice["bad_conditions"] = activity_info["bad_conditions"]
    
    return advice
