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
    
    # Get weather values - use None as default to detect missing data
    temp = weather.get("temp")
    humidity = weather.get("humidity")
    pop = weather.get("pop", 0)
    rain_1h = weather.get("rain_1h", 0)
    uvi = weather.get("uvi", 0)
    wind_speed = weather.get("wind_speed", 0)
    
    # Check for missing data
    if temp is None:
        issues.append("Thiếu dữ liệu nhiệt độ")
        recommendations.append("Không thể đánh giá - dữ liệu thời tiết không có sẵn")
    
    # Check for missing humidity (critical for activity advice)
    if humidity is None:
        issues.append("Thiếu dữ liệu độ ẩm")
        recommendations.append("Không thể đánh giá - dữ liệu thời tiết không có sẵn")
    
    # Temperature checks (only if temp is available)
    if temp is not None:
        if temp > KTTV_THRESHOLDS["NANG_NONG"]:
            issues.append(f"Nhiệt độ cao ({temp}°C)")
            recommendations.append("Nên chọn buổi sáng sớm (6-9h) hoặc chiều muộn (17h trở đi)")
        elif temp < KTTV_THRESHOLDS["RET_DAM"]:
            issues.append(f"Nhiệt độ thấp ({temp}°C)")
            recommendations.append("Mặc ấm, hạn chế ra ngoài vào ban đêm")
    
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
