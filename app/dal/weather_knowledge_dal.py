"""Weather Knowledge DAL - Detect Hanoi-specific weather phenomena."""

from app.dal.timezone_utils import now_ict
from typing import Dict, Any, List
from app.config.thresholds import KTTV_THRESHOLDS


def detect_hanoi_weather_phenomena(weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """Detect weather phenomena specific to Hanoi region.
    
    Uses KTTV Vietnam standards and seasonal patterns.
    
    Args:
        weather_data: Weather data dictionary (should include 'month' 1-12)
        
    Returns:
        Dictionary with detected phenomena
    """
    from datetime import datetime
    
    phenomena = []
    
    # Get current month (from data or now)
    month = weather_data.get("month", now_ict().month)
    
    humidity = weather_data.get("humidity")
    dew_point = weather_data.get("dew_point")
    temp = weather_data.get("temp")
    wind_deg = weather_data.get("wind_deg")
    wind_speed = weather_data.get("wind_speed", 0)
    weather_main = weather_data.get("weather_main", "")
    visibility = weather_data.get("visibility")
    clouds = weather_data.get("clouds")
    
    # Calculate dew_point difference (temp - dew_point)
    # Small difference = air is close to saturation = nom am
    dew_point_diff = temp - dew_point if temp is not None and dew_point is not None else 999
    
    # 1. Nồm Ẩm (Spring Dampness) - ONLY Tháng 2-4
    # True nồm ẩm: high humidity + dew_point very close to temp (chênh <= 2C)
    if month in [2, 3, 4]:
        if humidity is not None and humidity >= KTTV_THRESHOLDS["NOM_AM_HUMIDITY"] and dew_point_diff <= 2:
            phenomena.append({
                "type": "nom_am",
                "name": "Nồm ẩm",
                "description": f"Nồm ẩm mức độ cao! Độ ẩm {humidity}%, nhiệt {temp}°C, điểm sương {dew_point}°C - Sàn nhà lấy mồ hôi, quần áo khó khô",
                "severity": "high"
            })
        elif humidity is not None and humidity >= KTTV_THRESHOLDS["NOM_AM_HUMIDITY_MEDIUM"] and dew_point_diff <= KTTV_THRESHOLDS["NOM_AM_DEW_DIFF_MEDIUM"]:
            phenomena.append({
                "type": "nom_am",
                "name": "Nồm ẩm",
                "description": f"Độ ẩm cao mức độ vừa ({humidity}%) - Cảm giác ẩm ướt",
                "severity": "medium"
            })
    
    # 2. Gió Lào (Loo) - Tháng 5-8, Tây Nam + humidity < 55%
    # Gió Lào thực sự: gió Tây Nam qua Trường Sơn mất ẩm -> nóng khô
    if month in [5, 6, 7, 8]:
        if wind_deg is not None and 180 <= wind_deg <= 270 and wind_speed > 5 and humidity < 55:
            phenomena.append({
                "type": "gio_lao",
                "name": "Gió Lào",
                "description": f"Gió Lào! Gió nóng từ Tây Nam, độ ẩm chỉ {humidity}% - Trời oi nóng, không khí rất khô, cần uống nhiều nước",
                "severity": "high"
            })
        elif wind_deg is not None and 180 <= wind_deg <= 270 and wind_speed > 5:
            phenomena.append({
                "type": "gio_tay",
                "name": "Gió Tây",
                "description": f"Gió Tây mạnh ({wind_speed} m/s) - Cẩn thận cây gã",
                "severity": "medium"
            })
    
    # 3. Gió mùa Đông Bắc (Northeast Monsoon) - Tháng 10-3
    # Gió mùa Đông Bắc: tháng 10 đến tháng 3 năm sau
    if month in [10, 11, 12, 1, 2, 3]:
        if wind_deg is not None and ((wind_deg >= 315 or wind_deg <= 90)) and wind_speed > 5:
            phenomena.append({
                "type": "gio_dong_bac",
                "name": "Gió mùa Đông Bắc",
                "description": f"Gió mùa Đông Bắc! Gió lạnh từ Đông Bắc, nhiệt có thể xuống {temp}°C - Trời trở lạnh, cần mặc ấm",
                "severity": "medium"
            })
    
    # 4. Rét đậm (Cold Spell) - KTTV: Tavg <= 15C + clouds >= 70%
    # For current weather: check temp + clouds (proxy for cloudy day)
    if month in [11, 12, 1, 2, 3]:
        if temp is not None and temp < KTTV_THRESHOLDS["RET_DAM"] and clouds is not None and clouds >= 70:
            severity = "high" if temp < KTTV_THRESHOLDS["RET_HAI"] else "medium"
            phenomena.append({
                "type": "ret_dam",
                "name": "Rét đậm",
                "description": f"Rét đậm! Nhiệt độ dưới {KTTV_THRESHOLDS['RET_DAM']}°C, trời âm u - Cần mặc ấm, hạn chế ra ngoài",
                "severity": severity
            })
        elif temp is not None and temp < KTTV_THRESHOLDS["RET_DAM"]:
            phenomena.append({
                "type": "ret_nhe",
                "name": "Rét",
                "description": f"Nhiệt độ thấp ({temp}°C) - Cần mặc ấm nhẹ",
                "severity": "low"
            })
    
    # 5. Rét nàng Bân - Tháng 3, rét đột ngột sau khi đã ấm
    if month == 3:
        if temp is not None and temp < 18 and humidity is not None and humidity > 80:
            phenomena.append({
                "type": "ret_nang_ban",
                "name": "Rét nàng Bân",
                "description": f"Rét nàng Bân! Nhiệt đột ngột xuống {temp}°C sau chuỗi ngày ấm u - Thay đổi nhiệt độ đột ngột",
                "severity": "medium"
            })
    
    # 6. Nắng nóng (Heat Wave) - Tháng 5-9
    if month in [5, 6, 7, 8, 9]:
        if temp is not None and temp >= KTTV_THRESHOLDS["NANG_NONG_DB"]:
            severity = "high"
        elif temp is not None and temp >= KTTV_THRESHOLDS["NANG_NONG_GAY_GAT"]:
            severity = "medium"
        elif temp is not None and temp >= KTTV_THRESHOLDS["NANG_NONG"]:
            severity = "low"
        else:
            severity = None

        if severity:
            phenomena.append({
                "type": "nang_nong",
                "name": "Nắng nóng",
                "description": f"Nắng nóng mức độ {severity}: {temp}°C - Hạn chế ra ngoài giờ trưa, uống nhiều nước",
                "severity": severity
            })
    
    # 7. Sương mù (Fog/Mist) - Year-round in early morning
    # Priority: Dense fog (visibility < 200m) = always detect (affects traffic)
    if visibility is not None and visibility < 200:
        phenomena.append({
            "type": "suong_mu",
            "name": "Sương mù dày",
            "description": f"Sương mù rất dày! Tầm nhìn chỉ {visibility}m - CẨN THẬN NGHIÊM TRỌNG khi lái xe, hai cầm",
            "severity": "high"
        })
    elif visibility is not None and visibility < KTTV_THRESHOLDS["SUONG_MU_VISIBILITY"]:
        # Seasonal fog (mainly Jan-Feb, or Mar with low temp)
        if month in [1, 2] or (month == 3 and temp < 20):
            phenomena.append({
                "type": "suong_mu",
                "name": "Sương mù",
                "description": f"Sương mù! Tầm nhìn thấp ({visibility}m) - Cẩn thận khi lái xe",
                "severity": "medium"
            })
    
    # 8. Mưa dông (Thunderstorm) - Tháng 4-10
    if month in [4, 5, 6, 7, 8, 9, 10]:
        if weather_main == "Thunderstorm":
            phenomena.append({
                "type": "mua_dong",
                "name": "Mưa dông",
                "description": "Mưa dông! Có giông kèm mưa - Tránh ra ngoài, cẩn thận cây gã và lưới điện",
                "severity": "high"
            })
    
    return {
        "phenomena": phenomena,
        "has_dangerous": any(p.get("severity") == "high" for p in phenomena)
    }



def get_seasonal_average(month: int) -> Dict[str, Any]:
    """Get average weather for a specific month in Hanoi.
    
    Args:
        month: Month number (1-12)
        
    Returns:
        Dictionary with average weather data
    """
    seasonal_data = {
        1: {"temp_avg": 17, "temp_min": 14, "temp_max": 20, "humidity": 75, "rain_days": 6},
        2: {"temp_avg": 19, "temp_min": 15, "temp_max": 23, "humidity": 78, "rain_days": 8},
        3: {"temp_avg": 22, "temp_min": 18, "temp_max": 27, "humidity": 80, "rain_days": 12},
        4: {"temp_avg": 27, "temp_min": 22, "temp_max": 32, "humidity": 82, "rain_days": 15},
        5: {"temp_avg": 30, "temp_min": 25, "temp_max": 35, "humidity": 80, "rain_days": 18},
        6: {"temp_avg": 31, "temp_min": 26, "temp_max": 36, "humidity": 78, "rain_days": 16},
        7: {"temp_avg": 31, "temp_min": 26, "temp_max": 36, "humidity": 76, "rain_days": 14},
        8: {"temp_avg": 30, "temp_min": 26, "temp_max": 35, "humidity": 80, "rain_days": 17},
        9: {"temp_avg": 29, "temp_min": 24, "temp_max": 33, "humidity": 79, "rain_days": 14},
        10: {"temp_avg": 26, "temp_min": 21, "temp_max": 30, "humidity": 76, "rain_days": 10},
        11: {"temp_avg": 22, "temp_min": 18, "temp_max": 26, "humidity": 74, "rain_days": 7},
        12: {"temp_avg": 18, "temp_min": 14, "temp_max": 22, "humidity": 73, "rain_days": 5},
    }
    return seasonal_data.get(month, {"temp_avg": 25, "humidity": 75, "rain_days": 10})


def compare_with_seasonal(weather_data: Dict[str, Any], month: int = None) -> Dict[str, Any]:
    """Compare current weather with seasonal average.
    
    Args:
        weather_data: Current weather data
        month: Month (1-12). If not provided, uses current month.
        
    Returns:
        Dictionary with comparison results
    """
    from datetime import datetime
    if month is None:
        month = now_ict().month
    seasonal = get_seasonal_average(month)
    
    comparisons = []
    
    if weather_data.get("temp") is not None:
        diff = weather_data["temp"] - seasonal["temp_avg"]
        if diff > 3:
            comparisons.append(f"Nóng hơn bình thường {diff:.1f}°C")
        elif diff < -3:
            comparisons.append(f"Lạnh hơn bình thường {abs(diff):.1f}°C")
        else:
            comparisons.append("Nhiệt độ bình thường theo mùa")
    
    if weather_data.get("humidity") is not None:
        hum_diff = weather_data["humidity"] - seasonal["humidity"]
        if hum_diff > 10:
            comparisons.append("Độ ẩm cao hơn bình thường")
        elif hum_diff < -10:
            comparisons.append("Độ ẩm thấp hơn bình thường")
    
    # Validate month
    month_names = [
        "Tháng 1", "Tháng 2", "Tháng 3", "Tháng 4", "Tháng 5", "Tháng 6",
        "Tháng 7", "Tháng 8", "Tháng 9", "Tháng 10", "Tháng 11", "Tháng 12"
    ]
    if month < 1 or month > 12:
        month_name = "Không xác định"
    else:
        month_name = month_names[month - 1]
    
    return {
        "seasonal_avg": seasonal,
        "comparisons": comparisons,
        "month_name": month_name
    }


def get_weather_summary_text(weather_data: Dict[str, Any]) -> str:
    """Generate a natural language weather summary in Vietnamese.
    
    Args:
        weather_data: Weather data dictionary
        
    Returns:
        Vietnamese weather summary
    """
    temp = weather_data.get("temp")
    humidity = weather_data.get("humidity")
    wind_speed = weather_data.get("wind_speed")
    wind_deg = weather_data.get("wind_deg")
    weather_main = weather_data.get("weather_main", "")
    
    from app.dal.weather_helpers import wind_deg_to_vietnamese
    
    parts = []
    
    # Temperature
    if temp is not None:
        if temp < 15:
            parts.append(f"nhiệt độ {temp}°C, lạnh")
        elif temp < 25:
            parts.append(f"nhiệt độ {temp}°C, mát lạnh")
        elif temp < 32:
            parts.append(f"nhiệt độ {temp}°C, thoải mái")
        else:
            parts.append(f"nhiệt độ {temp}°C, nóng")
    
    # Weather condition
    if weather_main:
        from app.config.thresholds import WEATHER_DESCRIPTIONS
        desc = WEATHER_DESCRIPTIONS.get(weather_main, weather_main)
        parts.append(desc)
    
    # Humidity
    if humidity is not None:
        from app.config.thresholds import KTTV_THRESHOLDS
        if humidity >= KTTV_THRESHOLDS.get("NOM_AM_HUMIDITY_MEDIUM", 85):
            parts.append(f"độ ẩm cao {humidity}%")
    
    # Wind
    if wind_speed is not None and wind_deg is not None:
        wind_dir = wind_deg_to_vietnamese(wind_deg)
        parts.append(f"gió {wind_dir} {wind_speed} m/s")
    
    return ", ".join(parts) if parts else "Không có dữ liệu"