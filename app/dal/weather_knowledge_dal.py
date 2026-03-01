"""Weather Knowledge DAL - Detect Hanoi-specific weather phenomena."""

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
    month = weather_data.get("month", datetime.now().month)
    
    humidity = weather_data.get("humidity", 0)
    dew_point = weather_data.get("dew_point", 0)
    temp = weather_data.get("temp", 0)
    wind_deg = weather_data.get("wind_deg")
    wind_speed = weather_data.get("wind_speed", 0)
    weather_main = weather_data.get("weather_main", "")
    visibility = weather_data.get("visibility", 10000)
    clouds = weather_data.get("clouds", 0)
    
    # Calculate dew_point difference (temp - dew_point)
    # Small difference = air is close to saturation = nom am
    dew_point_diff = temp - dew_point if (temp and dew_point) else 999
    
    # 1. Nom Am (Spring Dampness) - ONLY Thang 2-4
    # True nom am: high humidity + dew_point very close to temp (chenh <= 2C)
    if month in [2, 3, 4]:
        if humidity >= KTTV_THRESHOLDS["NOM_AM_HUMIDITY"] and dew_point_diff <= 2:
            phenomena.append({
                "type": "nong_am",
                "name": "Nom am",
                "description": f"Nom am muc do cao! Do am {humidity}%, nhiet {temp}C, diem suong {dew_point}C - San nha lay mot, quan ao kho beo",
                "severity": "high"
            })
        elif humidity >= 85 and dew_point_diff <= 3:
            phenomena.append({
                "type": "nong_am",
                "name": "Nom am",
                "description": f"Do am cao muc do vua ({humidity}%) - Cam giac am uot",
                "severity": "medium"
            })
    
    # 2. Gio Lao (Loo) - Thang 5-8, Tay Nam + humidity < 55%
    # Gio Lao thuc su: gio Tay Nam qua Truong Son mat am -> nong kho
    if month in [5, 6, 7, 8]:
        if wind_deg and 180 <= wind_deg <= 270 and wind_speed > 5 and humidity < 55:
            phenomena.append({
                "type": "gio_lao",
                "name": "Gio Lao",
                "description": f"Gio Lao! Gio nong tu Tay Nam, do am chi {humidity}% - Troi oi nong, khong khi rat kho, can uong nhieu nuoc",
                "severity": "high"
            })
        elif wind_deg and 180 <= wind_deg <= 270 and wind_speed > 5:
            phenomena.append({
                "type": "gio_tay",
                "name": "Gio Tay",
                "description": f"Gio Tay manh ({wind_speed} m/s) - Can than cay ga",
                "severity": "medium"
            })
    
    # 3. Gio mua Dong Bac (Northeast Monsoon) - Thang 10-3
    # Gio mua Dong Bac: thang 10 den thang 3 nam sau
    if month in [10, 11, 12, 1, 2, 3]:
        if wind_deg and ((wind_deg >= 315 or wind_deg <= 90)) and wind_speed > 5:
            phenomena.append({
                "type": "gio_dong_bac",
                "name": "Gio mua Dong Bac",
                "description": f"Gio mua Dong Bac! Gio lanh tu Dong Bac, nhiet co the xuong {temp}C - Troi tro lanh, can mac am",
                "severity": "medium"
            })
    
    # 4. Ret dam (Cold Spell) - KTTV: Tavg <= 15C + clouds >= 70%
    # For current weather: check temp + clouds (proxy for cloudy day)
    if month in [11, 12, 1, 2, 3]:
        if temp < KTTV_THRESHOLDS["RET_DAM"] and clouds >= 70:
            severity = "high" if temp < KTTV_THRESHOLDS["RET_HAI"] else "medium"
            phenomena.append({
                "type": "ret_dam",
                "name": "Ret dam",
                "description": f"Ret dam! Nhiet do duoi {KTTV_THRESHOLDS['RET_DAM']}C, troi am u - Can mac am, han che ra ngoai",
                "severity": severity
            })
        elif temp < KTTV_THRESHOLDS["RET_DAM"]:
            phenomena.append({
                "type": "ret_nhe",
                "name": "Ret",
                "description": f"Nhiet do thap ({temp}C) - Can mac am nhe",
                "severity": "low"
            })
    
    # 5. Ret nang Ban - Thang 3, ret dot ngot sau khi da am
    if month == 3:
        if temp < 18 and humidity > 80:
            phenomena.append({
                "type": "ret_nang_ban",
                "name": "Ret nang Ban",
                "description": f"Ret nang Ban! Nhiet dot ngot xuong {temp}C sau chuoi ngay am u - Thay doi nhiet do dot ngot",
                "severity": "medium"
            })
    
    # 6. Nang nong (Heat Wave) - Thang 5-9
    if month in [5, 6, 7, 8, 9]:
        if temp >= KTTV_THRESHOLDS["NANG_NONG_DB"]:
            severity = "high"
        elif temp >= KTTV_THRESHOLDS["NANG_NONG_GAY_GAT"]:
            severity = "medium"
        elif temp >= KTTV_THRESHOLDS["NANG_NONG"]:
            severity = "low"
        else:
            severity = None
            
        if severity:
            phenomena.append({
                "type": "nang_nong",
                "name": "Nang nong",
                "description": f"Nang nong muc do {severity}: {temp}C - Han che ra ngoai gio trua, uong nhieu nuoc",
                "severity": severity
            })
    
    # 7. Suong mu (Fog/Mist) - Year-round in early morning
    # Priority: Dense fog (visibility < 200m) = always detect (affects traffic)
    if visibility < 200:
        phenomena.append({
            "type": "suong_mu",
            "name": "Suong mu day",
            "description": f"Suong mu rat day! Tam nhin chi {visibility}m - CAN THAN NGHIEM TRONG khi lai xe, hai cam",
            "severity": "high"
        })
    elif visibility < KTTV_THRESHOLDS["SUONG_MU_VISIBILITY"]:
        # Seasonal fog (mainly Jan-Feb, or Mar with low temp)
        if month in [1, 2] or (month == 3 and temp < 20):
            phenomena.append({
                "type": "suong_mu",
                "name": "Suong mu",
                "description": f"Suong mu! Tam nhin thap ({visibility}m) - Can than khi lai xe",
                "severity": "medium"
            })
    
    # 8. Mua dong (Thunderstorm) - Thang 5-9
    if month in [5, 6, 7, 8, 9]:
        if weather_main == "Thunderstorm":
            phenomena.append({
                "type": "mua_dong",
                "name": "Mua dong",
                "description": "Mua dong! Co giong kem mua - Tranh ra ngoai, can than cay ga va luoi dien",
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


def compare_with_seasonal(weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compare current weather with seasonal average.
    
    Args:
        weather_data: Current weather data
        
    Returns:
        Dictionary with comparison results
    """
    from datetime import datetime
    month = datetime.now().month
    seasonal = get_seasonal_average(month)
    
    comparisons = []
    
    if weather_data.get("temp"):
        diff = weather_data["temp"] - seasonal["temp_avg"]
        if diff > 3:
            comparisons.append(f"Nong hon binh thuong {diff:.1f}°C")
        elif diff < -3:
            comparisons.append(f"Lanh hon binh thuong {abs(diff):.1f}°C")
        else:
            comparisons.append("Nhiet do binh thuong theo mua")
    
    if weather_data.get("humidity"):
        hum_diff = weather_data["humidity"] - seasonal["humidity"]
        if hum_diff > 10:
            comparisons.append("Do am cao hon binh thuong")
        elif hum_diff < -10:
            comparisons.append("Do am thap hon binh thuong")
    
    return {
        "seasonal_avg": seasonal,
        "comparisons": comparisons,
        "month_name": [
            "Thang 1", "Thang 2", "Thang 3", "Thang 4", "Thang 5", "Thang 6",
            "Thang 7", "Thang 8", "Thang 9", "Thang 10", "Thang 11", "Thang 12"
        ][month - 1]
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
    if temp:
        if temp < 15:
            parts.append(f"nhiet do {temp}°C, lanh")
        elif temp < 25:
            parts.append(f"nhiet do {temp}°C, mat lanh")
        elif temp < 32:
            parts.append(f"nhiet do {temp}°C, thoai mai")
        else:
            parts.append(f"nhiet do {temp}°C, nong")
    
    # Weather condition
    if weather_main:
        from app.config.thresholds import WEATHER_DESCRIPTIONS
        desc = WEATHER_DESCRIPTIONS.get(weather_main, weather_main)
        parts.append(desc)
    
    # Humidity
    if humidity:
        if humidity >= 85:
            parts.append(f"do am cao {humidity}%")
    
    # Wind
    if wind_speed and wind_deg:
        from app.dal.weather_helpers import wind_deg_to_vietnamese
        wind_dir = wind_deg_to_vietnamese(wind_deg)
        parts.append(f"gio {wind_dir} {wind_speed} m/s")
    
    return ", ".join(parts) if parts else "Khong co du lieu"