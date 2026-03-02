"""Vietnamese weather helpers - Wind direction, UV, Dew Point, etc."""

from typing import Optional


def wind_deg_to_vietnamese(deg: Optional[int]) -> str:
    """Convert wind degrees (0-360) to Vietnamese direction.
    
    Uses: round(deg / 45) % 8
    - 0-22 -> Bac (0)
    - 23-67 -> Dong Bac (1)
    - 68-112 -> Dong (2)
    - 113-157 -> Dong Nam (3)
    - 158-202 -> Nam (4)
    - 203-247 -> Tay Nam (5)
    - 248-292 -> Tay (6)
    - 293-337 -> Tay Bac (7)
    - 338-360 -> Bac (0)
    
    Args:
        deg: Wind direction in degrees (0-360)
        
    Returns:
        Vietnamese direction name
    """
    if deg is None:
        return "Khong xac dinh"
    
    deg = deg % 360
    idx = round(deg / 45) % 8
    directions = ['Bac', 'Dong Bac', 'Dong', 'Dong Nam', 'Nam', 'Tay Nam', 'Tay', 'Tay Bac']
    return directions[idx]


def wind_speed_to_beaufort(speed: Optional[float]) -> int:
    """Convert wind speed (m/s) to Beaufort scale (0-12).
    
    Args:
        speed: Wind speed in m/s
        
    Returns:
        Beaufort scale (0-12)
    """
    if speed is None:
        return 0
    if speed < 0.5:
        return 0
    elif speed < 1.5:
        return 1
    elif speed < 3.3:
        return 2
    elif speed < 5.5:
        return 3
    elif speed < 8.0:
        return 4
    elif speed < 10.8:
        return 5
    elif speed < 13.9:
        return 6
    elif speed < 17.2:
        return 7
    elif speed < 20.8:
        return 8
    elif speed < 24.5:
        return 9
    elif speed < 28.5:
        return 10
    elif speed < 32.7:
        return 11
    else:
        return 12


def wind_beaufort_vietnamese(beaufort: int) -> str:
    """Convert Beaufort scale to Vietnamese description.
    
    Args:
        beaufort: Beaufort scale (0-12)
        
    Returns:
        Vietnamese description
    """
    descriptions = {
        0: "Gio lang",
        1: "Gio nhe",
        2: "Gio nhe",
        3: "Gio diu",
        4: "Gio vua",
        5: "Gio manh",
        6: "Gio rat manh",
        7: "Gio viet manh",
        8: "Gio bao",
        9: "Gio bao manh",
        10: "Gio bao rat manh",
        11: "Gio bao du doi",
        12: "Bao"
    }
    return descriptions.get(beaufort, "Khong xac dinh")


def get_uv_status(uvi: Optional[float]) -> str:
    """Get UV status according to WHO guidelines.
    
    Args:
        uvi: UV index
        
    Returns:
        Vietnamese UV status
    """
    if uvi is None:
        return "Khong xac dinh"
    if uvi <= 2:
        return "Thap - An toan"
    if uvi <= 5:
        return "Trung binh - Can che nang"
    if uvi <= 7:
        return "Cao - Han che ra ngoai"
    if uvi <= 10:
        return "Rat cao - Khong nen ra ngoai"
    return "Cuc cao - Nguy hiem"


def get_dew_point_status(dew_point: Optional[float]) -> str:
    """Get dew point status according to human perception.
    
    Args:
        dew_point: Dew point in Celsius
        
    Returns:
        Vietnamese dew point status
    """
    if dew_point is None:
        return "Khong xac dinh"
    if dew_point < 10:
        return "Kho rao, de chiu"
    if dew_point < 15:
        return "Hoi kho, de chiu"
    if dew_point < 18:
        return "Bat dau am"
    if dew_point < 21:
        return "Am, oi buc"
    if dew_point < 24:
        return "Rat am, kho chiu"
    return "Nguy hiem - Nom am"


def get_pressure_status(pressure: Optional[int]) -> str:
    """Get atmospheric pressure status.
    
    Args:
        pressure: Atmospheric pressure in hPa
        
    Returns:
        Vietnamese pressure status
    """
    if pressure is None:
        return "Khong xac dinh"
    if pressure < 1000:
        return "Ap thap"
    if pressure < 1010:
        return "Trung binh"
    if pressure < 1020:
        return "Ap trung binh cao"
    if pressure < 1030:
        return "Ap cao"
    return "Ap rat cao"


def get_feels_like_status(temp: Optional[float], feels_like: Optional[float]) -> str:
    """Compare actual temperature vs feels like temperature.
    
    Args:
        temp: Actual temperature in Celsius
        feels_like: Feels like temperature in Celsius
        
    Returns:
        Vietnamese comparison
    """
    if temp is None or feels_like is None:
        return "Khong xac dinh"
    diff = feels_like - temp
    if diff > 3:
        return "Nong hon thuc te"
    if diff < -3:
        return "Lanh hon thuc te"
    return "Nhu thuc te"


def weather_main_to_vietnamese(weather_main: str) -> str:
    """Convert weather main condition to Vietnamese.
    
    Args:
        weather_main: Weather main condition from API
        
    Returns:
        Vietnamese description with Hanoi context
    """
    translations = {
        # 8xx - May (Clouds)
        "Clear": "Troi quang, khong may",
        "Clouds": "Troi may",
        "Few clouds": "It may, troi trong",
        "Scattered clouds": "May rai rac",
        "Broken clouds": "Nhieu may",
        "Overcast": "Troi u am, day may",
        
        # 2xx - Gio (Thunderstorm) - Mua dong mua he
        "Thunderstorm": "Co giong kem mua",
        
        # 3xx - Mua phun (Drizzle) - Mua mua xuan
        "Drizzle": "Mua phun nhe",
        
        # 5xx - Mua (Rain) 
        "Rain": "Co mua",
        "Light rain": "Mua nho",
        "Moderate rain": "Mua vua",
        "Heavy intensity rain": "Mua to",
        "Very heavy rain": "Mua rat to",
        "Extreme rain": "Mua cuc lon",
        "Freezing rain": "Mua dong bang (vung nui cao)",
        
        # 7xx - Khi quyen (Atmosphere)
        "Mist": "Suong mu nhe (tam nhin 1-2km)",
        "Fog": "Suong mu day (tam nhin < 1km)",
        "Haze": "Co mu (tam nhin giam do bui/am)",
        "Smoke": "Co khoi (co the do dot ram)",
        "Dust": "Co bui",
        "Sand": "Co cat",
        "Ash": "Co tro",
        "Squall": "Gio giat",
        "Tornado": "Thien tai",
        
        # Snow (hiem vung nui cao)
        "Snow": "Tuyet (vung nui cao)",
        "Light snow": "Tuyet nhe",
        "Heavy snow": "Tuyet to",
        "Sleet": "Mua dong (vung nui cao)",
        
        # Additional
        "Sky is clear": "Troi quang",
    }
    return translations.get(weather_main, weather_main)


def compute_heat_index(temp_c: Optional[float], humidity: Optional[int]) -> Optional[dict]:
    """Compute Heat Index using NWS Rothfusz formula.
    
    Only applies when temp > 27°C (80°F).
    Based on National Weather Service formula.
    
    Args:
        temp_c: Temperature in Celsius
        humidity: Relative humidity in %
        
    Returns:
        Dict with heat_index (°C) and level, or None if not applicable
    """
    if temp_c is None or humidity is None or temp_c < 27:
        return None
    
    # Convert to Fahrenheit for NWS formula
    T = temp_c * 9/5 + 32
    RH = humidity
    
    # Rothfusz regression equation
    HI = (-42.379 + 2.04901523*T + 10.14333127*RH 
          - 0.22475541*T*RH - 0.00683783*T**2 
          - 0.05481717*RH**2 + 0.00122874*T**2*RH 
          + 0.00085282*T*RH**2 - 0.00000199*T**2*RH**2)
    
    # Convert back to Celsius
    hi_c = (HI - 32) * 5/9
    
    # Determine warning level
    if hi_c >= 52:
        level = "Cuc nguy hiem"
        description = "Say nan/soc nhiet gan nhu chắc chắn. Tránh ra ngoài."
    elif hi_c >= 40:
        level = "Nguy hiem"
        description = "Say nan rất có thể xảy ra. Hạn chế hoạt động ngoài trời."
    elif hi_c >= 33:
        level = "Canh bao cao"
        description = "Có thể say nắng, chuột rút. Uống nhiều nước."
    elif hi_c >= 27:
        level = "Than trong"
        description = "Mệt mỏi có thể xảy ra khi hoạt động kéo dài."
    else:
        level = "An toan"
        description = "Hoạt động ngoài trời bình thường."
    
    return {
        "heat_index": round(hi_c, 1),
        "level": level,
        "description": description
    }


def compute_wind_chill(temp_c: Optional[float], wind_speed_ms: Optional[float]) -> Optional[dict]:
    """Compute Wind Chill using NWS formula.
    
    Only applies when temp <= 10°C (50°F) and wind > 1.3 m/s (4.6 km/h).
    Based on National Weather Service formula.
    
    Args:
        temp_c: Temperature in Celsius
        wind_speed_ms: Wind speed in m/s
        
    Returns:
        Dict with wind_chill (°C) and level, or None if not applicable
    """
    if temp_c is None or wind_speed_ms is None:
        return None
    
    if temp_c > 10 or wind_speed_ms <= 1.3:
        return None
    
    # Convert to km/h for NWS formula
    V = wind_speed_ms * 3.6
    
    # NWS Wind Chill formula
    WC = 13.12 + 0.6215*temp_c - 11.37*V**0.16 + 0.3965*temp_c*V**0.16
    
    # Determine warning level
    if WC <= -20:
        level = "Cuc nguy hiem"
        description = "Nguy hiểm cho sức khỏe. Ở trong nhà."
    elif WC <= -10:
        level = "Nguy hiem"
        description = "Nguy cơ hạ thân nhiệt. Hạn chế ra ngoài."
    elif WC <= 0:
        level = "Rat lanh"
        description = "Cơ thể mất nhiệt nhanh. Mặc đủ ấm."
    else:
        level = "Lanh"
        description = "Có thể lạnh. Mặc áo ấm."
    
    return {
        "wind_chill": round(WC, 1),
        "level": level,
        "description": description
    }