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
        return "Không xác định"

    deg = deg % 360
    idx = round(deg / 45) % 8
    directions = ['Bắc', 'Đông Bắc', 'Đông', 'Đông Nam', 'Nam', 'Tây Nam', 'Tây', 'Tây Bắc']
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
        0: "Gió lặng",
        1: "Gió nhẹ",
        2: "Gió nhẹ",
        3: "Gió dịu",
        4: "Gió vừa",
        5: "Gió mạnh",
        6: "Gió rất mạnh",
        7: "Gió giật mạnh",
        8: "Gió bão",
        9: "Gió bão mạnh",
        10: "Gió bão rất mạnh",
        11: "Gió bão dữ dội",
        12: "Bão"
    }
    return descriptions.get(beaufort, "Không xác định")


def get_uv_status(uvi: Optional[float]) -> str:
    """Get UV status according to WHO guidelines.
    
    Args:
        uvi: UV index
        
    Returns:
        Vietnamese UV status
    """
    if uvi is None:
        return "Không xác định"
    if uvi <= 2:
        return "Thấp - An toàn"
    if uvi <= 5:
        return "Trung bình - Cần che nắng"
    if uvi <= 7:
        return "Cao - Hạn chế ra ngoài"
    if uvi <= 10:
        return "Rất cao - Không nên ra ngoài"
    return "Cực cao - Nguy hiểm"


def get_dew_point_status(dew_point: Optional[float]) -> str:
    """Get dew point status according to human perception.
    
    Args:
        dew_point: Dew point in Celsius
        
    Returns:
        Vietnamese dew point status
    """
    if dew_point is None:
        return "Không xác định"
    if dew_point < 10:
        return "Khô ráo, dễ chịu"
    if dew_point < 15:
        return "Hơi khô, dễ chịu"
    if dew_point < 18:
        return "Bắt đầu ẩm"
    if dew_point < 21:
        return "Ẩm, oi bức"
    if dew_point < 24:
        return "Rất ẩm, khó chịu"
    return "Nguy hiểm - Nồm ẩm"


def get_pressure_status(pressure: Optional[int]) -> str:
    """Get atmospheric pressure status.
    
    Args:
        pressure: Atmospheric pressure in hPa
        
    Returns:
        Vietnamese pressure status
    """
    if pressure is None:
        return "Không xác định"
    if pressure < 1000:
        return "Áp thấp"
    if pressure < 1010:
        return "Trung bình"
    if pressure < 1020:
        return "Áp trung bình cao"
    if pressure < 1030:
        return "Áp cao"
    return "Áp rất cao"


def get_feels_like_status(temp: Optional[float], feels_like: Optional[float]) -> str:
    """Compare actual temperature vs feels like temperature.
    
    Args:
        temp: Actual temperature in Celsius
        feels_like: Feels like temperature in Celsius
        
    Returns:
        Vietnamese comparison
    """
    if temp is None or feels_like is None:
        return "Không xác định"
    diff = feels_like - temp
    if diff > 3:
        return "Nóng hơn thực tế"
    if diff < -3:
        return "Lạnh hơn thực tế"
    return "Như thực tế"


def weather_main_to_vietnamese(weather_main: str) -> str:
    """Convert weather main condition to Vietnamese.
    
    Args:
        weather_main: Weather main condition from API
        
    Returns:
        Vietnamese description with Hanoi context
    """
    translations = {
        # 8xx - May (Clouds)
        "Clear": "Trời quang, không mây",
        "Clouds": "Trời mây",
        "Few clouds": "Ít mây, trời trong",
        "Scattered clouds": "Mây rải rác",
        "Broken clouds": "Nhiều mây",
        "Overcast": "Trời u ám, đầy mây",

        # 2xx - Gio (Thunderstorm) - Mua dong mua he
        "Thunderstorm": "Có giông kèm mưa",

        # 3xx - Mua phun (Drizzle) - Mua mua xuan
        "Drizzle": "Mưa phùn nhẹ",

        # 5xx - Mua (Rain)
        "Rain": "Có mưa",
        "Light rain": "Mưa nhỏ",
        "Moderate rain": "Mưa vừa",
        "Heavy intensity rain": "Mưa to",
        "Very heavy rain": "Mưa rất to",
        "Extreme rain": "Mưa cực lớn",
        "Freezing rain": "Mưa đóng băng (vùng núi cao)",

        # 7xx - Khi quyen (Atmosphere)
        "Mist": "Sương mù nhẹ (tầm nhìn 1-2km)",
        "Fog": "Sương mù dày (tầm nhìn < 1km)",
        "Haze": "Có mù (tầm nhìn giảm do bụi/ẩm)",
        "Smoke": "Có khói (có thể do đốt rơm)",
        "Dust": "Có bụi",
        "Sand": "Có cát",
        "Ash": "Có tro",
        "Squall": "Gió giật",
        "Tornado": "Thiên tai",

        # Snow (hiem vung nui cao)
        "Snow": "Tuyết (vùng núi cao)",
        "Light snow": "Tuyết nhẹ",
        "Heavy snow": "Tuyết to",
        "Sleet": "Mưa đông (vùng núi cao)",

        # Additional
        "Sky is clear": "Trời quang",
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
        level = "Cực nguy hiểm"
        description = "Say nan/soc nhiet gan nhu chắc chắn. Tránh ra ngoài."
    elif hi_c >= 40:
        level = "Nguy hiểm"
        description = "Say nan rất có thể xảy ra. Hạn chế hoạt động ngoài trời."
    elif hi_c >= 33:
        level = "Cảnh báo cao"
        description = "Có thể say nắng, chuột rút. Uống nhiều nước."
    elif hi_c >= 27:
        level = "Thận trọng"
        description = "Mệt mỏi có thể xảy ra khi hoạt động kéo dài."
    else:
        level = "An toàn"
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
        level = "Cực nguy hiểm"
        description = "Nguy hiểm cho sức khỏe. Ở trong nhà."
    elif WC <= -10:
        level = "Nguy hiểm"
        description = "Nguy cơ hạ thân nhiệt. Hạn chế ra ngoài."
    elif WC <= 0:
        level = "Rất lạnh"
        description = "Cơ thể mất nhiệt nhanh. Mặc đủ ấm."
    else:
        level = "Lạnh"
        description = "Có thể lạnh. Mặc áo ấm."
    
    return {
        "wind_chill": round(WC, 1),
        "level": level,
        "description": description
    }