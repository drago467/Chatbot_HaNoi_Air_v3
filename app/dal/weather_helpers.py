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
        description = "Say nắng/sốc nhiệt gần như chắc chắn. Tránh ra ngoài."
    elif hi_c >= 40:
        level = "Nguy hiểm"
        description = "Say nắng rất có thể xảy ra. Hạn chế hoạt động ngoài trời."
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


def compute_comfort_index(
    temp: Optional[float],
    humidity: Optional[float],
    wind_speed: Optional[float],
    uvi: float = 0,
    pop: float = 0
) -> Optional[dict]:
    """Compute a 0-100 comfort score combining multiple weather factors.

    Higher score = more comfortable for outdoor activities.
    Optimal: 22-28°C, 40-70% humidity, light wind, low UV, no rain.

    Args:
        temp: Temperature in °C
        humidity: Relative humidity in %
        wind_speed: Wind speed in m/s
        uvi: UV index (0-11+)
        pop: Probability of precipitation (0-1)

    Returns:
        Dict with score, label, breakdown, and recommendation. None if temp is missing.
    """
    if temp is None:
        return None

    score = 100
    breakdown = {}

    # Temperature: optimal 22-28°C
    if temp < 10:
        penalty = (10 - temp) * 5
        breakdown["temp"] = f"Rất lạnh ({temp:.1f}°C), -{penalty:.0f} điểm"
    elif temp < 15:
        penalty = (15 - temp) * 4
        breakdown["temp"] = f"Lạnh ({temp:.1f}°C), -{penalty:.0f} điểm"
    elif temp < 22:
        penalty = (22 - temp) * 1.5
        breakdown["temp"] = f"Se lạnh ({temp:.1f}°C), -{penalty:.0f} điểm"
    elif temp <= 28:
        penalty = 0
        breakdown["temp"] = f"Lý tưởng ({temp:.1f}°C)"
    elif temp <= 32:
        penalty = (temp - 28) * 2
        breakdown["temp"] = f"Hơi nóng ({temp:.1f}°C), -{penalty:.0f} điểm"
    elif temp <= 35:
        penalty = (temp - 28) * 3
        breakdown["temp"] = f"Nóng ({temp:.1f}°C), -{penalty:.0f} điểm"
    else:
        penalty = (temp - 28) * 5
        breakdown["temp"] = f"Rất nóng ({temp:.1f}°C), -{penalty:.0f} điểm"
    score -= penalty

    # Humidity: optimal 40-70%
    if humidity is not None:
        if humidity > 90:
            penalty = (humidity - 70) * 0.8
            breakdown["humidity"] = f"Rất ẩm ({humidity:.0f}%), -{penalty:.0f} điểm"
        elif humidity > 80:
            penalty = (humidity - 70) * 0.5
            breakdown["humidity"] = f"Ẩm ({humidity:.0f}%), -{penalty:.0f} điểm"
        elif humidity > 70:
            penalty = (humidity - 70) * 0.3
            breakdown["humidity"] = f"Hơi ẩm ({humidity:.0f}%), -{penalty:.0f} điểm"
        else:
            penalty = 0
            breakdown["humidity"] = f"Tốt ({humidity:.0f}%)"
        score -= penalty

    # Wind: mild is fine, strong is bad
    if wind_speed is not None:
        if wind_speed > 15:
            penalty = (wind_speed - 6) * 3
            breakdown["wind"] = f"Gió rất mạnh ({wind_speed:.1f} m/s), -{penalty:.0f} điểm"
        elif wind_speed > 10:
            penalty = (wind_speed - 6) * 2
            breakdown["wind"] = f"Gió mạnh ({wind_speed:.1f} m/s), -{penalty:.0f} điểm"
        elif wind_speed > 6:
            penalty = (wind_speed - 6) * 1
            breakdown["wind"] = f"Gió hơi mạnh ({wind_speed:.1f} m/s), -{penalty:.0f} điểm"
        else:
            penalty = 0
            breakdown["wind"] = f"Gió nhẹ ({wind_speed:.1f} m/s)"
        score -= penalty

    # UV
    if uvi > 10:
        penalty = (uvi - 6) * 3
        breakdown["uvi"] = f"UV cực cao ({uvi}), -{penalty:.0f} điểm"
    elif uvi > 8:
        penalty = (uvi - 6) * 2.5
        breakdown["uvi"] = f"UV rất cao ({uvi}), -{penalty:.0f} điểm"
    elif uvi > 6:
        penalty = (uvi - 6) * 1.5
        breakdown["uvi"] = f"UV cao ({uvi}), -{penalty:.0f} điểm"
    else:
        penalty = 0
    score -= penalty

    # Rain probability
    if pop > 0.7:
        penalty = pop * 30
        breakdown["rain"] = f"Rất có thể mưa ({pop*100:.0f}%), -{penalty:.0f} điểm"
    elif pop > 0.5:
        penalty = pop * 20
        breakdown["rain"] = f"Có thể mưa ({pop*100:.0f}%), -{penalty:.0f} điểm"
    elif pop > 0.3:
        penalty = pop * 10
        breakdown["rain"] = f"Có thể mưa nhẹ ({pop*100:.0f}%), -{penalty:.0f} điểm"
    else:
        penalty = 0
    score -= penalty

    score = max(0, min(100, round(score)))

    # Label
    if score >= 80:
        label = "Rất thoải mái"
        recommendation = "Thời tiết lý tưởng cho mọi hoạt động ngoài trời."
    elif score >= 60:
        label = "Thoải mái"
        recommendation = "Phù hợp hoạt động ngoài trời, lưu ý một vài yếu tố."
    elif score >= 40:
        label = "Chấp nhận được"
        recommendation = "Có thể ra ngoài nhưng cần chuẩn bị (ô, áo khoác, kem chống nắng...)."
    elif score >= 20:
        label = "Khó chịu"
        recommendation = "Nên hạn chế hoạt động ngoài trời, chọn thời điểm khác nếu có thể."
    else:
        label = "Rất khó chịu"
        recommendation = "Nên ở trong nhà, tránh ra ngoài nếu không cần thiết."

    return {
        "score": score,
        "label": label,
        "recommendation": recommendation,
        "breakdown": breakdown,
    }