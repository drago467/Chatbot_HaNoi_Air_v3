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
        5: "Gio tuong doi lon",
        6: "Gio manh",
        7: "Gio manh",
        8: "Gio bao",
        9: "Gio bao manh",
        10: "Gio bao",
        11: "Gio bao violent",
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
        Vietnamese description
    """
    translations = {
        "Clear": "Troi quang",
        "Clouds": "Troi co may",
        "Rain": "Co mua",
        "Drizzle": "Mua phun",
        "Thunderstorm": "Co giong",
        "Mist": "Co suong mu",
        "Fog": "Suong mu day",
        "Haze": "Co mu",
        "Smoke": "Co khoi",
        "Dust": "Co bui",
        "Sand": "Co cat",
        "Ash": "Tro",
        "Squall": "Gio giat",
        "Tornado": "Thien tai",
    }
    return translations.get(weather_main, weather_main)
