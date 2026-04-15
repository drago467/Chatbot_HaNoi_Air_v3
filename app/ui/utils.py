"""Weather display utilities for the Streamlit UI."""


def get_weather_emoji(weather_main: str) -> str:
    """Get weather emoji based on weather condition."""
    emoji_map = {
        "Clear": "☀️",
        "Clouds": "☁️",
        "Few clouds": "⛅",
        "Scattered clouds": "⛅",
        "Broken clouds": "☁️",
        "Overcast clouds": "☁️",
        "Rain": "🌧️",
        "Light rain": "🌦️",
        "Moderate rain": "🌧️",
        "Heavy rain": "⛈️",
        "Drizzle": "🌦️",
        "Thunderstorm": "⛈️",
        "Snow": "❄️",
        "Mist": "🌫️",
        "Fog": "🌫️",
        "Haze": "🌫️",
        "Smoke": "🌫️",
        "Dust": "💨",
        "Sand": "💨",
        "Ash": "🌋",
        "Squall": "🌬️",
        "Tornado": "🌪️",
    }
    return emoji_map.get(weather_main, "🌤️")


def get_wind_direction(deg: int) -> str:
    """Convert wind degree to Vietnamese direction.

    Standard wind sectors: N(0-22.5), NE(22.5-67.5), E(67.5-112.5),
    SE(112.5-157.5), S(157.5-202.5), SW(202.5-247.5), W(247.5-292.5), NW(292.5-337.5)
    """
    if deg is None:
        return ""
    directions = ["Bắc", "Đông Bắc", "Đông", "Đông Nam", "Nam", "Tây Nam", "Tây", "Tây Bắc"]
    idx = int((deg + 22.5) / 45) % 8
    return directions[idx]


def get_weather_description_vi(weather_main: str) -> str:
    """Translate weather condition to Vietnamese."""
    desc_map = {
        "Clear": "Trời quang",
        "Clouds": "Có mây",
        "Few clouds": "Mây ít",
        "Scattered clouds": "Mây rải rác",
        "Broken clouds": "Mây cụm",
        "Overcast clouds": "Mây đen",
        "Rain": "Mưa",
        "Light rain": "Mưa nhẹ",
        "Moderate rain": "Mưa vừa",
        "Heavy rain": "Mưa to",
        "Drizzle": "Mưa phùn",
        "Thunderstorm": "Giông",
        "Snow": "Tuyết",
        "Mist": "Sương mù",
        "Fog": "Sương mù",
        "Haze": "Mù",
        "Smoke": "Khói",
        "Dust": "Bụi",
        "Sand": "Cát",
        "Ash": "Tro",
        "Squall": "Gió giật",
        "Tornado": "Lốc xoáy",
    }
    return desc_map.get(weather_main, weather_main or "---")
