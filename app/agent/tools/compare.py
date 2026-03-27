"""Compare tools — compare_weather, compare_with_yesterday, seasonal_comparison.

Tất cả đều hỗ trợ 3 tier nhất quán.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============== Tool: compare_weather ==============

class CompareWeatherInput(BaseModel):
    location_hint1: str = Field(description="Tên địa điểm 1. Ví dụ: 'Cầu Giấy', 'Hoàn Kiếm'")
    location_hint2: str = Field(description="Tên địa điểm 2. Ví dụ: 'Đống Đa', 'Tây Hồ'")


@tool(args_schema=CompareWeatherInput)
def compare_weather(location_hint1: str, location_hint2: str) -> dict:
    """So sánh thời tiết HIỆN TẠI giữa HAI địa điểm.

    DÙNG KHI: "A và B nơi nào nóng/lạnh/ẩm hơn?", "so sánh thời tiết A với B",
    "Cầu Giấy hay Hoàn Kiếm mát hơn?".
    Hỗ trợ: so sánh giữa bất kỳ cặp phường-phường, quận-quận, phường-quận.
    KHÔNG DÙNG KHI: so sánh hôm nay vs hôm qua (dùng compare_with_yesterday),
    so sánh với trung bình mùa (dùng get_seasonal_comparison).
    """
    from app.agent.utils import auto_resolve_location
    from app.agent.dispatch import normalize_agg_keys

    # Resolve both locations
    r1 = auto_resolve_location(location_hint=location_hint1)
    r2 = auto_resolve_location(location_hint=location_hint2)

    if r1["status"] != "ok":
        return {"error": "location1_not_found", "message": f"Không tìm thấy địa điểm: {location_hint1}"}
    if r2["status"] != "ok":
        return {"error": "location2_not_found", "message": f"Không tìm thấy địa điểm: {location_hint2}"}

    # Get weather for each location at its natural level
    w1 = _get_weather_at_level(r1)
    w2 = _get_weather_at_level(r2)

    if w1.get("error"):
        return {"error": "no_data_location1", "message": w1.get("message", "")}
    if w2.get("error"):
        return {"error": "no_data_location2", "message": w2.get("message", "")}

    # Normalize keys for comparison
    w1 = normalize_agg_keys(w1)
    w2 = normalize_agg_keys(w2)

    # Build comparison
    temp1 = w1.get("temp") or w1.get("avg_temp")
    temp2 = w2.get("temp") or w2.get("avg_temp")
    hum1 = w1.get("humidity") or w1.get("avg_humidity")
    hum2 = w2.get("humidity") or w2.get("avg_humidity")

    name1 = _get_location_name(r1)
    name2 = _get_location_name(r2)

    # Temperature comparison text
    if temp1 is not None and temp2 is not None:
        temp_diff = temp1 - temp2
        if abs(temp_diff) <= 2:
            temp_text = "Nhiệt độ tương tự"
        elif temp_diff > 0:
            temp_text = f"{name1} nóng hơn {name2} {abs(temp_diff):.1f}C"
        else:
            temp_text = f"{name2} nóng hơn {name1} {abs(temp_diff):.1f}C"
    else:
        temp_text = "Không đủ dữ liệu nhiệt độ để so sánh"
        temp_diff = None

    return {
        "location1": {"name": name1, "weather": w1, "info": r1.get("data", {})},
        "location2": {"name": name2, "weather": w2, "info": r2.get("data", {})},
        "differences": {
            "temp_diff": round(temp_diff, 1) if temp_diff is not None else None,
            "humidity_diff": round((hum1 or 0) - (hum2 or 0), 1) if hum1 and hum2 else None,
        },
        "comparison_text": temp_text,
    }


def _get_weather_at_level(resolved: dict) -> dict:
    """Get current weather at natural level (ward/district/city)."""
    level = resolved.get("level", "ward")
    data = resolved.get("data", {})

    if level == "city":
        from app.dal.weather_aggregate_dal import get_city_current_weather
        return get_city_current_weather()
    elif level == "district":
        district_name = data.get("district_name_vi", "")
        from app.dal.weather_aggregate_dal import get_district_current_weather
        return get_district_current_weather(district_name)
    else:
        ward_id = data.get("ward_id", "")
        from app.dal.weather_dal import get_current_weather
        return get_current_weather(ward_id)


def _get_location_name(resolved: dict) -> str:
    """Extract human-readable name from resolved location."""
    data = resolved.get("data", {})
    level = resolved.get("level", "ward")
    if level == "city":
        return "Hà Nội"
    elif level == "district":
        return data.get("district_name_vi", "")
    else:
        return data.get("ward_name_vi", data.get("district_name_vi", ""))


# ============== Tool: compare_with_yesterday ==============

class CompareWithYesterdayInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")


@tool(args_schema=CompareWithYesterdayInput)
def compare_with_yesterday(ward_id: str = None, location_hint: str = None) -> dict:
    """So sánh thời tiết HÔM NAY với HÔM QUA cho một địa điểm.

    DÙNG KHI: "hôm nay nóng hơn hôm qua không?", "so với hôm qua thế nào?".
    Hỗ trợ: phường/xã, quận/huyện, toàn Hà Nội.
    KHÔNG DÙNG KHI: so sánh 2 địa điểm (dùng compare_weather).
    """
    from app.agent.dispatch import resolve_and_dispatch
    from app.dal.comparison_dal import (
        compare_with_previous_day as dal_ward,
        compare_district_with_previous_day as dal_district,
        compare_city_with_previous_day as dal_city,
    )

    return resolve_and_dispatch(
        ward_id=ward_id,
        location_hint=location_hint,
        default_scope="city",
        ward_fn=dal_ward,
        district_fn=dal_district,
        city_fn=dal_city,
        label="so sánh hôm nay vs hôm qua",
    )


# ============== Tool: get_seasonal_comparison ==============

class GetSeasonalComparisonInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")


@tool(args_schema=GetSeasonalComparisonInput)
def get_seasonal_comparison(ward_id: str = None, location_hint: str = None) -> dict:
    """So sánh thời tiết hiện tại với trung bình mùa (climatology Hà Nội).

    DÙNG KHI: "nóng hơn bình thường không?", "thời tiết có bất thường không?",
    "so với mùa này thế nào?".
    Hỗ trợ: phường/xã, quận/huyện, toàn Hà Nội.
    Trả về: nhiệt độ/độ ẩm hiện tại vs trung bình tháng, nhận xét chênh lệch.
    """
    from app.agent.utils import auto_resolve_location
    from app.dal.weather_knowledge_dal import compare_with_seasonal
    from app.agent.dispatch import normalize_agg_keys

    # Get current weather at appropriate level
    if not ward_id and not location_hint:
        from app.dal.weather_aggregate_dal import get_city_current_weather
        weather = get_city_current_weather()
        resolved_data = {"city_name": "Hà Nội"}
    else:
        resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
        if resolved["status"] != "ok":
            return {"error": resolved["status"], "message": resolved.get("message", "")}
        weather = _get_weather_at_level(resolved)
        resolved_data = resolved.get("data", {})

    if not weather or weather.get("error"):
        return {"error": "no_weather_data",
                "message": "Không lấy được dữ liệu thời tiết hiện tại để so sánh với mùa",
                "suggestion": "Thử hỏi thời tiết hiện tại trước"}

    weather = normalize_agg_keys(weather)
    seasonal = compare_with_seasonal(weather)

    return {
        "current": weather,
        "seasonal_avg": seasonal["seasonal_avg"],
        "comparisons": seasonal["comparisons"],
        "month_name": seasonal["month_name"],
        "resolved_location": resolved_data,
    }
