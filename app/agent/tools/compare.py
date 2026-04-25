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
    """SO SÁNH 2 ĐỊA ĐIỂM tại thời điểm HIỆN TẠI (1 call duy nhất, KHÔNG 2× current).

    DÙNG KHI: "A và B nơi nào nóng/lạnh/ẩm hơn?", "so sánh A với B", "A so với B".

    KHÔNG DÙNG KHI:
        - today vs yesterday 1 địa điểm → compare_with_yesterday.
        - hiện tại vs TB mùa → get_seasonal_comparison.
        - 3+ quận → get_district_ranking hoặc get_district_multi_compare.
        - So 2 ngày khác nhau của cùng 1 nơi → gọi 2 tool get_daily_summary riêng.

    Returns: Flat VN dict: `"địa điểm 1"` + `"địa điểm 2"` (mỗi block là flat VN weather),
    `"chênh lệch"` (nhiệt độ/độ ẩm), `"tóm tắt"`.
    """
    from app.agent.utils import auto_resolve_location
    from app.agent.dispatch import normalize_agg_keys, router_scope_var

    scope = router_scope_var.get(None)

    # Resolve both locations
    r1 = auto_resolve_location(location_hint=location_hint1, target_scope=scope)
    r2 = auto_resolve_location(location_hint=location_hint2, target_scope=scope)

    from app.agent.tools.output_builder import build_error_output
    if r1["status"] != "ok":
        return build_error_output({"error": "location1_not_found", "message": f"Không tìm thấy địa điểm: {location_hint1}"})
    if r2["status"] != "ok":
        return build_error_output({"error": "location2_not_found", "message": f"Không tìm thấy địa điểm: {location_hint2}"})

    # Get weather for each location at its natural level
    w1 = _get_weather_at_level(r1)
    w2 = _get_weather_at_level(r2)

    if w1.get("error"):
        return build_error_output({"error": "no_data_location1", "message": w1.get("message", "")})
    if w2.get("error"):
        return build_error_output({"error": "no_data_location2", "message": w2.get("message", "")})

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

    from app.agent.tools.output_builder import build_compare_output
    return build_compare_output({
        "location1": {"name": name1, "weather": w1, "info": r1.get("data", {})},
        "location2": {"name": name2, "weather": w2, "info": r2.get("data", {})},
        "differences": {
            "temp_diff": round(temp_diff, 1) if temp_diff is not None else None,
            "humidity_diff": round((hum1 or 0) - (hum2 or 0), 1) if hum1 and hum2 else None,
        },
        "comparison_text": temp_text,
    })


def _get_weather_at_level(resolved: dict) -> dict:
    """Get current weather at natural level (ward/district/city)."""
    level = resolved.get("level", "ward")
    data = resolved.get("data", {})

    if level == "city":
        from app.dal.weather_aggregate_dal import get_city_current_weather
        return get_city_current_weather()
    elif level == "district":
        district_id = data.get("district_id")
        if not district_id:
            return {"error": "no_district_id", "message": "Không xác định được quận/huyện"}
        from app.dal.weather_aggregate_dal import get_district_current_weather
        return get_district_current_weather(district_id)
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
    """SO SÁNH HÔM NAY vs HÔM QUA cho 1 địa điểm — PAST direction only.

    DÙNG KHI: user hỏi today-vs-yesterday:
        "hôm nay nóng hơn hôm qua không?", "so với hôm qua thế nào?",
        "mấy hôm nay hay thay đổi nhỉ?" (ám chỉ past).

    KHÔNG DÙNG KHI:
        - "so hôm nay vs NGÀY MAI" (FUTURE direction — tool này SAI hướng).
          Thay: gọi get_current_weather + get_daily_forecast(start_date=tomorrow_iso, days=1),
          so sánh 2 block trong câu trả lời.
        - So 2 địa điểm khác nhau cùng thời điểm → compare_weather.
        - So 1 địa điểm qua nhiều ngày → get_weather_period.

    Returns: Flat VN dict: `"hôm nay"` + `"hôm qua"` (block mini), `"thay đổi"` list,
    `"tóm tắt"`. Nếu error "not_enough_data" → gợi ý xem thời tiết hiện tại.
    """
    from app.agent.dispatch import resolve_and_dispatch
    from app.dal.comparison_dal import (
        compare_with_previous_day as dal_ward,
        compare_district_with_previous_day as dal_district,
        compare_city_with_previous_day as dal_city,
    )

    from app.agent.tools.output_builder import build_compare_with_yesterday_output
    raw = resolve_and_dispatch(
        ward_id=ward_id,
        location_hint=location_hint,
        default_scope="city",
        ward_fn=dal_ward,
        district_fn=dal_district,
        city_fn=dal_city,
        label="so sánh hôm nay vs hôm qua",
    )
    return build_compare_with_yesterday_output(raw)


# ============== Tool: get_seasonal_comparison ==============

class GetSeasonalComparisonInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")


@tool(args_schema=GetSeasonalComparisonInput)
def get_seasonal_comparison(ward_id: str = None, location_hint: str = None) -> dict:
    """SO HIỆN TẠI vs TRUNG BÌNH CLIMATOLOGY tháng hiện tại (Hà Nội).

    DÙNG KHI: user hỏi bất thường THEO MÙA:
        "nóng hơn bình thường không?", "thời tiết có bất thường không?",
        "dạo này khó chịu quá nhỉ?", "mùa này thường thế nào?".

    KHÔNG DÙNG KHI:
        - "so tuần trước / hôm qua / ngày mai" — đó là so 2 thời điểm, không phải climatology.
          Dùng compare_with_yesterday hoặc compare_weather.
        - "bây giờ mấy độ" — dùng get_current_weather.

    Returns: Flat VN dict: `"hiện tại"`, `"trung bình mùa"`, `"nhận xét"` (list string VN).
    Error "no_weather_data" → gợi ý hỏi thời tiết hiện tại trước.
    """
    from app.agent.utils import auto_resolve_location
    from app.dal.weather_knowledge_dal import compare_with_seasonal
    from app.agent.dispatch import normalize_agg_keys, router_scope_var

    scope = router_scope_var.get(None)

    # Get current weather at appropriate level
    if not ward_id and not location_hint:
        from app.dal.weather_aggregate_dal import get_city_current_weather
        weather = get_city_current_weather()
        resolved_data = {"city_name": "Hà Nội"}
    else:
        resolved = auto_resolve_location(
            ward_id=ward_id, location_hint=location_hint,
            target_scope=scope,
        )
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

    from app.agent.tools.output_builder import build_seasonal_comparison_output
    return build_seasonal_comparison_output({
        "current": weather,
        "seasonal_avg": seasonal["seasonal_avg"],
        "comparisons": seasonal["comparisons"],
        "month_name": seasonal["month_name"],
        "resolved_location": resolved_data,
    })
