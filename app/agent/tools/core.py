"""Core tools — resolve_location, get_current_weather, get_weather_alerts."""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============== Tool 1: resolve_location ==============

class ResolveLocationInput(BaseModel):
    location_hint: str = Field(
        description="Tên phường/xã hoặc quận/huyện tại Hà Nội. Ví dụ: Cầu Giấy, Đống Đa"
    )


@tool(args_schema=ResolveLocationInput)
def resolve_location(location_hint: str) -> dict:
    """Tìm phường/xã hoặc quận/huyện từ tên gần đúng.

    DÙNG KHI: cần xác định chính xác ward_id trước khi gọi tool khác,
    hoặc khi user nhập tên địa điểm không chính xác.
    KHÔNG DÙNG KHI: các tool khác đã có tham số location_hint (tự resolve bên trong).
    Trả về: ward_id, ward_name_vi, district_name_vi hoặc thông báo lỗi.
    """
    from app.dal.location_dal import resolve_location as dal_resolve
    from app.agent.tools.output_builder import build_resolve_location_output
    return build_resolve_location_output(dal_resolve(location_hint))


# ============== Tool 2: get_current_weather ==============

class GetCurrentWeatherInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="ward_id (ví dụ: ID_00169)")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")


@tool(args_schema=GetCurrentWeatherInput)
def get_current_weather(ward_id: str = None, location_hint: str = None) -> dict:
    """SNAPSHOT thời tiết tại THỜI ĐIỂM HIỆN TẠI (1 mốc thời gian duy nhất).

    DÙNG KHI: user dùng từ khoá SNAPSHOT thật sự:
        "bây giờ", "hiện tại", "đang", "lúc này", "vừa xong".

    KHÔNG DÙNG KHI (để tránh "reaching for the hammer"):
        - "chiều / tối / đêm / sáng mai / X giờ tối nay" → dùng get_hourly_forecast.
        - "ngày mai / thứ X / cuối tuần / 3 ngày tới" → dùng get_daily_forecast.
        - "cả ngày chi tiết sáng/trưa/chiều/tối" → dùng get_daily_summary.
        - "hôm qua / ngày đã qua" → dùng get_weather_history.
        - "max/min/đỉnh/mạnh nhất/cao nhất cả ngày" → dùng get_daily_summary (superlative
          từ snapshot CHỈ là 1 mốc, KHÔNG phải max cả ngày).
        - "so hôm nay vs ngày mai" → gọi CẢ current + get_daily_forecast(tomorrow), KHÔNG
          1 mình current rồi tự ngoại suy.

    Returns: Flat VN dict, value = "[nhãn] [số] [đơn vị]":
    - "địa điểm", "thời điểm" (HH:MM Thứ X NGÀY/THÁNG/NĂM), "thời tiết chung"
    - "nhiệt độ", "độ ẩm", "cảm giác" (CHỈ ward)
    - "điểm sương", "xác suất mưa" (từ forecast lân cận vì snapshot không có pop)
    - "gió" (avg + giật + hướng), "mây", "UV", "áp suất"
    - "cường độ mưa hiện tại" (CHỈ khi đang mưa)
    - "cảm giác nóng"/"cảm giác lạnh" (conditional theo ngưỡng)
    - "tầm nhìn" (CHỈ khi <5km), "tóm tắt"
    - "gợi ý dùng output": nếu xuất hiện → ĐỌC + làm theo.
    """
    from app.agent.dispatch import resolve_and_dispatch
    from app.agent.utils import enrich_weather_response, enrich_district_response, enrich_city_response
    from app.dal.weather_dal import get_current_weather as dal_ward
    from app.dal.weather_aggregate_dal import (
        get_district_current_weather as dal_district,
        get_city_current_weather as dal_city,
    )

    from app.agent.tools.output_builder import build_current_output
    raw = resolve_and_dispatch(
        ward_id=ward_id,
        location_hint=location_hint,
        default_scope="city",
        ward_fn=dal_ward,
        district_fn=lambda district_id: enrich_district_response(dal_district(district_id)),
        city_fn=lambda: enrich_city_response(dal_city()),
        enrich_fn=enrich_weather_response,  # Only applied to ward result
        label="thời tiết hiện tại",
    )
    return build_current_output(raw)


# ============== Tool 3: get_weather_alerts ==============

class GetWeatherAlertsInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="ward_id (mặc định: tất cả)")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")


@tool(args_schema=GetWeatherAlertsInput)
def get_weather_alerts(ward_id: str = None, location_hint: str = None) -> dict:
    """Lấy CẢNH BÁO thời tiết nguy hiểm trong 24h tới.

    DÙNG KHI: "có cảnh báo gì không?", "thời tiết có nguy hiểm không?",
    "có giông bão không?", "có rét hại không?".
    Hỗ trợ: phường/xã, toàn Hà Nội. Mặc định: toàn thành phố.
    Trả về: danh sách cảnh báo (gió giật > 20m/s, rét hại < 13C, nắng nóng > 39C, giông).
    """
    from app.dal.alerts_dal import get_weather_alerts as dal_get_alerts

    # Resolve ward_id if location_hint provided
    actual_id = None
    if ward_id:
        actual_id = ward_id
    elif location_hint:
        from app.agent.utils import auto_resolve_location
        from app.agent.dispatch import router_scope_var
        resolved = auto_resolve_location(
            location_hint=location_hint,
            target_scope=router_scope_var.get(None),
        )
        if resolved["status"] == "ok" and resolved.get("level") == "ward":
            actual_id = resolved["data"].get("ward_id")

    from app.agent.tools.output_builder import build_weather_alerts_output
    alerts = dal_get_alerts(actual_id)
    return build_weather_alerts_output({"alerts": alerts, "count": len(alerts)})
