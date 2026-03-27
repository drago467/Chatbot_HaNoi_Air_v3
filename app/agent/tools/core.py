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
    return dal_resolve(location_hint)


# ============== Tool 2: get_current_weather ==============

class GetCurrentWeatherInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="ward_id (ví dụ: ID_00169)")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")


@tool(args_schema=GetCurrentWeatherInput)
def get_current_weather(ward_id: str = None, location_hint: str = None) -> dict:
    """Lấy thời tiết HIỆN TẠI (real-time) cho phường/xã, quận/huyện, hoặc toàn Hà Nội.

    DÙNG KHI: user hỏi "bây giờ", "hiện tại", "đang", "lúc này".
    Hỗ trợ: phường/xã, quận/huyện, toàn Hà Nội (tự động dispatch).
    KHÔNG DÙNG KHI: hỏi về tương lai (dùng get_hourly_forecast),
    hỏi cả ngày (dùng get_daily_summary).
    Lưu ý: dữ liệu hiện tại KHÔNG có pop (xác suất mưa). Nếu user hỏi "có mưa không?",
    check weather_main + gọi thêm get_hourly_forecast 1-2h.
    """
    from app.agent.dispatch import resolve_and_dispatch
    from app.agent.utils import enrich_weather_response, enrich_district_response, enrich_city_response
    from app.dal.weather_dal import get_current_weather as dal_ward
    from app.dal.weather_aggregate_dal import (
        get_district_current_weather as dal_district,
        get_city_current_weather as dal_city,
    )

    return resolve_and_dispatch(
        ward_id=ward_id,
        location_hint=location_hint,
        default_scope="city",
        ward_fn=dal_ward,
        district_fn=lambda district_name: enrich_district_response(dal_district(district_name)),
        city_fn=lambda: enrich_city_response(dal_city()),
        enrich_fn=enrich_weather_response,  # Only applied to ward result
        label="thời tiết hiện tại",
    )


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
        resolved = auto_resolve_location(location_hint=location_hint)
        if resolved["status"] == "ok" and resolved.get("level") == "ward":
            actual_id = resolved["data"].get("ward_id")

    alerts = dal_get_alerts(actual_id)
    return {"alerts": alerts, "count": len(alerts)}
