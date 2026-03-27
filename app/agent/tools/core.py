"""Core tools — resolve_location, get_current_weather, get_weather_alerts."""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============== Tool 1: resolve_location ==============

class ResolveLocationInput(BaseModel):
    location_hint: str = Field(
        description="Ten phuong/xa hoac quan/huyen tai Ha Noi. Vi du: Cau Giay, Dong Da"
    )


@tool(args_schema=ResolveLocationInput)
def resolve_location(location_hint: str) -> dict:
    """Tim phuong/xa hoac quan/huyen tu ten gan dung.

    DUNG KHI: can xac dinh chinh xac ward_id truoc khi goi tool khac,
    hoac khi user nhap ten dia diem khong chinh xac.
    KHONG DUNG KHI: cac tool khac da co tham so location_hint (tu resolve ben trong).
    Tra ve: ward_id, ward_name_vi, district_name_vi hoac thong bao loi.
    """
    from app.dal.location_dal import resolve_location as dal_resolve
    return dal_resolve(location_hint)


# ============== Tool 2: get_current_weather ==============

class GetCurrentWeatherInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="ward_id (vi du: ID_00169)")
    location_hint: Optional[str] = Field(default=None, description="Ten phuong/xa hoac quan/huyen")


@tool(args_schema=GetCurrentWeatherInput)
def get_current_weather(ward_id: str = None, location_hint: str = None) -> dict:
    """Lay thoi tiet HIEN TAI (real-time) cho phuong/xa, quan/huyen, hoac toan Ha Noi.

    DUNG KHI: user hoi "bay gio", "hien tai", "dang", "luc nay".
    Ho tro: phuong/xa, quan/huyen, toan Ha Noi (tu dong dispatch).
    KHONG DUNG KHI: hoi ve tuong lai (dung get_hourly_forecast),
    hoi ca ngay (dung get_daily_summary).
    Luu y: du lieu hien tai KHONG co pop (xac suat mua). Neu user hoi "co mua khong?",
    check weather_main + goi them get_hourly_forecast 1-2h.
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
        label="thoi tiet hien tai",
    )


# ============== Tool 3: get_weather_alerts ==============

class GetWeatherAlertsInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="ward_id (mac dinh: tat ca)")
    location_hint: Optional[str] = Field(default=None, description="Ten phuong/xa hoac quan/huyen")


@tool(args_schema=GetWeatherAlertsInput)
def get_weather_alerts(ward_id: str = None, location_hint: str = None) -> dict:
    """Lay CANH BAO thoi tiet nguy hiem trong 24h toi.

    DUNG KHI: "co canh bao gi khong?", "thoi tiet co nguy hiem khong?",
    "co giong bao khong?", "co ret hai khong?".
    Ho tro: phuong/xa, toan Ha Noi. Mac dinh: toan thanh pho.
    Tra ve: danh sach canh bao (gio giat > 20m/s, ret hai < 13C, nang nong > 39C, giong).
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
