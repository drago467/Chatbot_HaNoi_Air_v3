"""Forecast tools — hourly, daily, rain_timeline, best_time.

Tất cả đều hỗ trợ 3 tier (ward/district/city) nhất quán thông qua dispatch_forecast.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============== Tool: get_hourly_forecast ==============

class GetHourlyForecastInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID (ví dụ: ID_00169)")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")
    hours: int = Field(default=24, description="Số giờ dự báo (1-48)")


@tool(args_schema=GetHourlyForecastInput)
def get_hourly_forecast(ward_id: str = None, location_hint: str = None, hours: int = 24) -> dict:
    """Lấy dự báo thời tiết THEO GIỜ (1-48 giờ tới).

    DÙNG KHI: user hỏi về chiều nay, tối nay, sáng mai, vài giờ tới,
    mưa lúc mấy giờ, nhiệt độ tối nay, gió đêm nay, khoảng thời gian cụ thể.
    Hỗ trợ: phường/xã, quận/huyện, toàn Hà Nội (tự động dispatch).
    KHÔNG DÙNG KHI: hỏi cả ngày mai/tuần này (dùng get_daily_forecast),
    hỏi hiện tại (dùng get_current_weather), hỏi mưa đến bao giờ (dùng get_rain_timeline).
    """
    from app.agent.dispatch import dispatch_forecast
    from app.dal.weather_dal import get_hourly_forecast as dal_ward
    from app.dal.weather_aggregate_dal import (
        get_district_hourly_forecast as dal_district,
        get_city_hourly_forecast as dal_city,
    )

    hours = max(1, min(hours, 48))
    return dispatch_forecast(
        ward_id=ward_id,
        location_hint=location_hint,
        ward_fn=dal_ward,
        district_fn=dal_district,
        city_fn=dal_city,
        ward_args={"hours": hours},
        district_args={"hours": hours},
        city_args={"hours": hours},
        forecast_type="hourly",
        default_scope="city",
    )


# ============== Tool: get_daily_forecast ==============

class GetDailyForecastInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID (ví dụ: ID_00169)")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")
    days: int = Field(default=7, description="Số ngày dự báo (1-8)")


@tool(args_schema=GetDailyForecastInput)
def get_daily_forecast(ward_id: str = None, location_hint: str = None, days: int = 7) -> dict:
    """Lấy dự báo thời tiết THEO NGÀY (1-8 ngày tới).

    DÙNG KHI: user hỏi "ngày mai", "cuối tuần", "3 ngày tới".
    Hỗ trợ: phường/xã, quận/huyện, toàn Hà Nội (tự động dispatch).
    KHÔNG DÙNG KHI: hỏi theo giờ (dùng get_hourly_forecast).
    """
    from app.agent.dispatch import dispatch_forecast
    from app.dal.weather_dal import get_daily_forecast as dal_ward
    from app.dal.weather_aggregate_dal import (
        get_district_daily_forecast as dal_district,
        get_city_daily_forecast as dal_city,
    )

    days = max(1, min(days, 8))
    return dispatch_forecast(
        ward_id=ward_id,
        location_hint=location_hint,
        ward_fn=dal_ward,
        district_fn=dal_district,
        city_fn=dal_city,
        ward_args={"days": days},
        district_args={"days": days},
        city_args={"days": days},
        forecast_type="daily",
        default_scope="city",
    )


# ============== Tool: get_rain_timeline ==============

class GetRainTimelineInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")
    hours: int = Field(default=24, description="Số giờ scan (1-48)")


@tool(args_schema=GetRainTimelineInput)
def get_rain_timeline(ward_id: str = None, location_hint: str = None, hours: int = 24) -> dict:
    """Timeline mưa: khi nào bắt đầu mưa, khi nào tạnh, max lượng mưa.

    DÙNG KHI: "lúc nào mưa?", "mưa đến bao giờ?", "có mưa không?", "trời tạnh lúc nào?".
    Hỗ trợ: phường/xã, quận/huyện, toàn Hà Nội.
    Trả về: rain_periods (start/end/max_pop), next_rain, next_clear.
    """
    from app.agent.dispatch import dispatch_forecast
    from app.dal.weather_dal import get_hourly_forecast as dal_ward_hourly
    from app.dal.weather_aggregate_dal import (
        get_district_hourly_forecast as dal_district_hourly,
        get_city_hourly_forecast as dal_city_hourly,
    )
    from app.dal.weather_dal import analyze_rain_from_forecasts

    hours = max(1, min(hours, 48))

    # We need raw forecasts first, then analyze
    result = dispatch_forecast(
        ward_id=ward_id,
        location_hint=location_hint,
        ward_fn=dal_ward_hourly,
        district_fn=dal_district_hourly,
        city_fn=dal_city_hourly,
        ward_args={"hours": hours},
        district_args={"hours": hours},
        city_args={"hours": hours},
        forecast_type="hourly",
        default_scope="city",
    )

    if result.get("error"):
        return result

    forecasts = result.get("forecasts", [])
    rain_analysis = analyze_rain_from_forecasts(forecasts, hours)
    rain_analysis["resolved_location"] = result.get("resolved_location", {})
    rain_analysis["level"] = result.get("level", "city")
    return rain_analysis


# ============== Tool: get_best_time ==============

class GetBestTimeInput(BaseModel):
    activity: str = Field(description="Hoạt động: chay_bo, picnic, bike, chup_anh, du_lich, cam_trai, ...")
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")
    hours: int = Field(default=24, description="Số giờ quét (1-48)")


@tool(args_schema=GetBestTimeInput)
def get_best_time(activity: str, ward_id: str = None, location_hint: str = None, hours: int = 24) -> dict:
    """Tìm KHUNG GIỜ TỐT NHẤT để thực hiện hoạt động ngoài trời.

    DÙNG KHI: "mấy giờ chạy bộ tốt?", "lúc nào đi chơi đẹp nhất?",
    "giờ nào nên picnic?".
    Hỗ trợ: phường/xã, quận/huyện, toàn Hà Nội.
    Trả về: top 5 giờ tốt nhất với score, và 3 giờ tồi nhất.
    """
    from app.agent.dispatch import dispatch_forecast
    from app.dal.weather_dal import get_hourly_forecast as dal_ward_hourly
    from app.dal.weather_aggregate_dal import (
        get_district_hourly_forecast as dal_district_hourly,
        get_city_hourly_forecast as dal_city_hourly,
    )
    from app.dal.activity_dal import get_best_time_for_activity

    hours = max(1, min(hours, 48))

    # Get forecasts first
    result = dispatch_forecast(
        ward_id=ward_id,
        location_hint=location_hint,
        ward_fn=dal_ward_hourly,
        district_fn=dal_district_hourly,
        city_fn=dal_city_hourly,
        ward_args={"hours": hours},
        district_args={"hours": hours},
        city_args={"hours": hours},
        forecast_type="hourly",
        default_scope="city",
    )

    if result.get("error"):
        return result

    forecasts = result.get("forecasts", [])
    best = get_best_time_for_activity(activity, forecasts=forecasts, hours=hours)
    best["resolved_location"] = result.get("resolved_location", {})
    best["level"] = result.get("level", "city")
    return best
