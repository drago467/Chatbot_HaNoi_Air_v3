"""Forecast tools — hourly, daily, rain_timeline, best_time.

Tat ca deu ho tro 3 tier (ward/district/city) nhat quan thong qua dispatch_forecast.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============== Tool: get_hourly_forecast ==============

class GetHourlyForecastInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID (vi du: ID_00169)")
    location_hint: Optional[str] = Field(default=None, description="Ten phuong/xa hoac quan/huyen")
    hours: int = Field(default=24, description="So gio du bao (1-48)")


@tool(args_schema=GetHourlyForecastInput)
def get_hourly_forecast(ward_id: str = None, location_hint: str = None, hours: int = 24) -> dict:
    """Lay du bao thoi tiet THEO GIO (1-48 gio toi).

    DUNG KHI: user hoi ve chieu nay, toi nay, sang mai, vai gio toi,
    mua luc may gio, nhiet do toi nay, gio dem nay, khoang thoi gian cu the.
    Ho tro: phuong/xa, quan/huyen, toan Ha Noi (tu dong dispatch).
    KHONG DUNG KHI: hoi ca ngay mai/tuan nay (dung get_daily_forecast),
    hoi hien tai (dung get_current_weather), hoi mua den bao gio (dung get_rain_timeline).
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
    ward_id: Optional[str] = Field(default=None, description="Ward ID (vi du: ID_00169)")
    location_hint: Optional[str] = Field(default=None, description="Ten phuong/xa hoac quan/huyen")
    days: int = Field(default=7, description="So ngay du bao (1-8)")


@tool(args_schema=GetDailyForecastInput)
def get_daily_forecast(ward_id: str = None, location_hint: str = None, days: int = 7) -> dict:
    """Lay du bao thoi tiet THEO NGAY (1-8 ngay toi).

    DUNG KHI: user hoi "ngay mai", "cuoi tuan", "3 ngay toi".
    Ho tro: phuong/xa, quan/huyen, toan Ha Noi (tu dong dispatch).
    KHONG DUNG KHI: hoi theo gio (dung get_hourly_forecast).
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
    location_hint: Optional[str] = Field(default=None, description="Ten phuong/xa hoac quan/huyen")
    hours: int = Field(default=24, description="So gio scan (1-48)")


@tool(args_schema=GetRainTimelineInput)
def get_rain_timeline(ward_id: str = None, location_hint: str = None, hours: int = 24) -> dict:
    """Timeline mua: khi nao bat dau mua, khi nao tanh, max luong mua.

    DUNG KHI: "luc nao mua?", "mua den bao gio?", "co mua khong?", "troi tanh luc nao?".
    Ho tro: phuong/xa, quan/huyen, toan Ha Noi.
    Tra ve: rain_periods (start/end/max_pop), next_rain, next_clear.
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
    activity: str = Field(description="Hoat dong: chay_bo, picnic, bike, chup_anh, du_lich, cam_trai, ...")
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Ten phuong/xa hoac quan/huyen")
    hours: int = Field(default=24, description="So gio quet (1-48)")


@tool(args_schema=GetBestTimeInput)
def get_best_time(activity: str, ward_id: str = None, location_hint: str = None, hours: int = 24) -> dict:
    """Tim KHUNG GIO TOT NHAT de thuc hien hoat dong ngoai troi.

    DUNG KHI: "may gio chay bo tot?", "luc nao di choi dep nhat?",
    "gio nao nen picnic?".
    Ho tro: phuong/xa, quan/huyen, toan Ha Noi.
    Tra ve: top 5 gio tot nhat voi score, va 3 gio toi nhat.
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
