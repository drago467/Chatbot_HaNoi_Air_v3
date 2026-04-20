"""Weather + location lookup endpoints.

Thay thế các truy vấn DB trực tiếp từ Streamlit (components.py::_fetch_weather,
get_districts, ...) bằng API call đi qua FastAPI + Redis cache.
"""

from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.schemas import ForecastPoint, WeatherCurrent
from app.core.cache import cache_get_or_fetch
from app.core.logging_config import get_logger
from app.db.dal import query

logger = get_logger(__name__)
router = APIRouter(tags=["weather"])

# TTL cache
_WEATHER_TTL = 300       # 5 phút
_LOCATION_TTL = 3600     # 1 giờ (districts / wards thay đổi hiếm)


@router.get("/locations/districts", response_model=list[str])
def list_districts():
    """Danh sách tất cả quận/huyện Hà Nội."""
    def _fetch() -> list[str]:
        rows = query("""
            SELECT DISTINCT district_name_vi
            FROM dim_ward
            ORDER BY district_name_vi
        """)
        return [r["district_name_vi"] for r in rows if r.get("district_name_vi")]

    return cache_get_or_fetch("locations:districts", _LOCATION_TTL, _fetch)


@router.get("/locations/wards/{district}", response_model=dict[str, str])
def list_wards(district: str):
    """Danh sách phường/xã thuộc 1 quận. Trả dict {ward_name: ward_id}."""
    key = f"locations:wards:{district}"

    def _fetch() -> dict[str, str]:
        rows = query("""
            SELECT ward_id, ward_name_vi
            FROM dim_ward
            WHERE district_name_vi = %s
            ORDER BY ward_name_vi
        """, (district,))
        return {r["ward_name_vi"]: r["ward_id"] for r in rows if r.get("ward_name_vi")}

    return cache_get_or_fetch(key, _LOCATION_TTL, _fetch)


@router.get("/weather/current/{ward_id}", response_model=WeatherCurrent)
def get_current_weather(ward_id: str):
    """Thời tiết hiện tại của 1 phường (bản ghi mới nhất từ fact_weather_hourly)."""
    key = f"weather:current:{ward_id}"

    def _fetch() -> dict[str, Any]:
        rows = query("""
            SELECT temp, humidity, weather_main, wind_speed, wind_deg
            FROM fact_weather_hourly
            WHERE ward_id = %s
            ORDER BY ts_utc DESC
            LIMIT 1
        """, (ward_id,))
        if not rows:
            return {}
        return rows[0]

    data = cache_get_or_fetch(key, _WEATHER_TTL, _fetch)
    if not data:
        raise HTTPException(status_code=404, detail="No weather data for ward")

    return WeatherCurrent(
        ward_id=ward_id,
        temp=data.get("temp"),
        humidity=data.get("humidity"),
        weather_main=data.get("weather_main"),
        wind_speed=data.get("wind_speed"),
        wind_deg=data.get("wind_deg"),
    )


@router.get("/weather/forecast/{ward_id}", response_model=list[ForecastPoint])
def get_hourly_forecast(ward_id: str, hours: int = 24):
    """Dự báo theo giờ (mặc định 24h tới)."""
    hours = max(1, min(hours, 48))
    key = f"weather:forecast:{ward_id}:{hours}"

    def _fetch() -> list[dict[str, Any]]:
        rows = query("""
            SELECT ts_utc AT TIME ZONE 'Asia/Ho_Chi_Minh' AS time_local,
                   temp, humidity
            FROM fact_weather_hourly
            WHERE ward_id = %s
              AND data_kind = 'forecast'
              AND ts_utc > NOW()
            ORDER BY ts_utc
            LIMIT %s
        """, (ward_id, hours))
        return rows

    data = cache_get_or_fetch(key, _WEATHER_TTL, _fetch) or []
    return [
        ForecastPoint(
            time_local=row["time_local"],
            temp=row.get("temp"),
            humidity=row.get("humidity"),
        )
        for row in data
    ]
