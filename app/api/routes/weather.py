"""Weather + location lookup endpoints.

Streamlit gọi qua FastAPI thay vì truy vấn DB trực tiếp. Sau R15 gỡ Redis,
query chạy thẳng DB (đủ nhanh cho scope single-user).
"""

from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.schemas import ForecastPoint, WeatherCurrent
from app.core.logging_config import get_logger
from app.db.dal import query

logger = get_logger(__name__)
router = APIRouter(tags=["weather"])


@router.get(
    "/locations/districts",
    response_model=list[str],
    summary="Danh sách quận/huyện Hà Nội",
    description="Trả 30+ tên quận/huyện (sau sắp xếp hành chính 2025) từ bảng `dim_district`.",
)
def list_districts():
    """Danh sách tất cả quận/huyện Hà Nội."""
    rows = query("""
        SELECT district_name_vi
        FROM dim_district
        ORDER BY district_name_vi
    """)
    return [r["district_name_vi"] for r in rows if r.get("district_name_vi")]


@router.get(
    "/locations/wards/{district}",
    response_model=dict[str, str],
    summary="Danh sách phường/xã của quận",
    description=(
        "Trả dict `{ward_name: ward_id}` cho mọi phường/xã thuộc quận/huyện "
        "được chỉ định. Tổng 126 phường sau sắp xếp 2025."
    ),
)
def list_wards(district: str):
    """Danh sách phường/xã thuộc 1 quận."""
    rows = query("""
        SELECT ward_id, ward_name_vi
        FROM dim_ward
        WHERE district_name_vi = %s
        ORDER BY ward_name_vi
    """, (district,))
    return {r["ward_name_vi"]: r["ward_id"] for r in rows if r.get("ward_name_vi")}


@router.get(
    "/weather/current/{ward_id}",
    response_model=WeatherCurrent,
    summary="Thời tiết hiện tại 1 phường",
    description="Bản ghi mới nhất từ `fact_weather_hourly` (data_kind='observation').",
    responses={
        404: {"description": "Không có data weather cho ward_id (chưa ingest?)"},
    },
)
def get_current_weather(ward_id: str):
    """Thời tiết hiện tại của 1 phường."""
    rows = query("""
        SELECT temp, humidity, weather_main, wind_speed, wind_deg
        FROM fact_weather_hourly
        WHERE ward_id = %s
        ORDER BY ts_utc DESC
        LIMIT 1
    """, (ward_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="No weather data for ward")

    data: dict[str, Any] = rows[0]
    return WeatherCurrent(
        ward_id=ward_id,
        temp=data.get("temp"),
        humidity=data.get("humidity"),
        weather_main=data.get("weather_main"),
        wind_speed=data.get("wind_speed"),
        wind_deg=data.get("wind_deg"),
    )


@router.get(
    "/weather/forecast/{ward_id}",
    response_model=list[ForecastPoint],
    summary="Dự báo theo giờ (tối đa 48h)",
    description=(
        "Trả list `ForecastPoint` theo thứ tự thời gian tăng dần. `hours` "
        "clamp trong [1, 48] — OpenWeather API chỉ cover 48h hourly."
    ),
)
def get_hourly_forecast(ward_id: str, hours: int = 24):
    """Dự báo theo giờ (mặc định 24h tới)."""
    hours = max(1, min(hours, 48))
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
    return [
        ForecastPoint(
            time_local=row["time_local"],
            temp=row.get("temp"),
            humidity=row.get("humidity"),
        )
        for row in rows
    ]
