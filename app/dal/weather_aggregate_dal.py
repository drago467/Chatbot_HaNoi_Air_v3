"""
Weather Aggregate DAL - Queries for district and city level weather.
Provides fast aggregated weather data from pre-computed tables.
"""

from typing import List, Dict, Any
from app.dal.timezone_utils import format_ict
from app.db.dal import query, query_one
from app.dal.weather_helpers import wind_deg_to_vietnamese


# Shared column lists to keep queries DRY
_DISTRICT_HOURLY_COLS = """
    district_name_vi, ts_utc,
    avg_temp, min_temp, max_temp,
    avg_humidity, avg_wind_speed,
    weather_main, ward_count,
    avg_dew_point, avg_pressure, avg_clouds, avg_visibility,
    avg_uvi, max_uvi, avg_pop, avg_rain_1h,
    avg_wind_deg, max_wind_gust
"""

_CITY_HOURLY_COLS = """
    ts_utc,
    avg_temp, min_temp, max_temp,
    avg_humidity, avg_wind_speed,
    weather_main, ward_count,
    avg_dew_point, avg_pressure, avg_clouds, avg_visibility,
    avg_uvi, max_uvi, avg_pop, avg_rain_1h,
    avg_wind_deg, max_wind_gust
"""

_DISTRICT_DAILY_COLS = """
    district_name_vi, date,
    avg_temp, temp_min, temp_max,
    avg_humidity, avg_pop, total_rain,
    weather_main, ward_count,
    avg_dew_point, avg_pressure, avg_clouds,
    max_uvi, avg_wind_deg, max_wind_gust
"""

_CITY_DAILY_COLS = """
    date,
    avg_temp, temp_min, temp_max,
    avg_humidity, avg_pop, total_rain,
    weather_main, ward_count,
    avg_dew_point, avg_pressure, avg_clouds,
    max_uvi, avg_wind_deg, max_wind_gust
"""
