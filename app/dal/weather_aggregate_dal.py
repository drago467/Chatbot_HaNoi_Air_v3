"""
Weather Aggregate DAL - Queries for district and city level weather.
Provides fast aggregated weather data from pre-computed tables.
"""

from typing import List, Dict, Any, Optional
from app.dal.timezone_utils import format_ict, to_ict
from app.db.dal import query, query_one
from app.dal.weather_helpers import (
    wind_deg_to_vietnamese,
    wind_speed_to_beaufort,
    wind_beaufort_vietnamese,
    get_uv_status,
)


def get_district_current_weather(district_name: str) -> Dict[str, Any]:
    """Get current weather for a district from aggregated table.
    
    Args:
        district_name: District name in Vietnamese (e.g., 'Quận Cầu Giấy', 'Huyện Ba Vì')
        
    Returns:
        Dictionary with current weather data or error
    """
    # Try to get fresh data (within 2 hours), fallback to latest
    result = query_one("""
        SELECT 
            district_name_vi,
            ts_utc,
            avg_temp,
            min_temp,
            max_temp,
            avg_humidity,
            avg_wind_speed,
            weather_main,
            ward_count,
            NOW() - ts_utc AS data_age
        FROM fact_weather_district_hourly
        WHERE district_name_vi = %s
          AND ts_utc > NOW() - INTERVAL '2 hours'
        ORDER BY ts_utc DESC
        LIMIT 1
    """, (district_name,))

    if not result:
        result = query_one("""
            SELECT 
                district_name_vi,
                ts_utc,
                avg_temp,
                min_temp,
                max_temp,
                avg_humidity,
                avg_wind_speed,
                weather_main,
                ward_count,
                NOW() - ts_utc AS data_age
            FROM fact_weather_district_hourly
            WHERE district_name_vi = %s
            ORDER BY ts_utc DESC
            LIMIT 1
        """, (district_name,))
        
        if result:
            result["data_stale"] = True
    
    if not result:
        return {"error": "no_data", "message": f"Khong co du lieu thoi tiet cho {district_name}"}
    
    # Add context
    data_age = result.pop('data_age', None)
    if data_age:
        result["data_age_minutes"] = int(data_age.total_seconds() / 60)
        result["data_age_hours"] = round(data_age.total_seconds() / 3600, 1)
    
    result["time_ict"] = format_ict(result.get("ts_utc"))
    result["level"] = "district"
    
    return result


def get_district_hourly_forecast(district_name: str, hours: int = 24) -> List[Dict[str, Any]]:
    """Get hourly forecast for a district."""
    hours = min(hours, 48)
    
    results = query("""
        SELECT 
            ts_utc,
            avg_temp,
            min_temp,
            max_temp,
            avg_humidity,
            avg_wind_speed,
            weather_main,
            ward_count
        FROM fact_weather_district_hourly
        WHERE district_name_vi = %s
          AND ts_utc > NOW()
        ORDER BY ts_utc
        LIMIT %s
    """, (district_name, hours))
    
    for r in results:
        r["time_ict"] = format_ict(r.get("ts_utc"))
        r["level"] = "district"
    
    return results


def get_district_daily_forecast(district_name: str, days: int = 7) -> List[Dict[str, Any]]:
    """Get daily forecast for a district."""
    days = min(days, 8)
    
    results = query("""
        SELECT 
            date,
            avg_temp,
            min_temp,
            max_temp,
            avg_humidity,
            avg_pop,
            total_rain,
            weather_main,
            ward_count
        FROM fact_weather_district_daily
        WHERE district_name_vi = %s
          AND date >= CURRENT_DATE
        ORDER BY date
        LIMIT %s
    """, (district_name, days))
    
    for r in results:
        r["level"] = "district"
    
    return results


def get_city_current_weather() -> Dict[str, Any]:
    """Get current weather for Hanoi city from aggregated table."""
    result = query_one("""
        SELECT 
            ts_utc,
            avg_temp,
            min_temp,
            max_temp,
            avg_humidity,
            avg_wind_speed,
            weather_main,
            ward_count,
            NOW() - ts_utc AS data_age
        FROM fact_weather_city_hourly
        WHERE ts_utc > NOW() - INTERVAL '2 hours'
        ORDER BY ts_utc DESC
        LIMIT 1
    """)

    if not result:
        result = query_one("""
            SELECT 
                ts_utc,
                avg_temp,
                min_temp,
                max_temp,
                avg_humidity,
                avg_wind_speed,
                weather_main,
                ward_count,
                NOW() - ts_utc AS data_age
            FROM fact_weather_city_hourly
            ORDER BY ts_utc DESC
            LIMIT 1
        """)
        
        if result:
            result["data_stale"] = True
    
    if not result:
        return {"error": "no_data", "message": "Khong co du lieu thoi tiet Ha Noi"}
    
    data_age = result.pop('data_age', None)
    if data_age:
        result["data_age_minutes"] = int(data_age.total_seconds() / 60)
        result["data_age_hours"] = round(data_age.total_seconds() / 3600, 1)
    
    result["time_ict"] = format_ict(result.get("ts_utc"))
    result["level"] = "city"
    result["city_name"] = "Ha Noi"
    
    return result


def get_city_hourly_forecast(hours: int = 24) -> List[Dict[str, Any]]:
    """Get hourly forecast for Hanoi city."""
    hours = min(hours, 48)
    
    results = query("""
        SELECT 
            ts_utc,
            avg_temp,
            min_temp,
            max_temp,
            avg_humidity,
            avg_wind_speed,
            weather_main,
            ward_count
        FROM fact_weather_city_hourly
        WHERE ts_utc > NOW()
        ORDER BY ts_utc
        LIMIT %s
    """, (hours,))
    
    for r in results:
        r["time_ict"] = format_ict(r.get("ts_utc"))
        r["level"] = "city"
        r["city_name"] = "Ha Noi"
    
    return results


def get_city_daily_forecast(days: int = 7) -> List[Dict[str, Any]]:
    """Get daily forecast for Hanoi city."""
    days = min(days, 8)
    
    results = query("""
        SELECT 
            date,
            avg_temp,
            min_temp,
            max_temp,
            avg_humidity,
            avg_pop,
            total_rain,
            weather_main,
            ward_count
        FROM fact_weather_city_daily
        WHERE date >= CURRENT_DATE
        ORDER BY date
        LIMIT %s
    """, (days,))
    
    for r in results:
        r["level"] = "city"
        r["city_name"] = "Ha Noi"
    
    return results


def get_all_districts_current_weather() -> List[Dict[str, Any]]:
    """Get current weather for ALL districts (for comparison/ranking)."""
    results = query("""
        SELECT 
            district_name_vi,
            ts_utc,
            avg_temp,
            min_temp,
            max_temp,
            avg_humidity,
            avg_wind_speed,
            weather_main,
            ward_count
        FROM fact_weather_district_hourly
        WHERE ts_utc > NOW() - INTERVAL '2 hours'
        ORDER BY avg_temp DESC
    """)
    
    for r in results:
        r["time_ict"] = format_ict(r.get("ts_utc"))
        r["level"] = "district"
    
    return results
