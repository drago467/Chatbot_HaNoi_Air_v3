"""Weather DAL - Core weather queries for LLM Agent."""

from typing import List, Dict, Any, Optional
from app.dal.timezone_utils import format_ict, to_ict
from app.db.dal import query, query_one
from app.dal.weather_helpers import (
    wind_deg_to_vietnamese,
    wind_speed_to_beaufort,
    wind_beaufort_vietnamese,
    get_uv_status,
    get_dew_point_status,
    get_pressure_status,
    get_feels_like_status,
)


def get_current_weather(ward_id: str) -> Dict[str, Any]:
    """Get current weather for a ward.
    
    Args:
        ward_id: Ward ID (e.g., 'ID_XXXXX')
        
    Returns:
        Dictionary with current weather data or error
    """
    # Step 1: Try to get fresh data (within 2 hours)
    result = query_one("""
        SELECT temp, feels_like, humidity, pressure, dew_point, wind_speed, wind_deg,
               wind_gust, clouds, visibility, uvi, pop, rain_1h,
               weather_main, weather_description, ts_utc,
               NOW() - ts_utc AS data_age
        FROM fact_weather_hourly
        WHERE ward_id = %s AND data_kind = 'current'
          AND ts_utc > NOW() - INTERVAL '2 hours'
        ORDER BY ts_utc DESC
        LIMIT 1
    """, (ward_id,))

    # Step 2: Fallback to latest data if no fresh data
    if not result:
        result = query_one("""
            SELECT temp, feels_like, humidity, pressure, dew_point, wind_speed, wind_deg,
                   wind_gust, clouds, visibility, uvi, pop, rain_1h,
                   weather_main, weather_description, ts_utc,
                   NOW() - ts_utc AS data_age
            FROM fact_weather_hourly
            WHERE ward_id = %s AND data_kind = 'current'
            ORDER BY ts_utc DESC
            LIMIT 1
        """, (ward_id,))
        
        if result:
            # Mark as stale data
            result["data_stale"] = True
            result["data_warning"] = "Du lieu cu, co the khong chinh xac"
    
    if not result:
        return {"error": "no_data", "message": "Khong co du lieu thoi tiet hien tai"}
    
    # Add data age info
    data_age = result.pop('data_age', None)
    if data_age:
        result["data_age_minutes"] = int(data_age.total_seconds() / 60)
        result["data_age_hours"] = round(data_age.total_seconds() / 3600, 1)
    
    # Add Vietnamese context
    result["wind_direction_vi"] = wind_deg_to_vietnamese(result.get("wind_deg"))
    result["wind_beaufort"] = wind_speed_to_beaufort(result.get("wind_speed"))
    result["wind_beaufort_vi"] = wind_beaufort_vietnamese(result["wind_beaufort"])
    result["uv_status"] = get_uv_status(result.get("uvi"))
    result["dew_point_status"] = get_dew_point_status(result.get("dew_point"))
    result["pressure_status"] = get_pressure_status(result.get("pressure"))
    result["feels_like_status"] = get_feels_like_status(result.get("temp"), result.get("feels_like"))
    result["time_ict"] = format_ict(result.get("ts_utc"))
    
    return result


def get_hourly_forecast(ward_id: str, hours: int = 48) -> List[Dict[str, Any]]:
    """Get hourly forecast for a ward.
    
    Args:
        ward_id: Ward ID
        hours: Number of hours to forecast (max 48)
        
    Returns:
        List of hourly forecast data
    """
    hours = min(hours, 48)  # Max 48 hours
    
    results = query("""
        SELECT ts_utc, temp, feels_like, humidity, dew_point, pop, rain_1h,
               wind_speed, wind_deg, clouds, weather_main, weather_description
        FROM fact_weather_hourly
        WHERE ward_id = %s 
          AND data_kind = 'forecast' 
          AND ts_utc > NOW()
        ORDER BY ts_utc
        LIMIT %s
    """, (ward_id, hours))
    
    # Add Vietnamese wind direction and ICT time
    for r in results:
        r["wind_direction_vi"] = wind_deg_to_vietnamese(r.get("wind_deg"))
        r["wind_beaufort"] = wind_speed_to_beaufort(r.get("wind_speed"))
        r["time_ict"] = format_ict(r.get("ts_utc"))
    
    return results


def get_daily_forecast(ward_id: str, days: int = 8) -> List[Dict[str, Any]]:
    """Get daily forecast for a ward.
    
    Args:
        ward_id: Ward ID
        days: Number of days to forecast (max 8)
        
    Returns:
        List of daily forecast data
    """
    days = min(days, 8)  # Max 8 days
    
    results = query("""
        SELECT date, temp_min, temp_max, temp_avg, temp_morn, temp_eve, temp_night,
               humidity, pop, rain_total, uvi, weather_main, weather_description, 
               summary, sunrise, sunset
        FROM fact_weather_daily
        WHERE ward_id = %s 
          AND date > CURRENT_DATE
        ORDER BY date
        LIMIT %s
    """, (ward_id, days))
    
    # Add ICT times
    for r in results:
        if r.get('sunrise'):
            r['sunrise_ict'] = format_ict(r['sunrise'])
        if r.get('sunset'):
            r['sunset_ict'] = format_ict(r['sunset'])
    
    return results


def get_weather_history(ward_id: str, date: str) -> Dict[str, Any]:
    """Get weather for a specific date in the past.
    
    IMPORTANT: Queries fact_weather_hourly with data_kind='history'
    because we only have 1 record/day (noon) in hourly table.
    
    Args:
        ward_id: Ward ID
        date: Date in YYYY-MM-DD format
        
    Returns:
        Dictionary with historical weather data or error
    """
    result = query_one("""
        SELECT temp, feels_like, humidity, dew_point, wind_speed, wind_deg, 
               weather_main, weather_description, ts_utc
        FROM fact_weather_hourly
        WHERE ward_id = %s 
          AND data_kind = 'history'
          AND ts_utc::date = %s::date
    """, (ward_id, date))
    
    if not result:
        return {
            "error": "no_data", 
            "message": f"Khong co du lieu thoi tiet ngay {date}",
            "note": "Chi co du lieu luc 12:00 ICT (noon)"
        }
    
    result["wind_direction_vi"] = wind_deg_to_vietnamese(result.get("wind_deg"))
    result["note"] = "Du lieu luc 12:00 ICT (noon)"
    
    return result


def get_weather_range(ward_id: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Get weather data for a date range.
    
    Args:
        ward_id: Ward ID
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        List of daily weather data
    """
    results = query("""
        SELECT date, temp_min, temp_max, temp_avg, humidity, 
               rain_total, weather_main, weather_description
        FROM fact_weather_daily
        WHERE ward_id = %s 
          AND date BETWEEN %s AND %s
        ORDER BY date
    """, (ward_id, start_date, end_date))
    
    return results


def get_latest_weather_time(ward_id: str) -> Optional[Dict[str, Any]]:
    """Get the latest weather timestamp for a ward.
    
    Args:
        ward_id: Ward ID
        
    Returns:
        Dictionary with latest timestamp or None
    """
    return query_one("""
        SELECT MAX(ts_utc) as latest, 
               NOW() - MAX(ts_utc) as age,
               data_kind
        FROM fact_weather_hourly
        WHERE ward_id = %s
        GROUP BY data_kind
        ORDER BY data_kind
    """, (ward_id,))
