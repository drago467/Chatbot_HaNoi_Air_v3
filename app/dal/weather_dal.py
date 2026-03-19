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
            result["data_warning"] = "Dữ liệu cũ, có thể không chính xác"

    if not result:
        return {"error": "no_data", "message": "Không có dữ liệu thời tiết hiện tại"}
    
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
          AND date >= CURRENT_DATE
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
          AND (ts_utc AT TIME ZONE 'Asia/Ho_Chi_Minh')::date = %s::date
    """, (ward_id, date))
    
    if not result:
        return {
            "error": "no_data", 
            "message": f"Không có dữ liệu thời tiết ngày {date}",
            "note": "Chỉ có dữ liệu lúc 12:00 ICT (noon)"
        }
    
    result["wind_direction_vi"] = wind_deg_to_vietnamese(result.get("wind_deg"))
    result["note"] = "Dữ liệu lúc 12:00 ICT (noon)"
    
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


def get_latest_weather_time(ward_id: str) -> List[Dict[str, Any]]:
    """Get the latest weather timestamp for a ward.
    
    Args:
        ward_id: Ward ID
        
    Returns:
        List of dictionaries with latest timestamp for each data_kind
    """
    return query("""
        SELECT MAX(ts_utc) as latest, 
               NOW() - MAX(ts_utc) as age,
               data_kind
        FROM fact_weather_hourly
        WHERE ward_id = %s
        GROUP BY data_kind
        ORDER BY data_kind
    """, (ward_id,))


def get_daily_summary_data(ward_id: str, query_date) -> Dict[str, Any]:
    """Get daily weather summary data for a ward.

    Args:
        ward_id: Ward ID
        query_date: Date object (date or datetime.date)

    Returns:
        Dictionary with daily weather data or error
    """
    row = query_one(
        "SELECT * FROM fact_weather_daily WHERE ward_id = %s AND date = %s",
        (ward_id, query_date)
    )

    if not row:
        return {"error": "no_data", "message": f"Không có dữ liệu ngày {query_date}"}

    # Temp range + bien do nhiet
    temp_min = row.get("temp_min")
    temp_max = row.get("temp_max")
    temp_range = temp_max - temp_min if temp_min is not None and temp_max is not None else 0
    bien_do_nhiet = f"Biên độ nhiệt {temp_range:.0f}°C" if temp_range > 0 else ""
    if temp_range > 10:
        bien_do_nhiet += " - Sáng lạnh, trưa nóng, nên mặc áo khoác"

    # Feels like gap
    feels_like_day = row.get("feels_like_day")
    temp_day = row.get("temp_day")
    feels_like_gap = (feels_like_day - temp_day) if feels_like_day is not None and temp_day is not None else 0

    # Rain assessment
    rain_total = row.get("rain_total") or 0
    if rain_total == 0:
        rain_assessment = "Không mưa"
    elif rain_total < 10:
        rain_assessment = f"Mưa nhẹ {rain_total:.1f}mm"
    elif rain_total < 25:
        rain_assessment = f"Mưa vừa {rain_total:.1f}mm"
    else:
        rain_assessment = f"Mưa to {rain_total:.1f}mm - Nên mang ô"

    # UV level
    uvi = row.get("uvi") or 0
    if uvi >= 11:
        uv_level = "Cực cao - Nguy hiểm"
    elif uvi >= 8:
        uv_level = "Rất cao - Hạn chế ra ngoài 10h-14h"
    elif uvi >= 6:
        uv_level = "Cao - Cần che nắng"
    elif uvi >= 3:
        uv_level = "Trung bình"
    else:
        uv_level = "Thấp"

    # Daylight hours
    daylight_hours = None
    if row.get("sunrise") and row.get("sunset"):
        try:
            sunrise = row["sunrise"]
            sunset = row["sunset"]
            if hasattr(sunrise, "replace"):
                sunrise = sunrise.replace(tzinfo=None)
                sunset = sunset.replace(tzinfo=None)
            daylight_hours = round((sunset - sunrise).total_seconds() / 3600, 1)
        except (TypeError, ValueError, AttributeError):
            pass

    # Wind direction
    wind_dir = wind_deg_to_vietnamese(row.get("wind_deg")) if row.get("wind_deg") is not None else None

    return {
        "date": str(query_date),
        "temp_range": {"min": temp_min, "max": temp_max, "bien_do": temp_range},
        "temp_progression": {
            "sang": row.get("temp_morn"),
            "trua": row.get("temp_day"),
            "chieu": row.get("temp_eve"),
            "toi": row.get("temp_night"),
        },
        "feels_like_gap": feels_like_gap,
        "humidity": row.get("humidity"),
        "dew_point": row.get("dew_point"),
        "pressure": row.get("pressure"),
        "rain_assessment": rain_assessment,
        "rain_total": rain_total,
        "pop": row.get("pop"),
        "uvi": uvi,
        "uv_level": uv_level,
        "daylight_hours": daylight_hours,
        "wind": {"speed": row.get("wind_speed"), "direction": wind_dir, "gust": row.get("wind_gust")},
        "clouds": row.get("clouds"),
        "weather_main": row.get("weather_main"),
        "weather_description": row.get("weather_description"),
        "sunrise": str(row.get("sunrise")) if row.get("sunrise") else None,
        "sunset": str(row.get("sunset")) if row.get("sunset") else None,
        "note": bien_do_nhiet,
    }


def get_rain_timeline(ward_id: str, hours: int = 24) -> Dict[str, Any]:
    """Scan hourly forecast to detect rain start/stop transitions.

    Returns rain periods, next rain time, next clear time.
    """
    rows = query("""
        SELECT ts_utc, pop, rain_1h, weather_main
        FROM fact_weather_hourly
        WHERE ward_id = %s
          AND data_kind = 'forecast'
          AND ts_utc > NOW()
        ORDER BY ts_utc
        LIMIT %s
    """, (ward_id, min(hours, 48)))

    if not rows:
        return {"error": "no_data", "message": "Không có dữ liệu dự báo"}

    # Detect rain periods (pop >= 0.3 or weather_main contains Rain/Drizzle/Thunderstorm)
    rain_keywords = {"Rain", "Drizzle", "Thunderstorm"}
    periods = []
    current_period = None

    for row in rows:
        pop = row.get("pop") or 0
        rain_1h = row.get("rain_1h") or 0
        wm = row.get("weather_main", "")
        is_rain = pop >= 0.3 or wm in rain_keywords or rain_1h > 0

        ts = row["ts_utc"]
        if is_rain and current_period is None:
            current_period = {"start": ts, "end": ts, "max_pop": pop, "max_rain_1h": rain_1h}
        elif is_rain and current_period is not None:
            current_period["end"] = ts
            current_period["max_pop"] = max(current_period["max_pop"], pop)
            current_period["max_rain_1h"] = max(current_period["max_rain_1h"], rain_1h)
        elif not is_rain and current_period is not None:
            periods.append(current_period)
            current_period = None

    if current_period is not None:
        periods.append(current_period)

    # Format periods
    from app.dal.timezone_utils import format_ict
    formatted = []
    for p in periods:
        formatted.append({
            "start": format_ict(p["start"]),
            "end": format_ict(p["end"]),
            "max_pop": round(p["max_pop"] * 100),
            "max_rain_1h": p["max_rain_1h"],
        })

    # Next rain / next clear
    first_rain = format_ict(periods[0]["start"]) if periods else None
    # Find first clear after first rain
    first_clear = None
    if periods:
        last_rain_end = periods[0]["end"]
        for row in rows:
            if row["ts_utc"] > last_rain_end:
                pop = row.get("pop") or 0
                wm = row.get("weather_main", "")
                if pop < 0.3 and wm not in rain_keywords:
                    first_clear = format_ict(row["ts_utc"])
                    break

    return {
        "rain_periods": formatted,
        "total_rain_periods": len(formatted),
        "next_rain": first_rain,
        "next_clear": first_clear,
        "hours_scanned": len(rows),
    }


def get_temperature_trend(ward_id: str, days: int = 7) -> Dict[str, Any]:
    """Analyze daily forecast to detect warming/cooling trend."""
    rows = query("""
        SELECT date, temp_min, temp_max, temp_avg, weather_main
        FROM fact_weather_daily
        WHERE ward_id = %s
          AND data_kind = 'forecast'
          AND date >= CURRENT_DATE
        ORDER BY date
        LIMIT %s
    """, (ward_id, min(days, 8)))

    if len(rows) < 2:
        return {"error": "no_data", "message": "Không đủ dữ liệu để phân tích xu hướng"}

    temps = [r["temp_avg"] for r in rows if r.get("temp_avg") is not None]
    if len(temps) < 2:
        return {"error": "no_data", "message": "Không đủ dữ liệu nhiệt độ"}

    # Simple linear trend: slope = (last - first) / n
    slope = (temps[-1] - temps[0]) / (len(temps) - 1)

    if slope > 0.5:
        trend = "warming"
        trend_vi = "Ấm dần lên"
    elif slope < -0.5:
        trend = "cooling"
        trend_vi = "Lạnh dần"
    else:
        trend = "stable"
        trend_vi = "Ổn định"

    # Find inflection point (day when trend reverses)
    inflection = None
    for i in range(1, len(temps) - 1):
        prev_diff = temps[i] - temps[i - 1]
        next_diff = temps[i + 1] - temps[i]
        if (prev_diff > 0 and next_diff < -0.5) or (prev_diff < 0 and next_diff > 0.5):
            inflection = str(rows[i]["date"])
            break

    # Find hottest and coldest days
    max_row = max(rows, key=lambda r: r.get("temp_max") or 0)
    min_row = min(rows, key=lambda r: r.get("temp_min") or 999)

    return {
        "trend": trend,
        "trend_vi": trend_vi,
        "slope_per_day": round(slope, 1),
        "days_analyzed": len(rows),
        "inflection_date": inflection,
        "hottest_day": {"date": str(max_row["date"]), "temp_max": max_row.get("temp_max")},
        "coldest_day": {"date": str(min_row["date"]), "temp_min": min_row.get("temp_min")},
        "daily_summary": [
            {"date": str(r["date"]), "min": r.get("temp_min"), "max": r.get("temp_max"),
             "avg": r.get("temp_avg"), "weather": r.get("weather_main")}
            for r in rows
        ],
    }
