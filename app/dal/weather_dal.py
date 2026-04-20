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
    clean_chinese_weather_desc,
)


def get_current_weather(ward_id: str) -> Dict[str, Any]:
    """Get current weather for a ward.
    
    Args:
        ward_id: Ward ID (e.g., 'ID_XXXXX')
        
    Returns:
        Dictionary with current weather data or error
    """
    # Step 1: Try to get fresh data (within 2 hours, KHÔNG lấy future data)
    # Bound trên ts_utc <= NOW() để tránh bắt forecast bị mislabel 'current'.
    result = query_one("""
        SELECT temp, feels_like, humidity, pressure, dew_point, wind_speed, wind_deg,
               wind_gust, clouds, visibility, uvi, pop, rain_1h,
               weather_main, weather_description, ts_utc,
               NOW() - ts_utc AS data_age
        FROM fact_weather_hourly
        WHERE ward_id = %s AND data_kind = 'current'
          AND ts_utc > NOW() - INTERVAL '2 hours'
          AND ts_utc <= NOW() + INTERVAL '30 minutes'
        ORDER BY ts_utc DESC
        LIMIT 1
    """, (ward_id,))

    # Step 2: Fallback — dữ liệu cũ nhưng không quá tương lai
    if not result:
        result = query_one("""
            SELECT temp, feels_like, humidity, pressure, dew_point, wind_speed, wind_deg,
                   wind_gust, clouds, visibility, uvi, pop, rain_1h,
                   weather_main, weather_description, ts_utc,
                   NOW() - ts_utc AS data_age
            FROM fact_weather_hourly
            WHERE ward_id = %s AND data_kind = 'current'
              AND ts_utc <= NOW() + INTERVAL '30 minutes'
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
    # Clean Chinese weather descriptions → Vietnamese
    result["weather_description"] = clean_chinese_weather_desc(result.get("weather_description"))
    
    return result


def get_hourly_forecast(ward_id: str, hours: int = 48) -> List[Dict[str, Any]]:
    """Get hourly forecast for a ward.
    
    Args:
        ward_id: Ward ID
        hours: Number of hours to forecast (max 48)
        
    Returns:
        List of hourly forecast data
    """
    hours = max(1, min(hours, 48))  # Clamp 1-48
    
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
        r["weather_description"] = clean_chinese_weather_desc(r.get("weather_description"))
    
    return results


def get_daily_forecast(ward_id: str, days: int = 8, start_date: str = None) -> List[Dict[str, Any]]:
    """Get daily forecast for a ward.

    Args:
        ward_id: Ward ID
        days: Number of days to forecast (max 8)
        start_date: Start date (YYYY-MM-DD). Defaults to today.

    Returns:
        List of daily forecast data
    """
    days = max(1, min(days, 8))  # Clamp 1-8

    if start_date:
        date_filter = "date >= %s::date"
        params = (ward_id, start_date, days)
    else:
        date_filter = "date >= (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh')::date"
        params = (ward_id, days)

    results = query(f"""
        SELECT date, temp_min, temp_max, temp_avg, temp_morn, temp_eve, temp_night,
               humidity, pop, rain_total, uvi, weather_main, weather_description,
               summary, sunrise, sunset,
               wind_speed, wind_deg, wind_gust
        FROM fact_weather_daily
        WHERE ward_id = %s
          AND {date_filter}
        ORDER BY date
        LIMIT %s
    """, params)
    
    # Add ICT times + wind direction + day_of_week
    from app.dal.weather_helpers import wind_deg_to_vietnamese
    from datetime import datetime

    # Vietnamese day-of-week mapping
    _DAYS_VI = {
        "Monday": "Thứ Hai", "Tuesday": "Thứ Ba", "Wednesday": "Thứ Tư",
        "Thursday": "Thứ Năm", "Friday": "Thứ Sáu", "Saturday": "Thứ Bảy",
        "Sunday": "Chủ Nhật",
    }

    for r in results:
        # Day of week (giúp model không tính sai thứ)
        if r.get('date'):
            try:
                date_obj = datetime.strptime(str(r['date']), '%Y-%m-%d')
                r['day_of_week'] = _DAYS_VI.get(date_obj.strftime('%A'), date_obj.strftime('%A'))
            except (ValueError, TypeError):
                pass
        if r.get('sunrise'):
            r['sunrise_ict'] = format_ict(r['sunrise'])
            r['sunrise_time'] = format_ict(r['sunrise'], fmt="%H:%M")
        if r.get('sunset'):
            r['sunset_ict'] = format_ict(r['sunset'])
            r['sunset_time'] = format_ict(r['sunset'], fmt="%H:%M")
        if r.get('wind_deg') is not None:
            r['wind_direction_vi'] = wind_deg_to_vietnamese(r['wind_deg'])
        r['weather_description'] = clean_chinese_weather_desc(r.get('weather_description'))

    return results


def get_weather_history(ward_id: str, date: str) -> Dict[str, Any]:
    """Get weather for a specific date in the past.

    Queries fact_weather_hourly with data_kind='history' first.
    Falls back to fact_weather_daily if no hourly history available.

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
        ORDER BY ABS(EXTRACT(HOUR FROM ts_utc AT TIME ZONE 'Asia/Ho_Chi_Minh') - 12)
        LIMIT 1
    """, (ward_id, date))

    if result:
        result["wind_direction_vi"] = wind_deg_to_vietnamese(result.get("wind_deg"))
        result["weather_description"] = clean_chinese_weather_desc(result.get("weather_description"))
        result["note"] = "Dữ liệu lúc 12:00 trưa"

    # Bổ sung daily summary (temp_min/max, rain_total, sunrise/sunset)
    daily = query_one(
        "SELECT temp_min, temp_max, temp_avg, humidity AS daily_humidity, "
        "rain_total, uvi, weather_main AS daily_weather_main, "
        "weather_description AS daily_weather_desc, sunrise, sunset "
        "FROM fact_weather_daily WHERE ward_id = %s AND date = %s::date",
        (ward_id, date)
    )

    if result and daily:
        daily["daily_weather_desc"] = clean_chinese_weather_desc(daily.get("daily_weather_desc"))
        result["daily_summary"] = daily
        result["note"] = "Dữ liệu trưa 12:00 + tổng hợp ngày"
        return result

    if result:
        return result

    # Fallback: use daily data only when no hourly history
    if daily:
        return {
            "temp": daily.get("temp_avg"),
            "temp_min": daily.get("temp_min"),
            "temp_max": daily.get("temp_max"),
            "humidity": daily.get("daily_humidity"),
            "rain_total": daily.get("rain_total"),
            "uvi": daily.get("uvi"),
            "weather_main": daily.get("daily_weather_main"),
            "weather_description": clean_chinese_weather_desc(daily.get("daily_weather_desc")),
            "sunrise": daily.get("sunrise"),
            "sunset": daily.get("sunset"),
            "note": "Dữ liệu tổng hợp ngày (không có chi tiết theo giờ)",
            "source": "daily_summary"
        }

    return {
        "error": "no_data",
        "message": f"Không có dữ liệu thời tiết ngày {date}",
        "note": "Dữ liệu lịch sử chỉ lưu trữ 14 ngày gần nhất",
        "suggestion": "Thử hỏi ngày gần hơn hoặc dùng dự báo cho ngày tới"
    }


def get_city_weather_history(date: str) -> Dict[str, Any]:
    """Get historical weather from fact_weather_city_daily."""
    row = query_one("""
        SELECT avg_temp, temp_min, temp_max, avg_humidity, avg_pop, total_rain,
               weather_main, avg_dew_point, avg_pressure, avg_clouds,
               max_uvi, avg_wind_deg, max_wind_gust, ward_count
        FROM fact_weather_city_daily
        WHERE date = %s::date
    """, (date,))
    if not row:
        return {"error": "no_data",
                "message": f"Không có dữ liệu lịch sử thành phố ngày {date}",
                "note": "Dữ liệu lịch sử chỉ lưu trữ 14 ngày gần nhất",
                "suggestion": "Thử hỏi ngày gần hơn hoặc dùng dự báo cho ngày tới"}
    row["level"] = "city"
    row["date"] = date
    row["data_coverage"] = f"Dữ liệu ngày {date} (tổng hợp toàn Hà Nội)"
    row["wind_direction_vi"] = wind_deg_to_vietnamese(row.get("avg_wind_deg"))
    return row


def get_district_weather_history(district_name: str, date: str) -> Dict[str, Any]:
    """Get historical weather from fact_weather_district_daily."""
    row = query_one("""
        SELECT district_name_vi, avg_temp, temp_min, temp_max, avg_humidity, avg_pop, total_rain,
               weather_main, avg_dew_point, avg_pressure, avg_clouds,
               max_uvi, avg_wind_deg, max_wind_gust, ward_count
        FROM fact_weather_district_daily
        WHERE district_name_vi = %s AND date = %s::date
    """, (district_name, date))
    if not row:
        return {"error": "no_data",
                "message": f"Không có dữ liệu lịch sử quận {district_name} ngày {date}",
                "note": "Dữ liệu lịch sử chỉ lưu trữ 14 ngày gần nhất",
                "suggestion": "Thử hỏi ngày gần hơn hoặc dùng dự báo cho ngày tới"}
    row["level"] = "district"
    row["date"] = date
    row["data_coverage"] = f"Dữ liệu ngày {date} (quận/huyện {district_name})"
    row["wind_direction_vi"] = wind_deg_to_vietnamese(row.get("avg_wind_deg"))
    return row


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

    for r in results:
        r['weather_description'] = clean_chinese_weather_desc(r.get('weather_description'))

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
        "weather_description": clean_chinese_weather_desc(row.get("weather_description")),
        "sunrise": str(row.get("sunrise")) if row.get("sunrise") else None,
        "sunset": str(row.get("sunset")) if row.get("sunset") else None,
        "note": bien_do_nhiet,
    }


def analyze_rain_from_forecasts(rows: list, hours: int = 24, source_note: str = "") -> Dict[str, Any]:
    """Analyze rain periods from forecast data (reusable for ward/district/city).

    Supports both ward-level (pop, rain_1h) and aggregate (avg_pop, avg_rain_1h) keys.

    Args:
        rows: List of forecast dicts with ts_utc + pop/avg_pop + rain_1h/avg_rain_1h + weather_main
        hours: Number of hours scanned (for metadata)
        source_note: Optional note about data source level

    Returns:
        Dict with rain_periods, next_rain, next_clear, hours_scanned, data_coverage
    """
    if not rows:
        return {"error": "no_data", "message": "Không có dữ liệu dự báo",
                "suggestion": "Thử hỏi thời tiết hiện tại hoặc dự báo ngắn hạn"}

    from app.dal.timezone_utils import format_ict

    rain_keywords = {"Rain", "Drizzle", "Thunderstorm"}
    periods = []
    current_period = None

    for row in rows:
        # Support both ward-level keys (pop, rain_1h) and aggregate keys (avg_pop, avg_rain_1h)
        pop = row.get("pop") or row.get("avg_pop") or 0
        rain_1h = row.get("rain_1h") or row.get("avg_rain_1h") or 0
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
    formatted = []
    for p in periods:
        formatted.append({
            "start": format_ict(p["start"]),
            "end": format_ict(p["end"]),
            "max_pop": round(p["max_pop"] * 100),
            "max_rain_1h": round(p["max_rain_1h"], 1),
        })

    # Next rain / next clear
    first_rain = format_ict(periods[0]["start"]) if periods else None
    first_clear = None
    if periods:
        last_rain_end = periods[0]["end"]
        for row in rows:
            if row["ts_utc"] > last_rain_end:
                pop = row.get("pop") or row.get("avg_pop") or 0
                wm = row.get("weather_main", "")
                if pop < 0.3 and wm not in rain_keywords:
                    first_clear = format_ict(row["ts_utc"])
                    break

    result = {
        "rain_periods": formatted,
        "total_rain_periods": len(formatted),
        "next_rain": first_rain,
        "next_clear": first_clear,
        "hours_scanned": len(rows),
        "data_coverage": f"Dự báo {len(rows)} giờ tới",
    }
    if source_note:
        result["source_note"] = source_note
    return result


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

    return analyze_rain_from_forecasts(rows, hours)


def get_temperature_trend(ward_id: str, days: int = 7) -> Dict[str, Any]:
    """Analyze daily forecast to detect warming/cooling trend."""
    rows = query("""
        SELECT date, temp_min, temp_max, temp_avg, weather_main
        FROM fact_weather_daily
        WHERE ward_id = %s
          AND data_kind = 'forecast'
          AND date >= (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh')::date
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


def get_weather_period_data(ward_id: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Get daily weather data for a date range (used by get_weather_period tool).

    Args:
        ward_id: Ward ID
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        List of daily weather rows
    """
    return query(
        "SELECT date, temp_min, temp_max, temp_avg, humidity, pop, rain_total, "
        "uvi, wind_speed, wind_deg, wind_gust, weather_main "
        "FROM fact_weather_daily "
        "WHERE ward_id = %s AND date BETWEEN %s AND %s "
        "AND data_kind IN ('forecast', 'history') ORDER BY date",
        (ward_id, start_date, end_date)
    )


def detect_weather_changes(ward_id: str, hours: int = 6) -> Dict[str, Any]:
    """Detect significant weather changes in the next few hours.

    Compares current weather with forecast to find:
    - Temperature drop/rise > 5°C
    - Rain probability jump > 50%
    - Wind speed increase > 5 m/s
    - Weather condition change (clear → rain, etc.)

    Args:
        ward_id: Ward ID
        hours: Hours ahead to scan (default 6)

    Returns:
        Dictionary with detected changes and timing
    """
    current = get_current_weather(ward_id)
    if "error" in current:
        return current

    forecasts = get_hourly_forecast(ward_id, hours=min(hours, 12))
    if not forecasts:
        return {"error": "no_data", "message": "Không có dữ liệu dự báo"}

    changes = []
    cur_temp = current.get("temp")
    cur_pop = current.get("pop") or 0
    cur_wind = current.get("wind_speed") or 0
    cur_weather = current.get("weather_main", "")

    rain_keywords = {"Rain", "Drizzle", "Thunderstorm"}
    cur_is_rain = cur_weather in rain_keywords

    for f in forecasts:
        f_temp = f.get("temp")
        f_pop = f.get("pop") or 0
        f_wind = f.get("wind_speed") or 0
        f_weather = f.get("weather_main", "")
        f_time = format_ict(f.get("ts_utc"))
        f_is_rain = f_weather in rain_keywords

        # Temperature change > 5°C
        if cur_temp is not None and f_temp is not None:
            temp_diff = f_temp - cur_temp
            if abs(temp_diff) >= 5:
                direction = "tăng" if temp_diff > 0 else "giảm"
                changes.append({
                    "type": "temperature",
                    "description": f"Nhiệt độ {direction} {abs(temp_diff):.1f}°C ({cur_temp:.1f}→{f_temp:.1f}°C)",
                    "time": f_time,
                    "severity": "high" if abs(temp_diff) >= 8 else "medium"
                })
                break  # Only report first significant temp change

        # Rain probability jump
        if f_pop - cur_pop >= 0.5:
            changes.append({
                "type": "rain_start",
                "description": f"Khả năng mưa tăng mạnh ({cur_pop*100:.0f}%→{f_pop*100:.0f}%)",
                "time": f_time,
                "severity": "high" if f_pop >= 0.8 else "medium"
            })
            break

        # Weather condition change: clear → rain
        if not cur_is_rain and f_is_rain:
            changes.append({
                "type": "weather_change",
                "description": f"Trời chuyển mưa ({cur_weather}→{f_weather})",
                "time": f_time,
                "severity": "high" if f_weather == "Thunderstorm" else "medium"
            })
            break

        # Rain → clear
        if cur_is_rain and not f_is_rain and f_pop < 0.3:
            changes.append({
                "type": "rain_stop",
                "description": f"Mưa có thể tạnh",
                "time": f_time,
                "severity": "low"
            })
            break

        # Wind increase > 5 m/s
        if f_wind - cur_wind >= 5:
            changes.append({
                "type": "wind_increase",
                "description": f"Gió mạnh lên ({cur_wind:.1f}→{f_wind:.1f} m/s)",
                "time": f_time,
                "severity": "high" if f_wind >= 15 else "medium"
            })
            break

    return {
        "changes": changes,
        "has_significant_change": len(changes) > 0,
        "hours_scanned": len(forecasts),
        "current_summary": {
            "temp": cur_temp,
            "weather_main": cur_weather,
            "wind_speed": cur_wind,
            "pop": cur_pop
        }
    }
