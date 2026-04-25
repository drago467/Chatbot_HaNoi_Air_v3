"""
Weather Aggregate DAL - Queries for district and city level weather.
Provides fast aggregated weather data from pre-computed tables.

Schema mới (sau refactor star schema):
    - fact_weather_district_* dùng district_id FK (dim_district).
    - fact_weather_city_* dùng city_id FK (dim_city).
    - JOIN dim_district / dim_city để lấy tên hiển thị cho response.
    - Các function district nhận district_id: int (thay cho district_name: str cũ).
"""

from typing import List, Dict, Any
from app.dal.timezone_utils import format_ict
from app.db.dal import query, query_one
from app.dal.weather_helpers import wind_deg_to_vietnamese


# Shared column lists — không bao gồm district_name_vi/city_name_vi
# (được enrich qua JOIN dim_district / dim_city trong từng query).
_DISTRICT_HOURLY_COLS = """
    fwh.district_id, fwh.ts_utc,
    fwh.avg_temp, fwh.min_temp, fwh.max_temp,
    fwh.avg_humidity, fwh.avg_wind_speed,
    fwh.weather_main, fwh.ward_count,
    fwh.avg_dew_point, fwh.avg_pressure, fwh.avg_clouds, fwh.avg_visibility,
    fwh.avg_uvi, fwh.max_uvi, fwh.avg_pop, fwh.avg_rain_1h,
    fwh.avg_wind_deg, fwh.max_wind_gust
"""

_CITY_HOURLY_COLS = """
    fwh.city_id, fwh.ts_utc,
    fwh.avg_temp, fwh.min_temp, fwh.max_temp,
    fwh.avg_humidity, fwh.avg_wind_speed,
    fwh.weather_main, fwh.ward_count,
    fwh.avg_dew_point, fwh.avg_pressure, fwh.avg_clouds, fwh.avg_visibility,
    fwh.avg_uvi, fwh.max_uvi, fwh.avg_pop, fwh.avg_rain_1h,
    fwh.avg_wind_deg, fwh.max_wind_gust
"""

_DISTRICT_DAILY_COLS = """
    fwd.district_id, fwd.date,
    fwd.avg_temp, fwd.temp_min, fwd.temp_max,
    fwd.avg_humidity, fwd.avg_pop, fwd.total_rain,
    fwd.weather_main, fwd.ward_count,
    fwd.avg_dew_point, fwd.avg_pressure, fwd.avg_clouds,
    fwd.max_uvi, fwd.avg_wind_deg, fwd.max_wind_gust, fwd.avg_wind_speed
"""

_CITY_DAILY_COLS = """
    fwd.city_id, fwd.date,
    fwd.avg_temp, fwd.temp_min, fwd.temp_max,
    fwd.avg_humidity, fwd.avg_pop, fwd.total_rain,
    fwd.weather_main, fwd.ward_count,
    fwd.avg_dew_point, fwd.avg_pressure, fwd.avg_clouds,
    fwd.max_uvi, fwd.avg_wind_deg, fwd.max_wind_gust, fwd.avg_wind_speed
"""


def _add_wind_dir(row: Dict[str, Any]) -> None:
    """Add Vietnamese wind direction from avg_wind_deg."""
    deg = row.get("avg_wind_deg")
    if deg is not None:
        row["wind_direction_vi"] = wind_deg_to_vietnamese(deg)


# ═══════════════════════════════════════════════════════════════════
# DISTRICT-LEVEL QUERIES
# ═══════════════════════════════════════════════════════════════════

def get_district_current_weather(district_id: int) -> Dict[str, Any]:
    """Get current weather for a district from aggregated table.

    Bound ts_utc <= NOW() + 30m để tránh lấy forecast aggregate (nếu có)
    thay cho current observation — agent sẽ tưởng đó là "lúc này" và báo
    sai timestamp.
    """
    result = query_one(f"""
        SELECT {_DISTRICT_HOURLY_COLS},
               dd.district_name_vi,
               NOW() - fwh.ts_utc AS data_age
        FROM fact_weather_district_hourly fwh
        JOIN dim_district dd ON fwh.district_id = dd.district_id
        WHERE fwh.district_id = %s
          AND fwh.ts_utc > NOW() - INTERVAL '2 hours'
          AND fwh.ts_utc <= NOW() + INTERVAL '30 minutes'
        ORDER BY fwh.ts_utc DESC
        LIMIT 1
    """, (district_id,))

    if not result:
        result = query_one(f"""
            SELECT {_DISTRICT_HOURLY_COLS},
                   dd.district_name_vi,
                   NOW() - fwh.ts_utc AS data_age
            FROM fact_weather_district_hourly fwh
            JOIN dim_district dd ON fwh.district_id = dd.district_id
            WHERE fwh.district_id = %s
              AND fwh.ts_utc <= NOW() + INTERVAL '30 minutes'
            ORDER BY fwh.ts_utc DESC
            LIMIT 1
        """, (district_id,))
        if result:
            result["data_stale"] = True

    if not result:
        return {"error": "no_data", "message": f"Không có dữ liệu thời tiết cho district_id={district_id}"}

    data_age = result.pop('data_age', None)
    if data_age:
        result["data_age_minutes"] = int(data_age.total_seconds() / 60)
        result["data_age_hours"] = round(data_age.total_seconds() / 3600, 1)

    result["time_ict"] = format_ict(result.get("ts_utc"))
    result["level"] = "district"
    _add_wind_dir(result)
    return result


def get_district_hourly_forecast(district_id: int, hours: int = 24) -> List[Dict[str, Any]]:
    """Get hourly forecast for a district."""
    hours = min(hours, 48)
    results = query(f"""
        SELECT {_DISTRICT_HOURLY_COLS},
               dd.district_name_vi
        FROM fact_weather_district_hourly fwh
        JOIN dim_district dd ON fwh.district_id = dd.district_id
        WHERE fwh.district_id = %s
          AND fwh.ts_utc > NOW()
        ORDER BY fwh.ts_utc
        LIMIT %s
    """, (district_id, hours))

    for r in results:
        r["time_ict"] = format_ict(r.get("ts_utc"))
        r["level"] = "district"
        _add_wind_dir(r)
    return results


def get_district_daily_forecast(district_id: int, days: int = 7, start_date: str = None) -> List[Dict[str, Any]]:
    """Get daily forecast for a district."""
    days = min(days, 8)

    if start_date:
        date_filter = "fwd.date >= %s::date"
        params = (district_id, start_date, days)
    else:
        date_filter = "fwd.date >= (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh')::date"
        params = (district_id, days)

    results = query(f"""
        SELECT {_DISTRICT_DAILY_COLS},
               dd.district_name_vi
        FROM fact_weather_district_daily fwd
        JOIN dim_district dd ON fwd.district_id = dd.district_id
        WHERE fwd.district_id = %s
          AND {date_filter}
        ORDER BY fwd.date
        LIMIT %s
    """, params)

    for r in results:
        r["level"] = "district"
        _add_wind_dir(r)
    return results


# ═══════════════════════════════════════════════════════════════════
# CITY-LEVEL QUERIES
# ═══════════════════════════════════════════════════════════════════

def get_city_current_weather() -> Dict[str, Any]:
    """Get current weather for Hanoi city from aggregated table.

    Bound ts_utc <= NOW() + 30m để tránh lấy forecast aggregate.
    Hiện tại hệ thống chỉ hỗ trợ 1 thành phố (Hà Nội) nên không cần city_id param.
    """
    result = query_one(f"""
        SELECT {_CITY_HOURLY_COLS},
               dc.city_name_vi,
               NOW() - fwh.ts_utc AS data_age
        FROM fact_weather_city_hourly fwh
        JOIN dim_city dc ON fwh.city_id = dc.city_id
        WHERE fwh.ts_utc > NOW() - INTERVAL '2 hours'
          AND fwh.ts_utc <= NOW() + INTERVAL '30 minutes'
        ORDER BY fwh.ts_utc DESC
        LIMIT 1
    """)

    if not result:
        result = query_one(f"""
            SELECT {_CITY_HOURLY_COLS},
                   dc.city_name_vi,
                   NOW() - fwh.ts_utc AS data_age
            FROM fact_weather_city_hourly fwh
            JOIN dim_city dc ON fwh.city_id = dc.city_id
            WHERE fwh.ts_utc <= NOW() + INTERVAL '30 minutes'
            ORDER BY fwh.ts_utc DESC
            LIMIT 1
        """)
        if result:
            result["data_stale"] = True

    if not result:
        return {"error": "no_data", "message": "Không có dữ liệu thời tiết Hà Nội"}

    data_age = result.pop('data_age', None)
    if data_age:
        result["data_age_minutes"] = int(data_age.total_seconds() / 60)
        result["data_age_hours"] = round(data_age.total_seconds() / 3600, 1)

    result["time_ict"] = format_ict(result.get("ts_utc"))
    result["level"] = "city"
    result["city_name"] = result.get("city_name_vi", "Hà Nội")
    _add_wind_dir(result)
    return result


def get_city_hourly_forecast(hours: int = 24) -> List[Dict[str, Any]]:
    """Get hourly forecast for Hanoi city."""
    hours = min(hours, 48)
    results = query(f"""
        SELECT {_CITY_HOURLY_COLS},
               dc.city_name_vi
        FROM fact_weather_city_hourly fwh
        JOIN dim_city dc ON fwh.city_id = dc.city_id
        WHERE fwh.ts_utc > NOW()
        ORDER BY fwh.ts_utc
        LIMIT %s
    """, (hours,))

    for r in results:
        r["time_ict"] = format_ict(r.get("ts_utc"))
        r["level"] = "city"
        r["city_name"] = r.get("city_name_vi", "Hà Nội")
        _add_wind_dir(r)
    return results


def get_city_daily_forecast(days: int = 7, start_date: str = None) -> List[Dict[str, Any]]:
    """Get daily forecast for Hanoi city."""
    days = min(days, 8)

    if start_date:
        date_filter = "fwd.date >= %s::date"
        params = (start_date, days)
    else:
        date_filter = "fwd.date >= (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh')::date"
        params = (days,)

    results = query(f"""
        SELECT {_CITY_DAILY_COLS},
               dc.city_name_vi
        FROM fact_weather_city_daily fwd
        JOIN dim_city dc ON fwd.city_id = dc.city_id
        WHERE {date_filter}
        ORDER BY fwd.date
        LIMIT %s
    """, params)

    for r in results:
        r["level"] = "city"
        r["city_name"] = r.get("city_name_vi", "Hà Nội")
        _add_wind_dir(r)
    return results


# ═══════════════════════════════════════════════════════════════════
# DAILY SUMMARY / TREND / PERIOD QUERIES
# ═══════════════════════════════════════════════════════════════════

def get_district_daily_summary_data(district_id: int, date) -> Dict[str, Any]:
    """Get daily summary for a district on a specific date."""
    row = query_one(f"""
        SELECT {_DISTRICT_DAILY_COLS},
               dd.district_name_vi
        FROM fact_weather_district_daily fwd
        JOIN dim_district dd ON fwd.district_id = dd.district_id
        WHERE fwd.district_id = %s AND fwd.date = %s::date
    """, (district_id, str(date)))
    if not row:
        return {}
    row["level"] = "district"
    _add_wind_dir(row)
    return row


def get_city_daily_summary_data(date) -> Dict[str, Any]:
    """Get daily summary for Hanoi city on a specific date."""
    row = query_one(f"""
        SELECT {_CITY_DAILY_COLS},
               dc.city_name_vi
        FROM fact_weather_city_daily fwd
        JOIN dim_city dc ON fwd.city_id = dc.city_id
        WHERE fwd.date = %s::date
    """, (str(date),))
    if not row:
        return {}
    row["level"] = "city"
    row["city_name"] = row.get("city_name_vi", "Hà Nội")
    _add_wind_dir(row)
    return row


def get_district_temperature_trend_data(
    district_id: int, days: int = 7
) -> List[Dict[str, Any]]:
    """Get daily temperature data for trend analysis (district level)."""
    days = min(days, 8)
    return query("""
        SELECT date, temp_min, temp_max, avg_temp AS temp_avg, weather_main
        FROM fact_weather_district_daily
        WHERE district_id = %s
          AND date >= (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh')::date
        ORDER BY date
        LIMIT %s
    """, (district_id, days))


def get_city_temperature_trend_data(days: int = 7) -> List[Dict[str, Any]]:
    """Get daily temperature data for trend analysis (city level)."""
    days = min(days, 8)
    return query("""
        SELECT date, temp_min, temp_max, avg_temp AS temp_avg, weather_main
        FROM fact_weather_city_daily
        WHERE date >= (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh')::date
        ORDER BY date
        LIMIT %s
    """, (days,))


def get_district_weather_period_data(
    district_id: int, start_date: str, end_date: str
) -> List[Dict[str, Any]]:
    """Get daily weather data for a date range (district level)."""
    return query("""
        SELECT date, avg_temp, temp_min, temp_max, avg_humidity, avg_pop, total_rain,
               weather_main, avg_wind_speed, avg_wind_deg, max_uvi
        FROM fact_weather_district_daily
        WHERE district_id = %s AND date BETWEEN %s AND %s
        ORDER BY date
    """, (district_id, start_date, end_date))


def get_city_weather_period_data(
    start_date: str, end_date: str
) -> List[Dict[str, Any]]:
    """Get daily weather data for a date range (city level)."""
    return query("""
        SELECT date, avg_temp, temp_min, temp_max, avg_humidity, avg_pop, total_rain,
               weather_main, avg_wind_speed, avg_wind_deg, max_uvi
        FROM fact_weather_city_daily
        WHERE date BETWEEN %s AND %s
        ORDER BY date
    """, (start_date, end_date))


def get_all_districts_current_weather() -> List[Dict[str, Any]]:
    """Get current weather for ALL districts (for comparison/ranking)."""
    results = query(f"""
        SELECT {_DISTRICT_HOURLY_COLS},
               dd.district_name_vi
        FROM fact_weather_district_hourly fwh
        JOIN dim_district dd ON fwh.district_id = dd.district_id
        WHERE fwh.ts_utc = (
            SELECT MAX(ts_utc) FROM fact_weather_district_hourly
            WHERE ts_utc > NOW() - INTERVAL '2 hours'
        )
        ORDER BY fwh.avg_temp DESC
    """)

    for r in results:
        r["time_ict"] = format_ict(r.get("ts_utc"))
        r["level"] = "district"
        _add_wind_dir(r)
    return results


# ═══════════════════════════════════════════════════════════════════
# RANKING QUERIES
# ═══════════════════════════════════════════════════════════════════

_METRIC_MAP = {
    "nhiet_do": ("avg_temp", "C"),
    "do_am": ("avg_humidity", "%"),
    "gio": ("avg_wind_speed", "m/s"),
    "mua": ("avg_rain_1h", "mm"),
    "uvi": ("max_uvi", ""),
    "ap_suat": ("avg_pressure", "hPa"),
    "diem_suong": ("avg_dew_point", "C"),
    "may": ("avg_clouds", "%"),
}


def get_district_rankings(
    metric: str = "nhiet_do",
    order: str = "cao_nhat",
    limit: int = 5,
) -> Dict[str, Any]:
    """Rank districts by a weather metric.

    Args:
        metric: One of nhiet_do, do_am, gio, mua, uvi, ap_suat, diem_suong, may
        order: cao_nhat (DESC) or thap_nhat (ASC)
        limit: Number of results (1-30)
    """
    col_info = _METRIC_MAP.get(metric)
    if not col_info:
        return {"error": "invalid_metric", "valid": list(_METRIC_MAP.keys())}

    col, unit = col_info
    direction = "DESC" if order == "cao_nhat" else "ASC"
    limit = max(1, min(limit, 30))

    results = query(f"""
        SELECT dd.district_name_vi, fwh.{col} AS value, fwh.ward_count
        FROM fact_weather_district_hourly fwh
        JOIN dim_district dd ON fwh.district_id = dd.district_id
        WHERE fwh.ts_utc = (
            SELECT MAX(ts_utc) FROM fact_weather_district_hourly
            WHERE ts_utc > NOW() - INTERVAL '2 hours'
        )
          AND fwh.{col} IS NOT NULL
        ORDER BY fwh.{col} {direction}
        LIMIT %s
    """, (limit,))

    return {
        "metric": metric,
        "column": col,
        "unit": unit,
        "order": order,
        "rankings": [
            {"rank": i + 1, "district": r["district_name_vi"],
             "value": r["value"], "unit": unit}
            for i, r in enumerate(results)
        ],
    }


def get_ward_rankings_in_district(
    district_id: int,
    metric: str = "nhiet_do",
    order: str = "cao_nhat",
    limit: int = 10,
) -> Dict[str, Any]:
    """Rank wards within a district by a weather metric."""
    col_map = {
        "nhiet_do": ("temp", "C"),
        "do_am": ("humidity", "%"),
        "gio": ("wind_speed", "m/s"),
        "uvi": ("uvi", ""),
    }
    col_info = col_map.get(metric)
    if not col_info:
        return {"error": "invalid_metric", "valid": list(col_map.keys())}

    col, unit = col_info
    direction = "DESC" if order == "cao_nhat" else "ASC"
    limit = max(1, min(limit, 30))

    # Lấy district_name_vi để trả cho response
    dist_row = query_one(
        "SELECT district_name_vi FROM dim_district WHERE district_id = %s",
        (district_id,),
    )
    district_name = dist_row["district_name_vi"] if dist_row else f"district_id={district_id}"

    results = query(f"""
        SELECT d.ward_name_vi, w.{col} AS value
        FROM fact_weather_hourly w
        JOIN dim_ward d ON w.ward_id = d.ward_id
        WHERE d.district_id = %s
          AND w.data_kind = 'current'
          AND w.ts_utc > NOW() - INTERVAL '2 hours'
          AND w.{col} IS NOT NULL
        ORDER BY w.{col} {direction}
        LIMIT %s
    """, (district_id, limit))

    return {
        "district": district_name,
        "metric": metric,
        "unit": unit,
        "order": order,
        "rankings": [
            {"rank": i + 1, "ward": r["ward_name_vi"],
             "value": r["value"], "unit": unit}
            for i, r in enumerate(results)
        ],
    }
