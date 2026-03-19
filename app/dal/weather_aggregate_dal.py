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


def _add_wind_dir(row: Dict[str, Any]) -> None:
    """Add Vietnamese wind direction from avg_wind_deg."""
    deg = row.get("avg_wind_deg")
    if deg is not None:
        row["wind_direction_vi"] = wind_deg_to_vietnamese(deg)


def get_district_current_weather(district_name: str) -> Dict[str, Any]:
    """Get current weather for a district from aggregated table."""
    result = query_one(f"""
        SELECT {_DISTRICT_HOURLY_COLS},
               NOW() - ts_utc AS data_age
        FROM fact_weather_district_hourly
        WHERE district_name_vi = %s
          AND ts_utc > NOW() - INTERVAL '2 hours'
        ORDER BY ts_utc DESC
        LIMIT 1
    """, (district_name,))

    if not result:
        result = query_one(f"""
            SELECT {_DISTRICT_HOURLY_COLS},
                   NOW() - ts_utc AS data_age
            FROM fact_weather_district_hourly
            WHERE district_name_vi = %s
            ORDER BY ts_utc DESC
            LIMIT 1
        """, (district_name,))
        if result:
            result["data_stale"] = True

    if not result:
        return {"error": "no_data", "message": f"Không có dữ liệu thời tiết cho {district_name}"}

    data_age = result.pop('data_age', None)
    if data_age:
        result["data_age_minutes"] = int(data_age.total_seconds() / 60)
        result["data_age_hours"] = round(data_age.total_seconds() / 3600, 1)

    result["time_ict"] = format_ict(result.get("ts_utc"))
    result["level"] = "district"
    _add_wind_dir(result)
    return result


def get_district_hourly_forecast(district_name: str, hours: int = 24) -> List[Dict[str, Any]]:
    """Get hourly forecast for a district."""
    hours = min(hours, 48)
    results = query(f"""
        SELECT {_DISTRICT_HOURLY_COLS}
        FROM fact_weather_district_hourly
        WHERE district_name_vi = %s
          AND ts_utc > NOW()
        ORDER BY ts_utc
        LIMIT %s
    """, (district_name, hours))

    for r in results:
        r["time_ict"] = format_ict(r.get("ts_utc"))
        r["level"] = "district"
        _add_wind_dir(r)
    return results


def get_district_daily_forecast(district_name: str, days: int = 7) -> List[Dict[str, Any]]:
    """Get daily forecast for a district."""
    days = min(days, 8)
    results = query(f"""
        SELECT {_DISTRICT_DAILY_COLS}
        FROM fact_weather_district_daily
        WHERE district_name_vi = %s
          AND date >= CURRENT_DATE
        ORDER BY date
        LIMIT %s
    """, (district_name, days))

    for r in results:
        r["level"] = "district"
        _add_wind_dir(r)
    return results


def get_city_current_weather() -> Dict[str, Any]:
    """Get current weather for Hanoi city from aggregated table."""
    result = query_one(f"""
        SELECT {_CITY_HOURLY_COLS},
               NOW() - ts_utc AS data_age
        FROM fact_weather_city_hourly
        WHERE ts_utc > NOW() - INTERVAL '2 hours'
        ORDER BY ts_utc DESC
        LIMIT 1
    """)

    if not result:
        result = query_one(f"""
            SELECT {_CITY_HOURLY_COLS},
                   NOW() - ts_utc AS data_age
            FROM fact_weather_city_hourly
            ORDER BY ts_utc DESC
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
    result["city_name"] = "Hà Nội"
    _add_wind_dir(result)
    return result


def get_city_hourly_forecast(hours: int = 24) -> List[Dict[str, Any]]:
    """Get hourly forecast for Hanoi city."""
    hours = min(hours, 48)
    results = query(f"""
        SELECT {_CITY_HOURLY_COLS}
        FROM fact_weather_city_hourly
        WHERE ts_utc > NOW()
        ORDER BY ts_utc
        LIMIT %s
    """, (hours,))

    for r in results:
        r["time_ict"] = format_ict(r.get("ts_utc"))
        r["level"] = "city"
        r["city_name"] = "Hà Nội"
        _add_wind_dir(r)
    return results


def get_city_daily_forecast(days: int = 7) -> List[Dict[str, Any]]:
    """Get daily forecast for Hanoi city."""
    days = min(days, 8)
    results = query(f"""
        SELECT {_CITY_DAILY_COLS}
        FROM fact_weather_city_daily
        WHERE date >= CURRENT_DATE
        ORDER BY date
        LIMIT %s
    """, (days,))

    for r in results:
        r["level"] = "city"
        r["city_name"] = "Hà Nội"
        _add_wind_dir(r)
    return results


def get_all_districts_current_weather() -> List[Dict[str, Any]]:
    """Get current weather for ALL districts (for comparison/ranking)."""
    results = query(f"""
        SELECT {_DISTRICT_HOURLY_COLS}
        FROM fact_weather_district_hourly
        WHERE ts_utc = (
            SELECT MAX(ts_utc) FROM fact_weather_district_hourly
            WHERE ts_utc > NOW() - INTERVAL '2 hours'
        )
        ORDER BY avg_temp DESC
    """)

    for r in results:
        r["time_ict"] = format_ict(r.get("ts_utc"))
        r["level"] = "district"
        _add_wind_dir(r)
    return results


# ---- Ranking queries ----

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
        SELECT district_name_vi, {col} AS value, ward_count
        FROM fact_weather_district_hourly
        WHERE ts_utc = (
            SELECT MAX(ts_utc) FROM fact_weather_district_hourly
            WHERE ts_utc > NOW() - INTERVAL '2 hours'
        )
          AND {col} IS NOT NULL
        ORDER BY {col} {direction}
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
    district_name: str,
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

    results = query(f"""
        SELECT d.ward_name_vi, w.{col} AS value
        FROM fact_weather_hourly w
        JOIN dim_ward d ON w.ward_id = d.ward_id
        WHERE d.district_name_vi = %s
          AND w.data_kind = 'current'
          AND w.ts_utc > NOW() - INTERVAL '2 hours'
          AND w.{col} IS NOT NULL
        ORDER BY w.{col} {direction}
        LIMIT %s
    """, (district_name, limit))

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
