"""
Aggregation functions for pre-aggregated weather tables.
Creates district-level and city-level aggregates from ward-level data.

Note on wind_deg: Uses circular mean (atan2 of sin/cos averages)
because simple AVG() gives wrong results for circular data.
Example: AVG(350, 10) = 180 (WRONG), circular mean = 0 (CORRECT).
"""

from app.db.connection import get_db_connection, release_connection


# Circular mean SQL for wind_deg (0-360 degrees)
# MOD(numeric, numeric) works in PostgreSQL; both args must be ::numeric
_WIND_DEG_CIRCULAR_MEAN = """ROUND(MOD(
        DEGREES(ATAN2(
            AVG(SIN(RADIANS(w.wind_deg))),
            AVG(COS(RADIANS(w.wind_deg)))
        ))::numeric + 360.0::numeric,
        360.0::numeric
    ), 1)"""


def aggregate_district_hourly(data_kind: str = 'current') -> dict:
    """Aggregate weather data from ward level to district level (hourly)."""
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                sql = f"""
                    INSERT INTO fact_weather_district_hourly (
                        district_name_vi, ts_utc,
                        avg_temp, min_temp, max_temp,
                        avg_humidity, avg_wind_speed,
                        weather_main, ward_count,
                        avg_dew_point, avg_pressure, avg_clouds,
                        avg_visibility, avg_uvi, avg_pop, avg_rain_1h,
                        avg_wind_deg, max_wind_gust, max_uvi
                    )
                    SELECT
                        d.district_name_vi,
                        w.ts_utc,
                        ROUND(AVG(w.temp)::numeric, 2),
                        MIN(w.temp),
                        MAX(w.temp),
                        ROUND(AVG(w.humidity)::numeric, 1),
                        ROUND(AVG(w.wind_speed)::numeric, 2),
                        (SELECT weather_main FROM fact_weather_hourly w2
                         INNER JOIN dim_ward d2 ON w2.ward_id = d2.ward_id
                         WHERE d2.district_name_vi = d.district_name_vi
                           AND w2.ts_utc = w.ts_utc
                         ORDER BY w2.temp DESC LIMIT 1),
                        COUNT(DISTINCT w.ward_id),
                        ROUND(AVG(w.dew_point)::numeric, 2),
                        ROUND(AVG(w.pressure)::numeric, 1),
                        ROUND(AVG(w.clouds)::numeric, 1),
                        ROUND(AVG(w.visibility)::numeric, 0),
                        ROUND(AVG(w.uvi)::numeric, 2),
                        ROUND(AVG(w.pop)::numeric, 3),
                        ROUND(AVG(w.rain_1h)::numeric, 2),
                        {_WIND_DEG_CIRCULAR_MEAN},
                        MAX(w.wind_gust),
                        MAX(w.uvi)
                    FROM fact_weather_hourly w
                    INNER JOIN dim_ward d ON w.ward_id = d.ward_id
                    WHERE w.data_kind = %s
                      AND d.district_name_vi IS NOT NULL
                    GROUP BY d.district_name_vi, w.ts_utc
                    ON CONFLICT (district_name_vi, ts_utc) DO UPDATE SET
                        avg_temp = EXCLUDED.avg_temp,
                        min_temp = EXCLUDED.min_temp,
                        max_temp = EXCLUDED.max_temp,
                        avg_humidity = EXCLUDED.avg_humidity,
                        avg_wind_speed = EXCLUDED.avg_wind_speed,
                        weather_main = EXCLUDED.weather_main,
                        ward_count = EXCLUDED.ward_count,
                        avg_dew_point = EXCLUDED.avg_dew_point,
                        avg_pressure = EXCLUDED.avg_pressure,
                        avg_clouds = EXCLUDED.avg_clouds,
                        avg_visibility = EXCLUDED.avg_visibility,
                        avg_uvi = EXCLUDED.avg_uvi,
                        avg_pop = EXCLUDED.avg_pop,
                        avg_rain_1h = EXCLUDED.avg_rain_1h,
                        avg_wind_deg = EXCLUDED.avg_wind_deg,
                        max_wind_gust = EXCLUDED.max_wind_gust,
                        max_uvi = EXCLUDED.max_uvi
                """
                cur.execute(sql, (data_kind,))
                cur.execute("SELECT COUNT(*) FROM fact_weather_district_hourly")
                row_count = cur.fetchone()[0]
        return {"status": "ok", "data_kind": data_kind, "records": row_count}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            release_connection(conn)


def aggregate_city_hourly(data_kind: str = 'current') -> dict:
    """Aggregate weather data from ward level to city level (hourly)."""
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                sql = f"""
                    INSERT INTO fact_weather_city_hourly (
                        ts_utc,
                        avg_temp, min_temp, max_temp,
                        avg_humidity, avg_wind_speed,
                        weather_main, ward_count,
                        avg_dew_point, avg_pressure, avg_clouds,
                        avg_visibility, avg_uvi, avg_pop, avg_rain_1h,
                        avg_wind_deg, max_wind_gust, max_uvi
                    )
                    SELECT
                        w.ts_utc,
                        ROUND(AVG(w.temp)::numeric, 2),
                        MIN(w.temp),
                        MAX(w.temp),
                        ROUND(AVG(w.humidity)::numeric, 1),
                        ROUND(AVG(w.wind_speed)::numeric, 2),
                        (SELECT weather_main FROM fact_weather_hourly w2
                         WHERE w2.ts_utc = w.ts_utc
                         ORDER BY w2.temp DESC LIMIT 1),
                        COUNT(DISTINCT w.ward_id),
                        ROUND(AVG(w.dew_point)::numeric, 2),
                        ROUND(AVG(w.pressure)::numeric, 1),
                        ROUND(AVG(w.clouds)::numeric, 1),
                        ROUND(AVG(w.visibility)::numeric, 0),
                        ROUND(AVG(w.uvi)::numeric, 2),
                        ROUND(AVG(w.pop)::numeric, 3),
                        ROUND(AVG(w.rain_1h)::numeric, 2),
                        {_WIND_DEG_CIRCULAR_MEAN},
                        MAX(w.wind_gust),
                        MAX(w.uvi)
                    FROM fact_weather_hourly w
                    WHERE w.data_kind = %s
                    GROUP BY w.ts_utc
                    ON CONFLICT (ts_utc) DO UPDATE SET
                        avg_temp = EXCLUDED.avg_temp,
                        min_temp = EXCLUDED.min_temp,
                        max_temp = EXCLUDED.max_temp,
                        avg_humidity = EXCLUDED.avg_humidity,
                        avg_wind_speed = EXCLUDED.avg_wind_speed,
                        weather_main = EXCLUDED.weather_main,
                        ward_count = EXCLUDED.ward_count,
                        avg_dew_point = EXCLUDED.avg_dew_point,
                        avg_pressure = EXCLUDED.avg_pressure,
                        avg_clouds = EXCLUDED.avg_clouds,
                        avg_visibility = EXCLUDED.avg_visibility,
                        avg_uvi = EXCLUDED.avg_uvi,
                        avg_pop = EXCLUDED.avg_pop,
                        avg_rain_1h = EXCLUDED.avg_rain_1h,
                        avg_wind_deg = EXCLUDED.avg_wind_deg,
                        max_wind_gust = EXCLUDED.max_wind_gust,
                        max_uvi = EXCLUDED.max_uvi
                """
                cur.execute(sql, (data_kind,))
                cur.execute("SELECT COUNT(*) FROM fact_weather_city_hourly")
                row_count = cur.fetchone()[0]
        return {"status": "ok", "data_kind": data_kind, "records": row_count}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            release_connection(conn)


def run_aggregation():
    """Run hourly aggregation functions (backward compat)."""
    results = {
        "district_hourly_current": aggregate_district_hourly('current'),
        "district_hourly_forecast": aggregate_district_hourly('forecast'),
        "district_hourly_history": aggregate_district_hourly('history'),
        "city_hourly_current": aggregate_city_hourly('current'),
        "city_hourly_forecast": aggregate_city_hourly('forecast'),
        "city_hourly_history": aggregate_city_hourly('history'),
    }
    return results


def aggregate_district_daily(data_kind: str = 'forecast') -> dict:
    """Aggregate weather data from ward level to district level (daily)."""
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                sql = f"""
                    INSERT INTO fact_weather_district_daily (
                        district_name_vi, date,
                        avg_temp, temp_min, temp_max,
                        avg_humidity, avg_pop, total_rain,
                        weather_main, ward_count,
                        avg_dew_point, avg_pressure, avg_clouds,
                        max_uvi, avg_wind_deg, max_wind_gust
                    )
                    SELECT
                        d.district_name_vi,
                        w.date,
                        ROUND(AVG(w.temp_avg)::numeric, 2),
                        MIN(w.temp_min),
                        MAX(w.temp_max),
                        ROUND(AVG(w.humidity)::numeric, 1),
                        ROUND(AVG(w.pop)::numeric, 2),
                        ROUND(AVG(w.rain_total)::numeric, 2),
                        (SELECT weather_main FROM fact_weather_daily w2
                         INNER JOIN dim_ward d2 ON w2.ward_id = d2.ward_id
                         WHERE d2.district_name_vi = d.district_name_vi
                           AND w2.date = w.date
                         ORDER BY w2.temp_avg DESC LIMIT 1),
                        COUNT(DISTINCT w.ward_id),
                        ROUND(AVG(w.dew_point)::numeric, 2),
                        ROUND(AVG(w.pressure)::numeric, 1),
                        ROUND(AVG(w.clouds)::numeric, 1),
                        MAX(w.uvi),
                        {_WIND_DEG_CIRCULAR_MEAN},
                        MAX(w.wind_gust)
                    FROM fact_weather_daily w
                    INNER JOIN dim_ward d ON w.ward_id = d.ward_id
                    WHERE w.data_kind = %s
                      AND d.district_name_vi IS NOT NULL
                    GROUP BY d.district_name_vi, w.date
                    ON CONFLICT (district_name_vi, date) DO UPDATE SET
                        avg_temp = EXCLUDED.avg_temp,
                        temp_min = EXCLUDED.temp_min,
                        temp_max = EXCLUDED.temp_max,
                        avg_humidity = EXCLUDED.avg_humidity,
                        avg_pop = EXCLUDED.avg_pop,
                        total_rain = EXCLUDED.total_rain,
                        weather_main = EXCLUDED.weather_main,
                        ward_count = EXCLUDED.ward_count,
                        avg_dew_point = EXCLUDED.avg_dew_point,
                        avg_pressure = EXCLUDED.avg_pressure,
                        avg_clouds = EXCLUDED.avg_clouds,
                        max_uvi = EXCLUDED.max_uvi,
                        avg_wind_deg = EXCLUDED.avg_wind_deg,
                        max_wind_gust = EXCLUDED.max_wind_gust
                """
                cur.execute(sql, (data_kind,))
                cur.execute("SELECT COUNT(*) FROM fact_weather_district_daily")
                row_count = cur.fetchone()[0]
        return {"status": "ok", "data_kind": data_kind, "records": row_count}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            release_connection(conn)


def aggregate_city_daily(data_kind: str = 'forecast') -> dict:
    """Aggregate weather data from ward level to city level (daily)."""
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                sql = f"""
                    INSERT INTO fact_weather_city_daily (
                        date,
                        avg_temp, temp_min, temp_max,
                        avg_humidity, avg_pop, total_rain,
                        weather_main, ward_count,
                        avg_dew_point, avg_pressure, avg_clouds,
                        max_uvi, avg_wind_deg, max_wind_gust
                    )
                    SELECT
                        w.date,
                        ROUND(AVG(w.temp_avg)::numeric, 2),
                        MIN(w.temp_min),
                        MAX(w.temp_max),
                        ROUND(AVG(w.humidity)::numeric, 1),
                        ROUND(AVG(w.pop)::numeric, 2),
                        ROUND(AVG(w.rain_total)::numeric, 2),
                        (SELECT weather_main FROM fact_weather_daily w2
                         WHERE w2.date = w.date
                         ORDER BY w2.temp_avg DESC LIMIT 1),
                        COUNT(DISTINCT w.ward_id),
                        ROUND(AVG(w.dew_point)::numeric, 2),
                        ROUND(AVG(w.pressure)::numeric, 1),
                        ROUND(AVG(w.clouds)::numeric, 1),
                        MAX(w.uvi),
                        {_WIND_DEG_CIRCULAR_MEAN},
                        MAX(w.wind_gust)
                    FROM fact_weather_daily w
                    WHERE w.data_kind = %s
                    GROUP BY w.date
                    ON CONFLICT (date) DO UPDATE SET
                        avg_temp = EXCLUDED.avg_temp,
                        temp_min = EXCLUDED.temp_min,
                        temp_max = EXCLUDED.temp_max,
                        avg_humidity = EXCLUDED.avg_humidity,
                        avg_pop = EXCLUDED.avg_pop,
                        total_rain = EXCLUDED.total_rain,
                        weather_main = EXCLUDED.weather_main,
                        ward_count = EXCLUDED.ward_count,
                        avg_dew_point = EXCLUDED.avg_dew_point,
                        avg_pressure = EXCLUDED.avg_pressure,
                        avg_clouds = EXCLUDED.avg_clouds,
                        max_uvi = EXCLUDED.max_uvi,
                        avg_wind_deg = EXCLUDED.avg_wind_deg,
                        max_wind_gust = EXCLUDED.max_wind_gust
                """
                cur.execute(sql, (data_kind,))
                cur.execute("SELECT COUNT(*) FROM fact_weather_city_daily")
                row_count = cur.fetchone()[0]
        return {"status": "ok", "data_kind": data_kind, "records": row_count}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            release_connection(conn)


def run_all_aggregations():
    """Run ALL aggregation functions (hourly + daily)."""
    results = {
        "district_hourly_current": aggregate_district_hourly('current'),
        "district_hourly_forecast": aggregate_district_hourly('forecast'),
        "district_hourly_history": aggregate_district_hourly('history'),
        "city_hourly_current": aggregate_city_hourly('current'),
        "city_hourly_forecast": aggregate_city_hourly('forecast'),
        "city_hourly_history": aggregate_city_hourly('history'),
        "district_daily_forecast": aggregate_district_daily('forecast'),
        "district_daily_history": aggregate_district_daily('history'),
        "city_daily_forecast": aggregate_city_daily('forecast'),
        "city_daily_history": aggregate_city_daily('history'),
    }
    return results


if __name__ == "__main__":
    results = run_all_aggregations()
    for key, result in results.items():
        print(f"{key}: {result}")
