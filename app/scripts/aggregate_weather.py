"""
Aggregation functions for pre-aggregated weather tables.
Creates district-level and city-level aggregates from ward-level data.
"""

from app.db.connection import get_db_connection, release_connection


def aggregate_district_hourly(data_kind: str = 'current') -> dict:
    """Aggregate weather data from ward level to district level (hourly).
    
    Args:
        data_kind: 'current', 'forecast', or 'history'
        
    Returns:
        Dict with status and count of aggregated records
    """
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                sql = """
                    INSERT INTO fact_weather_district_hourly (
                        district_name_vi,
                        ts_utc,
                        avg_temp,
                        min_temp,
                        max_temp,
                        avg_humidity,
                        avg_wind_speed,
                        weather_main,
                        ward_count
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
                         WHERE d2.district_name_vi = d.district_name_vi AND w2.ts_utc = w.ts_utc 
                         ORDER BY w2.temp DESC LIMIT 1),
                        COUNT(DISTINCT w.ward_id)
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
                        ward_count = EXCLUDED.ward_count
                """
                cur.execute(sql, (data_kind,))
                
                # Get actual count - rowcount doesn't work reliably with INSERT...ON CONFLICT
                # Note: fact_weather_district_hourly doesn't have data_kind column
                cur.execute("""
                    SELECT COUNT(*) FROM fact_weather_district_hourly
                """)
                result = cur.fetchone()
                row_count = result[0] if result else 0
        
        return {"status": "ok", "data_kind": data_kind, "records_upserted": row_count}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            release_connection(conn)


def aggregate_city_hourly(data_kind: str = 'current') -> dict:
    """Aggregate weather data from ward level to city level (hourly).
    
    Args:
        data_kind: 'current', 'forecast', or 'history'
        
    Returns:
        Dict with status and count of aggregated records
    """
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                sql = """
                    INSERT INTO fact_weather_city_hourly (
                        ts_utc,
                        avg_temp,
                        min_temp,
                        max_temp,
                        avg_humidity,
                        avg_wind_speed,
                        weather_main,
                        ward_count
                    )
                    SELECT 
                        w.ts_utc,
                        ROUND(AVG(w.temp)::numeric, 2),
                        MIN(w.temp),
                        MAX(w.temp),
                        ROUND(AVG(w.humidity)::numeric, 1),
                        ROUND(AVG(w.wind_speed)::numeric, 2),
                        (SELECT weather_main FROM fact_weather_hourly w2 WHERE w2.ts_utc = w.ts_utc ORDER BY w2.temp DESC LIMIT 1),
                        COUNT(DISTINCT w.ward_id)
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
                        ward_count = EXCLUDED.ward_count
                """
                cur.execute(sql, (data_kind,))
                
                # Get actual count - rowcount doesn't work reliably with INSERT...ON CONFLICT
                cur.execute("SELECT COUNT(*) FROM fact_weather_city_hourly")
                result = cur.fetchone()
                row_count = result[0] if result else 0
        
        return {"status": "ok", "data_kind": data_kind, "records_upserted": row_count}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            release_connection(conn)


def run_aggregation():
    """Run all aggregation functions.
    
    This should be called after weather ingestion is complete.
    """
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
    """Aggregate weather data from ward level to district level (daily).
    
    Args:
        data_kind: 'forecast' or 'history'
        
    Returns:
        Dict with status and count of aggregated records
    """
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                sql = """
                    INSERT INTO fact_weather_district_daily (
                        district_name_vi,
                        date,
                        avg_temp,
                        temp_min,
                        temp_max,
                        avg_humidity,
                        avg_pop,
                        total_rain,
                        weather_main,
                        ward_count
                    )
                    SELECT 
                        d.district_name_vi,
                        w.date,
                        ROUND(AVG(w.temp_avg)::numeric, 2),
                        MIN(w.temp_min),
                        MAX(w.temp_max),
                        ROUND(AVG(w.humidity)::numeric, 1),
                        ROUND(AVG(w.pop)::numeric, 2),
                        SUM(w.rain_total),
                        (SELECT weather_main FROM fact_weather_daily w2 
                         INNER JOIN dim_ward d2 ON w2.ward_id = d2.ward_id 
                         WHERE d2.district_name_vi = d.district_name_vi AND w2.date = w.date 
                         ORDER BY w2.temp_avg DESC LIMIT 1),
                        COUNT(DISTINCT w.ward_id)
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
                        ward_count = EXCLUDED.ward_count
                """
                cur.execute(sql, (data_kind,))
                
                # Get actual count - rowcount doesn't work reliably with INSERT...ON CONFLICT
                cur.execute("SELECT COUNT(*) FROM fact_weather_city_daily")
                result = cur.fetchone()
                row_count = result[0] if result else 0
        
        return {"status": "ok", "data_kind": data_kind, "records_upserted": row_count}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            release_connection(conn)


def aggregate_city_daily(data_kind: str = 'forecast') -> dict:
    """Aggregate weather data from ward level to city level (daily).
    
    Args:
        data_kind: 'forecast' or 'history'
        
    Returns:
        Dict with status and count of aggregated records
    """
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                sql = """
                    INSERT INTO fact_weather_city_daily (
                        date,
                        avg_temp,
                        temp_min,
                        temp_max,
                        avg_humidity,
                        avg_pop,
                        total_rain,
                        weather_main,
                        ward_count
                    )
                    SELECT 
                        w.date,
                        ROUND(AVG(w.temp_avg)::numeric, 2),
                        MIN(w.temp_min),
                        MAX(w.temp_max),
                        ROUND(AVG(w.humidity)::numeric, 1),
                        ROUND(AVG(w.pop)::numeric, 2),
                        SUM(w.rain_total),
                        (SELECT weather_main FROM fact_weather_daily w2 
                         WHERE w2.date = w.date 
                         ORDER BY w2.temp_avg DESC LIMIT 1),
                        COUNT(DISTINCT w.ward_id)
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
                        ward_count = EXCLUDED.ward_count
                """
                cur.execute(sql, (data_kind,))
                
                # Get actual count - rowcount doesn't work reliably with INSERT...ON CONFLICT
                cur.execute("SELECT COUNT(*) FROM fact_weather_city_daily")
                result = cur.fetchone()
                row_count = result[0] if result else 0
        
        return {"status": "ok", "data_kind": data_kind, "records_upserted": row_count}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            release_connection(conn)


def run_all_aggregations():
    """Run ALL aggregation functions (hourly + daily).
    
    This should be called after weather ingestion is complete.
    """
    results = {
        # Hourly
        "district_hourly_current": aggregate_district_hourly('current'),
        "district_hourly_forecast": aggregate_district_hourly('forecast'),
        "district_hourly_history": aggregate_district_hourly('history'),
        "city_hourly_current": aggregate_city_hourly('current'),
        "city_hourly_forecast": aggregate_city_hourly('forecast'),
        "city_hourly_history": aggregate_city_hourly('history'),
        # Daily
        "district_daily_forecast": aggregate_district_daily('forecast'),
        "district_daily_history": aggregate_district_daily('history'),
        "city_daily_forecast": aggregate_city_daily('forecast'),
        "city_daily_history": aggregate_city_daily('history'),
    }
    
    return results


if __name__ == "__main__":
    # Test run
    results = run_all_aggregations()
    for key, result in results.items():
        print(f"{key}: {result}")
