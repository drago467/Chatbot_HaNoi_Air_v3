from __future__ import annotations

from app.core.logging_config import get_logger, setup_logging
from app.db.connection import get_db_connection, release_connection


logger = get_logger(__name__)


def init_db() -> None:
    setup_logging()

    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                logger.info("Initializing database schema...")
                
                # 1) Extensions
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

                # 2) dim_ward
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS dim_ward (
                        ward_id TEXT PRIMARY KEY,
                        ward_name_vi TEXT NOT NULL,
                        ward_name_norm TEXT,
                        ward_name_core_norm TEXT,
                        ward_prefix_norm TEXT,
                        district_name_vi TEXT,
                        district_name_norm TEXT,
                        minx DOUBLE PRECISION,
                        miny DOUBLE PRECISION,
                        maxx DOUBLE PRECISION,
                        maxy DOUBLE PRECISION,
                        lat DOUBLE PRECISION,
                        lon DOUBLE PRECISION,
                        is_urban BOOLEAN,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)

                # 3) fact_air_pollution_hourly
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS fact_air_pollution_hourly (
                        ward_id TEXT REFERENCES dim_ward(ward_id),
                        ts_utc TIMESTAMPTZ NOT NULL,
                        aqi SMALLINT,
                        aqi_vn INT,
                        co DOUBLE PRECISION,
                        no DOUBLE PRECISION,
                        no2 DOUBLE PRECISION,
                        o3 DOUBLE PRECISION,
                        so2 DOUBLE PRECISION,
                        pm2_5 DOUBLE PRECISION,
                        pm10 DOUBLE PRECISION,
                        nh3 DOUBLE PRECISION,
                        data_kind TEXT,
                        source TEXT,
                        source_job TEXT,
                        ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (ward_id, ts_utc)
                    );
                """)

                # 4) fact_weather_hourly
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS fact_weather_hourly (
                        ward_id TEXT REFERENCES dim_ward(ward_id),
                        ts_utc TIMESTAMPTZ NOT NULL,
                        temp DOUBLE PRECISION,
                        feels_like DOUBLE PRECISION,
                        pressure INT,
                        humidity INT,
                        dew_point DOUBLE PRECISION,
                        clouds INT,
                        wind_speed DOUBLE PRECISION,
                        wind_deg INT,
                        wind_gust DOUBLE PRECISION,
                        visibility INT,
                        uvi DOUBLE PRECISION,
                        pop DOUBLE PRECISION,
                        rain_1h DOUBLE PRECISION,
                        snow_1h DOUBLE PRECISION,
                        weather_main TEXT,
                        weather_description TEXT,
                        data_kind TEXT,
                        source TEXT,
                        source_job TEXT,
                        ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (ward_id, ts_utc)
                    );
                """)

                # 5) fact_weather_daily
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS fact_weather_daily (
                        ward_id TEXT REFERENCES dim_ward(ward_id),
                        date DATE NOT NULL,
                        -- Temperature
                        temp_min DOUBLE PRECISION,
                        temp_max DOUBLE PRECISION,
                        temp_avg DOUBLE PRECISION,
                        temp_morn DOUBLE PRECISION,
                        temp_day DOUBLE PRECISION,
                        temp_eve DOUBLE PRECISION,
                        temp_night DOUBLE PRECISION,
                        -- Feels like
                        feels_like_morn DOUBLE PRECISION,
                        feels_like_day DOUBLE PRECISION,
                        feels_like_eve DOUBLE PRECISION,
                        feels_like_night DOUBLE PRECISION,
                        -- Other weather
                        humidity INT,
                        pressure INT,
                        dew_point DOUBLE PRECISION,
                        wind_speed DOUBLE PRECISION,
                        wind_deg INT,
                        wind_gust DOUBLE PRECISION,
                        clouds INT,
                        pop DOUBLE PRECISION,
                        rain_total DOUBLE PRECISION,
                        uvi DOUBLE PRECISION,
                        -- Weather condition
                        weather_main TEXT,
                        weather_description TEXT,
                        summary TEXT,
                        -- Sun times
                        sunrise TIMESTAMPTZ,
                        sunset TIMESTAMPTZ,
                        -- Metadata
                        data_kind TEXT,
                        source TEXT,
                        source_job TEXT,
                        ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (ward_id, date)
                    );
                """)

                # 6) fact_hanoiair_daily
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS fact_hanoiair_daily (
                        ward_id TEXT REFERENCES dim_ward(ward_id),
                        date DATE NOT NULL,
                        aqi DOUBLE PRECISION,
                        pm2_5 DOUBLE PRECISION,
                        is_forecast BOOLEAN NOT NULL DEFAULT FALSE,
                        source TEXT DEFAULT 'hanoiair',
                        source_job TEXT,
                        ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (ward_id, date)
                    );
                """)

                # 7) fact_hanoiair_ranking
                # NOTE: hanoiair_administrative_id stores the raw ID from
                # HanoiAir rankingprovince API (format: ID_XXXXX).
                # No FK to dim_ward because the ranking API may include
                # administrative units not in our ward dimension.
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS fact_hanoiair_ranking (
                        date DATE NOT NULL,
                        hanoiair_administrative_id TEXT NOT NULL,
                        rank INT,
                        aqi_avg DOUBLE PRECISION,
                        aqi_avg_pre DOUBLE PRECISION,
                        source TEXT DEFAULT 'hanoiair',
                        source_job TEXT DEFAULT 'ranking_daily',
                        ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (date, hanoiair_administrative_id)
                    );
                """)

                # Migration block for existing tables
                logger.info("Applying migrations (if any)...")
                
                # Add aqi_vn to air pollution
                cur.execute("ALTER TABLE fact_air_pollution_hourly ADD COLUMN IF NOT EXISTS aqi_vn INT;")
                
                # Add missing columns to fact_weather_daily (2026-02-28)
                daily_columns = [
                    ("temp_avg", "DOUBLE PRECISION"),
                    ("temp_morn", "DOUBLE PRECISION"),
                    ("temp_day", "DOUBLE PRECISION"),
                    ("temp_eve", "DOUBLE PRECISION"),
                    ("temp_night", "DOUBLE PRECISION"),
                    ("feels_like_morn", "DOUBLE PRECISION"),
                    ("feels_like_day", "DOUBLE PRECISION"),
                    ("feels_like_eve", "DOUBLE PRECISION"),
                    ("feels_like_night", "DOUBLE PRECISION"),
                    ("pressure", "INT"),
                    ("dew_point", "DOUBLE PRECISION"),
                    ("wind_deg", "INT"),
                    ("wind_gust", "DOUBLE PRECISION"),
                    ("clouds", "INT"),
                    ("weather_main", "TEXT"),
                    ("weather_description", "TEXT"),
                    ("sunrise", "TIMESTAMPTZ"),
                    ("sunset", "TIMESTAMPTZ"),
                ]
                # Safe migration: use SAVEPOINT to prevent transaction abort
                # Migration 1: fact_weather_daily columns
                cur.execute("SAVEPOINT sp1")
                try:
                    for col_name, col_type in daily_columns:
                        cur.execute(f"ALTER TABLE fact_weather_daily ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT sp1")
                    logger.warning(f"Could not add columns to fact_weather_daily: {e}")
                
                # Migration 2: source columns
                tables = [
                    "fact_air_pollution_hourly", "fact_weather_hourly", "fact_weather_daily",
                    "fact_hanoiair_daily", "fact_hanoiair_ranking"
                ]
                cur.execute("SAVEPOINT sp2")
                try:
                    for table in tables:
                        cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS source TEXT;")
                        cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS source_job TEXT;")
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT sp2")
                    logger.warning(f"Could not add source columns: {e}")
                
                # Migration 3: dim_ward columns
                cur.execute("SAVEPOINT sp3")
                try:
                    cur.execute("ALTER TABLE dim_ward ADD COLUMN IF NOT EXISTS ward_name_core_norm TEXT;")
                    cur.execute("ALTER TABLE dim_ward ADD COLUMN IF NOT EXISTS ward_prefix_norm TEXT;")
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT sp3")
                    logger.warning(f"Could not add dim_ward columns: {e}")

                # Trigram indexes
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_name_trgm ON dim_ward USING GIN (ward_name_vi gin_trgm_ops);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_district_name_trgm ON dim_ward USING GIN (district_name_vi gin_trgm_ops);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_name_norm_trgm ON dim_ward USING GIN (ward_name_norm gin_trgm_ops);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_name_core_norm_trgm ON dim_ward USING GIN (ward_name_core_norm gin_trgm_ops);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_prefix_norm_trgm ON dim_ward USING GIN (ward_prefix_norm gin_trgm_ops);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_district_name_norm_trgm ON dim_ward USING GIN (district_name_norm gin_trgm_ops);")

                # ============================================================
                # GIAI DOAN 2: Pre-aggregated Weather Tables
                # ============================================================
                
                # 1. fact_weather_district_hourly
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS fact_weather_district_hourly (
                        district_name_vi TEXT NOT NULL,
                        ts_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                        avg_temp DOUBLE PRECISION,
                        min_temp DOUBLE PRECISION,
                        max_temp DOUBLE PRECISION,
                        avg_humidity DOUBLE PRECISION,
                        avg_wind_speed DOUBLE PRECISION,
                        weather_main TEXT,
                        ward_count INTEGER,
                        PRIMARY KEY (district_name_vi, ts_utc)
                    );
                """)
                
                # 2. fact_weather_district_daily
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS fact_weather_district_daily (
                        district_name_vi TEXT NOT NULL,
                        date DATE NOT NULL,
                        avg_temp DOUBLE PRECISION,
                        temp_min DOUBLE PRECISION,
                        temp_max DOUBLE PRECISION,
                        avg_humidity DOUBLE PRECISION,
                        avg_pop DOUBLE PRECISION,
                        total_rain DOUBLE PRECISION,
                        weather_main TEXT,
                        ward_count INTEGER,
                        PRIMARY KEY (district_name_vi, date)
                    );
                """)
                
                # 3. fact_weather_city_hourly
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS fact_weather_city_hourly (
                        ts_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                        avg_temp DOUBLE PRECISION,
                        min_temp DOUBLE PRECISION,
                        max_temp DOUBLE PRECISION,
                        avg_humidity DOUBLE PRECISION,
                        avg_wind_speed DOUBLE PRECISION,
                        weather_main TEXT,
                        ward_count INTEGER,
                        PRIMARY KEY (ts_utc)
                    );
                """)
                
                # 4. fact_weather_city_daily
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS fact_weather_city_daily (
                        date DATE NOT NULL,
                        avg_temp DOUBLE PRECISION,
                        temp_min DOUBLE PRECISION,
                        temp_max DOUBLE PRECISION,
                        avg_humidity DOUBLE PRECISION,
                        avg_pop DOUBLE PRECISION,
                        total_rain DOUBLE PRECISION,
                        weather_main TEXT,
                        ward_count INTEGER,
                        PRIMARY KEY (date)
                    );
                """)
                
                # Indexes for aggregate tables
                cur.execute("CREATE INDEX IF NOT EXISTS idx_district_hourly_ts_utc ON fact_weather_district_hourly(ts_utc DESC);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_district_hourly_district_date ON fact_weather_district_hourly(district_name_vi, ts_utc DESC);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_district_daily_date ON fact_weather_district_daily(date DESC);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_district_daily_district_date ON fact_weather_district_daily(district_name_vi, date DESC);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_city_hourly_ts_utc ON fact_weather_city_hourly(ts_utc DESC);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_city_daily_date ON fact_weather_city_daily(date DESC);")

                # Migration: Add missing columns to aggregated tables (Phase 2.1)
                logger.info("Adding columns to aggregated tables...")

                # Hourly tables - add missing weather columns
                hourly_cols = [
                    ("avg_dew_point", "DOUBLE PRECISION"),
                    ("avg_pressure", "DOUBLE PRECISION"),
                    ("avg_clouds", "DOUBLE PRECISION"),
                    ("avg_visibility", "DOUBLE PRECISION"),
                    ("avg_uvi", "DOUBLE PRECISION"),
                    ("avg_pop", "DOUBLE PRECISION"),
                    ("avg_rain_1h", "DOUBLE PRECISION"),
                    ("avg_wind_deg", "DOUBLE PRECISION"),
                    ("max_wind_gust", "DOUBLE PRECISION"),
                    ("max_uvi", "DOUBLE PRECISION"),
                ]

                cur.execute("SAVEPOINT sp_hourly")
                try:
                    for col_name, col_type in hourly_cols:
                        cur.execute(f"ALTER TABLE fact_weather_district_hourly ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
                        cur.execute(f"ALTER TABLE fact_weather_city_hourly ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT sp_hourly")
                    logger.warning(f"Could not add hourly columns: {e}")

                # Daily tables - add missing weather columns
                daily_cols = [
                    ("avg_dew_point", "DOUBLE PRECISION"),
                    ("avg_pressure", "DOUBLE PRECISION"),
                    ("avg_clouds", "DOUBLE PRECISION"),
                    ("max_uvi", "DOUBLE PRECISION"),
                    ("avg_wind_deg", "DOUBLE PRECISION"),
                    ("max_wind_gust", "DOUBLE PRECISION"),
                ]

                cur.execute("SAVEPOINT sp_daily")
                try:
                    for col_name, col_type in daily_cols:
                        cur.execute(f"ALTER TABLE fact_weather_district_daily ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
                        cur.execute(f"ALTER TABLE fact_weather_city_daily ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT sp_daily")
                    logger.warning(f"Could not add daily columns: {e}")

        logger.info("Database init/migration completed.")
    finally:
        release_connection(conn)


if __name__ == "__main__":
    init_db()
