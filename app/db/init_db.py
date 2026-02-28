from __future__ import annotations

from app.core.logging_config import get_logger, setup_logging
from app.db.connection import get_db_connection


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
                for col_name, col_type in daily_columns:
                    try:
                        cur.execute(f"ALTER TABLE fact_weather_daily ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
                    except Exception as e:
                        logger.warning(f"Could not add column {col_name}: {e}")
                
                # Add source/source_job to all fact tables
                tables = [
                    "fact_air_pollution_hourly", "fact_weather_hourly", "fact_weather_daily",
                    "fact_hanoiair_daily", "fact_hanoiair_ranking"
                ]
                for table in tables:
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS source TEXT;")
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS source_job TEXT;")

                # Add new dim_ward columns
                cur.execute("ALTER TABLE dim_ward ADD COLUMN IF NOT EXISTS ward_name_core_norm TEXT;")
                cur.execute("ALTER TABLE dim_ward ADD COLUMN IF NOT EXISTS ward_prefix_norm TEXT;")

                # Trigram indexes
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_name_trgm ON dim_ward USING GIN (ward_name_vi gin_trgm_ops);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_district_name_trgm ON dim_ward USING GIN (district_name_vi gin_trgm_ops);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_name_norm_trgm ON dim_ward USING GIN (ward_name_norm gin_trgm_ops);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_name_core_norm_trgm ON dim_ward USING GIN (ward_name_core_norm gin_trgm_ops);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_prefix_norm_trgm ON dim_ward USING GIN (ward_prefix_norm gin_trgm_ops);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_district_name_norm_trgm ON dim_ward USING GIN (district_name_norm gin_trgm_ops);")

        logger.info("Database init/migration completed.")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
