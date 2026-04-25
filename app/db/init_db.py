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

                # 2) dim_city — danh mục thành phố
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS dim_city (
                        city_id SERIAL PRIMARY KEY,
                        city_name_vi TEXT NOT NULL UNIQUE,
                        city_name_norm TEXT,
                        timezone TEXT DEFAULT 'Asia/Bangkok',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)

                # 3) dim_district — danh mục quận/huyện
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS dim_district (
                        district_id SERIAL PRIMARY KEY,
                        city_id INT NOT NULL REFERENCES dim_city(city_id),
                        district_name_vi TEXT NOT NULL,
                        district_name_norm TEXT,
                        is_urban BOOLEAN,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE(city_id, district_name_vi)
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_district_norm_trgm ON dim_district USING GIN (district_name_norm gin_trgm_ops);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_district_city ON dim_district(city_id);")

                # 4) dim_ward — giữ district_name_vi/norm cho fuzzy search, thêm district_id FK
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS dim_ward (
                        ward_id TEXT PRIMARY KEY,
                        ward_name_vi TEXT NOT NULL,
                        ward_name_norm TEXT,
                        ward_name_core_norm TEXT,
                        ward_prefix_norm TEXT,
                        district_id INT REFERENCES dim_district(district_id),
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

                # 3) fact_weather_hourly
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

                # Migration block for existing tables
                logger.info("Applying migrations (if any)...")
                
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
                    "fact_weather_hourly", "fact_weather_daily",
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

                # Migration 4: dim_ward.district_id FK (nếu bảng đã tồn tại từ bản cũ)
                cur.execute("SAVEPOINT sp4")
                try:
                    cur.execute("ALTER TABLE dim_ward ADD COLUMN IF NOT EXISTS district_id INT REFERENCES dim_district(district_id);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_district_id ON dim_ward(district_id);")
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT sp4")
                    logger.warning(f"Could not add district_id to dim_ward: {e}")

                # Trigram indexes
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_name_trgm ON dim_ward USING GIN (ward_name_vi gin_trgm_ops);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_district_name_trgm ON dim_ward USING GIN (district_name_vi gin_trgm_ops);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_name_norm_trgm ON dim_ward USING GIN (ward_name_norm gin_trgm_ops);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_name_core_norm_trgm ON dim_ward USING GIN (ward_name_core_norm gin_trgm_ops);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_prefix_norm_trgm ON dim_ward USING GIN (ward_prefix_norm gin_trgm_ops);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_ward_district_name_norm_trgm ON dim_ward USING GIN (district_name_norm gin_trgm_ops);")

                # ============================================================
                # GIAI DOAN 2: Pre-aggregated Weather Tables (refactor)
                # Schema mới: dùng district_id / city_id FK thay cho district_name_vi text.
                # DROP + RECREATE 4 bảng aggregate để đảm bảo schema mới sạch.
                # Các bảng fact cấp phường (fact_weather_hourly/daily) KHÔNG bị đụng.
                # ============================================================

                logger.info("Recreating aggregate tables with star schema FK...")

                # 1. fact_weather_district_hourly
                cur.execute("DROP TABLE IF EXISTS fact_weather_district_hourly CASCADE;")
                cur.execute("""
                    CREATE TABLE fact_weather_district_hourly (
                        district_id INT NOT NULL REFERENCES dim_district(district_id),
                        ts_utc TIMESTAMPTZ NOT NULL,
                        avg_temp DOUBLE PRECISION,
                        min_temp DOUBLE PRECISION,
                        max_temp DOUBLE PRECISION,
                        avg_humidity DOUBLE PRECISION,
                        avg_wind_speed DOUBLE PRECISION,
                        avg_dew_point DOUBLE PRECISION,
                        avg_pressure DOUBLE PRECISION,
                        avg_clouds DOUBLE PRECISION,
                        avg_visibility DOUBLE PRECISION,
                        avg_uvi DOUBLE PRECISION,
                        avg_pop DOUBLE PRECISION,
                        avg_rain_1h DOUBLE PRECISION,
                        avg_wind_deg DOUBLE PRECISION,
                        max_wind_gust DOUBLE PRECISION,
                        max_uvi DOUBLE PRECISION,
                        weather_main TEXT,
                        ward_count INTEGER,
                        ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (district_id, ts_utc)
                    );
                """)

                # 2. fact_weather_district_daily
                cur.execute("DROP TABLE IF EXISTS fact_weather_district_daily CASCADE;")
                cur.execute("""
                    CREATE TABLE fact_weather_district_daily (
                        district_id INT NOT NULL REFERENCES dim_district(district_id),
                        date DATE NOT NULL,
                        avg_temp DOUBLE PRECISION,
                        temp_min DOUBLE PRECISION,
                        temp_max DOUBLE PRECISION,
                        avg_humidity DOUBLE PRECISION,
                        avg_pop DOUBLE PRECISION,
                        total_rain DOUBLE PRECISION,
                        avg_wind_speed DOUBLE PRECISION,
                        avg_wind_deg DOUBLE PRECISION,
                        max_wind_gust DOUBLE PRECISION,
                        avg_dew_point DOUBLE PRECISION,
                        avg_pressure DOUBLE PRECISION,
                        avg_clouds DOUBLE PRECISION,
                        max_uvi DOUBLE PRECISION,
                        weather_main TEXT,
                        ward_count INTEGER,
                        ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (district_id, date)
                    );
                """)

                # 3. fact_weather_city_hourly
                cur.execute("DROP TABLE IF EXISTS fact_weather_city_hourly CASCADE;")
                cur.execute("""
                    CREATE TABLE fact_weather_city_hourly (
                        city_id INT NOT NULL REFERENCES dim_city(city_id),
                        ts_utc TIMESTAMPTZ NOT NULL,
                        avg_temp DOUBLE PRECISION,
                        min_temp DOUBLE PRECISION,
                        max_temp DOUBLE PRECISION,
                        avg_humidity DOUBLE PRECISION,
                        avg_wind_speed DOUBLE PRECISION,
                        avg_dew_point DOUBLE PRECISION,
                        avg_pressure DOUBLE PRECISION,
                        avg_clouds DOUBLE PRECISION,
                        avg_visibility DOUBLE PRECISION,
                        avg_uvi DOUBLE PRECISION,
                        avg_pop DOUBLE PRECISION,
                        avg_rain_1h DOUBLE PRECISION,
                        avg_wind_deg DOUBLE PRECISION,
                        max_wind_gust DOUBLE PRECISION,
                        max_uvi DOUBLE PRECISION,
                        weather_main TEXT,
                        ward_count INTEGER,
                        ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (city_id, ts_utc)
                    );
                """)

                # 4. fact_weather_city_daily
                cur.execute("DROP TABLE IF EXISTS fact_weather_city_daily CASCADE;")
                cur.execute("""
                    CREATE TABLE fact_weather_city_daily (
                        city_id INT NOT NULL REFERENCES dim_city(city_id),
                        date DATE NOT NULL,
                        avg_temp DOUBLE PRECISION,
                        temp_min DOUBLE PRECISION,
                        temp_max DOUBLE PRECISION,
                        avg_humidity DOUBLE PRECISION,
                        avg_pop DOUBLE PRECISION,
                        total_rain DOUBLE PRECISION,
                        avg_wind_speed DOUBLE PRECISION,
                        avg_wind_deg DOUBLE PRECISION,
                        max_wind_gust DOUBLE PRECISION,
                        avg_dew_point DOUBLE PRECISION,
                        avg_pressure DOUBLE PRECISION,
                        avg_clouds DOUBLE PRECISION,
                        max_uvi DOUBLE PRECISION,
                        weather_main TEXT,
                        ward_count INTEGER,
                        ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (city_id, date)
                    );
                """)

                # Indexes cho các bảng aggregate
                cur.execute("CREATE INDEX IF NOT EXISTS idx_district_hourly_ts_utc ON fact_weather_district_hourly(ts_utc DESC);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_district_hourly_district_ts ON fact_weather_district_hourly(district_id, ts_utc DESC);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_district_daily_date ON fact_weather_district_daily(date DESC);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_district_daily_district_date ON fact_weather_district_daily(district_id, date DESC);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_city_hourly_city_ts ON fact_weather_city_hourly(city_id, ts_utc DESC);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_city_daily_city_date ON fact_weather_city_daily(city_id, date DESC);")

                # ============================================================
                # GIAI DOAN 3: Authentication — users table
                # ============================================================
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")

                # ============================================================
                # GIAI DOAN 4: Chat conversations persistence
                # ============================================================
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_conversations (
                        conv_id    TEXT PRIMARY KEY,
                        thread_id  TEXT NOT NULL UNIQUE,
                        title      TEXT NOT NULL DEFAULT 'Trò chuyện mới',
                        messages   JSONB NOT NULL DEFAULT '[]',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_conv_updated ON chat_conversations(updated_at DESC);")

        logger.info("Database init/migration completed.")
    finally:
        release_connection(conn)


if __name__ == "__main__":
    init_db()
