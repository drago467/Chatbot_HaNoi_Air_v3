from __future__ import annotations

from app.core.logging_config import get_logger, setup_logging
from app.db.connection import get_db_connection


logger = get_logger(__name__)


DDL = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS dim_ward (
    ward_id TEXT PRIMARY KEY,              -- HanoiAir internal_id: ID_XXXXX
    ward_name_vi TEXT NOT NULL,
    ward_name_norm TEXT,
    ward_name_core_norm TEXT,              -- Name without prefix (e.g., 'dich vong')
    ward_prefix_norm TEXT,                 -- Prefix only (e.g., 'phuong')
    district_name_vi TEXT,
    district_name_norm TEXT,

    -- Bounding box from HanoiAir
    minx DOUBLE PRECISION,
    miny DOUBLE PRECISION,
    maxx DOUBLE PRECISION,
    maxy DOUBLE PRECISION,

    -- Centroid derived from bbox
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,

    is_urban BOOLEAN,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fact_air_pollution_hourly (
    ward_id TEXT REFERENCES dim_ward(ward_id),
    ts_utc TIMESTAMPTZ NOT NULL,
    aqi SMALLINT,
    co DOUBLE PRECISION,
    no DOUBLE PRECISION,
    no2 DOUBLE PRECISION,
    o3 DOUBLE PRECISION,
    so2 DOUBLE PRECISION,
    pm2_5 DOUBLE PRECISION,
    pm10 DOUBLE PRECISION,
    nh3 DOUBLE PRECISION,
    data_kind TEXT, -- 'forecast' | 'history' | 'current'
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ward_id, ts_utc)
);

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
    data_kind TEXT, -- 'forecast' | 'history' | 'current'
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ward_id, ts_utc)
);

-- Trigram indexes for fuzzy search
CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_name_trgm
    ON dim_ward USING GIN (ward_name_vi gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_dim_ward_district_name_trgm
    ON dim_ward USING GIN (district_name_vi gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_name_norm_trgm
    ON dim_ward USING GIN (ward_name_norm gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_name_core_norm_trgm
    ON dim_ward USING GIN (ward_name_core_norm gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_prefix_norm_trgm
    ON dim_ward USING GIN (ward_prefix_norm gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_dim_ward_district_name_norm_trgm
    ON dim_ward USING GIN (district_name_norm gin_trgm_ops);
"""


def init_db() -> None:
    setup_logging()

    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                logger.info("Initializing database schema (pg_trgm, dim_ward, indexes)...")
                # 1) Ensure base table exists (older versions may not have new columns)
                logger.info("Ensuring base tables exist...")
                cur.execute(
                    """
                    CREATE EXTENSION IF NOT EXISTS pg_trgm;

                    CREATE TABLE IF NOT EXISTS dim_ward (
                        ward_id TEXT PRIMARY KEY,
                        ward_name_vi TEXT NOT NULL,
                        ward_name_norm TEXT,
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

                    CREATE TABLE IF NOT EXISTS fact_air_pollution_hourly (
                        ward_id TEXT REFERENCES dim_ward(ward_id),
                        ts_utc TIMESTAMPTZ NOT NULL,
                        aqi SMALLINT,
                        co DOUBLE PRECISION,
                        no DOUBLE PRECISION,
                        no2 DOUBLE PRECISION,
                        o3 DOUBLE PRECISION,
                        so2 DOUBLE PRECISION,
                        pm2_5 DOUBLE PRECISION,
                        pm10 DOUBLE PRECISION,
                        nh3 DOUBLE PRECISION,
                        data_kind TEXT,
                        ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (ward_id, ts_utc)
                    );

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
                        ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (ward_id, ts_utc)
                    );
                    """
                )

                # 2) Migration: add new columns
                logger.info("Applying migration for new columns (ward_prefix_norm, ward_name_core_norm)...")
                cur.execute(
                    """
                    ALTER TABLE dim_ward ADD COLUMN IF NOT EXISTS ward_name_core_norm TEXT;
                    ALTER TABLE dim_ward ADD COLUMN IF NOT EXISTS ward_prefix_norm TEXT;
                    """
                )

                # 3) Ensure indexes
                logger.info("Ensuring trigram indexes...")
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_name_trgm
                        ON dim_ward USING GIN (ward_name_vi gin_trgm_ops);

                    CREATE INDEX IF NOT EXISTS idx_dim_ward_district_name_trgm
                        ON dim_ward USING GIN (district_name_vi gin_trgm_ops);

                    CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_name_norm_trgm
                        ON dim_ward USING GIN (ward_name_norm gin_trgm_ops);

                    CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_name_core_norm_trgm
                        ON dim_ward USING GIN (ward_name_core_norm gin_trgm_ops);

                    CREATE INDEX IF NOT EXISTS idx_dim_ward_ward_prefix_norm_trgm
                        ON dim_ward USING GIN (ward_prefix_norm gin_trgm_ops);

                    CREATE INDEX IF NOT EXISTS idx_dim_ward_district_name_norm_trgm
                        ON dim_ward USING GIN (district_name_norm gin_trgm_ops);
                    """
                )

        logger.info("Database init/migration completed.")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
