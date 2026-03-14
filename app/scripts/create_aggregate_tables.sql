-- ============================================================
-- Pre-aggregated Weather Tables for District and City Level
-- Created for Giai đoạn 2: Multi-level weather queries
-- NOTE: This SQL file defines the ACTUAL schema used in production
-- ============================================================

-- 1. FACT_WEATHER_DISTRICT_HOURLY
-- Trung bình thời tiết theo giờ cho mỗi quận/huyện
-- ============================================================
CREATE TABLE IF NOT EXISTS fact_weather_district_hourly (
    district_name_vi TEXT NOT NULL,
    ts_utc TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Temperature
    avg_temp DOUBLE PRECISION,
    min_temp DOUBLE PRECISION,
    max_temp DOUBLE PRECISION,
    
    -- Humidity & Wind
    avg_humidity DOUBLE PRECISION,
    avg_wind_speed DOUBLE PRECISION,
    
    -- Weather condition
    weather_main TEXT,
    
    -- Count
    ward_count INTEGER,
    
    PRIMARY KEY (district_name_vi, ts_utc)
);

COMMENT ON TABLE fact_weather_district_hourly IS 'Trung bình thời tiết theo giờ cho mỗi quận/huyện';


-- 2. FACT_WEATHER_DISTRICT_DAILY
-- Trung bình thời tiết theo ngày cho mỗi quận/huyện
-- ============================================================
CREATE TABLE IF NOT EXISTS fact_weather_district_daily (
    district_name_vi TEXT NOT NULL,
    date DATE NOT NULL,
    
    -- Temperature
    avg_temp DOUBLE PRECISION,
    temp_min DOUBLE PRECISION,
    temp_max DOUBLE PRECISION,
    
    -- Humidity & Precipitation
    avg_humidity DOUBLE PRECISION,
    avg_pop DOUBLE PRECISION,
    total_rain DOUBLE PRECISION,
    
    -- Weather condition
    weather_main TEXT,
    
    -- Count
    ward_count INTEGER,
    
    PRIMARY KEY (district_name_vi, date)
);

COMMENT ON TABLE fact_weather_district_daily IS 'Trung bình thời tiết theo ngày cho mỗi quận/huyện';


-- 3. FACT_WEATHER_CITY_HOURLY
-- Trung bình thời tiết theo giờ cho toàn TP Hà Nội
-- ============================================================
CREATE TABLE IF NOT EXISTS fact_weather_city_hourly (
    ts_utc TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Temperature
    avg_temp DOUBLE PRECISION,
    min_temp DOUBLE PRECISION,
    max_temp DOUBLE PRECISION,
    
    -- Humidity & Wind
    avg_humidity DOUBLE PRECISION,
    avg_wind_speed DOUBLE PRECISION,
    
    -- Weather condition
    weather_main TEXT,
    
    -- Count
    ward_count INTEGER,
    
    PRIMARY KEY (ts_utc)
);

COMMENT ON TABLE fact_weather_city_hourly IS 'Trung bình thời tiết theo giờ cho toàn TP Hà Nội';


-- 4. FACT_WEATHER_CITY_DAILY
-- Trung bình thời tiết theo ngày cho toàn TP Hà Nội
-- ============================================================
CREATE TABLE IF NOT EXISTS fact_weather_city_daily (
    date DATE NOT NULL,
    
    -- Temperature
    avg_temp DOUBLE PRECISION,
    temp_min DOUBLE PRECISION,
    temp_max DOUBLE PRECISION,
    
    -- Humidity & Precipitation
    avg_humidity DOUBLE PRECISION,
    avg_pop DOUBLE PRECISION,
    total_rain DOUBLE PRECISION,
    
    -- Weather condition
    weather_main TEXT,
    
    -- Count
    ward_count INTEGER,
    
    PRIMARY KEY (date)
);

COMMENT ON TABLE fact_weather_city_daily IS 'Trung bình thời tiết theo ngày cho toàn TP Hà Nội';


-- ============================================================
-- Indexes for better query performance
-- ============================================================

-- District hourly indexes
CREATE INDEX IF NOT EXISTS idx_district_hourly_ts_utc 
    ON fact_weather_district_hourly(ts_utc DESC);
CREATE INDEX IF NOT EXISTS idx_district_hourly_district_date 
    ON fact_weather_district_hourly(district_name_vi, ts_utc DESC);

-- District daily indexes
CREATE INDEX IF NOT EXISTS idx_district_daily_date 
    ON fact_weather_district_daily(date DESC);
CREATE INDEX IF NOT EXISTS idx_district_daily_district_date 
    ON fact_weather_district_daily(district_name_vi, date DESC);

-- City hourly indexes
CREATE INDEX IF NOT EXISTS idx_city_hourly_ts_utc 
    ON fact_weather_city_hourly(ts_utc DESC);

-- City daily indexes
CREATE INDEX IF NOT EXISTS idx_city_daily_date 
    ON fact_weather_city_daily(date DESC);
