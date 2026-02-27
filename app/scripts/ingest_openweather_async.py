from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import aiohttp
import aqi
from psycopg2.extras import execute_values

from app.core.key_manager import OpenWeatherKeyManager
from app.core.logging_config import get_logger, setup_logging
from app.db.connection import get_db_connection

setup_logging()
logger = get_logger(__name__)


def calculate_vn_aqi_from_pm25(pm25_concentration):
    """Calculate Vietnam AQI from PM2.5 concentration using python-aqi library."""
    try:
        myaqi = aqi.to_iaqi(aqi.POLLUTANT_PM25, str(pm25_concentration))
        return int(myaqi)
    except Exception:
        return None

class OpenWeatherAsyncIngestor:
    """Async ingestor for OpenWeather APIs with multi-key rotation and service awareness."""

    def __init__(self, concurrency: int = 15):
        self.key_manager = OpenWeatherKeyManager()
        self.semaphore = asyncio.Semaphore(concurrency)
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.onecall_url = "https://api.openweathermap.org/data/3.0/onecall"

    async def fetch_json(self, session: aiohttp.ClientSession, url: str, params: Dict[str, Any], service: str) -> Optional[Dict[str, Any]]:
        """Fetch JSON from API with key rotation, service blacklisting, and retry logic.

        Key exhaustion handling: when all keys are exhausted (RuntimeError),
        the method waits for the minute window to reset then retries,
        instead of giving up immediately.
        """
        max_retries = 5
        for attempt in range(max_retries):
            api_key = None
            try:
                # Acquire semaphore FIRST to limit concurrency,
                # then get key inside to avoid exhausting the minute quota all at once
                async with self.semaphore:
                    try:
                        api_key = self.key_manager.get_key(service=service)
                    except RuntimeError:
                        # All keys exhausted — release semaphore, wait, then retry
                        raise  # caught by outer RuntimeError handler

                    current_params = {**params, "appid": api_key}
                    async with session.get(url, params=current_params, timeout=25) as resp:
                        if resp.status == 429:
                            self.key_manager.report_failure(api_key, 429, service=service)
                            logger.warning(f"Key {api_key[:8]} hit rate limit (429). Retrying...")
                            await asyncio.sleep(2 ** attempt)
                            continue

                        if resp.status == 401:
                            self.key_manager.report_failure(api_key, 401, service=service)
                            logger.error(f"Key {api_key[:8]} is UNAUTHORIZED (401) for {service}. Blacklisting.")
                            continue

                        resp.raise_for_status()
                        data = await resp.json()
                        self.key_manager.report_success(api_key)
                        return data
            except RuntimeError:
                # Keys exhausted — wait for minute window to reset, then retry
                wait_s = 15 + attempt * 10  # 15s, 25s, 35s, 45s, 55s
                if attempt == 0:
                    logger.warning(f"All keys exhausted for '{service}'. Waiting {wait_s}s before retry...")
                await asyncio.sleep(wait_s)
                continue
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed for {url} (Key: {api_key[:8] if api_key else 'None'}): {e}")
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(1)
        return None

    def _get_ward_list(self) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT ward_id, lat, lon, ward_name_vi FROM dim_ward")
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]
        finally:
            conn.close()

    async def run_nowcast(self):
        """Job 1: Air current + OneCall current (Hourly)."""
        wards = self._get_ward_list()
        logger.info(f"Starting Nowcast job for {len(wards)} wards...")
        
        async with aiohttp.ClientSession() as session:
            air_tasks = [self.fetch_json(session, f"{self.base_url}/air_pollution", {"lat": w["lat"], "lon": w["lon"]}, "pollution") for w in wards]
            weather_tasks = [self.fetch_json(session, self.onecall_url, {"lat": w["lat"], "lon": w["lon"], "units": "metric", "exclude": "minutely,hourly,daily,alerts"}, "onecall") for w in wards]
            
            air_results = await asyncio.gather(*air_tasks)
            weather_results = await asyncio.gather(*weather_tasks)

        # Process and Bulk Upsert
        air_records = []
        for ward, data in zip(wards, air_results):
            if data and data.get("list"):
                item = data["list"][0]
                comp = item["components"]
                pm25_val = comp.get("pm2_5")
                air_records.append((
                    ward["ward_id"], datetime.fromtimestamp(item["dt"], tz=timezone.utc),
                    item["main"]["aqi"], calculate_vn_aqi_from_pm25(pm25_val),
                    comp.get("co"), comp.get("no"), comp.get("no2"),
                    comp.get("o3"), comp.get("so2"), pm25_val, comp.get("pm10"),
                    comp.get("nh3"), 'current', 'openweather', 'nowcast'
                ))

        weather_records = []
        for ward, data in zip(wards, weather_results):
            if data and data.get("current"):
                curr = data["current"]
                w_info = curr["weather"][0] if curr.get("weather") else {}
                weather_records.append((
                    ward["ward_id"], datetime.fromtimestamp(curr["dt"], tz=timezone.utc),
                    curr.get("temp"), curr.get("feels_like"), curr.get("pressure"),
                    curr.get("humidity"), curr.get("dew_point"), curr.get("clouds"),
                    curr.get("wind_speed"), curr.get("wind_deg"), curr.get("wind_gust"),
                    curr.get("visibility"), curr.get("uvi"), None, # pop is not in current
                    (curr.get("rain") or {}).get("1h"), w_info.get("main"), w_info.get("description"),
                    'current', 'openweather', 'nowcast'
                ))

        self._bulk_upsert_air(air_records, priority='current')
        self._bulk_upsert_weather_hourly(weather_records, priority='current')

    def _bulk_upsert_air(self, records: List[tuple], priority: str):
        if not records: return
        conn = get_db_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    # Priority logic: history > current > forecast
                    # We only update if EXCLUDED record has higher or equal priority
                    condition = ""
                    if priority == 'forecast':
                        condition = "AND fact_air_pollution_hourly.data_kind NOT IN ('current', 'history')"
                    elif priority == 'current':
                        condition = "AND fact_air_pollution_hourly.data_kind <> 'history'"

                    query = f"""
                        INSERT INTO fact_air_pollution_hourly (
                            ward_id, ts_utc, aqi, aqi_vn, co, no, no2, o3, so2, pm2_5, pm10, nh3,
                            data_kind, source, source_job
                        ) VALUES %s
                        ON CONFLICT (ward_id, ts_utc) DO UPDATE SET
                            aqi = EXCLUDED.aqi, aqi_vn = EXCLUDED.aqi_vn,
                            co = EXCLUDED.co, no = EXCLUDED.no, no2 = EXCLUDED.no2,
                            o3 = EXCLUDED.o3, so2 = EXCLUDED.so2, pm2_5 = EXCLUDED.pm2_5, pm10 = EXCLUDED.pm10,
                            nh3 = EXCLUDED.nh3, data_kind = EXCLUDED.data_kind,
                            source = EXCLUDED.source, source_job = EXCLUDED.source_job, ingested_at = NOW()
                        WHERE TRUE {condition}
                    """
                    execute_values(cur, query, records)
            logger.info(f"Bulk upserted {len(records)} air records ({priority})")
        finally:
            conn.close()

    def _bulk_upsert_weather_hourly(self, records: List[tuple], priority: str):
        if not records: return
        conn = get_db_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    condition = ""
                    if priority == 'forecast':
                        condition = "AND fact_weather_hourly.data_kind NOT IN ('current', 'history')"
                    elif priority == 'current':
                        condition = "AND fact_weather_hourly.data_kind <> 'history'"

                    query = f"""
                        INSERT INTO fact_weather_hourly (
                            ward_id, ts_utc, temp, feels_like, pressure, humidity, dew_point, clouds,
                            wind_speed, wind_deg, wind_gust, visibility, uvi, pop, rain_1h,
                            weather_main, weather_description, data_kind, source, source_job
                        ) VALUES %s
                        ON CONFLICT (ward_id, ts_utc) DO UPDATE SET
                            temp = EXCLUDED.temp, feels_like = EXCLUDED.feels_like, pressure = EXCLUDED.pressure,
                            humidity = EXCLUDED.humidity, dew_point = EXCLUDED.dew_point, clouds = EXCLUDED.clouds,
                            wind_speed = EXCLUDED.wind_speed, wind_deg = EXCLUDED.wind_deg, 
                            wind_gust = EXCLUDED.wind_gust, visibility = EXCLUDED.visibility, 
                            uvi = EXCLUDED.uvi, pop = EXCLUDED.pop, rain_1h = EXCLUDED.rain_1h,
                            weather_main = EXCLUDED.weather_main, weather_description = EXCLUDED.weather_description,
                            data_kind = EXCLUDED.data_kind, source = EXCLUDED.source,
                            source_job = EXCLUDED.source_job, ingested_at = NOW()
                        WHERE TRUE {condition}
                    """
                    execute_values(cur, query, records)
            logger.info(f"Bulk upserted {len(records)} weather hourly records ({priority})")
        finally:
            conn.close()

    async def run_forecast(self):
        """Job 2: Air forecast + OneCall full (6-hourly)."""
        wards = self._get_ward_list()
        logger.info(f"Starting Forecast job for {len(wards)} wards...")
        
        async with aiohttp.ClientSession() as session:
            air_tasks = [self.fetch_json(session, f"{self.base_url}/air_pollution/forecast", {"lat": w["lat"], "lon": w["lon"]}, "pollution") for w in wards]
            weather_tasks = [self.fetch_json(session, self.onecall_url, {"lat": w["lat"], "lon": w["lon"], "units": "metric", "exclude": "minutely,alerts"}, "onecall") for w in wards]
            
            air_results = await asyncio.gather(*air_tasks)
            weather_results = await asyncio.gather(*weather_tasks)

        air_records = []
        for ward, data in zip(wards, air_results):
            if data and data.get("list"):
                for item in data["list"]:
                    comp = item["components"]
                    pm25_val = comp.get("pm2_5")
                    air_records.append((
                        ward["ward_id"], datetime.fromtimestamp(item["dt"], tz=timezone.utc),
                        item["main"]["aqi"], calculate_vn_aqi_from_pm25(pm25_val),
                        comp.get("co"), comp.get("no"), comp.get("no2"),
                        comp.get("o3"), comp.get("so2"), pm25_val, comp.get("pm10"),
                        comp.get("nh3"), 'forecast', 'openweather', 'forecast'
                    ))

        weather_hourly_records = []
        weather_daily_records = []
        for ward, data in zip(wards, weather_results):
            if not data: continue
            
            if data.get("hourly"):
                for h in data["hourly"]:
                    w_info = h["weather"][0] if h.get("weather") else {}
                    weather_hourly_records.append((
                        ward["ward_id"], datetime.fromtimestamp(h["dt"], tz=timezone.utc),
                        h.get("temp"), h.get("feels_like"), h.get("pressure"),
                        h.get("humidity"), h.get("dew_point"), h.get("clouds"),
                        h.get("wind_speed"), h.get("wind_deg"), h.get("wind_gust"),
                        h.get("visibility"), h.get("uvi"), h.get("pop"),
                        (h.get("rain") or {}).get("1h"), w_info.get("main"), w_info.get("description"),
                        'forecast', 'openweather', 'forecast'
                    ))
            
            if data.get("daily"):
                for d in data["daily"]:
                    weather_daily_records.append((
                        ward["ward_id"], datetime.fromtimestamp(d["dt"], tz=timezone.utc).date(),
                        d["temp"].get("min"), d["temp"].get("max"), d.get("humidity"),
                        d.get("wind_speed"), d.get("pop"), d.get("rain"), d.get("uvi"),
                        d.get("summary"), 'forecast', 'openweather', 'forecast'
                    ))

        self._bulk_upsert_air(air_records, priority='forecast')
        self._bulk_upsert_weather_hourly(weather_hourly_records, priority='forecast')
        self._bulk_upsert_weather_daily(weather_daily_records)

    def _bulk_upsert_weather_daily(self, records: List[tuple]):
        if not records: return
        conn = get_db_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    query = """
                        INSERT INTO fact_weather_daily (
                            ward_id, date, temp_min, temp_max, humidity, wind_speed, pop, rain_total,
                            uvi, summary, data_kind, source, source_job
                        ) VALUES %s
                        ON CONFLICT (ward_id, date) DO UPDATE SET
                            temp_min = EXCLUDED.temp_min, temp_max = EXCLUDED.temp_max,
                            humidity = EXCLUDED.humidity, wind_speed = EXCLUDED.wind_speed,
                            pop = EXCLUDED.pop, rain_total = EXCLUDED.rain_total,
                            uvi = EXCLUDED.uvi, summary = EXCLUDED.summary,
                            data_kind = EXCLUDED.data_kind, source = EXCLUDED.source,
                            source_job = EXCLUDED.source_job, ingested_at = NOW()
                    """
                    execute_values(cur, query, records)
            logger.info(f"Bulk upserted {len(records)} weather daily records")
        finally:
            conn.close()

    async def run_history_backfill(self, days: int = 14):
        """Job 3: Backfill Air Pollution history (One-time).

        OpenWeather Air Pollution History API has no hard per-request day limit,
        but very large ranges may time out. We chunk by 5-day windows to be safe.
        Each chunk is mapped back to its ward via a parallel `task_meta` list.
        """
        wards = self._get_ward_list()
        logger.info(f"Starting History Backfill for {len(wards)} wards, last {days} days...")

        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)

        # Build tasks + parallel metadata list so we can map results → ward_id
        tasks: List[Any] = []
        task_meta: List[Dict[str, Any]] = []  # [{"ward_id": ..., ...}, ...]

        async with aiohttp.ClientSession() as session:
            for w in wards:
                curr_start = start_dt
                while curr_start < end_dt:
                    curr_end = min(curr_start + timedelta(days=5), end_dt)
                    params = {
                        "lat": w["lat"], "lon": w["lon"],
                        "start": int(curr_start.timestamp()),
                        "end": int(curr_end.timestamp())
                    }
                    tasks.append(
                        self.fetch_json(session, f"{self.base_url}/air_pollution/history", params, "pollution")
                    )
                    task_meta.append({"ward_id": w["ward_id"]})
                    curr_start = curr_end

            logger.info(f"History backfill: {len(tasks)} API chunks across {len(wards)} wards")
            results = await asyncio.gather(*tasks)

        # Parse results using task_meta for ward_id mapping
        air_records = []
        for meta, data in zip(task_meta, results):
            if not data or not data.get("list"):
                continue
            ward_id = meta["ward_id"]
            for item in data["list"]:
                comp = item["components"]
                pm25_val = comp.get("pm2_5")
                air_records.append((
                    ward_id,
                    datetime.fromtimestamp(item["dt"], tz=timezone.utc),
                    item["main"]["aqi"],
                    calculate_vn_aqi_from_pm25(pm25_val),
                    comp.get("co"), comp.get("no"), comp.get("no2"),
                    comp.get("o3"), comp.get("so2"), pm25_val, comp.get("pm10"),
                    comp.get("nh3"), 'history', 'openweather', 'history_backfill'
                ))

        logger.info(f"History backfill: parsed {len(air_records)} air records from {len(results)} API responses")
        self._bulk_upsert_air(air_records, priority='history')

    async def run_smoke_test(self):
        """Mode: smoke_test. Test key 4 (One Call 3.0) with 1 ward to verify activation."""
        wards = self._get_ward_list()
        if not wards:
            return
        w = wards[0]
        logger.info(f"Starting smoke test for key 4 (OneCall) using ward: {w['ward_name_vi']}")
        
        async with aiohttp.ClientSession() as session:
            api_key = self.key_manager.get_key(service="onecall")
            params = {
                "lat": w["lat"],
                "lon": w["lon"],
                "units": "metric",
                "exclude": "minutely,hourly,daily,alerts",
                "appid": api_key,
            }

            async with session.get(self.onecall_url, params=params) as resp:
                if resp.status == 200:
                    logger.info("Result for key_4: SUCCESS (200 OK)")
                elif resp.status == 401:
                    logger.error("Result for key_4: FAILED (401 Unauthorized) - One Call 3.0 not activated")
                elif resp.status == 429:
                    logger.warning("Result for key_4: FAILED (429 Rate Limit)")
                else:
                    logger.error(f"Result for key_4: FAILED ({resp.status})")

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "nowcast"
    ingestor = OpenWeatherAsyncIngestor()
    if mode == "nowcast":
        asyncio.run(ingestor.run_nowcast())
    elif mode == "forecast":
        asyncio.run(ingestor.run_forecast())
    elif mode in ("history", "history_backfill"):
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 14
        asyncio.run(ingestor.run_history_backfill(days=days))
    elif mode == "smoke_test":
        asyncio.run(ingestor.run_smoke_test())
    else:
        print(f"Unknown mode: {mode}. Available: nowcast, forecast, history [days], smoke_test")
