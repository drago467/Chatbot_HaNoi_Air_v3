from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

ICT = timezone(timedelta(hours=7))
from typing import Any, Dict, List, Optional

import aiohttp
from psycopg2.extras import execute_values

from app.core.key_manager import OpenWeatherKeyManager
from app.core.logging_config import get_logger, setup_logging
from app.db.connection import get_db_connection, release_connection
from app.scripts.aggregate_weather import (
    aggregate_district_hourly,
    aggregate_city_hourly,
    aggregate_district_daily,
    aggregate_city_daily,
    _WIND_DEG_CIRCULAR_MEAN,
)

setup_logging()
logger = get_logger(__name__)


class OpenWeatherAsyncIngestor:
    """Async ingestor for OpenWeather APIs with multi-key rotation and service awareness."""

    def __init__(self, concurrency: int = 15):
        self.key_manager = OpenWeatherKeyManager()
        self.semaphore = asyncio.Semaphore(concurrency)
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.onecall_url = "https://api.openweathermap.org/data/3.0/onecall"

        # Shared state for coordinated key-exhaustion waiting.
        # When keys are exhausted, ONE coroutine becomes the "leader":
        # it logs the warning, sleeps, then wakes everyone up.
        # All other coroutines just await the same event — no duplicate logs.
        self._key_wait_lock = asyncio.Lock()
        self._key_ready_event = asyncio.Event()
        self._key_ready_event.set()  # starts as "ready"

    async def _wait_for_keys(self, service: str) -> None:
        """Coordinate waiting when all keys are exhausted.

        Only the first coroutine that detects exhaustion becomes the "leader":
        it logs the warning once, sleeps for the computed wait time, then signals
        all waiting coroutines to resume. Others simply await the event.
        """
        if self._key_ready_event.is_set():
            # No one is waiting yet — try to become the leader
            async with self._key_wait_lock:
                if self._key_ready_event.is_set():
                    # We are the leader
                    self._key_ready_event.clear()
                    wait_s = self.key_manager.get_wait_seconds(service=service)
                    wait_s = max(wait_s, 5.0)  # at least 5s
                    logger.warning(
                        f"All keys exhausted for '{service}'. "
                        f"Waiting {wait_s:.0f}s for rate limit window to reset..."
                    )
                    await asyncio.sleep(wait_s)
                    self._key_ready_event.set()
                    return

        # Not the leader — just wait for the leader to finish
        await self._key_ready_event.wait()

    async def fetch_json(self, session: aiohttp.ClientSession, url: str, params: Dict[str, Any], service: str) -> Optional[Dict[str, Any]]:
        """Fetch JSON from API with key rotation, service blacklisting, and retry logic.

        Key exhaustion handling: when all keys are exhausted (RuntimeError),
        coroutines coordinate via a shared event so that only ONE logs the
        warning and sleeps; all others silently await the same signal.
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
                # Keys exhausted — coordinate with other coroutines
                await self._wait_for_keys(service)
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
        """Job 1: OneCall current weather (Hourly)."""
        wards = self._get_ward_list()
        logger.info(f"Starting Nowcast job for {len(wards)} wards...")

        async with aiohttp.ClientSession() as session:
            weather_tasks = [self.fetch_json(session, self.onecall_url, {"lat": w["lat"], "lon": w["lon"], "units": "metric", "exclude": "minutely,hourly,daily,alerts"}, "onecall") for w in wards]
            weather_results = await asyncio.gather(*weather_tasks)

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

        self._bulk_upsert_weather_hourly(weather_records, priority='current')

        # === AGGREGATION: District + City level (after current data) ===
        logger.info("Starting aggregation for current data...")
        try:
            district_result = aggregate_district_hourly('current')
            city_result = aggregate_city_hourly('current')
            for label, res in [("district_hourly", district_result), ("city_hourly", city_result)]:
                if res.get('status') == 'error':
                    logger.error(f"Aggregation {label} failed: {res.get('message')}")
            logger.info(f"Aggregated hourly: current - district: {district_result.get('records_upserted', 0)}, city: {city_result.get('records_upserted', 0)}")
            logger.info("Aggregation completed for current data")
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")

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
        """Job 2: OneCall full forecast (6-hourly)."""
        wards = self._get_ward_list()
        logger.info(f"Starting Forecast job for {len(wards)} wards...")

        async with aiohttp.ClientSession() as session:
            weather_tasks = [self.fetch_json(session, self.onecall_url, {"lat": w["lat"], "lon": w["lon"], "units": "metric", "exclude": "minutely,alerts"}, "onecall") for w in wards]
            weather_results = await asyncio.gather(*weather_tasks)

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
                    w_info = d["weather"][0] if d.get("weather") else {}
                    feels_like = d.get("feels_like", {})
                    weather_daily_records.append((
                        ward["ward_id"], datetime.fromtimestamp(d["dt"], tz=ICT).date(),
                        # Temperature
                        d["temp"].get("min"), d["temp"].get("max"), d["temp"].get("day"),  # temp_min, temp_max, temp_avg (=day)
                        d["temp"].get("morn"), d["temp"].get("day"), d["temp"].get("eve"), d["temp"].get("night"),  # temp_morn, temp_day, temp_eve, temp_night
                        # Feels like
                        feels_like.get("morn"), feels_like.get("day"), feels_like.get("eve"), feels_like.get("night"),
                        # Other weather
                        d.get("humidity"), d.get("pressure"), d.get("dew_point"),
                        d.get("wind_speed"), d.get("wind_deg"), d.get("wind_gust"),
                        d.get("clouds"), d.get("pop"), d.get("rain"), d.get("uvi"),
                        # Weather condition
                        w_info.get("main"), w_info.get("description"), d.get("summary"),
                        # Sun times
                        datetime.fromtimestamp(d["sunrise"], tz=timezone.utc) if d.get("sunrise") else None,
                        datetime.fromtimestamp(d["sunset"], tz=timezone.utc) if d.get("sunset") else None,
                        # Metadata
                        'forecast', 'openweather', 'forecast'
                    ))

        self._bulk_upsert_weather_hourly(weather_hourly_records, priority='forecast')
        self._bulk_upsert_weather_daily(weather_daily_records)

        # === AGGREGATION: District + City level (after forecast data) ===
        logger.info("Starting aggregation for forecast data...")
        try:
            # Aggregate hourly (forecast)
            for data_kind in ['forecast']:
                district_result = aggregate_district_hourly(data_kind)
                city_result = aggregate_city_hourly(data_kind)
                for label, res in [("district_hourly", district_result), ("city_hourly", city_result)]:
                    if res.get('status') == 'error':
                        logger.error(f"Aggregation {label} ({data_kind}) failed: {res.get('message')}")
                logger.info(f"Aggregated hourly: {data_kind} - district: {district_result.get('records_upserted', 0)}, city: {city_result.get('records_upserted', 0)}")

            # Aggregate daily (forecast)
            for data_kind in ['forecast']:
                district_result = aggregate_district_daily(data_kind)
                city_result = aggregate_city_daily(data_kind)
                for label, res in [("district_daily", district_result), ("city_daily", city_result)]:
                    if res.get('status') == 'error':
                        logger.error(f"Aggregation {label} ({data_kind}) failed: {res.get('message')}")
                logger.info(f"Aggregated daily: {data_kind} - district: {district_result.get('records_upserted', 0)}, city: {city_result.get('records_upserted', 0)}")

            logger.info("Aggregation completed for forecast data")
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")

    def _bulk_upsert_weather_daily(self, records: List[tuple], priority='forecast'):
        if not records: return

        # Determine condition based on priority
        condition = ""
        if priority == 'forecast':
            condition = "AND fact_weather_daily.data_kind NOT IN ('history')"

        conn = get_db_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    query = f"""
                        INSERT INTO fact_weather_daily (
                            ward_id, date,
                            temp_min, temp_max, temp_avg, temp_morn, temp_day, temp_eve, temp_night,
                            feels_like_morn, feels_like_day, feels_like_eve, feels_like_night,
                            humidity, pressure, dew_point,
                            wind_speed, wind_deg, wind_gust, clouds, pop, rain_total, uvi,
                            weather_main, weather_description, summary,
                            sunrise, sunset,
                            data_kind, source, source_job
                        ) VALUES %s
                        ON CONFLICT (ward_id, date) DO UPDATE SET
                            temp_min = EXCLUDED.temp_min, temp_max = EXCLUDED.temp_max,
                            temp_avg = EXCLUDED.temp_avg, temp_morn = EXCLUDED.temp_morn,
                            temp_day = EXCLUDED.temp_day, temp_eve = EXCLUDED.temp_eve, temp_night = EXCLUDED.temp_night,
                            feels_like_morn = EXCLUDED.feels_like_morn, feels_like_day = EXCLUDED.feels_like_day,
                            feels_like_eve = EXCLUDED.feels_like_eve, feels_like_night = EXCLUDED.feels_like_night,
                            humidity = EXCLUDED.humidity, pressure = EXCLUDED.pressure, dew_point = EXCLUDED.dew_point,
                            wind_speed = EXCLUDED.wind_speed, wind_deg = EXCLUDED.wind_deg,
                            wind_gust = EXCLUDED.wind_gust, clouds = EXCLUDED.clouds,
                            pop = EXCLUDED.pop, rain_total = EXCLUDED.rain_total, uvi = EXCLUDED.uvi,
                            weather_main = EXCLUDED.weather_main, weather_description = EXCLUDED.weather_description,
                            summary = EXCLUDED.summary, sunrise = EXCLUDED.sunrise, sunset = EXCLUDED.sunset,
                            data_kind = EXCLUDED.data_kind, source = EXCLUDED.source,
                            source_job = EXCLUDED.source_job, ingested_at = NOW()
                        WHERE TRUE {condition}
                    """
                    execute_values(cur, query, records)
            logger.info(f"Bulk upserted {len(records)} weather daily records (priority={priority})")
        finally:
            conn.close()

    async def run_history_backfill(self, days: int = 14):
        """Job 3: Backfill Weather history (One-time).

        Args:
            days: Number of days to backfill (default 14)

        Data source:
        - Weather: /onecall/timemachine (1 call per day per ward)

        Weather history is needed for chatbot to answer questions about past weather:
        - "Hôm qua thời tiết thế nào?"
        - "Tuần trước có mưa không?"
        """
        wards = self._get_ward_list()
        logger.info(f"Starting History Backfill for {len(wards)} wards, last {days} days...")

        await self._fetch_weather_history(wards, days)

    async def _fetch_weather_history(self, wards: List[Dict], days: int = 14):
        """Fetch weather history using /onecall/timemachine API.

        Note: Timemachine returns 1 record per call (for a specific hour).
        For efficiency, we fetch once per day (at noon) as proxy for daily weather.
        """
        import math

        logger.info(f"Starting Weather History Backfill for {len(wards)} wards, last {days} days...")

        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)

        weather_records = []

        async with aiohttp.ClientSession() as session:
            # Fetch at noon each day as proxy for daily weather
            for w in wards:
                current = start_dt
                attempts = 0
                max_attempts = days

                while current < end_dt and attempts < max_attempts:
                    # Use noon of each day as the timestamp (middle of day)
                    noon = current.replace(hour=12, minute=0, second=0, microsecond=0)
                    params = {
                        "lat": w["lat"],
                        "lon": w["lon"],
                        "dt": int(noon.timestamp()),
                        "units": "metric"
                    }

                    try:
                        data = await self.fetch_json(
                            session,
                            f"{self.onecall_url}/timemachine",
                            params,
                            "timemachine"
                        )

                        if data and data.get("data"):
                            item = data["data"][0]
                            w_info = item.get("weather", [{}])[0] if item.get("weather") else {}

                            weather_records.append((
                                w["ward_id"],
                                datetime.fromtimestamp(item["dt"], tz=timezone.utc),
                                item.get("temp"),
                                item.get("feels_like"),
                                item.get("pressure"),
                                item.get("humidity"),
                                item.get("dew_point"),
                                item.get("clouds"),
                                item.get("wind_speed"),
                                item.get("wind_deg"),
                                item.get("wind_gust"),
                                item.get("visibility"),
                                item.get("uvi"),
                                item.get("pop"),
                                item.get("rain", {}).get("1h") if item.get("rain") else None,
                                w_info.get("main"),
                                w_info.get("description"),
                                'history',  # data_kind = history
                                'openweather',
                                'weather_history_backfill'
                            ))
                    except Exception as e:
                        logger.warning(f"Error fetching weather history for {w['ward_id']} at {noon}: {e}")

                    current += timedelta(days=1)
                    attempts += 1

        logger.info(f"Weather History Backfill: collected {len(weather_records)} records")

        if weather_records:
            # Use bulk upsert with history priority
            self._bulk_upsert_weather_hourly(weather_records, priority='history')
        else:
            logger.warning("No weather history records to insert")

        # Aggregate history to daily after fetching hourly history
        self._aggregate_history_to_daily()

    def _aggregate_history_to_daily(self):
        """Aggregate hourly history data into daily summary.

        This runs after weather history is ingested into fact_weather_hourly.
        It aggregates all history records into daily summaries and inserts
        into fact_weather_daily with data_kind='history'.

        This is needed because:
        - OpenWeather timemachine API only returns hourly data
        - We need daily aggregates for compare_with_yesterday() and historical queries
        """
        logger.info("Starting aggregation of hourly history to daily...")

        conn = get_db_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    # Aggregate query: group hourly history by date
                    # Note: We use ts_utc at noon as proxy for daily temp
                    agg_sql = f"""
                        INSERT INTO fact_weather_daily (
                            ward_id, date,
                            temp_min, temp_max, temp_avg,
                            humidity, pressure, dew_point,
                            wind_speed, wind_deg,
                            clouds, pop, rain_total, uvi,
                            weather_main, weather_description,
                            data_kind, source, source_job
                        )
                        SELECT
                            ward_id,
                            (ts_utc AT TIME ZONE 'Asia/Ho_Chi_Minh')::date as date,
                            MIN(temp) as temp_min,
                            MAX(temp) as temp_max,
                            AVG(temp) as temp_avg,
                            AVG(humidity)::int as humidity,
                            AVG(pressure)::int as pressure,
                            AVG(dew_point) as dew_point,
                            AVG(wind_speed) as wind_speed,
                            {_WIND_DEG_CIRCULAR_MEAN} as wind_deg,
                            AVG(clouds)::int as clouds,
                            MAX(pop) as pop,
                            SUM(COALESCE(rain_1h, 0)) as rain_total,
                            MAX(uvi) as uvi,
                            (array_agg(weather_main ORDER BY ts_utc DESC))[1] as weather_main,
                            (array_agg(weather_description ORDER BY ts_utc DESC))[1] as weather_description,
                            'history',
                            'openweather',
                            'history_aggregation'
                        FROM fact_weather_hourly w
                        WHERE w.data_kind = 'history'
                          AND w.ts_utc > NOW() - INTERVAL '30 days'
                        GROUP BY w.ward_id, (w.ts_utc AT TIME ZONE 'Asia/Ho_Chi_Minh')::date
                        ON CONFLICT (ward_id, date) DO UPDATE SET
                            temp_min = EXCLUDED.temp_min,
                            temp_max = EXCLUDED.temp_max,
                            temp_avg = EXCLUDED.temp_avg,
                            humidity = EXCLUDED.humidity,
                            pressure = EXCLUDED.pressure,
                            dew_point = EXCLUDED.dew_point,
                            wind_speed = EXCLUDED.wind_speed,
                            wind_deg = EXCLUDED.wind_deg,
                            clouds = EXCLUDED.clouds,
                            pop = EXCLUDED.pop,
                            rain_total = EXCLUDED.rain_total,
                            uvi = EXCLUDED.uvi,
                            weather_main = EXCLUDED.weather_main,
                            weather_description = EXCLUDED.weather_description,
                            data_kind = EXCLUDED.data_kind,
                            source = EXCLUDED.source,
                            source_job = 'history_aggregation'
                    """
                    cur.execute(agg_sql)
                    affected = cur.rowcount
                    logger.info(f"Aggregated {affected} daily history records")
        except Exception as e:
            logger.error(f"Error aggregating history to daily: {e}")
        finally:
            release_connection(conn)

        # === AGGREGATION: District + City level (after history data) ===
        logger.info("Starting aggregation for history data...")
        try:
            # Aggregate hourly (history)
            for data_kind in ['history']:
                district_result = aggregate_district_hourly(data_kind)
                city_result = aggregate_city_hourly(data_kind)
                for label, res in [("district_hourly", district_result), ("city_hourly", city_result)]:
                    if res.get('status') == 'error':
                        logger.error(f"Aggregation {label} ({data_kind}) failed: {res.get('message')}")
                logger.info(f"Aggregated hourly: {data_kind} - district: {district_result.get('records_upserted', 0)}, city: {city_result.get('records_upserted', 0)}")

            # Aggregate daily (history)
            for data_kind in ['history']:
                district_result = aggregate_district_daily(data_kind)
                city_result = aggregate_city_daily(data_kind)
                for label, res in [("district_daily", district_result), ("city_daily", city_result)]:
                    if res.get('status') == 'error':
                        logger.error(f"Aggregation {label} ({data_kind}) failed: {res.get('message')}")
                logger.info(f"Aggregated daily: {data_kind} - district: {district_result.get('records_upserted', 0)}, city: {city_result.get('records_upserted', 0)}")

            logger.info("Aggregation completed for history data")
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")

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
    import argparse

    parser = argparse.ArgumentParser(description="OpenWeather Weather Ingestion Script")
    parser.add_argument("--days", type=int, default=14, help="Number of days for history backfill (default: 14)")
    parser.add_argument("--history-only", action="store_true", help="Only ingest history data")
    parser.add_argument("--current-only", action="store_true", help="Only ingest current data")
    parser.add_argument("--forecast-only", action="store_true", help="Only ingest forecast data")

    args = parser.parse_args()

    print(f"Mode: Weather Ingestion")
    print(f"History days: {args.days}")
    print(f"History: {not (args.current_only or args.forecast_only)}")
    print(f"Current: {not (args.history_only or args.forecast_only)}")
    print(f"Forecast: {not (args.history_only or args.current_only)}")
    print()

    ingestor = OpenWeatherAsyncIngestor()

    # Run in order: history -> current -> forecast
    if not args.current_only and not args.forecast_only:
        # History
        print("=== STEP 1: HISTORY BACKFILL ===")
        asyncio.run(ingestor.run_history_backfill(days=args.days))

    if not args.history_only and not args.forecast_only:
        # Current
        print("\n=== STEP 2: CURRENT DATA ===")
        asyncio.run(ingestor.run_nowcast())

    if not args.history_only and not args.current_only:
        # Forecast
        print("\n=== STEP 3: FORECAST DATA ===")
        asyncio.run(ingestor.run_forecast())
