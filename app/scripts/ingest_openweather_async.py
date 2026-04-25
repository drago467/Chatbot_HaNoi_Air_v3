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

        # Not the leader — just wait for the leader to finish.
        # Note: asyncio.Event.wait() returns immediately if already set,
        # so there's no race condition between is_set() check and wait().
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
                        # All keys exhausted — raise exits the semaphore context
                        # manager (releasing it correctly), then caught by outer handler
                        raise

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
            release_connection(conn)

    def _parse_nowcast_record(self, ward: Dict, data: Dict) -> Optional[tuple]:
        """Parse a single nowcast API response into a DB record tuple."""
        if not data or not data.get("current"):
            return None
        curr = data["current"]
        w_info = curr["weather"][0] if curr.get("weather") else {}
        return (
            ward["ward_id"], datetime.fromtimestamp(curr["dt"], tz=timezone.utc),
            curr.get("temp"), curr.get("feels_like"), curr.get("pressure"),
            curr.get("humidity"), curr.get("dew_point"), curr.get("clouds"),
            curr.get("wind_speed"), curr.get("wind_deg"), curr.get("wind_gust"),
            curr.get("visibility"), curr.get("uvi"), None,  # pop is not in current
            (curr.get("rain") or {}).get("1h"), w_info.get("main"), w_info.get("description"),
            'current', 'openweather', 'nowcast'
        )

    async def run_nowcast(self):
        """Job 1: OneCall current weather (Hourly)."""
        wards = self._get_ward_list()
        logger.info(f"Starting Nowcast job for {len(wards)} wards...")

        nowcast_params = lambda w: {"lat": w["lat"], "lon": w["lon"], "units": "metric", "exclude": "minutely,hourly,daily,alerts"}

        async with aiohttp.ClientSession() as session:
            weather_tasks = [self.fetch_json(session, self.onecall_url, nowcast_params(w), "onecall") for w in wards]
            weather_results = await asyncio.gather(*weather_tasks)

        weather_records = []
        failed_wards = []
        for ward, data in zip(wards, weather_results):
            rec = self._parse_nowcast_record(ward, data)
            if rec:
                weather_records.append(rec)
            else:
                failed_wards.append(ward)

        # Retry failed wards (up to 2 additional rounds)
        for retry_round in range(1, 3):
            if not failed_wards:
                break
            logger.warning(f"Nowcast retry round {retry_round}: {len(failed_wards)}/{len(wards)} wards failed, retrying...")
            await asyncio.sleep(3 * retry_round)  # backoff between retries
            async with aiohttp.ClientSession() as session:
                retry_tasks = [self.fetch_json(session, self.onecall_url, nowcast_params(w), "onecall") for w in failed_wards]
                retry_results = await asyncio.gather(*retry_tasks)
            still_failed = []
            for ward, data in zip(failed_wards, retry_results):
                rec = self._parse_nowcast_record(ward, data)
                if rec:
                    weather_records.append(rec)
                else:
                    still_failed.append(ward)
            failed_wards = still_failed

        # Completeness report
        success_count = len(wards) - len(failed_wards)
        logger.info(f"Nowcast complete: {success_count}/{len(wards)} wards OK")
        if failed_wards:
            names = [w["ward_name_vi"] for w in failed_wards[:10]]
            logger.error(f"Nowcast INCOMPLETE: {len(failed_wards)} wards still failed after retries: {names}")

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
            release_connection(conn)

    def _parse_forecast_records(self, ward: Dict, data: Dict) -> tuple[list, list]:
        """Parse a single forecast API response into hourly + daily record lists."""
        hourly_records = []
        daily_records = []
        if not data:
            return hourly_records, daily_records

        if data.get("hourly"):
            for h in data["hourly"]:
                w_info = h["weather"][0] if h.get("weather") else {}
                hourly_records.append((
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
                daily_records.append((
                    ward["ward_id"], datetime.fromtimestamp(d["dt"], tz=ICT).date(),
                    d["temp"].get("min"), d["temp"].get("max"), d["temp"].get("day"),
                    d["temp"].get("morn"), d["temp"].get("day"), d["temp"].get("eve"), d["temp"].get("night"),
                    feels_like.get("morn"), feels_like.get("day"), feels_like.get("eve"), feels_like.get("night"),
                    d.get("humidity"), d.get("pressure"), d.get("dew_point"),
                    d.get("wind_speed"), d.get("wind_deg"), d.get("wind_gust"),
                    d.get("clouds"), d.get("pop"), d.get("rain"), d.get("uvi"),
                    w_info.get("main"), w_info.get("description"), d.get("summary"),
                    datetime.fromtimestamp(d["sunrise"], tz=timezone.utc) if d.get("sunrise") else None,
                    datetime.fromtimestamp(d["sunset"], tz=timezone.utc) if d.get("sunset") else None,
                    'forecast', 'openweather', 'forecast'
                ))

        return hourly_records, daily_records

    async def run_forecast(self):
        """Job 2: OneCall full forecast (6-hourly)."""
        wards = self._get_ward_list()
        logger.info(f"Starting Forecast job for {len(wards)} wards...")

        forecast_params = lambda w: {"lat": w["lat"], "lon": w["lon"], "units": "metric", "exclude": "minutely,alerts"}

        async with aiohttp.ClientSession() as session:
            weather_tasks = [self.fetch_json(session, self.onecall_url, forecast_params(w), "onecall") for w in wards]
            weather_results = await asyncio.gather(*weather_tasks)

        weather_hourly_records = []
        weather_daily_records = []
        failed_wards = []
        for ward, data in zip(wards, weather_results):
            if data is None:
                failed_wards.append(ward)
                continue
            hourly, daily = self._parse_forecast_records(ward, data)
            weather_hourly_records.extend(hourly)
            weather_daily_records.extend(daily)

        # Retry failed wards (up to 2 additional rounds)
        for retry_round in range(1, 3):
            if not failed_wards:
                break
            logger.warning(f"Forecast retry round {retry_round}: {len(failed_wards)}/{len(wards)} wards failed, retrying...")
            await asyncio.sleep(3 * retry_round)
            async with aiohttp.ClientSession() as session:
                retry_tasks = [self.fetch_json(session, self.onecall_url, forecast_params(w), "onecall") for w in failed_wards]
                retry_results = await asyncio.gather(*retry_tasks)
            still_failed = []
            for ward, data in zip(failed_wards, retry_results):
                if data is None:
                    still_failed.append(ward)
                    continue
                hourly, daily = self._parse_forecast_records(ward, data)
                weather_hourly_records.extend(hourly)
                weather_daily_records.extend(daily)
            failed_wards = still_failed

        # Completeness report
        success_count = len(wards) - len(failed_wards)
        logger.info(f"Forecast complete: {success_count}/{len(wards)} wards OK")
        if failed_wards:
            names = [w["ward_name_vi"] for w in failed_wards[:10]]
            logger.error(f"Forecast INCOMPLETE: {len(failed_wards)} wards still failed after retries: {names}")

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
            release_connection(conn)

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
        Uses async batched calls (like run_forecast) for performance.
        """
        logger.info(f"Starting Weather History Backfill for {len(wards)} wards, last {days} days...")

        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)

        # Build all (ward, noon_dt) pairs
        tasks_meta = []
        for w in wards:
            current = start_dt
            while current < end_dt:
                noon = current.replace(hour=12, minute=0, second=0, microsecond=0)
                tasks_meta.append((w, noon))
                current += timedelta(days=1)

        logger.info(f"History: {len(tasks_meta)} API calls ({len(wards)} wards x {days} days)")

        # Fetch all in parallel (semaphore in fetch_json limits concurrency)
        async with aiohttp.ClientSession() as session:
            api_tasks = [
                self.fetch_json(
                    session,
                    f"{self.onecall_url}/timemachine",
                    {"lat": w["lat"], "lon": w["lon"], "dt": int(noon.timestamp()), "units": "metric"},
                    "timemachine"
                )
                for w, noon in tasks_meta
            ]
            results = await asyncio.gather(*api_tasks)

        weather_records = []
        failed_count = 0
        for (ward, noon), data in zip(tasks_meta, results):
            if not data or not data.get("data"):
                failed_count += 1
                continue
            item = data["data"][0]
            w_info = item.get("weather", [{}])[0] if item.get("weather") else {}
            weather_records.append((
                ward["ward_id"],
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
                (item.get("rain") or {}).get("1h"),
                w_info.get("main"),
                w_info.get("description"),
                'history',
                'openweather',
                'weather_history_backfill'
            ))

        success_count = len(tasks_meta) - failed_count
        logger.info(f"Weather History Backfill: {success_count}/{len(tasks_meta)} calls OK, {len(weather_records)} records collected")
        if failed_count:
            logger.warning(f"History: {failed_count} API calls failed (some wards/days may be incomplete)")

        if weather_records:
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

def run_ingest(include_history: bool = False, history_days: int = 7) -> None:
    """Sync wrapper để gọi từ FastAPI /jobs/ingest sau R15 (gỡ Celery).

    Block đến khi mọi step xong. Chạy current + forecast luôn, thêm history
    nếu include_history=True.
    """
    async def _main():
        ingestor = OpenWeatherAsyncIngestor()
        if include_history and history_days > 0:
            await ingestor.run_history_backfill(days=history_days)
        await ingestor.run_nowcast()
        await ingestor.run_forecast()

    asyncio.run(_main())


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="OpenWeather Weather Ingestion Script")
    parser.add_argument("--days", type=int, default=0, help="Number of days for history backfill (default: 0 = skip history)")
    parser.add_argument("--history-only", action="store_true", help="Only ingest history data")
    parser.add_argument("--current-only", action="store_true", help="Only ingest current data")
    parser.add_argument("--forecast-only", action="store_true", help="Only ingest forecast data")

    args = parser.parse_args()

    do_history  = args.days > 0 and not (args.current_only or args.forecast_only)
    do_current  = not (args.history_only or args.forecast_only)
    do_forecast = not (args.history_only or args.current_only)

    # If --history-only is set but --days is 0, default to 14 days
    if args.history_only and args.days == 0:
        args.days = 14
        do_history = True

    print(f"Mode: Weather Ingestion")
    print(f"History: {do_history} (days={args.days})")
    print(f"Current: {do_current}")
    print(f"Forecast: {do_forecast}")
    print()

    async def main():
        ingestor = OpenWeatherAsyncIngestor()

        if do_history:
            print("=== STEP 1: HISTORY BACKFILL ===")
            await ingestor.run_history_backfill(days=args.days)

        if do_current:
            print("\n=== STEP 2: CURRENT DATA ===")
            await ingestor.run_nowcast()

        if do_forecast:
            print("\n=== STEP 3: FORECAST DATA ===")
            await ingestor.run_forecast()

    asyncio.run(main())
