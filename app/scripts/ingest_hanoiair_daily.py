"""HanoiAir Daily Ingestion Script.

Collects daily AQI/PM2.5 data from HanoiAir (https://geoi.com.vn) into:
  - fact_hanoiair_daily   (ward-level daily AQI + PM2.5, historical & forecast)
  - fact_hanoiair_ranking (ward-level ranking by AQI)

Jobs:
  1. district_avg   — POST /api/analysis/district_avg_statistic
                      → 126 wards × AQI avg for a given date range
  2. ranking        — POST /api/componentgeotiffdaily/rankingprovince
                      → 126 wards ranked by PM2.5/AQI for a given date
  3. forecast       — POST /api/componentgeotiffdaily/identify_district_id_list_geotiff
                      → per-ward historical (14d) + forecast (9d) PM2.5/AQI

Usage:
  python -m app.scripts.ingest_hanoiair_daily district_avg          # today
  python -m app.scripts.ingest_hanoiair_daily district_avg 2026-02-20
  python -m app.scripts.ingest_hanoiair_daily ranking               # today
  python -m app.scripts.ingest_hanoiair_daily forecast              # all 126 wards
  python -m app.scripts.ingest_hanoiair_daily all                   # run all 3 jobs
"""
from __future__ import annotations

import asyncio
import sys
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from psycopg2.extras import execute_values

from app.core.logging_config import get_logger, setup_logging
from app.db.connection import get_db_connection

setup_logging()
logger = get_logger(__name__)

BASE_URL = "https://geoi.com.vn"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_ward_ids() -> List[str]:
    """Return all ward_ids (ID_XXXXX) from dim_ward."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT ward_id FROM dim_ward ORDER BY ward_id")
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def _valid_ward_ids_set() -> set:
    """Return set of known ward_ids for filtering API responses."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT ward_id FROM dim_ward")
            return {r[0] for r in cur.fetchall()}
    finally:
        conn.close()


async def _post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """POST JSON to HanoiAir API with retry logic."""
    for attempt in range(max_retries):
        try:
            async with semaphore:
                async with session.post(url, json=payload, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("Code") == 200:
                            return data
                        logger.warning(f"API returned Code={data.get('Code')}: {data.get('Message')}")
                        return None
                    logger.warning(f"HTTP {resp.status} from {url} (attempt {attempt+1})")
        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed for {url}: {e}")
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)
    return None


# ─── Job 1: District Average Statistics ──────────────────────────────────────

async def run_district_avg(target_date: date | None = None):
    """Fetch daily AQI average for 126 wards from district_avg_statistic.

    Calls the API twice (component_id=aqi and component_id=pm25) to get
    both AQI and PM2.5 values, then merges into fact_hanoiair_daily.
    """
    target_date = target_date or date.today()
    date_str = target_date.strftime("%Y-%m-%d")
    logger.info(f"[district_avg] Starting for date={date_str}")

    valid_wards = _valid_ward_ids_set()
    semaphore = asyncio.Semaphore(5)

    async with aiohttp.ClientSession() as session:
        # Fetch AQI
        aqi_payload = {
            "id": "12",
            "from_date": f"{date_str} 00:00:00",
            "to_date": f"{date_str} 23:59:59",
            "component_id": "aqi",
            "lang_id": "vi",
        }
        # Fetch PM2.5
        pm25_payload = {
            "id": "12",
            "from_date": f"{date_str} 00:00:00",
            "to_date": f"{date_str} 23:59:59",
            "component_id": "pm25",
            "lang_id": "vi",
        }

        url = f"{BASE_URL}/api/analysis/district_avg_statistic"
        aqi_resp, pm25_resp = await asyncio.gather(
            _post_json(session, url, aqi_payload, semaphore),
            _post_json(session, url, pm25_payload, semaphore),
        )

    # Parse AQI response → {ward_id: aqi_val}
    aqi_map: Dict[str, float] = {}
    if aqi_resp and aqi_resp.get("Data", {}).get("comps"):
        for item in aqi_resp["Data"]["comps"]:
            ward_id = item.get("id")
            if ward_id and ward_id in valid_wards:
                aqi_map[ward_id] = item.get("val")

    # Parse PM2.5 response → {ward_id: pm25_val}
    pm25_map: Dict[str, float] = {}
    if pm25_resp and pm25_resp.get("Data", {}).get("comps"):
        for item in pm25_resp["Data"]["comps"]:
            ward_id = item.get("id")
            if ward_id and ward_id in valid_wards:
                pm25_map[ward_id] = item.get("val")

    # Merge into records
    all_ward_ids = set(aqi_map.keys()) | set(pm25_map.keys())
    records: List[Tuple] = []
    for wid in all_ward_ids:
        records.append((
            wid, target_date,
            aqi_map.get(wid), pm25_map.get(wid),
            False,  # is_forecast = False (actual data)
            'hanoiair', 'district_avg_daily',
        ))

    if records:
        _upsert_hanoiair_daily(records, is_actual=True)
    logger.info(f"[district_avg] Done: {len(records)} ward records for {date_str}")


def _upsert_hanoiair_daily(records: List[Tuple], is_actual: bool = False):
    """Bulk upsert into fact_hanoiair_daily.

    Priority: actual (is_forecast=FALSE) > forecast (is_forecast=TRUE).
    Actual data should not be overwritten by forecast.
    """
    if not records:
        return
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                if is_actual:
                    # Actual data always overwrites
                    query = """
                        INSERT INTO fact_hanoiair_daily (
                            ward_id, date, aqi, pm2_5, is_forecast, source, source_job
                        ) VALUES %s
                        ON CONFLICT (ward_id, date) DO UPDATE SET
                            aqi = EXCLUDED.aqi,
                            pm2_5 = EXCLUDED.pm2_5,
                            is_forecast = EXCLUDED.is_forecast,
                            source = EXCLUDED.source,
                            source_job = EXCLUDED.source_job,
                            ingested_at = NOW()
                    """
                else:
                    # Forecast data only writes if no actual data exists
                    query = """
                        INSERT INTO fact_hanoiair_daily (
                            ward_id, date, aqi, pm2_5, is_forecast, source, source_job
                        ) VALUES %s
                        ON CONFLICT (ward_id, date) DO UPDATE SET
                            aqi = EXCLUDED.aqi,
                            pm2_5 = EXCLUDED.pm2_5,
                            is_forecast = EXCLUDED.is_forecast,
                            source = EXCLUDED.source,
                            source_job = EXCLUDED.source_job,
                            ingested_at = NOW()
                        WHERE fact_hanoiair_daily.is_forecast = TRUE
                    """
                execute_values(cur, query, records)
        logger.info(f"Upserted {len(records)} rows into fact_hanoiair_daily (actual={is_actual})")
    finally:
        conn.close()


# ─── Job 2: Ranking ──────────────────────────────────────────────────────────

async def run_ranking(target_date: date | None = None):
    """Fetch ward-level ranking by AQI/PM2.5 from rankingprovince."""
    target_date = target_date or date.today()
    prev_date = target_date - timedelta(days=1)
    date_str = target_date.strftime("%Y-%m-%d")
    prev_str = prev_date.strftime("%Y-%m-%d")
    logger.info(f"[ranking] Starting for date={date_str}")

    semaphore = asyncio.Semaphore(5)
    payload = {
        "group_id": "satellite_aqi_pm25",
        "component_id": "pm25",
        "date_shooting": date_str,
        "date_shooting_pre": prev_str,
        "lang_id": "vi",
        "province_id": "12",
    }

    async with aiohttp.ClientSession() as session:
        url = f"{BASE_URL}/api/componentgeotiffdaily/rankingprovince"
        data = await _post_json(session, url, payload, semaphore)

    if not data or not data.get("Data", {}).get("comps"):
        logger.warning("[ranking] No data returned from API")
        return

    records: List[Tuple] = []
    for item in data["Data"]["comps"]:
        admin_id = item.get("administrative_id", "")
        records.append((
            target_date,
            admin_id,
            item.get("no"),       # rank
            item.get("avg"),      # aqi_avg
            item.get("avg_pre"),  # aqi_avg_pre
            'hanoiair', 'ranking_daily',
        ))

    if records:
        _upsert_hanoiair_ranking(records)
    logger.info(f"[ranking] Done: {len(records)} ward rankings for {date_str}")


def _upsert_hanoiair_ranking(records: List[Tuple]):
    """Bulk upsert into fact_hanoiair_ranking."""
    if not records:
        return
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                query = """
                    INSERT INTO fact_hanoiair_ranking (
                        date, hanoiair_administrative_id, rank, aqi_avg, aqi_avg_pre,
                        source, source_job
                    ) VALUES %s
                    ON CONFLICT (date, hanoiair_administrative_id) DO UPDATE SET
                        rank = EXCLUDED.rank,
                        aqi_avg = EXCLUDED.aqi_avg,
                        aqi_avg_pre = EXCLUDED.aqi_avg_pre,
                        source = EXCLUDED.source,
                        source_job = EXCLUDED.source_job,
                        ingested_at = NOW()
                """
                execute_values(cur, query, records)
        logger.info(f"Upserted {len(records)} rows into fact_hanoiair_ranking")
    finally:
        conn.close()


# ─── Job 3: Forecast + Historical (per-ward time series) ─────────────────────

async def run_forecast(target_date: date | None = None, predays: int = 14, nextdays: int = 9):
    """Fetch historical + forecast PM2.5/AQI per ward via identify_district_id_list_geotiff.

    For each of 126 wards, fetches a time series of (predays + 1 + nextdays) daily values.
    Dates <= target_date are marked as actual; dates > target_date as forecast.
    """
    target_date = target_date or date.today()
    date_str = target_date.strftime("%Y-%m-%d")
    ward_ids = _get_ward_ids()
    logger.info(
        f"[forecast] Starting for {len(ward_ids)} wards, "
        f"date={date_str}, predays={predays}, nextdays={nextdays}"
    )

    semaphore = asyncio.Semaphore(10)
    url = f"{BASE_URL}/api/componentgeotiffdaily/identify_district_id_list_geotiff"

    async with aiohttp.ClientSession() as session:
        tasks = []
        for wid in ward_ids:
            payload = {
                "district_id": wid,
                "groupcomponent_id": "63",  # PM2.5
                "date_request": date_str,
                "predays": predays,
                "nextdays": nextdays,
                "lang_id": "vi",
            }
            tasks.append(_post_json(session, url, payload, semaphore))

        results = await asyncio.gather(*tasks)

    actual_records: List[Tuple] = []
    forecast_records: List[Tuple] = []

    for ward_id, data in zip(ward_ids, results):
        if not data or not data.get("Data", {}).get("comps"):
            continue
        for item in data["Data"]["comps"]:
            req_date_str = item.get("requestdate")
            if not req_date_str:
                continue
            try:
                req_date = datetime.strptime(req_date_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            pm25_val = item.get("val")
            aqi_val = item.get("val_aqi")
            is_forecast = req_date > target_date

            record = (
                ward_id, req_date,
                aqi_val, pm25_val,
                is_forecast,
                'hanoiair', 'forecast_daily',
            )
            if is_forecast:
                forecast_records.append(record)
            else:
                actual_records.append(record)

    # Upsert actual first (higher priority), then forecast
    if actual_records:
        _upsert_hanoiair_daily(actual_records, is_actual=True)
    if forecast_records:
        _upsert_hanoiair_daily(forecast_records, is_actual=False)

    total = len(actual_records) + len(forecast_records)
    logger.info(
        f"[forecast] Done: {total} records "
        f"(actual={len(actual_records)}, forecast={len(forecast_records)})"
    )


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

async def run_all(target_date: date | None = None):
    """Run all 3 HanoiAir jobs sequentially."""
    target_date = target_date or date.today()
    logger.info(f"[all] Running all HanoiAir jobs for date={target_date}")
    await run_district_avg(target_date)
    await run_ranking(target_date)
    await run_forecast(target_date)
    logger.info("[all] All HanoiAir jobs completed")


def _parse_date_arg(arg: str | None) -> date | None:
    """Parse optional YYYY-MM-DD argument."""
    if not arg:
        return None
    try:
        return datetime.strptime(arg, "%Y-%m-%d").date()
    except ValueError:
        logger.error(f"Invalid date format: {arg}. Expected YYYY-MM-DD")
        sys.exit(1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    date_arg = _parse_date_arg(sys.argv[2] if len(sys.argv) > 2 else None)

    if mode == "district_avg":
        asyncio.run(run_district_avg(date_arg))
    elif mode == "ranking":
        asyncio.run(run_ranking(date_arg))
    elif mode == "forecast":
        asyncio.run(run_forecast(date_arg))
    elif mode == "all":
        asyncio.run(run_all(date_arg))
    else:
        print(
            f"Unknown mode: {mode}\n"
            "Available: district_avg [YYYY-MM-DD], ranking [YYYY-MM-DD], "
            "forecast [YYYY-MM-DD], all [YYYY-MM-DD]"
        )
        sys.exit(1)
