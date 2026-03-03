import requests
import pandas as pd
import unicodedata
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.core.logging_config import get_logger, setup_logging
from app.db.connection import get_db_connection
from app.core.normalize import normalize_name
from psycopg2.extras import execute_values

setup_logging()
logger = get_logger(__name__)

# Project root directory (parent of app/)
BASE_DIR = Path(__file__).parent.parent.parent.resolve()

BASE_URL = "https://geoi.com.vn"


def fetch_ward_list() -> List[Dict[str, Any]]:
    """Fetch all wards from HanoiAir administrative API."""
    logger.info("Fetching ward list from HanoiAir...")
    endpoint = f"{BASE_URL}/api/administrative/administrative_province_district"
    params = {"province_id": "12", "lang_id": "vi"}
    
    try:
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        wards = []
        for item in data:
            if item.get("type") == "district":  # HanoiAir labels wards as 'district' in this API
                wards.append({
                    "ward_id": item.get("id"),
                    "ward_name_vi": item.get("name")
                })
        logger.info(f"Found {len(wards)} wards.")
        return wards
    except Exception as e:
        logger.error(f"Error fetching ward list: {e}")
        return []

def fetch_ward_extent(ward_id: str) -> Optional[Dict[str, Any]]:
    """Fetch bounding box for a specific ward using endpoint A."""
    endpoint = f"{BASE_URL}/api/administrative/district_extent"
    params = {"district_id": ward_id, "lang_id": "vi"}
    
    try:
        response = requests.get(endpoint, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch extent for {ward_id}: {e}")
        return None

def split_ward_prefix_and_core(name_vi: str) -> tuple[str, str]:
    """Split Vietnamese admin prefix (Phường/Xã/Thị trấn) from core name.

    Returns:
        (prefix_norm, core_norm)

    Examples:
        "Phường Dịch Vọng" -> ("phuong", "dich vong")
        "Xã Bát Tràng" -> ("xa", "bat trang")
    """
    if not name_vi:
        return "", ""

    name_stripped = name_vi.strip()
    name_low = name_stripped.lower()

    prefixes = ["phường", "xã", "thị trấn"]
    for p in prefixes:
        if name_low.startswith(p):
            core = name_stripped[len(p):].strip()
            return normalize_name(p), normalize_name(core)

    # Fallback: no prefix
    return "", normalize_name(name_stripped)


def build_dim_ward():
    """Main script to build dim_ward table."""
    # 1. Fetch wards from HanoiAir
    wards = fetch_ward_list()
    if not wards:
        return

    # 2. Fetch extents and calculate centroids
    logger.info("Fetching bounding boxes and calculating centroids...")
    enriched_wards = []
    for w in wards:
        extent = fetch_ward_extent(w["ward_id"])
        if extent:
            w.update({
                "minx": extent.get("minx"),
                "miny": extent.get("miny"),
                "maxx": extent.get("maxx"),
                "maxy": extent.get("maxy"),
            })
            if all(w.get(k) is not None for k in ["minx", "miny", "maxx", "maxy"]):
                w["lat"] = (w["miny"] + w["maxy"]) / 2
                w["lon"] = (w["minx"] + w["maxx"]) / 2

        w["ward_name_norm"] = normalize_name(w["ward_name_vi"])
        prefix_norm, core_norm = split_ward_prefix_and_core(w["ward_name_vi"])
        w["ward_prefix_norm"] = prefix_norm
        w["ward_name_core_norm"] = core_norm

        enriched_wards.append(w)
        time.sleep(0.1)  # Be reasonable

    df_hanoiair = pd.DataFrame(enriched_wards)

    # 3. Merge with data/location.csv
    logger.info("Merging with data/location.csv...")
    try:
        df_local = pd.read_csv(BASE_DIR / "data" / "location.csv")
        # Ensure we have ward_name_norm in local csv
        if "ward_name_norm" not in df_local.columns:
            df_local["ward_name_norm"] = df_local["ward_name_vi"].apply(normalize_name)
        
        # Merge to get district_name_vi and district_name_norm
        df_final = pd.merge(
            df_hanoiair, 
            df_local[["ward_name_norm", "district_name_vi", "district_name_norm"]], 
            on="ward_name_norm", 
            how="left"
        )
    except Exception as e:
        logger.error(f"Error merging with location.csv: {e}")
        df_final = df_hanoiair

    # 4. Save to processed data
    os.makedirs(BASE_DIR / "data" / "processed", exist_ok=True)
    processed_path = BASE_DIR / "data" / "processed" / "dim_ward.csv"
    df_final.to_csv(processed_path, index=False)
    logger.info(f"Saved processed data to {processed_path}")

    # 5. Import into Postgres
    logger.info("Importing into Postgres dim_ward table...")
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                # Prepare data for upsert
                # Columns: ward_id, ward_name_vi, ward_name_norm, ward_name_core_norm, ward_prefix_norm,
                #          district_name_vi, district_name_norm, minx, miny, maxx, maxy, lat, lon
                data_to_upsert = []
                for _, row in df_final.iterrows():
                    data_to_upsert.append((
                        row["ward_id"],
                        row["ward_name_vi"],
                        row["ward_name_norm"],
                        row.get("ward_name_core_norm"),
                        row.get("ward_prefix_norm"),
                        row.get("district_name_vi"),
                        row.get("district_name_norm"),
                        row.get("minx"),
                        row.get("miny"),
                        row.get("maxx"),
                        row.get("maxy"),
                        row.get("lat"),
                        row.get("lon")
                    ))

                upsert_query = """
                INSERT INTO dim_ward (
                    ward_id, ward_name_vi, ward_name_norm, ward_name_core_norm, ward_prefix_norm,
                    district_name_vi, district_name_norm,
                    minx, miny, maxx, maxy, lat, lon
                ) VALUES %s
                ON CONFLICT (ward_id) DO UPDATE SET
                    ward_name_vi = EXCLUDED.ward_name_vi,
                    ward_name_norm = EXCLUDED.ward_name_norm,
                    ward_name_core_norm = EXCLUDED.ward_name_core_norm,
                    ward_prefix_norm = EXCLUDED.ward_prefix_norm,
                    district_name_vi = COALESCE(EXCLUDED.district_name_vi, dim_ward.district_name_vi),
                    district_name_norm = COALESCE(EXCLUDED.district_name_norm, dim_ward.district_name_norm),
                    minx = EXCLUDED.minx,
                    miny = EXCLUDED.miny,
                    maxx = EXCLUDED.maxx,
                    maxy = EXCLUDED.maxy,
                    lat = EXCLUDED.lat,
                    lon = EXCLUDED.lon,
                    updated_at = NOW();
                """
                execute_values(cur, upsert_query, data_to_upsert)
        logger.info("Import successful.")
    except Exception as e:
        logger.error(f"Error importing to DB: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    build_dim_ward()
