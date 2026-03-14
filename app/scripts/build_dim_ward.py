"""
Build dim_ward table from local CSV file.

Reads from: data/processed/dim_ward.csv
No API calls needed - fully offline.
"""
import pandas as pd
from pathlib import Path
from app.core.logging_config import get_logger, setup_logging
from app.db.connection import get_db_connection, release_connection
from psycopg2.extras import execute_values

setup_logging()
logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent.resolve()


def build_dim_ward():
    """Build dim_ward table from local CSV file."""
    
    # 1. Load dim_ward.csv
    logger.info("Loading data/processed/dim_ward.csv...")
    csv_path = BASE_DIR / "data" / "processed" / "dim_ward.csv"
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} wards from CSV")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return

    # 2. Log merge results
    total = len(df)
    with_district = df["district_name_vi"].notna().sum()
    logger.info(f"Districts: {with_district}/{total} wards have district_name_vi")

    # Check for missing districts
    missing = df[df["district_name_vi"].isna()]
    if len(missing) > 0:
        logger.warning(f"Wards without district: {missing['ward_name_vi'].tolist()}")

    # 3. Import into Postgres
    logger.info("Importing into Postgres dim_ward table...")
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                data_to_upsert = []
                for _, row in df.iterrows():
                    data_to_upsert.append((
                        row["ward_id"],
                        row["ward_name_vi"],
                        row.get("ward_name_norm"),
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
        
        # Verify
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM dim_ward WHERE district_name_vi IS NOT NULL")
            count = cur.fetchone()[0]
            logger.info(f"Verified: {count}/{total} wards have district in DB")
            
    except Exception as e:
        logger.error(f"Error importing to DB: {e}")
    finally:
        release_connection(conn)


if __name__ == "__main__":
    build_dim_ward()
