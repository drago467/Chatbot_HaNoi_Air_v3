"""
Seed dim_city and dim_district tables, then backfill dim_ward.district_id.

Run after build_dim_ward.py has populated the dim_ward table from CSV.

Logic:
  1. Insert row cho thành phố Hà Nội vào dim_city (idempotent).
  2. Trích DISTINCT district_name_vi + district_name_norm từ dim_ward,
     insert vào dim_district với city_id của Hà Nội (ON CONFLICT DO NOTHING).
  3. Backfill dim_ward.district_id từ dim_district thông qua district_name_vi.
  4. Verify: count wards có district_id IS NULL phải bằng 0.
"""
from app.core.logging_config import get_logger, setup_logging
from app.db.connection import get_db_connection, release_connection

setup_logging()
logger = get_logger(__name__)


CITY_NAME_VI = "Hà Nội"
CITY_NAME_NORM = "ha noi"
CITY_TIMEZONE = "Asia/Bangkok"


def build_dim_district():
    """Seed dim_city, dim_district, then backfill dim_ward.district_id."""
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                # 1. Seed dim_city (idempotent)
                logger.info("Seeding dim_city with Hà Nội...")
                cur.execute(
                    """
                    INSERT INTO dim_city (city_name_vi, city_name_norm, timezone)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (city_name_vi) DO NOTHING
                    """,
                    (CITY_NAME_VI, CITY_NAME_NORM, CITY_TIMEZONE),
                )

                cur.execute("SELECT city_id FROM dim_city WHERE city_name_vi = %s", (CITY_NAME_VI,))
                row = cur.fetchone()
                if not row:
                    raise RuntimeError("Không tìm được city_id của Hà Nội sau khi seed")
                city_id = row[0]
                logger.info(f"city_id = {city_id}")

                # 2. Seed dim_district từ DISTINCT trong dim_ward
                logger.info("Seeding dim_district từ DISTINCT dim_ward.district_name_vi...")
                cur.execute(
                    """
                    INSERT INTO dim_district (city_id, district_name_vi, district_name_norm)
                    SELECT %s, district_name_vi, district_name_norm
                    FROM (
                        SELECT DISTINCT district_name_vi, district_name_norm
                        FROM dim_ward
                        WHERE district_name_vi IS NOT NULL
                    ) AS distinct_districts
                    ON CONFLICT (city_id, district_name_vi) DO NOTHING
                    """,
                    (city_id,),
                )

                cur.execute("SELECT COUNT(*) FROM dim_district WHERE city_id = %s", (city_id,))
                district_count = cur.fetchone()[0]
                logger.info(f"Đã có {district_count} quận/huyện trong dim_district")

                # 3. Backfill dim_ward.district_id
                logger.info("Backfilling dim_ward.district_id...")
                cur.execute(
                    """
                    UPDATE dim_ward w
                    SET district_id = d.district_id
                    FROM dim_district d
                    WHERE w.district_name_vi = d.district_name_vi
                      AND d.city_id = %s
                      AND (w.district_id IS NULL OR w.district_id != d.district_id)
                    """,
                    (city_id,),
                )
                updated_rows = cur.rowcount
                logger.info(f"Cập nhật district_id cho {updated_rows} ward")

                # 4. Verify: không còn ward nào bị NULL
                cur.execute("SELECT COUNT(*) FROM dim_ward WHERE district_id IS NULL")
                null_count = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM dim_ward")
                total_wards = cur.fetchone()[0]

                if null_count > 0:
                    cur.execute(
                        """
                        SELECT ward_id, ward_name_vi, district_name_vi
                        FROM dim_ward
                        WHERE district_id IS NULL
                        LIMIT 10
                        """
                    )
                    sample = cur.fetchall()
                    logger.warning(
                        f"Có {null_count}/{total_wards} ward thiếu district_id. "
                        f"Mẫu: {sample}"
                    )
                else:
                    logger.info(f"Tất cả {total_wards}/{total_wards} ward đã có district_id")

        logger.info("build_dim_district hoàn tất.")

    except Exception as e:
        logger.error(f"Lỗi khi build dim_district: {e}")
        raise
    finally:
        release_connection(conn)


if __name__ == "__main__":
    build_dim_district()
