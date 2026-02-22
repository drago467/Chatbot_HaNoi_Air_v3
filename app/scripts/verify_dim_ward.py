from app.core.logging_config import get_logger, setup_logging
from app.db.connection import get_db_connection

setup_logging()
logger = get_logger(__name__)


def run_fuzzy_test(cur, term: str, use_core: bool = True):
    """Run a single fuzzy search test case."""
    column = "ward_name_core_norm" if use_core else "ward_name_norm"
    logger.info(f"--- Query: '{term}' (using {column}) ---")
    
    cur.execute(
        f"""
        SELECT ward_id, ward_name_vi, district_name_vi,
               similarity({column}, %s) AS score
        FROM dim_ward
        ORDER BY score DESC
        LIMIT 3
        """,
        (term.lower(),),
    )
    rows = cur.fetchall()
    for ward_id, ward_name_vi, district_name_vi, score in rows:
        logger.info(f"  [{score:.4f}] {ward_name_vi} ({district_name_vi}) | ID: {ward_id}")


def verify_dim_ward() -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 1) Basic stats
            cur.execute("SELECT COUNT(*) FROM dim_ward")
            count = cur.fetchone()[0]
            logger.info(f"Total wards in DB: {count}")

            # 2) Comprehensive Test Cases
            test_cases = [
                # (term, use_core_logic)
                ("dich vong", True),          # Không dấu, không prefix
                ("Dịch Vọng", True),          # Có dấu, không prefix
                ("phuong dich vong", False),   # Không dấu, có prefix
                ("Phường Dịch Vọng", False),   # Có dấu, có prefix
                ("bat trang", True),           # Xã, không dấu
                ("xa bat trang", False),       # Xã, có prefix
                ("van mieu", True),            # Tên ghép
                ("cau giay", True),            # Trùng tên quận/phường
            ]

            for term, use_core in test_cases:
                run_fuzzy_test(cur, term, use_core)

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    verify_dim_ward()
