"""Weather Alerts DAL - Get weather alerts for a ward or all wards."""

from typing import List, Dict, Any, Optional
from app.dal.timezone_utils import format_ict
from app.db.dal import query
from app.config.thresholds import KTTV_THRESHOLDS


def get_weather_alerts(ward_id: str = None) -> List[Dict[str, Any]]:
    """Get weather alerts for a ward or all districts.
    
    FIXED: When ward_id is None, only query 1 ward per district (30 wards)
    instead of scanning all 126 wards.
    
    Args:
        ward_id: Optional ward ID. If None, get alerts for all districts.
        
    Returns:
        List of weather alerts
    """
    if ward_id:
        # Query specific ward
        sql = """
            SELECT ward_id, ts_utc, temp, wind_gust, pop, weather_main, weather_description
            FROM fact_weather_hourly
            WHERE ward_id = %s 
              AND data_kind = 'forecast'
              AND ts_utc > NOW() 
              AND ts_utc < NOW() + INTERVAL '24 hours'
            ORDER BY ts_utc
        """
        results = query(sql, (ward_id,))
    else:
        # District-level: get all forecast hours for representative wards
        # (1 representative ward per district, chọn theo district_id)
        sql = """
            SELECT w.ward_id, w.ts_utc, w.temp, w.wind_gust, w.pop,
                   w.weather_main, w.weather_description,
                   d.district_id, d.district_name_vi
            FROM fact_weather_hourly w
            INNER JOIN dim_ward d ON w.ward_id = d.ward_id
            WHERE d.ward_id IN (
                SELECT DISTINCT ON (d.district_id) d.ward_id
                FROM dim_ward d
                WHERE d.district_id IS NOT NULL
                ORDER BY d.district_id, d.ward_id
            )
              AND w.data_kind = 'forecast'
              AND w.ts_utc > NOW()
              AND w.ts_utc < NOW() + INTERVAL '24 hours'
        """
        results = query(sql)
    
    # Map alert types to Vietnamese category names for clarity
    ALERT_CATEGORY_VI = {
        "wind": "gió_giật",
        "cold": "rét_hại",
        "heat": "nắng_nóng",
        "thunderstorm": "giông_sét",
    }

    # Pre-resolve ward_id → district name (avoid exposing raw IDs to LLM)
    _ward_district_cache: Dict[str, str] = {}
    if not ward_id:
        # District-level query already has district_name_vi in results
        for r in results:
            wid = r.get("ward_id", "")
            dname = r.get("district_name_vi", "")
            if wid and dname:
                _ward_district_cache[wid] = dname
    else:
        # Resolve single ward_id
        try:
            from app.dal.location_dal import get_ward_by_id
            ward_info = get_ward_by_id(ward_id) or {}
            _ward_district_cache[ward_id] = ward_info.get("district_name_vi", "Hà Nội")
        except Exception:
            _ward_district_cache[ward_id] = "Hà Nội"

    def _resolve_location(wid: str) -> str:
        return _ward_district_cache.get(wid, "một số khu vực")

    # Generate alerts from results
    alerts = []
    for r in results:
        ts_ict = format_ict(r.get("ts_utc"))
        wid = r.get("ward_id", "")
        location_name = _resolve_location(wid)
        base = {"location": location_name, "ts_ict": ts_ict}

        # Wind gust alert
        if r.get("wind_gust") is not None and r["wind_gust"] > 20:
            alerts.append({
                **base,
                "type": "wind",
                "category_vi": ALERT_CATEGORY_VI["wind"],
                "severity": "warning",
                "message": f"Gió giật {r['wind_gust']:.1f} m/s - Cẩn thận",
            })

        # Cold alert
        if r.get("temp") is not None and r["temp"] < KTTV_THRESHOLDS["RET_HAI"]:
            alerts.append({
                **base,
                "type": "cold",
                "category_vi": ALERT_CATEGORY_VI["cold"],
                "severity": "warning",
                "message": f"Rét hại {r['temp']:.1f}°C - Cần mặc ấm",
            })

        # Heat alert
        if r.get("temp") and r["temp"] > KTTV_THRESHOLDS["NANG_NONG_DB"]:
            alerts.append({
                **base,
                "type": "heat",
                "category_vi": ALERT_CATEGORY_VI["heat"],
                "severity": "warning",
                "message": f"Nắng nóng nguy hiểm {r['temp']:.1f}°C - Hạn chế ra ngoài",
            })

        # Thunderstorm alert
        if r.get("weather_main") == "Thunderstorm":
            alerts.append({
                **base,
                "type": "thunderstorm",
                "category_vi": ALERT_CATEGORY_VI["thunderstorm"],
                "severity": "warning",
                "message": "Có giông - Tránh ra ngoài",
            })

    return alerts


def get_all_district_alerts() -> Dict[str, List[Dict[str, Any]]]:
    """Get alerts grouped by district.

    Returns:
        Dictionary with district names as keys and list of alerts as values
    """
    alerts = get_weather_alerts()

    # Group by district (location already resolved in get_weather_alerts)
    district_alerts: Dict[str, List[Dict[str, Any]]] = {}
    for alert in alerts:
        district = alert.get("location", "Hà Nội")
        if district not in district_alerts:
            district_alerts[district] = []
        district_alerts[district].append(alert)

    return district_alerts
