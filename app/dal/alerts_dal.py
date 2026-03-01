"""Weather Alerts DAL - Get weather alerts for a ward or all wards."""

from typing import List, Dict, Any, Optional
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
        # FIXED: Only 1 ward per district (30 wards total)
        sql = """
            SELECT w.ward_id, w.ts_utc, w.temp, w.wind_gust, w.pop, 
                   w.weather_main, w.weather_description, d.district_name_vi
            FROM fact_weather_hourly w
            INNER JOIN (
                SELECT ward_id, MIN(ts_utc) as min_ts
                FROM fact_weather_hourly
                WHERE data_kind = 'forecast' 
                  AND ts_utc > NOW() 
                  AND ts_utc < NOW() + INTERVAL '24 hours'
                GROUP BY ward_id
            ) latest ON w.ward_id = latest.ward_id AND w.ts_utc = latest.min_ts
            INNER JOIN dim_ward d ON w.ward_id = d.ward_id
            WHERE d.ward_id IN (
                SELECT DISTINCT ON (d.district_name_vi) d.ward_id
                FROM dim_ward d
                ORDER BY d.district_name_vi, d.ward_id
            )
        """
        results = query(sql)
    
    # Generate alerts from results
    alerts = []
    for r in results:
        alert = {"ward_id": r.get("ward_id"), "ts_utc": str(r.get("ts_utc"))}
        
        # Wind gust alert
        if r.get("wind_gust") and r["wind_gust"] > 20:
            alert["type"] = "wind"
            alert["severity"] = "warning"
            alert["message"] = f"Gio giat {r['wind_gust']:.1f} m/s - Can than"
            alerts.append(alert)
        
        # Cold alert
        if r.get("temp") and r["temp"] < KTTV_THRESHOLDS["RET_HAI"]:
            alert = {"ward_id": r.get("ward_id"), "ts_utc": str(r.get("ts_utc"))}
            alert["type"] = "cold"
            alert["severity"] = "warning"
            alert["message"] = f"Ret hai {r['temp']:.1f}°C - Can mac am"
            alerts.append(alert)
        
        # Heat alert
        if r.get("temp") and r["temp"] > KTTV_THRESHOLDS["NANG_NONG_DB"]:
            alert = {"ward_id": r.get("ward_id"), "ts_utc": str(r.get("ts_utc"))}
            alert["type"] = "heat"
            alert["severity"] = "warning"
            alert["message"] = f"Nang nong nguy hiem {r['temp']:.1f}°C - Han che ra ngoai"
            alerts.append(alert)
        
        # Thunderstorm alert
        if r.get("weather_main") == "Thunderstorm":
            alert = {"ward_id": r.get("ward_id"), "ts_utc": str(r.get("ts_utc"))}
            alert["type"] = "thunderstorm"
            alert["severity"] = "warning"
            alert["message"] = f"Co giong - Tranh ra ngoai"
            alerts.append(alert)
    
    return alerts


def get_all_district_alerts() -> Dict[str, List[Dict[str, Any]]]:
    """Get alerts grouped by district.
    
    Returns:
        Dictionary with district names as keys and list of alerts as values
    """
    alerts = get_weather_alerts()
    
    # Group by district
    district_alerts = {}
    for alert in alerts:
        # Get district from ward_id - simplified for now
        ward_id = alert.get("ward_id", "")
        if ward_id not in district_alerts:
            district_alerts[ward_id] = []
        district_alerts[ward_id].append(alert)
    
    return district_alerts
