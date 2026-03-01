"""Comparison DAL - Compare weather between locations."""

from typing import Dict, Any, List
from app.db.dal import query
from app.dal.weather_helpers import wind_deg_to_vietnamese


def compare_weather(ward_id1: str, ward_id2: str) -> Dict[str, Any]:
    """Compare current weather between two wards.
    
    FIXED: Uses DISTINCT ON to ensure 1 row per ward.
    
    Args:
        ward_id1: First ward ID
        ward_id2: Second ward ID
        
    Returns:
        Dictionary with comparison data
    """
    results = query("""
        SELECT DISTINCT ON (ward_id) 
               ward_id, temp, humidity, wind_speed, wind_deg,
               weather_main, weather_description, ts_utc
        FROM fact_weather_hourly
        WHERE ward_id IN (%s, %s) AND data_kind = 'current'
        ORDER BY ward_id, ts_utc DESC
    """, (ward_id1, ward_id2))
    
    if len(results) != 2:
        return {
            "error": "missing_data",
            "message": f"Khong du lieu (chi co {len(results)}/2 wards)"
        }
    
    w1, w2 = results[0], results[1]
    
    # Add Vietnamese wind direction
    w1["wind_direction_vi"] = wind_deg_to_vietnamese(w1.get("wind_deg"))
    w2["wind_direction_vi"] = wind_deg_to_vietnamese(w2.get("wind_deg"))
    
    # Calculate differences
    temp_diff = (w1.get("temp") or 0) - (w2.get("temp") or 0)
    humidity_diff = (w1.get("humidity") or 0) - (w2.get("humidity") or 0)
    
    # Generate comparison text
    if abs(temp_diff) <= 2:
        temp_comparison = "Nhiet do tuong tu"
    elif temp_diff > 0:
        temp_comparison = f"{w1['ward_id']} nong hon {abs(temp_diff):.1f}°C"
    else:
        temp_comparison = f"{w2['ward_id']} nong hon {abs(temp_diff):.1f}°C"
    
    return {
        "location1": w1,
        "location2": w2,
        "differences": {
            "temp_diff": temp_diff,
            "humidity_diff": humidity_diff
        },
        "comparison_text": temp_comparison
    }


def compare_with_yesterday(ward_id: str) -> Dict[str, Any]:
    """Compare today's weather with yesterday's.
    
    Args:
        ward_id: Ward ID
        
    Returns:
        Dictionary with comparison data
    """
    results = query("""
        SELECT date, temp_avg, temp_min, temp_max, humidity, 
               rain_total, weather_main, data_kind
        FROM fact_weather_daily
        WHERE ward_id = %s
          AND data_kind = "history"
        ORDER BY date DESC
        LIMIT 2
    """, (ward_id,))
    
    if len(results) < 2:
        return {
            "error": "not_enough_data",
            "message": "Can co du lieu it nhat 2 ngay"
        }
    
    today, yesterday = results[1], results[0]  # Older first
    
    # Calculate changes
    temp_diff = (today.get("temp_avg") or 0) - (yesterday.get("temp_avg") or 0)
    rain_diff = (today.get("rain_total") or 0) - (yesterday.get("rain_total") or 0)
    
    changes = []
    
    if temp_diff > 2:
        changes.append(f"Nhiet do tang {temp_diff:.1f}°C so voi hom qua")
    elif temp_diff < -2:
        changes.append(f"Nhiet do giam {abs(temp_diff):.1f}°C so voi hom qua")
    
    if rain_diff > 5:
        changes.append("Mua nhieu hon hom qua")
    elif rain_diff < -5:
        changes.append("Mua it hon hom qua")
    
    return {
        "today": today,
        "yesterday": yesterday,
        "changes": changes,
        "temp_diff": temp_diff,
        "rain_diff": rain_diff
    }
