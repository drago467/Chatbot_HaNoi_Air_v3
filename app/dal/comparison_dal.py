"""Comparison DAL - Compare weather between locations."""

from typing import Dict, Any, List
from app.db.dal import query
from app.dal.weather_helpers import wind_deg_to_vietnamese


def compare_weather(ward_id1: str, ward_id2: str) -> Dict[str, Any]:
    """Compare current weather between two wards.
    
    FIXED: Maps results by ward_id to ensure correct order matching input.
    
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
            "message": f"Không đủ dữ liệu (chỉ có {len(results)}/2 wards)"
        }
    
    # Map results by ward_id to maintain input order
    results_by_id = {r["ward_id"]: r for r in results}
    w1 = results_by_id.get(ward_id1)
    w2 = results_by_id.get(ward_id2)
    
    if not w1 or not w2:
        return {"error": "missing_data", "message": "Không tìm thấy dữ liệu cho 1 trong 2 địa điểm"}
    
    # Add Vietnamese wind direction
    w1["wind_direction_vi"] = wind_deg_to_vietnamese(w1.get("wind_deg"))
    w2["wind_direction_vi"] = wind_deg_to_vietnamese(w2.get("wind_deg"))
    
    # Get ward names for display
    from app.dal.location_dal import get_ward_by_id
    loc1 = get_ward_by_id(ward_id1) or {}
    loc2 = get_ward_by_id(ward_id2) or {}
    name1 = loc1.get("ward_name_vi", ward_id1)
    name2 = loc2.get("ward_name_vi", ward_id2)
    
    # Add location names to results
    w1["ward_name"] = name1
    w2["ward_name"] = name2
    
    # Calculate differences
    if w1.get("temp") is None or w2.get("temp") is None:
        return {"error": "missing_data", "message": "Thiếu dữ liệu nhiệt độ cho một hoặc cả hai địa điểm"}
    temp_diff = w1.get("temp", 0) - w2.get("temp", 0)
    humidity_diff = (w1.get("humidity") or 0) - (w2.get("humidity") or 0)
    
    # Generate comparison text with location names
    if abs(temp_diff) <= 2:
        temp_comparison = "Nhiệt độ tương tự"
    elif temp_diff > 0:
        temp_comparison = f"{name1} nóng hơn {name2} {abs(temp_diff):.1f}°C"
    else:
        temp_comparison = f"{name2} nóng hơn {name1} {abs(temp_diff):.1f}°C"
    
    return {
        "location1": w1,
        "location2": w2,
        "differences": {
            "temp_diff": temp_diff,
            "humidity_diff": humidity_diff
        },
        "comparison_text": temp_comparison
    }


def compare_with_previous_day(ward_id: str) -> Dict[str, Any]:
    """Compare today's weather with the previous available day.
    
    Note: If data is missing for yesterday, compares with the most recent
    available previous day. Use compare_with_yesterday() for strict yesterday.
    
    Args:
        ward_id: Ward ID
        
    Returns:
        Dictionary with comparison data
    """
    # Query 2 most recent days (both history and forecast)
    # Prioritize history if available for the same date
    results = query("""
        SELECT DISTINCT ON (date) date, temp_avg, temp_min, temp_max, humidity, 
               rain_total, weather_main, data_kind
        FROM fact_weather_daily
        WHERE ward_id = %s
        ORDER BY date DESC, 
                 CASE WHEN data_kind = 'history' THEN 0 ELSE 1 END
        LIMIT 2
    """, (ward_id,))
    
    if len(results) < 2:
        return {
            "error": "not_enough_data",
            "message": "Cần có dữ liệu ít nhất 2 ngày"
        }
    
    today, previous = results[0], results[1]  # results[0] is newest (ORDER BY DESC)
    
    # Calculate days between
    from datetime import datetime
    today_date = today.get("date")
    prev_date = previous.get("date")
    if hasattr(today_date, 'date'):
        today_date = today_date.date()
    if hasattr(prev_date, 'date'):
        prev_date = prev_date.date()
    days_diff = (today_date - prev_date).days if today_date and prev_date else 1
    day_label = f"{days_diff} ngày trước" if days_diff > 1 else "hôm qua"
    
    # Calculate changes
    if today.get("temp_avg") is None or previous.get("temp_avg") is None:
        return {"error": "missing_data", "message": "Thiếu dữ liệu nhiệt độ cho một hoặc cả hai ngày"}
    temp_diff = today.get("temp_avg", 0) - previous.get("temp_avg", 0)
    rain_diff = (today.get("rain_total") or 0) - (previous.get("rain_total") or 0)
    
    changes = []
    
    if temp_diff > 2:
        changes.append(f"Nhiệt độ tăng {temp_diff:.1f}°C so với {day_label}")
    elif temp_diff < -2:
        changes.append(f"Nhiệt độ giảm {abs(temp_diff):.1f}°C so với {day_label}")

    if rain_diff > 5:
        changes.append(f"Mưa nhiều hơn {day_label}")
    elif rain_diff < -5:
        changes.append(f"Mưa ít hơn {day_label}")
    
    return {
        "today": today,
        "previous": previous,
        "changes": changes,
        "temp_diff": temp_diff,
        "rain_diff": rain_diff
    }

# Alias for backward compatibility
compare_with_yesterday = compare_with_previous_day

