"""Agent utilities - auto_resolve and enrich functions."""

from typing import Optional, Dict, Any


def auto_resolve_location(
    ward_id: Optional[str] = None,
    location_hint: Optional[str] = None
) -> Dict[str, Any]:
    """Resolve location from ward_id or location_hint.
    
    Args:
        ward_id: Ward ID (e.g., ID_00169)
        location_hint: Location name (e.g., "Cầu Giấy", "Đống Đa", "Hà Nội")
        
    Returns:
        Dict with status, level, ward_id, and data
        level: "ward" | "district" | "city"
    """
    from app.dal.location_dal import resolve_location, get_ward_by_id
    
    # If ward_id provided, get info
    if ward_id:
        ward = get_ward_by_id(ward_id)
        if ward:
            return {"status": "ok", "level": "ward", "ward_id": ward_id, "data": ward}
    
    # If location_hint provided, resolve it
    if location_hint:
        result = resolve_location(location_hint)
        
        # Get level from result
        level = result.get("level", "ward")
        
        if result["status"] in ("exact", "fuzzy"):
            if level == "ward":
                return {
                    "status": "ok",
                    "level": "ward",
                    "ward_id": result["data"]["ward_id"],
                    "data": result["data"]
                }
            elif level == "district":
                return {
                    "status": "ok",
                    "level": "district",
                    "district_name": result["data"]["district_name_vi"],
                    "data": result["data"]
                }
            elif level == "city":
                return {
                    "status": "ok",
                    "level": "city",
                    "city_name": result["data"]["city_name"],
                    "data": result["data"]
                }
        
        elif result["status"] in ("ambiguous", "not_found"):
            # Handle ambiguous or not found cases - ask user for clarification
            return {
                "status": result["status"],
                "level": "not_found",
                "message": result.get("message", "Khong tim thay dia diem"),
                "needs_clarification": result.get("needs_clarification", False),
                "alternatives": result.get("alternatives", []),
                "suggestion": result.get("suggestion", "Vui long cho biet them dia diem cu the")
            }
        
        elif result["status"] == "multiple":
            return {
                "status": "multiple",
                "level": "ward",
                "candidates": result["data"]
            }
        
        elif result["status"] == "not_found":
            return {
                "status": "not_found",
                "level": "not_found",
                "message": result.get("message", "Khong tim thay dia diem"),
                "needs_clarification": True,
                "suggestion": "Vui long cho biet them dia diem (vi du: 'quan TenQuan' hoac 'phuong TenPhuong')"
            }
    
    return {"status": "error", "level": "error", "message": "Khong xac dinh duoc dia diem"}


def enrich_weather_response(weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich weather data with insights.
    
    Adds:
    - Heat Index (when temp > 27°C)
    - Wind Chill (when temp <= 10°C and wind > 1.3 m/s)
    - Seasonal comparison
    - Phenomena detection
    
    Args:
        weather_data: Raw weather data from DAL
        
    Returns:
        Enriched weather data
    """
    # Return early if error
    if weather_data.get("error"):
        return weather_data
    
    try:
        from app.dal.weather_helpers import (
            compute_heat_index,
            compute_wind_chill
        )
        from app.dal.weather_knowledge_dal import (
            compare_with_seasonal,
            detect_hanoi_weather_phenomena
        )
    except ImportError as e:
        # If imports fail, return original data
        return weather_data
    
    # 1. Heat Index (only when temp > 27°C)
    temp = weather_data.get("temp")
    humidity = weather_data.get("humidity")
    if temp is not None and humidity is not None:
        heat_idx = compute_heat_index(
            weather_data["temp"],
            weather_data["humidity"]
        )
        if heat_idx is not None:
            weather_data["heat_index"] = heat_idx
    
    # 2. Wind Chill (only when temp <= 10°C and wind > 1.3 m/s)
    if weather_data.get("temp") is not None and weather_data.get("wind_speed") is not None:
        wind_chill = compute_wind_chill(
            weather_data["temp"],
            weather_data["wind_speed"]
        )
        if wind_chill is not None:
            weather_data["wind_chill"] = wind_chill
    
    # 3. Seasonal comparison
    try:
        seasonal = compare_with_seasonal(weather_data)
        weather_data["seasonal_comparison"] = seasonal.get("comparisons", [])
        weather_data["month_name"] = seasonal.get("month_name")
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Could not compute heat index: {e}")
    
    # 4. Phenomena detection
    try:
        phenomena = detect_hanoi_weather_phenomena(weather_data)
        if phenomena.get("phenomena"):
            weather_data["phenomena"] = phenomena["phenomena"]
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Could not compute heat index: {e}")
    
    return weather_data
