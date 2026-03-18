"""Agent utilities - auto_resolve and enrich functions."""

from typing import Optional, Dict, Any


# Enable/disable LLM-based question rewriting for location resolution
# Set to True to use the new rewrite_dal for better location resolution
USE_LLM_REWRITE = True


def auto_resolve_location(
    ward_id: Optional[str] = None,
    location_hint: Optional[str] = None
) -> Dict[str, Any]:
    """Resolve location from ward_id or location_hint.
    
    Args:
        ward_id: Ward ID (e.g., "ID_00169")
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
        # Use LLM-based rewrite for better resolution (NEW APPROACH)
        if USE_LLM_REWRITE:
            try:
                from app.dal.rewrite_dal import resolve_with_rewrite
                rewrite_result = resolve_with_rewrite(location_hint)
                
                # If rewrite succeeded, use the result
                if rewrite_result.get("status") == "ok":
                    district_name = rewrite_result.get("district_name")
                    
                    # SPECIAL CASE: If location_hint suggests city-level query
                    # Check if the original hint suggests city-level query
                    hint_lower = location_hint.lower().strip()
                    is_city_query = (
                        hint_lower in ['hà nội', 'ha noi', 'hn', 'hn capital', 'thành phố hà nội'] or
                        'nội thành' in hint_lower or
                        'ngoại thành' in hint_lower or
                        hint_lower.startswith('hà nội ')
                    )
                    
                    # Also check the rewritten location - if it's just "Hà Nội" (no ward specified)
                    rewrite_location = rewrite_result.get("location", "").lower().strip()
                    rewrite_ward = rewrite_result.get("ward", "").strip() if rewrite_result.get("ward") else ""
                    
                    # If location is ONLY "Hà Nội" with no ward info, it's city-level
                    if rewrite_location in ['hà nội', 'ha noi', 'hà nội', 'hanoi'] and not rewrite_ward:
                        is_city_query = True
                    
                    if is_city_query:
                        # Return city level for Hanoi
                        return {
                            "status": "ok",
                            "level": "city",
                            "city_name": "Hà Nội",
                            "data": {"city_name": "Hà Nội"}
                        }
                    
                    if district_name:
                        # Resolve from DB using the district name from LLM
                        db_result = resolve_location(district_name)
                        if db_result.get("status") in ("exact", "fuzzy"):
                            level = db_result.get("level", "district")
                            if level == "ward":
                                return {
                                    "status": "ok",
                                    "level": "ward",
                                    "ward_id": db_result["data"]["ward_id"],
                                    "data": db_result["data"]
                                }
                            elif level == "district":
                                return {
                                    "status": "ok",
                                    "level": "district",
                                    "district_name": db_result["data"]["district_name_vi"],
                                    "data": db_result["data"]
                                }
                            elif level == "city":
                                return {
                                    "status": "ok",
                                    "level": "city",
                                    "city_name": db_result["data"].get("city_name", "Hà Nội"),
                                    "data": db_result["data"]
                                }
                
                # If rewrite failed or returned ambiguous, fall back to old method
                if rewrite_result.get("needs_clarification"):
                    # Try old method as fallback
                    result = resolve_location(location_hint)
                    if result.get("status") in ("exact", "fuzzy"):
                        level = result.get("level", "ward")
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
                                "city_name": result["data"].get("city_name", "Hà Nội"),
                                "data": result["data"]
                            }
                    # Both failed - return clarification
                    return {
                        "status": "ambiguous",
                        "level": "not_found",
                        "message": rewrite_result.get("suggestion", "Không xác định được địa điểm"),
                        "needs_clarification": True,
                        "suggestion": rewrite_result.get("suggestion", "Vui lòng cho biết thêm địa điểm cụ thể")
                    }
            except Exception as e:
                # Fall back to old method if rewrite fails
                pass
        
        # OLD METHOD: Direct DB resolution
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
        logging.getLogger(__name__).warning(f"Could not compute seasonal comparison: {e}")

    # 4. Phenomena detection
    try:
        phenomena = detect_hanoi_weather_phenomena(weather_data)
        if phenomena.get("phenomena"):
            weather_data["phenomena"] = phenomena["phenomena"]
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Could not detect phenomena: {e}")

    return weather_data


def _base_enrich_aggregated(data: Dict[str, Any]) -> Dict[str, Any]:
    """Shared enrichment logic for district/city aggregated data.

    Adds: heat_index, wind_chill, seasonal_comparison, phenomena,
    spatial_variation, wind_direction_vi, uv_status, dew_point_status.
    """
    if data.get("error"):
        return data

    try:
        from app.dal.weather_helpers import (
            compute_heat_index,
            compute_wind_chill,
            wind_deg_to_vietnamese,
        )
        from app.dal.weather_knowledge_dal import (
            compare_with_seasonal,
            detect_hanoi_weather_phenomena,
        )
    except ImportError:
        return data

    avg_temp = data.get("avg_temp")
    humidity = data.get("avg_humidity")

    # Heat Index
    if avg_temp is not None and humidity is not None:
        hi = compute_heat_index(avg_temp, humidity)
        if hi is not None:
            data["heat_index"] = hi

    # Wind Chill
    wind_speed = data.get("avg_wind_speed")
    if avg_temp is not None and wind_speed is not None:
        wc = compute_wind_chill(avg_temp, wind_speed)
        if wc is not None:
            data["wind_chill"] = wc

    # Spatial variation (temp spread across wards)
    min_t = data.get("min_temp")
    max_t = data.get("max_temp")
    if min_t is not None and max_t is not None:
        spread = round(max_t - min_t, 1)
        data["temp_spread"] = spread
        if spread > 2:
            data["temp_spread_note"] = (
                f"Chenh lech nhiet do giua cac phuong/xa: {spread}C"
            )

    # Wind direction Vietnamese
    wind_deg = data.get("avg_wind_deg")
    if wind_deg is not None:
        data["wind_direction_vi"] = wind_deg_to_vietnamese(wind_deg)

    # UV status
    uvi = data.get("max_uvi") or data.get("avg_uvi")
    if uvi is not None:
        if uvi >= 11:
            data["uv_status"] = "Cuc cao - Nguy hiem"
        elif uvi >= 8:
            data["uv_status"] = "Rat cao - Han che ra ngoai 10h-14h"
        elif uvi >= 6:
            data["uv_status"] = "Cao - Can che nang"
        elif uvi >= 3:
            data["uv_status"] = "Trung binh"
        else:
            data["uv_status"] = "Thap"

    # Dew point comfort
    dp = data.get("avg_dew_point")
    if dp is not None:
        if dp > 24:
            data["dew_point_status"] = "Rat oi buc, kho chiu"
        elif dp > 20:
            data["dew_point_status"] = "Oi buc"
        elif dp > 16:
            data["dew_point_status"] = "De chiu"
        elif dp > 10:
            data["dew_point_status"] = "Kho rao"
        else:
            data["dew_point_status"] = "Rat kho"

    # Seasonal comparison (map avg_temp -> temp for the helper)
    try:
        proxy = {"temp": avg_temp, "humidity": humidity, "month": None}
        seasonal = compare_with_seasonal(proxy)
        data["seasonal_comparison"] = seasonal.get("comparisons", [])
    except Exception:
        pass

    # Phenomena detection
    try:
        proxy = {
            "temp": avg_temp,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "wind_gust": data.get("max_wind_gust"),
            "month": None,
        }
        phenomena = detect_hanoi_weather_phenomena(proxy)
        if phenomena.get("phenomena"):
            data["phenomena"] = phenomena["phenomena"]
    except Exception:
        pass

    return data


def enrich_district_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich district-level aggregated weather data."""
    return _base_enrich_aggregated(data)


def enrich_city_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich city-level aggregated weather data."""
    return _base_enrich_aggregated(data)
