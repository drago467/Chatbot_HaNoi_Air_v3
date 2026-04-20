"""Agent utilities - auto_resolve and enrich functions."""

from typing import Optional, Dict, Any
import json
import os

# Load POI mapping from JSON config (O(1) lookup, no LLM token cost)
_POI_MAPPING = None

def _get_poi_mapping() -> Dict[str, str]:
    """Load POI → district mapping from config file. Cached after first load."""
    global _POI_MAPPING
    if _POI_MAPPING is None:
        poi_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'poi_mapping.json')
        try:
            with open(poi_path, 'r', encoding='utf-8') as f:
                _POI_MAPPING = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _POI_MAPPING = {}
    return _POI_MAPPING


def _resolve_poi(location_hint: str) -> Optional[Dict[str, Any]]:
    """Try to resolve location via POI mapping before LLM rewrite.

    Returns resolved dict if POI matched, None otherwise.
    Uses case-insensitive matching with normalized hint.
    """
    poi_map = _get_poi_mapping()
    if not poi_map:
        return None

    hint = location_hint.strip()

    # Direct match
    if hint in poi_map:
        district_name = poi_map[hint]
        from app.dal.location_dal import resolve_location
        result = resolve_location(district_name)
        if result.get("status") in ("exact", "fuzzy") and result.get("level") == "district":
            return {
                "status": "ok",
                "level": "district",
                "district_name": result["data"]["district_name_vi"],
                "data": result["data"],
                "poi_matched": hint
            }

    # Case-insensitive match
    hint_lower = hint.lower()
    for poi_name, district_name in poi_map.items():
        if poi_name.lower() == hint_lower:
            from app.dal.location_dal import resolve_location
            result = resolve_location(district_name)
            if result.get("status") in ("exact", "fuzzy") and result.get("level") == "district":
                return {
                    "status": "ok",
                    "level": "district",
                    "district_name": result["data"]["district_name_vi"],
                    "data": result["data"],
                    "poi_matched": poi_name
                }

    # Substring match: check if hint contains a POI name
    for poi_name, district_name in poi_map.items():
        if poi_name.lower() in hint_lower or hint_lower in poi_name.lower():
            from app.dal.location_dal import resolve_location
            result = resolve_location(district_name)
            if result.get("status") in ("exact", "fuzzy") and result.get("level") == "district":
                return {
                    "status": "ok",
                    "level": "district",
                    "district_name": result["data"]["district_name_vi"],
                    "data": result["data"],
                    "poi_matched": poi_name
                }

    return None


def auto_resolve_location(
    ward_id: Optional[str] = None,
    location_hint: Optional[str] = None,
    target_scope: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve location from ward_id or location_hint.

    Args:
        ward_id: Ward ID (e.g., "ID_00169")
        location_hint: Location name (e.g., "Cầu Giấy", "Đống Đa", "Hà Nội")
        target_scope: Scope từ SLM router ("city"/"district"/"ward"/None)

    Returns:
        Dict with status, level, ward_id, and data
        level: "ward" | "district" | "city"
    """
    from app.dal.location_dal import resolve_location_scoped, get_ward_by_id

    # If ward_id provided, get info
    if ward_id:
        ward = get_ward_by_id(ward_id)
        if ward:
            return {"status": "ok", "level": "ward", "ward_id": ward_id, "data": ward}

    # If location_hint provided, resolve it
    if location_hint:
        # Step 0: Try POI mapping first (fast, no LLM cost)
        # SKIP POI khi scope=ward — POI thường map về district, nhưng nếu user
        # chỉ rõ scope=ward (qua router) thì ưu tiên search ward trước
        # (case Q3: "phường Cầu Giấy" bị POI "Công viên Cầu Giấy" override).
        if target_scope != "ward":
            poi_result = _resolve_poi(location_hint)
            if poi_result:
                # Respect scope: nếu POI trả district nhưng scope=city → upgrade
                if target_scope == "city":
                    return {"status": "ok", "level": "city",
                            "data": {"city_name": "Hà Nội"}}
                return poi_result

        # DB resolution với scope guidance
        result = resolve_location_scoped(location_hint, target_scope=target_scope)
        
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
                "message": result.get("message", "Không tìm thấy địa điểm"),
                "needs_clarification": result.get("needs_clarification", False),
                "alternatives": result.get("alternatives", []),
                "suggestion": result.get("suggestion", "Vui lòng cho biết thêm địa điểm cụ thể")
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
                "message": result.get("message", "Không tìm thấy địa điểm"),
                "needs_clarification": True,
                "suggestion": "Vui lòng cho biết thêm địa điểm (ví dụ: 'quận TênQuận' hoặc 'phường TênPhường')"
            }
    
    return {"status": "error", "level": "error", "message": "Không xác định được địa điểm"}


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
                f"Chênh lệch nhiệt độ giữa các phường/xã: {spread}°C"
            )

    # Wind direction Vietnamese
    wind_deg = data.get("avg_wind_deg")
    if wind_deg is not None:
        data["wind_direction_vi"] = wind_deg_to_vietnamese(wind_deg)

    # UV status
    uvi = data.get("max_uvi") or data.get("avg_uvi")
    if uvi is not None:
        if uvi >= 11:
            data["uv_status"] = "Cực cao - Nguy hiểm"
        elif uvi >= 8:
            data["uv_status"] = "Rất cao - Hạn chế ra ngoài 10h-14h"
        elif uvi >= 6:
            data["uv_status"] = "Cao - Cần che nắng"
        elif uvi >= 3:
            data["uv_status"] = "Trung bình"
        else:
            data["uv_status"] = "Thấp"

    # Dew point comfort
    dp = data.get("avg_dew_point")
    if dp is not None:
        if dp > 24:
            data["dew_point_status"] = "Rất oi bức, khó chịu"
        elif dp > 20:
            data["dew_point_status"] = "Oi bức"
        elif dp > 16:
            data["dew_point_status"] = "Dễ chịu"
        elif dp > 10:
            data["dew_point_status"] = "Khô ráo"
        else:
            data["dew_point_status"] = "Rất khô"

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
