"""OpenStreetMap (Nominatim) integration for location resolution."""

import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org"


async def search_osm(location_hint: str, country_code: str = "VN") -> Optional[Dict[str, Any]]:
    """Search OpenStreetMap/Nominatim for a location."""
    try:
        async with aiohttp.ClientSession() as session:
            params = {
                "q": f"{location_hint}, Vietnam",
                "format": "json",
                "limit": 5,
                "addressdetails": 1,
                "language": "vi"
            }
            headers = {"User-Agent": "WeatherChatbotHanoi/1.0"}
            
            async with session.get(
                f"{NOMINATIM_BASE_URL}/search",
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    return None
                
                results = await resp.json()
                
                if not results:
                    return None
                
                vn_results = [r for r in results if r.get("address", {}).get("country_code") == "vn"]
                
                if not vn_results:
                    results = results[:1]
                else:
                    results = vn_results[:3]
                
                return _parse_osm_results(results, location_hint)
                
    except Exception as e:
        logger.error(f"OSM API error: {e}")
        return None


def _parse_osm_results(results: List[Dict], original_query: str) -> Dict[str, Any]:
    """Parse OSM results and extract relevant information."""
    
    if not results:
        return {"status": "not_found", "message": "Location not found in OSM"}
    
    best = results[0]
    address = best.get("address", {})
    
    district = address.get("county") or address.get("city") or address.get("district")
    ward = address.get("suburb") or address.get("neighbourhood") or address.get("village")
    
    confidence = _calculate_confidence(best, original_query, district, ward)
    
    parsed = {
        "status": "found",
        "level": "unknown",
        "confidence": confidence,
        "data": {
            "osm_id": best.get("place_id"),
            "display_name": best.get("display_name"),
            "lat": float(best.get("lat", 0)),
            "lon": float(best.get("lon", 0)),
            "type": best.get("type"),
            "district_name_vi": district,
            "ward_name_vi": ward,
        },
        "alternatives": []
    }
    
    for r in results[1:]:
        addr = r.get("address", {})
        alt_district = addr.get("county") or addr.get("city") or addr.get("district")
        
        parsed["alternatives"].append({
            "display_name": r.get("display_name"),
            "lat": float(r.get("lat", 0)),
            "lon": float(r.get("lon", 0)),
            "district": alt_district,
        })
    
    if ward:
        parsed["level"] = "ward"
    elif district:
        parsed["level"] = "district"
    else:
        parsed["level"] = "city"
    
    return parsed


def _calculate_confidence(osm_result: Dict, original_query: str, district: Optional[str], ward: Optional[str]) -> float:
    """Calculate confidence score for OSM result."""
    score = 0.5
    
    result_type = osm_result.get("type", "")
    
    type_scores = {
        "city": 0.95, "town": 0.90, "municipality": 0.90,
        "district": 0.85, "suburb": 0.75, "neighbourhood": 0.70,
        "village": 0.65, "hamlet": 0.60,
    }
    
    score = type_scores.get(result_type, 0.5)
    
    if district and ward:
        score = min(0.98, score + 0.08)
    elif district:
        score = min(0.90, score + 0.05)
    
    display_name = osm_result.get("display_name", "").lower()
    if original_query.lower() in display_name:
        score = min(0.98, score + 0.05)
    
    return round(score, 2)


async def reverse_geocode(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Reverse geocode coordinates to get address."""
    try:
        async with aiohttp.ClientSession() as session:
            params = {
                "lat": lat, "lon": lon,
                "format": "json",
                "addressdetails": 1,
                "language": "vi"
            }
            headers = {"User-Agent": "WeatherChatbotHanoi/1.0"}
            
            async with session.get(
                f"{NOMINATIM_BASE_URL}/reverse",
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    return None
                
                result = await resp.json()
                
                if not result:
                    return None
                
                address = result.get("address", {})
                
                return {
                    "status": "ok",
                    "data": {
                        "display_name": result.get("display_name"),
                        "lat": float(result.get("lat", 0)),
                        "lon": float(result.get("lon", 0)),
                        "district": address.get("county") or address.get("city"),
                        "ward": address.get("suburb") or address.get("neighbourhood"),
                    }
                }
                
    except Exception as e:
        logger.error(f"Reverse geocode error: {e}")
        return None


def map_osm_to_ward(district_name: str, ward_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Map OSM location to database ward."""
    from app.db.dal import query_one
    from app.core.normalize import normalize_name
    
    if not district_name:
        return None
    
    norm_district = normalize_name(district_name).replace(" ", "_")
    
    db_district = query_one("""
        SELECT DISTINCT district_name_vi, district_name_norm
        FROM dim_ward 
        WHERE district_name_norm = %s
           OR district_name_norm LIKE CONCAT('%%_', %s)
        LIMIT 1
    """, (norm_district, norm_district))
    
    if db_district:
        if ward_name:
            norm_ward = normalize_name(ward_name).replace(" ", "_")
            db_ward = query_one("""
                SELECT ward_id, ward_name_vi, district_name_vi, lat, lon
                FROM dim_ward 
                WHERE district_name_vi = %s
                  AND (ward_name_norm = %s OR ward_name_norm LIKE CONCAT('%%_', %s))
                LIMIT 1
            """, (db_district["district_name_vi"], norm_ward, norm_ward))
            
            if db_ward:
                return {"status": "exact", "level": "ward", "data": db_ward}
        
        return {"status": "exact", "level": "district", "data": db_district}
    
    return None
