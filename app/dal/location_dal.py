"""Location resolution DAL - Fuzzy search for Vietnamese location names."""

from typing import List, Dict, Any, Optional
from app.db.dal import query, query_one
from app.core.normalize import normalize_name


def resolve_location(location_hint: str) -> Dict[str, Any]:
    """Resolve location with 3 levels: exact -> fuzzy -> district.
    
    Args:
        location_hint: User's location input
        
    Returns:
        Dictionary with status and resolved location data
    """
    norm = normalize_name(location_hint)
    
    # 1. Exact match (normalized)
    result = query_one("""
        SELECT ward_id, ward_name_vi, district_name_vi, lat, lon
        FROM dim_ward 
        WHERE ward_name_norm = %s OR district_name_norm = %s
        LIMIT 1
    """, (norm, norm))
    
    if result:
        return {"status": "exact", "data": result}
    
    # 2. Fuzzy match with trigram
    results = query("""
        SELECT ward_id, ward_name_vi, district_name_vi, lat, lon,
               similarity(ward_name_norm, %s) as score
        FROM dim_ward
        WHERE ward_name_norm %% %s
        ORDER BY score DESC
        LIMIT 5
    """, (norm, norm))
    
    if len(results) == 1:
        return {"status": "fuzzy", "data": results[0]}
    elif len(results) > 1:
        return {"status": "multiple", "data": results}
    
    # 3. Try district only
    results = query("""
        SELECT DISTINCT district_name_vi, district_name_norm
        FROM dim_ward
        WHERE district_name_norm %% %s
        LIMIT 5
    """, (norm,))
    
    if results:
        return {"status": "district_only", "data": results}
    
    return {"status": "not_found", "message": f"Khong tim thay '{location_hint}'"}


def get_ward_by_id(ward_id: str) -> Optional[Dict[str, Any]]:
    """Get ward information by ward_id.
    
    Args:
        ward_id: Ward ID
        
    Returns:
        Ward data or None
    """
    return query_one("""
        SELECT ward_id, ward_name_vi, district_name_vi, lat, lon
        FROM dim_ward
        WHERE ward_id = %s
    """, (ward_id,))


def get_all_wards() -> List[Dict[str, Any]]:
    """Get all wards.
    
    Returns:
        List of all wards
    """
    return query("""
        SELECT ward_id, ward_name_vi, district_name_vi, lat, lon
        FROM dim_ward
        ORDER BY district_name_vi, ward_name_vi
    """)


def get_districts() -> List[Dict[str, Any]]:
    """Get all districts.
    
    Returns:
        List of all districts
    """
    return query("""
        SELECT DISTINCT district_name_vi, district_name_norm
        FROM dim_ward
        ORDER BY district_name_vi
    """)


def search_wards(keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search wards by keyword.
    
    Args:
        keyword: Search keyword
        limit: Maximum results
        
    Returns:
        List of matching wards
    """
    norm = normalize_name(keyword)
    return query("""
        SELECT ward_id, ward_name_vi, district_name_vi, lat, lon,
               similarity(ward_name_norm, %s) as score
        FROM dim_ward
        WHERE ward_name_norm %% %s OR district_name_norm %% %s
        ORDER BY score DESC
        LIMIT %s
    """, (norm, norm, norm, limit))
