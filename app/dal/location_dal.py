"""Location resolution DAL - Fuzzy search for Vietnamese location names."""

from typing import List, Dict, Any, Optional
from app.db.dal import query, query_one
from app.core.normalize import normalize_name


def resolve_location(location_hint: str) -> Dict[str, Any]:
    """Resolve location with 4 levels: ward -> district -> city -> not_found.
    
    Priority logic:
    1. If input has "phuong", "xa" prefix → WARD
    2. If input has "quan", "huyen" prefix → DISTRICT  
    3. If input matches district name → DISTRICT (check ends with)
    4. If input matches ward name → WARD
    5. Fuzzy matching
    6. City level (Hà Nội)
    """
    norm = normalize_name(location_hint)
    norm_with_underscore = norm.replace(' ', '_')
    
    # STEP 1: Check for explicit WARD prefix
    ward_prefixes = ['phuong', 'xa']
    if any(norm.lower().startswith(p + ' ') for p in ward_prefixes):
        ward_result = query_one("""
        SELECT ward_id, ward_name_vi, district_name_vi, lat, lon
            FROM dim_ward WHERE ward_name_norm = %s LIMIT 1
        """, (norm_with_underscore,))
        if ward_result:
            return {"status": "exact", "level": "ward", "data": ward_result}
    
    # STEP 2: Check for explicit DISTRICT prefix
    district_prefixes = ['quan', 'huyen']
    if any(norm.lower().startswith(p + ' ') for p in district_prefixes):
        district_result = query_one("""
            SELECT DISTINCT district_name_vi, district_name_norm
            FROM dim_ward WHERE district_name_norm = %s LIMIT 1
        """, (norm_with_underscore,))
        if district_result:
            return {"status": "exact", "level": "district", "data": district_result}
    
    # STEP 3: NO PREFIX - Check DISTRICT FIRST (before ward)
    # "Cầu Giấy" should match "quan_cau_giay" (ends with)
    district_result = query_one("""
        SELECT DISTINCT district_name_vi, district_name_norm
        FROM dim_ward 
        WHERE district_name_norm = %s
           OR district_name_norm LIKE CONCAT('%%_', %s)
        LIMIT 1
    """, (norm_with_underscore, norm_with_underscore))
    
    if district_result:
        return {"status": "exact", "level": "district", "data": district_result}
    
    # STEP 4: Check WARD
    ward_result = query_one("""
        SELECT ward_id, ward_name_vi, district_name_vi, lat, lon
        FROM dim_ward WHERE ward_name_norm = %s LIMIT 1
    """, (norm_with_underscore,))
    
    if ward_result:
        # If ward name equals district name, treat as district
        district_name_only = ward_result["district_name_vi"].replace("Quận ", "").replace("Huyện ", "").replace("Thị xã ", "")
        district_norm = normalize_name(district_name_only).replace(' ', '_')
        
        if norm_with_underscore == district_norm:
            district_result = query_one("""
                SELECT DISTINCT district_name_vi, district_name_norm
                FROM dim_ward WHERE district_name_norm = %s LIMIT 1
            """, (district_norm,))
            if district_result:
                return {"status": "exact", "level": "district", "data": district_result}
        
        return {"status": "exact", "level": "ward", "data": ward_result}
    
    # STEP 5: Contains match for WARD
    ward_result = query_one("""
        SELECT ward_id, ward_name_vi, district_name_vi, lat, lon
        FROM dim_ward WHERE ward_name_norm LIKE %s LIMIT 1
    """, (f"%{norm_with_underscore}%",))
    
    if ward_result:
        return {"status": "fuzzy", "level": "ward", "data": ward_result}
    
    # STEP 6: Fuzzy match DISTRICT (trigram)
    fuzzy_district_results = query("""
        SELECT DISTINCT district_name_vi, district_name_norm
        FROM dim_ward WHERE district_name_norm %% %s LIMIT 5
    """, (norm_with_underscore,))
    
    if fuzzy_district_results:
        return {"status": "fuzzy", "level": "district", "data": fuzzy_district_results[0]}
    
    # STEP 7: Fuzzy match WARD (trigram)
    fuzzy_ward_results = query("""
        SELECT ward_id, ward_name_vi, district_name_vi, lat, lon,
               similarity(ward_name_norm, %s) as score
        FROM dim_ward WHERE ward_name_norm %% %s ORDER BY score DESC LIMIT 5
    """, (norm_with_underscore, norm_with_underscore))
    
    if len(fuzzy_ward_results) == 1:
        return {"status": "fuzzy", "level": "ward", "data": fuzzy_ward_results[0]}
    elif len(fuzzy_ward_results) > 1:
        return {"status": "multiple", "level": "ward", "data": fuzzy_ward_results}
    
    # STEP 8: City level (Hà Nội)
    city_keywords = ["ha_noi", "hanoi", "thanh_pho_ha_noi"]
    if norm in city_keywords or norm_with_underscore in city_keywords:
        return {"status": "exact", "level": "city", "data": {"city_name": "Hà Nội"}}
    
    # Return not_found with explicit level
    return {"status": "not_found", "level": "not_found", "message": f"Khong tim thay '{location_hint}'"}


def get_ward_by_id(ward_id: str) -> Optional[Dict[str, Any]]:
    return query_one("""
        SELECT ward_id, ward_name_vi, district_name_vi, lat, lon
        FROM dim_ward WHERE ward_id = %s
    """, (ward_id,))


def get_all_wards() -> List[Dict[str, Any]]:
    return query("""
        SELECT ward_id, ward_name_vi, district_name_vi, lat, lon
        FROM dim_ward ORDER BY district_name_vi, ward_name_vi
    """)


def get_districts() -> List[Dict[str, Any]]:
    return query("""
        SELECT DISTINCT district_name_vi, district_name_norm
        FROM dim_ward ORDER BY district_name_vi
    """)


def search_wards(keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
    norm = normalize_name(keyword).replace(' ', '_')
    return query("""
        SELECT ward_id, ward_name_vi, district_name_vi, lat, lon,
               similarity(ward_name_norm, %s) as score
        FROM dim_ward
        WHERE ward_name_norm %% %s OR district_name_norm %% %s
        ORDER BY score DESC LIMIT %s
    """, (norm, norm, norm, limit))
