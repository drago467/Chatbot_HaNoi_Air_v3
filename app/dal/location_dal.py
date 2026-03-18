"""Location resolution DAL - Fuzzy search for Vietnamese location names."""

import asyncio
from typing import List, Dict, Any, Optional
from app.db.dal import query, query_one
from app.core.normalize import normalize_name

# Import OSM module
from app.dal import osm_dal


# Confidence threshold below which we ask user for clarification
AMBIGUOUS_THRESHOLD = 0.6


def resolve_location(location_hint: str) -> Dict[str, Any]:
    """Resolve location with fallback to OSM and user clarification.

    Priority:
    1. Database exact match (high confidence)
    2. OpenStreetMap search (for validation/ambiguation)
    3. Ask user for clarification (if ambiguous or wrong)
    """
    
    # First, try database EXACT match only
    db_result = _resolve_from_database(location_hint)
    
    # If exact match found, return
    if db_result.get('status') == 'exact':
        return db_result
    
    # If not exact match, try OSM for better result
    try:
        osm_result = asyncio.run(osm_dal.search_osm(location_hint))
    except Exception:
        osm_result = None
    
    # If OSM finds something with good confidence, try to map to DB
    if osm_result and osm_result.get('confidence', 0) >= 0.5:
        district_name = osm_result.get('data', {}).get('district_name_vi')
        ward_name = osm_result.get('data', {}).get('ward_name_vi')
        
        if district_name:
            # Try to map OSM result to our database
            mapped = osm_dal.map_osm_to_ward(district_name, ward_name)
            if mapped:
                mapped['osm_confidence'] = osm_result.get('confidence')
                mapped['osm_display_name'] = osm_result.get('data', {}).get('display_name')
                return mapped
    
    # If database has fuzzy result but OSM suggests different location -> check for real conflict
    if db_result.get('status') == 'fuzzy' and osm_result:
        osm_district = osm_result.get('data', {}).get('district_name_vi')
        db_district = db_result.get('data', {}).get('district_name_vi', '')
        
        # Skip if OSM returns city level (e.g., "Thành phố Hà Nội") - not helpful
        skip_ambiguity = ['thành phố hà nội', 'hà nội', 'hanoi']
        if osm_district and osm_district.lower() not in skip_ambiguity:
            # Check if they're actually different
            osm_clean = osm_district.lower().replace('quận ', '').replace('huyện ', '').replace('thành phố ', '')
            db_clean = db_district.lower().replace('quận ', '').replace('huyện ', '').replace('thành phố ', '')
            
            if osm_clean != db_clean:
                return {
                    'status': 'ambiguous',
                    'level': 'district',
                    'needs_clarification': True,
                    'message': f'Co the la {osm_district} hoac {db_district}',
                    'alternatives': [
                        {'district_name_vi': db_district, 'source': 'database'},
                        {'district_name_vi': osm_district, 'source': 'osm'},
                    ],
                    'suggestion': 'Vui long cho biet them (vi du: "quan TenQuan")'
                }
    
    # If we have database fuzzy result, return it
    if db_result.get('status') in ('fuzzy', 'multiple'):
        return db_result
    
    # Not found anywhere - return clarification message
    alternatives = []
    if osm_result and osm_result.get('alternatives'):
        for alt in osm_result['alternatives'][:3]:
            alternatives.append(alt.get('display_name'))
    
    return {
        'status': 'not_found',
        'level': 'not_found',
        'message': f'Khong tim thay dia diem: {location_hint}',
        'needs_clarification': True,
        'alternatives': alternatives,
        'suggestion': 'Vui long cho biet them thong tin (vi du: "quan TenQuan" hoac "phuong TenPhuong")'
    }


def _resolve_from_database(location_hint: str) -> Dict[str, Any]:
    """Original database resolution logic."""
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
    
    # STEP 3: NO PREFIX - Check DISTRICT FIRST
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
    
    # STEP 6: Fuzzy match DISTRICT
    fuzzy_district_results = query("""
        SELECT DISTINCT district_name_vi, district_name_norm
        FROM dim_ward WHERE district_name_norm %% %s LIMIT 5
    """, (norm_with_underscore,))
    
    if fuzzy_district_results:
        return {"status": "fuzzy", "level": "district", "data": fuzzy_district_results[0]}
    
    # STEP 7: Fuzzy match WARD
    fuzzy_ward_results = query("""
        SELECT ward_id, ward_name_vi, district_name_vi, lat, lon,
               similarity(ward_name_norm, %s) as score
        FROM dim_ward WHERE ward_name_norm %% %s ORDER BY score DESC LIMIT 5
    """, (norm_with_underscore, norm_with_underscore))
    
    if len(fuzzy_ward_results) == 1:
        return {"status": "fuzzy", "level": "ward", "data": fuzzy_ward_results[0]}
    elif len(fuzzy_ward_results) > 1:
        return {"status": "multiple", "level": "ward", "data": fuzzy_ward_results}
    
    # STEP 8: City level
    city_keywords = ["ha_noi", "hanoi", "thanh_pho_ha_noi"]
    if norm in city_keywords or norm_with_underscore in city_keywords:
        return {"status": "exact", "level": "city", "data": {"city_name": "Hà Nội"}}
    
    # Not found in database
    return {"status": "not_found", "level": "not_found"}


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


def get_wards_in_district(district_name: str) -> List[Dict[str, Any]]:
    """Get all wards in a district."""
    return query("""
        SELECT ward_id, ward_name_vi, district_name_vi, lat, lon
        FROM dim_ward
        WHERE district_name_vi = %s
        ORDER BY ward_name_vi
    """, (district_name,))


def search_wards(keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
    norm = normalize_name(keyword).replace(' ', '_')
    return query("""
        SELECT ward_id, ward_name_vi, district_name_vi, lat, lon,
               similarity(ward_name_norm, %s) as score
        FROM dim_ward
        WHERE ward_name_norm %% %s OR district_name_norm %% %s
        ORDER BY score DESC LIMIT %s
    """, (norm, norm, norm, limit))
