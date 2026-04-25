"""Tier-aware dispatch — core abstraction cho 3-tier (ward/district/city).

Module này cung cấp:
- resolve_and_dispatch(): Hàm trung tâm xử lý location resolution + tier dispatch
- normalize_agg_keys(): Chuẩn hóa tên cột aggregate (avg_temp -> temp) để LLM nhận diện nhất quán
- _build_hourly_data_note(), _build_daily_data_note(): Data boundary notes
"""

from __future__ import annotations

import contextvars
from typing import Any, Callable, Dict, List, Optional

# Scope override from SLM router — set by stream_agent_routed/run_agent_routed.
# When set, dispatch uses this scope instead of resolve_location result.
router_scope_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "router_scope", default=None
)


# ---------------------------------------------------------------------------
# Data boundary notes (chống LLM hallucinate ngày/giờ không có trong DB)
# ---------------------------------------------------------------------------

def build_daily_data_note(forecasts: list) -> str:
    """Build data boundary note cho daily forecast — ngăn LLM bịa thêm ngày."""
    dates = [str(f.get("date", "")) for f in forecasts if f.get("date")]
    if not dates:
        return "Không có dữ liệu dự báo."
    return (
        f"⚠️ CHỈ CÓ dữ liệu {len(dates)} ngày: {', '.join(dates)}. "
        "TUYỆT ĐỐI KHÔNG bịa thêm ngày khác ngoài danh sách trên."
    )


def build_hourly_data_note(forecasts: list) -> str:
    """Build data boundary note cho hourly forecast — ngăn LLM bịa thêm giờ."""
    if not forecasts:
        return "Không có dữ liệu dự báo theo giờ."
    times = []
    for f in forecasts:
        t = f.get("time_ict") or f.get("ts_utc")
        if t:
            times.append(str(t))
    if not times:
        return "Không có dữ liệu dự báo theo giờ."
    return (
        f"⚠️ CHỈ CÓ dữ liệu {len(times)} giờ: từ {times[0]} đến {times[-1]}. "
        "TUYỆT ĐỐI KHÔNG bịa thêm giờ khác ngoài phạm vi trên. "
        "Nếu user hỏi giờ không có trong data → NÓI RÕ 'không có dữ liệu cho giờ đó'."
    )


# ---------------------------------------------------------------------------
# Column normalization: aggregate keys -> ward-like keys
# ---------------------------------------------------------------------------

_AGG_TO_WARD = {
    "avg_temp": "temp",
    "avg_humidity": "humidity",
    "avg_wind_speed": "wind_speed",
    "avg_wind_deg": "wind_deg",
    "avg_dew_point": "dew_point",
    "avg_pressure": "pressure",
    "avg_clouds": "clouds",
    "avg_visibility": "visibility",
    "avg_uvi": "uvi",
    "max_uvi": "uvi_max",
    "avg_pop": "pop",
    "avg_rain_1h": "rain_1h",
    "max_wind_gust": "wind_gust",
    "total_rain": "rain_total",
}


def normalize_agg_keys(row: Dict[str, Any]) -> Dict[str, Any]:
    """Chuẩn hóa aggregate keys (avg_temp -> temp) để downstream xử lý nhất quán.

    Giữ lại key gốc, chỉ THÊM key mới nếu chưa có.
    Ví dụ: {"avg_temp": 30, "min_temp": 28} -> thêm {"temp": 30}
    """
    for agg_key, ward_key in _AGG_TO_WARD.items():
        if agg_key in row and ward_key not in row:
            row[ward_key] = row[agg_key]
    return row


def normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize một list rows (forecasts)."""
    return [normalize_agg_keys(r) for r in rows]


# ---------------------------------------------------------------------------
# resolve_and_dispatch — core function thay thế mọi if/else boilerplate
# ---------------------------------------------------------------------------

def resolve_and_dispatch(
    *,
    ward_id: Optional[str] = None,
    location_hint: Optional[str] = None,
    default_scope: str = "city",
    ward_fn: Optional[Callable] = None,
    district_fn: Optional[Callable] = None,
    city_fn: Optional[Callable] = None,
    ward_args: Optional[dict] = None,
    district_args: Optional[dict] = None,
    city_args: Optional[dict] = None,
    enrich_fn: Optional[Callable] = None,
    normalize: bool = True,
    fallback_to_ward: bool = False,
    label: str = "",
) -> Dict[str, Any]:
    """Resolve location + dispatch to ward/district/city DAL function.

    Thay thế ~50 dòng boilerplate (auto_resolve + _get_ward_id_or_fallback + if/else)
    mà mỗi tool đều phải lặp lại.

    Args:
        ward_id: Ward ID trực tiếp
        location_hint: Tên địa điểm (VD: "Cầu Giấy", "Hà Nội")
        default_scope: Scope mặc định khi không có location ("city"/"district"/"ward")
        ward_fn: DAL function cho ward level
        district_fn: DAL function cho district level
        city_fn: DAL function cho city level
        ward_args: Extra kwargs cho ward_fn (ngoài ward_id)
        district_args: Extra kwargs cho district_fn (ngoài district_id)
        city_args: Extra kwargs cho city_fn
        enrich_fn: Optional function để enrich result (VD: enrich_weather_response)
        normalize: True để chuẩn hóa aggregate keys
        fallback_to_ward: True nếu tool chỉ có ward_fn, muốn fallback từ district->ward
        label: Label cho data_coverage (VD: "thời tiết hiện tại")

    Returns:
        Dict với data + metadata (resolved_location, source, level, data_note, ...)
    """
    from app.agent.utils import auto_resolve_location

    # --- Step 1: Resolve location (scope-guided từ SLM router) ---
    scope = router_scope_var.get(None)

    if not ward_id and not location_hint:
        # Không có location → dùng scope hoặc default_scope
        level = scope or default_scope
        resolved_data = {"city_name": "Hà Nội"} if level == "city" else {}
    else:
        resolved = auto_resolve_location(
            ward_id=ward_id,
            location_hint=location_hint,
            target_scope=scope,
        )
        if resolved["status"] != "ok":
            return {
                "error": resolved["status"],
                "message": resolved.get("message", "Không xác định được địa điểm"),
                "suggestion": resolved.get("suggestion", ""),
                "needs_clarification": resolved.get("needs_clarification", False),
                "alternatives": resolved.get("alternatives", []),
            }
        level = resolved.get("level", "ward")
        resolved_data = resolved.get("data", {})

    # --- Step 2: Dispatch to appropriate DAL function ---
    result = None
    source = "raw"

    if level == "city":
        if city_fn:
            result = city_fn(**(city_args or {}))
            source = "city_aggregated"
        elif fallback_to_ward:
            # Fallback: dùng 1 ward đại diện
            result, source, resolved_data = _fallback_to_ward(
                resolved_data, ward_fn, ward_args, "city"
            )
        else:
            return {"error": "unsupported_level",
                    "message": "Tool này chưa hỗ trợ cấp thành phố. Thử hỏi theo quận/huyện hoặc phường/xã."}

    elif level == "district":
        district_id = _extract_district_id(resolved_data)
        if district_fn and district_id:
            d_args = dict(district_args or {})
            d_args["district_id"] = district_id
            result = district_fn(**d_args)
            source = "district_aggregated"
        elif fallback_to_ward:
            result, source, resolved_data = _fallback_to_ward(
                resolved_data, ward_fn, ward_args, "district"
            )
        else:
            return {"error": "unsupported_level",
                    "message": "Tool này chưa hỗ trợ cấp quận/huyện. Thử hỏi theo phường/xã cụ thể."}

    elif level == "ward":
        wid = _extract_ward_id(resolved_data)
        if ward_fn and wid:
            w_args = dict(ward_args or {})
            w_args["ward_id"] = wid
            result = ward_fn(**w_args)
            source = "ward"
        elif not wid:
            return {"error": "need_ward",
                    "message": "Không xác định được phường/xã",
                    "suggestion": "Hãy nêu tên quận/huyện hoặc phường/xã cụ thể"}
        else:
            return {"error": "no_handler",
                    "message": "Không có hàm xử lý cho cấp phường/xã"}
    else:
        return {"error": "unknown_level", "message": f"Cấp độ '{level}' không được hỗ trợ"}

    # --- Step 3: Handle errors from DAL ---
    if result is None:
        return {"error": "no_data", "message": "Không có dữ liệu"}

    if isinstance(result, dict) and result.get("error"):
        return result

    # --- Step 4: Enrich + normalize ---
    if isinstance(result, dict):
        if normalize and level in ("district", "city"):
            result = normalize_agg_keys(result)
        if enrich_fn:
            result = enrich_fn(result)
        result["resolved_location"] = resolved_data
        result["source"] = source
        result["level"] = level
    elif isinstance(result, list):
        if normalize and level in ("district", "city"):
            result = normalize_rows(result)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_district_name(resolved_data: dict) -> Optional[str]:
    """Extract district_name từ resolved data."""
    return (
        resolved_data.get("district_name_vi")
        or resolved_data.get("district_name")
        or resolved_data.get("district")
    )


def _extract_district_id(resolved_data: dict) -> Optional[int]:
    """Extract district_id (integer FK) từ resolved data."""
    did = resolved_data.get("district_id")
    if did is None:
        return None
    try:
        return int(did)
    except (TypeError, ValueError):
        return None


def _extract_ward_id(resolved_data: dict) -> Optional[str]:
    """Extract ward_id từ resolved data."""
    return resolved_data.get("ward_id")


def _fallback_to_ward(
    resolved_data: dict,
    ward_fn: Optional[Callable],
    ward_args: Optional[dict],
    from_level: str,
) -> tuple:
    """Fallback từ district/city -> ward đại diện (ward đầu tiên trong district)."""
    if not ward_fn:
        return (
            {"error": "unsupported_level",
             "message": f"Tool này không hỗ trợ cấp {from_level}"},
            "error",
            resolved_data,
        )

    from app.dal.location_dal import get_wards_in_district

    if from_level == "city":
        # City fallback: dùng ward đầu tiên của quận đầu tiên
        # Thực tế sẽ dùng district aggregate hoặc city aggregate trước
        # Đây là fallback cuối cùng
        district_name = _extract_district_name(resolved_data) or "Hoàn Kiếm"
    else:
        district_name = _extract_district_name(resolved_data)

    if not district_name:
        return (
            {"error": "no_district", "message": "Không xác định được quận/huyện để fallback"},
            "error",
            resolved_data,
        )

    wards = get_wards_in_district(district_name)
    if not wards:
        return (
            {"error": "no_wards", "message": f"Không tìm thấy phường/xã trong {district_name}"},
            "error",
            resolved_data,
        )

    ward = wards[0]
    w_args = dict(ward_args or {})
    w_args["ward_id"] = ward["ward_id"]
    result = ward_fn(**w_args)

    # Add fallback note
    if isinstance(result, dict) and "error" not in result:
        result["_fallback_note"] = (
            f"Dữ liệu đại diện từ {ward.get('ward_name_vi', '')} "
            f"trong {district_name} (không phải trung bình toàn quận)"
        )
        result["_fallback_ward"] = True

    return result, f"ward_fallback_from_{from_level}", resolved_data


# ---------------------------------------------------------------------------
# Convenience: wrap list results (forecasts) với metadata
# ---------------------------------------------------------------------------

def wrap_forecast_result(
    forecasts: list,
    *,
    resolved_data: dict,
    source: str,
    level: str,
    forecast_type: str = "hourly",
    normalize: bool = True,
) -> Dict[str, Any]:
    """Wrap forecast list với metadata chuẩn.

    Args:
        forecasts: List of forecast dicts
        resolved_data: Resolved location data
        source: "ward" / "district_aggregated" / "city_aggregated"
        level: "ward" / "district" / "city"
        forecast_type: "hourly" or "daily"
        normalize: True để chuẩn hóa aggregate keys
    """
    if normalize and level in ("district", "city"):
        forecasts = normalize_rows(forecasts)

    if forecast_type == "hourly":
        data_note = build_hourly_data_note(forecasts)
    else:
        data_note = build_daily_data_note(forecasts)

    # Build coverage label
    location_label = ""
    if level == "city":
        location_label = "toàn Hà Nội"
    elif level == "district":
        dn = _extract_district_name(resolved_data) or ""
        location_label = f"quan {dn}" if dn else ""
    else:
        wn = resolved_data.get("ward_name_vi", "")
        location_label = wn

    return {
        "forecasts": forecasts,
        "count": len(forecasts),
        "resolved_location": resolved_data,
        "source": source,
        "level": level,
        "data_coverage": f"Dự báo {len(forecasts)} {'giờ' if forecast_type == 'hourly' else 'ngày'} tới"
                         + (f" ({location_label})" if location_label else ""),
        "data_note": data_note,
    }


def dispatch_forecast(
    *,
    ward_id: Optional[str] = None,
    location_hint: Optional[str] = None,
    ward_fn: Callable,
    district_fn: Callable,
    city_fn: Callable,
    ward_args: Optional[dict] = None,
    district_args: Optional[dict] = None,
    city_args: Optional[dict] = None,
    forecast_type: str = "hourly",
    default_scope: str = "city",
) -> Dict[str, Any]:
    """Shortcut cho dispatch forecast (hourly/daily) với auto-wrap metadata.

    Nhận ra pattern chung:
    1. Resolve location
    2. Gọi DAL (ward/district/city)
    3. Wrap với data_note + coverage
    """
    from app.agent.utils import auto_resolve_location

    # Resolve location (scope-guided từ SLM router)
    scope = router_scope_var.get(None)

    if not ward_id and not location_hint:
        level = scope or default_scope
        resolved_data = {"city_name": "Hà Nội"} if level == "city" else {}
    else:
        resolved = auto_resolve_location(
            ward_id=ward_id,
            location_hint=location_hint,
            target_scope=scope,
        )
        if resolved["status"] != "ok":
            return {
                "error": resolved["status"],
                "message": resolved.get("message", "Không xác định được địa điểm"),
                "suggestion": resolved.get("suggestion", ""),
            }
        level = resolved.get("level", "ward")
        resolved_data = resolved.get("data", {})

    # Dispatch
    if level == "city":
        forecasts = city_fn(**(city_args or {}))
        source = "city_aggregated"
    elif level == "district":
        district_id = _extract_district_id(resolved_data)
        if not district_id:
            return {"error": "no_district", "message": "Không xác định được quận/huyện"}
        d_args = dict(district_args or {})
        d_args["district_id"] = district_id
        forecasts = district_fn(**d_args)
        source = "district_aggregated"
    else:
        wid = _extract_ward_id(resolved_data)
        if not wid:
            return {"error": "need_ward", "message": "Không xác định được phường/xã"}
        w_args = dict(ward_args or {})
        w_args["ward_id"] = wid
        forecasts = ward_fn(**w_args)
        source = "ward"

    # Guard error
    if isinstance(forecasts, dict) and forecasts.get("error"):
        return forecasts
    if not isinstance(forecasts, list):
        forecasts = []

    return wrap_forecast_result(
        forecasts,
        resolved_data=resolved_data,
        source=source,
        level=level,
        forecast_type=forecast_type,
    )
