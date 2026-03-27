"""Tier-aware dispatch — core abstraction cho 3-tier (ward/district/city).

Module nay cung cap:
- resolve_and_dispatch(): Ham trung tam xu ly location resolution + tier dispatch
- normalize_agg_keys(): Chuan hoa ten cot aggregate (avg_temp -> temp) de LLM nhan dien nhat quan
- _build_hourly_data_note(), _build_daily_data_note(): Data boundary notes

Moi tool chi can goi resolve_and_dispatch() 1 lan thay vi copy-paste 50 dong boilerplate.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data boundary notes (chong LLM hallucinate ngay/gio khong co trong DB)
# ---------------------------------------------------------------------------

def build_daily_data_note(forecasts: list) -> str:
    """Build data boundary note cho daily forecast — ngan LLM bia them ngay."""
    dates = [str(f.get("date", "")) for f in forecasts if f.get("date")]
    if not dates:
        return "Khong co du lieu du bao."
    return (
        f"\u26a0\ufe0f CHI CO du lieu {len(dates)} ngay: {', '.join(dates)}. "
        "TUYET DOI KHONG bia them ngay khac ngoai danh sach tren."
    )


def build_hourly_data_note(forecasts: list) -> str:
    """Build data boundary note cho hourly forecast — ngan LLM bia them gio."""
    if not forecasts:
        return "Khong co du lieu du bao theo gio."
    times = []
    for f in forecasts:
        t = f.get("time_ict") or f.get("ts_utc")
        if t:
            times.append(str(t))
    if not times:
        return "Khong co du lieu du bao theo gio."
    return (
        f"\u26a0\ufe0f CHI CO du lieu {len(times)} gio: tu {times[0]} den {times[-1]}. "
        "TUYET DOI KHONG bia them gio khac ngoai pham vi tren. "
        "Neu user hoi gio khong co trong data \u2192 NOI RO 'khong co du lieu cho gio do'."
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
    """Chuan hoa aggregate keys (avg_temp -> temp) de downstream xu ly nhat quan.

    Giu lai key goc, chi THEM key moi neu chua co.
    Vi du: {"avg_temp": 30, "min_temp": 28} -> them {"temp": 30}
    """
    for agg_key, ward_key in _AGG_TO_WARD.items():
        if agg_key in row and ward_key not in row:
            row[ward_key] = row[agg_key]
    return row


def normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize mot list rows (forecasts)."""
    return [normalize_agg_keys(r) for r in rows]


# ---------------------------------------------------------------------------
# resolve_and_dispatch — core function thay the moi if/else boilerplate
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

    Thay the ~50 dong boilerplate (auto_resolve + _get_ward_id_or_fallback + if/else)
    ma MOJI tool deu phai lap lai.

    Args:
        ward_id: Ward ID truc tiep
        location_hint: Ten dia diem (VD: "Cau Giay", "Ha Noi")
        default_scope: Scope mac dinh khi khong co location ("city"/"district"/"ward")
        ward_fn: DAL function cho ward level
        district_fn: DAL function cho district level
        city_fn: DAL function cho city level
        ward_args: Extra kwargs cho ward_fn (ngoai ward_id)
        district_args: Extra kwargs cho district_fn (ngoai district_name)
        city_args: Extra kwargs cho city_fn
        enrich_fn: Optional function de enrich result (VD: enrich_weather_response)
        normalize: True de chuan hoa aggregate keys
        fallback_to_ward: True neu tool chi co ward_fn, muon fallback tu district->ward
        label: Label cho data_coverage (VD: "thoi tiet hien tai")

    Returns:
        Dict voi data + metadata (resolved_location, source, level, data_note, ...)
    """
    from app.agent.utils import auto_resolve_location

    # --- Step 1: Resolve location ---
    if not ward_id and not location_hint:
        # Khong co location -> dung default_scope
        level = default_scope
        resolved_data = {"city_name": "Ha Noi"} if level == "city" else {}
    else:
        resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
        if resolved["status"] != "ok":
            return {
                "error": resolved["status"],
                "message": resolved.get("message", "Khong xac dinh duoc dia diem"),
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
            # Fallback: dung 1 ward dai dien
            result, source, resolved_data = _fallback_to_ward(
                resolved_data, ward_fn, ward_args, "city"
            )
        else:
            return {"error": "unsupported_level",
                    "message": f"Tool nay chua ho tro cap thanh pho. Thu hoi theo quan/huyen hoac phuong/xa."}

    elif level == "district":
        district_name = _extract_district_name(resolved_data)
        if district_fn and district_name:
            d_args = dict(district_args or {})
            d_args["district_name"] = district_name
            result = district_fn(**d_args)
            source = "district_aggregated"
        elif fallback_to_ward:
            result, source, resolved_data = _fallback_to_ward(
                resolved_data, ward_fn, ward_args, "district"
            )
        else:
            return {"error": "unsupported_level",
                    "message": f"Tool nay chua ho tro cap quan/huyen. Thu hoi theo phuong/xa cu the."}

    elif level == "ward":
        wid = _extract_ward_id(resolved_data)
        if ward_fn and wid:
            w_args = dict(ward_args or {})
            w_args["ward_id"] = wid
            result = ward_fn(**w_args)
            source = "ward"
        elif not wid:
            return {"error": "need_ward",
                    "message": "Khong xac dinh duoc phuong/xa",
                    "suggestion": "Hay neu ten quan/huyen hoac phuong/xa cu the"}
        else:
            return {"error": "no_handler",
                    "message": "Khong co ham xu ly cho cap phuong/xa"}
    else:
        return {"error": "unknown_level", "message": f"Cap do '{level}' khong duoc ho tro"}

    # --- Step 3: Handle errors from DAL ---
    if result is None:
        return {"error": "no_data", "message": "Khong co du lieu"}

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
    """Extract district_name tu resolved data."""
    return (
        resolved_data.get("district_name_vi")
        or resolved_data.get("district_name")
        or resolved_data.get("district")
    )


def _extract_ward_id(resolved_data: dict) -> Optional[str]:
    """Extract ward_id tu resolved data."""
    return resolved_data.get("ward_id")


def _fallback_to_ward(
    resolved_data: dict,
    ward_fn: Optional[Callable],
    ward_args: Optional[dict],
    from_level: str,
) -> tuple:
    """Fallback tu district/city -> ward dai dien (ward dau tien trong district)."""
    if not ward_fn:
        return (
            {"error": "unsupported_level",
             "message": f"Tool nay khong ho tro cap {from_level}"},
            "error",
            resolved_data,
        )

    from app.dal.location_dal import get_wards_in_district

    if from_level == "city":
        # City fallback: dung ward dau tien cua quan dau tien
        # Thuc te se dung district aggregate hoac city aggregate truoc
        # Day la fallback cuoi cung
        district_name = _extract_district_name(resolved_data) or "Hoan Kiem"
    else:
        district_name = _extract_district_name(resolved_data)

    if not district_name:
        return (
            {"error": "no_district", "message": "Khong xac dinh duoc quan/huyen de fallback"},
            "error",
            resolved_data,
        )

    wards = get_wards_in_district(district_name)
    if not wards:
        return (
            {"error": "no_wards", "message": f"Khong tim thay phuong/xa trong {district_name}"},
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
            f"Du lieu dai dien tu {ward.get('ward_name_vi', '')} "
            f"trong {district_name} (khong phai trung binh toan quan)"
        )
        result["_fallback_ward"] = True

    return result, f"ward_fallback_from_{from_level}", resolved_data


# ---------------------------------------------------------------------------
# Convenience: wrap list results (forecasts) voi metadata
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
    """Wrap forecast list voi metadata chuan.

    Args:
        forecasts: List of forecast dicts
        resolved_data: Resolved location data
        source: "ward" / "district_aggregated" / "city_aggregated"
        level: "ward" / "district" / "city"
        forecast_type: "hourly" or "daily"
        normalize: True de chuan hoa aggregate keys
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
        location_label = "toan Ha Noi"
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
        "data_coverage": f"Du bao {len(forecasts)} {'gio' if forecast_type == 'hourly' else 'ngay'} toi"
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
    """Shortcut cho dispatch forecast (hourly/daily) voi auto-wrap metadata.

    Nhan ra pattern chung:
    1. Resolve location
    2. Goi DAL (ward/district/city)
    3. Wrap voi data_note + coverage
    """
    from app.agent.utils import auto_resolve_location

    # No location -> default scope
    if not ward_id and not location_hint:
        level = default_scope
        resolved_data = {"city_name": "Ha Noi"} if level == "city" else {}
    else:
        resolved = auto_resolve_location(ward_id=ward_id, location_hint=location_hint)
        if resolved["status"] != "ok":
            return {
                "error": resolved["status"],
                "message": resolved.get("message", "Khong xac dinh duoc dia diem"),
                "suggestion": resolved.get("suggestion", ""),
            }
        level = resolved.get("level", "ward")
        resolved_data = resolved.get("data", {})

    # Dispatch
    if level == "city":
        forecasts = city_fn(**(city_args or {}))
        source = "city_aggregated"
    elif level == "district":
        district_name = _extract_district_name(resolved_data)
        if not district_name:
            return {"error": "no_district", "message": "Khong xac dinh duoc quan/huyen"}
        d_args = dict(district_args or {})
        d_args["district_name"] = district_name
        forecasts = district_fn(**d_args)
        source = "district_aggregated"
    else:
        wid = _extract_ward_id(resolved_data)
        if not wid:
            return {"error": "need_ward", "message": "Khong xac dinh duoc phuong/xa"}
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
