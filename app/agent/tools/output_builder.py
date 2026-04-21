"""Flat Vietnamese tool output builders.

Mỗi builder biến raw dict từ DAL/dispatch → flat dict có:
- key = mô tả tiếng Việt ("nhiệt độ", "xác suất mưa", ...)
- value = chuỗi combined "<nhãn> <số> <đơn vị>"

LLM nhận output đã "thành câu" → chỉ copy/reshape, không tự diễn giải.

Dùng cho 27 tools chia 4 nhóm:
- Group 1 (11): full weather data builders (dưới)
- Group 2 (8): shape_labeled_dict với key_map
- Group 3 (5): window-list builders
- Group 4 (3): shape_ranking_output
"""

from __future__ import annotations

from datetime import datetime, date as _date_cls
from typing import Any, Dict, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from app.dal.weather_helpers import (
    compute_heat_index,
    compute_wind_chill,
    get_dew_point_status,
    get_pressure_status,
    get_uv_status,
    label_clouds,
    label_rain_intensity,
    label_rain_probability,
    label_rain_total,
    label_temp_hn,
    weather_main_to_vietnamese,
    wind_beaufort_vietnamese,
    wind_deg_to_vietnamese,
    wind_speed_to_beaufort,
)

_ICT = ZoneInfo("Asia/Ho_Chi_Minh")

_WEEKDAYS_VI = {
    0: "Thứ Hai", 1: "Thứ Ba", 2: "Thứ Tư", 3: "Thứ Năm",
    4: "Thứ Sáu", 5: "Thứ Bảy", 6: "Chủ Nhật",
}

_LEVEL_SUFFIX = {
    "ward": "phường/xã",
    "district": "quận",
    "city": "Hà Nội (toàn thành phố)",
}

# ── Shared helpers ──────────────────────────────────────────────────────────

def _as_date(v: Any) -> Optional[_date_cls]:
    if v is None:
        return None
    if isinstance(v, _date_cls) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, str):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(v, fmt).date()
            except ValueError:
                pass
    return None


def _format_date_vi(d: Any) -> str:
    dt = _as_date(d)
    if not dt:
        return str(d) if d is not None else ""
    return f"{dt.strftime('%d/%m/%Y')} ({_WEEKDAYS_VI[dt.weekday()]})"


def _format_dt_ict(v: Any) -> str:
    """Unix ts / ISO string / datetime → 'HH:MM Thứ X NGÀY/THÁNG/NĂM'."""
    if v is None:
        return ""
    dt: Optional[datetime] = None
    if isinstance(v, datetime):
        dt = v.astimezone(_ICT) if v.tzinfo else v.replace(tzinfo=_ICT)
    elif isinstance(v, (int, float)):
        dt = datetime.fromtimestamp(float(v), tz=_ICT)
    elif isinstance(v, str):
        raw = v.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(raw)
            dt = parsed.astimezone(_ICT) if parsed.tzinfo else parsed.replace(tzinfo=_ICT)
        except ValueError:
            return v
    if not dt:
        return str(v)
    return f"{dt.strftime('%H:%M')} {_WEEKDAYS_VI[dt.weekday()]} {dt.strftime('%d/%m/%Y')}"


def _format_hour_short(v: Any) -> str:
    """Unix ts / ISO → 'HH:MM NGÀY/THÁNG'."""
    if v is None:
        return ""
    dt: Optional[datetime] = None
    if isinstance(v, datetime):
        dt = v.astimezone(_ICT) if v.tzinfo else v.replace(tzinfo=_ICT)
    elif isinstance(v, (int, float)):
        dt = datetime.fromtimestamp(float(v), tz=_ICT)
    elif isinstance(v, str):
        # Already formatted like "HH:MM (ICT)" → keep as-is
        if "(ICT)" in v or ":" in v and len(v) <= 12:
            return v
        raw = v.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(raw)
            dt = parsed.astimezone(_ICT) if parsed.tzinfo else parsed.replace(tzinfo=_ICT)
        except ValueError:
            return v
    if not dt:
        return str(v)
    return f"{dt.strftime('%H:%M')} {dt.strftime('%d/%m')}"


def _format_time_only(v: Any) -> str:
    """Unix ts / ISO → 'HH:MM'."""
    if v is None:
        return ""
    if isinstance(v, datetime):
        dt = v.astimezone(_ICT) if v.tzinfo else v.replace(tzinfo=_ICT)
        return dt.strftime("%H:%M")
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(float(v), tz=_ICT).strftime("%H:%M")
    if isinstance(v, str):
        if ":" in v and len(v) <= 12:
            return v.split(" ")[0]  # "HH:MM (ICT)" → "HH:MM"
        raw = v.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(raw)
            dt = parsed.astimezone(_ICT) if parsed.tzinfo else parsed.replace(tzinfo=_ICT)
            return dt.strftime("%H:%M")
        except ValueError:
            return v
    return str(v)


def _format_location(
    resolved: Optional[Mapping[str, Any]],
    level: Optional[str],
    fallback_name: Optional[str] = None,
) -> str:
    """Tên địa điểm + cấp độ."""
    if not resolved:
        resolved = {}
    ward = resolved.get("ward_name_vi")
    district = resolved.get("district_name_vi")
    city = resolved.get("city_name") or "Hà Nội"
    if level == "ward" and ward:
        parts = [ward]
        if district:
            parts.append(f"{district}, {city}")
        return " — ".join(parts)
    if level == "district" and district:
        return f"{district} (quận/huyện)"
    if level == "city":
        return f"{city} (toàn thành phố)"
    return fallback_name or ward or district or city or "Hà Nội"


def _wind_text(speed: Optional[float], gust: Optional[float], deg: Optional[float]) -> str:
    """Combined 'Gió vừa cấp 4 (5.5 m/s), giật 9.4 m/s, hướng Nam'."""
    parts: List[str] = []
    if speed is not None:
        bf = wind_speed_to_beaufort(speed)
        label = wind_beaufort_vietnamese(bf)
        parts.append(f"{label} cấp {bf} ({speed:.1f} m/s)")
    if gust is not None and (speed is None or gust > speed + 0.5):
        parts.append(f"giật {gust:.1f} m/s")
    if deg is not None:
        parts.append(f"hướng {wind_deg_to_vietnamese(int(deg))}")
    return ", ".join(parts) if parts else "Không xác định"


def _wind_gust_only(gust: Optional[float], deg: Optional[float]) -> str:
    """Cho history daily: chỉ có wind_gust, không bịa avg."""
    parts: List[str] = []
    if gust is not None:
        parts.append(f"Giật {gust:.1f} m/s")
    if deg is not None:
        parts.append(f"hướng {wind_deg_to_vietnamese(int(deg))}")
    return ", ".join(parts) if parts else "Không xác định"


def _add_conditional_comfort(result: Dict[str, Any], temp: Optional[float],
                             humidity: Optional[float], wind_ms: Optional[float]) -> None:
    """Thêm 'cảm giác nóng'/'cảm giác lạnh' CHỈ khi ngưỡng trigger."""
    if temp is not None and humidity is not None and temp >= 27 and humidity >= 40:
        hi = compute_heat_index(temp, int(humidity))
        if hi and hi["level"] != "An toàn":
            result["cảm giác nóng"] = f"{hi['level']} {hi['heat_index']:.1f}°C"
    if temp is not None and wind_ms is not None and temp <= 10 and wind_ms >= 1.3:
        wc = compute_wind_chill(temp, wind_ms)
        if wc:
            result["cảm giác lạnh"] = f"{wc['level']} {wc['wind_chill']:.1f}°C"


def _add_visibility(result: Dict[str, Any], vis_m: Optional[float]) -> None:
    """Chỉ thêm key tầm nhìn khi <5km (đáng chú ý)."""
    if vis_m is not None and vis_m > 0 and vis_m < 5000:
        km = vis_m / 1000
        if km < 1:
            label = "Kém"
        elif km < 3:
            label = "Hạn chế"
        else:
            label = "Trung bình"
        result["tầm nhìn"] = f"{label} {km:.1f} km"


def _pick_temp(raw: Mapping[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        v = raw.get(k)
        if v is not None:
            return float(v)
    return None


def _narrative_current(raw: Mapping[str, Any], location_name: str,
                       temp: Optional[float], clouds: Optional[float],
                       pop_pct: Optional[float], rain_1h: Optional[float]) -> str:
    bits: List[str] = []
    time_str = _format_time_only(raw.get("time_ict") or raw.get("ts_utc"))
    if time_str:
        bits.append(f"Lúc {time_str}")
    bits.append(f"tại {location_name}")
    wm = weather_main_to_vietnamese(raw.get("weather_main") or "")
    if wm:
        bits.append(wm.lower())
    if temp is not None:
        bits.append(f"nhiệt độ {temp:.1f}°C")
    if rain_1h and rain_1h >= 0.1:
        bits.append(f"đang {label_rain_intensity(rain_1h).lower()}")
    elif pop_pct is not None and pop_pct >= 40:
        bits.append(f"xác suất mưa {int(round(pop_pct))}%")
    if clouds is not None:
        bits.append(f"mây {int(round(clouds))}%")
    text = ", ".join(bits)
    return (text[:1].upper() + text[1:] + ".") if text else ""


# ── Error ───────────────────────────────────────────────────────────────────

def build_error_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Chuẩn hoá {error, message, note?, suggestion?} → flat VN."""
    result: Dict[str, Any] = {
        "lỗi": raw.get("message") or raw.get("error") or "Không có dữ liệu",
    }
    if raw.get("note"):
        result["ghi chú"] = raw["note"]
    if raw.get("suggestion"):
        result["gợi ý"] = raw["suggestion"]
    if raw.get("data_stale"):
        result["ghi chú"] = (result.get("ghi chú") or "") + " Dữ liệu có thể cũ."
    return result


def _is_error(raw: Any) -> bool:
    return isinstance(raw, Mapping) and (raw.get("error") or raw.get("status") == "error")


# ── Group 1: full weather data builders ─────────────────────────────────────

def build_current_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """get_current_weather → flat VN dict."""
    if _is_error(raw):
        return build_error_output(raw)

    level = raw.get("level") or "ward"
    resolved = raw.get("resolved_location") or {}
    location_name = _format_location(resolved, level)

    temp = _pick_temp(raw, "temp", "avg_temp")
    feels_like = raw.get("feels_like")
    humidity = raw.get("humidity") or raw.get("avg_humidity")
    dew_point = raw.get("dew_point") or raw.get("avg_dew_point")
    pressure = raw.get("pressure") or raw.get("avg_pressure")
    wind_speed = raw.get("wind_speed") or raw.get("avg_wind_speed")
    wind_gust = raw.get("wind_gust") or raw.get("max_wind_gust")
    wind_deg = raw.get("wind_deg") or raw.get("avg_wind_deg")
    clouds = raw.get("clouds") or raw.get("avg_clouds")
    uvi = raw.get("uvi") or raw.get("avg_uvi")
    pop = raw.get("pop") or raw.get("avg_pop")
    rain_1h = raw.get("rain_1h") or raw.get("avg_rain_1h")
    visibility = raw.get("visibility") or raw.get("avg_visibility")
    weather_main = raw.get("weather_main")

    result: Dict[str, Any] = {
        "địa điểm": location_name,
        "thời điểm": _format_dt_ict(raw.get("time_ict") or raw.get("ts_utc")),
        "thời tiết chung": weather_main_to_vietnamese(weather_main or "") or "Không xác định",
        "nhiệt độ": label_temp_hn(temp),
    }
    if humidity is not None:
        result["độ ẩm"] = f"{int(round(humidity))}%"
    if level == "ward" and feels_like is not None:
        result["cảm giác"] = f"{label_temp_hn(feels_like)}"
    if dew_point is not None:
        result["điểm sương"] = f"{get_dew_point_status(dew_point)} {dew_point:.1f}°C"
    if pop is not None:
        result["xác suất mưa"] = label_rain_probability(pop)
    if rain_1h is not None and rain_1h >= 0.05:
        result["cường độ mưa hiện tại"] = label_rain_intensity(rain_1h)
    result["gió"] = _wind_text(wind_speed, wind_gust, wind_deg)
    if clouds is not None:
        result["mây"] = label_clouds(clouds)
    if uvi is not None:
        result["UV"] = f"{get_uv_status(uvi)} {uvi:.1f}"
    if pressure is not None:
        result["áp suất"] = f"{get_pressure_status(int(pressure))} {float(pressure):.1f} hPa"
    _add_conditional_comfort(result, temp, humidity, wind_speed)
    _add_visibility(result, visibility)

    pop_pct = pop * 100 if pop is not None else None
    result["tóm tắt"] = _narrative_current(raw, location_name, temp, clouds, pop_pct, rain_1h)
    result["gợi ý dùng output"] = (
        "Đây là SNAPSHOT tại thời điểm hiện tại. CHỈ dùng khi user hỏi 'bây giờ / hiện tại / đang / lúc này'. "
        "Nếu user hỏi 'chiều/tối/đêm/sáng mai/ngày mai/cuối tuần' → KHÔNG dùng số ở đây; "
        "gọi get_hourly_forecast (1-48h) hoặc get_daily_forecast. "
        "KHÔNG dán nhiệt độ/gió hiện tại làm 'nhiệt độ tối nay' hay 'gió giật mạnh nhất cả ngày'."
    )
    if raw.get("data_stale"):
        result["ghi chú dữ liệu"] = raw.get("data_warning") or "Dữ liệu có thể cũ hơn thường lệ."
    return result


def _build_hourly_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
    temp = _pick_temp(entry, "temp", "avg_temp")
    pop = entry.get("pop") or entry.get("avg_pop")
    rain_1h = entry.get("rain_1h") or entry.get("avg_rain_1h")
    clouds = entry.get("clouds") or entry.get("avg_clouds")
    wind_speed = entry.get("wind_speed") or entry.get("avg_wind_speed")
    wind_gust = entry.get("wind_gust") or entry.get("max_wind_gust")
    wind_deg = entry.get("wind_deg") or entry.get("avg_wind_deg")
    humidity = entry.get("humidity") or entry.get("avg_humidity")

    out: Dict[str, Any] = {
        "thời điểm": _format_hour_short(entry.get("time_ict") or entry.get("ts_utc")),
        "thời tiết": weather_main_to_vietnamese(entry.get("weather_main") or "") or "—",
        "nhiệt độ": label_temp_hn(temp),
    }
    if humidity is not None:
        out["độ ẩm"] = f"{int(round(humidity))}%"
    if pop is not None:
        out["xác suất mưa"] = label_rain_probability(pop)
    if rain_1h is not None and rain_1h >= 0.05:
        out["cường độ mưa"] = label_rain_intensity(rain_1h)
    out["gió"] = _wind_text(wind_speed, wind_gust, wind_deg)
    if clouds is not None:
        out["mây"] = label_clouds(clouds)
    return out


def _narrative_hourly(forecasts: Sequence[Mapping[str, Any]], location_name: str) -> str:
    if not forecasts:
        return ""
    temps = [_pick_temp(f, "temp", "avg_temp") for f in forecasts]
    temps = [t for t in temps if t is not None]
    pops = [f.get("pop") or f.get("avg_pop") or 0 for f in forecasts]
    rain_hours = sum(1 for f in forecasts
                     if (f.get("rain_1h") or f.get("avg_rain_1h") or 0) >= 0.1)
    max_pop = int(round(max(pops) * 100)) if pops else 0
    bits = [f"{len(forecasts)} giờ tới tại {location_name}"]
    if temps:
        bits.append(f"nhiệt {min(temps):.0f}–{max(temps):.0f}°C")
    if rain_hours:
        bits.append(f"{rain_hours} giờ có mưa")
    elif max_pop >= 40:
        bits.append(f"xác suất mưa cao nhất {max_pop}%")
    else:
        bits.append("không mưa")
    return ", ".join(bits) + "."


def _detect_forecast_range_gap(
    forecasts: Sequence[Mapping[str, Any]],
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Detect gap giữa forecast start và NOW → past-frame warning nếu có.

    Trả dict với các key optional:
    - "phạm vi thực tế": mô tả start→end (luôn có nếu forecasts ≥1)
    - "⚠ lưu ý khung đã qua": cảnh báo past-frame nếu data không cover các khung
      sáng/trưa/chiều/tối HÔM NAY đã qua.

    Lý do: tool hourly/rain_timeline chỉ trả forecast TỪ NOW. Khi NOW=22:14 tối,
    các khung "chiều/trưa/sáng nay" (13-18h / 11-13h / 6-11h) ĐÃ QUA hoàn toàn.
    Bot thường lấy data tương lai (23:00+ hoặc ngày mai) rồi dán nhãn khung đã
    qua → sai nghiêm trọng (audit v8 B1, 15 IDs).

    Args:
        forecasts: list forecast entries (có ts_utc).
        now: optional datetime override để test. Mặc định datetime.now(ICT).
    """
    result: Dict[str, Any] = {}
    if not forecasts:
        return result

    def _dt_of(entry: Mapping[str, Any]) -> Optional[datetime]:
        ts = entry.get("ts_utc")
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(float(ts), tz=_ICT)
        return None

    first_dt = _dt_of(forecasts[0])
    last_dt = _dt_of(forecasts[-1])
    if first_dt is None or last_dt is None:
        return result

    result["phạm vi thực tế"] = (
        f"từ {first_dt.strftime('%H:%M')} {_WEEKDAYS_VI[first_dt.weekday()]} "
        f"{first_dt.strftime('%d/%m/%Y')} đến {last_dt.strftime('%H:%M')} "
        f"{_WEEKDAYS_VI[last_dt.weekday()]} {last_dt.strftime('%d/%m/%Y')}"
    )

    now = now or datetime.now(_ICT)
    today = now.date()
    # Data có cover khung "quá khứ trong hôm nay" không?
    # Nếu first_dt > NOW (data chỉ forecast tương lai) → các khung đã qua KHÔNG cover.
    data_covers_today_past = first_dt.date() < today or (
        first_dt.date() == today and first_dt.hour <= now.hour
    )

    if data_covers_today_past:
        return result  # Data cover được → không cần warn

    # Liệt kê khung đã qua so với NOW (dựa giờ)
    past_frames: List[str] = []
    if now.hour >= 12:
        past_frames.append("sáng nay (6-11h)")
    if now.hour >= 14:
        past_frames.append("trưa nay (11-13h)")
    if now.hour >= 19:
        past_frames.append("chiều nay (13-18h)")
    if now.hour >= 23:
        past_frames.append("tối nay (18-22h)")

    if past_frames:
        result["⚠ lưu ý khung đã qua"] = (
            f"NOW={now.strftime('%H:%M %d/%m')}. Data forecast chỉ bắt đầu từ "
            f"{first_dt.strftime('%H:%M %d/%m')}. Các khung HÔM NAY đã qua: "
            f"{', '.join(past_frames)}. TOOL NÀY KHÔNG COVER khung đã qua — "
            f"nếu user hỏi về các khung đó → BÁO RÕ 'khung [X] hôm nay đã qua "
            f"(hiện là {now.strftime('%H:%M')})', TUYỆT ĐỐI KHÔNG dùng data "
            f"ngày mai dán nhãn khung hôm nay."
        )
    return result


def build_hourly_forecast_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    level = raw.get("level") or "ward"
    location_name = _format_location(raw.get("resolved_location") or {}, level)
    forecasts = raw.get("forecasts") or []

    entries = [_build_hourly_entry(f) for f in forecasts]
    result: Dict[str, Any] = {
        "địa điểm": location_name,
        "loại dự báo": raw.get("data_coverage") or f"{len(entries)} giờ tới",
        "dự báo": entries,
    }
    # Past-frame detection: thêm "phạm vi thực tế" + cảnh báo nếu khung đã qua
    result.update(_detect_forecast_range_gap(forecasts))
    result["tóm tắt tổng"] = _narrative_hourly(forecasts, location_name)
    if raw.get("data_note"):
        result["ghi chú dữ liệu"] = raw["data_note"]
    return result


def _build_daily_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
    date_v = entry.get("date")
    temp_min = entry.get("temp_min")
    temp_max = entry.get("temp_max")
    temp_avg = entry.get("temp_avg") or entry.get("temp") or entry.get("avg_temp")
    humidity = entry.get("humidity") or entry.get("avg_humidity")
    pop = entry.get("pop") or entry.get("avg_pop")
    rain_total = entry.get("rain_total") or entry.get("total_rain")
    uvi = entry.get("uvi") or entry.get("max_uvi") or entry.get("uvi_max")
    wind_speed = entry.get("wind_speed") or entry.get("avg_wind_speed")
    wind_gust = entry.get("wind_gust") or entry.get("max_wind_gust")
    wind_deg = entry.get("wind_deg") or entry.get("avg_wind_deg")
    weather_main = entry.get("weather_main")
    summary_ai = entry.get("summary")

    out: Dict[str, Any] = {
        "ngày": _format_date_vi(date_v),
        "thời tiết": weather_main_to_vietnamese(weather_main or "") or "—",
    }
    if temp_min is not None and temp_max is not None:
        tb = f" (TB {temp_avg:.1f}°C)" if temp_avg is not None else ""
        out["nhiệt độ"] = f"Thấp {temp_min:.1f}°C — Cao {temp_max:.1f}°C{tb}"
    elif temp_avg is not None:
        out["nhiệt độ"] = label_temp_hn(temp_avg)

    # Daily structured sub-temps (morn/day/eve/night) — ward forecast có từ OWM
    morn = entry.get("temp_morn")
    day = entry.get("temp_day")
    eve = entry.get("temp_eve")
    night = entry.get("temp_night")
    if any(x is not None for x in (morn, day, eve, night)):
        parts = []
        if morn is not None: parts.append(f"Sáng {morn:.1f}°C")
        if day is not None: parts.append(f"Trưa {day:.1f}°C")
        if eve is not None: parts.append(f"Chiều {eve:.1f}°C")
        if night is not None: parts.append(f"Tối {night:.1f}°C")
        out["nhiệt độ theo ngày"] = " / ".join(parts)

    if humidity is not None:
        out["độ ẩm"] = f"{int(round(humidity))}%"
    if pop is not None:
        out["xác suất mưa"] = label_rain_probability(pop)
    if rain_total is not None and rain_total >= 0.05:
        out["tổng lượng mưa"] = label_rain_total(rain_total)
    out["gió"] = _wind_text(wind_speed, wind_gust, wind_deg)
    if uvi is not None:
        out["UV"] = f"{get_uv_status(uvi)} {uvi:.1f}"

    sr = entry.get("sunrise_time") or entry.get("sunrise")
    ss = entry.get("sunset_time") or entry.get("sunset")
    sr_s = _format_time_only(sr) if sr else ""
    ss_s = _format_time_only(ss) if ss else ""
    if sr_s and ss_s:
        out["mọc-lặn"] = f"Mặt trời {sr_s} → {ss_s}"

    if summary_ai:
        out["tóm tắt"] = str(summary_ai)
    return out


def _compute_daily_superlatives(forecasts: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Pre-compute argmax/argmin trên list ngày forecast.

    Trả dict VN key → "DD/MM (Thứ X): <label> <value>".
    LLM không phải argmax qua list — chỉ copy key.
    """
    if not forecasts:
        return {}

    def _day_label(f: Mapping[str, Any]) -> str:
        date_v = _as_date(f.get("date"))
        if not date_v:
            return str(f.get("date", ""))
        return f"{date_v.strftime('%d/%m')} ({_WEEKDAYS_VI[date_v.weekday()]})"

    def _safe_num(f: Mapping[str, Any], *keys: str) -> Optional[float]:
        for k in keys:
            v = f.get(k)
            if v is not None:
                return float(v)
        return None

    result: Dict[str, str] = {}

    # Ngày nóng nhất (theo temp_max)
    hottest = max(
        ((f, _safe_num(f, "temp_max")) for f in forecasts),
        key=lambda x: x[1] if x[1] is not None else -999,
        default=(None, None),
    )
    if hottest[1] is not None:
        result["ngày nóng nhất"] = f"{_day_label(hottest[0])}: Cao {hottest[1]:.1f}°C"

    # Ngày mát nhất (theo temp_min)
    coolest = min(
        ((f, _safe_num(f, "temp_min")) for f in forecasts),
        key=lambda x: x[1] if x[1] is not None else 999,
        default=(None, None),
    )
    if coolest[1] is not None:
        result["ngày mát nhất"] = f"{_day_label(coolest[0])}: Thấp {coolest[1]:.1f}°C"

    # Ngày mưa nhiều nhất (theo rain_total)
    rainiest = max(
        ((f, _safe_num(f, "rain_total", "total_rain") or 0) for f in forecasts),
        key=lambda x: x[1],
        default=(None, 0),
    )
    if rainiest[0] is not None and rainiest[1] > 0.05:
        result["ngày mưa nhiều nhất"] = f"{_day_label(rainiest[0])}: {label_rain_total(rainiest[1])}"

    # Ngày ít mưa nhất / khô nhất — chỉ liệt kê khi có >= 2 ngày có data
    def _rain_of(f: Mapping[str, Any]) -> float:
        return _safe_num(f, "rain_total", "total_rain") or 0
    def _pop_of(f: Mapping[str, Any]) -> float:
        return _safe_num(f, "pop") or 0

    rains_per_day = [(f, _rain_of(f), _pop_of(f)) for f in forecasts]
    driest = min(rains_per_day, key=lambda x: (x[1], x[2]))
    if len(forecasts) >= 2 and driest[0] is not None:
        rain_txt = label_rain_total(driest[1]) if driest[1] >= 0.05 else "Không mưa"
        pop_txt = label_rain_probability(driest[2]) if driest[2] else ""
        tail = f" ({pop_txt})" if pop_txt else ""
        result["ngày khô/ít mưa nhất"] = f"{_day_label(driest[0])}: {rain_txt}{tail}"

    return result


def _narrative_daily(forecasts: Sequence[Mapping[str, Any]], location_name: str) -> str:
    if not forecasts:
        return ""
    temps_min = [f.get("temp_min") for f in forecasts if f.get("temp_min") is not None]
    temps_max = [f.get("temp_max") for f in forecasts if f.get("temp_max") is not None]
    rain_days = sum(1 for f in forecasts
                    if (f.get("rain_total") or f.get("total_rain") or 0) >= 1)
    bits = [f"{len(forecasts)} ngày tại {location_name}"]
    if temps_min and temps_max:
        bits.append(f"nhiệt {min(temps_min):.0f}–{max(temps_max):.0f}°C")
    if rain_days:
        bits.append(f"{rain_days} ngày có mưa")
    else:
        bits.append("không mưa đáng kể")
    return ", ".join(bits) + "."


def build_daily_forecast_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    level = raw.get("level") or "ward"
    location_name = _format_location(raw.get("resolved_location") or {}, level)
    forecasts = raw.get("forecasts") or []
    entries = [_build_daily_entry(f) for f in forecasts]
    result: Dict[str, Any] = {
        "địa điểm": location_name,
        "loại dự báo": raw.get("data_coverage") or f"{len(entries)} ngày tới",
        "dự báo": entries,
    }
    # Pre-computed superlatives để LLM không phải argmax qua list
    sup = _compute_daily_superlatives(forecasts)
    if sup:
        result["tổng hợp"] = sup
    result["tóm tắt tổng"] = _narrative_daily(forecasts, location_name)
    if raw.get("data_note"):
        result["ghi chú dữ liệu"] = raw["data_note"]
    return result


def build_rain_timeline_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    level = raw.get("level") or "ward"
    location_name = _format_location(raw.get("resolved_location") or {}, level)
    periods = raw.get("rain_periods") or []
    period_entries: List[Dict[str, Any]] = []
    for p in periods:
        max_pop = p.get("max_pop")
        max_rain = p.get("max_rain_1h") or 0
        rain_label = label_rain_intensity(max_rain) if max_rain >= 0.05 else ""
        entry = {
            "bắt đầu": p.get("start", ""),
            "kết thúc": p.get("end", ""),
            "xác suất cao nhất": f"{int(max_pop)}%" if max_pop is not None else "—",
        }
        if rain_label:
            entry["cường độ đỉnh"] = rain_label
        period_entries.append(entry)

    summary_bits: List[str] = []
    if raw.get("next_rain"):
        summary_bits.append(f"mưa bắt đầu lúc {raw['next_rain']}")
    if raw.get("next_clear"):
        summary_bits.append(f"tạnh lúc {raw['next_clear']}")
    if not periods and not summary_bits:
        summary_bits.append(f"không có đợt mưa nào trong {raw.get('hours_scanned', '?')} giờ tới")

    summary_text = "; ".join(summary_bits)
    if summary_text:
        summary_text = summary_text[:1].upper() + summary_text[1:] + "."
    else:
        summary_text = "Không có đợt mưa dự báo."
    result: Dict[str, Any] = {
        "địa điểm": location_name,
        "phạm vi": raw.get("data_coverage") or f"{raw.get('hours_scanned', 0)} giờ tới",
        "đợt mưa": period_entries,
        "tổng số đợt": raw.get("total_rain_periods") or len(periods),
        "tóm tắt": summary_text,
    }
    # Past-frame warning: nếu đợt mưa đầu tiên bắt đầu >1.5h sau NOW,
    # các khung sáng/trưa/chiều/tối đã qua trong hôm nay có thể không cover.
    # Tận dụng raw forecasts nếu dispatch layer forward qua.
    if isinstance(raw.get("forecasts"), list):
        result.update(_detect_forecast_range_gap(raw["forecasts"]))
    return result


def build_best_time_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    level = raw.get("level") or "ward"
    location_name = _format_location(raw.get("resolved_location") or {}, level)

    def _fmt_slot(s: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "thời điểm": s.get("time_ict", ""),
            "điểm": f"{s.get('score', 0)}/100",
            "nhiệt độ": label_temp_hn(s.get("temp")),
            "xác suất mưa": label_rain_probability((s.get("pop") or 0) / 100) if s.get("pop") is not None else "—",
            "ghi chú": ", ".join(s.get("issues") or []) or "Tốt",
        }

    return {
        "địa điểm": location_name,
        "hoạt động": raw.get("activity", ""),
        "giờ tốt nhất": [_fmt_slot(s) for s in (raw.get("best_hours") or [])],
        "giờ kém nhất": [_fmt_slot(s) for s in (raw.get("worst_hours") or [])],
        "tổng số giờ quét": raw.get("total_hours_scanned", 0),
    }


def build_weather_history_output(raw: Mapping[str, Any], date_hint: Optional[str] = None) -> Dict[str, Any]:
    """Timemachine/daily history — ward ONLY có wind_gust (không wind_speed)."""
    if _is_error(raw):
        return build_error_output(raw)
    level = raw.get("level") or "ward"
    location_name = _format_location(raw.get("resolved_location") or {}, level)

    temp = _pick_temp(raw, "temp", "avg_temp")
    feels_like = raw.get("feels_like")
    humidity = raw.get("humidity") or raw.get("avg_humidity")
    dew_point = raw.get("dew_point") or raw.get("avg_dew_point")
    wind_gust = raw.get("wind_gust") or raw.get("max_wind_gust")
    wind_speed = raw.get("wind_speed") or raw.get("avg_wind_speed")
    wind_deg = raw.get("wind_deg") or raw.get("avg_wind_deg")
    weather_main = raw.get("weather_main")
    daily = raw.get("daily_summary") or {}

    result: Dict[str, Any] = {
        "địa điểm": location_name,
        "ngày": _format_date_vi(date_hint or raw.get("date") or daily.get("date")),
        "thời tiết chung": weather_main_to_vietnamese(weather_main or "") or "—",
    }
    if raw.get("note"):
        result["phạm vi dữ liệu"] = raw["note"]

    if temp is not None:
        result["nhiệt độ"] = label_temp_hn(temp)
    if level == "ward" and feels_like is not None:
        result["cảm giác"] = label_temp_hn(feels_like)
    if daily.get("temp_min") is not None and daily.get("temp_max") is not None:
        result["nhiệt độ min-max"] = f"Thấp {daily['temp_min']:.1f}°C — Cao {daily['temp_max']:.1f}°C"
    elif raw.get("temp_min") is not None and raw.get("temp_max") is not None:
        result["nhiệt độ min-max"] = f"Thấp {raw['temp_min']:.1f}°C — Cao {raw['temp_max']:.1f}°C"
    if humidity is not None:
        result["độ ẩm"] = f"{int(round(humidity))}%"
    if dew_point is not None:
        result["điểm sương"] = f"{get_dew_point_status(dew_point)} {dew_point:.1f}°C"

    rain_total = daily.get("rain_total") or raw.get("rain_total") or raw.get("total_rain")
    if rain_total is not None and rain_total >= 0.05:
        result["tổng lượng mưa"] = label_rain_total(rain_total)

    # Wind: history ward chỉ có wind_gust, KHÔNG bịa avg
    if wind_speed is None and wind_gust is not None:
        result["gió"] = _wind_gust_only(wind_gust, wind_deg)
    else:
        result["gió"] = _wind_text(wind_speed, wind_gust, wind_deg)

    uvi = daily.get("uvi") or raw.get("uvi") or raw.get("max_uvi")
    if uvi is not None:
        result["UV"] = f"{get_uv_status(uvi)} {uvi:.1f}"
    return result


def build_daily_summary_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    level = raw.get("level") or "ward"
    location_name = _format_location(raw.get("resolved_location") or {}, level)

    result: Dict[str, Any] = {
        "địa điểm": location_name,
        "ngày": _format_date_vi(raw.get("date")),
        "thời tiết chung": weather_main_to_vietnamese(raw.get("weather_main") or "") or "—",
    }

    # Pre-compute chênh nhiệt ngày-đêm (arithmetic self-check, fix audit B4 ID 64)
    temp_min_val: Optional[float] = None
    temp_max_val: Optional[float] = None

    tr = raw.get("temp_range")
    if isinstance(tr, Mapping):
        temp_min_val = tr.get("min")
        temp_max_val = tr.get("max")
        result["nhiệt độ"] = f"Thấp {tr.get('min', 0):.1f}°C — Cao {tr.get('max', 0):.1f}°C (biên độ {tr.get('bien_do', 0):.1f}°C)"
    elif raw.get("temp_min") is not None and raw.get("temp_max") is not None:
        temp_min_val = raw["temp_min"]
        temp_max_val = raw["temp_max"]
        result["nhiệt độ"] = f"Thấp {raw['temp_min']:.1f}°C — Cao {raw['temp_max']:.1f}°C"
    elif raw.get("avg_temp") is not None:
        result["nhiệt độ"] = label_temp_hn(raw["avg_temp"])

    if temp_min_val is not None and temp_max_val is not None:
        diff = float(temp_max_val) - float(temp_min_val)
        result["chênh nhiệt ngày-đêm"] = f"{diff:.1f}°C (COPY thẳng, KHÔNG tự tính)"

    prog = raw.get("temp_progression")
    if isinstance(prog, Mapping):
        parts = []
        for k, vi in (("sang", "Sáng"), ("trua", "Trưa"), ("chieu", "Chiều"), ("toi", "Tối")):
            v = prog.get(k)
            if v is not None:
                parts.append(f"{vi} {v:.1f}°C")
        if parts:
            result["nhiệt độ theo ngày"] = " / ".join(parts)

    humidity = raw.get("humidity") or raw.get("avg_humidity")
    if humidity is not None:
        result["độ ẩm"] = f"{int(round(humidity))}%"

    pop = raw.get("pop") or raw.get("avg_pop")
    if pop is not None:
        result["xác suất mưa"] = label_rain_probability(pop)

    rain_total = raw.get("rain_total") or raw.get("total_rain")
    if rain_total is not None and rain_total >= 0.05:
        result["tổng lượng mưa"] = label_rain_total(rain_total)

    wind = raw.get("wind")
    if isinstance(wind, Mapping):
        result["gió"] = _wind_text(wind.get("speed"), wind.get("gust"),
                                   None if wind.get("direction") else None)
        if wind.get("direction"):
            # direction may already be VN string
            result["gió"] = result["gió"] + f", hướng {wind['direction']}"
    else:
        ws = raw.get("avg_wind_speed")
        wg = raw.get("max_wind_gust") or raw.get("wind_gust")
        wd = raw.get("avg_wind_deg") or raw.get("wind_deg")
        result["gió"] = _wind_text(ws, wg, wd)

    uvi = raw.get("uvi") or raw.get("max_uvi")
    if uvi is not None:
        result["UV"] = f"{get_uv_status(uvi)} {uvi:.1f}"
    if raw.get("daylight_hours") is not None:
        result["thời gian nắng"] = f"{raw['daylight_hours']:.1f} giờ"

    sr = _format_time_only(raw.get("sunrise")) if raw.get("sunrise") else ""
    ss = _format_time_only(raw.get("sunset")) if raw.get("sunset") else ""
    if sr and ss:
        result["mọc-lặn"] = f"Mặt trời {sr} → {ss}"

    if raw.get("note"):
        result["ghi chú"] = raw["note"]
    result["gợi ý dùng output"] = (
        "Đây là TỔNG HỢP CẢ NGÀY (min/max/TB + sáng/trưa/chiều/tối), KHÔNG phải thời điểm tức thời. "
        "Nếu user hỏi 'bây giờ / hiện tại / đang' → gọi get_current_weather. "
        "Khi trả lời câu hỏi theo khung (chiều/tối/sáng) → LẤY ĐÚNG giá trị từ 'nhiệt độ theo ngày', "
        "không gán cả dải 'Thấp X — Cao Y' làm giá trị tức thời. "
        "Key `chênh nhiệt ngày-đêm` đã pre-compute — COPY thẳng, KHÔNG tự trừ max-min."
    )
    return result


def build_weather_period_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    level = raw.get("level") or "ward"
    location_name = _format_location(raw.get("resolved_location") or {}, level)
    daily = raw.get("daily_data") or []
    stats = raw.get("statistics") or {}
    days = raw.get("days") or len(daily)

    result: Dict[str, Any] = {
        "địa điểm": location_name,
        "phạm vi": f"{days} ngày",
    }
    result["ngày"] = [_build_daily_entry(d) for d in daily]
    # Pre-computed superlatives trên list ngày
    sup = _compute_daily_superlatives(daily)
    if sup:
        result["tổng hợp"] = sup
    if stats:
        result["thống kê tổng"] = {
            "nhiệt độ TB": label_temp_hn(stats.get("avg_temp")) if stats.get("avg_temp") is not None else "—",
            "nhiệt độ thấp nhất": f"{stats.get('min_temp'):.1f}°C" if stats.get("min_temp") is not None else "—",
            "nhiệt độ cao nhất": f"{stats.get('max_temp'):.1f}°C" if stats.get("max_temp") is not None else "—",
            "tổng lượng mưa": label_rain_total(stats.get("total_rain") or 0),
            "số ngày có mưa": f"{stats.get('rain_days', 0)}/{days}",
        }
    if raw.get("note"):
        result["ghi chú dữ liệu"] = raw["note"]
    return result


def _compare_location_block(loc_info: Mapping[str, Any]) -> Dict[str, Any]:
    name = loc_info.get("name") or "?"
    w = loc_info.get("weather") or {}
    temp = _pick_temp(w, "temp", "avg_temp")
    humidity = w.get("humidity") or w.get("avg_humidity")
    wind_speed = w.get("wind_speed") or w.get("avg_wind_speed")
    wind_gust = w.get("wind_gust") or w.get("max_wind_gust")
    wind_deg = w.get("wind_deg") or w.get("avg_wind_deg")
    out = {
        "tên": name,
        "thời tiết": weather_main_to_vietnamese(w.get("weather_main") or "") or "—",
        "nhiệt độ": label_temp_hn(temp),
    }
    if humidity is not None:
        out["độ ẩm"] = f"{int(round(humidity))}%"
    out["gió"] = _wind_text(wind_speed, wind_gust, wind_deg)
    return out


def build_compare_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    loc1 = raw.get("location1") or {}
    loc2 = raw.get("location2") or {}
    diffs = raw.get("differences") or {}
    diff_bits = []
    if diffs.get("temp_diff") is not None:
        diff_bits.append(f"chênh nhiệt độ {diffs['temp_diff']:+.1f}°C")
    if diffs.get("humidity_diff") is not None:
        diff_bits.append(f"chênh độ ẩm {diffs['humidity_diff']:+.0f}%")
    return {
        "loại so sánh": "Hai địa điểm, thời điểm hiện tại",
        "địa điểm 1": _compare_location_block(loc1),
        "địa điểm 2": _compare_location_block(loc2),
        "chênh lệch": {
            "nhiệt độ": f"{diffs.get('temp_diff', 0):+.1f}°C",
            "độ ẩm": f"{diffs.get('humidity_diff', 0):+.0f}%" if diffs.get("humidity_diff") is not None else "—",
        },
        "tóm tắt": raw.get("comparison_text") or ", ".join(diff_bits) or "Chênh lệch không đáng kể.",
    }


def build_compare_with_yesterday_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    today = raw.get("today") or {}
    prev = raw.get("previous") or {}
    level = raw.get("level") or "ward"
    location_name = raw.get("location_name") or _format_location(raw.get("resolved_location") or {}, level)

    def _day_block(d: Mapping[str, Any], date_v: Any) -> Dict[str, Any]:
        temp_avg = d.get("temp_avg") or d.get("avg_temp")
        rain = d.get("rain_total") or d.get("total_rain") or 0
        humidity = d.get("humidity") or d.get("avg_humidity")
        out = {
            "ngày": _format_date_vi(date_v or d.get("date")),
            "nhiệt độ TB": label_temp_hn(temp_avg),
            "thời tiết": weather_main_to_vietnamese(d.get("weather_main") or "") or "—",
            "lượng mưa": label_rain_total(rain) if rain >= 0.05 else "Không mưa",
        }
        if humidity is not None:
            out["độ ẩm"] = f"{int(round(humidity))}%"
        return out

    today_block = _day_block(today, today.get("date"))
    prev_block = _day_block(prev, prev.get("date"))

    changes = list(raw.get("changes") or [])
    if not changes:
        temp_diff = raw.get("temp_diff")
        if temp_diff is not None:
            changes.append(f"Nhiệt độ thay đổi {temp_diff:+.1f}°C")
        rain_diff = raw.get("rain_diff")
        if rain_diff is not None and abs(rain_diff) >= 1:
            changes.append(f"Lượng mưa thay đổi {rain_diff:+.1f} mm")

    return {
        "loại so sánh": "Hôm nay vs Hôm qua",
        "địa điểm": location_name,
        "hôm nay": today_block,
        "hôm qua": prev_block,
        "thay đổi": changes or ["Thay đổi không đáng kể"],
        "tóm tắt": (changes[0] if changes else "Thời tiết hôm nay tương tự hôm qua."),
    }


def build_seasonal_comparison_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    cur = raw.get("current") or {}
    seasonal = raw.get("seasonal_avg") or {}
    # DAL compare_with_seasonal trả comparisons là list of VN strings, không phải dict
    comparisons = raw.get("comparisons") or []
    month_name = raw.get("month_name") or ""

    cur_temp = _pick_temp(cur, "temp", "avg_temp")
    seasonal_temp = seasonal.get("temp_avg")  # DAL dùng key "temp_avg"

    result: Dict[str, Any] = {
        "loại so sánh": f"Hiện tại vs trung bình {month_name}",
        "hiện tại": {
            "nhiệt độ": label_temp_hn(cur_temp),
            "độ ẩm": f"{int(round(cur.get('humidity') or cur.get('avg_humidity') or 0))}%",
        },
        "trung bình mùa": {
            "nhiệt độ TB": label_temp_hn(seasonal_temp),
        },
    }
    if seasonal.get("humidity") is not None:
        result["trung bình mùa"]["độ ẩm"] = f"{int(round(seasonal['humidity']))}%"
    if seasonal.get("temp_min") is not None and seasonal.get("temp_max") is not None:
        result["trung bình mùa"]["dải nhiệt TB"] = f"{seasonal['temp_min']}–{seasonal['temp_max']}°C"
    if seasonal.get("rain_days") is not None:
        result["trung bình mùa"]["số ngày mưa TB"] = f"{seasonal['rain_days']} ngày/tháng"

    # comparisons là list VN strings từ DAL — giữ nguyên, chỉ rename key
    if comparisons:
        result["nhận xét"] = [str(c) for c in comparisons]
    return result


# ── Group 2: generic shape_labeled_dict ─────────────────────────────────────

def shape_labeled_dict(raw: Mapping[str, Any],
                       key_map: Mapping[str, str],
                       extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Rename raw keys → VN keys. Keep values as-is (assume already labeled or simple).

    Args:
        raw: input dict
        key_map: {raw_key: vn_key} — keys not in map are dropped
        extra: optional dict merged into result

    Error handling: if raw has 'error', return build_error_output.
    """
    if _is_error(raw):
        return build_error_output(raw)
    result: Dict[str, Any] = {}
    for raw_key, vn_key in key_map.items():
        if raw_key in raw:
            v = raw[raw_key]
            if v is not None:
                result[vn_key] = v
    if extra:
        result.update(extra)
    return result


# ── Group 4: generic ranking output ─────────────────────────────────────────

_METRIC_VN_LABEL = {
    "nhiet_do": "nhiệt độ", "do_am": "độ ẩm", "gio": "gió",
    "mua": "lượng mưa", "uvi": "UV", "ap_suat": "áp suất",
    "diem_suong": "điểm sương", "may": "mây",
}

_UNIT_DISPLAY = {"C": "°C", "%": "%", "m/s": " m/s", "hPa": " hPa", "": ""}


def shape_ranking_output(raw: Mapping[str, Any],
                         location_vn_key: str) -> Dict[str, Any]:
    """Rankings list → flat VN per-entry.

    DAL trả: {metric, unit, order, rankings: [{rank, district|ward, value, unit}]}
    """
    if _is_error(raw):
        return build_error_output(raw)
    rankings = raw.get("rankings") or []
    metric = raw.get("metric") or ""
    unit = raw.get("unit") or ""
    metric_vn = _METRIC_VN_LABEL.get(metric, metric)
    order_vn = "cao nhất" if raw.get("order") == "cao_nhat" else "thấp nhất"

    entries: List[Dict[str, Any]] = []
    for r in rankings:
        # DAL dùng keys "district" / "ward" — không phải district_name_vi
        name = (r.get("district") or r.get("ward")
                or r.get("district_name_vi") or r.get("ward_name_vi")
                or r.get("name") or "")
        if not name:
            continue  # Skip empty (bug defensive)
        value = r.get("value")
        entry: Dict[str, Any] = {
            "hạng": r.get("rank") or len(entries) + 1,
            location_vn_key: name,
        }
        if value is not None:
            unit_disp = _UNIT_DISPLAY.get(unit, f" {unit}" if unit else "")
            if isinstance(value, float):
                entry[metric_vn] = f"{value:.1f}{unit_disp}"
            else:
                entry[metric_vn] = f"{value}{unit_disp}"
        entries.append(entry)

    return {
        "chỉ số": metric_vn,
        "thứ tự": order_vn,
        "xếp hạng": entries,
        "tổng số": len(entries),
    }


# ── Group 3: window/list builders (insight_advanced) ────────────────────────

def _fmt_window(w: Mapping[str, Any]) -> Dict[str, Any]:
    """Common window format."""
    out: Dict[str, Any] = {}
    if w.get("start"):
        out["bắt đầu"] = w["start"]
    if w.get("end"):
        out["kết thúc"] = w["end"]
    if w.get("time") or w.get("time_ict"):
        out["thời điểm"] = w.get("time") or w["time_ict"]
    if w.get("duration_hours") is not None:
        out["thời lượng"] = f"{w['duration_hours']} giờ"
    return out


def build_uv_safe_windows_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    level = raw.get("level") or "ward"
    location_name = _format_location(raw.get("resolved_location") or {}, level)

    def _uv_win(w: Mapping[str, Any]) -> Dict[str, Any]:
        base = _fmt_window(w)
        uvi = w.get("uvi") or w.get("max_uvi")
        if uvi is not None:
            base["UV"] = f"{get_uv_status(uvi)} {uvi:.1f}"
        return base

    return {
        "địa điểm": location_name,
        "UV đỉnh": f"{raw.get('peak_uvi', 0):.1f}" if raw.get("peak_uvi") is not None else "—",
        "giờ đỉnh": raw.get("peak_time", ""),
        "khung UV an toàn": [_uv_win(w) for w in (raw.get("safe_windows") or [])],
        "khung UV nguy hiểm": [_uv_win(w) for w in (raw.get("danger_windows") or [])],
        "tóm tắt": raw.get("summary") or "",
    }


def build_pressure_trend_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    level = raw.get("level") or "ward"
    location_name = _format_location(raw.get("resolved_location") or {}, level)
    return {
        "địa điểm": location_name,
        "xu hướng áp suất": raw.get("trend", ""),
        "thay đổi tổng": f"{raw.get('total_change', 0):+.1f} hPa" if raw.get("total_change") is not None else "—",
        "drop 3h lớn nhất": f"{raw.get('max_3h_drop', 0):.1f} hPa" if raw.get("max_3h_drop") is not None else "—",
        "cảnh báo front": raw.get("front_warning") or "Không có",
        "tóm tắt": raw.get("summary") or "",
    }


def build_daily_rhythm_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    level = raw.get("level") or "ward"
    location_name = _format_location(raw.get("resolved_location") or {}, level)
    rhythm = raw.get("rhythm") or {}

    def _bucket(b: Mapping[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        avg = b.get("avg_temp") or b.get("temp")
        if avg is not None:
            out["nhiệt độ TB"] = label_temp_hn(avg)
        if b.get("min_temp") is not None and b.get("max_temp") is not None:
            out["dải nhiệt"] = f"{b['min_temp']:.1f} – {b['max_temp']:.1f}°C"
        if b.get("avg_humidity") is not None:
            out["độ ẩm TB"] = f"{int(round(b['avg_humidity']))}%"
        return out

    result: Dict[str, Any] = {"địa điểm": location_name}
    for k, vi in (("morning", "sáng"), ("noon", "trưa"), ("afternoon", "chiều"), ("evening", "tối")):
        if isinstance(rhythm.get(k), Mapping):
            result[vi] = _bucket(rhythm[k])
    if raw.get("coolest_period"):
        result["khung mát nhất"] = raw["coolest_period"]
    if raw.get("hottest_period"):
        result["khung nóng nhất"] = raw["hottest_period"]
    return result


def build_humidity_timeline_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    level = raw.get("level") or "ward"
    location_name = _format_location(raw.get("resolved_location") or {}, level)

    def _hent(e: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "thời điểm": _format_hour_short(e.get("time_ict") or e.get("ts_utc")),
            "độ ẩm": f"{int(round(e.get('humidity') or 0))}%" if e.get("humidity") is not None else "—",
            "điểm sương": f"{e.get('dew_point'):.1f}°C" if e.get("dew_point") is not None else "—",
        }

    nom = raw.get("nom_am_periods") or []
    stats = raw.get("statistics") or {}
    return {
        "địa điểm": location_name,
        "timeline độ ẩm": [_hent(e) for e in (raw.get("timeline") or [])[:24]],
        "thống kê": {
            "độ ẩm TB": f"{int(round(stats.get('avg_humidity') or 0))}%" if stats.get("avg_humidity") is not None else "—",
            "độ ẩm cao nhất": f"{int(round(stats.get('max_humidity') or 0))}%" if stats.get("max_humidity") is not None else "—",
            "độ ẩm thấp nhất": f"{int(round(stats.get('min_humidity') or 0))}%" if stats.get("min_humidity") is not None else "—",
        },
        "khung nồm ẩm": [_fmt_window(w) for w in nom] if nom else ["Không có khung nồm ẩm"],
    }


def build_sunny_periods_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    level = raw.get("level") or "ward"
    location_name = _format_location(raw.get("resolved_location") or {}, level)

    def _sun_win(w: Mapping[str, Any]) -> Dict[str, Any]:
        base = _fmt_window(w)
        if w.get("avg_clouds") is not None:
            base["mây"] = label_clouds(w["avg_clouds"])
        return base

    return {
        "địa điểm": location_name,
        "khung nắng": [_sun_win(w) for w in (raw.get("sunny_windows") or [])],
        "khung nhiều mây": [_sun_win(w) for w in (raw.get("cloudy_windows") or [])],
        "khung nắng đẹp nhất": raw.get("best_sunny_time") or "—",
        "tóm tắt": raw.get("summary") or "",
    }


# ── Key maps for Group 2/4 (used by shape_labeled_dict / shape_ranking_output)

RESOLVE_LOCATION_KEYS = {
    "status": "trạng thái",
    "ward_id": "ward_id",
    "ward_name_vi": "phường/xã",
    "district_name_vi": "quận/huyện",
    "message": "ghi chú",
    "suggestions": "gợi ý",
}

WEATHER_ALERTS_KEYS = {
    "alerts": "cảnh báo",
    "count": "số lượng",
    "resolved_location": "địa điểm",
}

PHENOMENA_KEYS = {
    "phenomena": "hiện tượng",
    "has_dangerous": "có nguy hiểm",
    "weather_snapshot": "thời tiết hiện tại",
}

TEMP_TREND_KEYS = {
    "trend": "xu hướng",
    "slope_per_day": "thay đổi TB/ngày (°C)",
    "hottest_day": "ngày nóng nhất",
    "coldest_day": "ngày lạnh nhất",
    "daily_summary": "chi tiết theo ngày",
}

COMFORT_INDEX_KEYS = {
    "score": "điểm",
    "label": "mức độ",
    "recommendation": "khuyến nghị",
    "breakdown": "phân tích",
}

CHANGE_ALERT_KEYS = {
    "changes": "thay đổi",
    "has_significant_change": "có thay đổi đáng kể",
    "current_summary": "hiện trạng",
}

CLOTHING_ADVICE_KEYS = {
    "clothing_items": "trang phục đề xuất",
    "notes": "ghi chú",
    "temp": "nhiệt độ",
    "humidity": "độ ẩm",
    "pop": "xác suất mưa",
    "uvi": "UV",
    "wind_speed": "gió",
}

ACTIVITY_ADVICE_KEYS = {
    "advice": "khuyến nghị",
    "reason": "lý do",
    "recommendations": "gợi ý thêm",
}

DISTRICT_RANKING_METRIC_LABELS = {
    "avg_temp": "nhiệt độ TB",
    "temp": "nhiệt độ",
    "avg_humidity": "độ ẩm TB",
    "humidity": "độ ẩm",
    "total_rain": "tổng mưa (mm)",
    "rain_total": "tổng mưa (mm)",
    "avg_wind_speed": "gió TB (m/s)",
    "max_uvi": "UV đỉnh",
    "avg_pop": "xác suất mưa TB",
}


def build_resolve_location_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return shape_labeled_dict(raw, RESOLVE_LOCATION_KEYS)


def build_weather_alerts_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    alerts = raw.get("alerts") or []
    fmt = []
    for a in alerts:
        if isinstance(a, Mapping):
            fmt.append({
                "loại": a.get("type") or a.get("severity") or "cảnh báo",
                "mô tả": a.get("description") or a.get("message") or "",
                "thời điểm": a.get("time_ict") or a.get("time") or "",
            })
        else:
            fmt.append({"mô tả": str(a)})
    return {
        "cảnh báo": fmt,
        "số lượng": len(fmt),
    }


def build_detect_phenomena_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return shape_labeled_dict(raw, PHENOMENA_KEYS)


def build_temperature_trend_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return shape_labeled_dict(raw, TEMP_TREND_KEYS)


_ADVICE_NO_HALLUCINATE = (
    "⚠ KHÔNG suy diễn ngoài output: CHỈ dùng các field có trong output này. "
    "TUYỆT ĐỐI KHÔNG thêm nhãn hiện tượng (mưa phùn, sương mù, đợt lạnh mạnh, "
    "nắng đẹp, nồm ẩm...) không có trong recommendations/ghi chú. "
    "Nếu user hỏi chi tiết mưa/UV/giờ cụ thể → gọi tool tương ứng "
    "(get_rain_timeline / get_uv_safe_windows / get_hourly_forecast) để lấy data."
)


def build_comfort_index_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    score = raw.get("score")
    result = {
        "điểm thoải mái": f"{score}/100" if score is not None else "—",
        "mức độ": raw.get("label") or "",
        "khuyến nghị": raw.get("recommendation") or "",
    }
    if isinstance(raw.get("breakdown"), Mapping):
        result["phân tích"] = raw["breakdown"]
    result["⚠ KHÔNG suy diễn"] = _ADVICE_NO_HALLUCINATE
    return result


def build_weather_change_alert_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return shape_labeled_dict(raw, CHANGE_ALERT_KEYS)


def build_clothing_advice_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    return {
        "trang phục đề xuất": raw.get("clothing_items") or [],
        "ghi chú": raw.get("notes") or [],
        "⚠ KHÔNG suy diễn": _ADVICE_NO_HALLUCINATE,
    }


def build_activity_advice_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    return {
        "khuyến nghị": raw.get("advice") or "",
        "lý do": raw.get("reason") or "",
        "gợi ý thêm": raw.get("recommendations") or [],
        "⚠ KHÔNG suy diễn": _ADVICE_NO_HALLUCINATE,
    }


def build_district_ranking_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return shape_ranking_output(raw, "quận")


def build_ward_ranking_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return shape_ranking_output(raw, "phường/xã")


def build_district_multi_compare_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_error(raw):
        return build_error_output(raw)
    comparisons = raw.get("comparisons") or []
    # comparisons có thể là dict (do merge by district) hoặc list raw
    if isinstance(comparisons, list):
        fmt: List[Dict[str, Any]] = []
        for c in comparisons:
            if not isinstance(c, Mapping):
                continue
            name = (c.get("district") or c.get("district_name_vi")
                    or c.get("name") or "")
            if not name:
                continue
            entry: Dict[str, Any] = {"quận": name}
            # Copy các field giá trị nếu có
            for k, vn in _METRIC_VN_LABEL.items():
                v = c.get(k)
                if v is not None:
                    entry[vn] = f"{v:.1f}" if isinstance(v, float) else str(v)
            fmt.append(entry)
        return {
            "so sánh": fmt,
            "các chỉ số": raw.get("metrics_analyzed") or [],
            "tổng số quận": len(fmt),
        }
    return {
        "so sánh": comparisons,
        "các chỉ số": raw.get("metrics_analyzed") or [],
    }
