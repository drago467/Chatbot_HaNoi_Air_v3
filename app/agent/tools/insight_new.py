"""New Insight tools — 6 tool mới tận dụng data sẵn có trong DB.

1. get_uv_safe_windows — Tìm khung giờ UV an toàn để ra ngoài
2. get_pressure_trend — Xu hướng áp suất (front thời tiết)
3. get_daily_rhythm — Nhịp nhiệt độ trong ngày (sáng/trưa/chiều/tối)
4. get_humidity_timeline — Timeline độ ẩm + điểm sương
5. get_sunny_periods — Tìm khung giờ nắng đẹp (mây ít, UV vừa)
6. get_district_multi_compare — So sánh nhiều chỉ số cùng lúc giữa các quận
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============== Tool 1: get_uv_safe_windows ==============

class GetUvSafeWindowsInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")
    hours: int = Field(default=24, description="Số giờ quét (1-48)")
    max_uvi: float = Field(default=5.0, description="Ngưỡng UV an toàn (mặc định 5 = trung bình)")


@tool(args_schema=GetUvSafeWindowsInput)
def get_uv_safe_windows(ward_id: str = None, location_hint: str = None,
                        hours: int = 24, max_uvi: float = 5.0) -> dict:
    """Tìm KHUNG GIỜ UV AN TOÀN để hoạt động ngoài trời.

    DÙNG KHI: "lúc nào ra ngoài an toàn?", "UV thấp lúc mấy giờ?",
    "giờ nào nên đi bộ?", "bao giờ hết nắng gắt?".
    Hỗ trợ: phường/xã, quận/huyện, toàn Hà Nội.
    Trả về: safe_windows (khung giờ UV < ngưỡng), peak_uv_time, summary.
    """
    from app.agent.dispatch import dispatch_forecast, normalize_rows
    from app.dal.weather_dal import get_hourly_forecast as dal_ward
    from app.dal.weather_aggregate_dal import (
        get_district_hourly_forecast as dal_district,
        get_city_hourly_forecast as dal_city,
    )

    hours = max(1, min(hours, 48))
    result = dispatch_forecast(
        ward_id=ward_id,
        location_hint=location_hint,
        ward_fn=dal_ward,
        district_fn=dal_district,
        city_fn=dal_city,
        ward_args={"hours": hours},
        district_args={"hours": hours},
        city_args={"hours": hours},
        forecast_type="hourly",
        default_scope="city",
    )

    if result.get("error"):
        return result

    forecasts = result.get("forecasts", [])

    # Analyze UV windows
    safe_windows = []
    danger_windows = []
    current_window = None
    peak_uv = {"uvi": 0, "time": None}

    for f in forecasts:
        uvi = f.get("uvi") or f.get("uvi_max") or 0
        time_str = str(f.get("time_ict") or f.get("ts_utc", ""))

        if uvi > peak_uv["uvi"]:
            peak_uv = {"uvi": uvi, "time": time_str}

        is_safe = uvi <= max_uvi

        if is_safe:
            if current_window is None or current_window["type"] != "safe":
                if current_window:
                    (safe_windows if current_window["type"] == "safe" else danger_windows).append(current_window)
                current_window = {"type": "safe", "start": time_str, "end": time_str,
                                  "min_uvi": uvi, "max_uvi": uvi}
            else:
                current_window["end"] = time_str
                current_window["min_uvi"] = min(current_window["min_uvi"], uvi)
                current_window["max_uvi"] = max(current_window["max_uvi"], uvi)
        else:
            if current_window is None or current_window["type"] != "danger":
                if current_window:
                    (safe_windows if current_window["type"] == "safe" else danger_windows).append(current_window)
                current_window = {"type": "danger", "start": time_str, "end": time_str,
                                  "min_uvi": uvi, "max_uvi": uvi}
            else:
                current_window["end"] = time_str
                current_window["max_uvi"] = max(current_window["max_uvi"], uvi)

    if current_window:
        (safe_windows if current_window["type"] == "safe" else danger_windows).append(current_window)

    # Summary
    if not danger_windows:
        summary = f"UV an toàn suốt {hours} giờ tới (< {max_uvi})"
    elif not safe_windows:
        summary = f"UV cao suốt {hours} giờ tới — hạn chế ra ngoài"
    else:
        best = max(safe_windows, key=lambda w: w["end"])
        summary = f"Giờ an toàn nhất: {best['start']} - {best['end']} (UV {best['max_uvi']})"

    return {
        "safe_windows": safe_windows,
        "danger_windows": danger_windows,
        "peak_uv": peak_uv,
        "threshold": max_uvi,
        "summary": summary,
        "resolved_location": result.get("resolved_location", {}),
        "level": result.get("level", "city"),
    }


# ============== Tool 2: get_pressure_trend ==============

class GetPressureTrendInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")
    hours: int = Field(default=24, description="Số giờ phân tích (1-48)")


@tool(args_schema=GetPressureTrendInput)
def get_pressure_trend(ward_id: str = None, location_hint: str = None, hours: int = 24) -> dict:
    """Phân tích XU HƯỚNG ÁP SUẤT — phát hiện front thời tiết.

    DÙNG KHI: "áp suất thay đổi thế nào?", "có front thời tiết không?",
    "có khí áp thấp không?", "áp suất giảm/tăng mạnh không?".
    Hỗ trợ: phường/xã, quận/huyện, toàn Hà Nội.
    Trả về: trend (rising/falling/stable), drop_rate, front_warning, hourly_data.
    Ý nghĩa: áp suất giảm nhanh (>3 hPa/3h) = front lạnh/bão đến.
    """
    from app.agent.dispatch import dispatch_forecast
    from app.dal.weather_dal import get_hourly_forecast as dal_ward
    from app.dal.weather_aggregate_dal import (
        get_district_hourly_forecast as dal_district,
        get_city_hourly_forecast as dal_city,
    )

    hours = max(1, min(hours, 48))
    result = dispatch_forecast(
        ward_id=ward_id,
        location_hint=location_hint,
        ward_fn=dal_ward,
        district_fn=dal_district,
        city_fn=dal_city,
        ward_args={"hours": hours},
        district_args={"hours": hours},
        city_args={"hours": hours},
        forecast_type="hourly",
        default_scope="city",
    )

    if result.get("error"):
        return result

    forecasts = result.get("forecasts", [])
    pressures = []
    for f in forecasts:
        p = f.get("pressure") or f.get("avg_pressure")
        t = f.get("time_ict") or f.get("ts_utc")
        if p is not None:
            pressures.append({"pressure": p, "time": str(t) if t else ""})

    if len(pressures) < 2:
        return {"error": "no_data", "message": "Không đủ dữ liệu áp suất"}

    # Trend analysis
    p_first = pressures[0]["pressure"]
    p_last = pressures[-1]["pressure"]
    total_change = p_last - p_first

    if total_change > 3:
        trend, trend_vi = "rising", "Tăng"
    elif total_change < -3:
        trend, trend_vi = "falling", "Giảm"
    else:
        trend, trend_vi = "stable", "Ổn định"

    # 3-hour drop detection (front warning)
    max_3h_drop = 0
    front_time = None
    for i in range(min(3, len(pressures)), len(pressures)):
        drop = pressures[i - 3]["pressure"] - pressures[i]["pressure"]
        if drop > max_3h_drop:
            max_3h_drop = drop
            front_time = pressures[i]["time"]

    front_warning = None
    if max_3h_drop >= 6:
        front_warning = f"CẢNH BÁO: Áp suất giảm rất mạnh ({max_3h_drop:.1f} hPa/3h) — có thể có bão"
    elif max_3h_drop >= 3:
        front_warning = f"Lưu ý: Áp suất giảm ({max_3h_drop:.1f} hPa/3h) — có thể có front lạnh"

    return {
        "trend": trend, "trend_vi": trend_vi,
        "total_change": round(total_change, 1),
        "start_pressure": p_first, "end_pressure": p_last,
        "max_3h_drop": round(max_3h_drop, 1),
        "front_warning": front_warning, "front_time": front_time,
        "hourly_data": pressures[:12],  # Limit output size
        "resolved_location": result.get("resolved_location", {}),
        "level": result.get("level", "city"),
    }


# ============== Tool 3: get_daily_rhythm ==============

class GetDailyRhythmInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")
    date: Optional[str] = Field(default=None, description="Ngày (YYYY-MM-DD), mặc định hôm nay")


@tool(args_schema=GetDailyRhythmInput)
def get_daily_rhythm(ward_id: str = None, location_hint: str = None, date: str = None) -> dict:
    """Nhịp nhiệt độ TRONG NGÀY: sáng/trưa/chiều/tối, sunrise/sunset.

    DÙNG KHI: "sáng nay mấy độ?", "chiều nay nóng không?", "tối mát chưa?",
    "nhiệt độ thay đổi trong ngày như thế nào?".
    Hỗ trợ: phường/xã (chi tiết nhất với temp_morn/day/eve/night),
    quận/huyện và toàn Hà Nội (từ hourly forecast).
    Trả về: 4 khung giờ (sáng 6-10, trưa 10-14, chiều 14-18, tối 18-22)
    với temp, humidity, UV, wind trung bình.
    """
    from app.agent.dispatch import dispatch_forecast
    from app.dal.weather_dal import get_hourly_forecast as dal_ward
    from app.dal.weather_aggregate_dal import (
        get_district_hourly_forecast as dal_district,
        get_city_hourly_forecast as dal_city,
    )
    from app.dal.timezone_utils import now_ict

    # Get hourly data (24h)
    result = dispatch_forecast(
        ward_id=ward_id,
        location_hint=location_hint,
        ward_fn=dal_ward,
        district_fn=dal_district,
        city_fn=dal_city,
        ward_args={"hours": 24},
        district_args={"hours": 24},
        city_args={"hours": 24},
        forecast_type="hourly",
        default_scope="city",
    )

    if result.get("error"):
        return result

    forecasts = result.get("forecasts", [])

    # Parse hourly into time-of-day buckets
    buckets = {
        "sang": {"label": "Sáng (6h-10h)", "hours": range(6, 10), "data": []},
        "trua": {"label": "Trưa (10h-14h)", "hours": range(10, 14), "data": []},
        "chieu": {"label": "Chiều (14h-18h)", "hours": range(14, 18), "data": []},
        "toi": {"label": "Tối (18h-22h)", "hours": range(18, 22), "data": []},
    }

    for f in forecasts:
        time_str = f.get("time_ict") or f.get("ts_utc")
        if not time_str:
            continue
        try:
            from datetime import datetime
            if isinstance(time_str, str):
                # Try common formats
                for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
                    try:
                        dt = datetime.strptime(str(time_str), fmt)
                        break
                    except ValueError:
                        continue
                else:
                    continue
            elif isinstance(time_str, datetime):
                dt = time_str
            else:
                continue

            hour = dt.hour
            for bucket_key, bucket in buckets.items():
                if hour in bucket["hours"]:
                    bucket["data"].append(f)
                    break
        except Exception:
            continue

    # Summarize each bucket
    rhythm = {}
    for key, bucket in buckets.items():
        if not bucket["data"]:
            rhythm[key] = {"label": bucket["label"], "data_available": False}
            continue

        data = bucket["data"]
        temps = [d.get("temp") or d.get("avg_temp") for d in data if (d.get("temp") or d.get("avg_temp")) is not None]
        humids = [d.get("humidity") or d.get("avg_humidity") for d in data if (d.get("humidity") or d.get("avg_humidity")) is not None]
        uvis = [d.get("uvi") or d.get("max_uvi") or 0 for d in data]
        pops = [d.get("pop") or d.get("avg_pop") or 0 for d in data]
        winds = [d.get("wind_speed") or d.get("avg_wind_speed") for d in data if (d.get("wind_speed") or d.get("avg_wind_speed")) is not None]

        rhythm[key] = {
            "label": bucket["label"],
            "data_available": True,
            "avg_temp": round(sum(temps) / len(temps), 1) if temps else None,
            "min_temp": round(min(temps), 1) if temps else None,
            "max_temp": round(max(temps), 1) if temps else None,
            "avg_humidity": round(sum(humids) / len(humids)) if humids else None,
            "max_uvi": round(max(uvis), 1) if uvis else None,
            "max_pop": round(max(pops) * 100) if pops else None,
            "avg_wind": round(sum(winds) / len(winds), 1) if winds else None,
        }

    # Find best/worst periods
    available = {k: v for k, v in rhythm.items() if v.get("data_available") and v.get("avg_temp") is not None}
    coolest = min(available.items(), key=lambda x: x[1]["avg_temp"])[0] if available else None
    hottest = max(available.items(), key=lambda x: x[1]["avg_temp"])[0] if available else None

    return {
        "rhythm": rhythm,
        "coolest_period": coolest,
        "hottest_period": hottest,
        "date": date or str(now_ict().date()),
        "resolved_location": result.get("resolved_location", {}),
        "level": result.get("level", "city"),
    }


# ============== Tool 4: get_humidity_timeline ==============

class GetHumidityTimelineInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")
    hours: int = Field(default=24, description="Số giờ (1-48)")


@tool(args_schema=GetHumidityTimelineInput)
def get_humidity_timeline(ward_id: str = None, location_hint: str = None, hours: int = 24) -> dict:
    """Timeline ĐỘ ẨM + ĐIỂM SƯƠNG: khi nào khô ráo, khi nào oi bức.

    DÙNG KHI: "độ ẩm thay đổi thế nào?", "khi nào hết nồm ẩm?",
    "điểm sương bao nhiêu?", "lúc nào thoải mái nhất (độ ẩm)?".
    Hỗ trợ: phường/xã, quận/huyện, toàn Hà Nội.
    Trả về: hourly humidity + dew_point, comfort_zones, nom_am_periods.
    """
    from app.agent.dispatch import dispatch_forecast
    from app.dal.weather_dal import get_hourly_forecast as dal_ward
    from app.dal.weather_aggregate_dal import (
        get_district_hourly_forecast as dal_district,
        get_city_hourly_forecast as dal_city,
    )

    hours = max(1, min(hours, 48))
    result = dispatch_forecast(
        ward_id=ward_id,
        location_hint=location_hint,
        ward_fn=dal_ward,
        district_fn=dal_district,
        city_fn=dal_city,
        ward_args={"hours": hours},
        district_args={"hours": hours},
        city_args={"hours": hours},
        forecast_type="hourly",
        default_scope="city",
    )

    if result.get("error"):
        return result

    forecasts = result.get("forecasts", [])

    # Analyze humidity timeline
    timeline = []
    nom_am_periods = []
    current_nom = None

    for f in forecasts:
        hum = f.get("humidity") or f.get("avg_humidity")
        dp = f.get("dew_point") or f.get("avg_dew_point")
        temp = f.get("temp") or f.get("avg_temp")
        time_str = str(f.get("time_ict") or f.get("ts_utc", ""))

        # Comfort level based on dew point
        if dp is not None:
            if dp > 24:
                comfort = "Rất oi bức"
            elif dp > 20:
                comfort = "Oi bức"
            elif dp > 16:
                comfort = "Dễ chịu"
            elif dp > 10:
                comfort = "Khô ráo"
            else:
                comfort = "Rất khô"
        else:
            comfort = None

        # Nom am detection: humidity >= 85% AND temp-dp <= 2
        is_nom_am = False
        if hum is not None and dp is not None and temp is not None:
            is_nom_am = hum >= 85 and (temp - dp) <= 2

        entry = {"time": time_str, "humidity": hum, "dew_point": dp,
                 "temp": temp, "comfort": comfort, "is_nom_am": is_nom_am}
        timeline.append(entry)

        # Track nom_am periods
        if is_nom_am:
            if current_nom is None:
                current_nom = {"start": time_str, "end": time_str}
            else:
                current_nom["end"] = time_str
        else:
            if current_nom:
                nom_am_periods.append(current_nom)
                current_nom = None

    if current_nom:
        nom_am_periods.append(current_nom)

    # Summary stats
    humidities = [e["humidity"] for e in timeline if e["humidity"] is not None]
    dew_points = [e["dew_point"] for e in timeline if e["dew_point"] is not None]

    # Find most comfortable period
    best_entry = min(
        [e for e in timeline if e["dew_point"] is not None],
        key=lambda x: abs(x["dew_point"] - 18),
        default=None
    )

    return {
        "timeline": timeline[:12],  # Limit for token budget
        "statistics": {
            "avg_humidity": round(sum(humidities) / len(humidities)) if humidities else None,
            "min_humidity": min(humidities) if humidities else None,
            "max_humidity": max(humidities) if humidities else None,
            "avg_dew_point": round(sum(dew_points) / len(dew_points), 1) if dew_points else None,
        },
        "nom_am_periods": nom_am_periods,
        "has_nom_am": len(nom_am_periods) > 0,
        "most_comfortable_time": best_entry["time"] if best_entry else None,
        "resolved_location": result.get("resolved_location", {}),
        "level": result.get("level", "city"),
    }


# ============== Tool 5: get_sunny_periods ==============

class GetSunnyPeriodsInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Tên phường/xã hoặc quận/huyện")
    hours: int = Field(default=24, description="Số giờ (1-48)")


@tool(args_schema=GetSunnyPeriodsInput)
def get_sunny_periods(ward_id: str = None, location_hint: str = None, hours: int = 24) -> dict:
    """Tìm khung giờ NẮNG ĐẸP (ít mây, không mưa, UV vừa phải).

    DÙNG KHI: "khi nào có nắng?", "lúc nào trời quang?", "có nắng để phơi đồ không?",
    "khi nào trời đẹp nhất?".
    Hỗ trợ: phường/xã, quận/huyện, toàn Hà Nội.
    Trả về: sunny_windows, cloudy_windows, best_sunny_time.
    """
    from app.agent.dispatch import dispatch_forecast
    from app.dal.weather_dal import get_hourly_forecast as dal_ward
    from app.dal.weather_aggregate_dal import (
        get_district_hourly_forecast as dal_district,
        get_city_hourly_forecast as dal_city,
    )

    hours = max(1, min(hours, 48))
    result = dispatch_forecast(
        ward_id=ward_id,
        location_hint=location_hint,
        ward_fn=dal_ward,
        district_fn=dal_district,
        city_fn=dal_city,
        ward_args={"hours": hours},
        district_args={"hours": hours},
        city_args={"hours": hours},
        forecast_type="hourly",
        default_scope="city",
    )

    if result.get("error"):
        return result

    forecasts = result.get("forecasts", [])

    sunny_windows = []
    cloudy_windows = []
    current_window = None
    best_sunny = None

    for f in forecasts:
        clouds = f.get("clouds") or f.get("avg_clouds") or 50
        pop = f.get("pop") or f.get("avg_pop") or 0
        uvi = f.get("uvi") or f.get("max_uvi") or 0
        weather_main = f.get("weather_main", "")
        time_str = str(f.get("time_ict") or f.get("ts_utc", ""))

        # Sunny = clouds < 40% AND pop < 30% AND no rain
        rain_types = {"Rain", "Drizzle", "Thunderstorm"}
        is_sunny = clouds < 40 and pop < 0.3 and weather_main not in rain_types

        if is_sunny:
            if current_window is None or current_window["type"] != "sunny":
                if current_window:
                    (sunny_windows if current_window["type"] == "sunny" else cloudy_windows).append(current_window)
                current_window = {"type": "sunny", "start": time_str, "end": time_str,
                                  "avg_clouds": clouds, "max_uvi": uvi, "_count": 1,
                                  "_clouds_sum": clouds, "_uvi_max": uvi}
            else:
                current_window["end"] = time_str
                current_window["_count"] += 1
                current_window["_clouds_sum"] += clouds
                current_window["avg_clouds"] = round(current_window["_clouds_sum"] / current_window["_count"])
                current_window["_uvi_max"] = max(current_window["_uvi_max"], uvi)
                current_window["max_uvi"] = current_window["_uvi_max"]
        else:
            if current_window is None or current_window["type"] != "cloudy":
                if current_window:
                    (sunny_windows if current_window["type"] == "sunny" else cloudy_windows).append(current_window)
                current_window = {"type": "cloudy", "start": time_str, "end": time_str,
                                  "clouds": clouds, "pop": pop, "_count": 1}
            else:
                current_window["end"] = time_str
                current_window["_count"] += 1

    if current_window:
        (sunny_windows if current_window["type"] == "sunny" else cloudy_windows).append(current_window)

    # Best sunny: longest with moderate UV (must be before cleanup)
    if sunny_windows:
        best_sunny = max(sunny_windows, key=lambda w: w.get("_count", 1) if "_count" in w else 1)

    # Clean internal keys
    for w in sunny_windows + cloudy_windows:
        for k in list(w.keys()):
            if k.startswith("_"):
                del w[k]
    if not sunny_windows:
        summary = f"Không có khung giờ nắng đẹp trong {hours} giờ tới"
    else:
        summary = f"Có {len(sunny_windows)} khung giờ nắng đẹp"
        if best_sunny:
            summary += f", tốt nhất: {best_sunny['start']} - {best_sunny['end']}"

    return {
        "sunny_windows": sunny_windows,
        "cloudy_windows": cloudy_windows[:5],  # Limit output
        "best_sunny_time": best_sunny,
        "summary": summary,
        "resolved_location": result.get("resolved_location", {}),
        "level": result.get("level", "city"),
    }


# ============== Tool 6: get_district_multi_compare ==============

class GetDistrictMultiCompareInput(BaseModel):
    metrics: str = Field(
        default="nhiet_do,do_am,uvi",
        description="Các chỉ số cần so sánh, cách nhau bởi dấu phẩy. "
                    "Ví dụ: 'nhiet_do,do_am,uvi,gio,mua,ap_suat'"
    )
    limit: int = Field(default=5, description="Số quận/huyện top (1-30)")


@tool(args_schema=GetDistrictMultiCompareInput)
def get_district_multi_compare(metrics: str = "nhiet_do,do_am,uvi", limit: int = 5) -> dict:
    """So sánh NHIỀU CHỈ SỐ cùng lúc giữa các quận/huyện.

    DÙNG KHI: "tổng hợp thời tiết các quận", "so sánh toàn diện các quận",
    "quận nào thoải mái nhất?" (cần nhiều chỉ số).
    Chỉ số hỗ trợ: nhiet_do, do_am, gio, mua, uvi, ap_suat, diem_suong, may.
    Trả về: cho mỗi chỉ số, top N quận nóng/lạnh/ẩm/... nhất.
    """
    from app.dal.weather_aggregate_dal import get_district_rankings

    metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
    valid_metrics = {"nhiet_do", "do_am", "gio", "mua", "uvi", "ap_suat", "diem_suong", "may"}
    metric_list = [m for m in metric_list if m in valid_metrics]

    if not metric_list:
        return {"error": "invalid_metrics",
                "message": "Không có chỉ số hợp lệ. Chọn từ: nhiet_do, do_am, gio, mua, uvi, ap_suat, diem_suong, may"}

    result = {}
    for metric in metric_list:
        top = get_district_rankings(metric, "cao_nhat", limit)
        bottom = get_district_rankings(metric, "thap_nhat", limit)
        result[metric] = {
            "top": top.get("rankings", []),
            "bottom": bottom.get("rankings", []),
            "unit": top.get("unit", ""),
        }

    return {
        "comparisons": result,
        "metrics_analyzed": metric_list,
        "districts_per_metric": limit,
    }
