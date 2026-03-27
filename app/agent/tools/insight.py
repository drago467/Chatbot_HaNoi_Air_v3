"""Insight tools — phenomena, temperature_trend, comfort_index,
weather_change_alert, clothing_advice, activity_advice.

Tat ca deu ho tro 3 tier nhat quan.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============== Tool: detect_phenomena ==============

class DetectPhenomenaInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Ten phuong/xa hoac quan/huyen")


@tool(args_schema=DetectPhenomenaInput)
def detect_phenomena(ward_id: str = None, location_hint: str = None) -> dict:
    """Phat hien cac HIEN TUONG THOI TIET DAC BIET tai Ha Noi.

    DUNG KHI: "co hien tuong gi dac biet khong?", "co nom am khong?",
    "co gio mua dong bac khong?", "co ret dam khong?".
    Ho tro: phuong/xa, quan/huyen, toan Ha Noi.
    Tra ve: danh sach hien tuong (nom am, gio Lao, gio mua DB, ret dam, suong mu, mua dong).
    """
    from app.agent.dispatch import resolve_and_dispatch, normalize_agg_keys
    from app.dal.weather_knowledge_dal import detect_hanoi_weather_phenomena

    def _detect_ward(ward_id):
        from app.dal.weather_dal import get_current_weather
        weather = get_current_weather(ward_id)
        if weather.get("error"):
            return weather
        phenomena = detect_hanoi_weather_phenomena(weather)
        return {"phenomena": phenomena.get("phenomena", []),
                "has_dangerous": phenomena.get("has_dangerous", False),
                "weather_snapshot": weather}

    def _detect_district(district_name):
        from app.dal.weather_aggregate_dal import get_district_current_weather
        weather = get_district_current_weather(district_name)
        if weather.get("error"):
            return weather
        weather = normalize_agg_keys(weather)
        phenomena = detect_hanoi_weather_phenomena(weather)
        return {"phenomena": phenomena.get("phenomena", []),
                "has_dangerous": phenomena.get("has_dangerous", False),
                "weather_snapshot": weather}

    def _detect_city():
        from app.dal.weather_aggregate_dal import get_city_current_weather
        weather = get_city_current_weather()
        if weather.get("error"):
            return weather
        weather = normalize_agg_keys(weather)
        phenomena = detect_hanoi_weather_phenomena(weather)
        return {"phenomena": phenomena.get("phenomena", []),
                "has_dangerous": phenomena.get("has_dangerous", False),
                "weather_snapshot": weather}

    return resolve_and_dispatch(
        ward_id=ward_id,
        location_hint=location_hint,
        default_scope="city",
        ward_fn=_detect_ward,
        district_fn=_detect_district,
        city_fn=_detect_city,
        normalize=False,
        label="hien tuong thoi tiet",
    )


# ============== Tool: get_temperature_trend ==============

class GetTemperatureTrendInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Ten phuong/xa hoac quan/huyen")
    days: int = Field(default=7, description="So ngay phan tich (2-8)")


@tool(args_schema=GetTemperatureTrendInput)
def get_temperature_trend(ward_id: str = None, location_hint: str = None, days: int = 7) -> dict:
    """Phan tich XU HUONG NHIET DO (am dan len / lanh dan / on dinh).

    DUNG KHI: "nhiet do thay doi the nao?", "co lanh dan khong?",
    "xu huong nhiet do tuan nay?".
    Ho tro: phuong/xa, quan/huyen, toan Ha Noi.
    Tra ve: trend (warming/cooling/stable), slope, inflection_date, hottest/coldest day.
    """
    from app.dal.weather_dal import get_temperature_trend as dal_ward_trend

    days = max(2, min(days, 8))

    def _district_trend(district_name):
        from app.db.dal import query
        rows = query("""
            SELECT date, temp_min, temp_max, avg_temp AS temp_avg, weather_main
            FROM fact_weather_district_daily
            WHERE district_name_vi = %s
              AND date >= (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh')::date
            ORDER BY date LIMIT %s
        """, (district_name, days))
        return _analyze_trend(rows)

    def _city_trend():
        from app.db.dal import query
        rows = query("""
            SELECT date, temp_min, temp_max, avg_temp AS temp_avg, weather_main
            FROM fact_weather_city_daily
            WHERE date >= (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh')::date
            ORDER BY date LIMIT %s
        """, (days,))
        return _analyze_trend(rows)

    from app.agent.dispatch import resolve_and_dispatch
    return resolve_and_dispatch(
        ward_id=ward_id,
        location_hint=location_hint,
        default_scope="city",
        ward_fn=lambda ward_id: dal_ward_trend(ward_id, days),
        district_fn=_district_trend,
        city_fn=_city_trend,
        normalize=False,
        label="xu huong nhiet do",
    )


def _analyze_trend(rows: list) -> dict:
    """Reusable trend analysis cho district/city daily data."""
    if len(rows) < 2:
        return {"error": "no_data", "message": "Khong du du lieu de phan tich xu huong"}

    temps = [r["temp_avg"] for r in rows if r.get("temp_avg") is not None]
    if len(temps) < 2:
        return {"error": "no_data", "message": "Khong du du lieu nhiet do"}

    slope = (temps[-1] - temps[0]) / (len(temps) - 1)
    if slope > 0.5:
        trend, trend_vi = "warming", "Am dan len"
    elif slope < -0.5:
        trend, trend_vi = "cooling", "Lanh dan"
    else:
        trend, trend_vi = "stable", "On dinh"

    # Find inflection
    inflection = None
    for i in range(1, len(temps) - 1):
        prev_diff = temps[i] - temps[i - 1]
        next_diff = temps[i + 1] - temps[i]
        if (prev_diff > 0 and next_diff < -0.5) or (prev_diff < 0 and next_diff > 0.5):
            inflection = str(rows[i]["date"])
            break

    max_row = max(rows, key=lambda r: r.get("temp_max") or 0)
    min_row = min(rows, key=lambda r: r.get("temp_min") or 999)

    return {
        "trend": trend, "trend_vi": trend_vi,
        "slope_per_day": round(slope, 1),
        "days_analyzed": len(rows),
        "inflection_date": inflection,
        "hottest_day": {"date": str(max_row["date"]), "temp_max": max_row.get("temp_max")},
        "coldest_day": {"date": str(min_row["date"]), "temp_min": min_row.get("temp_min")},
        "daily_summary": [
            {"date": str(r["date"]), "min": r.get("temp_min"), "max": r.get("temp_max"),
             "avg": r.get("temp_avg"), "weather": r.get("weather_main")}
            for r in rows
        ],
    }


# ============== Tool: get_comfort_index ==============

class GetComfortIndexInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Ten phuong/xa hoac quan/huyen")


@tool(args_schema=GetComfortIndexInput)
def get_comfort_index(ward_id: str = None, location_hint: str = None) -> dict:
    """Tinh diem THOAI MAI (0-100) ket hop nhiet do, do am, gio, UV, mua.

    DUNG KHI: "hom nay thoai mai khong?", "diem thoai mai bao nhieu?",
    "co de chiu khong?".
    Ho tro: phuong/xa, quan/huyen, toan Ha Noi.
    Tra ve: score (0-100), label, recommendation, breakdown tung yeu to.
    """
    from app.dal.weather_helpers import compute_comfort_index
    from app.agent.dispatch import resolve_and_dispatch, normalize_agg_keys

    def _comfort_ward(ward_id):
        from app.dal.weather_dal import get_current_weather
        weather = get_current_weather(ward_id)
        if weather.get("error"):
            return weather
        return _compute_comfort(weather)

    def _comfort_district(district_name):
        from app.dal.weather_aggregate_dal import get_district_current_weather
        weather = get_district_current_weather(district_name)
        if weather.get("error"):
            return weather
        weather = normalize_agg_keys(weather)
        return _compute_comfort(weather)

    def _comfort_city():
        from app.dal.weather_aggregate_dal import get_city_current_weather
        weather = get_city_current_weather()
        if weather.get("error"):
            return weather
        weather = normalize_agg_keys(weather)
        return _compute_comfort(weather)

    return resolve_and_dispatch(
        ward_id=ward_id,
        location_hint=location_hint,
        default_scope="city",
        ward_fn=_comfort_ward,
        district_fn=_comfort_district,
        city_fn=_comfort_city,
        normalize=False,
        label="chi so thoai mai",
    )


def _compute_comfort(weather: dict) -> dict:
    """Tinh comfort index tu weather data (da normalize)."""
    from app.dal.weather_helpers import compute_comfort_index
    comfort = compute_comfort_index(
        temp=weather.get("temp"),
        humidity=weather.get("humidity"),
        wind_speed=weather.get("wind_speed"),
        uvi=weather.get("uvi") or weather.get("uvi_max") or 0,
        pop=weather.get("pop") or 0,
    )
    if comfort is None:
        return {"error": "no_data", "message": "Khong du du lieu de tinh chi so thoai mai"}
    comfort["weather_snapshot"] = {
        "temp": weather.get("temp"),
        "humidity": weather.get("humidity"),
        "wind_speed": weather.get("wind_speed"),
        "uvi": weather.get("uvi"),
        "weather_main": weather.get("weather_main"),
    }
    return comfort


# ============== Tool: get_weather_change_alert ==============

class GetWeatherChangeAlertInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Ten phuong/xa hoac quan/huyen")
    hours: int = Field(default=6, description="So gio toi scan (1-12)")


@tool(args_schema=GetWeatherChangeAlertInput)
def get_weather_change_alert(ward_id: str = None, location_hint: str = None, hours: int = 6) -> dict:
    """Phat hien THAY DOI THOI TIET LON sap xay ra.

    DUNG KHI: "sap co gi thay doi khong?", "thoi tiet co bien dong khong?",
    "co chuyen mua khong?".
    Ho tro: phuong/xa, quan/huyen, toan Ha Noi.
    Tra ve: changes (temp drop/rise >5C, rain start/stop, wind increase, weather condition change).
    """
    from app.dal.weather_dal import detect_weather_changes as dal_ward_detect

    hours = max(1, min(hours, 12))

    def _district_detect(district_name):
        from app.dal.weather_aggregate_dal import (
            get_district_current_weather, get_district_hourly_forecast
        )
        from app.agent.dispatch import normalize_agg_keys
        current = get_district_current_weather(district_name)
        if current.get("error"):
            return current
        current = normalize_agg_keys(current)
        forecasts = get_district_hourly_forecast(district_name, hours)
        from app.agent.dispatch import normalize_rows
        forecasts = normalize_rows(forecasts)
        return _detect_changes(current, forecasts)

    def _city_detect():
        from app.dal.weather_aggregate_dal import (
            get_city_current_weather, get_city_hourly_forecast
        )
        from app.agent.dispatch import normalize_agg_keys
        current = get_city_current_weather()
        if current.get("error"):
            return current
        current = normalize_agg_keys(current)
        forecasts = get_city_hourly_forecast(hours)
        from app.agent.dispatch import normalize_rows
        forecasts = normalize_rows(forecasts)
        return _detect_changes(current, forecasts)

    from app.agent.dispatch import resolve_and_dispatch
    return resolve_and_dispatch(
        ward_id=ward_id,
        location_hint=location_hint,
        default_scope="city",
        ward_fn=lambda ward_id: dal_ward_detect(ward_id, hours),
        district_fn=_district_detect,
        city_fn=_city_detect,
        normalize=False,
        label="bien dong thoi tiet",
    )


def _detect_changes(current: dict, forecasts: list) -> dict:
    """Detect significant changes between current and forecasts (reusable)."""
    from app.dal.timezone_utils import format_ict

    if not forecasts:
        return {"error": "no_data", "message": "Khong co du lieu du bao"}

    changes = []
    cur_temp = current.get("temp")
    cur_pop = current.get("pop") or 0
    cur_wind = current.get("wind_speed") or 0
    cur_weather = current.get("weather_main", "")
    rain_keywords = {"Rain", "Drizzle", "Thunderstorm"}
    cur_is_rain = cur_weather in rain_keywords

    for f in forecasts:
        f_temp = f.get("temp")
        f_pop = f.get("pop") or 0
        f_wind = f.get("wind_speed") or 0
        f_weather = f.get("weather_main", "")
        f_time = format_ict(f.get("ts_utc"))
        f_is_rain = f_weather in rain_keywords

        if cur_temp is not None and f_temp is not None:
            temp_diff = f_temp - cur_temp
            if abs(temp_diff) >= 5:
                direction = "tang" if temp_diff > 0 else "giam"
                changes.append({
                    "type": "temperature",
                    "description": f"Nhiet do {direction} {abs(temp_diff):.1f}C ({cur_temp:.1f}->{f_temp:.1f}C)",
                    "time": f_time,
                    "severity": "high" if abs(temp_diff) >= 8 else "medium"
                })
                break

        if f_pop - cur_pop >= 0.5:
            changes.append({
                "type": "rain_start",
                "description": f"Kha nang mua tang manh ({cur_pop*100:.0f}%->{f_pop*100:.0f}%)",
                "time": f_time,
                "severity": "high" if f_pop >= 0.8 else "medium"
            })
            break

        if not cur_is_rain and f_is_rain:
            changes.append({
                "type": "weather_change",
                "description": f"Troi chuyen mua ({cur_weather}->{f_weather})",
                "time": f_time,
                "severity": "high" if f_weather == "Thunderstorm" else "medium"
            })
            break

        if cur_is_rain and not f_is_rain and f_pop < 0.3:
            changes.append({
                "type": "rain_stop", "description": "Mua co the tanh",
                "time": f_time, "severity": "low"
            })
            break

        if f_wind - cur_wind >= 5:
            changes.append({
                "type": "wind_increase",
                "description": f"Gio manh len ({cur_wind:.1f}->{f_wind:.1f} m/s)",
                "time": f_time,
                "severity": "high" if f_wind >= 15 else "medium"
            })
            break

    return {
        "changes": changes,
        "has_significant_change": len(changes) > 0,
        "hours_scanned": len(forecasts),
        "current_summary": {
            "temp": cur_temp, "weather_main": cur_weather,
            "wind_speed": cur_wind, "pop": cur_pop
        }
    }


# ============== Tool: get_clothing_advice ==============

class GetClothingAdviceInput(BaseModel):
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Ten phuong/xa hoac quan/huyen")
    hours_ahead: int = Field(default=0, description="So gio toi (0=hien tai)")


@tool(args_schema=GetClothingAdviceInput)
def get_clothing_advice(ward_id: str = None, location_hint: str = None, hours_ahead: int = 0) -> dict:
    """Khuyen nghi TRANG PHUC phu hop voi thoi tiet.

    DUNG KHI: "mac gi hom nay?", "can ao khoac khong?", "nen mang o khong?".
    Ho tro: phuong/xa, quan/huyen, toan Ha Noi.
    Tra ve: clothing_items, notes, va du lieu thoi tiet co ban.
    """
    from app.dal.activity_dal import get_clothing_advice as dal_ward_clothing
    from app.agent.dispatch import resolve_and_dispatch, normalize_agg_keys

    def _district_clothing(district_name):
        from app.dal.weather_aggregate_dal import get_district_current_weather
        weather = get_district_current_weather(district_name)
        if weather.get("error"):
            return weather
        weather = normalize_agg_keys(weather)
        return _clothing_from_weather(weather, hours_ahead)

    def _city_clothing():
        from app.dal.weather_aggregate_dal import get_city_current_weather
        weather = get_city_current_weather()
        if weather.get("error"):
            return weather
        weather = normalize_agg_keys(weather)
        return _clothing_from_weather(weather, hours_ahead)

    return resolve_and_dispatch(
        ward_id=ward_id,
        location_hint=location_hint,
        default_scope="city",
        ward_fn=lambda ward_id: dal_ward_clothing(ward_id, hours_ahead),
        district_fn=_district_clothing,
        city_fn=_city_clothing,
        normalize=False,
        label="khuyen nghi trang phuc",
    )


def _clothing_from_weather(weather: dict, hours_ahead: int = 0) -> dict:
    """Generate clothing advice from weather data (reusable for district/city)."""
    temp = weather.get("temp")
    humidity = weather.get("humidity") or 50
    pop = weather.get("pop") or 0
    wind = weather.get("wind_speed") or 0
    uvi = weather.get("uvi") or weather.get("uvi_max") or 0
    wm = weather.get("weather_main", "")

    if temp is None:
        return {"error": "Khong co du lieu nhiet do"}

    items = []
    notes = []

    if temp < 10:
        items.extend(["Ao phao/ao khoac day", "Khan quang co", "Gang tay", "Mu len"])
        notes.append("Ret dam - mac nhieu lop, giu am co va tay")
    elif temp < 15:
        items.extend(["Ao khoac day", "Ao len", "Quan dai"])
        notes.append("Lanh - nen mac ao khoac day")
    elif temp < 20:
        items.extend(["Ao khoac nhe", "Ao dai tay"])
        notes.append("Se lanh - ao khoac nhe la du")
    elif temp < 25:
        items.extend(["Ao thun dai tay hoac ngan tay", "Quan dai hoac short"])
    elif temp < 32:
        items.extend(["Ao mong thoang", "Quan short", "Mu chong nang"])
        notes.append("Nong - chon vai thoang mat")
    else:
        items.extend(["Ao mong thoang mat nhat", "Mu rong vanh", "Kinh ram"])
        notes.append("Rat nong - han che ra ngoai, uong nhieu nuoc")

    if pop > 0.5 or wm in ("Rain", "Drizzle", "Thunderstorm"):
        items.append("O/ao mua")
        notes.append("Co mua - nho mang o")
    elif pop > 0.3:
        items.append("O gap nho")
        notes.append("Co the mua - mang o phong")

    if humidity > 90 and temp > 20:
        notes.append("Nom am - tranh vai cotton, chon vai nhanh kho")

    if uvi >= 8:
        items.append("Kem chong nang SPF50+")
        if "Kinh ram" not in items:
            items.append("Kinh ram")
        notes.append("UV rat cao - bao ve da")
    elif uvi >= 5:
        items.append("Kem chong nang SPF30+")

    if wind > 8:
        notes.append("Gio manh - tranh ao rong, chon ao sat nguoi")

    return {
        "clothing_items": items, "notes": notes,
        "temp": temp, "humidity": humidity,
        "pop": round(pop * 100), "uvi": uvi, "wind_speed": wind,
    }


# ============== Tool: get_activity_advice ==============

class GetActivityAdviceInput(BaseModel):
    activity: str = Field(description="Hoat dong: chay_bo, picnic, bike, chup_anh, du_lich, cam_trai, ...")
    ward_id: Optional[str] = Field(default=None, description="Ward ID")
    location_hint: Optional[str] = Field(default=None, description="Ten phuong/xa hoac quan/huyen")


@tool(args_schema=GetActivityAdviceInput)
def get_activity_advice(activity: str, ward_id: str = None, location_hint: str = None) -> dict:
    """Khuyen cao co NEN thuc hien hoat dong ngoai troi khong.

    DUNG KHI: "di choi duoc khong?", "chay bo co on khong?", "co nen picnic khong?".
    Ho tro: phuong/xa, quan/huyen, toan Ha Noi.
    KHONG DUNG KHI: hoi may gio tot nhat (dung get_best_time),
    hoi mac gi (dung get_clothing_advice).
    Tra ve: muc khuyen cao (nen/co_the/han_che/khong_nen), ly do, khuyen nghi.
    """
    from app.dal.activity_dal import get_activity_advice as dal_ward_activity
    from app.agent.dispatch import resolve_and_dispatch, normalize_agg_keys

    def _district_activity(district_name):
        from app.dal.weather_aggregate_dal import get_district_current_weather
        weather = get_district_current_weather(district_name)
        if weather.get("error"):
            return weather
        weather = normalize_agg_keys(weather)
        # Reuse the same activity logic with normalized weather
        return _activity_from_weather(activity, weather)

    def _city_activity():
        from app.dal.weather_aggregate_dal import get_city_current_weather
        weather = get_city_current_weather()
        if weather.get("error"):
            return weather
        weather = normalize_agg_keys(weather)
        return _activity_from_weather(activity, weather)

    return resolve_and_dispatch(
        ward_id=ward_id,
        location_hint=location_hint,
        default_scope="city",
        ward_fn=lambda ward_id: dal_ward_activity(activity, ward_id),
        district_fn=_district_activity,
        city_fn=_city_activity,
        normalize=False,
        label="khuyen cao hoat dong",
    )


def _activity_from_weather(activity: str, weather: dict) -> dict:
    """Generate activity advice from weather data (district/city level)."""
    from app.dal.weather_knowledge_dal import detect_hanoi_weather_phenomena
    from app.config.thresholds import KTTV_THRESHOLDS, THRESHOLDS

    issues = []
    recommendations = []
    temp = weather.get("temp")
    humidity = weather.get("humidity")
    pop = weather.get("pop") or 0
    uvi = weather.get("uvi") or weather.get("uvi_max") or 0
    wind_speed = weather.get("wind_speed") or 0
    weather_main = weather.get("weather_main", "")

    if pop > THRESHOLDS.get("POP_VERY_LIKELY", 0.8):
        issues.append(f"Kha nang mua cao ({pop*100:.0f}%)")
        recommendations.append("Nen mang ao mua hoac o")
    elif pop > THRESHOLDS.get("POP_LIKELY", 0.5):
        issues.append(f"Co the mua ({pop*100:.0f}%)")
        recommendations.append("Nen mang o phong mua")

    if temp is not None:
        if temp > KTTV_THRESHOLDS["NANG_NONG"]:
            issues.append(f"Nhiet do cao ({temp}C)")
            recommendations.append("Nen chon buoi sang som (6-9h) hoac chieu muon (17h tro di)")
        elif temp < KTTV_THRESHOLDS["RET_DAM"]:
            issues.append(f"Nhiet do thap ({temp}C)")
            recommendations.append("Mac am, han che ra ngoai vao ban dem")

    if uvi >= 10:
        issues.append(f"UV cuc cao ({uvi})")
        recommendations.append("Han che ra ngoai 10h-14h, dung kem chong nang SPF50+")
    elif uvi >= 7:
        issues.append(f"UV cao ({uvi})")
        recommendations.append("Doi mu, dung kem chong nang")

    if wind_speed > 20:
        issues.append(f"Gio rat manh ({wind_speed} m/s)")
        recommendations.append("NGUY HIEM - Khong nen ra ngoai")
    elif wind_speed > 10:
        issues.append(f"Gio manh ({wind_speed} m/s)")

    phenomena = detect_hanoi_weather_phenomena(weather)
    for p in phenomena.get("phenomena", []):
        issues.append(p["name"])
        recommendations.append(p["description"])

    if len(issues) == 0:
        advice, reason = "nen", "Thoi tiet thuan loi cho hoat dong ngoai troi"
    elif len(issues) == 1:
        advice, reason = "co_the", f"Can luu y: {issues[0]}"
    elif any("NGUY HIEM" in r for r in recommendations):
        advice, reason = "khong_nen", f"Thoi tiet nguy hiem: {', '.join(issues)}"
    else:
        advice, reason = "han_che", f"Nhieu yeu to bat loi: {', '.join(issues)}"

    return {
        "advice": advice, "reason": reason, "recommendations": recommendations,
        "activity": activity, "temp": temp, "humidity": humidity,
        "pop": pop, "uvi": uvi, "wind_speed": wind_speed,
        "phenomena": phenomena.get("phenomena", []),
    }
