"""Page routes — serve full HTML pages via Jinja2 + HTMX."""

from urllib.parse import unquote

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(tags=["pages"])

# Default ward: Dịch Vọng Hậu, Cầu Giấy
DEFAULT_WARD_ID = "ID_00364"
DEFAULT_DISTRICT = "Cầu Giấy"


def _templates(request: Request):
    return request.app.state.templates


def _ward_id(request: Request) -> str:
    """Get ward_id from cookie or default."""
    return request.cookies.get("ward_id", DEFAULT_WARD_ID)


def _district(request: Request) -> str:
    return unquote(request.cookies.get("district", DEFAULT_DISTRICT))


def _user(request: Request):
    """Get current user from auth middleware (or None)."""
    return getattr(request.state, "user", None)


@router.get("/", response_class=HTMLResponse)
async def page_today(request: Request):
    """Trang Hôm nay — current weather + ngày/đêm."""
    from app.dal.weather_dal import get_current_weather, get_daily_forecast
    from app.dal.location_dal import get_districts

    ward_id = _ward_id(request)
    district = _district(request)

    current = get_current_weather(ward_id)
    daily = get_daily_forecast(ward_id, days=8)
    districts = get_districts()

    return _templates(request).TemplateResponse(request, "pages/today.html", {
        "current": current,
        "daily": daily,
        "districts": districts,
        "active_page": "today",
        "district": district,
        "ward_id": ward_id,
        "user": _user(request),
    })


@router.get("/hourly", response_class=HTMLResponse)
async def page_hourly(request: Request):
    """Trang Hàng giờ — 48h forecast."""
    from app.dal.weather_dal import get_hourly_forecast, get_current_weather
    from app.dal.location_dal import get_districts
    from app.db.dal import query as db_query

    ward_id = _ward_id(request)
    district = _district(request)
    hourly = get_hourly_forecast(ward_id, hours=48)
    current = get_current_weather(ward_id)

    # Fallback: if no future forecasts, show most recent forecast data
    if not hourly:
        from app.dal.timezone_utils import format_ict
        from app.dal.weather_helpers import wind_deg_to_vietnamese, wind_speed_to_beaufort
        rows = db_query("""
            SELECT ts_utc, temp, feels_like, humidity, dew_point, pop, rain_1h,
                   wind_speed, wind_deg, clouds, weather_main, weather_description
            FROM fact_weather_hourly
            WHERE ward_id = %s AND data_kind = 'forecast'
            ORDER BY ts_utc DESC
            LIMIT 48
        """, (ward_id,))
        rows.reverse()
        for r in rows:
            r["wind_direction_vi"] = wind_deg_to_vietnamese(r.get("wind_deg"))
            r["wind_beaufort"] = wind_speed_to_beaufort(r.get("wind_speed"))
            r["time_ict"] = format_ict(r.get("ts_utc"))
        hourly = rows

    districts = get_districts()

    return _templates(request).TemplateResponse(request, "pages/hourly.html", {
        "hourly": hourly,
        "current": current,
        "districts": districts,
        "active_page": "hourly",
        "district": district,
        "ward_id": ward_id,
        "user": _user(request),
    })


@router.get("/daily", response_class=HTMLResponse)
async def page_daily(request: Request):
    """Trang 8 ngày — daily forecast."""
    from app.dal.weather_dal import get_daily_forecast, get_current_weather
    from app.dal.location_dal import get_districts

    ward_id = _ward_id(request)
    district = _district(request)
    daily = get_daily_forecast(ward_id, days=8)
    current = get_current_weather(ward_id)
    districts = get_districts()

    return _templates(request).TemplateResponse(request, "pages/daily.html", {
        "daily": daily,
        "current": current,
        "districts": districts,
        "active_page": "daily",
        "district": district,
        "ward_id": ward_id,
        "user": _user(request),
    })


@router.get("/chat", response_class=HTMLResponse)
async def page_chat(request: Request):
    """Trang Trợ lý thời tiết — chatbot (requires login)."""
    if not _user(request):
        return RedirectResponse("/login?next=/chat", status_code=303)

    from app.dal.location_dal import get_districts

    district = _district(request)
    districts = get_districts()

    return _templates(request).TemplateResponse(request, "pages/chat.html", {
        "districts": districts,
        "active_page": "chat",
        "district": district,
        "user": _user(request),
    })


@router.get("/alerts", response_class=HTMLResponse)
async def page_alerts(request: Request):
    """Trang Cảnh báo thời tiết."""
    from app.dal.location_dal import get_districts
    from app.dal.alerts_dal import get_weather_alerts

    ward_id = _ward_id(request)
    district = _district(request)
    districts = get_districts()

    try:
        alerts = get_weather_alerts(ward_id)
    except Exception:
        alerts = []

    try:
        from app.dal.weather_dal import get_current_weather
        current = get_current_weather(ward_id)
    except Exception:
        current = None

    return _templates(request).TemplateResponse(request, "pages/alerts.html", {
        "districts": districts,
        "alerts": alerts,
        "current": current,
        "active_page": "alerts",
        "district": district,
        "ward_id": ward_id,
        "user": _user(request),
    })
