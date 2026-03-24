"""Weather API — JSON endpoints + HTMX partial renderers."""

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter()


def _templates(request: Request):
    return request.app.state.templates


@router.get("/current")
async def api_current_weather(
    request: Request,
    ward_id: str = Query(default="ID_00364"),
    format: str = Query(default="json", pattern="^(json|html)$"),
):
    """Current weather — JSON or HTMX partial."""
    from app.dal.weather_dal import get_current_weather
    data = get_current_weather(ward_id)

    if format == "html":
        return _templates(request).TemplateResponse(request, "partials/weather_now.html", {
            "current": data,
        })
    return data


@router.get("/hourly")
async def api_hourly_forecast(
    request: Request,
    ward_id: str = Query(default="ID_00364"),
    hours: int = Query(default=48, ge=1, le=48),
    format: str = Query(default="json", pattern="^(json|html)$"),
):
    """Hourly forecast — JSON or HTMX partial."""
    from app.dal.weather_dal import get_hourly_forecast
    data = get_hourly_forecast(ward_id, hours=hours)

    # Fallback: if no future forecasts, show most recent forecast data
    if not data:
        from app.db.dal import query as db_query
        from app.dal.timezone_utils import format_ict
        from app.dal.weather_helpers import wind_deg_to_vietnamese, wind_speed_to_beaufort
        rows = db_query("""
            SELECT ts_utc, temp, feels_like, humidity, dew_point, pop, rain_1h,
                   wind_speed, wind_deg, clouds, weather_main, weather_description
            FROM fact_weather_hourly
            WHERE ward_id = %s AND data_kind = 'forecast'
            ORDER BY ts_utc DESC
            LIMIT %s
        """, (ward_id, hours))
        rows.reverse()
        for r in rows:
            r["wind_direction_vi"] = wind_deg_to_vietnamese(r.get("wind_deg"))
            r["wind_beaufort"] = wind_speed_to_beaufort(r.get("wind_speed"))
            r["time_ict"] = format_ict(r.get("ts_utc"))
        data = rows

    if format == "html":
        return _templates(request).TemplateResponse(request, "partials/hourly_table.html", {
            "hourly": data,
        })
    return data


@router.get("/daily")
async def api_daily_forecast(
    request: Request,
    ward_id: str = Query(default="ID_00364"),
    days: int = Query(default=8, ge=1, le=8),
    format: str = Query(default="json", pattern="^(json|html)$"),
):
    """Daily forecast — JSON or HTMX partial."""
    from app.dal.weather_dal import get_daily_forecast
    data = get_daily_forecast(ward_id, days=days)

    if format == "html":
        return _templates(request).TemplateResponse(request, "partials/daily_list.html", {
            "daily": data,
        })
    return data


@router.get("/districts")
async def api_districts():
    """Get all districts."""
    from app.dal.location_dal import get_districts
    return get_districts()


@router.get("/wards")
async def api_wards(district: str = Query(...)):
    """Get wards in a district — returns HTML <option> elements for HTMX."""
    from app.dal.location_dal import get_wards_in_district
    wards = get_wards_in_district(district)
    options = '<option value="">-- Chọn phường/xã --</option>'
    for w in wards:
        options += f'<option value="{w["ward_id"]}">{w["ward_name_vi"]}</option>'
    return HTMLResponse(content=options)
