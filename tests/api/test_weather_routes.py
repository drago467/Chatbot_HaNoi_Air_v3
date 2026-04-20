"""Smoke test cho /weather/* + /locations/*.

Mock DB query + Redis cache để test độc lập với infra.
"""

from unittest.mock import patch


def _no_cache(key, ttl, fetch_fn):
    """Bỏ qua Redis — gọi thẳng fetch_fn."""
    return fetch_fn()


def test_list_districts(api_client):
    fake_rows = [
        {"district_name_vi": "Cầu Giấy"},
        {"district_name_vi": "Đống Đa"},
    ]
    with patch("app.api.routes.weather.cache_get_or_fetch", side_effect=_no_cache), \
         patch("app.api.routes.weather.query", return_value=fake_rows):
        resp = api_client.get("/locations/districts")

    assert resp.status_code == 200
    assert resp.json() == ["Cầu Giấy", "Đống Đa"]


def test_list_wards(api_client):
    fake_rows = [
        {"ward_id": "ID_00001", "ward_name_vi": "Nghĩa Tân"},
        {"ward_id": "ID_00002", "ward_name_vi": "Dịch Vọng"},
    ]
    with patch("app.api.routes.weather.cache_get_or_fetch", side_effect=_no_cache), \
         patch("app.api.routes.weather.query", return_value=fake_rows):
        resp = api_client.get("/locations/wards/Cầu Giấy")

    assert resp.status_code == 200
    assert resp.json() == {"Nghĩa Tân": "ID_00001", "Dịch Vọng": "ID_00002"}


def test_get_current_weather(api_client):
    fake_rows = [{
        "temp": 28.5, "humidity": 65.0, "weather_main": "Clouds",
        "wind_speed": 2.3, "wind_deg": 180.0,
    }]
    with patch("app.api.routes.weather.cache_get_or_fetch", side_effect=_no_cache), \
         patch("app.api.routes.weather.query", return_value=fake_rows):
        resp = api_client.get("/weather/current/ID_00001")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ward_id"] == "ID_00001"
    assert body["temp"] == 28.5
    assert body["weather_main"] == "Clouds"


def test_get_current_weather_not_found(api_client):
    with patch("app.api.routes.weather.cache_get_or_fetch", side_effect=_no_cache), \
         patch("app.api.routes.weather.query", return_value=[]):
        resp = api_client.get("/weather/current/ID_NOT_EXIST")

    assert resp.status_code == 404
