"""Smoke test cho /health + /ready + root."""


def test_root(api_client):
    resp = api_client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["docs"] == "/docs"


def test_health(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
