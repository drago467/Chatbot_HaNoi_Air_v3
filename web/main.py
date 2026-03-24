"""HanoiAir Weather — FastAPI + Jinja2 + HTMX Web Application."""

import json
import os
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is in path so `app.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.routers import pages, weather_api, chat_api, auth
from web.core.security import decode_access_token


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown events."""
    yield


app = FastAPI(
    title="HanoiAir Weather",
    description="Hệ thống Tra cứu và Cảnh báo Thời tiết Hà Nội",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Static files & Templates ---
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

templates = Jinja2Templates(directory=BASE_DIR / "templates")


# Custom JSON serializer for Jinja2 tojson filter (handles datetime, date, Decimal, timedelta)
def _json_serializer(obj, **kwargs):
    def _default(o):
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, date):
            return o.isoformat()
        if isinstance(o, timedelta):
            return str(o)
        if isinstance(o, Decimal):
            return float(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    return json.dumps(obj, default=_default, **kwargs)

templates.env.policies["json.dumps_function"] = _json_serializer


# Custom Jinja2 filters for safe numeric operations
def _safe_round(value, precision=0):
    """Round that handles None gracefully."""
    if value is None:
        return 0
    try:
        return round(float(value), precision)
    except (TypeError, ValueError):
        return 0

templates.env.filters["safe_round"] = _safe_round
templates.env.globals["safe"] = lambda v, default=0: v if v is not None else default

# Share templates instance with routers
app.state.templates = templates

# --- Auth middleware ---
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Decode JWT from cookie and attach user info to request.state."""
    request.state.user = None
    token = request.cookies.get("access_token")
    if token:
        payload = decode_access_token(token)
        if payload:
            request.state.user = payload  # {"sub": "id", "username": "...", "exp": ...}
    response = await call_next(request)
    return response

# --- Routers ---
app.include_router(pages.router)
app.include_router(weather_api.router, prefix="/api/weather", tags=["weather"])
app.include_router(chat_api.router, prefix="/api/chat", tags=["chat"])
app.include_router(auth.router, tags=["auth"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.main:app", host="0.0.0.0", port=8000, reload=True)
