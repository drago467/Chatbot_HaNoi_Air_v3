"""FastAPI app entry point.

Khởi tạo app, middleware, logging, routes. Dùng lifespan thay cho on_event
(cách mới, tránh deprecation warning).

Run:
    uvicorn app.api.main:app --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Monkey-patch langchain trước khi import agent
from app.core import compat  # noqa: F401
from app.core.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan: log startup. Shutdown: cleanup."""
    logger.info("FastAPI starting up...")
    yield
    logger.info("FastAPI shutting down...")


app = FastAPI(
    title="Chatbot Thời tiết Hà Nội API",
    description="API cho chatbot thời tiết Hà Nội với SLM router + multi-turn + 27 tools",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — thesis scope cho phép tất cả, production thì tighten
_cors_origins = os.getenv("API_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký routes
from app.api.routes import chat, conversations, health, jobs, weather  # noqa: E402

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(jobs.router)
app.include_router(weather.router)


@app.get("/")
def root():
    """Trả link tới Swagger UI."""
    return {
        "name": "Chatbot Thời tiết Hà Nội API",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
    }
