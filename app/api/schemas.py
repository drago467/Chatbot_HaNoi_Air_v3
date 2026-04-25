"""Pydantic schemas cho request/response của FastAPI."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Câu hỏi của user")
    thread_id: Optional[str] = Field(None, description="ID hội thoại (optional, ưu tiên header X-Thread-Id)")


class ChatSyncResponse(BaseModel):
    thread_id: str
    result: dict[str, Any]


class IngestRequest(BaseModel):
    include_history: bool = False
    history_days: int = 7


class IngestResponse(BaseModel):
    status: str
    include_history: bool
    history_days: int
    message: str


class ConversationSummary(BaseModel):
    conv_id: str
    thread_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationDetail(BaseModel):
    conv_id: str
    thread_id: str
    title: str
    messages: list[dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class WeatherCurrent(BaseModel):
    ward_id: str
    temp: Optional[float] = None
    humidity: Optional[float] = None
    weather_main: Optional[str] = None
    wind_speed: Optional[float] = None
    wind_deg: Optional[float] = None


class ForecastPoint(BaseModel):
    time_local: datetime
    temp: Optional[float] = None
    humidity: Optional[float] = None


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    postgres: str
    router: str
    llm: str
