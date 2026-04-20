"""Celery app cho Chatbot Thời tiết Hà Nội.

Chức năng:
- Chạy background tasks: ingest OpenWeather, aggregate, chat async, telemetry.
- Beat scheduler: ingest định kỳ mỗi giờ.

Usage (trong terminal):
    celery -A app.workers.celery_app worker --loglevel=info
    celery -A app.workers.celery_app beat --loglevel=info
"""

import os

from celery import Celery
from celery.schedules import crontab

from app.core.logging_config import setup_logging

# Đảm bảo logging cấu hình trước khi worker khởi động
setup_logging()

# Broker + result backend dùng Redis (2 DB khác nhau cho tách biệt)
BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

celery = Celery("hanoiweather", broker=BROKER_URL, backend=RESULT_BACKEND)

celery.conf.update(
    # Acks muộn: worker crash giữa chừng -> task tự re-queue cho worker khác
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Track khi task START để endpoint /tasks/{id} thấy trạng thái PROGRESS sớm
    task_track_started=True,
    # Mỗi worker giữ 1 task 1 lúc -> fair dispatch, tránh pile-up
    worker_prefetch_multiplier=1,
    # Kết quả hết hạn sau 1 giờ
    result_expires=3600,
    timezone="Asia/Ho_Chi_Minh",
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Beat: mỗi giờ, vào phút thứ 5 -> ingest + aggregate
    beat_schedule={
        "ingest-hourly": {
            "task": "app.workers.tasks.ingest.ingest_and_aggregate_chain",
            "schedule": crontab(minute=5),
        },
    },
)

# Auto discover tasks trong các submodule của app/workers/tasks/
celery.autodiscover_tasks(["app.workers.tasks"], related_name=None)

# Import sẵn tasks để celery worker nhận biết khi boot
from app.workers.tasks import ingest, chat, telemetry  # noqa: E402,F401
