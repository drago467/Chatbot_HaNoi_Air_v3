"""Celery tasks cho ingest + aggregate weather data từ OpenWeather.

- ingest_weather_task: gọi OpenWeatherAsyncIngestor.run_nowcast/forecast/history.
- aggregate_weather_task: chạy aggregation sau khi ingest xong.
- ingest_and_aggregate_chain: chain 2 task trên -> dùng cho Beat hourly.
"""

import asyncio

from celery import chain, shared_task

from app.core.logging_config import get_logger

logger = get_logger(__name__)


@shared_task(bind=True, name="app.workers.tasks.ingest.ingest_weather_task")
def ingest_weather_task(self, include_history: bool = False, history_days: int = 7):
    """Ingest weather data từ OpenWeather API.

    Args:
        include_history: có backfill lịch sử không.
        history_days: số ngày lịch sử nếu include_history=True.

    Cập nhật progress qua self.update_state để endpoint /tasks/{id} hiển thị.
    """
    from app.scripts.ingest_openweather_async import OpenWeatherAsyncIngestor

    ingestor = OpenWeatherAsyncIngestor()

    logger.info("ingest_weather_task start (history=%s, days=%d)", include_history, history_days)

    self.update_state(state="PROGRESS", meta={"progress": 0.1, "step": "nowcast"})
    asyncio.run(ingestor.run_nowcast())

    self.update_state(state="PROGRESS", meta={"progress": 0.5, "step": "forecast"})
    asyncio.run(ingestor.run_forecast())

    if include_history:
        self.update_state(state="PROGRESS", meta={"progress": 0.75, "step": "history"})
        asyncio.run(ingestor.run_history_backfill(days=history_days))

    self.update_state(state="PROGRESS", meta={"progress": 1.0, "step": "done"})
    logger.info("ingest_weather_task completed")
    return {"status": "ok", "include_history": include_history}


@shared_task(name="app.workers.tasks.ingest.aggregate_weather_task")
def aggregate_weather_task(_prev_result=None):
    """Aggregate ward-level data lên district + city. Chạy sau ingest."""
    from app.scripts.run_aggregation import run_aggregation

    logger.info("aggregate_weather_task start")
    results = run_aggregation()
    errors = [k for k, v in results.items() if v.get("status") == "error"]
    if errors:
        logger.warning("aggregate_weather_task partial errors: %s", errors)
    logger.info("aggregate_weather_task completed")
    return {"status": "ok", "errors": errors}


@shared_task(name="app.workers.tasks.ingest.ingest_and_aggregate_chain")
def ingest_and_aggregate_chain():
    """Chain ingest -> aggregate. Được Beat gọi mỗi giờ."""
    logger.info("Beat triggered ingest_and_aggregate_chain")
    # Dùng .si() (immutable signature) để aggregate không nhận result từ ingest
    workflow = chain(
        ingest_weather_task.si(include_history=False),
        aggregate_weather_task.si(),
    )
    async_result = workflow.apply_async()
    return {"chain_id": async_result.id}
