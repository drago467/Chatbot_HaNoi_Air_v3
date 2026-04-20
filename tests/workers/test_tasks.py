"""Smoke test cho Celery tasks — chạy eager (autouse fixture trong conftest).

Mock agent + DB để test không phụ thuộc infra thật.
"""

from unittest.mock import MagicMock, patch


def test_run_chat_task_calls_agent():
    """chat task gọi run_agent_routed đúng cách."""
    from app.workers.tasks.chat import run_chat_task

    fake_result = {"messages": [{"role": "assistant", "content": "Hi"}]}
    with patch("app.agent.agent.run_agent_routed", return_value=fake_result) as m:
        result = run_chat_task.apply(args=["Hello", "thread-1"]).get()

    m.assert_called_once_with("Hello", thread_id="thread-1")
    assert result == {"status": "ok", "result": fake_result}


def test_log_conversation_task_writes_telemetry():
    """telemetry task gọi evaluation logger."""
    from app.workers.tasks.telemetry import log_conversation_task

    mock_logger = MagicMock()
    with patch("app.agent.telemetry.get_evaluation_logger", return_value=mock_logger):
        result = log_conversation_task.apply(
            kwargs={
                "session_id": "s1",
                "turn_number": 1,
                "user_query": "Hello",
                "llm_response": "Hi",
                "response_time_ms": 123.4,
            }
        ).get()

    assert result == {"status": "ok"}
    mock_logger.log_conversation.assert_called_once()


def test_ingest_weather_task_runs():
    """ingest task gọi OpenWeatherAsyncIngestor nowcast + forecast."""
    from app.workers.tasks.ingest import ingest_weather_task

    mock_ingestor = MagicMock()
    mock_ingestor.run_nowcast = MagicMock(return_value=_async_noop())
    mock_ingestor.run_forecast = MagicMock(return_value=_async_noop())

    with patch(
        "app.scripts.ingest_openweather_async.OpenWeatherAsyncIngestor",
        return_value=mock_ingestor,
    ):
        result = ingest_weather_task.apply(kwargs={"include_history": False}).get()

    assert result["status"] == "ok"
    assert result["include_history"] is False
    mock_ingestor.run_nowcast.assert_called_once()
    mock_ingestor.run_forecast.assert_called_once()


async def _async_noop():
    """Helper: async function trả None (dùng cho asyncio.run)."""
    return None
