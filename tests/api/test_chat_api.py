"""Smoke test cho /chat routes.

Mock agent core để test chạy nhanh + không cần LLM/DB thật.
"""

from unittest.mock import patch

import pytest


def test_chat_sync_returns_result(api_client):
    """POST /chat → 200 + result dict. Agent bị mock."""
    fake_result = {"messages": [{"role": "assistant", "content": "Hello"}]}

    with patch("app.agent.agent.run_agent_routed", return_value=fake_result):
        resp = api_client.post(
            "/chat",
            json={"message": "Thời tiết hôm nay?"},
            headers={"X-Thread-Id": "test-sync-1"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["thread_id"] == "test-sync-1"
    assert body["result"] == fake_result


def test_chat_stream_returns_sse(api_client):
    """POST /chat/stream → 200 + content-type text/event-stream, first 2 tokens match."""
    def fake_gen(msg, thread_id):
        yield "Hello "
        yield "world"

    with patch("app.agent.agent.stream_agent_routed", side_effect=fake_gen):
        with api_client.stream(
            "POST",
            "/chat/stream",
            json={"message": "Test"},
            headers={"X-Thread-Id": "test-stream-1"},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")

            # Đọc vài dòng đầu để xác nhận có token event
            lines = []
            for line in resp.iter_lines():
                lines.append(line)
                if len(lines) >= 6:
                    break

    text = "\n".join(lines)
    assert "event: token" in text
    assert "Hello " in text


