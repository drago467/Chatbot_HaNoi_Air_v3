"""Chat API — SSE streaming endpoint for chatbot."""

import asyncio
import queue as sync_queue
import uuid

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

router = APIRouter()


def _templates(request: Request):
    return request.app.state.templates


def _require_auth(request: Request) -> dict | None:
    """Return user payload or None if not authenticated."""
    return getattr(request.state, "user", None)


async def _stream_agent(prompt: str, thread_id: str):
    """Wrap synchronous agent stream into async SSE generator (non-blocking).

    Runs the blocking LangGraph stream in a thread-pool executor and
    forwards chunks via a thread-safe queue so the event loop stays free.
    """
    from app.agent.agent import stream_agent_routed

    q: sync_queue.Queue = sync_queue.Queue()

    def _produce():
        try:
            full = ""
            for chunk in stream_agent_routed(prompt, thread_id=thread_id):
                full += chunk
                q.put(("chunk", chunk))
            # Try to get follow-up suggestions from ConversationState
            suggestions = []
            try:
                from app.agent.conversation_state import get_follow_up_suggestions
                suggestions = get_follow_up_suggestions(thread_id) or []
            except Exception:
                pass
            q.put(("done", full))
            if suggestions:
                import json as _json
                q.put(("suggestions", _json.dumps(suggestions, ensure_ascii=False)))
        except Exception as exc:
            q.put(("error", str(exc)))

    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(None, _produce)

    while True:
        # Poll without blocking the event loop
        try:
            item = q.get_nowait()
        except sync_queue.Empty:
            await asyncio.sleep(0.02)
            continue

        event_type, data = item
        yield {"event": event_type, "data": data}
        if event_type in ("done", "error"):
            break

    await fut  # ensure the thread finishes cleanly


@router.post("/send")
async def chat_send(
    request: Request,
    message: str = Form(...),
    thread_id: str = Form(default=""),
):
    """Send a chat message — returns assistant message HTML partial via HTMX."""
    if not _require_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if not thread_id:
        thread_id = str(uuid.uuid4())

    from app.agent.router.config import USE_SLM_ROUTER
    if USE_SLM_ROUTER:
        from app.agent.agent import run_agent_routed
        result = run_agent_routed(message, thread_id=thread_id)
    else:
        from app.agent.agent import run_agent
        result = run_agent(message, thread_id=thread_id)

    return _templates(request).TemplateResponse(request, "partials/chat_message.html", {
        "role": "assistant",
        "content": result,
        "thread_id": thread_id,
    })


@router.post("/stream")
async def chat_stream(
    request: Request,
    message: str = Form(...),
    thread_id: str = Form(default=""),
):
    """SSE streaming chat — for real-time response rendering."""
    if not _require_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if not thread_id:
        thread_id = str(uuid.uuid4())

    return EventSourceResponse(_stream_agent(message, thread_id))
