"""Server-Sent Events streaming for the chat flow.

Emits incremental events as the agent parses the request, plans, routes to a
sub-agent, runs each tool, and generates the final itinerary. This mirrors
the streaming UX of the Azure AI Travel Agents sample ("Agents Reasoning…"
panel) while keeping the existing synchronous tool implementations intact.

Streaming is truly progressive: the chat core runs in a background thread and
pushes events to a queue as they happen (plan steps, agent_start, tool_start,
tool_end, agent_end); the generator yields each event as an SSE frame the
moment it is produced, then streams the final message in chunks and a final
``result`` frame carrying the full ChatResponse.

Event shape (one JSON object per SSE ``data:`` line):
    {"type": "plan", "step": "Parse destination ..."}
    {"type": "route", "agent": "ItineraryAgent"}
    {"type": "agent_start", "agent": "ItineraryAgent"}
    {"type": "tool_start", "tool": "destination_rag", "query": "..."}
    {"type": "tool_end", "tool": "weather_tool", "status": "completed"}
    {"type": "agent_end", "agent": "ItineraryAgent"}
    {"type": "clarification", "question": "..."}
    {"type": "message", "delta": "Here is ..."}
    {"type": "result", "response": { ...full ChatResponse... }}
    {"type": "error", "message": "..."}

The final result is identical to what ``handle_chat`` returns, so streaming
and non-streaming clients receive the same payload.
"""

from __future__ import annotations

import json
import queue as _queue
import threading
from collections.abc import Generator
from typing import Any

from app.agent.orchestrator import AgentServices, _run_chat_core
from app.memory.long_term import long_term_memory
from app.memory.short_term import short_term_memory
from app.schemas import ChatRequest


def stream_chat(
    request: ChatRequest,
    memory: Any | None = None,
    user_memory: Any | None = None,
    services: AgentServices | None = None,
) -> Generator[str, None, None]:
    """Yield SSE-formatted frames describing the chat flow as it happens.

    Runs ``_run_chat_core`` in a background thread with an emitter that pushes
    events onto a queue; this generator yields each event as an SSE frame the
    moment it is produced, then streams the final message and a result frame.
    """
    memory = memory or short_term_memory
    user_memory = user_memory or long_term_memory
    services = services or AgentServices()

    event_queue: _queue.Queue = _queue.Queue()
    sentinel = object()
    outcome: dict[str, Any] = {}

    def emitter(event: dict[str, Any]) -> None:
        event_queue.put(event)

    def worker() -> None:
        try:
            outcome["response"] = _run_chat_core(
                request,
                memory=memory,
                user_memory=user_memory,
                services=services,
                event_emitter=emitter,
            )
        except Exception as error:  # noqa: BLE001 - surface any failure to the client
            outcome["error"] = error
        finally:
            event_queue.put(sentinel)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        while True:
            item = event_queue.get()
            if item is sentinel:
                break
            yield _frame(item)
        if "error" in outcome:
            yield _frame({"type": "error", "message": str(outcome["error"])})
            return
        response = outcome["response"]
        if response.needs_clarification:
            yield _frame({"type": "clarification", "question": response.clarifying_question or ""})
        yield from _stream_message(response.message)
        yield _frame({"type": "result", "response": response.model_dump()})
    except Exception as error:  # noqa: BLE001 - never leave the client hanging
        yield _frame({"type": "error", "message": str(error)})
    finally:
        thread.join(timeout=1.0)


def _stream_message(message: str) -> Generator[str, None, None]:
    """Yield message deltas in word chunks to simulate token streaming."""
    if not message:
        return
    words = message.split()
    chunk_size = 4
    for index in range(0, len(words), chunk_size):
        delta = " ".join(words[index : index + chunk_size])
        yield _frame(
            {"type": "message", "delta": delta + (" " if index + chunk_size < len(words) else "")}
        )


def _frame(event: dict[str, Any]) -> str:
    """Format a single SSE data frame from an event dict."""
    event_type = event.get("type", "event")
    payload = {key: value for key, value in event.items() if key != "type"}
    data = json.dumps({"type": event_type, **payload}, default=str, ensure_ascii=False)
    return f"data: {data}\n\n"
