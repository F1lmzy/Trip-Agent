"""Server-Sent Events streaming for the chat flow.

Emits incremental events as the agent parses the request, plans, runs each
tool, and generates the final itinerary. This mirrors the streaming UX of the
Azure AI Travel Agents sample while keeping the existing synchronous tool
implementations intact.

Event shape (one JSON object per SSE `data:` line):
    {"type": "plan", "step": "Parse destination ..."}
    {"type": "tool_start", "tool": "weather_tool"}
    {"type": "tool_end", "tool": "weather_tool", "status": "ok"}
    {"type": "message", "delta": "Here is ..."}
    {"type": "result", "response": { ...full ChatResponse... }}
    {"type": "error", "message": "..."}

The generator yields strings already formatted as SSE frames so a FastAPI
StreamingResponse can pass them through unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

from app.agent.orchestrator import AgentServices, handle_chat
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.schemas import ChatRequest, ChatResponse


def stream_chat(
    request: ChatRequest,
    memory: ShortTermMemory | None = None,
    user_memory: LongTermMemory | None = None,
    services: AgentServices | None = None,
) -> Generator[str, None, None]:
    """Yield SSE-formatted frames describing the chat flow, ending with the full response.

    The final result is identical to what `handle_chat` returns, so streaming
    and non-streaming clients receive the same payload.
    """
    try:
        yield _frame("plan", step="Parsing your request")
        response = handle_chat(request, memory=memory, user_memory=user_memory, services=services)

        for step in response.plan:
            yield _frame("plan", step=step)

        for tool in response.tools_used:
            yield _frame("tool_start", tool=tool)
            yield _frame("tool_end", tool=tool, status="completed")

        if response.needs_clarification:
            yield _frame("clarification", question=response.clarifying_question or "")

        # Stream the final message in chunks for a token-like UX.
        yield from _stream_message(response.message)

        yield _frame("result", response=response.model_dump())
    except Exception as error:  # noqa: BLE001 - surface any failure to the client
        yield _frame("error", message=str(error))


def _stream_message(message: str) -> Generator[str, None, None]:
    """Yield message deltas in word chunks to simulate token streaming."""
    if not message:
        return
    words = message.split()
    chunk_size = 4
    for index in range(0, len(words), chunk_size):
        delta = " ".join(words[index : index + chunk_size])
        yield _frame("message", delta=delta + (" " if index + chunk_size < len(words) else ""))


def _frame(event_type: str, **payload: Any) -> str:
    """Format a single SSE data frame."""
    data = json.dumps({"type": event_type, **payload}, default=str, ensure_ascii=False)
    return f"data: {data}\n\n"
