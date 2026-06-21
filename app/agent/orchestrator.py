from dataclasses import dataclass
from typing import Any

import httpx

from app.agent.parser import parse_user_request
from app.agent.planner import create_trip_plan
from app.agent.response_builders import (
    build_clarification_response as _build_clarification_response,
)
from app.agent.response_builders import build_follow_up_response as _build_follow_up_response
from app.agent.response_builders import save_stable_preferences as _save_stable_preferences
from app.agent.service_utils import service_value as _service_value
from app.agent.tool_executor import attach_images as _attach_images
from app.agent.tool_executor import execute_tools as _execute_tools
from app.agent.tool_executor import fresh_travel_search_intent as _fresh_travel_search_intent
from app.agent.tool_executor import resolve_departure_date as _resolve_departure_date
from app.agent.tool_executor import resolve_return_date as _resolve_return_date
from app.agent.tool_executor import run_flight as _run_flight
from app.agent.tool_executor import run_hotel as _run_hotel
from app.agents import AgentContext, Supervisor
from app.memory.long_term import LongTermMemory, long_term_memory
from app.memory.short_term import ShortTermMemory, short_term_memory
from app.schemas import ChatRequest, ChatResponse
from app.tools.attraction_rag_tool import AttractionRagTool, has_curated_rag_city


@dataclass
class AgentServices:
    attraction_rag_tool: AttractionRagTool | None = None
    weather_client: httpx.Client | None = None
    web_search_tool: Any | None = None
    openrouter_client: httpx.Client | None = None
    openweather_api_key: str | None = None
    openrouter_api_key: str | None = None
    openrouter_model: str | None = None
    serpapi_api_key: str | None = None
    serpapi_client: httpx.Client | None = None
    image_client: httpx.Client | None = None
    vector_store: Any | None = None
    use_environment: bool = True


def handle_chat(
    request: ChatRequest,
    memory: ShortTermMemory | None = None,
    user_memory: LongTermMemory | None = None,
    services: AgentServices | None = None,
) -> ChatResponse:
    return _run_chat_core(
        request,
        memory=memory or short_term_memory,
        user_memory=user_memory or long_term_memory,
        services=services or AgentServices(),
        event_emitter=None,
    )


def _run_chat_core(
    request: ChatRequest,
    memory: ShortTermMemory,
    user_memory: LongTermMemory,
    services: AgentServices,
    event_emitter: Any = None,
) -> ChatResponse:
    """Shared core for the synchronous and streaming chat paths.

    ``event_emitter`` is an optional callable invoked with event dicts as the
    flow progresses (plan steps, agent start/end, tool calls). When None, events
    are still recorded on the AgentContext but not forwarded — keeping the
    synchronous /chat path identical to before.
    """
    parsed = parse_user_request(request.message)
    rag_context_is_weak = parsed.city is not None and not has_curated_rag_city(parsed.city)
    plan = create_trip_plan(parsed, rag_context_is_weak=rag_context_is_weak)

    # Delegate to the supervisor-coordinated multi-agent system. The routing
    # mirrors the original branching exactly (clarification / follow-up /
    # fresh itinerary / destination suggestions), so the synchronous /chat
    # behavior is unchanged.
    ctx = AgentContext(
        parsed=parsed,
        plan=plan,
        services=services,
        memory=memory,
        user_memory=user_memory,
        user_id=request.user_id,
        event_emitter=event_emitter,
    )
    # Surface plan steps as progressive events (streaming UI shows them first).
    for step in plan.plan:
        ctx.emit({"type": "plan", "step": step})
    response = Supervisor().run(ctx)

    memory.add_message(request.user_id, "user", request.message)
    memory.add_message(request.user_id, "assistant", response.message)

    return response
