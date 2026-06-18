from dataclasses import dataclass
from typing import Any

import httpx

from app.agent.parser import ParsedRequest, parse_user_request
from app.agent.planner import PlanningResult, create_trip_plan
from app.agent.response_generator import generate_itinerary_response
from app.config import get_settings
from app.memory.long_term import LongTermMemory, long_term_memory
from app.memory.short_term import ShortTermMemory, short_term_memory
from app.schemas import ChatRequest, ChatResponse
from app.tools.attraction_rag_tool import AttractionRagTool, has_curated_rag_city
from app.tools.budget_tool import run_budget_tool
from app.tools.hotel_tool import run_hotel_tool
from app.tools.weather_tool import run_weather_tool
from app.tools.web_search_tool import run_web_search_tool


@dataclass
class AgentServices:
    attraction_rag_tool: AttractionRagTool | None = None
    weather_client: httpx.Client | None = None
    web_search_tool: Any | None = None
    openrouter_client: httpx.Client | None = None
    openweather_api_key: str | None = None
    openrouter_api_key: str | None = None
    openrouter_model: str | None = None
    use_environment: bool = True
    rag_seeded: bool = False


def handle_chat(
    request: ChatRequest,
    memory: ShortTermMemory | None = None,
    user_memory: LongTermMemory | None = None,
    services: AgentServices | None = None,
) -> ChatResponse:
    memory = memory or short_term_memory
    user_memory = user_memory or long_term_memory
    services = services or AgentServices()

    parsed = parse_user_request(request.message)
    rag_context_is_weak = parsed.city is not None and not has_curated_rag_city(parsed.city)
    plan = create_trip_plan(parsed, rag_context_is_weak=rag_context_is_weak)
    has_prior_context = memory.has_history(request.user_id)

    if plan.needs_clarification:
        response = _build_clarification_response(plan)
    elif parsed.is_follow_up:
        memory_used = user_memory.search_preferences(request.user_id, request.message)
        response = _build_follow_up_response(parsed, plan, has_prior_context, memory_used)
    else:
        memory_used = user_memory.search_preferences(request.user_id, request.message)
        tool_outputs = _execute_tools(parsed, plan, services)
        _save_stable_preferences(request.user_id, parsed, user_memory)
        response = generate_itinerary_response(
            parsed=parsed,
            plan=plan,
            tool_outputs=tool_outputs,
            memory_used=memory_used,
            api_key=_service_value(services, "openrouter_api_key", "openrouter_api_key"),
            model=_service_value(services, "openrouter_model", "openrouter_model"),
            client=services.openrouter_client,
        )

    memory.add_message(request.user_id, "user", request.message)
    memory.add_message(request.user_id, "assistant", response.message)

    return response


def _execute_tools(parsed: ParsedRequest, plan: PlanningResult, services: AgentServices) -> dict[str, Any]:
    if parsed.city is None:
        return {}

    tool_outputs: dict[str, Any] = {}
    selected_tools = set(plan.selected_tools)

    if "attraction_rag_tool" in selected_tools:
        rag_tool = services.attraction_rag_tool or AttractionRagTool()
        if not services.rag_seeded:
            rag_tool.seed()
            services.rag_seeded = True
        services.attraction_rag_tool = rag_tool
        tool_outputs["attraction_rag_tool"] = rag_tool.run(city=parsed.city, interests=parsed.interests)

    if "weather_tool" in selected_tools:
        tool_outputs["weather_tool"] = run_weather_tool(
            parsed.city,
            api_key=_service_value(services, "openweather_api_key", "openweather_api_key"),
            client=services.weather_client,
        )

    if "budget_tool" in selected_tools:
        tool_outputs["budget_tool"] = run_budget_tool(parsed.budget)

    if "web_search_tool" in selected_tools:
        tool_outputs["web_search_tool"] = run_web_search_tool(
            parsed.city,
            query_intent=_fresh_travel_search_intent(parsed),
            search_tool=services.web_search_tool,
        )

    if "hotel_tool" in selected_tools:
        tool_outputs["hotel_tool"] = run_hotel_tool(parsed.city, parsed.budget)

    return tool_outputs


def _fresh_travel_search_intent(parsed: ParsedRequest) -> str:
    interests = ", ".join(parsed.interests) if parsed.interests else "highlights, food, markets, neighborhoods"
    return f"top attractions, specific places, {interests}, {parsed.duration_days} day itinerary"


def _build_clarification_response(plan: PlanningResult) -> ChatResponse:
    question = plan.clarifying_question or "Could you share a few more details for the trip?"
    return ChatResponse(
        message=question,
        itinerary={},
        memory_used=[],
        tools_used=[],
        plan=plan.plan,
        needs_clarification=True,
        clarifying_question=question,
    )


def _build_follow_up_response(
    parsed: ParsedRequest,
    plan: PlanningResult,
    has_prior_context: bool,
    memory_used: list[str],
) -> ChatResponse:
    if not has_prior_context:
        question = "What trip should I update? Please share the destination or original itinerary request."
        return ChatResponse(
            message=question,
            itinerary={},
            memory_used=[],
            tools_used=[],
            plan=["Ask for original trip context before applying follow-up request"],
            needs_clarification=True,
            clarifying_question=question,
        )

    return ChatResponse(
        message="I understood this as a follow-up to your previous trip request and planned the update.",
        itinerary={
            "status": "follow_up_planned_not_generated_yet",
            "follow_up_intent": parsed.follow_up_intent,
        },
        memory_used=["Recent conversation history", *memory_used],
        tools_used=plan.selected_tools,
        plan=plan.plan,
        needs_clarification=False,
        clarifying_question=None,
    )


def _save_stable_preferences(user_id: str, parsed: ParsedRequest, user_memory: LongTermMemory) -> None:
    existing = set(user_memory.get_preferences(user_id))
    for preference in _stable_preferences_from(parsed):
        if preference not in existing:
            user_memory.add_preference(user_id, preference)
            existing.add(preference)


def _stable_preferences_from(parsed: ParsedRequest) -> list[str]:
    preferences: list[str] = []
    preferences.extend(f"Interest preference: {interest}" for interest in parsed.interests)
    preferences.extend(f"Dietary need: {need}" for need in parsed.dietary_needs)
    preferences.extend(f"Constraint: {constraint}" for constraint in parsed.constraints)

    if parsed.budget:
        preferences.append(f"Budget preference: {parsed.budget}")
    if parsed.travel_style:
        preferences.append(f"Travel style: {parsed.travel_style}")

    return preferences


def _service_value(services: AgentServices, field_name: str, settings_name: str) -> Any:
    value = getattr(services, field_name)
    if value is not None:
        return value
    if not services.use_environment:
        return "" if field_name.endswith("_api_key") else None
    return getattr(get_settings(), settings_name)
