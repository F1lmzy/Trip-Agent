from dataclasses import dataclass
from typing import Any
import re

import httpx

from app.agent.parser import ParsedRequest, parse_user_request
from app.agent.planner import PlanningResult, create_trip_plan
from app.agent.response_generator import generate_itinerary_response
from app.agents import AgentContext, Supervisor
from app.config import get_settings
from app.memory.long_term import LongTermMemory, long_term_memory
from app.memory.short_term import ShortTermMemory, short_term_memory
from app.schemas import ChatRequest, ChatResponse
from app.tools.attraction_rag_tool import AttractionRagTool, has_curated_rag_city
from app.tools.budget_tool import run_budget_tool
from app.tools.flight_tool import run_flight_tool
from app.tools.hotel_tool import run_hotel_tool
from app.tools.serpapi_flight_tool import run_serpapi_flight_tool
from app.tools.serpapi_hotel_tool import run_serpapi_hotel_tool
from app.tools.weather_tool import run_weather_tool
from app.tools.web_search_tool import run_web_search_tool
from app.tools.wikimedia_image_tool import resolve_place_image


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
    rag_seeded: bool = False


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
        tool_outputs["attraction_rag_tool"] = rag_tool.run(
            city=parsed.city,
            interests=parsed.interests,
            http_client=services.weather_client,
        )

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
        tool_outputs["hotel_tool"] = _run_hotel(parsed, services)

    if "flight_tool" in selected_tools and parsed.origin_city:
        tool_outputs["flight_tool"] = _run_flight(parsed, services)

    _attach_images(tool_outputs, services)

    return tool_outputs


def _run_hotel(parsed: ParsedRequest, services: AgentServices) -> dict[str, Any]:
    """Run the SerpAPI hotel tool, falling back to the mock tool.

    SerpAPI is only called when a key is available (so the agent never breaks
    and the SerpAPI request quota is only consumed on real, key-backed
    requests). On any no_results/error, fall back to the mock tool so the
    itinerary still has hotel content.
    """
    key = _service_value(services, "serpapi_api_key", "serpapi_api_key")
    check_in = _resolve_departure_date(parsed)
    check_out = _resolve_return_date(parsed, parsed.duration_days)
    if key:
        result = run_serpapi_hotel_tool(
            city=parsed.city,
            check_in_date=check_in,
            check_out_date=check_out or check_in,
            budget=parsed.budget,
            api_key=key,
            client=services.serpapi_client,
        )
        if result.get("status") == "ok":
            return result
    return run_hotel_tool(parsed.city, parsed.budget)


def _run_flight(parsed: ParsedRequest, services: AgentServices) -> dict[str, Any]:
    """Run the SerpAPI flight tool, falling back to the mock tool.

    Same key-gating and fallback semantics as ``_run_hotel``.
    """
    key = _service_value(services, "serpapi_api_key", "serpapi_api_key")
    departure_date = _resolve_departure_date(parsed)
    # One-way requests skip the return date entirely; both flight tools treat
    # return_date=None as a one-way search (no return_flights).
    return_date = _resolve_return_date(parsed, parsed.duration_days) if parsed.trip_type != "one_way" else None
    if key:
        result = run_serpapi_flight_tool(
            from_location=parsed.origin_city,
            to_location=parsed.city,
            departure_date=departure_date,
            return_date=return_date,
            budget=parsed.budget,
            api_key=key,
            client=services.serpapi_client,
        )
        if result.get("status") == "ok":
            return result
    return run_flight_tool(
        from_location=parsed.origin_city,
        to_location=parsed.city,
        departure_date=departure_date,
        return_date=return_date,
        budget=parsed.budget,
    )


def _attach_images(tool_outputs: dict[str, Any], services: AgentServices) -> None:
    """Attach Wikimedia Commons image_url to attraction and hotel results.

    Best-effort and never blocking: a None image_url means "no image". Uses
    the shared ``image_client`` when provided (testability), else a transient
    client. Attraction results come from the RAG tool (a list under
    ``results``); hotel results come from the hotel tool (``results`` list).
    """
    attraction = tool_outputs.get("attraction_rag_tool")
    if isinstance(attraction, dict) and isinstance(attraction.get("results"), list):
        for item in attraction["results"]:
            if isinstance(item, dict) and "image_url" not in item:
                name = item.get("name")
                if name:
                    item["image_url"] = resolve_place_image(name, client=services.image_client)

    hotel = tool_outputs.get("hotel_tool")
    if isinstance(hotel, dict) and isinstance(hotel.get("results"), list):
        for item in hotel["results"]:
            if isinstance(item, dict) and "image_url" not in item:
                name = item.get("name")
                if name:
                    item["image_url"] = resolve_place_image(name, client=services.image_client)



def _fresh_travel_search_intent(parsed: ParsedRequest) -> str:
    interests = ", ".join(parsed.interests) if parsed.interests else "highlights, food, markets, neighborhoods"
    return f"top attractions, specific places, {interests}, {parsed.duration_days} day itinerary"


def _resolve_departure_date(parsed: ParsedRequest) -> str:
    """Resolve an ISO departure date.

    Priority: an explicit departure_date parsed from the message (e.g.
    'from June 21 to June 25'), then the first date in parsed.dates, else today.
    """
    from datetime import date, datetime

    if parsed.departure_date:
        return parsed.departure_date
    if parsed.dates:
        match = re.match(r"([A-Za-z]+\s+\d{1,2})", parsed.dates)
        if match:
            try:
                parsed_date = datetime.strptime(f"{match.group(1)} {date.today().year}", "%B %d %Y").date()
                return parsed_date.isoformat()
            except ValueError:
                pass
    return date.today().isoformat()


def _resolve_return_date(parsed: ParsedRequest, duration_days: int) -> str | None:
    """Resolve an ISO return date.

    Priority: an explicit return_date parsed from the message, then departure
    plus duration_days, else None.
    """
    from datetime import date, datetime, timedelta

    if parsed.return_date:
        return parsed.return_date
    departure = _resolve_departure_date(parsed)
    try:
        departure_date = datetime.strptime(departure, "%Y-%m-%d").date()
    except ValueError:
        return None
    return (departure_date + timedelta(days=max(1, duration_days))).isoformat()


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
