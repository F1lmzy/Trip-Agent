from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from app.agent.parser import ParsedRequest
from app.agent.planner import PlanningResult
from app.agent.service_utils import service_value
from app.agent.tool_names import ToolName
from app.tools.attraction_rag_tool import AttractionRagTool
from app.tools.budget_tool import run_budget_tool
from app.tools.flight_tool import run_flight_tool
from app.tools.hotel_tool import run_hotel_tool
from app.tools.serpapi_flight_tool import run_serpapi_flight_tool
from app.tools.serpapi_hotel_tool import run_serpapi_hotel_tool
from app.tools.weather_tool import run_weather_tool
from app.tools.web_search_tool import run_web_search_tool
from app.tools.wikimedia_image_tool import resolve_place_image

if TYPE_CHECKING:
    from app.agent.orchestrator import AgentServices


def execute_tools(parsed: ParsedRequest, plan: PlanningResult, services: AgentServices) -> dict[str, Any]:
    if parsed.city is None:
        return {}

    tool_outputs: dict[str, Any] = {}
    selected_tools = set(plan.selected_tools)

    if ToolName.ATTRACTION_RAG in selected_tools:
        rag_tool = services.attraction_rag_tool or AttractionRagTool()
        rag_tool.seed_if_needed()
        services.attraction_rag_tool = rag_tool
        tool_outputs[ToolName.ATTRACTION_RAG] = rag_tool.run(
            city=parsed.city,
            interests=parsed.interests,
            http_client=services.weather_client,
        )

    if ToolName.WEATHER in selected_tools:
        tool_outputs[ToolName.WEATHER] = run_weather_tool(
            parsed.city,
            api_key=service_value(services, "openweather_api_key", "openweather_api_key"),
            client=services.weather_client,
        )

    if ToolName.BUDGET in selected_tools:
        tool_outputs[ToolName.BUDGET] = run_budget_tool(parsed.budget)

    if ToolName.WEB_SEARCH in selected_tools:
        tool_outputs[ToolName.WEB_SEARCH] = run_web_search_tool(
            parsed.city,
            query_intent=fresh_travel_search_intent(parsed),
            search_tool=services.web_search_tool,
        )

    if ToolName.HOTEL in selected_tools:
        tool_outputs[ToolName.HOTEL] = run_hotel(parsed, services)

    if ToolName.FLIGHT in selected_tools and parsed.origin_city:
        tool_outputs[ToolName.FLIGHT] = run_flight(parsed, services)

    attach_images(tool_outputs, services)

    return tool_outputs


def run_hotel(parsed: ParsedRequest, services: AgentServices) -> dict[str, Any]:
    """Run the SerpAPI hotel tool, falling back to the mock tool.

    SerpAPI is only called when a key is available (so the agent never breaks
    and the SerpAPI request quota is only consumed on real, key-backed
    requests). On any no_results/error, fall back to the mock tool so the
    itinerary still has hotel content.
    """
    key = service_value(services, "serpapi_api_key", "serpapi_api_key")
    check_in = resolve_departure_date(parsed)
    check_out = resolve_return_date(parsed, parsed.duration_days)
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


def run_flight(parsed: ParsedRequest, services: AgentServices) -> dict[str, Any]:
    """Run real SerpAPI flights when configured, else local mock flights.

    If a SerpAPI key is configured, return the SerpAPI result even for
    no-results/error responses. Falling back to mock data after a live SerpAPI
    failure makes the API look successful with fake flights.
    """
    key = service_value(services, "serpapi_api_key", "serpapi_api_key")
    departure_date = resolve_departure_date(parsed)
    # One-way requests skip the return date entirely; both flight tools treat
    # return_date=None as a one-way search (no return_flights).
    return_date = resolve_return_date(parsed, parsed.duration_days) if parsed.trip_type != "one_way" else None
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
        return result
    return run_flight_tool(
        from_location=parsed.origin_city,
        to_location=parsed.city,
        departure_date=departure_date,
        return_date=return_date,
        budget=parsed.budget,
    )


def attach_images(tool_outputs: dict[str, Any], services: AgentServices) -> None:
    """Attach Wikimedia Commons image_url to attraction and hotel results.

    Best-effort and never blocking: a None image_url means "no image". Uses
    the shared ``image_client`` when provided (testability), else a transient
    client. Attraction results come from the RAG tool (a list under
    ``results``); hotel results come from the hotel tool (``results`` list).
    """
    attraction = tool_outputs.get(ToolName.ATTRACTION_RAG)
    if isinstance(attraction, dict) and isinstance(attraction.get("results"), list):
        for item in attraction["results"]:
            if isinstance(item, dict) and "image_url" not in item:
                name = item.get("name")
                city = (
                    item.get("city")
                    if isinstance(item.get("city"), str)
                    else (item.get("metadata") or {}).get("city")
                    if isinstance(item.get("metadata"), dict)
                    else None
                )
                if name:
                    item["image_url"] = resolve_place_image(name, city=city, client=services.image_client)

    hotel = tool_outputs.get(ToolName.HOTEL)
    if isinstance(hotel, dict) and isinstance(hotel.get("results"), list):
        for item in hotel["results"]:
            if isinstance(item, dict) and "image_url" not in item:
                name = item.get("name")
                city = (
                    item.get("city")
                    if isinstance(item.get("city"), str)
                    else (item.get("metadata") or {}).get("city")
                    if isinstance(item.get("metadata"), dict)
                    else None
                )
                if name:
                    item["image_url"] = resolve_place_image(name, city=city, client=services.image_client)


def fresh_travel_search_intent(parsed: ParsedRequest) -> str:
    interests = ", ".join(parsed.interests) if parsed.interests else "highlights, food, markets, neighborhoods"
    return f"top attractions, specific places, {interests}, {parsed.duration_days} day itinerary"


def resolve_departure_date(parsed: ParsedRequest) -> str:
    """Resolve an ISO departure date.

    Priority: an explicit departure_date parsed from the message (e.g.
    'from June 21 to June 25'), then the first date in parsed.dates, else
    tomorrow. Google Flights can reject same-day searches depending on timezone,
    so undated flight requests use the next day rather than today.
    """
    from datetime import date, datetime, timedelta

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
    return (date.today() + timedelta(days=1)).isoformat()


def resolve_return_date(parsed: ParsedRequest, duration_days: int) -> str | None:
    """Resolve an ISO return date.

    Priority: an explicit return_date parsed from the message, then departure
    plus duration_days, else None.
    """
    from datetime import datetime, timedelta

    if parsed.return_date:
        return parsed.return_date
    departure = resolve_departure_date(parsed)
    try:
        departure_date = datetime.strptime(departure, "%Y-%m-%d").date()
    except ValueError:
        return None
    return (departure_date + timedelta(days=max(1, duration_days))).isoformat()
