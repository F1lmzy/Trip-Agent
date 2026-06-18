import json
import queue
import threading
from typing import Any

import httpx

from app.agent.openrouter_client import call_openrouter
from app.agent.parser import ParsedRequest
from app.agent.planner import PlanningResult
from app.config import get_settings
from app.schemas import ChatResponse


def generate_itinerary_response(
    parsed: ParsedRequest,
    plan: PlanningResult,
    tool_outputs: dict[str, Any],
    memory_used: list[str],
    api_key: str | None = None,
    model: str | None = None,
    client: httpx.Client | None = None,
) -> ChatResponse:
    settings = get_settings()
    resolved_model = model or settings.openrouter_model
    messages = build_itinerary_messages(parsed, plan, tool_outputs, memory_used)
    llm_result = _call_openrouter_with_deadline(
        messages=messages,
        api_key=api_key,
        model=resolved_model,
        client=client,
        timeout_seconds=settings.openrouter_timeout_seconds,
    )

    itinerary = _fallback_itinerary(parsed, tool_outputs)
    rag_trace = _extract_rag_trace(tool_outputs)

    if llm_result["status"] == "ok" and llm_result.get("content"):
        message = llm_result["content"]
    else:
        message = _fallback_message(parsed, llm_result["status"])

    return ChatResponse(
        message=message,
        itinerary=itinerary,
        memory_used=memory_used,
        tools_used=plan.selected_tools,
        plan=plan.plan,
        rag_trace=rag_trace,
        needs_clarification=False,
        clarifying_question=None,
    )


def _call_openrouter_with_deadline(
    messages: list[dict[str, str]],
    api_key: str | None,
    model: str | None,
    client: httpx.Client | None,
    timeout_seconds: float = 45.0,
) -> dict[str, Any]:
    result_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        result_queue.put(call_openrouter(messages, api_key=api_key, model=model, client=client))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        return result_queue.get(timeout=timeout_seconds)
    except queue.Empty:
        return {
            "status": "fallback_openrouter_timeout",
            "source": "fallback",
            "model": model,
            "content": None,
            "message": "OpenRouter timed out while generating a response.",
        }


def build_itinerary_messages(
    parsed: ParsedRequest,
    plan: PlanningResult,
    tool_outputs: dict[str, Any],
    memory_used: list[str],
) -> list[dict[str, str]]:
    system_message = (
        "You are a travel planning assistant. Generate a personalized itinerary using only the provided context. "
        "Return a concise response with a structured itinerary organized by day, morning, afternoon, and evening. "
        "Use weather, budget, RAG attraction context, memory, and web search context when available. "
        "Mention hotel suggestions only when hotel output is present. "
        "If context is missing, make safe general recommendations and state assumptions briefly."
    )
    context = {
        "parsed_request": parsed.model_dump(),
        "plan_steps": plan.plan,
        "selected_tools": plan.selected_tools,
        "assumptions": plan.assumptions,
        "memory_used": memory_used,
        "tool_outputs": tool_outputs,
        "rag_trace": _extract_rag_trace(tool_outputs),
        "required_format": {
            f"day_{day}": ["morning", "afternoon", "evening"] for day in range(1, parsed.duration_days + 1)
        },
    }
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": json.dumps(context, indent=2, sort_keys=True)},
    ]


def _fallback_itinerary(parsed: ParsedRequest, tool_outputs: dict[str, Any]) -> dict[str, Any]:
    city = parsed.city or "your destination"
    duration_days = max(1, parsed.duration_days)
    attractions = _attraction_names(tool_outputs) or _web_search_place_suggestions(tool_outputs)
    if not attractions:
        attractions = _generic_activity_pool(parsed)

    hotel_note = None
    hotel_results = tool_outputs.get("hotel_tool", {}).get("results", [])
    if hotel_results:
        hotel_note = f"Consider staying near {hotel_results[0].get('area', city)} at {hotel_results[0].get('name')}."

    weather_note = _weather_note(tool_outputs)
    budget_note = _budget_note(tool_outputs)
    web_note = _web_search_note(tool_outputs)

    itinerary: dict[str, Any] = {
        "city": city,
        "duration_days": duration_days,
        "status": "generated_with_fallback_template",
        "notes": [note for note in [weather_note, budget_note, hotel_note, web_note] if note],
    }

    for day in range(1, duration_days + 1):
        first = attractions[(day - 1) * 3 % len(attractions)]
        second = attractions[((day - 1) * 3 + 1) % len(attractions)]
        third = attractions[((day - 1) * 3 + 2) % len(attractions)]
        itinerary[f"day_{day}"] = {
            "morning": f"Start day {day} in {city} with {first}.",
            "afternoon": f"Continue with {second}, keeping travel time clustered by neighborhood.",
            "evening": f"Finish with {third} or a nearby dinner area that matches your budget.",
        }

    return itinerary


def _fallback_message(parsed: ParsedRequest, status: str) -> str:
    city = parsed.city or "your destination"
    return (
        f"I generated a usable {parsed.duration_days}-day itinerary for {city} with a deterministic fallback "
        f"because OpenRouter returned {status}."
    )


def _attraction_names(tool_outputs: dict[str, Any]) -> list[str]:
    results = tool_outputs.get("attraction_rag_tool", {}).get("results", [])
    return [str(item.get("name")) for item in results if item.get("name")]


def _web_search_place_suggestions(tool_outputs: dict[str, Any]) -> list[str]:
    results = tool_outputs.get("web_search_tool", {}).get("results", [])
    suggestions: list[str] = []
    blocked_terms = ["itinerary", "day trip", "days in", "packages", "all inclusive", "last minute"]
    for result in results:
        title = str(result.get("title", "")).strip()
        description = str(result.get("description", "")).strip()
        if title and not any(term in title.lower() for term in blocked_terms):
            suggestions.append(title)
        for phrase in _extract_place_like_phrases(description):
            if phrase not in suggestions:
                suggestions.append(phrase)
    return suggestions[:8]


def _extract_place_like_phrases(text: str) -> list[str]:
    import re

    candidates = re.findall(r"\b(?:[A-Z][A-Za-z'’-]+|of|the|and|&)(?:\s+(?:[A-Z][A-Za-z'’-]+|of|the|and|&)){1,5}\b", text)
    stop_starts = {"The Best", "How To", "When To", "Where To", "Things To"}
    phrases: list[str] = []
    for candidate in candidates:
        cleaned = " ".join(candidate.split()).strip(" .,:;-/")
        for prefix in ["Visit ", "Explore ", "See "]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :]
        if any(cleaned.startswith(stop) for stop in stop_starts):
            continue
        if len(cleaned) < 5 or cleaned.lower() in {"travel guide", "day itinerary"}:
            continue
        phrases.append(cleaned)
    return phrases


def _generic_activity_pool(parsed: ParsedRequest) -> list[str]:
    activities = [
        "a central neighborhood walk to understand the city layout",
        "a local food stop or market area",
        "a cultural landmark or heritage district",
        "a museum, gallery, or indoor attraction",
        "a scenic viewpoint, waterfront, park, or relaxed photo stop",
        "a low-stress evening food area close to your accommodation",
    ]
    interest_options = {
        "food": "a food-focused neighborhood, market, or well-reviewed local dining area",
        "museums": "a major museum or gallery cluster",
        "nature": "a park, garden, waterfront, or scenic outdoor area",
        "culture": "a temple, heritage street, old town, or cultural district",
        "history": "a historic landmark and nearby walking route",
        "shopping": "a market, craft district, or shopping street",
        "nightlife": "an evening district with safe transport back to your stay",
        "photography": "a viewpoint, waterfront, or photogenic neighborhood",
    }
    for interest in reversed(parsed.interests):
        option = interest_options.get(interest)
        if option:
            activities.insert(0, option)
    return activities


def _web_search_note(tool_outputs: dict[str, Any]) -> str | None:
    web_search = tool_outputs.get("web_search_tool", {})
    results = web_search.get("results", [])
    if not results:
        return None
    titles = [str(item.get("title")) for item in results[:3] if item.get("title")]
    if not titles:
        return None
    return "Web search context consulted: " + "; ".join(titles) + "."


def _weather_note(tool_outputs: dict[str, Any]) -> str | None:
    weather = tool_outputs.get("weather_tool", {})
    forecast = weather.get("forecast", [])
    if not forecast:
        return None
    first_day = forecast[0]
    summary = first_day.get("summary")
    suitability = first_day.get("outdoor_suitability")
    if weather.get("source") == "fallback":
        return f"Weather note: {summary} Outdoor suitability is {suitability}."
    return f"Weather note: {summary} with outdoor suitability marked {suitability}."


def _budget_note(tool_outputs: dict[str, Any]) -> str | None:
    budget = tool_outputs.get("budget_tool", {})
    if not budget:
        return None
    return f"Budget note: using {budget.get('budget_level')} budget guidance."


def _extract_rag_trace(tool_outputs: dict[str, Any]) -> dict[str, Any]:
    return tool_outputs.get("attraction_rag_tool", {}).get("rag_trace", {})
