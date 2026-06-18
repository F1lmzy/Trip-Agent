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
    timeout_seconds: float = 12.0,
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
            "day_1": ["morning", "afternoon", "evening"],
            "day_2": ["morning", "afternoon", "evening"],
        },
    }
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": json.dumps(context, indent=2, sort_keys=True)},
    ]


def _fallback_itinerary(parsed: ParsedRequest, tool_outputs: dict[str, Any]) -> dict[str, Any]:
    city = parsed.city or "your destination"
    attractions = _attraction_names(tool_outputs) or _web_search_titles(tool_outputs)
    if not attractions:
        attractions = ["a central neighborhood walk", "a local food stop", "a museum or cultural highlight"]

    hotel_note = None
    hotel_results = tool_outputs.get("hotel_tool", {}).get("results", [])
    if hotel_results:
        hotel_note = f"Consider staying near {hotel_results[0].get('area', city)} at {hotel_results[0].get('name')}."

    weather_note = _weather_note(tool_outputs)
    budget_note = _budget_note(tool_outputs)

    return {
        "city": city,
        "duration_days": parsed.duration_days,
        "status": "generated_with_fallback_template",
        "day_1": {
            "morning": f"Start in {city} with {attractions[0]}.",
            "afternoon": f"Continue with {attractions[1] if len(attractions) > 1 else 'a nearby cultural stop'}.",
            "evening": "Choose a dinner area that matches your budget and keep transit simple.",
        },
        "day_2": {
            "morning": f"Visit {attractions[2] if len(attractions) > 2 else 'a relaxed neighborhood highlight'}.",
            "afternoon": "Add an indoor museum, market, or cafe break depending on the weather.",
            "evening": "Finish with a viewpoint, walkable food area, or low-stress local experience.",
        },
        "notes": [note for note in [weather_note, budget_note, hotel_note] if note],
    }


def _fallback_message(parsed: ParsedRequest, status: str) -> str:
    city = parsed.city or "your destination"
    return (
        f"I generated a usable {parsed.duration_days}-day itinerary for {city} with a deterministic fallback "
        f"because OpenRouter returned {status}."
    )


def _attraction_names(tool_outputs: dict[str, Any]) -> list[str]:
    results = tool_outputs.get("attraction_rag_tool", {}).get("results", [])
    return [str(item.get("name")) for item in results if item.get("name")]


def _web_search_titles(tool_outputs: dict[str, Any]) -> list[str]:
    results = tool_outputs.get("web_search_tool", {}).get("results", [])
    return [str(item.get("title")) for item in results if item.get("title")]


def _weather_note(tool_outputs: dict[str, Any]) -> str | None:
    forecast = tool_outputs.get("weather_tool", {}).get("forecast", [])
    if not forecast:
        return None
    first_day = forecast[0]
    return f"Weather note: {first_day.get('summary')} with outdoor suitability marked {first_day.get('outdoor_suitability')}."


def _budget_note(tool_outputs: dict[str, Any]) -> str | None:
    budget = tool_outputs.get("budget_tool", {})
    if not budget:
        return None
    return f"Budget note: using {budget.get('budget_level')} budget guidance."


def _extract_rag_trace(tool_outputs: dict[str, Any]) -> dict[str, Any]:
    return tool_outputs.get("attraction_rag_tool", {}).get("rag_trace", {})
