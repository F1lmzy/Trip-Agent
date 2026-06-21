import concurrent.futures
import json
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

    rag_trace = _extract_rag_trace(tool_outputs)

    content = llm_result.get("content")
    if llm_result["status"] == "ok" and _usable_llm_content(content) and _content_matches_itinerary_contract(content):
        message = content
        itinerary = _itinerary_from_llm_content(parsed, message, tool_outputs)
    else:
        fallback_status = llm_result["status"] if llm_result["status"] != "ok" else "fallback_unparseable_content"
        message = _fallback_message(parsed, fallback_status)
        itinerary = _fallback_itinerary(parsed, tool_outputs)

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

# Bounded thread pool for OpenRouter calls. Threads are reused across requests
# so a timeout does not leave an accumulating trail of daemon threads. When a
# timeout fires the worker thread returns to the pool once the HTTP call finishes
# (subject to the httpx timeout in call_openrouter, default 120 s).
_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="openrouter",
)


def _call_openrouter_with_deadline(
    messages: list[dict[str, str]],
    api_key: str | None,
    model: str | None,
    client: httpx.Client | None,
    timeout_seconds: float = 45.0,
) -> dict[str, Any]:
    future = _executor.submit(call_openrouter, messages, api_key=api_key, model=model, client=client)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        return {
            "status": "fallback_openrouter_timeout",
            "source": "fallback",
            "model": model,
            "content": None,
            "message": "OpenRouter timed out while generating a response.",
        }


def _usable_llm_content(content: Any) -> bool:
    return isinstance(content, str) and bool(content.strip()) and content.strip().lower() not in {"none", "null"}


def _content_matches_itinerary_contract(content: Any) -> bool:
    """Reject non-itinerary OpenRouter responses such as safety labels.

    Some upstream models can return short moderation/status text like
    ``User Safety: safe`` with HTTP 200. Treat that as unusable unless the
    content contains the itinerary structure we explicitly requested.
    """
    if not isinstance(content, str):
        return False

    import re

    text = content.strip()
    if re.fullmatch(r"user\s+safety\s*:\s*\w+", text, flags=re.IGNORECASE):
        return False

    has_day_heading = re.search(r"(?:\*\*)?Day\s+\d+", text, flags=re.IGNORECASE) is not None
    has_time_slot = re.search(r"\b(Morning|Afternoon|Evening)\b", text, flags=re.IGNORECASE) is not None
    has_itinerary_table = bool(_parse_markdown_table(text))
    return has_itinerary_table or (has_day_heading and has_time_slot)


def build_itinerary_messages(
    parsed: ParsedRequest,
    plan: PlanningResult,
    tool_outputs: dict[str, Any],
    memory_used: list[str],
) -> list[dict[str, str]]:
    system_message = (
        "You are a travel planning assistant. Generate a personalized itinerary using only the provided context. "
        "Return the itinerary in this exact parseable Markdown shape for every day: "
        "**Day N – YYYY-MM-DD (weather if available)**, then exactly three bullets: "
        "- **Morning**: <specific activities>, - **Afternoon**: <specific activities>, - **Evening**: <specific activities>. "
        "Use those slot labels exactly, one slot per line, and prefer ':' after each label. Never return None/null/empty content. "
        "Do not output moderation/status labels such as 'User Safety: safe'; output only the itinerary Markdown. "
        "Use weather, budget, RAG attraction context, memory, and web search context when available. "
        "When using weather, copy the provided forecast dates, summaries, temperatures, and suitability exactly; do not invent weather details. "
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
            "markdown_contract": "For each day output: **Day N – date** then '- **Morning**: ...', '- **Afternoon**: ...', '- **Evening**: ...' on separate lines.",
            "days": {
                f"day_{day}": ["morning", "afternoon", "evening"] for day in range(1, parsed.duration_days + 1)
            },
        },
    }
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": json.dumps(context, indent=2, sort_keys=True)},
    ]


def _itinerary_from_llm_content(parsed: ParsedRequest, content: str, tool_outputs: dict[str, Any]) -> dict[str, Any]:
    import re

    city = parsed.city or "your destination"
    duration_days = max(1, parsed.duration_days)
    itinerary: dict[str, Any] = {
        "city": city,
        "duration_days": duration_days,
        "status": "generated_with_openrouter",
        "notes": [note for note in [_weather_note(tool_outputs), _budget_note(tool_outputs), _web_search_note(tool_outputs), _flight_note(tool_outputs)] if note],
    }

    fallback = _fallback_itinerary(parsed, tool_outputs)
    table_rows = _parse_markdown_table(content)
    for day in range(1, duration_days + 1):
        table_slots = _table_slots_for_day(table_rows, day)
        if table_slots:
            itinerary[f"day_{day}"] = {
                slot: table_slots.get(slot) or fallback[f"day_{day}"][slot]
                for slot in ("morning", "afternoon", "evening")
            }
            continue

        post_itinerary_heading = _post_itinerary_heading_pattern()
        pattern = rf"(?:\*\*)?Day\s+{day}(?:\*\*)?(.+?)(?=(?:\*\*)?Day\s+{day + 1}(?:\*\*)?|{post_itinerary_heading}|\Z)"
        match = re.search(pattern, content, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if not match:
            itinerary[f"day_{day}"] = fallback[f"day_{day}"]
            continue

        section = match.group(1)
        itinerary[f"day_{day}"] = {
            "morning": _extract_time_slot(section, "morning") or fallback[f"day_{day}"]["morning"],
            "afternoon": _extract_time_slot(section, "afternoon") or fallback[f"day_{day}"]["afternoon"],
            "evening": _extract_time_slot(section, "evening") or fallback[f"day_{day}"]["evening"],
        }

    return _enrich_itinerary(itinerary, tool_outputs)


def _parse_markdown_table(content: str) -> list[dict[str, str]]:
    """Parse a markdown table into a list of row dicts keyed by header.

    Returns an empty list when no table is present. Handles pipe-delimited
    rows with a separator row (---). Column headers are matched to
    morning/afternoon/evening case-insensitively.
    """
    import re

    lines = content.splitlines()
    table_lines = [line.strip() for line in lines if line.strip().startswith("|")]
    if len(table_lines) < 2:
        return []

    header = _split_table_row(table_lines[0])
    if len(table_lines) > 1 and set(re.sub(r"[^:-]", "", table_lines[1])) <= {"-", ":"}:
        data_lines = table_lines[2:]
    else:
        data_lines = table_lines[1:]

    rows: list[dict[str, str]] = []
    for line in data_lines:
        cells = _split_table_row(line)
        if not cells:
            continue
        row: dict[str, str] = {}
        for index, cell in enumerate(cells):
            if index < len(header):
                row[header[index].lower()] = cell
        rows.append(row)
    return rows


def _split_table_row(line: str) -> list[str]:
    """Split a markdown table row into trimmed cell values."""
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def _table_slots_for_day(table_rows: list[dict[str, str]], day: int) -> dict[str, str] | None:
    """Extract morning/afternoon/evening slots for a given day from table rows.

    Matches a row whose first column mentions the day number. Returns None
    when no matching row is found.
    """
    import re

    slot_keys = {"morning", "afternoon", "evening"}
    for row in table_rows:
        first_cell = next(iter(row.values()), "")
        if re.search(rf"\bday\s*{day}\b", first_cell, flags=re.IGNORECASE):
            slots: dict[str, str] = {}
            for key, value in row.items():
                normalized = key.lower()
                for slot in slot_keys:
                    if slot in normalized and value:
                        cleaned = " ".join(value.replace("**", "").split()).strip(" -.;|")
                        if cleaned:
                            slots[slot] = cleaned
            if slots:
                return slots
    return None


def _post_itinerary_heading_pattern() -> str:
    return r"^\s*(?:[-•]\s*)?\*{0,2}(?:assumptions?|notes?|considerations?|caveats?|important notes?|booking notes?)\*{0,2}\s*:"


def _extract_time_slot(section: str, slot: str) -> str | None:
    import re

    # Common separators a model might use after the bold label.
    sep = r"(?:\s*[\-–—:]+\s*)"
    post_itinerary_heading = _post_itinerary_heading_pattern()

    # Format 1: label with colon — "**Morning**: content"
    label_colon = rf"\*{{0,2}}\s*{slot}\s*\*{{0,2}}\s*:"
    next_label = r"\*{0,2}\s*(?:morning|afternoon|evening|budget|memory|note)\s*\*{0,2}\s*(?::|\s*[\-–—]+)\s*"
    pattern_colon = rf"{label_colon}\s*(.+?)(?=(?:{next_label})|\n\s*\*{{0,2}}Day|{post_itinerary_heading}|$)"
    match = re.search(pattern_colon, section, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    if match:
        value = " ".join(match.group(1).replace("**", "").split()).strip(" -.;")
        return value or None

    # Format 2: label as standalone header without colon — "**Morning**\n- content"
    label_header = rf"\*{{0,2}}\s*{slot}\s*\*{{0,2}}\s*\n"
    next_label_header = r"\*{0,2}\s*(?:morning|afternoon|evening|budget|memory|note|day)\s*\*{0,2}\s*(?:\n|:|\s*[\-–—:]|$)"
    pattern_header = rf"{label_header}(.+?)(?=(?:{next_label_header})|{post_itinerary_heading}|$)"
    match = re.search(pattern_header, section, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    if match:
        raw = match.group(1).replace("**", "")
        raw = re.sub(r"^\s*[-•]\s*", "", raw, flags=re.MULTILINE)
        value = " ".join(raw.split()).strip(" -.;|")
        return value or None

    # Format 3: bullet + bold label + separator — "- **Morning** – content" or "- **Morning**: content"
    bullet_label = rf"^\s*[-•]\s+\*{{0,2}}{slot}\s*\*{{0,2}}{sep}"
    next_slot_or_day = rf"(?:^\s*[-•]\s+\*{{0,2}}(?:morning|afternoon|evening|budget|memory|note)\s*\*{{0,2}}{sep}|\*{{0,2}}Day\b|$)"
    pattern_bullet = rf"{bullet_label}(.+?)(?={next_slot_or_day}|{post_itinerary_heading}|\n\n|\n-\s*\*|$)"
    match = re.search(pattern_bullet, section, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    if match:
        raw = match.group(1).replace("**", "")
        raw = re.sub(r"^\s*[-•]\s*", "", raw, flags=re.MULTILINE)
        value = " ".join(raw.split()).strip(" -.;|")
        return value or None

    return None


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
    flight_note = _flight_note(tool_outputs)

    itinerary: dict[str, Any] = {
        "city": city,
        "duration_days": duration_days,
        "status": "generated_with_fallback_template",
        "notes": [note for note in [weather_note, budget_note, hotel_note, web_note, flight_note] if note],
    }

    morning_templates = [
        "Begin day {day} in {city} at {place}.",
        "Start with {place} in {city}.",
        "Spend the morning around {place}.",
    ]
    afternoon_templates = [
        "Pair that with {place} for the afternoon.",
        "Head next to {place} for a different side of {city}.",
        "Use the afternoon for {place}.",
    ]

    for day in range(1, duration_days + 1):
        first = attractions[(day - 1) * 3 % len(attractions)]
        second = attractions[((day - 1) * 3 + 1) % len(attractions)]
        third = attractions[((day - 1) * 3 + 2) % len(attractions)]
        itinerary[f"day_{day}"] = {
            "morning": morning_templates[(day - 1) % len(morning_templates)].format(day=day, city=city, place=first),
            "afternoon": afternoon_templates[(day - 1) % len(afternoon_templates)].format(city=city, place=second),
            "evening": _fallback_evening_slot(parsed, city, third),
        }


    return _enrich_itinerary(itinerary, tool_outputs)


def _fallback_evening_slot(parsed: ParsedRequest, city: str, place: str) -> str:
    if "nightlife" in parsed.interests:
        return f"Use the evening for local bars near {place}, choosing well-reviewed spots that fit your budget."
    if "food" in parsed.interests:
        return f"End near {place} with a local dinner that fits your budget."
    return f"Finish around {place} or a nearby dinner area that matches your budget."


def _fallback_message(parsed: ParsedRequest, status: str) -> str:
    city = parsed.city or "your destination"
    return (
        f"I generated a usable {parsed.duration_days}-day itinerary for {city} with a deterministic fallback "
        f"because OpenRouter returned {status}."
    )


def _attraction_names(tool_outputs: dict[str, Any]) -> list[str]:
    """Extract clean attraction names from RAG results.

    Skips externally-ingested results whose "name" is a raw text chunk
    (e.g. "Stockholm is Sweden's capital...") rather than a real attraction
    name. Those results are still useful for the LLM prompt context, but not
    for the fallback itinerary template.
    """
    results = tool_outputs.get("attraction_rag_tool", {}).get("results", [])
    names: list[str] = []
    for item in results:
        name = item.get("name")
        if not name:
            continue
        # External ingestion sets name to a truncated text chunk — skip those.
        description = item.get("description", "")
        if name in description[: len(name)] and len(name) > 40:
            continue
        names.append(str(name))
    return names


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
    if weather.get("source") == "fallback":
        first_day = forecast[0]
        summary = first_day.get("summary")
        suitability = first_day.get("outdoor_suitability")
        return f"Weather note: {summary} Outdoor suitability is {suitability}."

    day_summaries = [_weather_day_text(day) for day in forecast[:3]]
    day_summaries = [text for text in day_summaries if text]
    if not day_summaries:
        return None
    return "Weather note: " + "; ".join(day_summaries) + "."


def _weather_day_text(day: dict[str, Any]) -> str:
    date = day.get("date")
    summary = day.get("summary")
    suitability = day.get("outdoor_suitability")
    temp = day.get("temperature_c")
    feels_like = day.get("feels_like_c")
    if temp is None:
        temp = day.get("temperature_f")
        feels_like = day.get("feels_like_f")
        unit = "°F"
    else:
        unit = "°C"

    parts = []
    if date:
        parts.append(str(date))
    if summary:
        parts.append(str(summary))
    if temp is not None:
        parts.append(f"{temp}{unit}")
    if feels_like is not None:
        parts.append(f"feels like {feels_like}{unit}")
    if suitability:
        parts.append(f"outdoor suitability {suitability}")
    return ", ".join(parts)


def _budget_note(tool_outputs: dict[str, Any]) -> str | None:
    budget = tool_outputs.get("budget_tool", {})
    if not budget:
        return None
    return f"Budget note: using {budget.get('budget_level')} budget guidance."


def _extract_rag_trace(tool_outputs: dict[str, Any]) -> dict[str, Any]:
    return tool_outputs.get("attraction_rag_tool", {}).get("rag_trace", {})


def _flight_note(tool_outputs: dict[str, Any]) -> str | None:
    flight = tool_outputs.get("flight_tool", {})
    if flight.get("status") != "ok":
        return None
    departure = flight.get("results", {}).get("departure_flights", [])
    if not departure:
        return None
    first = departure[0]
    return (
        f"Flight note: {first.get('airline')} {first.get('flight_number')} from "
        f"{flight.get('from_location')} to {flight.get('to_location')} from "
        f"${first.get('price')}."
    )


def _enrich_itinerary(itinerary: dict[str, Any], tool_outputs: dict[str, Any]) -> dict[str, Any]:
    """Attach structured places/hotels/flights sections to the itinerary.

    Backward-compatible: only adds new keys, never removes or renames existing
    ones. Each place and hotel entry carries an ``image_url`` (a Wikimedia
    Commons URL or None) so the front-end can render images. Flights are
    surfaced as structured cards (airline, price, duration, stops, booking
    link) for the front-end flight section. All sections are best-effort and
    default to empty lists when the corresponding tool did not run or returned
    no usable results.
    """
    itinerary["places"] = _structured_places(tool_outputs)
    itinerary["hotels"] = _structured_hotels(tool_outputs)
    itinerary["flights"] = _structured_flights(tool_outputs)
    itinerary["weather"] = _structured_weather(tool_outputs)
    return itinerary


def _structured_weather(tool_outputs: dict[str, Any]) -> dict[str, Any]:
    weather = tool_outputs.get("weather_tool", {})
    forecast = weather.get("forecast") or []
    if not weather:
        return {"status": "not_run", "forecast": []}
    return {
        "status": weather.get("status"),
        "city": weather.get("city"),
        "source": weather.get("source"),
        "forecast": [
            {
                "date": day.get("date"),
                "summary": day.get("summary"),
                "temperature_c": day.get("temperature_c"),
                "feels_like_c": day.get("feels_like_c"),
                "temperature_f": day.get("temperature_f"),
                "feels_like_f": day.get("feels_like_f"),
                "humidity": day.get("humidity"),
                "wind_speed": day.get("wind_speed"),
                "outdoor_suitability": day.get("outdoor_suitability"),
            }
            for day in forecast
        ],
    }


def _structured_places(tool_outputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a list of place cards with image_url from RAG attraction results."""
    results = tool_outputs.get("attraction_rag_tool", {}).get("results", [])
    places: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in results or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        # Skip externally-ingested chunks whose name is a raw text fragment.
        description = item.get("description", "") or ""
        if name in description[: len(name)] and len(name) > 40:
            continue
        key = str(name).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        places.append({
            "name": str(name),
            "image_url": item.get("image_url"),
            "description": description,
            "categories": item.get("categories"),
        })
    return places


def _structured_hotels(tool_outputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a list of hotel cards with image_url from the hotel tool output."""
    hotel = tool_outputs.get("hotel_tool", {})
    if hotel.get("status") != "ok":
        return []
    hotels: list[dict[str, Any]] = []
    for item in hotel.get("results", []) or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        hotels.append({
            "name": item.get("name"),
            "image_url": item.get("image_url"),
            "hotel_class": item.get("hotel_class"),
            "rating": item.get("rating"),
            "price_usd_per_night": item.get("price_usd_per_night") or item.get("price"),
            "booking_link": item.get("booking_link"),
            "area": item.get("area"),
            "budget_level": item.get("budget_level"),
        })
    return hotels


def _structured_flights(tool_outputs: dict[str, Any]) -> dict[str, Any]:
    """Build a structured flights section from the flight tool output."""
    flight = tool_outputs.get("flight_tool", {})
    if not flight:
        return {"status": "not_run", "departure_flights": [], "return_flights": []}
    if flight.get("status") != "ok":
        return {
            "status": flight.get("status", "not_run"),
            "message": flight.get("message"),
            "reason": flight.get("reason"),
            "from_location": flight.get("from_location"),
            "to_location": flight.get("to_location"),
            "departure_date": flight.get("departure_date"),
            "return_date": flight.get("return_date"),
            "departure_id": flight.get("departure_id"),
            "arrival_id": flight.get("arrival_id"),
            "departure_flights": [],
            "return_flights": [],
        }
    results = flight.get("results", {}) or {}
    return {
        "status": "ok",
        "from_location": flight.get("from_location"),
        "to_location": flight.get("to_location"),
        "departure_date": flight.get("departure_date"),
        "return_date": flight.get("return_date"),
        "departure_id": flight.get("departure_id"),
        "arrival_id": flight.get("arrival_id"),
        "departure_flights": results.get("departure_flights", []) or [],
        "return_flights": results.get("return_flights", []) or [],
    }
