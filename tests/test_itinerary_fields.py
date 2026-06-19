"""Contract tests for the response_generator itinerary image/structured fields.

Asserts the iteration-4 contract: the itinerary dict surfaces structured
``places`` (with image_url), ``hotels`` (with image_url), and ``flights``
sections derived from tool_outputs, while preserving the existing day_N slots
and notes. Uses FakeImageClient (no live network) and the mock hotel/flight
tools (no SerpAPI quota use).
"""

from app.agent.parser import ParsedRequest
from app.agent.planner import PlanningResult
from app.agent.response_generator import generate_itinerary_response
from tests.fakes import FakeImageClient


def _parsed(**kwargs) -> ParsedRequest:
    base = dict(raw_message="Plan a 2-day Tokyo trip. I like anime and food. Medium budget.")
    base.update(kwargs)
    return ParsedRequest(**base)


def _plan(parsed: ParsedRequest) -> PlanningResult:
    return PlanningResult(
        plan=["step"],
        selected_tools=["attraction_rag_tool", "hotel_tool", "flight_tool"],
        needs_clarification=False,
        clarifying_question=None,
    )


def _tool_outputs_with_images() -> dict:
    return {
        "attraction_rag_tool": {
            "status": "ok",
            "results": [
                {"name": "Senso-ji Temple", "description": "ancient temple in Asakusa", "image_url": "https://upload.wikimedia.org/senso.jpg", "categories": "culture, history"},
                {"name": "Shibuya Sky", "description": "rooftop viewing deck", "image_url": "https://upload.wikimedia.org/shibuya.jpg", "categories": "views, photography"},
            ],
            "rag_trace": {"hop_1": [], "hop_2": []},
        },
        "hotel_tool": {
            "tool_name": "hotel_tool",
            "status": "ok",
            "city": "Tokyo",
            "results": [
                {"name": "Shinjuku Grand Hotel", "area": "Shinjuku", "budget_level": "medium", "image_url": "https://upload.wikimedia.org/shinjuku.jpg"},
            ],
        },
        "flight_tool": {
            "tool_name": "flight_tool",
            "status": "ok",
            "from_location": "London",
            "to_location": "Tokyo",
            "departure_date": "2024-10-01",
            "return_date": "2024-10-03",
            "results": {
                "departure_flights": [
                    {"airline": "SkyWings", "flight_number": "S100", "price": 450.0, "duration_minutes": 600, "stops": 0, "is_direct": True, "booking_link": None}
                ],
                "return_flights": [],
            },
        },
    }


def test_itinerary_places_carry_image_url():
    parsed = _parsed(city="Tokyo", origin_city="London")
    response = generate_itinerary_response(
        parsed=parsed,
        plan=_plan(parsed),
        tool_outputs=_tool_outputs_with_images(),
        api_key="",
        memory_used=[],
    )

    places = response.itinerary.get("places", [])
    assert len(places) == 2
    assert places[0]["name"] == "Senso-ji Temple"
    assert places[0]["image_url"] == "https://upload.wikimedia.org/senso.jpg"
    assert places[1]["name"] == "Shibuya Sky"
    assert places[1]["image_url"] == "https://upload.wikimedia.org/shibuya.jpg"


def test_itinerary_hotels_carry_image_url():
    parsed = _parsed(city="Tokyo", origin_city="London")
    response = generate_itinerary_response(
        parsed=parsed,
        plan=_plan(parsed),
        tool_outputs=_tool_outputs_with_images(),
        api_key="",
        memory_used=[],
    )

    hotels = response.itinerary.get("hotels", [])
    assert len(hotels) == 1
    assert hotels[0]["name"] == "Shinjuku Grand Hotel"
    assert hotels[0]["image_url"] == "https://upload.wikimedia.org/shinjuku.jpg"
    assert hotels[0]["area"] == "Shinjuku"


def test_itinerary_flights_structured():
    parsed = _parsed(city="Tokyo", origin_city="London")
    response = generate_itinerary_response(
        parsed=parsed,
        plan=_plan(parsed),
        tool_outputs=_tool_outputs_with_images(),
        api_key="",
        memory_used=[],
    )

    flights = response.itinerary["flights"]
    assert flights["status"] == "ok"
    assert flights["from_location"] == "London"
    assert flights["to_location"] == "Tokyo"
    assert len(flights["departure_flights"]) == 1
    assert flights["departure_flights"][0]["airline"] == "SkyWings"
    assert flights["departure_flights"][0]["price"] == 450.0


def test_itinerary_preserves_day_slots_and_notes():
    parsed = _parsed(city="Tokyo", origin_city="London")
    response = generate_itinerary_response(
        parsed=parsed,
        plan=_plan(parsed),
        tool_outputs=_tool_outputs_with_images(),
        api_key="",
        memory_used=[],
    )

    # Existing fields are preserved (backward compatibility).
    assert response.itinerary["city"] == "Tokyo"
    assert response.itinerary["duration_days"] == 2
    assert "day_1" in response.itinerary
    assert "morning" in response.itinerary["day_1"]
    assert "notes" in response.itinerary


def test_itinerary_image_url_is_none_when_not_found():
    # Attractions with image_url=None (no Commons hit) still appear, with None.
    tool_outputs = _tool_outputs_with_images()
    tool_outputs["attraction_rag_tool"]["results"][0]["image_url"] = None
    parsed = _parsed(city="Tokyo", origin_city="London")
    response = generate_itinerary_response(
        parsed=parsed,
        plan=_plan(parsed),
        tool_outputs=tool_outputs,
        api_key="",
        memory_used=[],
    )

    places = response.itinerary["places"]
    assert places[0]["image_url"] is None
    assert places[0]["name"] == "Senso-ji Temple"


def test_itinerary_places_skips_external_text_fragment_names():
    tool_outputs = _tool_outputs_with_images()
    # A long name that is a prefix of its description (external chunk) is skipped.
    tool_outputs["attraction_rag_tool"]["results"].append(
        {"name": "Tokyo is the capital of Japan and a bustling metropolis", "description": "Tokyo is the capital of Japan and a bustling metropolis that...", "image_url": None}
    )
    parsed = _parsed(city="Tokyo", origin_city="London")
    response = generate_itinerary_response(
        parsed=parsed,
        plan=_plan(parsed),
        tool_outputs=tool_outputs,
        api_key="",
        memory_used=[],
    )

    names = [p["name"] for p in response.itinerary["places"]]
    assert "Tokyo is the capital of Japan and a bustling metropolis" not in names
    assert "Senso-ji Temple" in names


def test_itinerary_empty_sections_when_tools_not_run():
    parsed = _parsed(city="Tokyo")
    response = generate_itinerary_response(
        parsed=parsed,
        plan=PlanningResult(plan=["step"], selected_tools=[], needs_clarification=False, clarifying_question=None),
        tool_outputs={},
        api_key="",
        memory_used=[],
    )

    assert response.itinerary["places"] == []
    assert response.itinerary["hotels"] == []
    assert response.itinerary["flights"] == {"status": "not_run", "departure_flights": [], "return_flights": []}


def test_itinerary_dedupes_places_by_name():
    tool_outputs = _tool_outputs_with_images()
    tool_outputs["attraction_rag_tool"]["results"].append(
        {"name": "Senso-ji Temple", "description": "duplicate", "image_url": "https://x/2.jpg"}
    )
    parsed = _parsed(city="Tokyo", origin_city="London")
    response = generate_itinerary_response(
        parsed=parsed,
        plan=_plan(parsed),
        tool_outputs=tool_outputs,
        api_key="",
        memory_used=[],
    )

    names = [p["name"] for p in response.itinerary["places"]]
    assert names.count("Senso-ji Temple") == 1
