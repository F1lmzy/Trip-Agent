"""Tests for the MCP server tool registration and invocation.

These verify that the FastMCP server exposes all eight tools and that calling
them returns the correct shape. We call the registered tool functions directly
via the FastMCP tool manager to avoid spinning up a full HTTP transport.

``call_tool`` returns a tuple of ``(content_blocks, structured_output)``.
The structured output (second element) is the dict returned by the tool function.
"""

import asyncio

from app.mcp_server import mcp


def _tool_names() -> set[str]:
    """Return the set of tool names registered on the FastMCP server."""
    tools = asyncio.run(mcp.list_tools())
    return {tool.name for tool in tools}


def _call(tool_name: str, arguments: dict) -> dict:
    """Call an MCP tool and return its structured output."""
    _content, structured = asyncio.run(mcp.call_tool(tool_name, arguments))
    return structured


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_mcp_server_registers_all_tools():
    names = _tool_names()

    assert names == {
        "search_attractions",
        "get_weather",
        "apply_budget",
        "search_hotels",
        "search_flights",
        "web_search",
        "search_destinations",
        "lookup_place_image",
    }


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_mcp_apply_budget_tool_returns_budget_guidance():
    result = _call("apply_budget", {"budget": "medium"})

    assert result["tool_name"] == "budget_tool"
    assert result["budget_level"] == "medium"
    assert result["guidance"]["activities"]


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------


def test_mcp_get_weather_tool_returns_fallback_without_api_key(monkeypatch):
    # Force an empty key even when a developer's local .env contains one.
    monkeypatch.setenv("OPENWEATHER_API_KEY", "")
    from app.config import get_settings

    get_settings.cache_clear()

    result = _call("get_weather", {"city": "Tokyo"})

    assert result["tool_name"] == "weather_tool"
    assert result["source"] == "fallback"


# ---------------------------------------------------------------------------
# Hotels via SerpAPI
# ---------------------------------------------------------------------------


def test_mcp_search_hotels_tool_returns_serpapi_shape(monkeypatch):
    """search_hotels delegates to run_serpapi_hotel_tool; verify shape."""

    def mock_serpapi_hotel(city, check_in_date, check_out_date, **kwargs):
        return {
            "tool_name": "serpapi_hotel_tool",
            "status": "ok",
            "city": city,
            "budget_level": kwargs.get("budget") or "medium",
            "results": [
                {
                    "name": "Le Test Hôtel",
                    "price_usd_per_night": 250.0,
                    "price_usd_total": 1750.0,
                    "rating": 4.7,
                    "reviews": 312,
                }
            ],
        }

    monkeypatch.setattr("app.mcp_server.run_serpapi_hotel_tool", mock_serpapi_hotel)

    result = _call(
        "search_hotels",
        {
            "city": "Paris",
            "check_in_date": "2026-07-15",
            "check_out_date": "2026-07-22",
            "budget": "luxury",
        },
    )

    assert result["tool_name"] == "serpapi_hotel_tool"
    assert result["status"] == "ok"
    assert result["city"] == "Paris"
    assert len(result["results"]) == 1
    assert result["results"][0]["name"] == "Le Test Hôtel"


# ---------------------------------------------------------------------------
# Flights via SerpAPI
# ---------------------------------------------------------------------------


def test_mcp_search_flights_tool_returns_serpapi_shape(monkeypatch):
    """search_flights delegates to run_serpapi_flight_tool; verify shape."""

    def mock_serpapi_flight(from_location, to_location, departure_date, **kwargs):
        return {
            "tool_name": "serpapi_flight_tool",
            "status": "ok",
            "from_location": from_location,
            "to_location": to_location,
            "departure_date": departure_date,
            "return_date": kwargs.get("return_date"),
            "budget_level": kwargs.get("budget") or "medium",
            "results": {
                "departure_flights": [
                    {
                        "airline": "Air France",
                        "price": 850.0,
                        "currency": "USD",
                        "is_direct": True,
                        "stops": 0,
                        "departure": "2026-07-15T08:00:00",
                        "arrival": "2026-07-15T09:30:00",
                    }
                ],
                "return_flights": [],
            },
        }

    monkeypatch.setattr("app.mcp_server.run_serpapi_flight_tool", mock_serpapi_flight)

    result = _call(
        "search_flights",
        {
            "from_location": "London",
            "to_location": "Paris",
            "departure_date": "2026-07-15",
            "budget": "medium",
        },
    )

    assert result["tool_name"] == "serpapi_flight_tool"
    assert result["status"] == "ok"
    assert result["from_location"] == "London"
    assert result["to_location"] == "Paris"
    assert len(result["results"]["departure_flights"]) == 1
    assert result["results"]["departure_flights"][0]["airline"] == "Air France"


# ---------------------------------------------------------------------------
# Destination search
# ---------------------------------------------------------------------------


def test_mcp_search_destinations_tool_returns_results(monkeypatch):
    """search_destinations delegates to run_destination_search_tool."""

    def mock_dest_search(query_intent, **kwargs):
        return {
            "tool_name": "web_search_tool",
            "status": "ok",
            "query": query_intent,
            "results": [
                {"title": "Top Beach Destinations", "url": "http://example.com", "description": "Best beaches in Asia"}
            ],
        }

    monkeypatch.setattr("app.mcp_server.run_destination_search_tool", mock_dest_search)

    result = _call("search_destinations", {"query": "beach vacation asia"})

    assert result["tool_name"] == "web_search_tool"
    assert result["status"] == "ok"
    assert "beach" in result["query"]
    assert len(result["results"]) >= 1


# ---------------------------------------------------------------------------
# Wikimedia image lookup
# ---------------------------------------------------------------------------


def test_mcp_lookup_place_image_tool_returns_url(monkeypatch):
    """lookup_place_image delegates to resolve_place_image."""

    def mock_resolve(place_name, **kwargs):
        return "https://upload.wikimedia.org/wikipedia/commons/thumb/test.jpg"

    monkeypatch.setattr("app.mcp_server.resolve_place_image", mock_resolve)

    result = _call("lookup_place_image", {"place_name": "Eiffel Tower"})

    assert result["place_name"] == "Eiffel Tower"
    assert result["image_url"] == "https://upload.wikimedia.org/wikipedia/commons/thumb/test.jpg"


def test_mcp_lookup_place_image_tool_returns_none_when_not_found(monkeypatch):
    """When resolve_place_image returns None, image_url should be None."""

    def mock_resolve(place_name, **kwargs):
        return None

    monkeypatch.setattr("app.mcp_server.resolve_place_image", mock_resolve)

    result = _call("lookup_place_image", {"place_name": "NonexistentPlaceXYZ"})

    assert result["place_name"] == "NonexistentPlaceXYZ"
    assert result["image_url"] is None
