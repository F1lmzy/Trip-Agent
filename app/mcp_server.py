"""Model Context Protocol (MCP) server for the Travel Agent tools.

Exposes the project's existing tools as MCP tools using FastMCP, so any
MCP-compatible client (Claude Desktop, Cursor, VS Code, etc.) can call them.
The server can be mounted into the FastAPI app at `/mcp` or run standalone.

Run standalone:
    python -m app.mcp_server

Mount in FastAPI (see app/main.py):
    app.mount("/mcp", mcp_app)

Connect from an MCP client:
    URL: http://127.0.0.1:8000/mcp
"""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app.config import get_settings
from app.tools.attraction_rag_tool import AttractionRagTool
from app.tools.budget_tool import run_budget_tool
from app.tools.serpapi_flight_tool import run_serpapi_flight_tool
from app.tools.serpapi_hotel_tool import run_serpapi_hotel_tool
from app.tools.weather_tool import run_weather_tool
from app.tools.web_search_tool import run_destination_search_tool, run_web_search_tool
from app.tools.wikimedia_image_tool import resolve_place_image

mcp = FastMCP("Travel Agent Tools")

# Module-level RAG tool instance, seeded lazily on first attraction query.
_rag_tool: AttractionRagTool | None = None


def _get_seeded_rag_tool() -> AttractionRagTool:
    global _rag_tool
    if _rag_tool is None:
        _rag_tool = AttractionRagTool()
    _rag_tool.seed_if_needed()
    return _rag_tool


# ---------------------------------------------------------------------------
# Attractions (RAG / curated)
# ---------------------------------------------------------------------------


@mcp.tool()
def search_attractions(
    city: Annotated[str, Field(description="City to find attractions in, e.g. Tokyo")],
    interests: Annotated[list[str] | None, Field(description="Optional interests like food, anime, museums")] = None,
    limit: Annotated[int, Field(description="Maximum attractions to return", ge=1, le=20)] = 5,
) -> dict[str, Any]:
    """Multi-hop RAG retrieval of attractions for a city using ChromaDB.

    Performs a city-overview hop then an interest-specific hop and returns
    matched attractions plus a rag_trace with hop summaries.
    """
    return _get_seeded_rag_tool().run(city=city, interests=interests or [], limit=limit)


# ---------------------------------------------------------------------------
# Weather (OpenWeatherMap)
# ---------------------------------------------------------------------------


@mcp.tool()
def get_weather(
    city: Annotated[str, Field(description="City to get the forecast for, e.g. Paris")],
) -> dict[str, Any]:
    """Get a near-term weather forecast for a city via OpenWeatherMap.

    Falls back to a graceful message when OPENWEATHER_API_KEY is not set.
    """
    return run_weather_tool(city, api_key=get_settings().openweather_api_key)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@mcp.tool()
def apply_budget(
    budget: Annotated[str | None, Field(description="Budget level: low, medium, or luxury")] = None,
) -> dict[str, Any]:
    """Apply low/medium/luxury budget guidance for meals, hotels, activities, transport.

    Defaults to medium when the budget is missing or unrecognized.
    """
    return run_budget_tool(budget)


# ---------------------------------------------------------------------------
# Hotels via SerpAPI (real-time)
# ---------------------------------------------------------------------------


@mcp.tool()
def search_hotels(
    city: Annotated[str, Field(description="City to search for hotels, e.g. Singapore")],
    check_in_date: Annotated[str, Field(description="Check-in date in ISO format (YYYY-MM-DD)")],
    check_out_date: Annotated[str, Field(description="Check-out date in ISO format (YYYY-MM-DD)")],
    api_key: Annotated[str | None, Field(description="SerpAPI API key (optional; falls back to SERPAPI_API_KEY env)")] = None,
    budget: Annotated[str | None, Field(description="Budget level: low, medium, or luxury")] = None,
    limit: Annotated[int, Field(description="Maximum hotels to return", ge=1, le=10)] = 5,
) -> dict[str, Any]:
    """Search real-time hotel listings via SerpAPI Google Hotels.

    Requires a SerpAPI API key. Provide via ``api_key`` parameter or set
    the ``SERPAPI_API_KEY`` environment variable. Returns live pricing,
    ratings, and booking links when available.
    """
    key = api_key if api_key is not None else get_settings().serpapi_api_key
    return run_serpapi_hotel_tool(
        city=city,
        check_in_date=check_in_date,
        check_out_date=check_out_date,
        budget=budget,
        limit=limit,
        api_key=key,
    )


# ---------------------------------------------------------------------------
# Flights via SerpAPI (real-time)
# ---------------------------------------------------------------------------


@mcp.tool()
def search_flights(
    from_location: Annotated[str, Field(description="Departure city or airport")],
    to_location: Annotated[str, Field(description="Destination city or airport")],
    departure_date: Annotated[str, Field(description="Departure date in ISO format (YYYY-MM-DD)")],
    return_date: Annotated[str | None, Field(description="Optional return date in ISO format (YYYY-MM-DD)")] = None,
    api_key: Annotated[str | None, Field(description="SerpAPI API key (optional; falls back to SERPAPI_API_KEY env)")] = None,
    budget: Annotated[str | None, Field(description="Budget level: low, medium, or luxury")] = None,
) -> dict[str, Any]:
    """Search real-time flight options via SerpAPI Google Flights.

    Requires a SerpAPI API key. Provide via ``api_key`` parameter or set
    the ``SERPAPI_API_KEY`` environment variable. Returns live pricing,
    airlines, segments, and booking links when available.
    """
    key = api_key if api_key is not None else get_settings().serpapi_api_key
    return run_serpapi_flight_tool(
        from_location=from_location,
        to_location=to_location,
        departure_date=departure_date,
        return_date=return_date,
        budget=budget,
        api_key=key,
    )


# ---------------------------------------------------------------------------
# Web search (DuckDuckGo, no API key required)
# ---------------------------------------------------------------------------


@mcp.tool()
def web_search(
    city: Annotated[str, Field(description="City the search should focus on, e.g. Tokyo")],
    query_intent: Annotated[str, Field(description="What to search for, e.g. current events and food spots")] = "",
) -> dict[str, Any]:
    """Search the web for fresh travel context using LangChain's DuckDuckGo integration.

    Useful for current events, closures, or recent recommendations that are
    not in the curated RAG knowledge base. No API key required.
    """
    return run_web_search_tool(city, query_intent=query_intent or "highlights, food, travel tips")


# ---------------------------------------------------------------------------
# Destination search (DuckDuckGo, no API key required)
# ---------------------------------------------------------------------------


@mcp.tool()
def search_destinations(
    query: Annotated[str, Field(description="Search query for destinations, e.g. 'beach vacation asia'")],
) -> dict[str, Any]:
    """Search for travel destinations matching a query using DuckDuckGo.

    Useful for destination inspiration and discovery when the user has not
    yet chosen a specific city. No API key required.
    """
    return run_destination_search_tool(query_intent=query)


# ---------------------------------------------------------------------------
# Wikimedia Commons image lookup
# ---------------------------------------------------------------------------


@mcp.tool()
def lookup_place_image(
    place_name: Annotated[str, Field(description="Name of the place, attraction, or hotel, e.g. 'Eiffel Tower'")],
) -> dict[str, Any]:
    """Resolve a place name to a Wikimedia Commons image URL.

    Returns a thumbnail image URL (width 480) for the given place, or
    ``image_url: null`` if no suitable image is found. Free API, no key
    required.
    """
    url = resolve_place_image(place_name)
    return {"place_name": place_name, "image_url": url}


# ---------------------------------------------------------------------------
# ASGI app and standalone runner
# ---------------------------------------------------------------------------


def create_mcp_app():
    """Return the ASGI app for the MCP streamable-http transport.

    Mount this under a FastAPI/Starlette route, e.g.:
        app.mount("/mcp", create_mcp_app())
    """
    return mcp.streamable_http_app()


def main() -> None:
    """Run the MCP server standalone with streamable-http transport."""
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
