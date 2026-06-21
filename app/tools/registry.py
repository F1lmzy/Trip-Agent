"""Tool registry metadata for the /api/tools endpoint and health checks.

Stores descriptive metadata about each available tool and checks reachability
without performing real work. Mirrors the tool-registry pattern from the Azure
AI Travel Agents sample, kept lightweight for this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolInfo:
    id: str
    name: str
    description: str
    type: str = "local"
    selected: bool = True
    tools: list[dict[str, str]] = field(default_factory=list)


_TOOL_REGISTRY: dict[str, ToolInfo] = {
    "attraction_rag_tool": ToolInfo(
        id="attraction_rag_tool",
        name="Attraction RAG",
        description="Multi-hop ChromaDB retrieval of attractions for a city.",
    ),
    "weather_tool": ToolInfo(
        id="weather_tool",
        name="Weather",
        description="Near-term weather forecast via OpenWeatherMap with fallback.",
    ),
    "budget_tool": ToolInfo(
        id="budget_tool",
        name="Budget",
        description="Low/medium/luxury budget guidance for the itinerary.",
    ),
    "serpapi_hotel_tool": ToolInfo(
        id="serpapi_hotel_tool",
        name="Hotels (SerpAPI)",
        description="Real-time hotel search via SerpAPI Google Hotels with live pricing and ratings.",
    ),
    "serpapi_flight_tool": ToolInfo(
        id="serpapi_flight_tool",
        name="Flights (SerpAPI)",
        description="Real-time flight search via SerpAPI Google Flights with live pricing and segments.",
    ),
    "web_search_tool": ToolInfo(
        id="web_search_tool",
        name="Web Search",
        description="Fresh travel context via LangChain DuckDuckGo search.",
    ),
    "destination_search_tool": ToolInfo(
        id="destination_search_tool",
        name="Destination Search",
        description="Destination discovery and inspiration via DuckDuckGo search.",
    ),
    "wikimedia_image_tool": ToolInfo(
        id="wikimedia_image_tool",
        name="Wikimedia Image Lookup",
        description="Resolve a place or attraction name to a Wikimedia Commons image URL.",
    ),
}


def list_tools() -> list[ToolInfo]:
    """Return metadata for every registered tool."""
    return list(_TOOL_REGISTRY.values())


def get_tool(tool_id: str) -> ToolInfo | None:
    """Return metadata for a single tool by id, or None if unknown."""
    return _TOOL_REGISTRY.get(tool_id)


def tools_count() -> int:
    """Return the number of registered tools."""
    return len(_TOOL_REGISTRY)
