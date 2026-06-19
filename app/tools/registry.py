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
    "hotel_tool": ToolInfo(
        id="hotel_tool",
        name="Hotels",
        description="Curated mock hotel suggestions filtered by budget.",
    ),
    "flight_tool": ToolInfo(
        id="flight_tool",
        name="Flights",
        description="Mock flight suggestions between two locations and dates.",
    ),
    "web_search_tool": ToolInfo(
        id="web_search_tool",
        name="Web Search",
        description="Fresh travel context via LangChain DuckDuckGo search.",
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
