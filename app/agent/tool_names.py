"""Shared tool name constants used by planner, executor, and registry.

Each member is a ``str`` (via ``StrEnum``), so it can be used anywhere a
string tool name is expected: ``selected_tools`` lists, ``tool_outputs``
dict keys, and string comparisons all work directly.
"""

from enum import StrEnum


class ToolName(StrEnum):
    ATTRACTION_RAG = "attraction_rag_tool"
    WEATHER = "weather_tool"
    BUDGET = "budget_tool"
    WEB_SEARCH = "web_search_tool"
    HOTEL = "hotel_tool"
    FLIGHT = "flight_tool"
