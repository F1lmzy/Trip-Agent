"""Tests for the MCP server tool registration and invocation.

These verify that the FastMCP server exposes all six tools and that calling
them returns the same shape as the underlying tool functions. We call the
registered tool functions directly via the FastMCP tool manager to avoid
spinning up a full HTTP transport in unit tests.

`call_tool` returns a tuple of (content_blocks, structured_output). The
structured output (second element) is the dict returned by the tool function.
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


def test_mcp_server_registers_all_six_tools():
    names = _tool_names()

    assert names == {
        "search_attractions",
        "get_weather",
        "apply_budget",
        "suggest_hotels",
        "suggest_flights",
        "web_search",
    }


def test_mcp_apply_budget_tool_returns_budget_guidance():
    result = _call("apply_budget", {"budget": "medium"})

    assert result["tool_name"] == "budget_tool"
    assert result["budget_level"] == "medium"
    assert result["guidance"]["activities"]


def test_mcp_suggest_flights_tool_returns_flight_suggestions():
    result = _call(
        "suggest_flights",
        {
            "from_location": "London",
            "to_location": "Tokyo",
            "departure_date": "2026-07-15",
            "return_date": "2026-07-22",
            "budget": "medium",
        },
    )

    assert result["tool_name"] == "flight_tool"
    assert result["status"] == "ok"
    assert len(result["results"]["departure_flights"]) >= 3


def test_mcp_get_weather_tool_returns_fallback_without_api_key(monkeypatch):
    # Force an empty key even when a developer's local .env contains one.
    monkeypatch.setenv("OPENWEATHER_API_KEY", "")
    from app.config import get_settings

    get_settings.cache_clear()

    result = _call("get_weather", {"city": "Tokyo"})

    assert result["tool_name"] == "weather_tool"
    assert result["source"] == "fallback"
