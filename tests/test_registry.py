from app.tools.registry import get_tool, list_tools, tools_count


def test_list_tools_returns_all_tools():
    tools = list_tools()

    tool_ids = {tool.id for tool in tools}
    assert tool_ids == {
        "attraction_rag_tool",
        "weather_tool",
        "budget_tool",
        "serpapi_hotel_tool",
        "serpapi_flight_tool",
        "web_search_tool",
        "destination_search_tool",
        "wikimedia_image_tool",
    }


def test_tools_count_matches_list():
    assert tools_count() == len(list_tools()) == 8


def test_get_tool_returns_metadata_for_known_id():
    tool = get_tool("serpapi_flight_tool")

    assert tool is not None
    assert tool.name == "Flights (SerpAPI)"
    assert tool.description


def test_get_tool_returns_none_for_unknown_id():
    assert get_tool("nonexistent_tool") is None
