from app.agent.parser import parse_user_request
from app.agent.planner import create_trip_plan


def test_planner_clarifies_when_city_missing():
    parsed = parse_user_request("Plan me a trip")
    result = create_trip_plan(parsed)

    assert result.needs_clarification is True
    assert result.selected_tools == []
    assert result.clarifying_question


def test_planner_selects_core_tools_for_normal_trip():
    parsed = parse_user_request("Plan a 2-day trip to Tokyo. I like anime and food. Medium budget.")
    result = create_trip_plan(parsed)

    assert result.needs_clarification is False
    assert "attraction_rag_tool" in result.selected_tools
    assert "weather_tool" in result.selected_tools
    assert "budget_tool" in result.selected_tools
    assert "hotel_tool" not in result.selected_tools
    assert "web_search_tool" not in result.selected_tools
    assert result.plan


def test_planner_selects_hotel_only_when_requested():
    parsed = parse_user_request("Plan a Paris trip and suggest hotels.")
    result = create_trip_plan(parsed)

    assert "hotel_tool" in result.selected_tools


def test_planner_selects_web_search_for_current_info():
    parsed = parse_user_request("Plan a Tokyo trip with current events and latest food spots.")
    result = create_trip_plan(parsed)

    assert "web_search_tool" in result.selected_tools


def test_planner_selects_web_search_when_rag_context_is_weak():
    parsed = parse_user_request("Plan a 2-day trip to Reykjavik.")
    result = create_trip_plan(parsed, rag_context_is_weak=True)

    assert "web_search_tool" in result.selected_tools


def test_planner_follow_up_cheaper_uses_budget_tool():
    parsed = parse_user_request("Make it cheaper.")
    result = create_trip_plan(parsed)

    assert result.needs_clarification is False
    assert "budget_tool" in result.selected_tools
