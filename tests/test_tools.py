from app.tools.budget_tool import run_budget_tool
from app.tools.hotel_tool import load_hotels, run_hotel_tool


def test_budget_tool_defaults_to_medium_when_missing():
    result = run_budget_tool(None)

    assert result["tool_name"] == "budget_tool"
    assert result["status"] == "ok"
    assert result["budget_level"] == "medium"
    assert result["assumed"] is True
    assert result["guidance"]["activities"]


def test_budget_tool_returns_low_medium_luxury_guidance():
    assert run_budget_tool("low")["guidance"]["activities"]
    assert run_budget_tool("medium")["guidance"]["activities"]
    assert run_budget_tool("luxury")["guidance"]["activities"]


def test_budget_tool_unknown_value_falls_back_to_medium():
    result = run_budget_tool("surprise me")

    assert result["status"] == "fallback"
    assert result["budget_level"] == "medium"
    assert result["assumed"] is True


def test_hotel_seed_data_includes_demo_city_budget_levels():
    hotels = load_hotels()
    pairs = {(hotel["city"], hotel["budget_level"]) for hotel in hotels}

    for city in ["Tokyo", "Singapore", "Paris", "New York"]:
        assert {(city, "low"), (city, "medium"), (city, "luxury")} <= pairs


def test_hotel_tool_returns_city_budget_matches():
    result = run_hotel_tool("Tokyo", "medium")

    assert result["tool_name"] == "hotel_tool"
    assert result["status"] == "ok"
    assert result["results"]
    assert all(hotel["city"] == "Tokyo" for hotel in result["results"])
    assert all(hotel["budget_level"] == "medium" for hotel in result["results"])


def test_hotel_tool_defaults_missing_budget_to_medium():
    result = run_hotel_tool("Singapore", None)

    assert result["budget_level"] == "medium"
    assert result["results"]
    assert all(hotel["budget_level"] == "medium" for hotel in result["results"])


def test_hotel_tool_unknown_city_returns_no_results():
    result = run_hotel_tool("Atlantis", "low")

    assert result["status"] == "no_results"
    assert result["city"] == "Atlantis"
    assert result["budget_level"] == "low"
    assert result["results"] == []
