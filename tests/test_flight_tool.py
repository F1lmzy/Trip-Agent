from app.tools.flight_tool import run_flight_tool


def test_flight_tool_returns_ok_with_departure_and_return_flights():
    result = run_flight_tool(
        from_location="London",
        to_location="Tokyo",
        departure_date="2026-07-15",
        return_date="2026-07-22",
        budget="medium",
        seed=42,
    )

    assert result["tool_name"] == "flight_tool"
    assert result["status"] == "ok"
    assert result["from_location"] == "London"
    assert result["to_location"] == "Tokyo"
    assert result["budget_level"] == "medium"
    flights = result["results"]
    assert len(flights["departure_flights"]) >= 3
    assert len(flights["return_flights"]) >= 3
    first = flights["departure_flights"][0]
    assert first["airline"]
    assert first["flight_number"]
    assert first["from_airport"]["city"] == "London"
    assert first["to_airport"]["city"] == "Tokyo"
    assert first["price"] > 0
    assert first["currency"] == "USD"


def test_flight_tool_omits_return_flights_when_return_date_missing():
    result = run_flight_tool("Mumbai", "Paris", "2026-08-01", seed=1)

    assert result["status"] == "ok"
    assert result["results"]["return_flights"] == []
    assert len(result["results"]["departure_flights"]) >= 3


def test_flight_tool_is_deterministic_with_same_seed():
    first = run_flight_tool("London", "Tokyo", "2026-07-15", "2026-07-22", seed=7)
    second = run_flight_tool("London", "Tokyo", "2026-07-15", "2026-07-22", seed=7)

    assert first == second


def test_flight_tool_rejects_invalid_departure_date_format():
    result = run_flight_tool("London", "Tokyo", "15 July 2026")

    assert result["status"] == "error"
    assert "ISO format" in result["message"]
    assert result["results"]["departure_flights"] == []


def test_flight_tool_rejects_return_before_departure():
    result = run_flight_tool("London", "Tokyo", "2026-07-15", "2026-07-10")

    assert result["status"] == "error"
    assert "after departure_date" in result["message"]


def test_flight_tool_prices_scale_with_budget_level():
    low = run_flight_tool("London", "Tokyo", "2026-07-15", seed=5, budget="low")
    luxury = run_flight_tool("London", "Tokyo", "2026-07-15", seed=5, budget="luxury")

    low_prices = [flight["price"] for flight in low["results"]["departure_flights"]]
    luxury_prices = [flight["price"] for flight in luxury["results"]["departure_flights"]]

    assert min(luxury_prices) > max(low_prices)


def test_flight_tool_unknown_budget_defaults_to_medium():
    result = run_flight_tool("London", "Tokyo", "2026-07-15", budget="surprise me", seed=3)

    assert result["budget_level"] == "medium"
