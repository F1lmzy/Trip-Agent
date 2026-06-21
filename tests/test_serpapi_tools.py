"""Tests for the SerpAPI flight and hotel tools.

Uses httpx.MockTransport with realistic SerpAPI google_flights / google_hotels
JSON payloads — no real network, no real API key required.
"""

import httpx

from app.tools.serpapi_flight_tool import run_serpapi_flight_tool
from app.tools.serpapi_hotel_tool import _budget_to_class, run_serpapi_hotel_tool


# --- Realistic SerpAPI google_flights payloads ---

_ROUND_TRIP_PAYLOAD = {
    "search_metadata": {"status": "Success"},
    "best_flights": [
        {
            "type": "round trip",
            "price": 412,
            "total_duration": 130,
            "booking_link": "https://www.google.com/flights/rt/abc",
            "departures": [
                {
                    "flights": [
                        {
                            "airline": "Air France",
                            "flight_number": "AF 1080",
                            "departure_airport": {
                                "name": "London Heathrow",
                                "id": "LHR",
                                "time": "2024-10-01T08:00:00",
                            },
                            "arrival_airport": {
                                "name": "Charles de Gaulle",
                                "id": "CDG",
                                "time": "2024-10-01T10:10:00",
                            },
                            "duration": 130,
                            "airplane": "Airbus A320",
                            "travel_class": "Economy",
                        }
                    ],
                    "total_duration": 130,
                    "price": 412,
                }
            ],
            "returns": [
                {
                    "flights": [
                        {
                            "airline": "Air France",
                            "flight_number": "AF 1081",
                            "departure_airport": {
                                "name": "Charles de Gaulle",
                                "id": "CDG",
                                "time": "2024-10-04T19:00:00",
                            },
                            "arrival_airport": {
                                "name": "London Heathrow",
                                "id": "LHR",
                                "time": "2024-10-04T19:15:00",
                            },
                            "duration": 75,
                            "airplane": "Airbus A320",
                            "travel_class": "Economy",
                        }
                    ],
                    "total_duration": 75,
                }
            ],
        }
    ],
    "other_flights": [
        {
            "type": "round trip",
            "price": 520,
            "departures": [
                {
                    "flights": [
                        {
                            "airline": "British Airways",
                            "flight_number": "BA 308",
                            "departure_airport": {
                                "name": "London Heathrow",
                                "id": "LHR",
                                "time": "2024-10-01T09:00:00",
                            },
                            "arrival_airport": {
                                "name": "Charles de Gaulle",
                                "id": "CDG",
                                "time": "2024-10-01T11:15:00",
                            },
                            "duration": 135,
                        }
                    ],
                }
            ],
            "returns": [],
        }
    ],
}

_ROUND_TRIP_FLAT_PAYLOAD = {
    "search_metadata": {"status": "Success"},
    "best_flights": [
        {
            "type": "Round trip",
            "price": 525,
            "total_duration": 390,
            "booking_link": "https://www.google.com/flights/rt/flat",
            "departure_token": "outbound-token-123",
            "flights": [
                {
                    "airline": "Singapore Airlines",
                    "flight_number": "SQ 620",
                    "departure_airport": {"name": "Singapore Changi Airport", "id": "SIN", "time": "2026-06-21T08:30:00"},
                    "arrival_airport": {"name": "Kansai International Airport", "id": "KIX", "time": "2026-06-21T16:00:00"},
                    "duration": 390,
                    "airplane": "Boeing 787",
                    "travel_class": "Economy",
                }
            ],
        }
    ],
}

_RETURN_FLAT_PAYLOAD = {
    "search_metadata": {"status": "Success"},
    "best_flights": [
        {
            "type": "Round trip",
            "price": 525,
            "total_duration": 410,
            "flights": [
                {
                    "airline": "Singapore Airlines",
                    "flight_number": "SQ 621",
                    "departure_airport": {"name": "Kansai International Airport", "id": "KIX", "time": "2026-06-24T17:30:00"},
                    "arrival_airport": {"name": "Singapore Changi Airport", "id": "SIN", "time": "2026-06-24T23:20:00"},
                    "duration": 410,
                    "airplane": "Boeing 787",
                    "travel_class": "Economy",
                }
            ],
        }
    ],
}

_ONE_WAY_PAYLOAD = {
    "search_metadata": {"status": "Success"},
    "best_flights": [
        {
            "type": "one way",
            "price": 89,
            "total_duration": 130,
            "booking_link": "https://www.google.com/flights/ow/xyz",
            "flights": [
                {
                    "airline": "EasyJet",
                    "flight_number": "U2 8001",
                    "departure_airport": {"name": "London Gatwick", "id": "LGW", "time": "2024-10-01T07:00:00"},
                    "arrival_airport": {"name": "Charles de Gaulle", "id": "CDG", "time": "2024-10-01T09:10:00"},
                    "duration": 130,
                    "travel_class": "Economy",
                }
            ],
        }
    ],
}


_HOTELS_PAYLOAD = {
    "search_metadata": {"status": "Success"},
    "properties": [
        {
            "name": "Hotel Rivoli Paris",
            "hotel_class": 4,
            "rating": 4.5,
            "reviews": 1200,
            "rate_per_night": {"lowest": "$180", "extracted_lowest": 180.0},
            "total_rate": {"lowest": "$540", "extracted_lowest": 540.0},
            "check_in_date": "2024-10-01",
            "check_out_date": "2024-10-04",
            "link": "https://www.google.com/travel/hotels/rivoli",
            "images": [{"thumbnail": "https://example.com/rivoli.jpg", "original": "https://example.com/rivoli_full.jpg"}],
            "address": "1 Rue de Rivoli, Paris",
        },
        {
            "name": "Budget Inn",
            "hotel_class": 2,
            "rating": 3.2,
            "reviews": 300,
            "rate_per_night": {"extracted_lowest": 60.0},
            "total_rate": {"extracted_lowest": 180.0},
            "link": "https://www.google.com/travel/hotels/budget",
        },
    ],
}


def _mock_flight_client(payload: dict) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "serpapi.com" in str(request.url)
        assert "google_flights" in str(request.url)
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _mock_hotel_client(payload: dict) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "serpapi.com" in str(request.url)
        assert "google_hotels" in str(request.url)
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


# --- Flight tool tests ---


def test_serpapi_flight_parses_round_trip():
    client = _mock_flight_client(_ROUND_TRIP_PAYLOAD)
    result = run_serpapi_flight_tool(
        from_location="London",
        to_location="Paris",
        departure_date="2024-10-01",
        return_date="2024-10-04",
        api_key="fake-key",
        client=client,
    )

    assert result["tool_name"] == "serpapi_flight_tool"
    assert result["status"] == "ok"
    assert result["from_location"] == "London"
    assert result["to_location"] == "Paris"
    departures = result["results"]["departure_flights"]
    returns = result["results"]["return_flights"]
    assert len(departures) >= 2  # best_flights + other_flights departures
    assert len(returns) >= 1

    first = departures[0]
    assert first["airline"] == "Air France"
    assert first["flight_number"] == "AF 1080"
    assert first["price"] == 412.0
    assert first["currency"] == "USD"
    assert first["is_direct"] is True
    assert first["stops"] == 0
    assert first["from_airport"]["code"] == "LHR"
    assert first["to_airport"]["code"] == "CDG"
    assert first["departure"] == "2024-10-01T08:00:00"
    assert first["arrival"] == "2024-10-01T10:10:00"
    assert first["duration_minutes"] == 130
    assert first["booking_link"].startswith("https://www.google.com/flights")
    assert first["segments"][0]["airplane"] == "Airbus A320"


def test_serpapi_flight_parses_round_trip_flat_payload_and_fetches_return_flights_with_token():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        params = request.url.params
        if "departure_token" in params:
            assert params["departure_token"] == "outbound-token-123"
            assert params["type"] == "1"
            assert params["return_date"] == "2026-06-24"
            assert params["departure_id"] == "SIN"
            assert params["arrival_id"] == "KIX"
            assert params["outbound_date"] == "2026-06-21"
            return httpx.Response(200, json=_RETURN_FLAT_PAYLOAD)
        assert params["departure_id"] == "SIN"
        assert params["arrival_id"] == "KIX"
        return httpx.Response(200, json=_ROUND_TRIP_FLAT_PAYLOAD)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = run_serpapi_flight_tool(
        from_location="Singapore",
        to_location="Osaka",
        departure_date="2026-06-21",
        return_date="2026-06-24",
        api_key="fake-key",
        client=client,
    )

    assert result["tool_name"] == "serpapi_flight_tool"
    assert result["status"] == "ok"
    assert result["departure_id"] == "SIN"
    assert result["arrival_id"] == "KIX"
    assert result["departure_token"] == "outbound-token-123"
    departures = result["results"]["departure_flights"]
    returns = result["results"]["return_flights"]
    assert len(requests) == 2
    assert len(departures) == 1
    assert len(returns) == 1
    assert departures[0]["flight_number"] == "SQ 620"
    assert returns[0]["flight_number"] == "SQ 621"
    assert departures[0]["from_airport"]["code"] == "SIN"
    assert departures[0]["to_airport"]["code"] == "KIX"
    assert returns[0]["from_airport"]["code"] == "KIX"
    assert returns[0]["to_airport"]["code"] == "SIN"


def test_serpapi_flight_keeps_outbound_when_return_token_lookup_fails():
    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "departure_token" in params:
            return httpx.Response(400, json={"error": "Invalid departure_token"})
        return httpx.Response(200, json=_ROUND_TRIP_FLAT_PAYLOAD)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = run_serpapi_flight_tool(
        from_location="Singapore",
        to_location="Osaka",
        departure_date="2026-06-21",
        return_date="2026-06-24",
        api_key="fake-key",
        client=client,
    )

    assert result["status"] == "ok"
    assert result["results"]["departure_flights"]
    assert result["results"]["return_flights"] == []


def test_serpapi_flight_parses_one_way():
    client = _mock_flight_client(_ONE_WAY_PAYLOAD)
    result = run_serpapi_flight_tool(
        from_location="London",
        to_location="Paris",
        departure_date="2024-10-01",
        return_date=None,
        api_key="fake-key",
        client=client,
    )

    assert result["status"] == "ok"
    departures = result["results"]["departure_flights"]
    returns = result["results"]["return_flights"]
    assert len(departures) == 1
    assert returns == []
    assert departures[0]["airline"] == "EasyJet"
    assert departures[0]["price"] == 89.0
    assert departures[0]["is_direct"] is True


def test_serpapi_flight_no_key_degrades_to_no_results():
    client = _mock_flight_client(_ROUND_TRIP_PAYLOAD)
    result = run_serpapi_flight_tool(
        from_location="London",
        to_location="Paris",
        departure_date="2024-10-01",
        api_key="",  # explicit empty -> no key
        client=client,
    )

    assert result["status"] == "no_results"
    assert result["reason"] == "serpapi_key_missing"
    assert result["results"]["departure_flights"] == []


def test_serpapi_flight_unmapped_city_degrades():
    client = _mock_flight_client(_ROUND_TRIP_PAYLOAD)
    result = run_serpapi_flight_tool(
        from_location="Atlantis",
        to_location="Paris",
        departure_date="2024-10-01",
        api_key="fake-key",
        client=client,
    )

    assert result["status"] == "no_results"
    assert result["reason"] == "departure_city_not_resolved"


def test_serpapi_flight_api_error_degrades():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "Invalid API key. Expected key was not provided."})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = run_serpapi_flight_tool(
        from_location="London",
        to_location="Paris",
        departure_date="2024-10-01",
        api_key="bad-key",
        client=client,
    )

    assert result["status"] == "error"
    assert "Invalid API key" in result["message"]


def test_serpapi_flight_http_error_degrades():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = run_serpapi_flight_tool(
        from_location="London",
        to_location="Paris",
        departure_date="2024-10-01",
        api_key="fake-key",
        client=client,
    )

    assert result["status"] == "error"


# --- Hotel tool tests ---


def test_serpapi_hotel_parses_properties():
    client = _mock_hotel_client(_HOTELS_PAYLOAD)
    result = run_serpapi_hotel_tool(
        city="Paris",
        check_in_date="2024-10-01",
        check_out_date="2024-10-04",
        budget="medium",
        api_key="fake-key",
        client=client,
    )

    assert result["tool_name"] == "serpapi_hotel_tool"
    assert result["status"] == "ok"
    assert result["city"] == "Paris"
    assert result["budget_level"] == "medium"
    hotels = result["results"]
    assert len(hotels) == 2

    first = hotels[0]
    assert first["name"] == "Hotel Rivoli Paris"
    assert first["hotel_class"] == 4
    assert first["rating"] == 4.5
    assert first["reviews"] == 1200
    assert first["price_usd_per_night"] == 180.0
    assert first["price_usd_total"] == 540.0
    assert first["booking_link"].startswith("https://www.google.com/travel/hotels")
    assert first["image"] == "https://example.com/rivoli.jpg"
    assert first["check_in_date"] == "2024-10-01"

    assert hotels[1]["name"] == "Budget Inn"
    assert hotels[1]["hotel_class"] == 2
    assert hotels[1]["price_usd_per_night"] == 60.0


def test_serpapi_hotel_respects_limit():
    client = _mock_hotel_client(_HOTELS_PAYLOAD)
    result = run_serpapi_hotel_tool(
        city="Paris",
        check_in_date="2024-10-01",
        check_out_date="2024-10-04",
        api_key="fake-key",
        client=client,
        limit=1,
    )

    assert len(result["results"]) == 1


def test_serpapi_hotel_no_key_degrades_to_no_results():
    client = _mock_hotel_client(_HOTELS_PAYLOAD)
    result = run_serpapi_hotel_tool(
        city="Paris",
        check_in_date="2024-10-01",
        check_out_date="2024-10-04",
        api_key="",
        client=client,
    )

    assert result["status"] == "no_results"
    assert result["reason"] == "serpapi_key_missing"
    assert result["results"] == []


def test_serpapi_hotel_empty_properties_degrades():
    client = _mock_hotel_client({"properties": []})
    result = run_serpapi_hotel_tool(
        city="Paris",
        check_in_date="2024-10-01",
        check_out_date="2024-10-04",
        api_key="fake-key",
        client=client,
    )

    assert result["status"] == "no_results"
    assert result["reason"] == "no_properties_in_response"


def test_serpapi_hotel_api_error_degrades():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "Quota exceeded."})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = run_serpapi_hotel_tool(
        city="Paris",
        check_in_date="2024-10-01",
        check_out_date="2024-10-04",
        api_key="bad-key",
        client=client,
    )

    assert result["status"] == "error"
    assert "Quota exceeded" in result["message"]


def test_budget_to_class_mapping():
    assert _budget_to_class("luxury") == "4,5"
    assert _budget_to_class("medium") == "3,4"
    assert _budget_to_class("low") == "1,2,3"
    assert _budget_to_class("unknown") is None
