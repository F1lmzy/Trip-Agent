"""Tests for the orchestrator SerpAPI/mock fallback wiring and image attachment.

Asserts the iteration-3 contract:
  (a) with a SerpAPI key + ok response -> SerpAPI tool used, mock not called;
  (b) without a key -> mock tool used, SerpAPI not called;
  (c) SerpAPI no_results/error is surfaced instead of hidden by fake mock flights;
  (d) Wikimedia image_url is attached to attraction and hotel results.

All HTTP is mocked via httpx.MockTransport — no real network, no SerpAPI quota.
"""

from datetime import date, timedelta

import httpx

from app.agent.orchestrator import AgentServices
from app.agent.parser import ParsedRequest
from app.agent.tool_executor import attach_images, resolve_departure_date, run_flight, run_hotel
from app.tools.serpapi_flight_tool import run_serpapi_flight_tool
from app.tools.serpapi_hotel_tool import run_serpapi_hotel_tool
from tests.fakes import FakeImageClient


def _parsed(**kwargs) -> ParsedRequest:
    base = dict(raw_message="Plan a Paris trip from London, 3 days, medium budget.")
    base.update(kwargs)
    return ParsedRequest(**base)


def _serpapi_ok_hotel_client() -> httpx.Client:
    payload = {
        "properties": [
            {
                "name": "Hotel Rivoli Paris",
                "hotel_class": 4,
                "rating": 4.5,
                "reviews": 1200,
                "rate_per_night": {"extracted_lowest": 180.0},
                "total_rate": {"extracted_lowest": 540.0},
                "link": "https://www.google.com/travel/hotels/rivoli",
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert "google_hotels" in str(request.url)
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _serpapi_ok_flight_client() -> httpx.Client:
    payload = {
        "best_flights": [
            {
                "type": "one way",
                "price": 412,
                "total_duration": 130,
                "booking_link": "https://www.google.com/flights/ow/abc",
                "flights": [
                    {
                        "airline": "Air France",
                        "flight_number": "AF 1080",
                        "departure_airport": {"name": "London Heathrow", "id": "LHR", "time": "2024-10-01T08:00:00"},
                        "arrival_airport": {"name": "Charles de Gaulle", "id": "CDG", "time": "2024-10-01T10:10:00"},
                        "duration": 130,
                    }
                ],
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert "google_flights" in str(request.url)
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _serpapi_error_client(engine: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "Invalid API key."})

    return httpx.Client(transport=httpx.MockTransport(handler))


# --- Hotel wiring ---


def test_run_hotel_uses_serpapi_when_key_present_and_ok():
    parsed = _parsed(city="Paris", budget="medium", duration_days=3)
    services = AgentServices(
        serpapi_api_key="fake-key",
        serpapi_client=_serpapi_ok_hotel_client(),
        image_client=FakeImageClient(),
        use_environment=False,
    )

    result = run_hotel(parsed, services)

    assert result["tool_name"] == "serpapi_hotel_tool"
    assert result["status"] == "ok"
    assert result["results"][0]["name"] == "Hotel Rivoli Paris"


def test_run_hotel_falls_back_to_mock_when_key_absent():
    parsed = _parsed(city="Tokyo", budget="medium", duration_days=2)
    services = AgentServices(
        serpapi_api_key="",  # no key
        serpapi_client=_serpapi_ok_hotel_client(),  # would-be ok; must NOT be called
        image_client=FakeImageClient(),
        use_environment=False,
    )

    result = run_hotel(parsed, services)

    # Mock hotel tool returns tool_name "hotel_tool" (not serpapi).
    assert result["tool_name"] == "hotel_tool"
    # Tokyo has curated mock hotels in data/hotels.json.
    assert result["status"] == "ok"


def test_run_hotel_falls_back_to_mock_on_serpapi_error():
    parsed = _parsed(city="Tokyo", budget="medium", duration_days=2)
    services = AgentServices(
        serpapi_api_key="fake-key",
        serpapi_client=_serpapi_error_client("google_hotels"),
        image_client=FakeImageClient(),
        use_environment=False,
    )

    result = run_hotel(parsed, services)

    # SerpAPI errored -> fallback to mock.
    assert result["tool_name"] == "hotel_tool"
    assert result["status"] == "ok"


def test_run_hotel_falls_back_to_mock_on_serpapi_no_results(monkeypatch):
    parsed = _parsed(city="Tokyo", budget="medium", duration_days=2)

    def _no_results(*args, **kwargs):
        return {"tool_name": "serpapi_hotel_tool", "status": "no_results", "results": []}

    monkeypatch.setattr("app.agent.tool_executor.run_serpapi_hotel_tool", _no_results)
    services = AgentServices(
        serpapi_api_key="fake-key",
        image_client=FakeImageClient(),
        use_environment=False,
    )

    result = run_hotel(parsed, services)
    assert result["tool_name"] == "hotel_tool"  # fell back to mock


# --- Flight wiring ---


def test_run_flight_uses_serpapi_when_key_present_and_ok():
    parsed = _parsed(city="Paris", origin_city="London", budget="medium", duration_days=3)
    services = AgentServices(
        serpapi_api_key="fake-key",
        serpapi_client=_serpapi_ok_flight_client(),
        image_client=FakeImageClient(),
        use_environment=False,
    )

    result = run_flight(parsed, services)

    assert result["tool_name"] == "serpapi_flight_tool"
    assert result["status"] == "ok"
    dep = result["results"]["departure_flights"]
    assert dep and dep[0]["airline"] == "Air France"
    assert dep[0]["price"] == 412.0


def test_run_flight_falls_back_to_mock_when_key_absent():
    parsed = _parsed(city="Tokyo", origin_city="London", budget="medium", duration_days=2)
    services = AgentServices(
        serpapi_api_key="",
        serpapi_client=_serpapi_ok_flight_client(),  # must NOT be called
        image_client=FakeImageClient(),
        use_environment=False,
    )

    result = run_flight(parsed, services)

    assert result["tool_name"] == "flight_tool"  # mock
    assert result["status"] == "ok"


def test_run_flight_one_way_omits_return_flights():
    parsed = _parsed(city="Milan", origin_city="London", budget="medium", duration_days=2)
    parsed = parsed.model_copy(update={"trip_type": "one_way", "asks_for_flights": True})
    services = AgentServices(
        serpapi_api_key="",
        image_client=FakeImageClient(),
        use_environment=False,
    )

    result = run_flight(parsed, services)

    assert result["tool_name"] == "flight_tool"
    assert result["status"] == "ok"
    assert result["results"]["departure_flights"], "one-way still has outbound flights"
    assert result["results"]["return_flights"] == [], "one-way must not produce return flights"


def test_run_flight_round_trip_produces_return_flights():
    parsed = _parsed(city="Milan", origin_city="London", budget="medium", duration_days=2)
    parsed = parsed.model_copy(update={"trip_type": "round_trip", "asks_for_flights": True})
    services = AgentServices(
        serpapi_api_key="",
        image_client=FakeImageClient(),
        use_environment=False,
    )

    result = run_flight(parsed, services)

    assert result["results"]["return_flights"], "round-trip must produce return flights"


def test_resolve_departure_date_defaults_undated_flights_to_tomorrow():
    parsed = _parsed(city="Osaka", origin_city="Singapore", budget="luxury", duration_days=3)

    assert resolve_departure_date(parsed) == (date.today() + timedelta(days=1)).isoformat()


def test_run_flight_uses_explicit_date_range():
    # 'from June 21 to June 25' must honor those exact dates for outbound and
    # return flights, not duration_days-based computation.
    parsed = _parsed(city="Milan", origin_city="London", budget="medium", duration_days=2)
    parsed = parsed.model_copy(
        update={"asks_for_flights": True, "departure_date": "2026-06-21", "return_date": "2026-06-25"}
    )
    services = AgentServices(
        serpapi_api_key="",
        image_client=FakeImageClient(),
        use_environment=False,
    )

    result = run_flight(parsed, services)

    assert result["status"] == "ok"
    dep = result["results"]["departure_flights"][0]
    assert dep["departure"].startswith("2026-06-21"), dep["departure"]
    ret = result["results"]["return_flights"][0]
    assert ret["departure"].startswith("2026-06-25"), ret["departure"]


def test_run_flight_explicit_dates_override_duration():
    # Even with duration_days=2, an explicit return_date of June 25 wins over
    # the duration-based June 23.
    parsed = _parsed(city="Milan", origin_city="London", budget="medium", duration_days=2)
    parsed = parsed.model_copy(
        update={"asks_for_flights": True, "departure_date": "2026-06-21", "return_date": "2026-06-25"}
    )
    services = AgentServices(
        serpapi_api_key="",
        image_client=FakeImageClient(),
        use_environment=False,
    )

    result = run_flight(parsed, services)

    ret = result["results"]["return_flights"][0]
    assert ret["departure"].startswith("2026-06-25")


def test_run_flight_returns_serpapi_error_without_mock_fallback_when_key_present():
    parsed = _parsed(city="Tokyo", origin_city="London", budget="medium", duration_days=2)
    services = AgentServices(
        serpapi_api_key="fake-key",
        serpapi_client=_serpapi_error_client("google_flights"),
        image_client=FakeImageClient(),
        use_environment=False,
    )

    result = run_flight(parsed, services)
    assert result["tool_name"] == "serpapi_flight_tool"
    assert result["status"] == "error"
    assert result["results"]["departure_flights"] == []


# --- Image attachment ---


def _image_client_returning_url(url: str) -> httpx.Client:
    payload = {
        "query": {
            "pages": [
                {
                    "title": "File:Sample.jpg",
                    "index": 1,
                    "imageinfo": [{"thumburl": url, "mime": "image/jpeg"}],
                }
            ]
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_attach_images_adds_image_url_to_attractions():
    tool_outputs = {
        "attraction_rag_tool": {
            "status": "ok",
            "results": [
                {"name": "Senso-ji Temple", "description": "ancient temple"},
                {"name": "Shibuya Sky", "description": "viewing deck"},
            ],
        }
    }
    services = AgentServices(image_client=_image_client_returning_url("https://upload.wikimedia.org/a.jpg"), use_environment=False)

    attach_images(tool_outputs, services)

    assert tool_outputs["attraction_rag_tool"]["results"][0]["image_url"] == "https://upload.wikimedia.org/a.jpg"
    assert tool_outputs["attraction_rag_tool"]["results"][1]["image_url"] == "https://upload.wikimedia.org/a.jpg"


def test_attach_images_adds_image_url_to_hotels():
    tool_outputs = {
        "hotel_tool": {
            "status": "ok",
            "results": [{"name": "Hotel Rivoli Paris", "price_usd_per_night": 180.0}],
        }
    }
    services = AgentServices(image_client=_image_client_returning_url("https://upload.wikimedia.org/h.jpg"), use_environment=False)

    attach_images(tool_outputs, services)

    assert tool_outputs["hotel_tool"]["results"][0]["image_url"] == "https://upload.wikimedia.org/h.jpg"


def test_attach_images_sets_none_when_no_image_found():
    tool_outputs = {
        "attraction_rag_tool": {
            "status": "ok",
            "results": [{"name": "Obscure Place XYZ", "description": "unknown"}],
        }
    }
    # FakeImageClient returns an empty Commons payload -> None.
    services = AgentServices(image_client=FakeImageClient(), use_environment=False)

    attach_images(tool_outputs, services)

    assert tool_outputs["attraction_rag_tool"]["results"][0]["image_url"] is None


def test_attach_images_skips_items_without_name():
    tool_outputs = {
        "attraction_rag_tool": {
            "status": "ok",
            "results": [{"description": "no name here"}],
        }
    }
    services = AgentServices(image_client=FakeImageClient(), use_environment=False)

    attach_images(tool_outputs, services)

    # No name -> no image lookup, no image_url key added.
    assert "image_url" not in tool_outputs["attraction_rag_tool"]["results"][0]


def test_attach_images_does_not_overwrite_existing_image_url():
    tool_outputs = {
        "attraction_rag_tool": {
            "status": "ok",
            "results": [{"name": "Senso-ji Temple", "image_url": "https://already.set/img.jpg"}],
        }
    }
    services = AgentServices(image_client=_image_client_returning_url("https://upload.wikimedia.org/other.jpg"), use_environment=False)

    attach_images(tool_outputs, services)

    assert tool_outputs["attraction_rag_tool"]["results"][0]["image_url"] == "https://already.set/img.jpg"


def test_attach_images_reads_city_from_rag_metadata(monkeypatch):
    """RAG items store city in metadata sub-dict; attach_images should
    pass it to resolve_place_image for disambiguated image search."""
    captured_args = {}

    def fake_resolve(place_name, city=None, client=None, timeout=15.0):
        captured_args["place_name"] = place_name
        captured_args["city"] = city
        return "https://upload.wikimedia.org/mock.jpg"

    monkeypatch.setattr(
        "app.agent.tool_executor.resolve_place_image",
        fake_resolve,
    )

    tool_outputs = {
        "attraction_rag_tool": {
            "status": "ok",
            "results": [
                {
                    "name": "Old Town",
                    "description": "Old Town in Edinburgh.",
                    "metadata": {"city": "Edinburgh", "source": "external_wikivoyage"},
                }
            ],
        }
    }
    services = AgentServices(image_client=FakeImageClient(), use_environment=False)

    attach_images(tool_outputs, services)

    assert captured_args["place_name"] == "Old Town"
    assert captured_args["city"] == "Edinburgh"


def test_attach_images_top_level_city_takes_precedence_over_metadata(monkeypatch):
    captured_args = {}

    def fake_resolve(place_name, city=None, client=None, timeout=15.0):
        captured_args["place_name"] = place_name
        captured_args["city"] = city
        return "https://upload.wikimedia.org/mock.jpg"

    monkeypatch.setattr(
        "app.agent.tool_executor.resolve_place_image",
        fake_resolve,
    )

    tool_outputs = {
        "attraction_rag_tool": {
            "status": "ok",
            "results": [
                {
                    "name": "Old Town",
                    "description": "Old Town in Prague.",
                    "city": "Prague",
                    "metadata": {"city": "Edinburgh"},
                }
            ],
        }
    }
    services = AgentServices(image_client=FakeImageClient(), use_environment=False)

    attach_images(tool_outputs, services)

    # Top-level city wins over metadata.city
    assert captured_args["city"] == "Prague"
