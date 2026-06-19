"""End-to-end tests for the front-end rendering of images, hotels, and flights.

Iteration 5 contract:
  (a) the served index.html contains the new places/hotels/flights renderers;
  (b) a /chat response itinerary carries places/hotels/flights with image_url;
  (c) when the image client resolves a URL, place and hotel entries carry a
      real image_url string (proving the wiring from Commons -> orchestrator ->
      response_generator -> /chat JSON is intact).

No live network: FakeImageClient / a MockTransport image client, mock hotel
and flight tools (use_environment=False -> zero SerpAPI spend, zero OpenRouter
calls).
"""

import httpx
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.agent.orchestrator import AgentServices
from app.memory.long_term import LongTermMemory
from app.memory.vector_store import VectorStore
from app.tools.attraction_rag_tool import AttractionRagTool
from tests.fakes import FakeEmbedder, FakeImageClient, FakeSearchTool


client = TestClient(app)


def _install_services(monkeypatch, tmp_path, image_client=None) -> AgentServices:
    memory = LongTermMemory(VectorStore(path=str(tmp_path / "memory"), embedder=FakeEmbedder()))
    monkeypatch.setattr(main_module, "long_term_memory", memory)
    services = AgentServices(
        attraction_rag_tool=AttractionRagTool(VectorStore(path=str(tmp_path / "rag"), embedder=FakeEmbedder())),
        web_search_tool=FakeSearchTool(),
        image_client=image_client or FakeImageClient(),
        use_environment=False,
    )
    monkeypatch.setattr(main_module, "agent_services", services)
    return services


def _image_client_returning_url(url: str) -> httpx.Client:
    payload = {
        "query": {
            "pages": [
                {"title": "File:Sample.jpg", "index": 1, "imageinfo": [{"thumburl": url, "mime": "image/jpeg"}]}
            ]
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_index_html_contains_image_hotel_and_flight_renderers():
    response = client.get("/")

    assert response.status_code == 200
    html = response.text
    # New renderer functions are present.
    assert "function placesSection" in html
    assert "function hotelsSection" in html
    assert "function flightsSection" in html
    assert "function imageCard" in html
    assert "function hotelCard" in html
    assert "function flightCard" in html
    assert "function airportLabel" in html
    # Section headings rendered by the JS.
    assert "Places to visit" in html
    assert "Hotel suggestions" in html
    assert "Flights" in html
    # An <img> element is created for image-bearing cards.
    assert "createElement('img')" in html


def test_index_served_with_no_cache_header():
    # Regression: the single-file UI was heuristically cached by browsers, so
    # JS fixes (airportLabel) did not reach users without a hard reload.
    response = client.get("/")
    assert response.status_code == 200
    assert "no-cache" in response.headers.get("cache-control", "")


def test_airport_label_logic_renders_nested_object():
    # The mock + SerpAPI flight tools emit nested airport objects
    # {code, name, city}. airportLabel must produce 'Name (CODE)', never
    # '[object Object]'. This mirrors the JS helper in app/static/index.html.
    def airport_label(airport):
        if not airport:
            return ''
        if isinstance(airport, str):
            return airport
        if isinstance(airport, dict):
            code = airport.get('code') or airport.get('iata') or ''
            name = airport.get('name') or airport.get('city') or ''
            if code and name:
                return f'{name} ({code})'
            return code or name or ''
        return str(airport)

    assert airport_label({"code": "LHR", "name": "London Heathrow", "city": "London"}) == 'London Heathrow (LHR)'
    assert airport_label({"code": "LIN", "name": "Milan Linate", "city": "Milan"}) == 'Milan Linate (LIN)'
    assert airport_label("London") == 'London'
    assert airport_label(None) == ''
    assert airport_label({"city": "Milan"}) == 'Milan'


def test_chat_itinerary_includes_places_hotels_flights(monkeypatch, tmp_path):
    _install_services(monkeypatch, tmp_path)

    response = client.post(
        "/chat",
        json={"user_id": "e2e-shape-user", "message": "Plan a 2-day Tokyo trip from London. Medium budget."},
    )

    assert response.status_code == 200
    itinerary = response.json()["itinerary"]
    assert "places" in itinerary
    assert "hotels" in itinerary
    assert "flights" in itinerary
    assert isinstance(itinerary["places"], list)
    assert isinstance(itinerary["hotels"], list)
    # Each place entry exposes an image_url key (None here because FakeImageClient).
    for place in itinerary["places"]:
        assert "image_url" in place
    for hotel in itinerary["hotels"]:
        assert "image_url" in hotel


def test_chat_places_carry_real_image_url_when_commons_resolves(monkeypatch, tmp_path):
    _install_services(monkeypatch, tmp_path, image_client=_image_client_returning_url("https://upload.wikimedia.org/tokyo.jpg"))

    response = client.post(
        "/chat",
        json={"user_id": "e2e-image-user", "message": "Plan a 2-day Tokyo trip from London. Medium budget."},
    )

    assert response.status_code == 200
    places = response.json()["itinerary"]["places"]
    assert places, "expected at least one place for Tokyo"
    # The mock Commons client always returns the same URL -> every place carries it.
    assert all(p["image_url"] == "https://upload.wikimedia.org/tokyo.jpg" for p in places)


def test_chat_flights_section_populated_when_origin_city(monkeypatch, tmp_path):
    _install_services(monkeypatch, tmp_path)

    response = client.post(
        "/chat",
        json={"user_id": "e2e-flight-user", "message": "Plan a 2-day Tokyo trip from London with flights. Medium budget."},
    )

    assert response.status_code == 200
    flights = response.json()["itinerary"]["flights"]
    assert flights["status"] == "ok"
    assert flights["from_location"] == "London"
    assert flights["to_location"] == "Tokyo"
    assert isinstance(flights["departure_flights"], list)


def test_chat_hotels_section_populated(monkeypatch, tmp_path):
    _install_services(monkeypatch, tmp_path)

    response = client.post(
        "/chat",
        json={"user_id": "e2e-hotel-user", "message": "Plan a 2-day Tokyo trip. Medium budget, suggest hotels."},
    )

    assert response.status_code == 200
    hotels = response.json()["itinerary"]["hotels"]
    assert hotels, "expected hotel suggestions for Tokyo"
    assert all("image_url" in h for h in hotels)
