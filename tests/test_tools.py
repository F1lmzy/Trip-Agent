import pytest
import httpx

from app.tools.budget_tool import run_budget_tool
from app.tools.hotel_tool import load_hotels, run_hotel_tool
from app.tools.weather_tool import run_weather_tool
from app.tools.web_search_tool import run_web_search_tool


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


def test_budget_tool_small_budget_normalized_to_low():
    assert run_budget_tool("small budget")["budget_level"] == "low"
    assert run_budget_tool("tight budget")["budget_level"] == "low"
    assert run_budget_tool("shoestring")["budget_level"] == "low"
    assert run_budget_tool("on a budget")["budget_level"] == "low"
    assert run_budget_tool("limited budget")["budget_level"] == "low"


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


def test_weather_tool_missing_api_key_returns_fallback():
    result = run_weather_tool("Tokyo", api_key=None)

    assert result["tool_name"] == "weather_tool"
    assert result["status"] == "fallback_missing_api_key"
    assert result["city"] == "Tokyo"
    assert result["source"] == "fallback"
    assert result["forecast"]


def test_weather_tool_calls_openweathermap_and_normalizes_forecast():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["q"] == "Tokyo"
        assert request.url.params["appid"] == "test-key"
        return httpx.Response(
            200,
            json={
                "list": [
                    {
                        "dt_txt": "2026-06-18 09:00:00",
                        "main": {"temp": 24.0, "feels_like": 24.5, "humidity": 70},
                        "weather": [{"main": "Clear"}],
                        "wind": {"speed": 2.0},
                    },
                    {
                        "dt_txt": "2026-06-18 12:00:00",
                        "main": {"temp": 26.0, "feels_like": 26.5, "humidity": 60},
                        "weather": [{"main": "Clear"}],
                        "wind": {"speed": 4.0},
                    },
                    {
                        "dt_txt": "2026-06-19 09:00:00",
                        "main": {"temp": 22.0, "feels_like": 22.2, "humidity": 80},
                        "weather": [{"main": "Clouds"}],
                        "wind": {"speed": 3.0},
                    },
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = run_weather_tool("Tokyo", api_key="test-key", client=client)

    assert result["status"] == "ok"
    assert result["source"] == "openweathermap"
    assert result["forecast"][0]["date"] == "2026-06-18"
    assert result["forecast"][0]["summary"] == "Clear"
    assert result["forecast"][0]["temperature_c"] == 25.0
    assert result["forecast"][0]["feels_like_c"] == 25.5
    assert result["forecast"][0]["humidity"] == 65.0
    assert result["forecast"][0]["wind_speed"] == 3.0
    assert result["forecast"][0]["outdoor_suitability"] == "good"


def test_weather_tool_api_error_returns_fallback():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "server error"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = run_weather_tool("Tokyo", api_key="test-key", client=client)

    assert result["status"] == "fallback_api_error"
    assert result["source"] == "fallback"
    assert result["forecast"]


def test_weather_tool_marks_rain_as_poor_outdoor_suitability():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "list": [
                    {
                        "dt_txt": "2026-06-18 09:00:00",
                        "main": {"temp": 24.0, "feels_like": 24.5, "humidity": 90},
                        "weather": [{"main": "Rain"}],
                        "wind": {"speed": 4.0},
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = run_weather_tool("Tokyo", api_key="test-key", client=client)

    assert result["forecast"][0]["outdoor_suitability"] == "poor"


class FakeDuckDuckGoTool:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def invoke(self, query: str):
        self.queries.append(query)
        return self.results


class FailingDuckDuckGoTool:
    def invoke(self, query: str):
        raise RuntimeError("search failed")


def test_web_search_tool_calls_langchain_duckduckgo_and_normalizes_results():
    search_tool = FakeDuckDuckGoTool(
        [
            {
                "title": "Tokyo event guide",
                "link": "https://example.com/tokyo",
                "snippet": "Recent Tokyo travel events.",
            },
            {
                "title": "Tokyo museum closure update",
                "href": "https://example.com/museums",
                "body": "Latest travel closure details.",
            },
        ]
    )

    result = run_web_search_tool("Tokyo", "current events", count=3, search_tool=search_tool)

    assert result["status"] == "ok"
    assert result["source"] == "duckduckgo_langchain"
    assert result["query"] == "Tokyo current events travel"
    assert search_tool.queries == ["Tokyo current events travel"]
    assert result["results"][0] == {
        "title": "Tokyo event guide",
        "url": "https://example.com/tokyo",
        "description": "Recent Tokyo travel events.",
    }
    assert result["results"][1]["url"] == "https://example.com/museums"


def test_web_search_tool_handles_langchain_tuple_output():
    search_tool = FakeDuckDuckGoTool(
        ([{"title": "Paris guide", "link": "https://example.com/paris", "snippet": "Paris update."}], [])
    )

    result = run_web_search_tool("Paris", "latest museum closures", search_tool=search_tool)

    assert result["status"] == "ok"
    assert result["results"] == [
        {"title": "Paris guide", "url": "https://example.com/paris", "description": "Paris update."}
    ]


def test_web_search_tool_search_error_returns_fallback():
    result = run_web_search_tool("Tokyo", "current events", search_tool=FailingDuckDuckGoTool())

    assert result["status"] == "fallback_search_error"
    assert result["source"] == "fallback"
    assert result["results"] == []
