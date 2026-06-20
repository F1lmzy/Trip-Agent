"""Tests for the multi-agent scaffold (iteration 1).

Asserts the structural refactor preserved behavior exactly:
- supervisor routes city-None -> clarification (CustomerQueryAgent)
- supervisor routes city-present -> itinerary (ItineraryAgent)
- handle_chat still returns the same ChatResponse shape
- no live network is used (use_environment=False + FakeImageClient)
- agents emit agent_start/agent_end events on the context

These tests deliberately exercise the scaffold with no behavior change. The
destination-recommendation branch (city-None -> suggestions) is added in a
later iteration and tested there.
"""

import httpx

from app.agent.orchestrator import AgentServices, handle_chat
from app.agent.parser import parse_user_request
from app.agent.planner import create_trip_plan
from app.agents import (
    AgentContext,
    CustomerQueryAgent,
    DestinationRecommendationAgent,
    ItineraryAgent,
    Supervisor,
)
from app.agents.base import Agent
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.memory.vector_store import VectorStore
from app.schemas import ChatRequest
from app.tools.attraction_rag_tool import AttractionRagTool
from tests.fakes import FakeEmbedder, FakeImageClient, FakeSearchTool


def _make_services(tmp_path) -> AgentServices:
    """Build AgentServices with a FakeEmbedder-backed RAG store (no live network).

    Mirrors the tests/test_agent.py make_services pattern so the itinerary
    path never calls the real OpenRouter embeddings API.
    """
    rag_store = VectorStore(path=str(tmp_path / "rag"), embedder=FakeEmbedder())
    return AgentServices(
        attraction_rag_tool=AttractionRagTool(vector_store=rag_store),
        web_search_tool=FakeSearchTool(),
        image_client=FakeImageClient(),
        use_environment=False,
    )


def _make_memories(tmp_path):
    memory = ShortTermMemory()
    user_memory = LongTermMemory(VectorStore(path=str(tmp_path / "memory"), embedder=FakeEmbedder()))
    return memory, user_memory


def _ctx(message: str, services: AgentServices | None = None) -> AgentContext:
    services = services or AgentServices(image_client=FakeImageClient(), use_environment=False)
    parsed = parse_user_request(message)
    plan = create_trip_plan(parsed)
    return AgentContext(
        parsed=parsed,
        plan=plan,
        services=services,
        memory=None,  # type: ignore[arg-value] - not needed for routing
        user_memory=None,  # type: ignore[arg-value]
        user_id="test-user",
    )


def test_supervisor_routes_city_none_to_destination_agent():
    ctx = _ctx("I like anime, food and photography, medium budget, 2 days")
    assert ctx.plan.needs_clarification is True  # planner still flags it

    agent = Supervisor().route(ctx)

    # ...but the supervisor now routes to the destination agent (suggestions)
    # instead of the customer-query clarification agent.
    assert isinstance(agent, DestinationRecommendationAgent)


def test_supervisor_routes_city_present_to_itinerary_agent():
    ctx = _ctx("Plan a 2-day trip to Tokyo. I like anime and food. Medium budget.")
    assert ctx.plan.needs_clarification is False

    agent = Supervisor().route(ctx)

    assert isinstance(agent, ItineraryAgent)


def test_customer_query_agent_returns_clarification_response():
    ctx = _ctx("I like anime, food and photography, medium budget, 2 days")

    response = CustomerQueryAgent().run(ctx)

    assert response.needs_clarification is True
    assert "Which city" in (response.clarifying_question or "")
    assert response.itinerary == {}
    assert response.tools_used == []


def test_itinerary_agent_produces_itinerary_matching_parsed_city(tmp_path):
    # No live OpenRouter: FakeEmbedder backs the RAG store and use_environment=False
    # makes the LLM api_key resolve to '' which forces the fallback itinerary path.
    services = _make_services(tmp_path)
    memory, user_memory = _make_memories(tmp_path)
    ctx = _ctx(
        "Plan a 2-day trip to Tokyo. I like anime and food. Medium budget.",
        services=services,
    )
    ctx.memory = memory
    ctx.user_memory = user_memory

    response = ItineraryAgent().run(ctx)

    assert response.needs_clarification is False
    assert response.itinerary.get("city") == "Tokyo"


def test_handle_chat_returns_same_shape_for_tokyo_request(tmp_path):
    response = handle_chat(
        ChatRequest(user_id="test-user", message="Plan a 2-day trip to Tokyo. I like anime and food. Medium budget."),
        services=_make_services(tmp_path),
    )

    # ChatResponse schema fields are unchanged.
    assert response.itinerary.get("city") == "Tokyo"
    assert isinstance(response.plan, list)
    assert isinstance(response.tools_used, list)
    assert response.needs_clarification is False


def test_handle_chat_city_none_now_suggests_destinations(tmp_path):
    # Iteration 2: city-None now returns ranked suggestions (not a clarifying
    # question). Pin the new contract so a future regression to a bare
    # clarifying question is caught.
    response = handle_chat(
        ChatRequest(
            user_id="test-user",
            message="I like anime, food and photography, medium budget, 2 days",
        ),
        services=_make_services(tmp_path),
    )

    assert response.needs_clarification is False
    assert response.itinerary["status"] == "destination_suggestions"
    assert response.itinerary["suggested_cities"]
    assert response.tools_used == ["destination_rag"]


def test_agents_emit_start_and_end_events():
    ctx = _ctx("I like anime, food and photography, medium budget, 2 days")

    CustomerQueryAgent().run(ctx)

    event_types = [e["type"] for e in ctx.events]
    assert "agent_start" in event_types
    assert "agent_end" in event_types
    assert any(e.get("agent") == "CustomerQueryAgent" for e in ctx.events)


def test_event_emitter_callback_is_invoked_when_provided():
    ctx = _ctx("I like anime, food and photography, medium budget, 2 days")
    received: list[dict] = []
    ctx.event_emitter = received.append

    CustomerQueryAgent().run(ctx)

    assert received  # emitter received at least the agent_start/agent_end events
    assert any(e["type"] == "agent_start" for e in received)


def test_supervisor_run_returns_chat_response(tmp_path):
    ctx = _ctx(
        "I like anime, food and photography, medium budget, 2 days",
        services=_make_services(tmp_path),
    )

    response = Supervisor().run(ctx)

    assert response.itinerary["status"] == "destination_suggestions"


def test_agent_base_is_abstract():
    # The base Agent class raises NotImplementedError when run directly.
    ctx = _ctx("I like anime, food and photography, medium budget, 2 days")
    try:
        Agent().run(ctx)
    except NotImplementedError:
        return
    raise AssertionError("Base Agent.run should raise NotImplementedError")


# --- DestinationRecommendationAgent (iteration 2) ---


def test_destination_agent_suggests_ranked_cities(tmp_path):
    services = _make_services(tmp_path)
    ctx = _ctx(
        "I like anime, food and photography, medium budget, 2 days",
        services=services,
    )

    response = DestinationRecommendationAgent().run(ctx)

    assert response.needs_clarification is False
    assert response.itinerary["status"] == "destination_suggestions"
    suggestions = response.itinerary["suggested_cities"]
    assert suggestions, "expected at least one suggested city"
    # Every suggestion has a city name and a rationale.
    assert all("city" in s and "rationale" in s and "match_score" in s for s in suggestions)
    # Tokyo's curated overview mentions anime/akihabara + food + photography,
    # so it must rank within the top suggestions for this query.
    top_cities = [s["city"].lower() for s in suggestions]
    assert "tokyo" in top_cities, top_cities


def test_destination_agent_excludes_origin_city(tmp_path):
    services = _make_services(tmp_path)
    ctx = _ctx(
        "I like anime, food and photography, medium budget, 2 days. Flying from Singapore.",
        services=services,
    )
    assert ctx.parsed.origin_city == "Singapore"
    assert ctx.parsed.city is None  # parser must not treat the origin as destination

    response = DestinationRecommendationAgent().run(ctx)

    suggestions = response.itinerary["suggested_cities"]
    suggested = [s["city"].lower() for s in suggestions]
    assert "singapore" not in suggested, suggested


def test_handle_chat_origin_only_routes_to_destination_suggestions(tmp_path):
    # Regression guard: a user who names only an origin ("flying from Singapore")
    # with no destination must get city suggestions, NOT an itinerary for their
    # origin city. This is the "picks one place" complaint fixed.
    response = handle_chat(
        ChatRequest(
            user_id="test-user",
            message="I like anime, food and photography, medium budget, 2 days. Flying from Singapore.",
        ),
        services=_make_services(tmp_path),
    )

    assert response.itinerary["status"] == "destination_suggestions"
    suggested = [s["city"].lower() for s in response.itinerary["suggested_cities"]]
    assert "singapore" not in suggested
    assert response.needs_clarification is False


def test_destination_agent_message_lists_cities(tmp_path):
    services = _make_services(tmp_path)
    ctx = _ctx(
        "I like anime, food and photography, medium budget, 2 days",
        services=services,
    )

    response = DestinationRecommendationAgent().run(ctx)

    assert "destinations worth considering" in response.message
    # The message invites the user to pick one so the loop can continue.
    assert "Which one sounds good" in response.message


def test_destination_agent_returns_suggestions_via_handle_chat(tmp_path):
    response = handle_chat(
        ChatRequest(
            user_id="test-user",
            message="I like anime, food and photography, medium budget, 2 days",
        ),
        services=_make_services(tmp_path),
    )

    assert response.itinerary["status"] == "destination_suggestions"
    assert response.itinerary["suggested_cities"]
    assert response.tools_used == ["destination_rag"]


def test_destination_agent_uses_broader_catalog_not_only_five_curated_cities(tmp_path):
    services = _make_services(tmp_path)
    ctx = _ctx(
        "I want beaches, nature, wellness and a relaxed trip, medium budget",
        services=services,
    )

    response = DestinationRecommendationAgent().run(ctx)

    cities = [s["city"] for s in response.itinerary["suggested_cities"]]
    assert cities[0] == "Bali", cities
    assert any(city in cities for city in ["Barcelona", "Lisbon", "Dubai"]), cities
    # Regression: before the catalog, the agent could only return these five.
    assert set(cities) != {"Tokyo", "Paris", "Singapore", "New York", "Mumbai"}


def test_destination_agent_understands_regional_theme_asian_history(tmp_path):
    search = FakeSearchTool(
        results=[
            {
                "title": "Best Asian history destinations: Siem Reap, Beijing and Xi'an",
                "link": "https://example.com/asian-history",
                "snippet": "Siem Reap is useful for Angkor temples, Beijing for imperial history, and Xi'an for the Terracotta Army.",
            },
            {
                "title": "Historic Asia city breaks",
                "link": "https://example.com/asia-cities",
                "snippet": "Hanoi, Kyoto and Ayutthaya are popular choices for travelers interested in Asian history.",
            },
        ]
    )
    services = _make_services(tmp_path)
    services.web_search_tool = search
    ctx = _ctx("Plan a 2-day trip, I like asian history", services=services)

    response = DestinationRecommendationAgent().run(ctx)

    cities = [s["city"] for s in response.itinerary["suggested_cities"]]
    assert cities, "expected Asian history suggestions"
    assert all(city not in cities for city in ["Paris", "Barcelona", "Rome"]), cities
    # These are discovered from web snippets, not hardcoded in the local catalog.
    assert any(city in cities for city in ["Siem Reap", "Beijing", "Xi'an", "Hanoi", "Ayutthaya"]), cities
    assert any(s.get("source") == "web" for s in response.itinerary["suggested_cities"])
    assert search.queries, "history theme should use the web booster"
    assert "asian" in search.queries[0]
    assert "history" in search.queries[0]


def test_destination_agent_rejects_generic_title_case_web_phrases(tmp_path):
    search = FakeSearchTool(
        results=[
            {
                "title": "Amazing Historic Holidays Where History and Modernity Collide - Touripia",
                "link": "https://example.com/noisy",
                "snippet": "Every location offers iconic landmarks and unique attractions steeped in history. Perfect for history buffs or anyone looking for the best historic holidays.",
            },
            {
                "title": "The History of Ottoman Empire: A Global Power",
                "link": "https://example.com/ottoman",
                "snippet": "Travel Guides and Tips. Wildlife Safaris: The Best Destinations for Animal Encounters.",
            },
            {
                "title": "United Kingdom historic cities",
                "link": "https://example.com/uk",
                "snippet": "The United Kingdom is home to an incredible array of cities, each with its unique blend of history and modernity. From the capital city of London to the old streets of York.",
            },
        ]
    )
    services = _make_services(tmp_path)
    services.web_search_tool = search
    ctx = _ctx("Plan a 2-day trip, I like western history", services=services)

    response = DestinationRecommendationAgent().run(ctx)

    cities = [s["city"] for s in response.itinerary["suggested_cities"]]
    assert not any(
        city in cities for city in ["Amazing", "Modernity Collide", "Touripia", "Ottoman Empire", "Global Power", "Animal Encounters"]
    ), cities
    assert any(city in cities for city in ["London", "Rome", "Paris", "Barcelona", "Prague"]), cities


def test_destination_agent_rejects_country_region_and_site_names_from_web_results(tmp_path):
    search = FakeSearchTool(
        results=[
            {
                "title": "Which Asian destination should I pick? - Mumsnet",
                "link": "https://example.com/forum",
                "snippet": "23 Sept 2025 · Vietnam has some treasures as well as Cambodia. Taiwan was fabulous. South Korea has many gems as well as Japan. Sri Lanka is on my list for SE Asia.",
            },
            {
                "title": "Best Southeast Asian history trips",
                "link": "https://example.com/asia",
                "snippet": "Southeast Asian routes often include countries rather than cities in short forum answers.",
            },
        ]
    )
    services = _make_services(tmp_path)
    services.web_search_tool = search
    ctx = _ctx("Plan a 2-day trip, I like asian history", services=services)

    response = DestinationRecommendationAgent().run(ctx)

    cities = [s["city"] for s in response.itinerary["suggested_cities"]]
    assert not any(
        city in cities for city in ["Sri Lanka", "Southeast Asian", "SE Asia", "Mumsnet", "Which Asian", "Vietnam", "Cambodia", "Taiwan", "South Korea", "Japan"]
    ), cities
    assert any(city in cities for city in ["Kyoto", "Seoul", "Tokyo", "Mumbai", "Bangkok"]), cities


def test_destination_agent_rejects_dates_and_sentence_words_from_web_results(tmp_path):
    search = FakeSearchTool(
        results=[
            {
                "title": "Athens Comes to Life for Western History",
                "link": "https://example.com/bad-history",
                "snippet": "March 9, 2026 - Egyptian pharaohs believed that once they passed away, they’d become Gods in the afterlife, so they built massive temples and grand pyramids.",
            }
        ]
    )
    services = _make_services(tmp_path)
    services.web_search_tool = search
    ctx = _ctx("Plan a 2-day trip, I like western history", services=services)

    response = DestinationRecommendationAgent().run(ctx)

    cities = [s["city"] for s in response.itinerary["suggested_cities"]]
    assert not any(city in cities for city in ["March", "Comes", "Life", "Egyptian", "Gods", "Athens Comes"]), cities
    # The agent should fall back to clean catalog-backed Western-history cities
    # instead of emitting garbage web phrases.
    assert any(city in cities for city in ["Rome", "Paris", "Barcelona", "London", "Prague"]), cities


def test_destination_agent_rejects_generic_web_phrases_for_western_history(tmp_path):
    search = FakeSearchTool(
        results=[
            {
                "title": "History Buffs and The Best Travel Destinations For History",
                "link": "https://example.com/bad",
                "snippet": "Model Desac says the best travel destinations for history buffs provide an immersive experience.",
            },
            {
                "title": "Western history city breaks: Rome, Prague and London",
                "link": "https://example.com/western-history",
                "snippet": "Rome, Prague and London are useful bases for Western history, museums and architecture.",
            },
        ]
    )
    services = _make_services(tmp_path)
    services.web_search_tool = search
    ctx = _ctx("Plan a 2-day trip, I like western history", services=services)

    response = DestinationRecommendationAgent().run(ctx)

    cities = [s["city"] for s in response.itinerary["suggested_cities"]]
    assert not any(city in cities for city in ["History Buffs", "Model Desac", "The", "The Best Travel", "Destinations For History"]), cities
    assert any(city in cities for city in ["Rome", "Prague", "London"]), cities
    assert response.message.startswith("Based on what you're looking for")


def test_destination_agent_web_boost_uses_search_results_for_recommendation_prompts(tmp_path):
    search = FakeSearchTool(
        results=[
            {
                "title": "Best affordable food destinations: Lisbon and Naples",
                "link": "https://example.com/food",
                "snippet": "Lisbon and Naples are often recommended for food, budget hotels, and coastal city breaks.",
            },
            {
                "title": "Why Prague is good for nightlife and architecture",
                "link": "https://example.com/prague",
                "snippet": "Prague is a cheap destination for beer, history, nightlife, and architecture.",
            },
        ]
    )
    services = _make_services(tmp_path)
    services.web_search_tool = search
    ctx = _ctx("Recommend the best cheap food destinations with nightlife", services=services)

    response = DestinationRecommendationAgent().run(ctx)

    assert search.queries, "expected the web search booster to run"
    cities = [s["city"] for s in response.itinerary["suggested_cities"]]
    assert cities[0] in {"Lisbon", "Naples", "Prague"}, cities
    assert any(s["source"] == "catalog+web" for s in response.itinerary["suggested_cities"])
    assert any("Recent web results" in s["rationale"] for s in response.itinerary["suggested_cities"])


def test_destination_agent_web_searches_for_plain_theme_prompt(tmp_path):
    search = FakeSearchTool()
    services = _make_services(tmp_path)
    services.web_search_tool = search
    ctx = _ctx("anime, gaming, food and photography, medium budget", services=services)

    DestinationRecommendationAgent().run(ctx)

    assert search.queries, "theme prompts should use the web booster"
    assert "anime" in search.queries[0]
    assert "photography" in search.queries[0]


def test_destination_agent_web_searches_for_unparsed_theme_terms(tmp_path):
    search = FakeSearchTool()
    services = _make_services(tmp_path)
    services.web_search_tool = search
    ctx = _ctx("wellness, beer and cycling destinations", services=services)

    DestinationRecommendationAgent().run(ctx)

    assert search.queries, "raw theme terms should also trigger the web booster"


def test_destination_agent_varies_results_by_interest(tmp_path):
    services = _make_services(tmp_path)
    anime = DestinationRecommendationAgent().run(
        _ctx("anime, gaming, food and photography, medium budget", services=services)
    )
    beaches = DestinationRecommendationAgent().run(
        _ctx("beaches, wellness, nature and relaxed cafes, medium budget", services=services)
    )

    anime_cities = [s["city"] for s in anime.itinerary["suggested_cities"]]
    beach_cities = [s["city"] for s in beaches.itinerary["suggested_cities"]]
    assert anime_cities != beach_cities
    assert anime_cities[0] in {"Tokyo", "Seoul"}
    assert beach_cities[0] == "Bali"


def test_destination_agent_with_no_interests_still_suggests(tmp_path):
    services = _make_services(tmp_path)
    ctx = _ctx("Plan a 2-day trip, medium budget.", services=services)
    assert ctx.parsed.interests == []

    response = DestinationRecommendationAgent().run(ctx)

    # Generic query still returns sensible defaults instead of a dead-end.
    assert response.itinerary["suggested_cities"]
