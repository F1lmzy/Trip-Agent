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


def test_destination_agent_seeds_city_docs_lazily(tmp_path):
    # A fresh, empty store must still yield suggestions because the agent seeds
    # curated city overviews from app/data/city_docs/*.md (local files, no net).
    services = _make_services(tmp_path)
    ctx = _ctx(
        "I like museums and food, medium budget, 2 days",
        services=services,
    )

    response = DestinationRecommendationAgent().run(ctx)

    assert response.itinerary["suggested_cities"], "lazy seed should populate city docs"


def test_destination_agent_with_no_interests_still_suggests(tmp_path):
    services = _make_services(tmp_path)
    ctx = _ctx("Plan a 2-day trip, medium budget.", services=services)
    assert ctx.parsed.interests == []

    response = DestinationRecommendationAgent().run(ctx)

    # Generic query still returns sensible defaults instead of a dead-end.
    assert response.itinerary["suggested_cities"]
