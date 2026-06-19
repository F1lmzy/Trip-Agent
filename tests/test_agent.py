import httpx

from app.agent.openrouter_client import call_openrouter
from app.agent.orchestrator import AgentServices, handle_chat
from app.agent.parser import parse_user_request
from app.agent.planner import create_trip_plan
from app.agent.response_generator import generate_itinerary_response
from app.config import get_settings
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.memory.vector_store import VectorStore
from app.tools.attraction_rag_tool import AttractionRagTool
from app.schemas import ChatRequest
from tests.fakes import FakeEmbedder, FakeSearchTool


def make_long_term_memory(tmp_path) -> LongTermMemory:
    return LongTermMemory(VectorStore(path=str(tmp_path / "memory"), embedder=FakeEmbedder()))


def make_services(tmp_path) -> AgentServices:
    rag_store = VectorStore(path=str(tmp_path / "rag"), embedder=FakeEmbedder())
    return AgentServices(
        attraction_rag_tool=AttractionRagTool(vector_store=rag_store),
        web_search_tool=FakeSearchTool(),
        use_environment=False,
    )


def test_handle_chat_returns_clarification_without_tools_when_city_missing():
    memory = ShortTermMemory()

    response = handle_chat(ChatRequest(user_id="kavin", message="Plan me a trip"), memory=memory)

    assert response.needs_clarification is True
    assert response.clarifying_question == "Which city would you like to visit?"
    assert response.tools_used == []
    assert memory.has_history("kavin") is True


def test_handle_chat_executes_required_tools_for_normal_trip(tmp_path):
    memory = ShortTermMemory()
    user_memory = make_long_term_memory(tmp_path)

    response = handle_chat(
        ChatRequest(user_id="kavin", message="Plan a 2-day trip to Tokyo. I like anime and food. Medium budget."),
        memory=memory,
        user_memory=user_memory,
        services=make_services(tmp_path),
    )

    assert response.needs_clarification is False
    assert response.itinerary["city"] == "Tokyo"
    assert response.itinerary["duration_days"] == 2
    assert response.itinerary["day_1"]["morning"]
    assert "attraction_rag_tool" in response.tools_used
    assert "weather_tool" in response.tools_used
    assert "budget_tool" in response.tools_used
    assert response.rag_trace["hop_1"]
    assert response.plan


def test_follow_up_uses_short_term_memory_when_history_exists(tmp_path):
    memory = ShortTermMemory()
    user_memory = make_long_term_memory(tmp_path)
    services = make_services(tmp_path)
    handle_chat(
        ChatRequest(user_id="kavin", message="Plan a 2-day trip to Tokyo."),
        memory=memory,
        user_memory=user_memory,
        services=services,
    )

    response = handle_chat(
        ChatRequest(user_id="kavin", message="Make it cheaper."),
        memory=memory,
        user_memory=user_memory,
        services=services,
    )

    assert response.needs_clarification is False
    assert response.itinerary["follow_up_intent"] == "cheaper"
    assert response.memory_used == ["Recent conversation history"]
    assert response.tools_used == ["budget_tool"]


def test_follow_up_without_history_asks_for_context(tmp_path):
    memory = ShortTermMemory()

    response = handle_chat(
        ChatRequest(user_id="new-user", message="Make it cheaper."),
        memory=memory,
        user_memory=make_long_term_memory(tmp_path),
    )

    assert response.needs_clarification is True
    assert response.tools_used == []
    assert "What trip should I update?" in response.message


def test_handle_chat_includes_long_term_memory_when_available(tmp_path):
    user_memory = make_long_term_memory(tmp_path)
    user_memory.add_preference("kavin", "I prefer vegetarian food")

    response = handle_chat(
        ChatRequest(user_id="kavin", message="Plan a 2-day trip to Singapore with food."),
        memory=ShortTermMemory(),
        user_memory=user_memory,
        services=make_services(tmp_path),
    )

    assert "I prefer vegetarian food" in response.memory_used


def test_handle_chat_uses_web_search_for_city_without_curated_rag(tmp_path):
    response = handle_chat(
        ChatRequest(user_id="kavin", message="Plan a 2-day trip to Kyoto with temples and food."),
        memory=ShortTermMemory(),
        user_memory=make_long_term_memory(tmp_path),
        services=make_services(tmp_path),
    )

    assert response.needs_clarification is False
    assert response.itinerary["city"] == "Kyoto"
    assert "web_search_tool" in response.tools_used
    assert "attraction_rag_tool" in response.tools_used


def test_handle_chat_executes_web_search_for_current_info(tmp_path):
    response = handle_chat(
        ChatRequest(user_id="kavin", message="Plan a Tokyo trip with current events and latest food spots."),
        memory=ShortTermMemory(),
        user_memory=make_long_term_memory(tmp_path),
        services=make_services(tmp_path),
    )

    assert "web_search_tool" in response.tools_used
    assert response.itinerary["day_1"]["morning"]


def test_handle_chat_executes_hotel_tool_when_requested(tmp_path):
    response = handle_chat(
        ChatRequest(user_id="kavin", message="Plan a 2-day Tokyo trip and suggest hotels."),
        memory=ShortTermMemory(),
        user_memory=make_long_term_memory(tmp_path),
        services=make_services(tmp_path),
    )

    assert "hotel_tool" in response.tools_used
    assert any("hotel" in note.lower() or "staying" in note.lower() for note in response.itinerary["notes"])


def test_handle_chat_executes_flight_tool_when_origin_and_flights_requested(tmp_path):
    response = handle_chat(
        ChatRequest(user_id="kavin", message="Plan a 2-day Tokyo trip flying from London. Medium budget."),
        memory=ShortTermMemory(),
        user_memory=make_long_term_memory(tmp_path),
        services=make_services(tmp_path),
    )

    assert "flight_tool" in response.tools_used
    assert response.itinerary["city"] == "Tokyo"
    assert any("flight" in note.lower() for note in response.itinerary["notes"])


def test_handle_chat_does_not_call_flight_tool_without_origin(tmp_path):
    response = handle_chat(
        ChatRequest(user_id="kavin", message="Plan a 2-day Tokyo trip with flights. Medium budget."),
        memory=ShortTermMemory(),
        user_memory=make_long_term_memory(tmp_path),
        services=make_services(tmp_path),
    )

    assert "flight_tool" not in response.tools_used


def test_handle_chat_saves_stable_preferences(tmp_path):
    user_memory = make_long_term_memory(tmp_path)

    handle_chat(
        ChatRequest(user_id="kavin", message="Plan Tokyo with anime and vegetarian food. Medium budget."),
        memory=ShortTermMemory(),
        user_memory=user_memory,
        services=make_services(tmp_path),
    )

    prefs = set(user_memory.get_preferences("kavin"))
    assert "Interest preference: anime" in prefs
    assert "Interest preference: food" in prefs
    assert "Dietary need: vegetarian" in prefs
    assert "Budget preference: medium" in prefs


def test_openrouter_client_missing_key_returns_fallback(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    get_settings.cache_clear()

    result = call_openrouter([{"role": "user", "content": "hello"}], api_key="", model="test-model")

    assert result["status"] == "fallback_missing_api_key"
    assert result["source"] == "fallback"
    assert result["model"] == "test-model"
    assert result["content"] is None


def test_openrouter_client_explicit_empty_key_disables_environment_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-env-key")
    get_settings.cache_clear()

    result = call_openrouter([{"role": "user", "content": "hello"}], api_key="", model="test-model")

    assert result["status"] == "fallback_missing_api_key"
    assert result["source"] == "fallback"
    assert result["content"] is None


def test_openrouter_client_calls_api_and_extracts_content():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = request.read().decode()
        assert "nvidia/nemotron-3-ultra" in payload
        assert "Plan Tokyo" in payload
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Here is your itinerary..."}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = call_openrouter(
        [{"role": "user", "content": "Plan Tokyo"}],
        api_key="test-key",
        model="nvidia/nemotron-3-ultra",
        client=client,
    )

    assert result["status"] == "ok"
    assert result["source"] == "openrouter"
    assert result["content"] == "Here is your itinerary..."


def test_openrouter_client_api_error_returns_fallback():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server error"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = call_openrouter([{"role": "user", "content": "hello"}], api_key="test-key", client=client)

    assert result["status"] == "fallback_api_error"
    assert result["source"] == "fallback"
    assert result["content"] is None


def test_response_generator_falls_back_without_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_TIMEOUT_SECONDS", "45")
    get_settings.cache_clear()
    parsed = parse_user_request("Plan a 2-day trip to Tokyo with anime and food.")
    plan = create_trip_plan(parsed)

    response = generate_itinerary_response(
        parsed=parsed,
        plan=plan,
        tool_outputs={
            "attraction_rag_tool": {"results": [{"name": "Akihabara"}], "rag_trace": {"hop_1": [], "hop_2": []}},
            "budget_tool": {"budget_level": "medium"},
        },
        memory_used=["I prefer vegetarian food"],
        api_key="",
        model="test-model",
    )

    assert response.itinerary["day_1"]["morning"]
    assert response.itinerary["day_2"]["evening"]
    assert response.tools_used == plan.selected_tools
    assert response.memory_used == ["I prefer vegetarian food"]
    assert "fallback_missing_api_key" in response.message


def test_response_generator_parses_bold_markdown_openrouter_slots():
    parsed = parse_user_request("Plan a 2-day trip to Beijing with food.")
    plan = create_trip_plan(parsed, rag_context_is_weak=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "**Day 1**\n- **Morning**: Explore **Wangfujing Snack Street**.\n- **Afternoon**: Visit **Temple of Heaven**.\n- **Evening**: Dine at **Dongzhimen Night Market**.\n\n**Day 2**\n- **Morning**: Walk **Beihai Park**.\n- **Afternoon**: Visit **Summer Palace**.\n- **Evening**: Try **Peking Duck**."
                        }
                    }
                ]
            },
        )

    response = generate_itinerary_response(
        parsed=parsed,
        plan=plan,
        tool_outputs={"budget_tool": {"budget_level": "medium"}},
        memory_used=[],
        api_key="test-key",
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert response.itinerary["status"] == "generated_with_openrouter"
    assert response.itinerary["day_1"]["morning"] == "Explore Wangfujing Snack Street"
    assert response.itinerary["day_1"]["afternoon"] == "Visit Temple of Heaven"
    assert response.itinerary["day_1"]["evening"] == "Dine at Dongzhimen Night Market"
    assert response.itinerary["day_2"]["evening"] == "Try Peking Duck"


def test_response_generator_structures_itinerary_from_openrouter_text():
    parsed = parse_user_request("Plan a 2-day trip to Beijing with food.")
    plan = create_trip_plan(parsed, rag_context_is_weak=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "**Day 1**\n- Morning: Visit Tiananmen Square. Afternoon: Explore Forbidden City. Evening: Eat at Wangfujing.\n\n**Day 2**\n- Morning: Walk the Summer Palace. Afternoon: Visit Temple of Heaven. Evening: Try hutong snacks."
                        }
                    }
                ]
            },
        )

    response = generate_itinerary_response(
        parsed=parsed,
        plan=plan,
        tool_outputs={"budget_tool": {"budget_level": "medium"}},
        memory_used=[],
        api_key="test-key",
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert response.itinerary["status"] == "generated_with_openrouter"
    assert response.itinerary["day_1"]["morning"] == "Visit Tiananmen Square"
    assert response.itinerary["day_1"]["afternoon"] == "Explore Forbidden City"
    assert response.itinerary["day_1"]["evening"] == "Eat at Wangfujing"
    assert response.itinerary["day_2"]["afternoon"] == "Visit Temple of Heaven"


def test_response_generator_fallback_matches_requested_duration_and_uses_specific_search_context():
    parsed = parse_user_request("Plan a 4-day trip to Hokkaido with food and nature.")
    plan = create_trip_plan(parsed, rag_context_is_weak=True)

    response = generate_itinerary_response(
        parsed=parsed,
        plan=plan,
        tool_outputs={
            "web_search_tool": {
                "results": [
                    {
                        "title": "3 Days in Hokkaido: The Best Short Trip Itinerary",
                        "url": "https://example.com",
                        "description": "Visit Sapporo Clock Tower, Otaru Canal, and Nijo Market for food.",
                    }
                ]
            },
            "budget_tool": {"budget_level": "medium"},
        },
        memory_used=[],
        api_key="",
        model="test-model",
    )

    assert response.itinerary["duration_days"] == 4
    assert "day_4" in response.itinerary
    assert "3 Days in Hokkaido" not in response.itinerary["day_1"]["morning"]
    assert "Sapporo Clock Tower" in response.itinerary["day_1"]["morning"]
    assert any("3 Days in Hokkaido" in note for note in response.itinerary["notes"])


def test_response_generator_includes_context_in_openrouter_prompt():
    parsed = parse_user_request("Plan a 2-day trip to Tokyo with anime and food.")
    plan = create_trip_plan(parsed)
    tool_outputs = {
        "attraction_rag_tool": {
            "results": [{"name": "Akihabara"}],
            "rag_trace": {"hop_1": [{"summary": "Tokyo overview"}], "hop_2": [{"summary": "Akihabara anime"}]},
        },
        "weather_tool": {"forecast": [{"summary": "Clear", "outdoor_suitability": "good"}]},
        "budget_tool": {"budget_level": "medium"},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read().decode()
        assert "Tokyo" in payload
        assert "Akihabara" in payload
        assert "Tokyo overview" in payload
        assert "I like museums" in payload
        assert "Generate a structured itinerary" in payload
        return httpx.Response(200, json={"choices": [{"message": {"content": "LLM itinerary"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = generate_itinerary_response(
        parsed=parsed,
        plan=plan,
        tool_outputs=tool_outputs,
        memory_used=["I like museums"],
        api_key="test-key",
        model="test-model",
        client=client,
    )

    assert response.message == "LLM itinerary"
    assert response.itinerary["day_1"]["morning"]
    assert response.rag_trace == tool_outputs["attraction_rag_tool"]["rag_trace"]
