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
from tests.fakes import FakeEmbedder, FakeImageClient, FakeSearchTool


def make_long_term_memory(tmp_path) -> LongTermMemory:
    return LongTermMemory(VectorStore(path=str(tmp_path / "memory"), embedder=FakeEmbedder()))


def make_services(tmp_path) -> AgentServices:
    rag_store = VectorStore(path=str(tmp_path / "rag"), embedder=FakeEmbedder())
    return AgentServices(
        attraction_rag_tool=AttractionRagTool(vector_store=rag_store),
        web_search_tool=FakeSearchTool(),
        image_client=FakeImageClient(),
        use_environment=False,
    )


def test_handle_chat_suggests_destinations_when_city_missing(tmp_path):
    memory = ShortTermMemory()
    user_memory = make_long_term_memory(tmp_path)

    response = handle_chat(
        ChatRequest(user_id="kavin", message="Plan me a trip. I like anime, food and photography. Medium budget."),
        memory=memory,
        user_memory=user_memory,
        services=make_services(tmp_path),
    )

    # No city given -> the DestinationRecommendationAgent suggests ranked
    # cities from RAG instead of asking a bare clarifying question.
    assert response.needs_clarification is False
    assert response.itinerary["status"] == "destination_suggestions"
    suggestions = response.itinerary["suggested_cities"]
    assert suggestions, "expected at least one suggested city"
    assert all("city" in s and "rationale" in s for s in suggestions)
    assert response.tools_used == ["destination_rag"]
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
        ChatRequest(user_id="kavin", message="Plan a 2-day trip to Singapore with food. Medium budget."),
        memory=ShortTermMemory(),
        user_memory=user_memory,
        services=make_services(tmp_path),
    )

    assert "I prefer vegetarian food" in response.memory_used


def test_handle_chat_uses_web_search_for_city_without_curated_rag(tmp_path):
    response = handle_chat(
        ChatRequest(user_id="kavin", message="Plan a 2-day trip to Kyoto with temples and food. Medium budget."),
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
        ChatRequest(user_id="kavin", message="Plan a Tokyo trip with current events and latest food spots. Medium budget."),
        memory=ShortTermMemory(),
        user_memory=make_long_term_memory(tmp_path),
        services=make_services(tmp_path),
    )

    assert "web_search_tool" in response.tools_used
    assert response.itinerary["day_1"]["morning"]


def test_handle_chat_executes_hotel_tool_when_requested(tmp_path):
    response = handle_chat(
        ChatRequest(user_id="kavin", message="Plan a 2-day Tokyo trip and suggest hotels. Medium budget."),
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


def test_current_budget_overrides_stale_long_term_budget_memory(tmp_path):
    user_memory = make_long_term_memory(tmp_path)
    user_memory.add_preference("kavin", "Budget preference: low")

    response = handle_chat(
        ChatRequest(user_id="kavin", message="Plan a 2-day trip to Tokyo with an unlimited budget."),
        memory=ShortTermMemory(),
        user_memory=user_memory,
        services=make_services(tmp_path),
    )

    assert "Budget preference: low" not in response.memory_used
    assert "Budget preference: luxury" in response.memory_used
    prefs = set(user_memory.get_preferences("kavin"))
    assert "Budget preference: low" not in prefs
    assert "Budget preference: luxury" in prefs


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


def test_openrouter_client_null_content_returns_fallback_not_string_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = call_openrouter([{"role": "user", "content": "hello"}], api_key="test-key", client=client)

    assert result["status"] == "fallback_empty_content"
    assert result["source"] == "fallback"
    assert result["content"] is None


def test_openrouter_client_uses_configured_timeout(monkeypatch):
    """The owned httpx client timeout should respect OPENROUTER_TIMEOUT_SECONDS."""
    monkeypatch.setenv("OPENROUTER_TIMEOUT_SECONDS", "90")
    get_settings.cache_clear()

    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    import httpx as _httpx

    original_client_init = _httpx.Client.__init__

    def spy_client_init(self, *args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        # Inject a mock transport so no real network call is made.
        kwargs["transport"] = httpx.MockTransport(handler)
        original_client_init(self, *args, **kwargs)

    monkeypatch.setattr(_httpx.Client, "__init__", spy_client_init)

    result = call_openrouter(
        [{"role": "user", "content": "Plan Tokyo"}],
        api_key="test-key",
        model="nvidia/nemotron-3-ultra",
    )

    assert result["status"] == "ok"
    timeout = captured.get("timeout")
    assert isinstance(timeout, _httpx.Timeout)
    assert timeout.read == 90.0

    get_settings.cache_clear()


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


def test_response_generator_rejects_openrouter_user_safety_status_content():
    parsed = parse_user_request("Plan a 3-day trip to Delhi with markets and history. Medium budget.")
    plan = create_trip_plan(parsed, rag_context_is_weak=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "User Safety: safe"}}]})

    response = generate_itinerary_response(
        parsed=parsed,
        plan=plan,
        tool_outputs={
            "attraction_rag_tool": {"results": [{"name": "Red Fort"}], "rag_trace": {"hop_1": [], "hop_2": []}},
            "budget_tool": {"budget_level": "medium"},
        },
        memory_used=[],
        api_key="test-key",
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert response.message != "User Safety: safe"
    assert "fallback_unparseable_content" in response.message
    assert response.itinerary["status"] == "generated_with_fallback_template"
    assert response.itinerary["day_1"]["morning"]


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


def test_response_generator_parses_markdown_table_itinerary():
    parsed = parse_user_request("Plan a 2-day trip to Kyoto with food.")
    plan = create_trip_plan(parsed, rag_context_is_weak=True)

    table_content = (
        "| Day | Morning | Afternoon | Evening |\n"
        "|---|---|---|---|\n"
        "| **Day 1: Shrines** | Visit **Fushimi Inari** early. | Head to **Nishiki Market** for food. | Stroll **Pontocho Alley**. |\n"
        "| **Day 2: Bamboo** | Go to **Arashiyama** for the bamboo grove. | Continue with temple sightseeing. | Budget dinner near Kyoto Station. |"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": table_content}}]})

    response = generate_itinerary_response(
        parsed=parsed,
        plan=plan,
        tool_outputs={"budget_tool": {"budget_level": "low"}},
        memory_used=[],
        api_key="test-key",
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert response.itinerary["status"] == "generated_with_openrouter"
    assert response.itinerary["day_1"]["morning"] == "Visit Fushimi Inari early"
    assert response.itinerary["day_1"]["afternoon"] == "Head to Nishiki Market for food"
    assert response.itinerary["day_1"]["evening"] == "Stroll Pontocho Alley"
    assert response.itinerary["day_2"]["morning"] == "Go to Arashiyama for the bamboo grove"
    assert response.itinerary["day_2"]["evening"] == "Budget dinner near Kyoto Station"


def test_response_generator_table_falls_back_to_label_parser_when_no_table():
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

    assert response.itinerary["day_1"]["morning"] == "Visit Tiananmen Square"


def test_response_generator_parses_header_without_colon_format():
    """The LLM often uses **Morning** as a standalone header with bullet content
    on the following lines, without a colon. This should be parsed correctly
    instead of falling back to the template text."""
    parsed = parse_user_request("Plan a 2-day trip to Stockholm with food.")
    plan = create_trip_plan(parsed, rag_context_is_weak=True)

    content = (
        "**Day 1**  \n"
        "**Morning**  \n"
        "- **Free Exploration**: Walk around Gamla Stan and Södermalm.  \n"
        "- **Food**: Grab affordable pastries from a local bakery.  \n\n"
        "**Afternoon**  \n"
        "- **Low-Cost Attraction**: Explore the Stockholm Archipelago.  \n"
        "- **Food**: Lunch at a market stall.  \n\n"
        "**Evening**  \n"
        "- **Free Activity**: Stroll along the waterfront at sunset.  \n\n"
        "**Day 2**  \n"
        "**Morning**  \n"
        "- **Free Activity**: Visit the Vasa Museum exterior.  \n"
        "- **Food**: Breakfast at a budget café.  \n\n"
        "**Afternoon**  \n"
        "- **Budget Attraction**: Free transit to Norrmalm for shopping.  \n\n"
        "**Evening**  \n"
        "- **Free Activity**: Relax in a park or attend a free event."
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    response = generate_itinerary_response(
        parsed=parsed,
        plan=plan,
        tool_outputs={"budget_tool": {"budget_level": "low"}},
        memory_used=[],
        api_key="test-key",
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert response.itinerary["status"] == "generated_with_openrouter"
    assert "Gamla Stan" in response.itinerary["day_1"]["morning"]
    assert "Archipelago" in response.itinerary["day_1"]["afternoon"]
    assert "waterfront" in response.itinerary["day_1"]["evening"]
    assert "Vasa Museum" in response.itinerary["day_2"]["morning"]
    assert "Norrmalm" in response.itinerary["day_2"]["afternoon"]
    assert "park" in response.itinerary["day_2"]["evening"].lower()


def test_attraction_names_skips_external_text_chunks():
    """Externally-ingested RAG results have raw text chunks as 'name' — those
    should be skipped so the fallback itinerary doesn't use sentence fragments
    as attraction names."""
    from app.agent.response_generator import _attraction_names

    tool_outputs = {
        "attraction_rag_tool": {
            "results": [
                {
                    "name": "Akihabara",
                    "description": "Akihabara in Tokyo. Anime and electronics district.",
                },
                {
                    "name": "Stockholm is Sweden's capital and largest city, with nearly",
                    "description": "Stockholm is Sweden's capital and largest city, with nearly 1 million residents.",
                },
                {
                    "name": "Tsukiji Outer Market",
                    "description": "Tsukiji Outer Market in Tokyo. Food-focused market area.",
                },
            ]
        }
    }

    names = _attraction_names(tool_outputs)

    assert "Akihabara" in names
    assert "Tsukiji Outer Market" in names
    assert "Stockholm is Sweden's capital and largest city, with nearly" not in names


def test_response_generator_fallback_matches_requested_duration_and_uses_specific_search_context():
    parsed = parse_user_request("Plan a 4-day trip to Hokkaido with food and nature. Medium budget.")
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


def test_response_generator_fallback_uses_nightlife_interest_without_repetitive_phrase():
    parsed = parse_user_request("Plan a 4 day trip to Osaka, I want to visit a lot of local bars, medium budget")
    plan = create_trip_plan(parsed, rag_context_is_weak=True)

    response = generate_itinerary_response(
        parsed=parsed,
        plan=plan,
        tool_outputs={
            "attraction_rag_tool": {
                "results": [
                    {"name": "Dotonbori"},
                    {"name": "Namba"},
                    {"name": "Shinsekai"},
                ],
                "rag_trace": {"hop_1": [], "hop_2": []},
            },
            "budget_tool": {"budget_level": "medium"},
        },
        memory_used=[],
        api_key="",
        model="test-model",
    )

    assert response.message != "None"
    assert "local bars" in response.itinerary["day_1"]["evening"]
    assert "keeping travel time clustered" not in str(response.itinerary)


def test_response_generator_includes_context_in_openrouter_prompt():
    parsed = parse_user_request("Plan a 2-day trip to Tokyo with anime and food. Medium budget.")
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
        assert "exactly three bullets" in payload
        content = "**Day 1**\n- **Morning**: Visit Akihabara.\n- **Afternoon**: Explore Ueno.\n- **Evening**: Eat ramen.\n\n**Day 2**\n- **Morning**: Visit Meiji Shrine.\n- **Afternoon**: Explore Harajuku.\n- **Evening**: Try izakaya food."
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

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

    assert "**Day 1**" in response.message
    assert response.itinerary["day_1"]["morning"]
    assert response.rag_trace == tool_outputs["attraction_rag_tool"]["rag_trace"]


def test_extract_time_slot_bullet_bold_dash_format():
    """_extract_time_slot handles '- **Morning** – content' format."""
    from app.agent.response_generator import _extract_time_slot

    section = (
        "- **Morning** – Arrive & settle in. Start with a private guided tour of Edinburgh Castle.\n"
        "- **Afternoon** – Lunch at Number 9. Follow with private walk up Arthur\'s Seat.\n"
        "- **Evening** – Dinner at The Kitchin. Finish with a cocktail at the Balmoral rooftop bar."
    )

    morning = _extract_time_slot(section, "morning")
    assert morning is not None
    assert "Edinburgh Castle" in morning
    assert "Arrive" in morning

    afternoon = _extract_time_slot(section, "afternoon")
    assert afternoon is not None
    assert "Number 9" in afternoon
    assert "Arthur" in afternoon

    evening = _extract_time_slot(section, "evening")
    assert evening is not None
    assert "Kitchin" in evening
    assert "Balmoral" in evening


def test_itinerary_from_llm_extracts_rich_descriptions_from_user_sample():
    """Full itinerary extraction using the user's actual OpenRouter output."""
    from app.agent.response_generator import _itinerary_from_llm_content

    raw_content = (
        "**Day 1 – 2026‑06‑21 (Rain, outdoor suitability: poor)**  \n"
        "- **Morning** – Arrive & settle in. Start with a private guided tour of **Edinburgh Castle** (indoor, climate‑controlled).  \n"
        "- **Afternoon** – Lunch at **Number 9** (fine‑dining, contemporary Scottish cuisine). Follow with a private walk up **Arthur’s Seat** (short, sheltered paths).  \n"
        "- **Evening** – Dinner at **The Kitchin** (Michelin‑starred, tasting menu). Finish with a cocktail at **The Balmoral Hotel’s** rooftop bar for a panoramic city view.  \n"
        "\n"
        "**Day 2 – 2026‑06‑22 (Clouds, outdoor suitability: fair)**  \n"
        "- **Morning** – Private transfer to the **National Museum of Scotland**; enjoy a curated, quiet tour.  \n"
        "- **Afternoon** – Explore the historic **Old Town** on a guided walking tour (including the Royal Mile). Lunch at **The Witchery by the Castle** (luxury ambience).  \n"
        "- **Evening** – Attend a private performance at the **Edinburgh Playhouse** or a VIP wine tasting at **The Scotch Whisky Experience**.  \n"
        "\n"
        "**Day 3 – 2026‑06‑23 (Clouds, outdoor suitability: fair)**  \n"
        "- **Morning** – Exclusive, early‑access visit to the **Royal Yacht Britannia** (private tour).  \n"
        "- **Afternoon** – Lunch at **The Gardener’s Cottage** (farm‑to‑table luxury). Then a private boat cruise on the **Firth of Forth** for sunset views of the harbor.  \n"
        "- **Evening** – Farewell dinner at **Restaurant Martin Wishart** (Michelin‑starred, modern Scottish cuisine). Conclude with a nightcap at **The Royal Mile Hotel’s** speakeasy lounge.  \n"
    )

    parsed = parse_user_request("Plan a 3-day trip to Edinburgh with unlimited budget.")
    itinerary = _itinerary_from_llm_content(parsed, raw_content, {})

    # All slots should contain rich descriptions, not fallback phrases.
    assert itinerary["status"] == "generated_with_openrouter"
    assert itinerary["city"] == "Edinburgh"

    # Day 1
    assert "Edinburgh Castle" in itinerary["day_1"]["morning"]
    assert "Number 9" in itinerary["day_1"]["afternoon"]
    assert "Kitchin" in itinerary["day_1"]["evening"]

    # Day 2
    assert "National Museum of Scotland" in itinerary["day_2"]["morning"]
    assert "Witchery" in itinerary["day_2"]["afternoon"]
    assert "Scotch Whisky" in itinerary["day_2"]["evening"]

    # Day 3
    assert "Royal Yacht Britannia" in itinerary["day_3"]["morning"]
    assert "Firth of Forth" in itinerary["day_3"]["afternoon"]
    assert "Martin Wishart" in itinerary["day_3"]["evening"]

    # Verify fallback phrases are NOT present
    day_1_text = str(itinerary["day_1"])
    assert "keeping travel time clustered" not in day_1_text
    assert "Start day" not in day_1_text


def test_itinerary_from_llm_stops_before_trailing_assumptions_section():
    from app.agent.response_generator import _itinerary_from_llm_content

    raw_content = (
        "**Day 1 – 2026-06-21 (Rain)**  \n"
        "- **Morning**: Visit the National Museum of Scotland.  \n"
        "- **Afternoon**: Explore the Scottish National Gallery of Modern Art.  \n"
        "- **Evening**: Dine at The Sheep Heid Inn.  \n\n"
        "**Day 2 – 2026-06-22 (Clouds, Fair Outdoor Suitability)**  \n"
        "- **Morning**: Tour Edinburgh Castle.  \n"
        "- **Afternoon**: Wander through Holyrood Palace gardens.  \n"
        "- **Evening**: Enjoy a sunset drink at The Balmoral.  \n\n"
        "**Day 3 – 2026-06-23 (Clouds, Fair Outdoor Suitability)**  \n"
        "- **Morning**: Explore the Old Town’s historic sites.  \n"
        "- **Afternoon**: Relax in Holyrood Park or visit the Royal Botanic Garden Edinburgh.  \n"
        "- **Evening**: Experience a traditional Scottish ceilidh or dinner at a fine dining spot like The Kitchin.  \n\n"
        "**Assumptions**:  \n"
        "- Dates default to the first three days of the weather forecast.  \n"
        "- Budget conflict resolved by prioritizing medium-level activities.\n"
    )

    parsed = parse_user_request("Plan a 3-day trip to Edinburgh with medium budget.")
    itinerary = _itinerary_from_llm_content(parsed, raw_content, {})

    evening = itinerary["day_3"]["evening"]
    assert "traditional Scottish ceilidh" in evening
    assert "The Kitchin" in evening
    assert "Assumptions" not in evening
    assert "Budget conflict" not in evening

