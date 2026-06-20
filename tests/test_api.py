from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.agent.orchestrator import AgentServices
from app.memory.long_term import LongTermMemory
from app.memory.vector_store import VectorStore
from app.tools.attraction_rag_tool import AttractionRagTool
from tests.fakes import FakeEmbedder, FakeImageClient, FakeSearchTool


client = TestClient(app)


def install_test_long_term_memory(monkeypatch, tmp_path) -> LongTermMemory:
    test_memory = LongTermMemory(VectorStore(path=str(tmp_path / "memory"), embedder=FakeEmbedder()))
    monkeypatch.setattr(main_module, "long_term_memory", test_memory)
    return test_memory


def install_test_agent_services(monkeypatch, tmp_path) -> AgentServices:
    services = AgentServices(
        attraction_rag_tool=AttractionRagTool(VectorStore(path=str(tmp_path / "rag"), embedder=FakeEmbedder())),
        web_search_tool=FakeSearchTool(),
        image_client=FakeImageClient(),
        use_environment=False,
    )
    monkeypatch.setattr(main_module, "agent_services", services)
    return services


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["tools_available"] == 6
    assert body["mcp_endpoint"] == "/mcp"
    assert isinstance(body["openrouter_configured"], bool)
    assert isinstance(body["openweather_configured"], bool)


def test_index_returns_html():
    response = client.get("/")

    assert response.status_code == 200
    assert "Travel Planning Agent" in response.text
    assert "Tools used" in response.text
    assert "Planning steps" in response.text
    assert "RAG trace" in response.text


def test_chat_returns_response_shape(monkeypatch, tmp_path):
    install_test_long_term_memory(monkeypatch, tmp_path)
    install_test_agent_services(monkeypatch, tmp_path)

    response = client.post("/chat", json={"user_id": "api-shape-user", "message": "Plan Tokyo"})

    assert response.status_code == 200
    body = response.json()
    assert body["message"]
    assert body["itinerary"]["city"] == "Tokyo"
    assert body["itinerary"]["day_1"]["morning"]
    assert isinstance(body["tools_used"], list)
    assert "attraction_rag_tool" in body["tools_used"]
    assert "weather_tool" in body["tools_used"]
    assert "budget_tool" in body["tools_used"]
    assert body["rag_trace"]["hop_1"]
    assert isinstance(body["plan"], list)
    assert body["needs_clarification"] is False


def test_chat_suggests_destinations_when_city_missing(monkeypatch, tmp_path):
    install_test_long_term_memory(monkeypatch, tmp_path)
    install_test_agent_services(monkeypatch, tmp_path)

    response = client.post(
        "/chat",
        json={"user_id": "api-clarify-user", "message": "Plan me a trip. I like anime, food and photography. Medium budget."},
    )

    assert response.status_code == 200
    body = response.json()
    # No city given -> destination suggestions, not a clarifying question.
    assert body["needs_clarification"] is False
    assert body["tools_used"] == ["destination_rag"]
    assert body["itinerary"]["status"] == "destination_suggestions"
    assert body["itinerary"]["suggested_cities"], "expected at least one suggested city"


def test_memory_endpoints_store_get_and_clear_preferences(monkeypatch, tmp_path):
    install_test_long_term_memory(monkeypatch, tmp_path)
    user_id = "api-memory-user"

    add_response = client.post(f"/memory/{user_id}", json={"preference": "I like museums"})
    get_response = client.get(f"/memory/{user_id}")
    delete_response = client.delete(f"/memory/{user_id}")
    after_delete_response = client.get(f"/memory/{user_id}")

    assert add_response.status_code == 200
    assert add_response.json() == {"status": "saved"}
    assert get_response.status_code == 200
    assert get_response.json() == {"user_id": user_id, "memories": ["I like museums"]}
    assert delete_response.status_code == 200
    assert delete_response.json() == {"status": "memory cleared"}
    assert after_delete_response.status_code == 200
    assert after_delete_response.json() == {"user_id": user_id, "memories": []}


def test_api_tools_lists_all_registered_tools():
    response = client.get("/api/tools")

    assert response.status_code == 200
    body = response.json()
    tool_ids = {tool["id"] for tool in body["tools"]}
    assert tool_ids == {
        "attraction_rag_tool",
        "weather_tool",
        "budget_tool",
        "hotel_tool",
        "flight_tool",
        "web_search_tool",
    }
    assert body["total"] == 6
    assert all(tool["name"] for tool in body["tools"])


def test_chat_stream_emits_sse_events_with_result(monkeypatch, tmp_path):
    install_test_long_term_memory(monkeypatch, tmp_path)
    install_test_agent_services(monkeypatch, tmp_path)

    with client.stream(
        "POST", "/chat/stream", json={"user_id": "stream-user", "message": "Plan Tokyo"}
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        frames = [line for line in response.iter_lines() if line.startswith("data: ")]

    assert frames, "expected at least one SSE frame"
    import json

    events = [json.loads(frame[len("data: ") :]) for frame in frames]
    event_types = {event["type"] for event in events}
    assert "plan" in event_types
    assert "result" in event_types
    result_event = next(event for event in events if event["type"] == "result")
    assert result_event["response"]["itinerary"]["city"] == "Tokyo"


def test_chat_stream_emits_destination_suggestion_when_city_missing(monkeypatch, tmp_path):
    install_test_long_term_memory(monkeypatch, tmp_path)
    install_test_agent_services(monkeypatch, tmp_path)

    with client.stream(
        "POST",
        "/chat/stream",
        json={"user_id": "stream-clarify", "message": "Plan me a trip. I like anime, food and photography. Medium budget."},
    ) as response:
        frames = [line for line in response.iter_lines() if line.startswith("data: ")]

    import json

    events = [json.loads(frame[len("data: ") :]) for frame in frames]
    # The destination suggestion flows through the streaming endpoint. The
    # final result frame carries the suggestions. (Progressive per-agent
    # agent_start/agent_end events are added in the streaming iteration.)
    result = next(event for event in events if event["type"] == "result")
    assert result["response"]["itinerary"]["status"] == "destination_suggestions"
