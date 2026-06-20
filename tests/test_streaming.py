"""Tests for progressive SSE streaming (iteration 3).

Asserts /chat/stream emits truly progressive per-agent and per-tool events
with an "agent" field as work happens, plus the existing plan/result/clarification
frames. All HTTP is mocked/offline: use_environment=False + FakeEmbedder-backed
RAG store so no live OpenRouter/SerpAPI/Wikimedia calls happen.
"""

import json

import app.main as main_module
from app.main import app
from app.agent.orchestrator import AgentServices
from app.agent.streaming import stream_chat
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.memory.vector_store import VectorStore
from app.schemas import ChatRequest
from app.tools.attraction_rag_tool import AttractionRagTool
from fastapi.testclient import TestClient
from tests.fakes import FakeEmbedder, FakeImageClient, FakeSearchTool

client = TestClient(app)


def _install_services(monkeypatch, tmp_path) -> AgentServices:
    rag_store = VectorStore(path=str(tmp_path / "rag"), embedder=FakeEmbedder())
    services = AgentServices(
        attraction_rag_tool=AttractionRagTool(vector_store=rag_store),
        web_search_tool=FakeSearchTool(),
        image_client=FakeImageClient(),
        use_environment=False,
    )
    monkeypatch.setattr(main_module, "agent_services", services)
    return services


def _install_memory(monkeypatch, tmp_path) -> LongTermMemory:
    mem = LongTermMemory(VectorStore(path=str(tmp_path / "memory"), embedder=FakeEmbedder()))
    monkeypatch.setattr(main_module, "long_term_memory", mem)
    return mem


def _frames_to_events(frames):
    return [json.loads(frame[len("data: ") :]) for frame in frames]


def test_stream_chat_emits_progressive_agent_events(tmp_path):
    memory = ShortTermMemory()
    user_memory = LongTermMemory(VectorStore(path=str(tmp_path / "mem"), embedder=FakeEmbedder()))
    rag_store = VectorStore(path=str(tmp_path / "rag"), embedder=FakeEmbedder())
    services = AgentServices(
        attraction_rag_tool=AttractionRagTool(vector_store=rag_store),
        web_search_tool=FakeSearchTool(),
        image_client=FakeImageClient(),
        use_environment=False,
    )

    frames = list(
        stream_chat(
            ChatRequest(user_id="s", message="Plan a 2-day trip to Tokyo. I like anime and food. Medium budget."),
            memory=memory,
            user_memory=user_memory,
            services=services,
        )
    )
    events = _frames_to_events(frames)
    types = [e["type"] for e in events]

    # Progressive agent events are present.
    assert "agent_start" in types, types
    assert "agent_end" in types, types
    # At least one frame carries an "agent" field (the agent name).
    assert any(e.get("agent") for e in events), events
    # The itinerary agent handled this (city present).
    assert any(e.get("agent") == "ItineraryAgent" for e in events)
    # Plan steps stream before the result.
    assert "plan" in types
    assert "result" in types
    result = next(e for e in events if e["type"] == "result")
    assert result["response"]["itinerary"]["city"] == "Tokyo"


def test_stream_chat_emits_tool_events_for_itinerary(tmp_path):
    rag_store = VectorStore(path=str(tmp_path / "rag"), embedder=FakeEmbedder())
    services = AgentServices(
        attraction_rag_tool=AttractionRagTool(vector_store=rag_store),
        web_search_tool=FakeSearchTool(),
        image_client=FakeImageClient(),
        use_environment=False,
    )

    frames = list(
        stream_chat(
            ChatRequest(user_id="s", message="Plan a 2-day trip to Tokyo. I like anime and food. Medium budget."),
            memory=ShortTermMemory(),
            user_memory=LongTermMemory(VectorStore(path=str(tmp_path / "mem"), embedder=FakeEmbedder())),
            services=services,
        )
    )
    events = _frames_to_events(frames)

    tool_ends = [e for e in events if e["type"] == "tool_end"]
    assert tool_ends, "expected tool_end events for the itinerary tools"
    tool_names = {e["tool"] for e in tool_ends}
    # attraction_rag_tool is always selected for a city trip.
    assert "attraction_rag_tool" in tool_names, tool_names


def test_stream_chat_destination_agent_events(tmp_path):
    rag_store = VectorStore(path=str(tmp_path / "rag"), embedder=FakeEmbedder())
    services = AgentServices(
        attraction_rag_tool=AttractionRagTool(vector_store=rag_store),
        web_search_tool=FakeSearchTool(),
        image_client=FakeImageClient(),
        use_environment=False,
    )

    frames = list(
        stream_chat(
            ChatRequest(
                user_id="s",
                message="I like anime, food and photography, medium budget, 2 days",
            ),
            memory=ShortTermMemory(),
            user_memory=LongTermMemory(VectorStore(path=str(tmp_path / "mem"), embedder=FakeEmbedder())),
            services=services,
        )
    )
    events = _frames_to_events(frames)

    # The destination agent runs and emits its own tool (destination_rag).
    assert any(e.get("agent") == "DestinationRecommendationAgent" for e in events), events
    tool_ends = {e["tool"] for e in events if e["type"] == "tool_end"}
    assert "destination_rag" in tool_ends, tool_ends
    result = next(e for e in events if e["type"] == "result")
    assert result["response"]["itinerary"]["status"] == "destination_suggestions"


def test_stream_chat_route_event_names_the_agent(tmp_path):
    rag_store = VectorStore(path=str(tmp_path / "rag"), embedder=FakeEmbedder())
    services = AgentServices(
        attraction_rag_tool=AttractionRagTool(vector_store=rag_store),
        web_search_tool=FakeSearchTool(),
        image_client=FakeImageClient(),
        use_environment=False,
    )

    frames = list(
        stream_chat(
            ChatRequest(user_id="s", message="Plan a 2-day trip to Tokyo. Medium budget."),
            memory=ShortTermMemory(),
            user_memory=LongTermMemory(VectorStore(path=str(tmp_path / "mem"), embedder=FakeEmbedder())),
            services=services,
        )
    )
    events = _frames_to_events(frames)

    route_events = [e for e in events if e["type"] == "route"]
    assert route_events, "expected a route event naming the chosen agent"
    assert route_events[0]["agent"] == "ItineraryAgent"


def test_stream_chat_and_handle_chat_produce_equivalent_payload(tmp_path):
    # Streaming and synchronous paths share _run_chat_core, so the final
    # ChatResponse payload must match for the same request.
    from app.agent.orchestrator import handle_chat

    rag_store = VectorStore(path=str(tmp_path / "rag"), embedder=FakeEmbedder())
    services = AgentServices(
        attraction_rag_tool=AttractionRagTool(vector_store=rag_store),
        web_search_tool=FakeSearchTool(),
        image_client=FakeImageClient(),
        use_environment=False,
    )
    user_memory = LongTermMemory(VectorStore(path=str(tmp_path / "mem"), embedder=FakeEmbedder()))

    sync_response = handle_chat(
        ChatRequest(user_id="s", message="Plan a 2-day trip to Tokyo. Medium budget."),
        memory=ShortTermMemory(),
        user_memory=LongTermMemory(VectorStore(path=str(tmp_path / "mem2"), embedder=FakeEmbedder())),
        services=services,
    )

    frames = list(
        stream_chat(
            ChatRequest(user_id="s", message="Plan a 2-day trip to Tokyo. Medium budget."),
            memory=ShortTermMemory(),
            user_memory=user_memory,
            services=services,
        )
    )
    events = _frames_to_events(frames)
    streamed = next(e for e in events if e["type"] == "result")

    assert streamed["response"]["itinerary"]["city"] == sync_response.itinerary["city"]
    assert streamed["response"]["needs_clarification"] == sync_response.needs_clarification
    assert streamed["response"]["tools_used"] == sync_response.tools_used


def test_api_stream_emits_agent_start_frame(monkeypatch, tmp_path):
    _install_memory(monkeypatch, tmp_path)
    _install_services(monkeypatch, tmp_path)

    with client.stream(
        "POST", "/chat/stream", json={"user_id": "api-stream", "message": "Plan a 2-day trip to Tokyo. Medium budget."}
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        frames = [line for line in response.iter_lines() if line.startswith("data: ")]

    events = _frames_to_events(frames)
    assert any(e["type"] == "agent_start" and e.get("agent") for e in events), events
