from app.agent.orchestrator import handle_chat
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.memory.vector_store import VectorStore
from app.schemas import ChatRequest
from tests.fakes import FakeEmbedder


def make_long_term_memory(tmp_path) -> LongTermMemory:
    return LongTermMemory(VectorStore(path=str(tmp_path), embedder=FakeEmbedder()))


def test_handle_chat_returns_clarification_without_tools_when_city_missing():
    memory = ShortTermMemory()

    response = handle_chat(ChatRequest(user_id="kavin", message="Plan me a trip"), memory=memory)

    assert response.needs_clarification is True
    assert response.clarifying_question == "Which city would you like to visit?"
    assert response.tools_used == []
    assert memory.has_history("kavin") is True


def test_handle_chat_returns_planned_tools_for_normal_trip():
    memory = ShortTermMemory()

    response = handle_chat(
        ChatRequest(user_id="kavin", message="Plan a 2-day trip to Tokyo. I like anime and food. Medium budget."),
        memory=memory,
    )

    assert response.needs_clarification is False
    assert response.itinerary["city"] == "Tokyo"
    assert response.itinerary["duration_days"] == 2
    assert "attraction_rag_tool" in response.tools_used
    assert "weather_tool" in response.tools_used
    assert "budget_tool" in response.tools_used
    assert response.plan


def test_follow_up_uses_short_term_memory_when_history_exists():
    memory = ShortTermMemory()
    handle_chat(ChatRequest(user_id="kavin", message="Plan a 2-day trip to Tokyo."), memory=memory)

    response = handle_chat(ChatRequest(user_id="kavin", message="Make it cheaper."), memory=memory)

    assert response.needs_clarification is False
    assert response.itinerary["follow_up_intent"] == "cheaper"
    assert response.memory_used == ["Recent conversation history"]
    assert response.tools_used == ["budget_tool"]


def test_follow_up_without_history_asks_for_context():
    memory = ShortTermMemory()

    response = handle_chat(ChatRequest(user_id="new-user", message="Make it cheaper."), memory=memory)

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
    )

    assert "I prefer vegetarian food" in response.memory_used
