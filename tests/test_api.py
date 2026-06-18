from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.memory.long_term import LongTermMemory
from app.memory.vector_store import VectorStore
from tests.fakes import FakeEmbedder


client = TestClient(app)


def install_test_long_term_memory(monkeypatch, tmp_path) -> LongTermMemory:
    test_memory = LongTermMemory(VectorStore(path=str(tmp_path), embedder=FakeEmbedder()))
    monkeypatch.setattr(main_module, "long_term_memory", test_memory)
    return test_memory


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_returns_html():
    response = client.get("/")

    assert response.status_code == 200
    assert "Travel Planning Agent" in response.text


def test_chat_returns_response_shape(monkeypatch, tmp_path):
    install_test_long_term_memory(monkeypatch, tmp_path)

    response = client.post("/chat", json={"user_id": "api-shape-user", "message": "Plan Tokyo"})

    assert response.status_code == 200
    body = response.json()
    assert body["message"]
    assert body["itinerary"]["city"] == "Tokyo"
    assert isinstance(body["tools_used"], list)
    assert "attraction_rag_tool" in body["tools_used"]
    assert isinstance(body["plan"], list)
    assert body["needs_clarification"] is False


def test_chat_clarifies_when_city_missing(monkeypatch, tmp_path):
    install_test_long_term_memory(monkeypatch, tmp_path)

    response = client.post("/chat", json={"user_id": "api-clarify-user", "message": "Plan me a trip"})

    assert response.status_code == 200
    body = response.json()
    assert body["needs_clarification"] is True
    assert body["tools_used"] == []
    assert body["clarifying_question"] == "Which city would you like to visit?"


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
