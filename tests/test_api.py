from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_returns_html():
    response = client.get("/")

    assert response.status_code == 200
    assert "Travel Planning Agent" in response.text


def test_chat_returns_response_shape():
    response = client.post("/chat", json={"user_id": "api-shape-user", "message": "Plan Tokyo"})

    assert response.status_code == 200
    body = response.json()
    assert body["message"]
    assert body["itinerary"]["city"] == "Tokyo"
    assert isinstance(body["tools_used"], list)
    assert "attraction_rag_tool" in body["tools_used"]
    assert isinstance(body["plan"], list)
    assert body["needs_clarification"] is False


def test_chat_clarifies_when_city_missing():
    response = client.post("/chat", json={"user_id": "api-clarify-user", "message": "Plan me a trip"})

    assert response.status_code == 200
    body = response.json()
    assert body["needs_clarification"] is True
    assert body["tools_used"] == []
    assert body["clarifying_question"] == "Which city would you like to visit?"


def test_memory_placeholders_return_valid_shapes():
    get_response = client.get("/memory/kavin")
    add_response = client.post("/memory/kavin", json={"preference": "I like museums"})
    delete_response = client.delete("/memory/kavin")

    assert get_response.status_code == 200
    assert get_response.json() == {"user_id": "kavin", "memories": []}
    assert add_response.status_code == 200
    assert delete_response.status_code == 200
