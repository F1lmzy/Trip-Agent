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
    response = client.post("/chat", json={"user_id": "kavin", "message": "Plan Tokyo"})

    assert response.status_code == 200
    body = response.json()
    assert body["message"]
    assert isinstance(body["itinerary"], dict)
    assert isinstance(body["tools_used"], list)
    assert isinstance(body["plan"], list)
    assert body["needs_clarification"] is False


def test_memory_placeholders_return_valid_shapes():
    get_response = client.get("/memory/kavin")
    add_response = client.post("/memory/kavin", json={"preference": "I like museums"})
    delete_response = client.delete("/memory/kavin")

    assert get_response.status_code == 200
    assert get_response.json() == {"user_id": "kavin", "memories": []}
    assert add_response.status_code == 200
    assert delete_response.status_code == 200
