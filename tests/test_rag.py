from app.memory.vector_store import VectorStore
from app.tools.attraction_rag_tool import AttractionRagTool, load_attraction_documents
from tests.fakes import FakeEmbedder


def make_tool(tmp_path) -> AttractionRagTool:
    vector_store = VectorStore(path=str(tmp_path), embedder=FakeEmbedder())
    tool = AttractionRagTool(vector_store=vector_store)
    tool.seed()
    return tool


def test_rag_seed_data_includes_demo_cities():
    docs = load_attraction_documents()
    cities = {doc["city"] for doc in docs}

    assert {"Tokyo", "Singapore", "Paris", "New York", "Mumbai"} <= cities


def test_rag_hop_1_retrieves_city_overview(tmp_path):
    tool = make_tool(tmp_path)

    result = tool.run(city="Tokyo", interests=["anime", "food", "photography"])

    assert result["status"] == "ok"
    assert result["rag_trace"]["hop_1"]
    assert result["rag_trace"]["hop_1"][0]["metadata"]["type"] == "city_overview"


def test_rag_hop_2_uses_interests_for_tokyo_matches(tmp_path):
    tool = make_tool(tmp_path)

    result = tool.run(city="Tokyo", interests=["anime", "food", "photography"])
    names = {item["name"] for item in result["results"]}

    assert "Akihabara" in names
    assert names & {"Tsukiji Outer Market", "Shibuya Sky", "Senso-ji Temple"}
    assert result["rag_trace"]["hop_2"]


def test_rag_filters_results_by_city(tmp_path):
    tool = make_tool(tmp_path)

    result = tool.run(city="Paris", interests=["museums", "art"])
    names = {item["name"] for item in result["results"]}

    assert "Louvre Museum" in names
    assert "Akihabara" not in names


def test_rag_supports_mumbai_food_and_culture(tmp_path):
    tool = make_tool(tmp_path)

    result = tool.run(city="mumbai", interests=["food", "culture", "photography"])
    names = {item["name"] for item in result["results"]}

    assert result["status"] == "ok"
    assert "Gateway of India and Colaba" in names
    assert names & {"Marine Drive and Girgaum Chowpatty", "Bandra West and Bandstand"}


def test_rag_unknown_city_returns_empty_result(tmp_path):
    tool = make_tool(tmp_path)

    result = tool.run(city="Atlantis", interests=["museums"])

    assert result["status"] == "no_results"
    assert result["results"] == []
    assert result["rag_trace"]["hop_1"] == []
    assert result["rag_trace"]["hop_2"] == []
