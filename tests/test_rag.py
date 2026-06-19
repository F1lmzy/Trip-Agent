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
    import httpx

    from app.tools.external_content import clear_failed_cache

    clear_failed_cache()
    tool = make_tool(tmp_path)

    # Use a mock client that returns empty pages so no external docs are ingested.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"query": {"pages": {"1": {"title": "Atlantis"}}}})

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))

    result = tool.run(city="Atlantis", interests=["museums"], http_client=mock_client)

    assert result["status"] == "no_results"
    assert result["results"] == []
    assert result["rag_trace"]["hop_1"] == []
    assert result["rag_trace"]["hop_2"] == []


def test_rag_auto_ingests_external_docs_for_unknown_city(tmp_path):
    """When hop_1 is empty for an unknown city, the tool fetches external docs
    and retries, producing real hop_1 results from the ingested content."""
    import httpx

    from app.tools.external_content import clear_failed_cache

    clear_failed_cache()
    vector_store = VectorStore(path=str(tmp_path), embedder=FakeEmbedder())
    tool = AttractionRagTool(vector_store=vector_store)
    tool.seed()

    wikivoyage_extract = (
        "Kyoto is a beautiful city with many temples and shrines. "
        "Fushimi Inari is famous for its torii gates. "
        "Arashiyama has a bamboo grove. "
        "Nishiki Market is great for food. "
        "Gion is the historic geisha district."
    ) * 5

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": {
                        "1": {"pageid": 1, "title": "Kyoto", "extract": wikivoyage_extract}
                    }
                }
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))

    result = tool.run(city="Kyoto", interests=["food", "culture"], http_client=mock_client)

    assert result["status"] == "ok"
    assert result["rag_trace"]["hop_1"]
    assert result["rag_trace"]["hop_1"][0]["metadata"]["source"] == "external_wikivoyage"


def test_rag_auto_ingest_caches_so_second_call_skips_fetch(tmp_path):
    """After the first call ingests external docs, a second call for the same
    city should find them in ChromaDB without fetching again."""
    import httpx

    from app.tools.external_content import clear_failed_cache

    clear_failed_cache()
    vector_store = VectorStore(path=str(tmp_path), embedder=FakeEmbedder())
    tool = AttractionRagTool(vector_store=vector_store)
    tool.seed()

    fetch_count = 0
    wikivoyage_extract = "Kyoto has temples and food markets. " * 20

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal fetch_count
        fetch_count += 1
        return httpx.Response(
            200,
            json={"query": {"pages": {"1": {"title": "Kyoto", "extract": wikivoyage_extract}}}},
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))

    tool.run(city="Kyoto", interests=["food"], http_client=mock_client)
    first_fetch_count = fetch_count

    tool.run(city="Kyoto", interests=["food"], http_client=mock_client)

    assert fetch_count == first_fetch_count


def test_rag_detects_and_replaces_stale_external_attractions(tmp_path):
    """When hop_2 returns old-format external entries (text-chunk names),
    the tool should detect them, delete them, and re-ingest with real
    attraction names from Wikivoyage's See/Do sections."""
    import httpx

    from app.tools.external_content import clear_failed_cache

    clear_failed_cache()
    vector_store = VectorStore(path=str(tmp_path), embedder=FakeEmbedder())
    tool = AttractionRagTool(vector_store=vector_store)
    tool.seed()

    # Manually insert stale external city docs (so hop_1 finds them)
    # and stale external attractions with text-chunk names (so hop_2 finds them).
    vector_store.add_documents(
        "travel_city_docs",
        documents=["Liverpool is a big city in Merseyside famed for music and culture."],
        metadatas=[{"city": "Liverpool", "type": "city_overview", "source": "external_wikivoyage"}],
        ids=["ext-wikivoyage-liverpool-0"],
    )
    vector_store.add_documents(
        "travel_attractions",
        documents=["Liverpool is a big city in Merseyside famed for music and culture."],
        metadatas=[
            {
                "city": "Liverpool",
                "type": "attraction",
                "name": "Liverpool is a big city in Merseyside, England famed for its",
                "source": "external_wikivoyage",
                "categories": "",
            }
        ],
        ids=["ext-wikivoyage-liverpool-0-attr"],
    )

    # Mock API that returns real attractions from the See section.
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "prop=sections" in url:
            return httpx.Response(
                200,
                json={"parse": {"sections": [{"index": "17", "line": "See", "level": "2"}]}},
            )
        if "prop=wikitext" in url:
            return httpx.Response(
                200,
                json={
                    "parse": {
                        "wikitext": {
                            "*": "==See==\n* {{see\n| name=Museum of Liverpool\n| content=A large museum about Liverpool history.\n}}\n* {{see\n| name=Royal Liver Building\n| content=Iconic waterfront building.\n}}"
                        }
                    }
                },
            )
        return httpx.Response(200, json={"query": {"pages": {"1": {"title": "Liverpool"}}}})

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))

    result = tool.run(city="Liverpool", interests=["music"], http_client=mock_client)

    assert result["status"] == "ok"
    # The stale text-chunk name should be gone, replaced with real attraction names.
    names = {item["name"] for item in result["results"]}
    assert "Liverpool is a big city in Merseyside, England famed for its" not in names
    assert any(name in names for name in {"Museum of Liverpool", "Royal Liver Building"})
