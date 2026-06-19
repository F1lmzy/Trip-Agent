"""Tests for the external content fetcher and chunker."""

import httpx

from app.tools.external_content import (
    _chunk_extract,
    external_docs_to_vectors,
    fetch_city_docs,
)


def _wikivoyage_response(extract: str, title: str = "Kyoto") -> dict:
    return {
        "query": {
            "pages": {
                "1": {
                    "pageid": 1,
                    "title": title,
                    "extract": extract,
                }
            }
        }
    }


def _wikipedia_response(extract: str, title: str = "Kyoto") -> dict:
    return {
        "query": {
            "pages": {
                "37652": {
                    "pageid": 37652,
                    "title": title,
                    "extract": extract,
                }
            }
        }
    }


def _mock_client(wikivoyage_extract: str | None, wikipedia_extract: str | None) -> httpx.Client:
    """Build a mock httpx client that returns canned API responses."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "wikivoyage.org" in url:
            if wikivoyage_extract is not None:
                return httpx.Response(200, json=_wikivoyage_response(wikivoyage_extract))
            return httpx.Response(200, json={"query": {"pages": {"1": {"title": "Kyoto"}}}})
        if "wikipedia.org" in url:
            if wikipedia_extract is not None:
                return httpx.Response(200, json=_wikipedia_response(wikipedia_extract))
            return httpx.Response(200, json={"query": {"pages": {"37652": {"title": "Kyoto"}}}})
        return httpx.Response(404, json={"error": "not found"})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_city_docs_uses_wikivoyage_first():
    extract = (
        "Kyoto is a beautiful city. " * 20
        + "\n\n"
        + "It has many temples and shrines. " * 10
    )
    client = _mock_client(wikivoyage_extract=extract, wikipedia_extract=None)

    docs = fetch_city_docs("Kyoto", client=client)

    assert len(docs) >= 2
    assert all(doc.city == "Kyoto" for doc in docs)
    assert all(doc.source == "wikivoyage" for doc in docs)
    assert all(doc.text for doc in docs)


def test_fetch_city_docs_falls_back_to_wikipedia_when_wikivoyage_empty():
    wikipedia_extract = "Kyoto is the capital of Kyoto Prefecture. " * 20
    client = _mock_client(wikivoyage_extract=None, wikipedia_extract=wikipedia_extract)

    docs = fetch_city_docs("Kyoto", client=client)

    assert len(docs) >= 2
    assert all(doc.source == "wikipedia" for doc in docs)


def test_fetch_city_docs_returns_empty_when_both_sources_fail():
    client = _mock_client(wikivoyage_extract=None, wikipedia_extract=None)

    docs = fetch_city_docs("Nonexistent City", client=client)

    assert docs == []


def test_fetch_city_docs_returns_empty_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server error"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    docs = fetch_city_docs("Kyoto", client=client)

    assert docs == []


def test_chunk_extract_splits_into_paragraphs():
    extract = (
        "First paragraph about Kyoto temples and their historical significance in Japanese culture.\n\n"
        "Second paragraph about food markets and local cuisine options for travelers visiting.\n\n"
        "Third paragraph about the bamboo grove in Arashiyama and its scenic walking paths."
    )

    docs = _chunk_extract(extract, "Kyoto", "wikivoyage")

    assert len(docs) == 3
    assert "temples" in docs[0].text
    assert "food markets" in docs[1].text
    assert "bamboo grove" in docs[2].text


def test_chunk_extract_merges_short_paragraphs():
    extract = (
        "Long enough paragraph about Kyoto history and culture for the first chunk here.\n\n"
        "Short.\n\n"
        "Another long paragraph about temples and shrines in the area for the second chunk."
    )

    docs = _chunk_extract(extract, "Kyoto", "wikivoyage")

    assert len(docs) == 2
    assert "Short." in docs[0].text


def test_chunk_extract_splits_long_paragraphs_at_sentences():
    long_paragraph = ". ".join([f"Sentence number {i} about Kyoto" for i in range(30)]) + "."

    docs = _chunk_extract(long_paragraph, "Kyoto", "wikipedia")

    assert len(docs) >= 2
    for doc in docs:
        assert len(doc.text) <= 600


def test_external_docs_to_vectors_returns_correct_shape():
    from app.tools.external_content import ExternalDoc

    docs = [
        ExternalDoc(id="ext-wikipedia-kyoto-0", city="Kyoto", text="About Kyoto.", source="wikipedia"),
        ExternalDoc(id="ext-wikipedia-kyoto-1", city="Kyoto", text="More about Kyoto.", source="wikipedia"),
    ]

    payload = external_docs_to_vectors(docs)

    assert payload["documents"] == ["About Kyoto.", "More about Kyoto."]
    assert payload["ids"] == ["ext-wikipedia-kyoto-0", "ext-wikipedia-kyoto-1"]
    assert payload["metadatas"][0] == {"city": "Kyoto", "type": "city_overview", "source": "external_wikipedia"}
    assert len(payload["metadatas"]) == 2
