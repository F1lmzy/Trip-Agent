"""Tests for the external content fetcher and chunker."""

import httpx

from app.tools.external_content import (
    _chunk_extract,
    _retry_after_seconds,
    clear_failed_cache,
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
    clear_failed_cache()

    docs = fetch_city_docs("Kyoto", client=client, sleep_fn=_noop_sleep)

    assert len(docs) >= 2
    assert all(doc.city == "Kyoto" for doc in docs)
    assert all(doc.source == "wikivoyage" for doc in docs)
    assert all(doc.text for doc in docs)


def test_fetch_city_docs_falls_back_to_wikipedia_when_wikivoyage_empty():
    wikipedia_extract = "Kyoto is the capital of Kyoto Prefecture. " * 20
    client = _mock_client(wikivoyage_extract=None, wikipedia_extract=wikipedia_extract)
    clear_failed_cache()

    docs = fetch_city_docs("Kyoto", client=client, sleep_fn=_noop_sleep)

    assert len(docs) >= 2
    assert all(doc.source == "wikipedia" for doc in docs)


def test_fetch_city_docs_returns_empty_when_both_sources_fail():
    client = _mock_client(wikivoyage_extract=None, wikipedia_extract=None)
    clear_failed_cache()

    docs = fetch_city_docs("Nonexistent City", client=client, sleep_fn=_noop_sleep)

    assert docs == []


def test_fetch_city_docs_returns_empty_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server error"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    clear_failed_cache()
    docs = fetch_city_docs("Kyoto", client=client, sleep_fn=_noop_sleep)

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


# --- Retry and rate-limit tests ---


def _noop_sleep(_seconds: float) -> None:
    """A sleep function that does nothing, for fast tests."""
    pass


def test_fetch_retries_on_429_then_succeeds():
    """A 429 on the first attempt should trigger a retry that succeeds."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if "wikivoyage" in str(request.url) and call_count == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={})
        return httpx.Response(
            200,
            json=_wikivoyage_response("Kyoto is a beautiful city with temples. " * 20),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    clear_failed_cache()
    docs = fetch_city_docs("Kyoto", client=client, sleep_fn=_noop_sleep)

    assert len(docs) >= 2
    assert call_count >= 2  # At least one retry


def test_fetch_respects_retry_after_header():
    """The Retry-After header value should be used as the wait time."""
    response = httpx.Response(429, headers={"Retry-After": "5"})

    wait = _retry_after_seconds(response, default=2.0)

    assert wait == 5.0


def test_fetch_falls_back_to_default_backoff_without_retry_after():
    """Without a Retry-After header, the default backoff should be used."""
    response = httpx.Response(429)

    wait = _retry_after_seconds(response, default=3.0)

    assert wait == 3.0


def test_fetch_caches_failures_to_avoid_repeated_429s():
    """After a city fails all retries, a second fetch should skip the API entirely."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(429, headers={"Retry-After": "0"}, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    clear_failed_cache()

    docs1 = fetch_city_docs("Stockholm", client=client, sleep_fn=_noop_sleep)
    calls_after_first = call_count

    docs2 = fetch_city_docs("Stockholm", client=client, sleep_fn=_noop_sleep)
    calls_after_second = call_count

    assert docs1 == []
    assert docs2 == []
    # The second fetch should not have made any new API calls (cached failure).
    assert calls_after_second == calls_after_first


def test_fetch_retries_on_500_then_succeeds():
    """A 5xx on the first attempt should trigger a retry that succeeds."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(503, json={"error": "service unavailable"})
        return httpx.Response(
            200,
            json=_wikivoyage_response("Kyoto is a beautiful city with temples. " * 20),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    clear_failed_cache()
    docs = fetch_city_docs("Kyoto", client=client, sleep_fn=_noop_sleep)

    assert len(docs) >= 2
    assert call_count >= 2


def test_fetch_gives_up_after_max_retries_on_persistent_429():
    """Persistent 429s should exhaust retries and return empty without hanging."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(429, headers={"Retry-After": "0"}, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    clear_failed_cache()
    docs = fetch_city_docs("Oslo", client=client, sleep_fn=_noop_sleep)

    assert docs == []
    # Wikivoyage: 4 attempts (1 + 3 retries), Wikipedia: 4 attempts = 8 total.
    assert call_count == 8
