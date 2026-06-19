"""Tests for the external content fetcher and chunker."""

import httpx

from app.tools.external_content import (
    _chunk_extract,
    _clean_wikitext,
    _parse_attractions_from_wikitext,
    _retry_after_seconds,
    clear_failed_cache,
    external_docs_to_vectors,
    fetch_city_attractions,
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


def test_fetch_city_docs_returns_empty_when_wikivoyage_empty():
    """No Wikipedia fallback: if Wikivoyage has no usable article, return [].

    The Wikipedia fallback was removed because disambiguation pages (e.g.
    "Newcastle usually refers to:") were being ingested as city context and
    served as attractions. Wikivoyage is now the single external source.
    """
    client = _mock_client(wikivoyage_extract=None, wikipedia_extract="Kyoto is the capital of Kyoto Prefecture. " * 20)
    clear_failed_cache()

    docs = fetch_city_docs("Kyoto", client=client, sleep_fn=_noop_sleep)

    assert docs == []


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
    # Wikivoyage only (no Wikipedia fallback): 4 attempts = 1 + 3 retries.
    assert call_count == 4


# --- Attraction parsing tests ---


_WIKIVOYAGE_SEE_WIKITEXT = """==See==
{{Mapframe|53.4029393|-2.9956251|zoom=15}}
Liverpool is particularly famous for its architecture.

=== Pier Head ===
* {{see
| name=Museum of Liverpool | alt= | url=http://www.liverpoolmuseums.org.uk/mol/ | email=
| address=Pier Head, Liverpool Waterfront, Liverpool L3 1DG | lat=53.402939 | long=-2.995625
| phone=+44 151 478-4545 | tollfree=
| hours=10AM-5PM | price=Free
| wikipedia=Museum of Liverpool
| content=A large museum all about the city of Liverpool and its history from ancient inhabitants to its modern revival.
}}
* {{see
| name=Royal Liver Building | alt= | url= | email=
| address= | lat=53.40585 | long=-2.99592
| content=Iconic symbol of Liverpool waterfront. This 1911 skyscraper still dominates the distinctive Liverpool skyline.
}}
* '''Titanic Memorial''' is north side of the Royal Liver Building, a sober granite monument to the 244 engineers lost with the ship.

=== Albert Dock ===
* {{see
| name=Merseyside Maritime Museum | alt= | url= | email=
| content=Museum with permanent gallery devoted to the ''Titanic'', ''Lusitania'' and ''Fortyacre''.
}}
* {{see
| name=The Beatles Story | alt=Fab4D Cinema | url=http://www.beatlesstory.com/
| content=A film telling a story using The Beatles as a theme. Not to be confused with the [[Cavern Club]].
}}
"""


def test_parse_attractions_extracts_see_template_entries():
    attractions = _parse_attractions_from_wikitext(_WIKIVOYAGE_SEE_WIKITEXT, "Liverpool")

    names = {a.name for a in attractions}
    assert "Museum of Liverpool" in names
    assert "Royal Liver Building" in names
    assert "Merseyside Maritime Museum" in names
    assert "The Beatles Story" in names


def test_parse_attractions_extracts_bold_name_entries():
    attractions = _parse_attractions_from_wikitext(_WIKIVOYAGE_SEE_WIKITEXT, "Liverpool")

    names = {a.name for a in attractions}
    assert "Titanic Memorial" in names


def test_parse_attractions_cleans_wikitext_markup_from_descriptions():
    attractions = _parse_attractions_from_wikitext(_WIKIVOYAGE_SEE_WIKITEXT, "Liverpool")

    beatles = next(a for a in attractions if a.name == "The Beatles Story")
    # Wiki links like [[Cavern Club]] should be cleaned to plain text.
    assert "[[" not in beatles.description
    assert "Cavern Club" in beatles.description
    # Italic markers '' should be removed.
    maritime = next(a for a in attractions if a.name == "Merseyside Maritime Museum")
    assert "''" not in maritime.description


def test_parse_attractions_returns_empty_for_no_templates():
    wikitext = "This is just a plain paragraph with no attraction listings."

    attractions = _parse_attractions_from_wikitext(wikitext, "TestCity")

    assert attractions == []


def test_clean_wikitext_strips_links_bold_italic_and_templates():
    raw = "Visit the '''[[Royal Albert Dock|Albert Dock]]''' for {{free}} entry to ''galleries''."

    cleaned = _clean_wikitext(raw)

    assert "''" not in cleaned
    assert "[[" not in cleaned
    assert "{{" not in cleaned
    assert "Albert Dock" in cleaned
    assert "galleries" in cleaned


def test_fetch_city_attractions_with_mock_api():
    """End-to-end test with a mock Wikivoyage API returning sections and wikitext."""
    clear_failed_cache()

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "prop=sections" in url:
            return httpx.Response(
                200,
                json={
                    "parse": {
                        "title": "Liverpool",
                        "sections": [
                            {"index": "1", "line": "Understand", "level": "2"},
                            {"index": "17", "line": "See", "level": "2"},
                            {"index": "24", "line": "Do", "level": "2"},
                        ],
                    }
                },
            )
        if "prop=wikitext" in url:
            return httpx.Response(
                200,
                json={
                    "parse": {
                        "title": "Liverpool",
                        "wikitext": {"*": _WIKIVOYAGE_SEE_WIKITEXT},
                    }
                },
            )
        return httpx.Response(404, json={"error": "not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    attractions = fetch_city_attractions("Liverpool", client=client, sleep_fn=_noop_sleep)

    assert len(attractions) >= 4
    assert all(a.city == "Liverpool" for a in attractions)
    assert all(a.name for a in attractions)
    assert any(a.name == "Museum of Liverpool" for a in attractions)


def test_fetch_city_attractions_returns_empty_when_no_see_section():
    clear_failed_cache()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "parse": {
                    "title": "SmallTown",
                    "sections": [
                        {"index": "1", "line": "Understand", "level": "2"},
                        {"index": "2", "line": "Get in", "level": "2"},
                    ],
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    attractions = fetch_city_attractions("SmallTown", client=client, sleep_fn=_noop_sleep)

    assert attractions == []
