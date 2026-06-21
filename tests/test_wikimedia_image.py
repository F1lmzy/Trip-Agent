"""Tests for the Wikimedia Commons image resolver.

Uses httpx.MockTransport with realistic Commons API payloads — no real network.
"""

import httpx

from app.tools.wikimedia_image_tool import (
    resolve_place_image,
    special_filepath_url,
)


_SEARCH_PAYLOAD_EIFFEL = {
    "batchcomplete": "",
    "query": {
        "pages": [
            {
                "pageid": 422551,
                "ns": 6,
                "title": "File:Lightning striking the Eiffel Tower - NOAA.jpg",
                "index": 3,
                "imageinfo": [
                    {
                        "thumburl": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/lightning.jpg/480px-lightning.jpg",
                        "url": "https://upload.wikimedia.org/wikipedia/commons/d/d7/lightning.jpg",
                        "mime": "image/jpeg",
                    }
                ],
            },
            {
                "pageid": 21375035,
                "ns": 6,
                "title": "File:Paris - The Eiffel Tower in spring - 2307.jpg",
                "index": 2,
                "imageinfo": [
                    {
                        "thumburl": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/eiffel_spring.jpg/480px-eiffel_spring.jpg",
                        "url": "https://upload.wikimedia.org/wikipedia/commons/b/b7/eiffel_spring.jpg",
                        "mime": "image/jpeg",
                    }
                ],
            },
        ]
    },
}

_SEARCH_PAYLOAD_MAP_ONLY = {
    "batchcomplete": "",
    "query": {
        "pages": [
            {
                "pageid": 1,
                "ns": 6,
                "title": "File:Paris locator map.svg",
                "index": 1,
                "imageinfo": [
                    {
                        "thumburl": "https://upload.wikimedia.org/wikipedia/commons/thumb/x/x/paris_map.svg/480px-paris_map.svg.png",
                        "url": "https://upload.wikimedia.org/wikipedia/commons/x/x/paris_map.svg",
                        "mime": "image/svg+xml",
                    }
                ],
            },
            {
                "pageid": 2,
                "ns": 6,
                "title": "File:Paris flag.svg",
                "index": 2,
                "imageinfo": [{"thumburl": "https://upload.wikimedia.org/flag.svg.png", "mime": "image/svg+xml"}],
            },
        ]
    },
}

_EMPTY_PAYLOAD = {"batchcomplete": "", "query": {"pages": []}}


def _mock_client(payload: dict) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "commons.wikimedia.org" in str(request.url)
        assert "generator=search" in str(request.url)
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_resolve_place_image_adds_city_context_to_search_query():
    seen_queries = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_queries.append(request.url.params["gsrsearch"])
        return httpx.Response(200, json=_EMPTY_PAYLOAD)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resolve_place_image("Old Town", city="Edinburgh", client=client)

    assert seen_queries == ["Old Town Edinburgh"]


def test_resolve_place_image_returns_thumburl_preferring_name_match():
    client = _mock_client(_SEARCH_PAYLOAD_EIFFEL)
    url = resolve_place_image("Eiffel Tower", client=client)

    # The "Lightning striking..." result is index 3 but contains the name;
    # "Paris - The Eiffel Tower in spring" is index 2 (better rank) and also
    # contains the name. Both match the needle; we accept either as long as it
    # is a photographic, name-matching title (NOT a fallback to a non-match).
    assert url is not None
    assert url.startswith("https://upload.wikimedia.org/")
    assert "eiffel" in url.lower() or "lightning" in url.lower()


def test_resolve_place_image_skips_non_photographic_files():
    client = _mock_client(_SEARCH_PAYLOAD_MAP_ONLY)
    url = resolve_place_image("Paris", client=client)

    # Both results are SVG maps/flags — no photographic file -> None.
    assert url is None


def test_resolve_place_image_returns_none_when_no_results():
    client = _mock_client(_EMPTY_PAYLOAD)
    url = resolve_place_image("Nonexistent Place XYZ123", client=client)
    assert url is None


def test_resolve_place_image_empty_name_returns_none():
    assert resolve_place_image("   ") is None


def test_resolve_place_image_http_error_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    url = resolve_place_image("Eiffel Tower", client=client)
    assert url is None


def test_resolve_place_image_formatversion1_dict_pages_handled():
    # formatversion=1 returns pages as a dict keyed by pageid.
    payload_v1 = {
        "query": {
            "pages": {
                "10": {
                    "pageid": 10,
                    "title": "File:British Museum.jpg",
                    "index": 1,
                    "imageinfo": [
                        {"thumburl": "https://upload.wikimedia.org/bm.jpg/480px-bm.jpg", "mime": "image/jpeg"}
                    ],
                }
            }
        }
    }
    client = _mock_client(payload_v1)
    url = resolve_place_image("British Museum", client=client)
    assert url == "https://upload.wikimedia.org/bm.jpg/480px-bm.jpg"


def test_special_filepath_url_strips_file_prefix_and_encodes_spaces():
    assert (
        special_filepath_url("File:Eiffel Tower.jpg")
        == "https://commons.wikimedia.org/wiki/Special:FilePath/Eiffel_Tower.jpg"
    )
    assert (
        special_filepath_url("British Museum.jpg")
        == "https://commons.wikimedia.org/wiki/Special:FilePath/British_Museum.jpg"
    )
