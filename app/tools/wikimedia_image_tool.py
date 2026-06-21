"""Wikimedia Commons image resolver.

Resolves a place/attraction/hotel name to a usable image URL from Wikimedia
Commons (free, no API key required). Uses the MediaWiki ``generator=search``
API on the File namespace (6) with ``imageinfo`` to get a thumbnail URL.

Designed to be MockTransport-testable (injectable ``httpx.Client``) and to
degrade gracefully: returns ``None`` for an unfound place, never crashes, and
never blocks itinerary generation on a missing image.

The returned URL is a ``upload.wikimedia.org`` thumbnail (width 480) which
browsers render directly. Special:FilePath is used as a deterministic fallback
when the search API returns nothing for an exact title match.
"""

from __future__ import annotations

from typing import Any

import logging

import httpx

logger = logging.getLogger(__name__)

_COMMONS_API = "https://commons.wikimedia.org/w/api.php"
_DEFAULT_TIMEOUT = 15.0
_THUMB_WIDTH = 480
_USER_AGENT = "TripAgent/1.0 (https://github.com/srirajkavin/Trip-Agent; educational project)"

# File extensions we accept as photographic images (skip PDFs, SVGs of maps,
# logos, audio, video, etc. which are common in Commons search results).
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".gif")


def resolve_place_image(
    place_name: str,
    city: str | None = None,
    client: httpx.Client | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> str | None:
    """Resolve a place name to a Wikimedia Commons image URL.

    Args:
        place_name: Place, attraction, or hotel name (e.g. "Eiffel Tower").
        city: Optional city context used to disambiguate generic names like
            "Old Town" or "Central Park".
        client: Optional httpx client for testability. If None, a transient
            client is created.
        timeout: Request timeout in seconds.

    Returns:
        A direct image URL (upload.wikimedia.org thumbnail) or None if no
        suitable image is found or the request fails.
    """
    normalized = place_name.strip()
    normalized_city = city.strip() if city else ""
    if not normalized:
        return None
    search_query = normalized
    if normalized_city and normalized_city.lower() not in normalized.lower():
        search_query = f"{normalized} {normalized_city}"

    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": search_query,
        "gsrnamespace": "6",  # File namespace
        "gsrlimit": "8",
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "iiurlwidth": str(_THUMB_WIDTH),
        "format": "json",
        "formatversion": "2",
    }
    headers = {"User-Agent": _USER_AGENT}

    try:
        owns_client = client is None
        http_client = client or httpx.Client(timeout=timeout, headers=headers)
        try:
            response = http_client.get(_COMMONS_API, params=params, headers=headers)
            response.raise_for_status()
            payload = response.json()
        finally:
            if owns_client:
                http_client.close()
    except Exception:  # noqa: BLE001 - never crash the agent over an image
        logger.warning("Wikimedia image lookup failed for %r", place_name, exc_info=True)
        return None

    return _pick_best_image(payload, normalized)


def _pick_best_image(payload: dict[str, Any], place_name: str) -> str | None:
    """Pick the most relevant image URL from a Commons search payload.

    Relevance heuristics (best-effort, not perfect):
      1. Skip non-photographic files by extension (PDF/SVG-map/audio/etc.).
      2. Prefer titles whose normalized name is contained in the file title
         (e.g. "Eiffel Tower" in "Paris - The Eiffel Tower in spring").
      3. Among matching titles, prefer the first by search rank (Commons
         returns a relevance-ranked index).
    Fallback: if no title matches by name, accept the first photographic file.
    """
    pages = _ordered_pages(payload)
    if not pages:
        return None

    needle = place_name.strip().lower()
    photographic = [
        page for page in pages
        if _is_photographic(page.get("title", ""))
    ]
    if not photographic:
        return None

    name_match = [
        page for page in photographic
        if needle and needle in (page.get("title") or "").lower()
    ]
    chosen = name_match[0] if name_match else photographic[0]

    info = _first_imageinfo(chosen)
    if not info:
        return None
    return info.get("thumburl") or info.get("url")


def _ordered_pages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    query = payload.get("query") or {}
    pages = query.get("pages")
    if not isinstance(pages, list):
        # formatversion=1 returns a dict keyed by pageid; normalize to a list.
        if isinstance(pages, dict):
            pages = list(pages.values())
        else:
            return []
    # Commons search ranks by relevance; preserve API order (formatversion=2
    # preserves order, but be defensive and sort by the `index` field if present).
    return sorted(
        [p for p in pages if isinstance(p, dict)],
        key=lambda p: p.get("index", 10**9),
    )


def _first_imageinfo(page: dict[str, Any]) -> dict[str, Any] | None:
    info = page.get("imageinfo")
    if isinstance(info, list) and info and isinstance(info[0], dict):
        return info[0]
    return None


def _is_photographic(title: str) -> bool:
    if not title:
        return False
    lower = title.lower()
    if not lower.endswith(_IMAGE_EXTENSIONS):
        return False
    # Skip obvious non-photo artifacts even when the extension matches.
    skip_markers = (" logo", "icon", "map", ".svg", " diagram", " flag")
    return not any(marker in lower for marker in skip_markers)


def special_filepath_url(file_title: str) -> str:
    """Build a deterministic Special:FilePath URL for an exact Commons file title.

    Used as a fallback when the search API is unavailable. ``file_title`` should
    be the full Commons title (e.g. "File:Eiffel Tower.jpg") or just the file
    name; the "File:" prefix is stripped.
    """
    name = file_title.strip()
    if name.lower().startswith("file:"):
        name = name[5:]
    name = name.replace(" ", "_")
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{name}"
