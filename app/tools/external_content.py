"""Fetch and chunk external travel content for RAG ingestion.

Fetches city articles from Wikivoyage (travel-focused) with a fallback to
Wikipedia (general encyclopedic content) via their Action API. The returned
plain-text extracts are chunked into paragraphs suitable for embedding into
ChromaDB.

Rate-limit handling:
- Retries 429 responses with exponential backoff, respecting the Retry-After header.
- Adds a small delay between the Wikivoyage and Wikipedia fallback to avoid bursts.
- Caches failed fetches in-memory so a repeated request for the same city does
  not hammer the API again within the same process lifetime.

All network calls use httpx and accept an optional client for testability.
A descriptive User-Agent header is set per the Wikimedia API etiquette policy.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

_WIKIVOYAGE_API = "https://en.wikivoyage.org/w/api.php"
_USER_AGENT = "TripAgent/1.0 (https://github.com/srirajkavin/Trip-Agent; educational project)"
_MIN_EXTRACT_CHARS = 200
_MAX_CHUNK_CHARS = 600
_MIN_CHUNK_CHARS = 80

# Retry configuration for 429 / 5xx responses.
_MAX_RETRIES = 3
_INITIAL_BACKOFF_SECONDS = 2.0
_MAX_BACKOFF_SECONDS = 30.0

# In-memory cache of cities that failed to fetch in this process.
# Prevents repeated hammering of the API after a 429 or network failure.
# Maps "city:source" -> failure timestamp.
_failed_fetch_cache: dict[str, float] = {}
_FAILED_CACHE_TTL_SECONDS = 300.0  # 5 minutes

# Section names in Wikivoyage that list actual attractions.
_SEE_SECTION_NAMES = ["See", "Do"]


@dataclass
class ExternalDoc:
    """A single chunk of external content ready for embedding."""

    id: str
    city: str
    text: str
    source: str


@dataclass
class ExternalAttraction:
    """A parsed attraction from Wikivoyage with a real name and description."""

    name: str
    city: str
    description: str
    source: str = "wikivoyage"


def fetch_city_attractions(
    city: str,
    client: httpx.Client | None = None,
    timeout: float = 15.0,
    sleep_fn: Callable[[float], None] | None = None,
) -> list[ExternalAttraction]:
    """Fetch real attractions for a city from Wikivoyage's See/Do sections.

    Uses the parse API to get section wikitext, then extracts {{see}} and
    {{do}} template entries which contain structured name/description pairs.

    Args:
        city: City name (e.g. "Liverpool").
        client: Optional httpx client for testability.
        timeout: Request timeout in seconds.
        sleep_fn: Optional sleep function for testability.

    Returns:
        List of ExternalAttraction with real names, or empty list on failure.
    """
    normalized = city.strip().replace("_", " ").title()
    sleeper = sleep_fn or time.sleep

    section_index = _find_section_index(normalized, "See", client, timeout, sleeper)
    if section_index is None:
        section_index = _find_section_index(normalized, "Do", client, timeout, sleeper)
    if section_index is None:
        return []

    wikitext = _fetch_section_wikitext(normalized, section_index, client, timeout, sleeper)
    if not wikitext:
        return []

    return _parse_attractions_from_wikitext(wikitext, normalized)


def _find_section_index(
    city: str,
    section_name: str,
    client: httpx.Client | None,
    timeout: float,
    sleep_fn: Callable[[float], None],
) -> str | None:
    """Find the section index for a given section name via the parse API."""
    cache_key = f"{city.lower()}:sections"
    if _is_recently_failed(cache_key):
        return None

    params = {
        "action": "parse",
        "format": "json",
        "page": city,
        "prop": "sections",
    }
    headers = {"User-Agent": _USER_AGENT}

    try:
        response = _do_request(params, headers, client, timeout, sleep_fn)
        if response is None:
            _mark_failed(cache_key)
            return None

        sections = response.get("parse", {}).get("sections", [])
        for section in sections:
            if section.get("line", "").strip().lower() == section_name.lower():
                return str(section.get("index"))
    except (KeyError, TypeError, ValueError):
        _mark_failed(cache_key)
        return None

    return None


def _fetch_section_wikitext(
    city: str,
    section_index: str,
    client: httpx.Client | None,
    timeout: float,
    sleep_fn: Callable[[float], None],
) -> str:
    """Fetch the wikitext of a specific section."""
    cache_key = f"{city.lower()}:section-{section_index}"
    if _is_recently_failed(cache_key):
        return ""

    params = {
        "action": "parse",
        "format": "json",
        "page": city,
        "section": section_index,
        "prop": "wikitext",
    }
    headers = {"User-Agent": _USER_AGENT}

    try:
        response = _do_request(params, headers, client, timeout, sleep_fn)
        if response is None:
            _mark_failed(cache_key)
            return ""

        return response.get("parse", {}).get("wikitext", {}).get("*", "")
    except (KeyError, TypeError, ValueError):
        _mark_failed(cache_key)
        return ""


def _do_request(
    params: dict[str, str],
    headers: dict[str, str],
    client: httpx.Client | None,
    timeout: float,
    sleep_fn: Callable[[float], None],
) -> dict | None:
    """Make an API request with retry logic, returning parsed JSON or None."""
    backoff = _INITIAL_BACKOFF_SECONDS
    for attempt in range(_MAX_RETRIES + 1):
        try:
            if client is None:
                with httpx.Client(timeout=timeout) as owned:
                    response = owned.get(_WIKIVOYAGE_API, params=params, headers=headers)
            else:
                response = client.get(_WIKIVOYAGE_API, params=params, headers=headers)

            if response.status_code == 429:
                if attempt < _MAX_RETRIES:
                    sleep_fn(_retry_after_seconds(response, backoff))
                    backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
                    continue
                return None

            if response.status_code >= 500:
                if attempt < _MAX_RETRIES:
                    sleep_fn(backoff)
                    backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
                    continue
                return None

            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            if attempt < _MAX_RETRIES:
                sleep_fn(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
                continue
            return None

    return None


def _parse_attractions_from_wikitext(wikitext: str, city: str) -> list[ExternalAttraction]:
    """Parse {{see}} and {{do}} templates from Wikivoyage wikitext.

    Also captures '''Bold Name''' entries as a fallback format.
    """
    attractions: list[ExternalAttraction] = []

    # Format 1: {{see | name=... | content=... }} and {{do | name=... | content=... }}
    template_pattern = r"\{\{(see|do)\s*\n\| name=([^|]+)\|.*?\| content=([^}]+)\}\}"
    for _template_type, name, content in re.findall(template_pattern, wikitext, flags=re.DOTALL):
        clean_name = _clean_wikitext(name.strip())
        clean_content = _clean_wikitext(content.strip())
        if clean_name and len(clean_name) < 80:
            attractions.append(
                ExternalAttraction(name=clean_name, city=city, description=clean_content)
            )

    # Format 2: '''Bold Name''' entries (simpler listings without templates)
    bold_pattern = r"'''([^']{2,60})'''(?: is |,)?(.+?)(?=\n\n|\n\*|\n\{|$)"
    for name, desc in re.findall(bold_pattern, wikitext):
        clean_name = _clean_wikitext(name.strip())
        clean_desc = _clean_wikitext(desc.strip())
        if clean_name and not any(a.name == clean_name for a in attractions):
            attractions.append(
                ExternalAttraction(name=clean_name, city=city, description=clean_desc)
            )

    return attractions


def _clean_wikitext(text: str) -> str:
    """Strip wikitext markup from a string to produce readable plain text."""
    # Remove wiki links: [[Article|Display]] -> Display, [[Article]] -> Article
    text = re.sub(r"\[\[[^]]*\|([^]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^]]+)\]\]", r"\1", text)
    # Remove file/image links
    text = re.sub(r"\[\[(?:File|Image):[^]]+\]\]", "", text)
    # Remove templates: {{...}}
    text = re.sub(r"\{\{[^}]+\}\}", "", text)
    # Remove bold/italic markers
    text = text.replace("'''", "").replace("''", "")
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip(" -.;|")


def fetch_city_docs(
    city: str,
    client: httpx.Client | None = None,
    timeout: float = 15.0,
    sleep_fn: Callable[[float], None] | None = None,
) -> list[ExternalDoc]:
    """Fetch and chunk a city article from Wikivoyage.

    Wikivoyage is the single external source for city overviews. A previous
    Wikipedia fallback was removed because disambiguation pages (e.g.
    "Newcastle usually refers to:") were being ingested as city context and
    served as attractions. If Wikivoyage has no article for the city (or the
    exact title does not match), an empty list is returned so the caller can
    return ``no_results`` rather than serving junk.

    Args:
        city: City name to fetch content for (e.g. "Kyoto").
        client: Optional httpx client for testability.
        timeout: Request timeout in seconds.
        sleep_fn: Optional sleep function for testability (defaults to time.sleep).

    Returns:
        List of ExternalDoc chunks, or an empty list if Wikivoyage has no
        usable article for the city.
    """
    normalized = city.strip().replace("_", " ").title()
    sleeper = sleep_fn or time.sleep

    extract, source = _try_fetch_with_retries(normalized, "wikivoyage", client, timeout, sleeper)

    if not extract or len(extract) < _MIN_EXTRACT_CHARS:
        return []

    return _chunk_extract(extract, normalized, source)


def _try_fetch_with_retries(
    city: str,
    source: str,
    client: httpx.Client | None,
    timeout: float,
    sleep_fn: Callable[[float], None],
) -> tuple[str, str]:
    """Attempt to fetch from a source, retrying on 429/5xx with backoff.

    Returns (extract_text, source_name) or ("", source) on failure.
    """
    cache_key = f"{city.lower()}:{source}"
    if _is_recently_failed(cache_key):
        return "", source

    api_url = _WIKIVOYAGE_API
    params = {
        "action": "query",
        "format": "json",
        "titles": city,
        "prop": "extracts",
        "exintro": "false",
        "explaintext": "true",
    }
    headers = {"User-Agent": _USER_AGENT}

    backoff = _INITIAL_BACKOFF_SECONDS
    for attempt in range(_MAX_RETRIES + 1):
        try:
            if client is None:
                with httpx.Client(timeout=timeout) as owned:
                    response = owned.get(api_url, params=params, headers=headers)
            else:
                response = client.get(api_url, params=params, headers=headers)

            if response.status_code == 429:
                if attempt < _MAX_RETRIES:
                    wait = _retry_after_seconds(response, backoff)
                    sleep_fn(wait)
                    backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
                    continue
                _mark_failed(cache_key)
                return "", source

            if response.status_code >= 500:
                if attempt < _MAX_RETRIES:
                    sleep_fn(backoff)
                    backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
                    continue
                _mark_failed(cache_key)
                return "", source

            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            if attempt < _MAX_RETRIES:
                sleep_fn(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
                continue
            _mark_failed(cache_key)
            return "", source

        pages = data.get("query", {}).get("pages", {})
        for _page_id, page in pages.items():
            extract = page.get("extract", "")
            if extract and len(extract) >= _MIN_EXTRACT_CHARS:
                return str(extract), source

        return "", source

    _mark_failed(cache_key)
    return "", source


def _retry_after_seconds(response: httpx.Response, default: float) -> float:
    """Extract the Retry-After header value, falling back to the default backoff."""
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return min(float(retry_after), _MAX_BACKOFF_SECONDS)
        except ValueError:
            pass
    return default


def _is_recently_failed(cache_key: str) -> bool:
    """Check if a city:source was recently marked as failed and is still in cooldown."""
    failed_at = _failed_fetch_cache.get(cache_key)
    if failed_at is None:
        return False
    if time.time() - failed_at > _FAILED_CACHE_TTL_SECONDS:
        _failed_fetch_cache.pop(cache_key, None)
        return False
    return True


def _mark_failed(cache_key: str) -> None:
    """Record that a city:source fetch failed so we skip it for a while."""
    _failed_fetch_cache[cache_key] = time.time()


def clear_failed_cache() -> None:
    """Clear the in-memory failed-fetch cache. Primarily for tests."""
    _failed_fetch_cache.clear()


def _chunk_extract(extract: str, city: str, source: str) -> list[ExternalDoc]:
    """Split a plain-text extract into paragraph chunks for embedding.

    Merges very short paragraphs into the previous chunk and splits overly
    long paragraphs at sentence boundaries.
    """
    raw_paragraphs = [p.strip() for p in re.split(r"\n{2,}", extract) if p.strip()]
    chunks: list[str] = []

    for paragraph in raw_paragraphs:
        if len(paragraph) > _MAX_CHUNK_CHARS:
            chunks.extend(_split_long_paragraph(paragraph))
        elif chunks and len(paragraph) < _MIN_CHUNK_CHARS:
            chunks[-1] = chunks[-1] + " " + paragraph
        else:
            chunks.append(paragraph)

    docs: list[ExternalDoc] = []
    for index, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        docs.append(
            ExternalDoc(
                id=f"ext-{source}-{city.lower().replace(' ', '-')}-{index}",
                city=city,
                text=chunk,
                source=source,
            )
        )
    return docs


def _split_long_paragraph(paragraph: str) -> list[str]:
    """Split a paragraph longer than _MAX_CHUNK_CHARS at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + len(sentence) + 1 <= _MAX_CHUNK_CHARS:
            current = current + " " + sentence
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def external_docs_to_vectors(docs: list[ExternalDoc]) -> dict[str, list[Any]]:
    """Convert ExternalDoc chunks into the shape VectorStore.add_documents expects.

    Returns a dict with keys: documents, metadatas, ids.
    """
    return {
        "documents": [doc.text for doc in docs],
        "metadatas": [
            {"city": doc.city, "type": "city_overview", "source": f"external_{doc.source}"}
            for doc in docs
        ],
        "ids": [doc.id for doc in docs],
    }
