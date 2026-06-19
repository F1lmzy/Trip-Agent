"""Fetch and chunk external travel content for RAG ingestion.

Fetches city articles from Wikivoyage (travel-focused) with a fallback to
Wikipedia (general encyclopedic content) via their Action API. The returned
plain-text extracts are chunked into paragraphs suitable for embedding into
ChromaDB.

All network calls use httpx and accept an optional client for testability.
A User-Agent header is set per the Wikimedia API etiquette policy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx

_WIKIVOYAGE_API = "https://en.wikivoyage.org/w/api.php"
_WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
_USER_AGENT = "TripAgent/1.0 (educational project; contact@example.com)"
_MIN_EXTRACT_CHARS = 200
_MAX_CHUNK_CHARS = 600
_MIN_CHUNK_CHARS = 80


@dataclass
class ExternalDoc:
    """A single chunk of external content ready for embedding."""

    id: str
    city: str
    text: str
    source: str


def fetch_city_docs(
    city: str,
    client: httpx.Client | None = None,
    timeout: float = 15.0,
) -> list[ExternalDoc]:
    """Fetch and chunk a city article from Wikivoyage, falling back to Wikipedia.

    Args:
        city: City name to fetch content for (e.g. "Kyoto").
        client: Optional httpx client for testability.
        timeout: Request timeout in seconds.

    Returns:
        List of ExternalDoc chunks, or an empty list if both sources fail.
    """
    normalized = city.strip().replace("_", " ").title()

    extract, source = _try_fetch(normalized, "wikivoyage", client, timeout)
    if not extract or len(extract) < _MIN_EXTRACT_CHARS:
        extract, source = _try_fetch(normalized, "wikipedia", client, timeout)

    if not extract or len(extract) < _MIN_EXTRACT_CHARS:
        return []

    return _chunk_extract(extract, normalized, source)


def _try_fetch(
    city: str,
    source: str,
    client: httpx.Client | None,
    timeout: float,
) -> tuple[str, str]:
    """Attempt to fetch a plain-text extract from a given source.

    Returns (extract_text, source_name) or ("", source) on failure.
    """
    api_url = _WIKIVOYAGE_API if source == "wikivoyage" else _WIKIPEDIA_API
    params = {
        "action": "query",
        "format": "json",
        "titles": city,
        "prop": "extracts",
        "exintro": "false",
        "explaintext": "true",
    }
    headers = {"User-Agent": _USER_AGENT}

    try:
        if client is None:
            with httpx.Client(timeout=timeout) as owned:
                response = owned.get(api_url, params=params, headers=headers)
        else:
            response = client.get(api_url, params=params, headers=headers)

        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return "", source

    pages = data.get("query", {}).get("pages", {})
    for _page_id, page in pages.items():
        extract = page.get("extract", "")
        if extract and len(extract) >= _MIN_EXTRACT_CHARS:
            return str(extract), source

    return "", source


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
