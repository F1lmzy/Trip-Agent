from typing import Any

import httpx

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


def run_web_search_tool(
    city: str,
    query_intent: str,
    api_key: str | None = None,
    count: int = 5,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    normalized_city = city.strip().replace("_", " ").title()
    query = _build_query(normalized_city, query_intent)

    if not api_key:
        return _fallback_result(
            city=normalized_city,
            query=query,
            status="fallback_missing_api_key",
            message="Web search unavailable because BRAVE_SEARCH_API_KEY is not configured.",
        )

    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
    }
    params = {
        "q": query,
        "count": max(1, min(count, 20)),
        "search_lang": "en",
    }

    try:
        if client is None:
            with httpx.Client(timeout=10) as owned_client:
                response = owned_client.get(BRAVE_SEARCH_URL, headers=headers, params=params)
        else:
            response = client.get(BRAVE_SEARCH_URL, headers=headers, params=params)

        response.raise_for_status()
        payload = response.json()
        results = _normalize_results(payload)
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return _fallback_result(
            city=normalized_city,
            query=query,
            status="fallback_api_error",
            message="Web search unavailable because Brave Search failed or returned an invalid response.",
        )

    return {
        "tool_name": "web_search_tool",
        "status": "ok",
        "city": normalized_city,
        "source": "brave_search",
        "query": query,
        "results": results,
    }


def _build_query(city: str, query_intent: str) -> str:
    normalized_intent = " ".join(query_intent.strip().split()) or "current travel information"
    return f"{city} {normalized_intent} travel"


def _normalize_results(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw_results = payload.get("web", {}).get("results", [])
    return [
        {
            "title": str(result.get("title", "")),
            "url": str(result.get("url", "")),
            "description": str(result.get("description", "")),
        }
        for result in raw_results
    ]


def _fallback_result(city: str, query: str, status: str, message: str) -> dict[str, Any]:
    return {
        "tool_name": "web_search_tool",
        "status": status,
        "city": city,
        "source": "fallback",
        "query": query,
        "results": [],
        "message": message,
    }
