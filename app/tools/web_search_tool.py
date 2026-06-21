import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class SearchTool(Protocol):
        ...


def run_destination_search_tool(
    query_intent: str,
    count: int = 5,
    search_tool: SearchTool | None = None,
) -> dict[str, Any]:
    query = _build_destination_query(query_intent)

    try:
        tool = search_tool or _build_duckduckgo_tool(count=count)
        raw_results = tool.invoke(query)
        results = _normalize_results(raw_results)
    except ImportError:
        return _fallback_result(
            city="",
            query=query,
            status="fallback_missing_dependency",
            message="DuckDuckGo search unavailable because langchain-community and ddgs are not installed.",
        )
    except Exception:
        logger.warning(
            "Web search (destination) failed unexpectedly; returning fallback",
            exc_info=True,
        )
        return _fallback_result(
            city="",
            query=query,
            status="fallback_search_error",
            message="DuckDuckGo search unavailable because the LangChain search tool failed.",
        )

    return {
        "tool_name": "web_search_tool",
        "status": "ok",
        "city": "",
        "source": "duckduckgo_langchain",
        "query": query,
        "results": results,
    }


def run_web_search_tool(
    city: str,
    query_intent: str,
    count: int = 5,
    search_tool: SearchTool | None = None,
) -> dict[str, Any]:
    normalized_city = city.strip().replace("_", " ").title()
    query = _build_query(normalized_city, query_intent)

    try:
        tool = search_tool or _build_duckduckgo_tool(count=count)
        raw_results = tool.invoke(query)
        results = _normalize_results(raw_results)
    except ImportError:
        return _fallback_result(
            city=normalized_city,
            query=query,
            status="fallback_missing_dependency",
            message="DuckDuckGo search unavailable because langchain-community and ddgs are not installed.",
        )
    except Exception:
        logger.warning(
            "Web search failed unexpectedly; returning fallback",
            exc_info=True,
        )
        return _fallback_result(
            city=normalized_city,
            query=query,
            status="fallback_search_error",
            message="DuckDuckGo search unavailable because the LangChain search tool failed.",
        )

    return {
        "tool_name": "web_search_tool",
        "status": "ok",
        "city": normalized_city,
        "source": "duckduckgo_langchain",
        "query": query,
        "results": results,
    }


def _build_duckduckgo_tool(count: int) -> SearchTool:
    from langchain_community.tools import DuckDuckGoSearchResults

    return DuckDuckGoSearchResults(
        num_results=max(1, min(count, 10)),
        output_format="list",
        keys_to_include=["title", "link", "snippet"],
    )


def _build_destination_query(query_intent: str) -> str:
    normalized_intent = " ".join(query_intent.strip().split()) or "best travel destinations"
    return f"{normalized_intent} travel destinations"


def _build_query(city: str, query_intent: str) -> str:
    normalized_intent = " ".join(query_intent.strip().split()) or "current travel information"
    return f"{city} {normalized_intent} travel"


def _normalize_results(raw_results: Any) -> list[dict[str, str]]:
    if isinstance(raw_results, tuple):
        raw_results = raw_results[0]
    if not isinstance(raw_results, list):
        return []

    normalized: list[dict[str, str]] = []
    for result in raw_results:
        if not isinstance(result, dict):
            continue
        normalized.append(
            {
                "title": str(result.get("title", "")),
                "url": str(result.get("link") or result.get("href") or result.get("url") or ""),
                "description": str(result.get("snippet") or result.get("body") or result.get("description") or ""),
            }
        )
    return normalized


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
