"""SerpAPI Google Hotels tool.

Calls the SerpAPI ``google_hotels`` engine via the SerpAPI REST endpoint
(https://serpapi.com/search) using an injectable ``httpx.Client`` so the tool
is testable with ``httpx.MockTransport`` (the repo's established test pattern).

Gracefully degrades when ``SERPAPI_API_KEY`` is missing — never crashes, never
serves junk. The returned shape mirrors ``app.tools.hotel_tool.run_hotel_tool``
(``results`` list of hotel dicts) so the orchestrator can treat mock and real
hotel tools interchangeably. Hotel images from SerpAPI (when present) are kept
on each result for the upcoming front-end image rendering.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.config import get_settings
from app.tools.budget_tool import _normalize_budget

_SERPAPI_ENDPOINT = "https://serpapi.com/search"
_DEFAULT_TIMEOUT = 30.0


def run_serpapi_hotel_tool(
    city: str,
    check_in_date: str,
    check_out_date: str,
    budget: str | None = None,
    adults: int = 1,
    rooms: int = 1,
    hotel_class: str | None = None,
    limit: int = 5,
    api_key: str | None = None,
    client: httpx.Client | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Search real hotels via SerpAPI Google Hotels.

    Args:
        city: Destination city / location query (e.g. "Paris").
        check_in_date: ISO date string (YYYY-MM-DD).
        check_out_date: ISO date string (YYYY-MM-DD).
        budget: Optional budget level (low/medium/luxury); mapped to a
            ``hotel_class`` hint when ``hotel_class`` is not provided.
        adults: Number of adults (default 1).
        rooms: Number of rooms (default 1).
        hotel_class: Optional comma-separated hotel class filter (e.g. "3,4").
        limit: Maximum number of hotels to return (default 5).
        api_key: SerpAPI key. If None, reads SERPAPI_API_KEY from settings.
        client: Optional httpx client for testability. If None, a transient
            client is created.

    Returns:
        Dict with ``tool_name`` ("serpapi_hotel_tool"), ``status`` ("ok",
        "no_results", or "error"), ``city``, ``budget_level``, and ``results``
        (list of hotel dicts: name, hotel_class, price_usd_per_night,
        price_usd_total, rating, reviews, booking_link, image, check_in_date,
        check_out_date).
    """
    normalized_city = city.strip().replace("_", " ").title()
    budget_level = _normalize_budget(budget)

    key = api_key if api_key is not None else get_settings().serpapi_api_key
    if not key:
        return _no_results(normalized_city, reason="serpapi_key_missing")

    class_filter = hotel_class or _budget_to_class(budget_level)

    params = {
        "engine": "google_hotels",
        "hl": "en",
        "gl": "us",
        "q": normalized_city,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "currency": "USD",
        "adults": str(adults),
        "rooms": str(rooms),
        "sort_by": "8",  # highest rating
        "api_key": key,
    }
    if class_filter:
        params["hotel_class"] = class_filter

    try:
        owns_client = client is None
        http_client = client or httpx.Client(timeout=timeout)
        try:
            response = http_client.get(_SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            payload = response.json()
        finally:
            if owns_client:
                http_client.close()
    except httpx.HTTPError as error:
        return _error_result(normalized_city, f"serpapi_request_failed: {error}")
    except Exception as error:  # noqa: BLE001 - defensive: never crash the agent
        return _error_result(normalized_city, f"serpapi_unexpected: {error}")

    if payload.get("error"):
        return _error_result(normalized_city, str(payload["error"]))

    hotels = _parse_hotels_payload(payload, limit)
    if not hotels:
        return _no_results(normalized_city, reason="no_properties_in_response")

    return {
        "tool_name": "serpapi_hotel_tool",
        "status": "ok",
        "city": normalized_city,
        "budget_level": budget_level,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "results": hotels,
    }


def _parse_hotels_payload(payload: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    properties = payload.get("properties") or []
    hotels: list[dict[str, Any]] = []
    for prop in properties:
        if not isinstance(prop, dict):
            continue
        hotel = _parse_property(prop)
        if hotel:
            hotels.append(hotel)
        if len(hotels) >= limit:
            break
    return hotels


def _parse_property(prop: dict[str, Any]) -> dict[str, Any] | None:
    name = prop.get("name")
    if not name:
        return None

    rate_per_night = prop.get("rate_per_night") or {}
    total_rate = prop.get("total_rate") or {}
    images = prop.get("images") or []
    image = None
    if images and isinstance(images, list):
        first = images[0]
        if isinstance(first, dict):
            image = first.get("thumbnail") or first.get("original") or first.get("image")
        elif isinstance(first, str):
            image = first

    return {
        "name": name,
        "hotel_class": prop.get("hotel_class"),
        "rating": _to_float(prop.get("rating")),
        "reviews": prop.get("reviews"),
        "price_usd_per_night": _to_float(rate_per_night.get("extracted_lowest") or rate_per_night.get("lowest")),
        "price_usd_total": _to_float(total_rate.get("extracted_lowest") or total_rate.get("lowest")),
        "check_in_date": prop.get("check_in_date"),
        "check_out_date": prop.get("check_out_date"),
        "booking_link": prop.get("link") or prop.get("booking_link"),
        "image": image,
        "address": prop.get("address") or prop.get("location"),
        "gps": prop.get("gps"),
    }


def _budget_to_class(budget_level: str) -> str | None:
    """Map a budget level to a SerpAPI hotel_class hint (optional)."""
    if budget_level == "luxury":
        return "4,5"
    if budget_level == "medium":
        return "3,4"
    if budget_level == "low":
        return "1,2,3"
    return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").replace("$", ""))
        except ValueError:
            return None
    return None


def _no_results(city: str, reason: str) -> dict[str, Any]:
    return {
        "tool_name": "serpapi_hotel_tool",
        "status": "no_results",
        "city": city,
        "reason": reason,
        "results": [],
    }


def _error_result(city: str, message: str) -> dict[str, Any]:
    return {
        "tool_name": "serpapi_hotel_tool",
        "status": "error",
        "city": city,
        "message": message,
        "results": [],
    }
