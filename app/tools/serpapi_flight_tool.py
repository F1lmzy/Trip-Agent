"""SerpAPI Google Flights tool.

Calls the SerpAPI ``google_flights`` engine via the SerpAPI REST endpoint
(https://serpapi.com/search) using an injectable ``httpx.Client`` so the tool
is testable with ``httpx.MockTransport`` (the repo's established test pattern).
The ``serpapi`` SDK is kept as the canonical service dependency, but the REST
endpoint is called directly so there is a single, mockable HTTP code path.

Gracefully degrades when ``SERPAPI_API_KEY`` is missing or the city is not
mapped to an IATA code — never crashes, never serves junk. The returned shape
mirrors ``app.tools.flight_tool.run_flight_tool`` (``results`` with
``departure_flights`` / ``return_flights`` lists) so the orchestrator can treat
mock and real flight tools interchangeably.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx

from app.config import get_settings
from app.tools.budget_tool import _normalize_budget

logger = logging.getLogger(__name__)

_SERPAPI_ENDPOINT = "https://serpapi.com/search"
_DEFAULT_TIMEOUT = 30.0

# City/city-group IATA codes accepted by SerpAPI google_flights departure_id /
# arrival_id. Intentionally a small, curated set; unmapped cities degrade to
# no_results rather than guessing a wrong airport code.
_CITY_TO_IATA: dict[str, str] = {
    "London": "LON", "Paris": "PAR", "New York": "NYC", "Tokyo": "TYO",
    "Rome": "ROM", "Madrid": "MAD", "Barcelona": "BCN", "Amsterdam": "AMS",
    "Berlin": "BER", "Munich": "MUC", "Dublin": "DUB", "Edinburgh": "EDI",
    "Manchester": "MAN", "Liverpool": "LPL", "Newcastle Upon Tyne": "NCL",
    "Newcastle": "NCL", "Lisbon": "LIS", "Vienna": "VIE", "Prague": "PRG",
    "Budapest": "BUD", "Stockholm": "STO", "Copenhagen": "CPH", "Oslo": "OSL",
    "Helsinki": "HEL", "Zurich": "ZRH", "Geneva": "GVA", "Milan": "MIL",
    "Florence": "FLR", "Venice": "VCE", "Naples": "NAP", "Athens": "ATH",
    "Istanbul": "IST", "Dubai": "DXB", "Singapore": "SIN", "Bangkok": "BKK",
    "Hong Kong": "HKG", "Seoul": "ICN", "Sydney": "SYD", "Melbourne": "MEL",
    "Toronto": "YTO", "Vancouver": "YVR", "Los Angeles": "LAX",
    "San Francisco": "SFO", "Chicago": "CHI", "Boston": "BOS",
    "Washington": "WAS", "Miami": "MIA", "Seattle": "SEA", "Las Vegas": "LAS",
}


def run_serpapi_flight_tool(
    from_location: str,
    to_location: str,
    departure_date: str,
    return_date: str | None = None,
    budget: str | None = None,
    adults: int = 1,
    api_key: str | None = None,
    client: httpx.Client | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Search real flights via SerpAPI Google Flights.

    Args:
        from_location: Departure city (must be in the city->IATA map).
        to_location: Destination city (must be in the city->IATA map).
        departure_date: ISO date string (YYYY-MM-DD).
        return_date: Optional return ISO date string (YYYY-MM-DD).
        budget: Optional budget level (low/medium/luxury); used only to tag the
            result, not to filter SerpAPI (SerpAPI ranks by price).
        adults: Number of adults (default 1).
        api_key: SerpAPI key. If None, reads SERPAPI_API_KEY from settings.
        client: Optional httpx client for testability. If None, a transient
            client is created.

    Returns:
        Dict with ``tool_name`` ("serpapi_flight_tool"), ``status`` ("ok",
        "no_results", or "error"), and ``results`` mirroring the mock flight
        tool (``departure_flights`` / ``return_flights`` lists).
    """
    normalized_from = _normalize_location(from_location)
    normalized_to = _normalize_location(to_location)
    budget_level = _normalize_budget(budget)

    key = api_key if api_key is not None else get_settings().serpapi_api_key
    if not key:
        return _no_results(normalized_from, normalized_to, reason="serpapi_key_missing")

    from_code = _CITY_TO_IATA.get(normalized_from)
    to_code = _CITY_TO_IATA.get(normalized_to)
    if not from_code:
        return _no_results(normalized_from, normalized_to, reason="departure_city_not_mapped")
    if not to_code:
        return _no_results(normalized_from, normalized_to, reason="destination_city_not_mapped")

    params = {
        "engine": "google_flights",
        "hl": "en",
        "gl": "us",
        "departure_id": from_code,
        "arrival_id": to_code,
        "outbound_date": departure_date,
        "currency": "USD",
        "adults": str(adults),
        "stops": "1",
        "api_key": key,
    }
    if return_date:
        params["return_date"] = return_date

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
        return _error_result(normalized_from, normalized_to, f"serpapi_request_failed: {error}")
    except Exception as error:  # noqa: BLE001 - defensive: never crash the agent
        logger.warning("SerpAPI flight search failed unexpectedly: %s", error)
        return _error_result(normalized_from, normalized_to, f"serpapi_unexpected: {error}")

    if payload.get("error"):
        return _error_result(normalized_from, normalized_to, str(payload["error"]))

    departure_flights, return_flights = _parse_flights_payload(payload, return_date is not None)
    if not departure_flights:
        return _no_results(normalized_from, normalized_to, reason="no_flights_in_response")

    return {
        "tool_name": "serpapi_flight_tool",
        "status": "ok",
        "from_location": normalized_from,
        "to_location": normalized_to,
        "departure_date": departure_date,
        "return_date": return_date,
        "budget_level": budget_level,
        "results": {
            "departure_flights": departure_flights,
            "return_flights": return_flights,
        },
    }


def _parse_flights_payload(payload: dict[str, Any], is_round_trip: bool) -> tuple[list[dict], list[dict]]:
    """Parse a SerpAPI google_flights response into flight dicts.

    SerpAPI returns ``best_flights`` and ``other_flights`` arrays. Each offer is
    either one-way (has a ``flights`` segment list) or round-trip (has
    ``departures`` and ``returns`` leg lists, each leg holding its own
    ``flights``). We parse defensively, skipping malformed offers.
    """
    departure_flights: list[dict] = []
    return_flights: list[dict] = []

    offers = []
    for key in ("best_flights", "other_flights"):
        offers.extend(payload.get(key) or [])

    for offer in offers:
        if not isinstance(offer, dict):
            continue
        if is_round_trip and "departures" in offer:
            for leg in offer.get("departures") or []:
                flight = _parse_leg(leg, offer)
                if flight:
                    departure_flights.append(flight)
            for leg in offer.get("returns") or []:
                flight = _parse_leg(leg, offer)
                if flight:
                    return_flights.append(flight)
        elif "flights" in offer:
            flight = _parse_leg(offer, offer)
            if flight:
                departure_flights.append(flight)

    return departure_flights, return_flights


def _parse_leg(leg: dict[str, Any], offer: dict[str, Any]) -> dict[str, Any] | None:
    segments_raw = leg.get("flights") or []
    if not segments_raw or not isinstance(segments_raw, list):
        return None

    segments = [_parse_segment(seg) for seg in segments_raw if isinstance(seg, dict)]
    segments = [seg for seg in segments if seg]
    if not segments:
        return None

    first = segments[0]
    last = segments[-1]
    total_duration = leg.get("total_duration") or offer.get("total_duration")
    duration_minutes = int(total_duration) if isinstance(total_duration, (int, float)) else sum(
        seg["duration_minutes"] for seg in segments
    )
    airlines = {seg["airline"] for seg in segments if seg.get("airline")}
    airline = next(iter(airlines)) if len(airlines) == 1 else (segments[0].get("airline") or "Multiple airlines")

    return {
        "flight_id": offer.get("flight_id") or leg.get("id"),
        "airline": airline,
        "flight_number": first.get("flight_number"),
        "price": _to_float(offer.get("price") if "price" in offer else leg.get("price")),
        "currency": "USD",
        "from_airport": first["from_airport"],
        "to_airport": last["to_airport"],
        "departure": first["departure"],
        "arrival": last["arrival"],
        "duration_minutes": duration_minutes,
        "is_direct": len(segments) == 1,
        "stops": len(segments) - 1,
        "segments": segments,
        "booking_link": offer.get("booking_link") or leg.get("booking_link"),
        "type": offer.get("type") or leg.get("type"),
    }


def _parse_segment(seg: dict[str, Any]) -> dict[str, Any] | None:
    dep_airport = seg.get("departure_airport") or {}
    arr_airport = seg.get("arrival_airport") or {}
    if not dep_airport or not arr_airport:
        return None
    return {
        "flight_number": seg.get("flight_number"),
        "airline": seg.get("airline"),
        "from_airport": {
            "code": dep_airport.get("id"),
            "name": dep_airport.get("name"),
            "city": dep_airport.get("city") or dep_airport.get("name"),
        },
        "to_airport": {
            "code": arr_airport.get("id"),
            "name": arr_airport.get("name"),
            "city": arr_airport.get("city") or arr_airport.get("name"),
        },
        "departure": dep_airport.get("time"),
        "arrival": arr_airport.get("time"),
        "duration_minutes": int(seg["duration"]) if isinstance(seg.get("duration"), (int, float)) else 0,
        "airplane": seg.get("airplane"),
        "travel_class": seg.get("travel_class"),
    }


def _normalize_location(location: str) -> str:
    return location.strip().replace("_", " ").title()


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return None
    return None


def _no_results(from_location: str, to_location: str, reason: str) -> dict[str, Any]:
    return {
        "tool_name": "serpapi_flight_tool",
        "status": "no_results",
        "from_location": from_location,
        "to_location": to_location,
        "reason": reason,
        "results": {"departure_flights": [], "return_flights": []},
    }


def _error_result(from_location: str, to_location: str, message: str) -> dict[str, Any]:
    return {
        "tool_name": "serpapi_flight_tool",
        "status": "error",
        "from_location": from_location,
        "to_location": to_location,
        "message": message,
        "results": {"departure_flights": [], "return_flights": []},
    }
