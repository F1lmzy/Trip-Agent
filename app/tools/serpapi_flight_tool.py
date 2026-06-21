"""SerpAPI Google Flights tool.

Calls the SerpAPI ``google_flights`` engine via the SerpAPI REST endpoint
(https://serpapi.com/search) using an injectable ``httpx.Client`` so the tool
is testable with ``httpx.MockTransport`` (the repo's established test pattern).
The ``serpapi`` SDK is kept as the canonical service dependency, but the REST
endpoint is called directly so there is a single, mockable HTTP code path.

Gracefully degrades when ``SERPAPI_API_KEY`` is missing or a city cannot be
resolved to an IATA code — never crashes, never serves junk. The returned shape
mirrors ``app.tools.flight_tool.run_flight_tool`` (``results`` with
``departure_flights`` / ``return_flights`` lists) so the orchestrator can treat
mock and real flight tools interchangeably.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

import httpx

from app.config import get_settings
from app.tools.budget_tool import _normalize_budget

logger = logging.getLogger(__name__)

_SERPAPI_ENDPOINT = "https://serpapi.com/search"
_DEFAULT_TIMEOUT = 30.0

# Preferred airports for ambiguous multi-airport cities and common tourist
# places whose nearest practical airport is in a neighboring city. The dynamic
# resolver below handles the broad world airport database; these entries keep
# high-traffic ambiguous prompts stable and avoid SerpAPI-rejected city codes
# such as NYC, LON, TYO, CHI, etc.
_CITY_TO_IATA: dict[str, str] = {
    "London": "LHR", "Paris": "CDG", "New York": "JFK", "Tokyo": "NRT",
    "Rome": "FCO", "Madrid": "MAD", "Barcelona": "BCN", "Amsterdam": "AMS",
    "Berlin": "BER", "Munich": "MUC", "Dublin": "DUB", "Edinburgh": "EDI",
    "Manchester": "MAN", "Liverpool": "LPL", "Newcastle Upon Tyne": "NCL",
    "Newcastle": "NCL", "Lisbon": "LIS", "Vienna": "VIE", "Prague": "PRG",
    "Budapest": "BUD", "Stockholm": "ARN", "Copenhagen": "CPH", "Oslo": "OSL",
    "Helsinki": "HEL", "Zurich": "ZRH", "Geneva": "GVA", "Milan": "MXP",
    "Florence": "FLR", "Venice": "VCE", "Naples": "NAP", "Athens": "ATH",
    "Istanbul": "IST", "Dubai": "DXB", "Singapore": "SIN", "Bangkok": "BKK", "Osaka": "KIX",
    "Hong Kong": "HKG", "Seoul": "ICN", "Sydney": "SYD", "Melbourne": "MEL",
    "Toronto": "YYZ", "Vancouver": "YVR", "Los Angeles": "LAX",
    "San Francisco": "SFO", "Chicago": "ORD", "Boston": "BOS",
    "Washington": "DCA", "Miami": "MIA", "Seattle": "SEA", "Las Vegas": "LAS",
    "Kyoto": "KIX", "Oxford": "LHR", "Cambridge": "STN", "Anaheim": "LAX",
}


@lru_cache(maxsize=1)
def _airport_index() -> dict[str, dict[str, Any]]:
    try:
        import airportsdata

        return airportsdata.load("IATA")
    except Exception as error:  # noqa: BLE001 - optional resolver dependency
        logger.warning("Airport database unavailable; falling back to curated city map: %s", error)
        return {}


def _resolve_iata_code(location: str) -> str | None:
    """Resolve a city/airport prompt to a SerpAPI-compatible airport IATA code.

    SerpAPI Google Flights accepts real airport IATA codes, not broad city codes
    such as NYC or LON. This resolver first honors explicit IATA prompts and the
    curated ambiguity map, then falls back to the bundled airportsdata database
    so cities outside the hand-maintained list (Austin, Beijing, Osaka, etc.)
    work without adding entries one by one.
    """
    cleaned = " ".join(location.replace(",", " ").split()).strip()
    if not cleaned:
        return None

    airports = _airport_index()
    explicit_code = cleaned.upper()
    if len(explicit_code) == 3 and explicit_code in airports:
        return explicit_code

    normalized = _normalize_location(cleaned)
    if normalized in _CITY_TO_IATA:
        return _CITY_TO_IATA[normalized]

    query = normalized.casefold()
    matches: list[tuple[int, str]] = []
    for code, airport in airports.items():
        city = str(airport.get("city") or "").casefold()
        name = str(airport.get("name") or "").casefold()
        if city == query:
            matches.append((_airport_score(airport, exact_city=True), code))
        elif re.search(rf"\b{re.escape(query)}\b", name):
            matches.append((_airport_score(airport, exact_city=False), code))

    if not matches:
        return None

    matches.sort(key=lambda item: (-item[0], item[1]))
    return matches[0][1]


def _airport_score(airport: dict[str, Any], exact_city: bool) -> int:
    name = str(airport.get("name") or "").casefold()
    score = 100 if exact_city else 40
    if "international" in name:
        score += 30
    if any(term in name for term in ("capital", "heathrow", "kennedy", "dulles", "major")):
        score += 10
    if any(term in name for term in ("municipal", "regional", "county", "airpark", "heliport", "seaplane")):
        score -= 25
    return score


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
        from_location: Departure city or IATA airport code.
        to_location: Destination city or IATA airport code.
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

    from_code = _resolve_iata_code(normalized_from)
    to_code = _resolve_iata_code(normalized_to)
    if not from_code:
        return _no_results(normalized_from, normalized_to, reason="departure_city_not_resolved")
    if not to_code:
        return _no_results(normalized_from, normalized_to, reason="destination_city_not_resolved")

    params = {
        "engine": "google_flights",
        "hl": "en",
        "gl": "us",
        "departure_id": from_code,
        "arrival_id": to_code,
        "outbound_date": departure_date,
        "currency": "USD",
        "adults": str(adults),
        "api_key": key,
    }
    if return_date:
        params["type"] = "1"
        params["return_date"] = return_date
    else:
        params["type"] = "2"

    try:
        owns_client = client is None
        http_client = client or httpx.Client(timeout=timeout)
        try:
            response = http_client.get(_SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                return _error_result(normalized_from, normalized_to, str(payload["error"]), departure_date, return_date, from_code, to_code)

            departure_flights, return_flights = _parse_flights_payload(payload, return_date is not None)
            departure_token = _first_departure_token(payload)
            if return_date and departure_flights and not return_flights and departure_token:
                fetched_returns = _fetch_return_flights(
                    http_client=http_client,
                    base_params=params,
                    departure_token=departure_token,
                )
                if fetched_returns is not None:
                    return_flights = fetched_returns
        finally:
            if owns_client:
                http_client.close()
    except httpx.HTTPError as error:
        return _error_result(normalized_from, normalized_to, f"serpapi_request_failed: {error}", departure_date, return_date, from_code, to_code)
    except Exception as error:  # noqa: BLE001 - defensive: never crash the agent
        logger.warning("SerpAPI flight search failed unexpectedly: %s", error)
        return _error_result(normalized_from, normalized_to, f"serpapi_unexpected: {error}", departure_date, return_date, from_code, to_code)

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
        "departure_id": from_code,
        "arrival_id": to_code,
        "departure_token": departure_token,
        "results": {
            "departure_flights": departure_flights,
            "return_flights": return_flights,
        },
    }


def _first_departure_token(payload: dict[str, Any]) -> str | None:
    for key in ("best_flights", "other_flights"):
        for offer in payload.get(key) or []:
            if isinstance(offer, dict) and offer.get("departure_token"):
                return str(offer["departure_token"])
    return None


def _fetch_return_flights(
    http_client: httpx.Client,
    base_params: dict[str, Any],
    departure_token: str,
) -> list[dict] | None:
    """Fetch SerpAPI's second-step return flights for a selected outbound.

    SerpAPI's first round-trip request often returns only outbound choices plus
    a departure_token. Per SerpAPI docs, return flight options require a second
    request with that token. If the token lookup fails, keep the outbound
    results instead of converting the whole flight section to an error.
    """
    full_token_params = dict(base_params)
    full_token_params["departure_token"] = departure_token
    compact_token_params = {
        key: value
        for key, value in full_token_params.items()
        if key not in {"departure_id", "arrival_id", "outbound_date"}
    }

    for token_params in (full_token_params, compact_token_params):
        try:
            response = http_client.get(_SERPAPI_ENDPOINT, params=token_params)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as error:
            logger.warning("SerpAPI return flight token lookup failed: %s", error)
            continue

        if payload.get("error"):
            logger.warning("SerpAPI return flight lookup failed: %s", payload["error"])
            continue

        return_flights, _ = _parse_flights_payload(payload, is_round_trip=False)
        if return_flights:
            return return_flights

    return None


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
            continue

        # SerpAPI's documented google_flights response uses a flat "flights"
        # list even when return_date/type=1 are present. Treat that as the
        # outbound leg rather than returning no_results and triggering mock
        # fallback in the orchestrator.
        if "flights" in offer:
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


def _error_result(
    from_location: str,
    to_location: str,
    message: str,
    departure_date: str | None = None,
    return_date: str | None = None,
    departure_id: str | None = None,
    arrival_id: str | None = None,
) -> dict[str, Any]:
    return {
        "tool_name": "serpapi_flight_tool",
        "status": "error",
        "from_location": from_location,
        "to_location": to_location,
        "departure_date": departure_date,
        "return_date": return_date,
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "message": message,
        "results": {"departure_flights": [], "return_flights": []},
    }
