"""Mock flight suggestion tool.

Generates deterministic-ish mock flight suggestions for a city pair and dates.
Modeled on the Azure AI Travel Agents sample's itinerary-planning MCP server,
but kept self-contained and synchronous to match this project's tool conventions.

Returns a dict with `tool_name`, `status`, and a `results` payload, mirroring
the other tools in this package.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

from app.tools.budget_tool import _normalize_budget


@dataclass
class Airport:
    code: str
    name: str
    city: str


@dataclass
class FlightSegment:
    flight_number: str
    from_airport: Airport
    to_airport: Airport
    departure: str
    arrival: str
    duration_minutes: int


@dataclass
class FlightConnection:
    airport_code: str
    duration_minutes: int


@dataclass
class Flight:
    flight_id: str
    airline: str
    flight_number: str
    aircraft: str
    from_airport: Airport
    to_airport: Airport
    departure: str
    arrival: str
    duration_minutes: int
    is_direct: bool
    price: float
    currency: str
    available_seats: int
    cabin_class: str
    segments: list[FlightSegment] = field(default_factory=list)
    connection: FlightConnection | None = None


@dataclass
class FlightSuggestions:
    departure_flights: list[Flight]
    return_flights: list[Flight]


_AIRLINES = [
    "SkyWings",
    "Global Air",
    "Atlantic Airways",
    "Pacific Express",
    "Mountain Jets",
    "Stellar Airlines",
]
_AIRCRAFT = ["Boeing 737", "Airbus A320", "Boeing 787", "Airbus A350", "Embraer E190"]
_CABIN_CLASSES = ["Economy", "Premium Economy", "Business", "First"]
_CONNECTION_CODES = ["ATL", "ORD", "DFW", "LHR", "CDG", "DXB", "AMS", "FRA"]

_PRICE_BY_BUDGET = {
    "low": (99, 299),
    "medium": (200, 599),
    "luxury": (600, 1499),
}


def run_flight_tool(
    from_location: str,
    to_location: str,
    departure_date: str,
    return_date: str | None = None,
    budget: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Suggest mock flights between two locations for given dates.

    Args:
        from_location: Departure city or airport.
        to_location: Destination city or airport.
        departure_date: ISO date string (YYYY-MM-DD).
        return_date: Optional return ISO date string (YYYY-MM-DD).
        budget: Optional budget level (low/medium/luxury) used to price flights.
        seed: Optional integer seed for deterministic output (useful in tests).

    Returns:
        Dict with `tool_name`, `status`, normalized locations, and `results`.
    """
    normalized_from = _normalize_location(from_location)
    normalized_to = _normalize_location(to_location)
    budget_level = _normalize_budget(budget)
    if budget_level not in _PRICE_BY_BUDGET:
        budget_level = "medium"

    try:
        dep_date = _validate_iso_date(departure_date, "departure_date")
    except ValueError as error:
        return _error_result(str(error), normalized_from, normalized_to)

    ret_date: datetime.date | None = None
    if return_date:
        try:
            ret_date = _validate_iso_date(return_date, "return_date")
        except ValueError as error:
            return _error_result(str(error), normalized_from, normalized_to)
        if ret_date <= dep_date:
            return _error_result(
                "return_date must be after departure_date",
                normalized_from,
                normalized_to,
            )

    rng = _SeededRandom(seed)
    from_code = _airport_code(normalized_from, rng)
    to_code = _airport_code(normalized_to, rng)

    departure_flights = [
        _build_flight(normalized_from, normalized_to, from_code, to_code, dep_date, budget_level, rng)
        for _ in range(rng.randint(3, 5))
    ]
    return_flights: list[Flight] = []
    if ret_date is not None:
        return_flights = [
            _build_flight(normalized_to, normalized_from, to_code, from_code, ret_date, budget_level, rng)
            for _ in range(rng.randint(3, 5))
        ]

    suggestions = FlightSuggestions(departure_flights=departure_flights, return_flights=return_flights)
    return {
        "tool_name": "flight_tool",
        "status": "ok",
        "from_location": normalized_from,
        "to_location": normalized_to,
        "departure_date": departure_date,
        "return_date": return_date,
        "budget_level": budget_level,
        "results": _suggestions_to_dict(suggestions),
    }


def _build_flight(
    from_city: str,
    to_city: str,
    from_code: str,
    to_code: str,
    date: datetime.date,
    budget_level: str,
    rng: "_SeededRandom",
) -> Flight:
    from_airport = Airport(code=from_code, name=f"{from_city} International Airport", city=from_city)
    to_airport = Airport(code=to_code, name=f"{to_city} International Airport", city=to_city)

    hour = rng.randint(6, 22)
    minute = rng.choice([0, 15, 30, 45])
    dep_time = datetime.combine(date, datetime.min.time()).replace(hour=hour, minute=minute)
    flight_minutes = rng.randint(60, 480)
    arr_time = dep_time + timedelta(minutes=flight_minutes)

    is_direct = rng.random() < 0.6
    segments: list[FlightSegment] = []
    connection: FlightConnection | None = None

    if not is_direct:
        connection_code = rng.choice(_CONNECTION_CODES)
        segment1_duration = round(flight_minutes * rng.uniform(0.3, 0.7))
        segment2_duration = flight_minutes - segment1_duration
        connection_time = rng.randint(45, 180)
        segment1_arrival = dep_time + timedelta(minutes=segment1_duration)
        segment2_departure = segment1_arrival + timedelta(minutes=connection_time)
        connection_airport = Airport(
            code=connection_code,
            name=f"{connection_code} International Airport",
            city=connection_code,
        )
        segments = [
            FlightSegment(
                flight_number=rng.flight_number(),
                from_airport=from_airport,
                to_airport=connection_airport,
                departure=dep_time.isoformat(),
                arrival=segment1_arrival.isoformat(),
                duration_minutes=segment1_duration,
            ),
            FlightSegment(
                flight_number=rng.flight_number(),
                from_airport=connection_airport,
                to_airport=to_airport,
                departure=segment2_departure.isoformat(),
                arrival=arr_time.isoformat(),
                duration_minutes=segment2_duration,
            ),
        ]
        connection = FlightConnection(airport_code=connection_code, duration_minutes=connection_time)

    price_range = _PRICE_BY_BUDGET[budget_level]
    return Flight(
        flight_id=rng.flight_id(),
        airline=rng.choice(_AIRLINES),
        flight_number=rng.flight_number(),
        aircraft=rng.choice(_AIRCRAFT),
        from_airport=from_airport,
        to_airport=to_airport,
        departure=dep_time.isoformat(),
        arrival=arr_time.isoformat(),
        duration_minutes=flight_minutes,
        is_direct=is_direct,
        price=round(rng.uniform(*price_range), 2),
        currency="USD",
        available_seats=rng.randint(1, 30),
        cabin_class=rng.choice(_CABIN_CLASSES),
        segments=segments,
        connection=connection,
    )


def _suggestions_to_dict(suggestions: FlightSuggestions) -> dict[str, Any]:
    return {
        "departure_flights": [_flight_to_dict(flight) for flight in suggestions.departure_flights],
        "return_flights": [_flight_to_dict(flight) for flight in suggestions.return_flights],
    }


def _flight_to_dict(flight: Flight) -> dict[str, Any]:
    data = asdict(flight)
    return data


def _airport_code(city: str, rng: "_SeededRandom") -> str:
    vowels = "AEIOU"
    consonants = "BCDFGHJKLMNPQRSTVWXYZ"
    first_char = city[0].upper() if city else rng.choice(consonants)
    code = first_char if first_char in consonants else rng.choice(consonants)
    for _ in range(2):
        code += rng.choice(consonants) if rng.random() < 0.7 else rng.choice(vowels)
    return code


def _validate_iso_date(date_str: str, param_name: str) -> datetime.date:
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        raise ValueError(f"{param_name} must be in ISO format (YYYY-MM-DD), got: {date_str}")
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _normalize_location(location: str) -> str:
    return location.strip().replace("_", " ").title()


def _error_result(message: str, from_location: str, to_location: str) -> dict[str, Any]:
    return {
        "tool_name": "flight_tool",
        "status": "error",
        "from_location": from_location,
        "to_location": to_location,
        "results": {"departure_flights": [], "return_flights": []},
        "message": message,
    }


class _SeededRandom:
    """Small deterministic RNG wrapper so tests can request stable output."""

    def __init__(self, seed: int | None) -> None:
        import random

        self._rng = random.Random(seed) if seed is not None else random.Random()

    def randint(self, low: int, high: int) -> int:
        return self._rng.randint(low, high)

    def choice(self, items: list[Any]) -> Any:
        return self._rng.choice(items)

    def uniform(self, low: float, high: float) -> float:
        return self._rng.uniform(low, high)

    def random(self) -> float:
        return self._rng.random()

    def flight_number(self) -> str:
        return f"{self._rng.choice('ABCDEFG')}{self._rng.randint(100, 9999)}"

    def flight_id(self) -> str:
        return f"{self._rng.randint(0, 0xFFFFFFFF):08X}"
