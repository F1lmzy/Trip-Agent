import re

from pydantic import BaseModel, Field


KNOWN_CITIES = [
    "Tokyo",
    "Singapore",
    "Paris",
    "New York",
    "Mumbai",
    "London",
    "Rome",
    "Barcelona",
    "Bangkok",
    "Dubai",
    "Seoul",
    "Reykjavik",
]

INTEREST_KEYWORDS = {
    "food": ["food", "restaurant", "restaurants", "eat", "eats", "dining", "cafe", "cafes"],
    "anime": ["anime", "manga", "gaming", "arcade"],
    "photography": ["photography", "photo", "photos", "viewpoint", "views"],
    "museums": ["museum", "museums", "gallery", "galleries"],
    "nature": ["nature", "garden", "gardens", "park", "parks", "outdoors"],
    "nightlife": ["nightlife", "bars", "clubs"],
    "shopping": ["shopping", "shops", "market", "markets"],
    "culture": ["culture", "cultural", "temple", "temples", "heritage"],
    "history": ["history", "historic", "historical"],
    "family": ["family", "family-friendly", "kids", "children"],
    "art": ["art"],
    "architecture": ["architecture", "buildings"],
    "beaches": ["beach", "beaches"],
}

BUDGET_KEYWORDS = {
    "low": [
        "cheap",
        "low budget",
        "low-budget",
        "affordable",
        "low-cost",
        "budget-friendly",
        "small budget",
        "small-budget",
        "tight budget",
        "tight-budget",
        "shoestring",
        "on a budget",
        "budget trip",
        "limited budget",
    ],
    "medium": ["moderate", "medium", "mid-range", "midrange"],
    "luxury": ["luxury", "premium", "high-end", "expensive"],
}

DIETARY_KEYWORDS = ["vegetarian", "vegan", "halal", "kosher", "gluten-free", "gluten free"]
CONSTRAINT_KEYWORDS = ["wheelchair", "accessible", "pet-friendly", "pet friendly", "indoor"]
CURRENT_INFO_KEYWORDS = [
    "current",
    "recent",
    "latest",
    "today",
    "this week",
    "events",
    "closure",
    "closed",
    "new attraction",
    "new attractions",
    "recent food",
]


class ParsedRequest(BaseModel):
    raw_message: str
    city: str | None = None
    duration_days: int = 2
    interests: list[str] = Field(default_factory=list)
    budget: str | None = None
    dates: str | None = None
    departure_date: str | None = None
    return_date: str | None = None
    travel_style: str | None = None
    dietary_needs: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    asks_for_hotel: bool = False
    asks_for_current_info: bool = False
    asks_for_flights: bool = False
    origin_city: str | None = None
    trip_type: str = "round_trip"
    is_follow_up: bool = False
    follow_up_intent: str | None = None


def parse_user_request(message: str) -> ParsedRequest:
    normalized = message.lower().strip()
    origin_city = _extract_origin_city(message)

    return ParsedRequest(
        raw_message=message,
        city=_extract_city(message, origin_city=origin_city),
        duration_days=_extract_duration_days(normalized),
        interests=_extract_interests(normalized),
        budget=_extract_budget(normalized),
        dates=_extract_dates(message),
        departure_date=_extract_departure_date(message),
        return_date=_extract_return_date(message),
        travel_style=_extract_travel_style(normalized),
        dietary_needs=_extract_keywords(normalized, DIETARY_KEYWORDS),
        constraints=_extract_keywords(normalized, CONSTRAINT_KEYWORDS),
        asks_for_hotel=_contains_any(
            normalized, ["hotel", "hotels", "stay", "accommodation", "lodging", "place to stay", "places to stay"]
        ),
        asks_for_current_info=_contains_any(normalized, CURRENT_INFO_KEYWORDS),
        asks_for_flights=_contains_any(
            normalized, ["flight", "flights", "fly", "airfare", "plane", "air ticket", "air tickets"]
        ),
        origin_city=origin_city,
        trip_type=_extract_trip_type(normalized),
        is_follow_up=_extract_follow_up_intent(normalized) is not None,
        follow_up_intent=_extract_follow_up_intent(normalized),
    )


def _extract_city(message: str, origin_city: str | None = None) -> str | None:
    # Route phrases. The destination is whichever city is paired with "to",
    # regardless of word order, and is accepted even when it is not a known
    # city (e.g. "to kyoto from singapore" -> Kyoto, "from london to milan"
    # -> Milan). The origin is never mistaken for the destination.
    origin_lower = (origin_city or "").strip().lower()
    return (
        _extract_route_city(message)
        or _extract_reversed_city(message)
        or _extract_verb_city(message, origin_lower)
        or _extract_known_city(message, origin_lower)
        or _extract_dynamic_city_phrase(message)
    )


def _known_city_or_candidate(candidate: str) -> str:
    for city in KNOWN_CITIES:
        if city.lower() == candidate.lower():
            return city
    return candidate


def _extract_route_city(message: str) -> str | None:
    # Order 1: "from <origin> to <destination>".
    route_match = re.search(
        r"\bfrom\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+to\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
        message,
        flags=re.IGNORECASE,
    )
    if not route_match:
        return None
    dest = _clean_city_candidate(route_match.group(2))
    return _known_city_or_candidate(dest) if dest else None


def _extract_reversed_city(message: str) -> str | None:
    # Order 2: "to <destination> from <origin>" (reversed wording).
    # The destination capture excludes origin-marker verbs (flying/departing/
    # leaving) so "to Tokyo flying from London" yields Tokyo, not "Tokyo Flying".
    reversed_match = re.search(
        r"\bto\s+([A-Za-z]+(?:\s+(?!flying|departing|leaving|from\b)[A-Za-z]+)?)\s+from\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
        message,
        flags=re.IGNORECASE,
    )
    if not reversed_match:
        return None
    dest = _clean_city_candidate(reversed_match.group(1))
    return _known_city_or_candidate(dest) if dest else None


def _extract_verb_city(message: str, origin_lower: str) -> str | None:
    # "<verb> to <destination>" with no explicit origin (e.g. "trip to tokyo",
    # "fly to paris"). Only accept known cities here so common verbs like
    # "to plan" / "to visit" are not mistaken for a destination. Skip the
    # origin so "flying from tokyo" (no destination) does not pick Tokyo.
    to_match = re.search(r"\bto\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)", message, flags=re.IGNORECASE)
    if not to_match:
        return None
    candidate = _clean_city_candidate(to_match.group(1))
    if not candidate or candidate.lower() == origin_lower:
        return None
    for city in KNOWN_CITIES:
        if city.lower() == candidate.lower():
            return city
    return None


def _extract_known_city(message: str, origin_lower: str) -> str | None:
    # Known-cities fallback: the first known city mentioned. Skip the origin so
    # a message that names only an origin ("I'm flying from Singapore, where
    # should I go?") does not treat that origin as the destination — instead
    # city stays None and the DestinationRecommendationAgent suggests places.
    for city in KNOWN_CITIES:
        if city.lower() == origin_lower:
            continue
        if re.search(rf"\b{re.escape(city)}\b", message, flags=re.IGNORECASE):
            return city
    return None


def _extract_dynamic_city_phrase(message: str) -> str | None:
    patterns = [
        r"(?:trip to|visit|in|for)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return _clean_city_candidate(match.group(1))
    return None


def _clean_city_candidate(candidate: str) -> str | None:
    stop_words = {"with", "for", "from", "on", "and", "or", "that", "where", "to"}
    words = candidate.strip(" .,!?").split()
    while words and words[-1].lower() in stop_words:
        words.pop()
    if not words:
        return None
    return " ".join(words).title()


def _extract_duration_days(normalized: str) -> int:
    match = re.search(r"\b(\d{1,2})\s*-?\s*days?\b", normalized)
    if match:
        return int(match.group(1))
    match = re.search(r"\bfor\s+(\d{1,2})\s+days?\b", normalized)
    if match:
        return int(match.group(1))
    return 2


def _extract_interests(normalized: str) -> list[str]:
    interests: list[str] = []
    for interest, keywords in INTEREST_KEYWORDS.items():
        if _contains_any(normalized, keywords):
            interests.append(interest)
    return interests


def _extract_budget(normalized: str) -> str | None:
    for budget, keywords in BUDGET_KEYWORDS.items():
        if _contains_any(normalized, keywords):
            return budget
    return None


def _extract_dates(message: str) -> str | None:
    month_pattern = (
        r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|"
        r"Sep|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2}(?:st|nd|rd|th)?"
        r"(?:\s*-\s*\d{1,2}(?:st|nd|rd|th)?)?\b"
    )
    match = re.search(month_pattern, message, flags=re.IGNORECASE)
    return match.group(0) if match else None


_MONTHS = (
    "Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|"
    "Sep|September|Oct|October|Nov|November|Dec|December"
)
# Day number with an optional ordinal suffix (1st, 2nd, 3rd, 4th, 25th).
_DAY = r"\d{1,2}(?:st|nd|rd|th)?"


def _parse_month_day(token: str, year: int) -> str | None:
    """Parse 'June 21' / 'Jun 21st' / 'Dec 28' into an ISO date for the year."""
    from datetime import datetime

    cleaned = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", token.strip(), flags=re.IGNORECASE)
    for fmt in ("%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(f"{cleaned} {year}", fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _month_day(token: str):
    """Parse a 'Month day' token (full or abbreviated, with optional ordinal)
    into a (month, day) tuple, or None."""
    from datetime import datetime

    cleaned = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", token.strip(), flags=re.IGNORECASE)
    for fmt in ("%B %d", "%b %d"):
        try:
            d = datetime.strptime(cleaned, fmt)
            return d.month, d.day
        except ValueError:
            continue
    return None


def _resolve_year(month: int, day: int) -> int:
    """Pick the year for a month/day: current year, or next year if it's past."""
    from datetime import date

    today = date.today()
    candidate = date(today.year, month, day)
    return today.year if candidate >= today else today.year + 1


def _return_year_iso(month: int, day: int, departure_iso: str | None) -> str | None:
    """Return ISO date for a return month/day that is on/after the departure.

    Tries the departure's year first; if that return date falls before the
    departure, rolls forward one year so return >= departure across year
    boundaries (e.g. departure 2026-12-28, return Jan 5 -> 2027-01-05).
    """
    from datetime import date, datetime

    if departure_iso is None:
        return None
    dep = datetime.strptime(departure_iso, "%Y-%m-%d").date()
    for year in (dep.year, dep.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate >= dep:
            return candidate.isoformat()
    return None


def _extract_date_range(message: str) -> tuple[str | None, str | None]:
    """Extract an explicit departure/return date range as ISO strings.

    Handles 'from June 21 to June 25', 'June 22nd to June 25th',
    'June 21 - 25', and 'June 21-25' (second date inherits the first's month
    when omitted). Ordinal suffixes (st/nd/rd/th) are accepted on any day.
    Returns (departure_iso, return_iso); each may be None independently.
    """
    month_group = rf"(?:{_MONTHS})"
    # Two full month-day anchors, separated by 'to' / '-' / '–'.
    full = re.search(
        rf"\b(?:from\s+)?({month_group}\s+{_DAY})\s*(?:to|[-\u2013])\s*({month_group}\s+{_DAY})\b",
        message,
        flags=re.IGNORECASE,
    )
    if full:
        return _parse_full_month_range(full.group(1), full.group(2))

    # First anchor has a month; second is a bare day sharing that month.
    partial = re.search(
        rf"\b(?:from\s+)?({month_group})\s+({_DAY})\s*(?:to|[-\u2013])\s*({_DAY})\b",
        message,
        flags=re.IGNORECASE,
    )
    if partial:
        return _parse_single_month_range(partial.group(1), partial.group(2), partial.group(3))

    return None, None


def _parse_full_month_range(start_token: str, end_token: str) -> tuple[str | None, str | None]:
    ma = _month_day(start_token)
    mb = _month_day(end_token)
    if not ma or not mb:
        return None, None
    dep_year = _resolve_year(ma[0], ma[1])
    dep = _parse_month_day(start_token, dep_year)
    # Return year is chosen relative to the departure so the return date
    # is never before the departure (handles 'June 3 to July 10' when
    # June 3 has already passed this year, and 'Dec 28 to Jan 5').
    ret = _return_year_iso(mb[0], mb[1], dep)
    return dep, ret


def _parse_single_day_range(day_token: str) -> int:
    cleaned = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", day_token, flags=re.IGNORECASE)
    return int(cleaned)


def _parse_single_month_range(month_name: str, start_day_token: str, end_day_token: str) -> tuple[str | None, str | None]:
    from datetime import date

    d1 = _parse_single_day_range(start_day_token)
    d2 = _parse_single_day_range(end_day_token)
    md = _month_day(f"{month_name} {d1}")
    if not md:
        return None, None
    m = md[0]

    year = _resolve_year(m, d1)
    dep = date(year, m, d1).isoformat()
    ret_year, ret_month = year, m
    if d2 < d1:
        ret_month = m % 12 + 1
        if ret_month == 1:
            ret_year += 1
    try:
        ret = date(ret_year, ret_month, d2).isoformat()
    except ValueError:
        ret = None
    return dep, ret


def _extract_departure_date(message: str) -> str | None:
    dep, _ = _extract_date_range(message)
    return dep


def _extract_return_date(message: str) -> str | None:
    _, ret = _extract_date_range(message)
    return ret


def _extract_travel_style(normalized: str) -> str | None:
    if "family-friendly" in normalized or "family friendly" in normalized:
        return "family-friendly"
    if "relaxed" in normalized:
        return "relaxed"
    if "packed" in normalized:
        return "packed"
    return None


def _extract_keywords(normalized: str, keywords: list[str]) -> list[str]:
    return [keyword.replace(" ", "-") for keyword in keywords if keyword in normalized]


def _extract_follow_up_intent(normalized: str) -> str | None:
    if _contains_any(normalized, ["make it cheaper", "cheaper", "more affordable", "lower budget"]):
        return "cheaper"
    if _contains_any(normalized, ["more indoor", "indoor activities", "inside activities"]):
        return "more_indoor"
    if _contains_any(normalized, ["more museums", "add museums", "add more museums"]):
        return "more_museums"
    if _contains_any(normalized, ["make it relaxed", "more relaxed", "slower pace"]):
        return "change_pace"
    return None


def _extract_trip_type(normalized: str) -> str:
    """Classify the request as one-way or round-trip.

    Default is round-trip (preserves the existing behavior of always
    requesting return flights). Only an explicit "one way" / "oneway" /
    "single" phrase flips it to one-way.
    """
    if _contains_any(normalized, ["one way", "one-way", "oneway", "single trip", "single-ticket", "outbound only"]):
        return "one_way"
    return "round_trip"


def _extract_origin_city(message: str) -> str | None:
    """Extract a departure/origin city from phrases like 'from London' or 'flying from Mumbai'.

    The second-word group uses a negative lookahead for ``to`` so phrases like
    "from london to milan" capture "london" (the origin) and do not swallow
    the "to milan" part of the route.
    """
    patterns = [
        r"\b(?:from|flying from|departing from|leaving from)\s+([A-Za-z]+(?:\s+(?!to\b)[A-Za-z]+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            candidate = _clean_city_candidate(match.group(1))
            if candidate and candidate.lower() not in {"home", "here", "there", "airport"}:
                return candidate
    return None


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)
