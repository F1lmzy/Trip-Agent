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
    "low": ["cheap", "low budget", "low-budget", "affordable", "low-cost", "budget-friendly"],
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
    travel_style: str | None = None
    dietary_needs: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    asks_for_hotel: bool = False
    asks_for_current_info: bool = False
    is_follow_up: bool = False
    follow_up_intent: str | None = None


def parse_user_request(message: str) -> ParsedRequest:
    normalized = message.lower().strip()

    return ParsedRequest(
        raw_message=message,
        city=_extract_city(message),
        duration_days=_extract_duration_days(normalized),
        interests=_extract_interests(normalized),
        budget=_extract_budget(normalized),
        dates=_extract_dates(message),
        travel_style=_extract_travel_style(normalized),
        dietary_needs=_extract_keywords(normalized, DIETARY_KEYWORDS),
        constraints=_extract_keywords(normalized, CONSTRAINT_KEYWORDS),
        asks_for_hotel=_contains_any(
            normalized, ["hotel", "hotels", "stay", "accommodation", "lodging", "place to stay", "places to stay"]
        ),
        asks_for_current_info=_contains_any(normalized, CURRENT_INFO_KEYWORDS),
        is_follow_up=_extract_follow_up_intent(normalized) is not None,
        follow_up_intent=_extract_follow_up_intent(normalized),
    )


def _extract_city(message: str) -> str | None:
    for city in KNOWN_CITIES:
        if re.search(rf"\b{re.escape(city)}\b", message, flags=re.IGNORECASE):
            return city

    patterns = [
        r"(?:trip to|visit|in|for)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().title()
    return None


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
        r"Sep|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2}(?:\s*-\s*\d{1,2})?\b"
    )
    match = re.search(month_pattern, message, flags=re.IGNORECASE)
    return match.group(0) if match else None


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


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)
