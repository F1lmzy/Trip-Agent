import json
from pathlib import Path
from typing import Any

from app.tools.budget_tool import _normalize_budget

_HOTELS_PATH = Path(__file__).parents[1] / "data" / "hotels.json"


def load_hotels(path: Path = _HOTELS_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_hotel_tool(city: str, budget: str | None = None, limit: int = 3) -> dict[str, Any]:
    normalized_city = city.strip().replace("_", " ").title()
    budget_level = _normalize_budget(budget)
    if budget_level not in {"low", "medium", "luxury"}:
        budget_level = "medium"

    city_hotels = [hotel for hotel in load_hotels() if hotel["city"] == normalized_city]
    if not city_hotels:
        return {
            "tool_name": "hotel_tool",
            "status": "no_results",
            "city": normalized_city,
            "budget_level": budget_level,
            "results": [],
        }

    matching_hotels = [hotel for hotel in city_hotels if hotel["budget_level"] == budget_level]
    results = matching_hotels or city_hotels

    return {
        "tool_name": "hotel_tool",
        "status": "ok",
        "city": normalized_city,
        "budget_level": budget_level,
        "results": results[:limit],
    }
