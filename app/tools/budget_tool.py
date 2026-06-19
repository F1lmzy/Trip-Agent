BUDGET_GUIDANCE = {
    "low": {
        "meals": "street food, markets, bakeries, hawker centres, and casual cheap eats",
        "hotel": "hostel, capsule hotel, guesthouse, or budget hotel",
        "activities": "free viewpoints, parks, self-guided walks, and low-cost attractions",
        "transport": "public transit and walking",
    },
    "medium": {
        "meals": "casual restaurants, local favorites, food halls, and a few paid experiences",
        "hotel": "3-star hotel, boutique guesthouse, or well-rated apartment stay",
        "activities": "mix of free attractions, paid museums, viewpoints, and guided experiences",
        "transport": "public transit with occasional rideshare or taxi",
    },
    "luxury": {
        "meals": "fine dining, tasting menus, cocktail bars, and premium local restaurants",
        "hotel": "4-5 star hotel or luxury boutique property",
        "activities": "private tours, premium viewpoints, reservations, and curated experiences",
        "transport": "taxis, private transfers, and premium rail where useful",
    },
}


def run_budget_tool(budget: str | None) -> dict:
    normalized_budget = _normalize_budget(budget)
    assumed = budget is None or normalized_budget != budget.lower().strip()
    status = "ok" if normalized_budget in BUDGET_GUIDANCE else "fallback"

    if normalized_budget not in BUDGET_GUIDANCE:
        normalized_budget = "medium"
        assumed = True

    return {
        "tool_name": "budget_tool",
        "status": status,
        "budget_level": normalized_budget,
        "assumed": assumed,
        "guidance": BUDGET_GUIDANCE[normalized_budget],
    }


def _normalize_budget(budget: str | None) -> str:
    if budget is None:
        return "medium"

    normalized = budget.lower().strip()
    if normalized in {
        "cheap",
        "budget",
        "low",
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
    }:
        return "low"
    if normalized in {"moderate", "medium", "mid-range", "midrange"}:
        return "medium"
    if normalized in {"luxury", "premium", "high-end", "expensive"}:
        return "luxury"
    return normalized
