from pydantic import BaseModel, Field

from app.agent.parser import ParsedRequest


CORE_PLAN_STEPS = [
    "Parse destination, duration, interests, budget, dates, and constraints",
    "Retrieve long-term user preferences from ChromaDB",
    "Run city-level RAG retrieval for broad destination context",
    "Run interest-specific RAG retrieval using city context and user preferences",
    "Call weather tool for destination forecast",
    "Apply budget rules and constraints",
]


class PlanningResult(BaseModel):
    plan: list[str] = Field(default_factory=list)
    selected_tools: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarifying_question: str | None = None
    assumptions: list[str] = Field(default_factory=list)


def create_trip_plan(parsed: ParsedRequest, rag_context_is_weak: bool = False) -> PlanningResult:
    if parsed.city is None and not parsed.is_follow_up:
        return PlanningResult(
            plan=["Ask for missing destination before calling travel-planning tools"],
            selected_tools=[],
            needs_clarification=True,
            clarifying_question="Which city would you like to visit?",
        )

    if parsed.is_follow_up and parsed.city is None:
        return _plan_follow_up(parsed)

    selected_tools = ["attraction_rag_tool", "weather_tool", "budget_tool"]
    assumptions: list[str] = []
    plan = list(CORE_PLAN_STEPS)

    if parsed.budget is None:
        assumptions.append("Budget not provided; defaulting to medium for this itinerary.")

    if parsed.asks_for_current_info or rag_context_is_weak:
        selected_tools.append("web_search_tool")
        plan.append("Search the web for fresh travel context")

    if parsed.asks_for_hotel:
        selected_tools.append("hotel_tool")
        plan.append("Retrieve hotel recommendations")

    if parsed.asks_for_flights and parsed.origin_city:
        selected_tools.append("flight_tool")
        plan.append("Suggest flights from origin to destination")

    plan.extend(
        [
            "Generate a structured itinerary",
            "Save stable user preferences to long-term memory",
        ]
    )

    return PlanningResult(
        plan=plan,
        selected_tools=selected_tools,
        needs_clarification=False,
        assumptions=assumptions,
    )


def _plan_follow_up(parsed: ParsedRequest) -> PlanningResult:
    if parsed.follow_up_intent == "cheaper":
        return PlanningResult(
            plan=[
                "Reuse the previous itinerary from short-term memory",
                "Apply lower-budget constraints",
                "Regenerate the itinerary with cheaper options",
            ],
            selected_tools=["budget_tool"],
        )

    if parsed.follow_up_intent == "more_indoor":
        return PlanningResult(
            plan=[
                "Reuse the previous itinerary from short-term memory",
                "Find more indoor attractions",
                "Use weather context if relevant",
                "Regenerate the itinerary with more indoor options",
            ],
            selected_tools=["attraction_rag_tool", "weather_tool"],
        )

    if parsed.follow_up_intent == "more_museums":
        return PlanningResult(
            plan=[
                "Reuse the previous itinerary from short-term memory",
                "Find additional museum options",
                "Regenerate the itinerary with more museums",
            ],
            selected_tools=["attraction_rag_tool"],
        )

    return PlanningResult(
        plan=[
            "Reuse the previous itinerary from short-term memory",
            "Apply the requested follow-up change",
            "Regenerate the itinerary",
        ],
        selected_tools=[],
    )
