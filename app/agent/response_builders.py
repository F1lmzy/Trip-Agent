from __future__ import annotations

from app.agent.parser import ParsedRequest
from app.agent.planner import PlanningResult
from app.memory.long_term import LongTermMemory
from app.schemas import ChatResponse


def build_clarification_response(plan: PlanningResult) -> ChatResponse:
    question = plan.clarifying_question or "Could you share a few more details for the trip?"
    return ChatResponse(
        message=question,
        itinerary={},
        memory_used=[],
        tools_used=[],
        plan=plan.plan,
        needs_clarification=True,
        clarifying_question=question,
    )


def build_follow_up_response(
    parsed: ParsedRequest,
    plan: PlanningResult,
    has_prior_context: bool,
    memory_used: list[str],
) -> ChatResponse:
    if not has_prior_context:
        question = "What trip should I update? Please share the destination or original itinerary request."
        return ChatResponse(
            message=question,
            itinerary={},
            memory_used=[],
            tools_used=[],
            plan=["Ask for original trip context before applying follow-up request"],
            needs_clarification=True,
            clarifying_question=question,
        )

    return ChatResponse(
        message="I understood this as a follow-up to your previous trip request and planned the update.",
        itinerary={
            "status": "follow_up_planned_not_generated_yet",
            "follow_up_intent": parsed.follow_up_intent,
        },
        memory_used=["Recent conversation history", *memory_used],
        tools_used=plan.selected_tools,
        plan=plan.plan,
        needs_clarification=False,
        clarifying_question=None,
    )


def save_stable_preferences(user_id: str, parsed: ParsedRequest, user_memory: LongTermMemory) -> None:
    existing = set(user_memory.get_preferences(user_id))
    for preference in _stable_preferences_from(parsed):
        if preference not in existing:
            user_memory.add_preference(user_id, preference)
            existing.add(preference)


def _stable_preferences_from(parsed: ParsedRequest) -> list[str]:
    preferences: list[str] = []
    preferences.extend(f"Interest preference: {interest}" for interest in parsed.interests)
    preferences.extend(f"Dietary need: {need}" for need in parsed.dietary_needs)
    preferences.extend(f"Constraint: {constraint}" for constraint in parsed.constraints)

    if parsed.budget:
        preferences.append(f"Budget preference: {parsed.budget}")
    if parsed.travel_style:
        preferences.append(f"Travel style: {parsed.travel_style}")

    return preferences
