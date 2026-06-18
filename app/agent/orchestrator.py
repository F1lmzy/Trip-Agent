from app.agent.parser import ParsedRequest, parse_user_request
from app.agent.planner import PlanningResult, create_trip_plan
from app.memory.short_term import ShortTermMemory, short_term_memory
from app.schemas import ChatRequest, ChatResponse


def handle_chat(request: ChatRequest, memory: ShortTermMemory = short_term_memory) -> ChatResponse:
    parsed = parse_user_request(request.message)
    plan = create_trip_plan(parsed)
    response = _build_response(parsed, plan, memory.has_history(request.user_id))

    memory.add_message(request.user_id, "user", request.message)
    memory.add_message(request.user_id, "assistant", response.message)

    return response


def _build_response(parsed: ParsedRequest, plan: PlanningResult, has_prior_context: bool) -> ChatResponse:
    if plan.needs_clarification:
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

    if parsed.is_follow_up:
        return _build_follow_up_response(parsed, plan, has_prior_context)

    city = parsed.city or "your destination"
    interests = parsed.interests or ["general highlights"]
    budget = parsed.budget or "medium"

    return ChatResponse(
        message=(
            f"I can plan a {parsed.duration_days}-day trip to {city}. "
            "The next implementation step will execute the selected tools and generate the full itinerary."
        ),
        itinerary={
            "city": city,
            "duration_days": parsed.duration_days,
            "preferences_used": interests,
            "budget": budget,
            "status": "planned_not_generated_yet",
        },
        memory_used=[],
        tools_used=plan.selected_tools,
        plan=plan.plan,
        needs_clarification=False,
        clarifying_question=None,
    )


def _build_follow_up_response(parsed: ParsedRequest, plan: PlanningResult, has_prior_context: bool) -> ChatResponse:
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
        memory_used=["Recent conversation history"],
        tools_used=plan.selected_tools,
        plan=plan.plan,
        needs_clarification=False,
        clarifying_question=None,
    )
