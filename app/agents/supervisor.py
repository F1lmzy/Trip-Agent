"""Supervisor: routes an AgentContext to the appropriate specialized agent.

A lightweight router (no LangGraph/MAF) that decides which agent runs for a
given chat turn. In this scaffold it mirrors the original monolithic
branching of ``handle_chat`` exactly:

- plan.needs_clarification  -> CustomerQueryAgent
- otherwise                 -> ItineraryAgent

The DestinationRecommendationAgent branch (city-None -> suggest cities) is
added in a later iteration; until then city-None requests still clarify,
preserving current behavior and keeping all existing tests green.
"""

from __future__ import annotations

from app.agents.base import Agent, AgentContext
from app.agents.customer_query_agent import CustomerQueryAgent
from app.agents.destination_recommendation_agent import DestinationRecommendationAgent
from app.agents.itinerary_agent import ItineraryAgent


class Supervisor:
    """Routes a context to the agent that should handle it.

    Routing rules (mirror the Azure-Samples supervisor pattern, lightweight):
    - follow-up request            -> ItineraryAgent (reuses prior context;
      it asks for prior context if none exists)
    - no destination city given    -> DestinationRecommendationAgent (suggest
      ranked cities from RAG instead of asking a clarifying question)
    - destination city given       -> ItineraryAgent (run tools, generate)
    - plan still needs clarification for other reasons -> CustomerQueryAgent
      (kept as a fallback for future clarification cases)

    Stateless and cheap to construct; a single instance is reused.
    """

    def __init__(self) -> None:
        self._customer_query = CustomerQueryAgent()
        self._itinerary = ItineraryAgent()
        self._destination = DestinationRecommendationAgent()

    def route(self, ctx: AgentContext) -> Agent:
        # Follow-ups are handled by the itinerary agent (it knows how to ask
        # for prior context or apply the follow-up change).
        if ctx.parsed.is_follow_up:
            return self._itinerary
        # No destination -> suggest cities instead of asking a bare question.
        if ctx.parsed.city is None:
            return self._destination
        # Any other clarification need the planner flagged (future cases).
        if ctx.plan.needs_clarification:
            return self._customer_query
        return self._itinerary

    def run(self, ctx: AgentContext):
        """Convenience: route and run in one call, returning a ChatResponse."""
        agent = self.route(ctx)
        ctx.emit({"type": "route", "agent": agent.name})
        return agent.run(ctx)


# Module-level convenience used by the orchestrator.
def route(ctx: AgentContext) -> Agent:
    return Supervisor().route(ctx)
