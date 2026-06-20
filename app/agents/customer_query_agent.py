"""CustomerQueryAgent: builds a clarification response when info is missing.

Thin wrapper over the existing planner clarification logic — no new behavior.
When the supervisor detects that the plan needs clarification (e.g. no
destination was given), this agent produces the same ChatResponse the
monolithic orchestrator used to build directly.

In a later iteration the DestinationRecommendationAgent will intercept the
"no city" case to suggest cities instead of asking; until then this agent
preserves the original clarification behavior exactly.
"""

from __future__ import annotations

from app.agents.base import Agent, AgentContext
from app.schemas import ChatResponse


class CustomerQueryAgent(Agent):
    name = "CustomerQueryAgent"

    def run(self, ctx: AgentContext) -> ChatResponse:
        # Lazy import avoids an import-time cycle with app.agent.orchestrator,
        # which imports this package's supervisor.
        from app.agent.orchestrator import _build_clarification_response

        ctx.emit({"type": "agent_start", "agent": self.name})
        response = _build_clarification_response(ctx.plan)
        ctx.emit({"type": "agent_end", "agent": self.name})
        return response
