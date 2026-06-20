"""ItineraryAgent: runs tools and generates the itinerary.

Thin wrapper over the existing orchestrator tool-execution and response-
generation path. Handles two cases exactly as the monolithic orchestrator did:

1. Follow-up request (parsed.is_follow_up): build a follow-up response that
   either asks for prior context (when there is none) or acknowledges the
   planned update. No tools are run.
2. Fresh request: run the selected tools, save stable preferences, and
   generate the itinerary via the LLM/fallback path.

No behavior change — the helpers are imported lazily from
app.agent.orchestrator to avoid an import-time cycle.
"""

from __future__ import annotations

from app.agents.base import Agent, AgentContext
from app.schemas import ChatResponse


class ItineraryAgent(Agent):
    name = "ItineraryAgent"

    def run(self, ctx: AgentContext) -> ChatResponse:
        # Lazy imports avoid an import-time cycle: app.agent.orchestrator
        # imports this package's supervisor at module load time.
        from app.agent.orchestrator import (
            _build_follow_up_response,
            _execute_tools,
            _save_stable_preferences,
            _service_value,
        )

        ctx.emit({"type": "agent_start", "agent": self.name})

        if ctx.parsed.is_follow_up:
            has_prior_context = ctx.memory.has_history(ctx.user_id)
            memory_used = ctx.user_memory.search_preferences(ctx.user_id, ctx.parsed.raw_message)
            response = _build_follow_up_response(
                ctx.parsed, ctx.plan, has_prior_context, memory_used
            )
            ctx.emit({"type": "agent_end", "agent": self.name})
            return response

        memory_used = ctx.user_memory.search_preferences(ctx.user_id, ctx.parsed.raw_message)
        ctx.emit({"type": "tools_start"})
        tool_outputs = _execute_tools(ctx.parsed, ctx.plan, ctx.services)
        for tool_name in tool_outputs:
            ctx.emit({"type": "tool_end", "tool": tool_name, "status": "completed"})
        ctx.emit({"type": "tools_end"})

        _save_stable_preferences(ctx.user_id, ctx.parsed, ctx.user_memory)

        from app.agent.response_generator import generate_itinerary_response

        response = generate_itinerary_response(
            parsed=ctx.parsed,
            plan=ctx.plan,
            tool_outputs=tool_outputs,
            memory_used=memory_used,
            api_key=_service_value(ctx.services, "openrouter_api_key", "openrouter_api_key"),
            model=_service_value(ctx.services, "openrouter_model", "openrouter_model"),
            client=ctx.services.openrouter_client,
        )
        ctx.emit({"type": "agent_end", "agent": self.name})
        return response
