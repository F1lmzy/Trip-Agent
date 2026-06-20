"""Multi-agent orchestration package.

Specialized agents coordinated by a supervisor router, modeled on the
Azure-Samples/azure-ai-travel-agents supervisor pattern but kept lightweight
(no LangGraph/MAF dependency). Each agent is a thin wrapper over the existing
orchestrator helpers so behavior is preserved exactly.

Agents:
- CustomerQueryAgent: builds a clarification response when the request is
  missing required information (e.g. no destination yet).
- ItineraryAgent: runs tools and generates the itinerary (or a follow-up
  response) when the request is ready to plan.
- DestinationRecommendationAgent: suggests ranked cities from RAG when no
  destination is given (added in a later iteration).
- Supervisor: routes an AgentContext to the appropriate agent.
"""

from app.agents.base import Agent, AgentContext
from app.agents.customer_query_agent import CustomerQueryAgent
from app.agents.destination_recommendation_agent import DestinationRecommendationAgent
from app.agents.itinerary_agent import ItineraryAgent
from app.agents.supervisor import Supervisor, route

__all__ = [
    "Agent",
    "AgentContext",
    "CustomerQueryAgent",
    "DestinationRecommendationAgent",
    "ItineraryAgent",
    "Supervisor",
    "route",
]
