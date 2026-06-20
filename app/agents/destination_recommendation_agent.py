"""DestinationRecommendationAgent: suggest cities when no destination is given.

Instead of asking a clarifying question when the user hasn't named a city
(e.g. "I like anime, food and photography, medium budget, 2 days"), this
agent ranks curated city overviews from the ``travel_city_docs`` ChromaDB
collection by semantic similarity to the user's interests + budget + travel
style, excludes the origin city (when one is named), and returns the top
matches as suggestions. This turns a dead-end into a productive response,
mirroring the Azure-Samples destination-recommendation agent.

The collection has only curated city overviews (no category metadata), so
ranking is by embedding distance over the overview text using the interest
terms as the query. City docs are seeded lazily from the curated
``app/data/city_docs/*.md`` files if the collection is empty.
"""

from __future__ import annotations

from typing import Any

from app.agents.base import Agent, AgentContext
from app.schemas import ChatResponse

# Reused from the RAG tool so the destination agent reads the same collection
# the itinerary agent later queries for city context.
_CITY_DOCS_COLLECTION = "travel_city_docs"
_MAX_SUGGESTIONS = 5
_QUERY_CANDIDATES = 8


class DestinationRecommendationAgent(Agent):
    name = "DestinationRecommendationAgent"

    def run(self, ctx: AgentContext) -> ChatResponse:
        ctx.emit({"type": "agent_start", "agent": self.name})

        vector_store = self._resolve_vector_store(ctx)
        self._ensure_city_docs_seeded(vector_store)

        query = self._build_query(ctx.parsed)
        ctx.emit({"type": "tool_start", "tool": "destination_rag", "query": query})
        results = vector_store.query(
            _CITY_DOCS_COLLECTION,
            query_text=query,
            n_results=_QUERY_CANDIDATES,
        )
        ctx.emit({"type": "tool_end", "tool": "destination_rag", "status": "completed"})

        origin = (ctx.parsed.origin_city or "").strip().lower()
        suggestions: list[dict[str, Any]] = []
        seen_cities: set[str] = set()
        for result in results:
            city = (result.metadata.get("city") or "").strip()
            if not city:
                continue
            key = city.lower()
            if key in seen_cities:
                continue
            if origin and key == origin:
                # Never recommend the city the user is flying out of.
                continue
            seen_cities.add(key)
            suggestions.append(
                {
                    "city": city,
                    "rationale": self._rationale(result.document),
                    "match_score": self._score(result.distance),
                }
            )
            if len(suggestions) >= _MAX_SUGGESTIONS:
                break

        message = self._build_message(ctx.parsed, suggestions)
        itinerary: dict[str, Any] = {
            "status": "destination_suggestions",
            "suggested_cities": suggestions,
        }
        if ctx.parsed.origin_city:
            itinerary["origin_city"] = ctx.parsed.origin_city
        if ctx.parsed.budget:
            itinerary["budget"] = ctx.parsed.budget

        ctx.emit({"type": "agent_end", "agent": self.name})
        return ChatResponse(
            message=message,
            itinerary=itinerary,
            memory_used=[],
            tools_used=["destination_rag"],
            plan=ctx.plan.plan,
            needs_clarification=False,
            clarifying_question=None,
        )

    @staticmethod
    def _resolve_vector_store(ctx: AgentContext):
        vs = ctx.services.vector_store
        if vs is not None:
            return vs
        if ctx.services.attraction_rag_tool is not None:
            return ctx.services.attraction_rag_tool.vector_store
        from app.memory.vector_store import VectorStore

        return VectorStore()

    @staticmethod
    def _ensure_city_docs_seeded(vector_store) -> None:
        """Seed curated city overviews if the collection is empty.

        Reads local curated files (no network). Idempotent via upsert.
        """
        collection = vector_store.get_or_create_collection(_CITY_DOCS_COLLECTION)
        if collection.count() > 0:
            return
        from app.tools.attraction_rag_tool import load_city_documents

        docs = load_city_documents()
        if not docs:
            return
        vector_store.add_documents(
            _CITY_DOCS_COLLECTION,
            documents=[doc["text"] for doc in docs],
            metadatas=[
                {"city": doc["city"], "type": "city_overview", "source": "curated_wikivoyage_style"}
                for doc in docs
            ],
            ids=[doc["id"] for doc in docs],
        )

    @staticmethod
    def _build_query(parsed) -> str:
        terms: list[str] = []
        terms.extend(parsed.interests or [])
        if parsed.travel_style:
            terms.append(parsed.travel_style)
        if parsed.budget:
            terms.append(f"{parsed.budget} budget")
        # Generic travel terms help match the city-overview document style.
        terms.extend(["travel", "overview", "attractions", "neighborhoods"])
        if not parsed.interests:
            # No interests to rank by: lean on a broad travel query so we still
            # return sensible defaults rather than asking a clarifying question.
            terms = ["highlights", "food", "museums", "nature", "photography", "travel", "overview"]
        return " ".join(terms)

    @staticmethod
    def _rationale(document: str) -> str:
        text = (document or "").strip()
        # Drop the leading "# <City> Travel Overview" heading line for a
        # cleaner one-line rationale.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        body = " ".join(lines[1:]) if lines and lines[0].startswith("#") else " ".join(lines)
        return body[:180].rstrip() + ("..." if len(body) > 180 else "")

    @staticmethod
    def _score(distance: float | None) -> float:
        """Convert a Chroma cosine distance (0=identical, <=2) into a 0-1 score."""
        if distance is None:
            return 0.0
        try:
            d = max(0.0, float(distance))
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, 1.0 - d / 2.0)

    @staticmethod
    def _build_message(parsed, suggestions: list[dict[str, Any]]) -> str:
        if not suggestions:
            return (
                "I couldn't match your preferences to a specific destination from the curated set. "
                "Could you name a city you're considering, or share a bit more about what you'd like to do?"
            )
        lines = [
            f"Based on what you're looking for"
            + (f" ({', '.join(parsed.interests)})" if parsed.interests else "")
            + ", here are some destinations worth considering:",
            "",
        ]
        for index, suggestion in enumerate(suggestions, start=1):
            city = suggestion["city"]
            rationale = suggestion["rationale"]
            lines.append(f"{index}. **{city}** — {rationale}")
        lines.append("")
        lines.append("Which one sounds good? Tell me the city and I'll plan the full itinerary.")
        return "\n".join(lines)
