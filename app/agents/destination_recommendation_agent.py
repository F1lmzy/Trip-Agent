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

import json
import re
from pathlib import Path
from typing import Any

from app.agents.base import Agent, AgentContext
from app.schemas import ChatResponse
from app.tools.web_search_tool import run_destination_search_tool

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_DESTINATION_CATALOG = _DATA_DIR / "destination_catalog.json"

# Reused from the RAG tool so the destination agent reads the same collection
# the itinerary agent later queries for city context.
_CITY_DOCS_COLLECTION = "travel_city_docs"
_MAX_SUGGESTIONS = 5
_QUERY_CANDIDATES = 8


def _load_destination_catalog() -> list[dict[str, Any]]:
    try:
        with _DESTINATION_CATALOG.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _catalog_city_mentions(catalog: list[dict[str, Any]], results: list[dict[str, str]]) -> dict[str, int]:
    text = " ".join(
        f"{item.get('title', '')} {item.get('description', '')}" for item in results if isinstance(item, dict)
    ).lower()
    mentions: dict[str, int] = {}
    for item in catalog:
        city = str(item.get("city", "")).strip()
        if not city:
            continue
        pattern = r"(?<![a-z])" + re.escape(city.lower()) + r"(?![a-z])"
        count = len(re.findall(pattern, text))
        if count:
            mentions[city.lower()] = count
    return mentions


def _web_boosted_rationale(item: dict[str, Any], web_mentions: dict[str, int]) -> str:
    rationale = str(item.get("rationale", ""))
    city = str(item.get("city", "")).lower()
    if web_mentions.get(city, 0):
        return rationale + " Recent web results also mention it for this kind of trip."
    return rationale


_DESTINATION_STOPWORDS = {
    "asia",
    "asian",
    "europe",
    "european",
    "history",
    "travel",
    "travels",
    "destination",
    "destinations",
    "city",
    "cities",
    "guide",
    "guides",
    "trip",
    "trips",
    "itinerary",
    "itineraries",
    "places",
    "place",
    "best",
    "top",
    "current",
    "fresh",
    "recent",
    "latest",
    "ancient",
    "historic",
    "historical",
    "culture",
    "cultural",
    "temples",
    "temple",
    "museum",
    "museums",
    "things",
    "days",
    "day",
    "buffs",
    "history buffs",
    "model desac",
    "the",
    "the best travel",
    "destinations for history",
    "world",
    "western",
    "exploring",
    "true history",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "comes",
    "life",
    "gods",
    "god",
    "egyptian",
    "egyptians",
    "pharaohs",
    "pharaoh",
    "afterlife",
    "south korea",
    "south east asia",
    "southeast asia",
    "east asia",
    "historic asia",
    "angkor",
    "angkor wat",
    "terracotta army",
    "forbidden city",
    "great wall",
    "japan",
    "china",
    "vietnam",
    "cambodia",
    "thailand",
    "india",
    "indonesia",
    "singapore",
    "sri lanka",
    "southeast asian",
    "se asia",
    "which asian",
    "mumsnet",
    "reddit",
    "tripadvisor",
    "quora",
    "lonely planet",
    "rough guides",
    "time out",
    "timeout",
    "cn traveller",
    "condé nast traveller",
    "conde nast traveller",
}

# Countries/regions/publication names are useful web context, but should not be
# emitted as city suggestions. Keep this as a guardrail around noisy snippets;
# catalog items can still include city-states such as Singapore intentionally.
_NON_CITY_WEB_CANDIDATES = {
    "afghanistan", "albania", "algeria", "andorra", "argentina", "armenia", "australia",
    "austria", "azerbaijan", "bahamas", "bahrain", "bangladesh", "belgium", "bhutan",
    "bolivia", "brazil", "bulgaria", "cambodia", "canada", "chile", "china", "colombia",
    "croatia", "cyprus", "czech republic", "denmark", "egypt", "estonia", "ethiopia",
    "finland", "france", "georgia", "germany", "greece", "hungary", "iceland", "india",
    "indonesia", "iran", "iraq", "ireland", "israel", "italy", "japan", "jordan", "kenya",
    "laos", "latvia", "lebanon", "lithuania", "malaysia", "maldives", "mexico", "morocco",
    "myanmar", "nepal", "netherlands", "new zealand", "norway", "oman", "pakistan",
    "peru", "philippines", "poland", "portugal", "qatar", "romania", "russia",
    "saudi arabia", "serbia", "slovakia", "slovenia", "south africa", "south korea",
    "spain", "sri lanka", "sweden", "switzerland", "syria", "taiwan", "thailand",
    "turkey", "ukraine", "united arab emirates", "united kingdom", "united states",
    "usa", "vietnam",
    "asia", "east asia", "south asia", "southeast asia", "south east asia", "se asia",
    "southeast asian", "western europe", "eastern europe", "middle east", "which asian",
    "mumsnet", "reddit", "tripadvisor", "quora", "lonely planet", "rough guides",
}

_DESTINATION_NAME_RE = re.compile(r"\b[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){0,2}\b")
_LIST_CONTEXT_RE = re.compile(
    r"(?:destinations?|cities|city breaks?|places to visit|places to go|historic hubs|history trips)"
    r"(?:\s*:\s*|\s+(?:include|including|such as)\s+)"
    r"([^.!?]{0,160})",
    flags=re.IGNORECASE,
)
_CITY_OF_RE = re.compile(
    r"(?:capital city|historic city|city)\s+of\s+([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){0,2})",
)


def _extract_destination_names(text: str) -> list[str]:
    """Extract city-like names from explicit destination-list contexts only.

    Do not run open-ended title-case extraction. Search snippets contain article
    titles, months, publishers, countries, regions, historical empires, and SEO
    phrases in Title Case; treating those as cities caused repeated bad output
    like "Amazing", "Modernity Collide", and "Ottoman Empire".
    """
    candidates: list[str] = []
    for match in _LIST_CONTEXT_RE.finditer(text or ""):
        candidates.extend(_split_city_list(match.group(1)))
    for match in _CITY_OF_RE.finditer(text or ""):
        candidates.append(match.group(1))

    clean: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip(" .,:;()[]{}")
        normalized = candidate.lower().replace("’", "'")
        if normalized in seen:
            continue
        if _is_valid_web_city_candidate(candidate):
            seen.add(normalized)
            clean.append(candidate.replace("’", "'"))
    return clean


def _split_city_list(fragment: str) -> list[str]:
    # Keep separators simple and list-like. This intentionally avoids extracting
    # arbitrary Title Case phrases from full prose sentences.
    normalized = re.sub(r"\b(?:and|or)\b", ",", fragment)
    parts = re.split(r",|;|/|\||\u00b7|•", normalized)
    candidates: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = _DESTINATION_NAME_RE.search(part)
        if match:
            candidates.append(match.group(0))
    return candidates


def _is_valid_web_city_candidate(candidate: str) -> bool:
    normalized = candidate.lower().replace("’", "'")
    if len(candidate) <= 2:
        return False
    if normalized in _DESTINATION_STOPWORDS or normalized in _NON_CITY_WEB_CANDIDATES:
        return False
    tokens = normalized.split()
    if tokens[0] in {"best", "top", "why", "where", "when", "how", "plan", "visit", "historic", "the", "amazing"}:
        return False
    if tokens[-1] in {
        "army",
        "temple",
        "temples",
        "museum",
        "museums",
        "wall",
        "buffs",
        "history",
        "travel",
        "destinations",
        "destination",
        "empire",
        "power",
        "collide",
    }:
        return False
    blocked_tokens = {
        "history",
        "buffs",
        "travel",
        "destinations",
        "destination",
        "model",
        "desac",
        "comes",
        "life",
        "gods",
        "egyptian",
        "pharaohs",
        "pharaoh",
        "amazing",
        "modernity",
        "touripia",
        "ottoman",
        "empire",
        "global",
        "power",
    }
    if any(token in blocked_tokens for token in tokens):
        return False
    return True


def _looks_like_destination_candidate(candidate: str, source_text: str) -> bool:
    normalized = candidate.lower()
    tokens = normalized.split()
    # Single-word discoveries are very noisy. Accept them only when the source
    # context looks like a destination list or city/travel phrase, not arbitrary
    # prose such as dates, adjectives, or sentence-start nouns.
    destination_context = re.search(
        r"(destinations?|cities|city breaks?|places to (?:visit|go)|travel to|visit|in)[:\s][^.!?]{0,120}"
        + re.escape(candidate),
        source_text,
        flags=re.IGNORECASE,
    )
    country_context = re.search(
        re.escape(candidate) + r"\s*,\s*[A-Z][A-Za-z ]{3,}",
        source_text,
    )
    if len(tokens) == 1:
        if normalized in _DESTINATION_STOPWORDS:
            return False
        # Reject adjective-like demonyms and generic capitalized nouns unless
        # there is clear destination/list context.
        if normalized.endswith(("ian", "ese", "ish")) and not destination_context:
            return False
        return bool(destination_context or country_context)
    return bool(destination_context or country_context or len(tokens) <= 2)


def _web_rationale(city: str, title: str, description: str) -> str:
    source_text = description or title
    source_text = " ".join(source_text.split())
    if len(source_text) > 150:
        source_text = source_text[:150].rstrip() + "..."
    if source_text:
        return f"Web search results mention {city} for this theme: {source_text}"
    return f"Web search results mention {city} for this theme."


def _requested_region_countries(raw: str) -> set[str]:
    regions: set[str] = set()
    if any(term in raw for term in ["asian", "asia", "east asia", "southeast asia", "south east asia"]):
        regions.update(
            {
                "japan",
                "south korea",
                "china",
                "vietnam",
                "cambodia",
                "thailand",
                "singapore",
                "indonesia",
                "india",
            }
        )
    if any(term in raw for term in ["european", "europe", "western", "western history", "western civilization"]):
        regions.update(
            {
                "france",
                "spain",
                "italy",
                "united kingdom",
                "portugal",
                "netherlands",
                "czech republic",
                "iceland",
                "united states",
            }
        )
    if "mediterranean" in raw:
        regions.update({"spain", "italy", "france", "portugal"})
    return regions


class DestinationRecommendationAgent(Agent):
    name = "DestinationRecommendationAgent"

    def run(self, ctx: AgentContext) -> ChatResponse:
        ctx.emit({"type": "agent_start", "agent": self.name})

        query = self._build_query(ctx.parsed)
        ctx.emit({"type": "tool_start", "tool": "destination_rag", "query": query})
        web_results = self._destination_web_results(ctx)
        catalog_suggestions = self._rank_catalog(ctx.parsed, web_results=web_results)
        web_suggestions = self._web_discovered_suggestions(ctx.parsed, web_results)
        suggestions = self._merge_suggestions(web_suggestions, catalog_suggestions)
        if not suggestions:
            # Fallback for deployments without the catalog file: use the older
            # Chroma city-doc ranking path.
            vector_store = self._resolve_vector_store(ctx)
            self._ensure_city_docs_seeded(vector_store)
            suggestions = self._rank_vector_city_docs(vector_store, ctx.parsed, query)
        ctx.emit({"type": "tool_end", "tool": "destination_rag", "status": "completed"})

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

    def _rank_vector_city_docs(self, vector_store, parsed, query: str) -> list[dict[str, Any]]:
        results = vector_store.query(
            _CITY_DOCS_COLLECTION,
            query_text=query,
            n_results=_QUERY_CANDIDATES,
        )
        origin = (parsed.origin_city or "").strip().lower()
        suggestions: list[dict[str, Any]] = []
        seen_cities: set[str] = set()
        for result in results:
            city = (result.metadata.get("city") or "").strip()
            if not city:
                continue
            key = city.lower()
            if key in seen_cities or (origin and key == origin):
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
        return suggestions

    def _destination_web_results(self, ctx: AgentContext) -> list[dict[str, str]]:
        if not self._should_use_web_boost(ctx.parsed):
            return []
        query_intent = self._web_query_intent(ctx.parsed)
        ctx.emit({"type": "tool_start", "tool": "destination_web_search", "query": query_intent})
        result = run_destination_search_tool(
            query_intent=query_intent,
            count=5,
            search_tool=ctx.services.web_search_tool,
        )
        results = result.get("results", []) if result.get("status") == "ok" else []
        ctx.emit(
            {
                "type": "tool_end",
                "tool": "destination_web_search",
                "status": result.get("status", "unknown"),
                "count": len(results),
            }
        )
        return results

    @staticmethod
    def _should_use_web_boost(parsed) -> bool:
        raw = (parsed.raw_message or "").lower()
        # Destination suggestions should feel current and theme-aware. If the
        # user names a theme/interest such as anime, photography, food, beaches,
        # museums, nightlife, etc., run the web booster even without words like
        # "best" or "recommend".
        if parsed.interests:
            return True
        theme_terms = [
            "anime",
            "photography",
            "photo",
            "photos",
            "food",
            "restaurant",
            "restaurants",
            "beach",
            "beaches",
            "wellness",
            "fashion",
            "beer",
            "cycling",
            "music",
            "nightlife",
            "museum",
            "museums",
            "art",
            "architecture",
            "history",
            "nature",
            "family",
            "shopping",
        ]
        if any(term in raw for term in theme_terms):
            return True
        triggers = [
            "best",
            "recommend",
            "recommendation",
            "suggest",
            "where should",
            "trending",
            "current",
            "latest",
            "this year",
            "2026",
            "summer",
            "winter",
            "spring",
            "fall",
            "autumn",
            "cheap destinations",
            "affordable destinations",
        ]
        return any(trigger in raw for trigger in triggers)

    @staticmethod
    def _web_query_intent(parsed) -> str:
        terms: list[str] = []
        if parsed.interests:
            terms.extend(parsed.interests)
        if parsed.budget:
            terms.append(f"{parsed.budget} budget")
        if parsed.travel_style:
            terms.append(parsed.travel_style)
        raw = (parsed.raw_message or "").lower()
        for term in [
            "asian",
            "asia",
            "european",
            "europe",
            "mediterranean",
            "western",
            "western history",
            "beaches",
            "wellness",
            "nightlife",
            "food",
            "museums",
            "history",
            "family",
            "cheap",
            "luxury",
        ]:
            if term in raw and term not in terms:
                terms.append(term)
        terms.append("best")
        return " ".join(terms)

    @staticmethod
    def _rank_catalog(parsed, web_results: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
        catalog = _load_destination_catalog()
        if not catalog:
            return []
        origin = (parsed.origin_city or "").strip().lower()
        interests = {interest.lower() for interest in (parsed.interests or [])}
        raw = (parsed.raw_message or "").lower()
        raw_tokens = set(raw.replace("-", " ").split())
        budget = (parsed.budget or "").lower()
        style = (parsed.travel_style or "").lower()

        web_mentions = _catalog_city_mentions(catalog, web_results or [])
        requested_countries = _requested_region_countries(raw)
        ranked: list[tuple[float, int, dict[str, Any]]] = []
        for index, item in enumerate(catalog):
            city = str(item.get("city", "")).strip()
            if not city or city.lower() == origin:
                continue
            tags = {str(tag).lower() for tag in item.get("tags", [])}
            budget_levels = {str(level).lower() for level in item.get("budget_levels", [])}
            styles = {str(s).lower() for s in item.get("travel_styles", [])}
            country = str(item.get("country", "")).lower()

            score = 0.0
            matched_interests = interests & tags
            score += 4.0 * len(matched_interests)
            # Let natural words in the prompt match catalog tags too, covering
            # terms not yet in parser.INTEREST_KEYWORDS (e.g. wellness, fashion,
            # beer, cycling, northern lights).
            score += 1.5 * len(raw_tokens & tags)
            if budget and budget in budget_levels:
                score += 1.25
            if style and style in styles:
                score += 1.5
            if requested_countries and country in requested_countries:
                # Preserve geographic qualifiers from the raw prompt, e.g.
                # "Asian history" should prioritize Asia + history over generic
                # European history cities like Paris/Rome/Barcelona.
                score += 5.0
                if "history" in interests or "history" in raw_tokens or "ancient" in raw_tokens:
                    score += 2.0 if {"history", "ancient history", "temples", "archaeology"} & tags else 0.0
            if any(term in raw for term in ["beach", "beaches", "coast", "sea"]):
                score += 2.5 if {"beaches", "coast"} & tags else 0.0
            if any(term in raw for term in ["cheap", "affordable", "budget"]):
                score += 1.5 if "low" in budget_levels or "budget" in tags else 0.0
            if any(term in raw for term in ["luxury", "premium", "high-end"]):
                score += 1.5 if "luxury" in budget_levels or "luxury" in tags else 0.0
            if any(term in raw for term in ["relax", "relaxed", "wellness", "romantic"]):
                score += 1.5 if {"relaxed", "wellness", "romantic"} & (tags | styles) else 0.0
            mentions = web_mentions.get(city.lower(), 0)
            if mentions:
                score += 3.0 * mentions
            if not interests and not raw_tokens & tags:
                # Broad trip request: keep a balanced default without hardcoding
                # the same five cities at the top forever.
                score += max(0.0, 2.0 - index * 0.03)

            if score <= 0:
                continue
            ranked.append((score, -index, item))

        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        top_score = ranked[0][0] if ranked else 1.0
        suggestions = []
        for score, _, item in ranked[:_MAX_SUGGESTIONS]:
            suggestions.append(
                {
                    "city": item["city"],
                    "rationale": _web_boosted_rationale(item, web_mentions),
                    "match_score": round(min(1.0, score / max(top_score, 1.0)), 3),
                    "source": "catalog+web" if web_mentions.get(str(item["city"]).lower(), 0) else "catalog",
                }
            )
        return suggestions

    @staticmethod
    def _web_discovered_suggestions(parsed, web_results: list[dict[str, str]]) -> list[dict[str, Any]]:
        if not web_results:
            return []
        catalog_cities = {str(item.get("city", "")).lower() for item in _load_destination_catalog()}
        origin = (parsed.origin_city or "").strip().lower()
        counts: dict[str, int] = {}
        rationales: dict[str, str] = {}
        first_seen: dict[str, int] = {}

        for idx, result in enumerate(web_results):
            title = str(result.get("title", ""))
            description = str(result.get("description", ""))
            text = f"{title}. {description}"
            for candidate in _extract_destination_names(text):
                key = candidate.lower()
                if key == origin or key in catalog_cities:
                    continue
                counts[candidate] = counts.get(candidate, 0) + 1
                first_seen.setdefault(candidate, idx)
                rationales.setdefault(candidate, _web_rationale(candidate, title, description))

        ranked = sorted(counts, key=lambda city: (counts[city], -first_seen[city]), reverse=True)
        suggestions: list[dict[str, Any]] = []
        for city in ranked[:_MAX_SUGGESTIONS]:
            suggestions.append(
                {
                    "city": city,
                    "rationale": rationales[city],
                    "match_score": round(min(0.95, 0.78 + 0.05 * counts[city]), 3),
                    "source": "web",
                }
            )
        return suggestions

    @staticmethod
    def _merge_suggestions(
        web_suggestions: list[dict[str, Any]], catalog_suggestions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        # Web-discovered cities come first because they are the whole point of
        # using live search for broad/themed destination discovery. Catalog
        # suggestions then fill any remaining slots with stable local options.
        for suggestion in [*web_suggestions, *catalog_suggestions]:
            key = str(suggestion.get("city", "")).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(suggestion)
            if len(merged) >= _MAX_SUGGESTIONS:
                break
        return merged

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
    def _preference_summary(parsed) -> str:
        raw = (parsed.raw_message or "").lower()
        interests = list(parsed.interests or [])
        if any(term in raw for term in ["asian", "asia"]):
            if "history" in interests:
                return "Asian history"
            if interests:
                return "Asian " + ", ".join(interests)
            return "Asia"
        if "western" in raw:
            if "history" in interests:
                return "Western history"
            if interests:
                return "Western " + ", ".join(interests)
            return "Western destinations"
        return ", ".join(interests)

    @staticmethod
    def _build_message(parsed, suggestions: list[dict[str, Any]]) -> str:
        if not suggestions:
            return (
                "I couldn't match your preferences to a specific destination from the curated set. "
                "Could you name a city you're considering, or share a bit more about what you'd like to do?"
            )
        preference_summary = DestinationRecommendationAgent._preference_summary(parsed)
        lines = [
            "Based on what you're looking for"
            + (f" ({preference_summary})" if preference_summary else "")
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
