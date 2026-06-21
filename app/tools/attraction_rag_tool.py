import json
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.memory.vector_store import VectorSearchResult, VectorStore
from app.tools.external_content import (
    external_docs_to_vectors,
    fetch_city_attractions,
    fetch_city_docs,
)


CITY_DOCS_COLLECTION = "travel_city_docs"
ATTRACTIONS_COLLECTION = "travel_attractions"
_VALID_EXTERNAL_SOURCE = "external_wikivoyage"
_DATA_DIR = Path(__file__).parents[1] / "data"
_ATTRACTIONS_PATH = _DATA_DIR / "attractions.json"
_CITY_DOCS_DIR = _DATA_DIR / "city_docs"


class AttractionRagResult(BaseModel):
    tool_name: str = "attraction_rag_tool"
    status: str
    city: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    rag_trace: dict[str, list[dict[str, Any]]] = Field(default_factory=lambda: {"hop_1": [], "hop_2": []})


def load_attraction_documents(path: Path = _ATTRACTIONS_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_city_documents(directory: Path = _CITY_DOCS_DIR) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    for path in sorted(directory.glob("*.md")):
        city = path.stem.replace("_", " ").title()
        docs.append({"id": f"city-{path.stem}", "city": city, "text": path.read_text(encoding="utf-8")})
    return docs


def supported_rag_cities() -> set[str]:
    return {doc["city"] for doc in load_city_documents()}


def has_curated_rag_city(city: str) -> bool:
    return _normalize_city(city) in supported_rag_cities()


class AttractionRagTool:
    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self.vector_store = vector_store or VectorStore()
        self._seeded = False

    def seed(self) -> None:
        self._seed_city_docs()
        self._seed_attractions()

    def seed_if_needed(self) -> None:
        """Seed the vector store exactly once. Idempotent — safe to call repeatedly."""
        if not self._seeded:
            self.seed()
            self._seeded = True

    def run(
        self,
        city: str,
        interests: list[str] | None = None,
        limit: int = 5,
        http_client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        normalized_city = _normalize_city(city)
        interest_terms = " ".join(interests or ["highlights"])

        hop_1 = self._run_hop1(normalized_city, http_client)
        if isinstance(hop_1, dict):
            return hop_1  # early exit — no city context found
        hop_1_results, city_context = hop_1

        hop_2_results = self._run_hop2(
            normalized_city, interest_terms, city_context, hop_1_results, limit, http_client,
        )

        if not hop_2_results:
            return AttractionRagResult(
                status="no_results",
                city=normalized_city,
                rag_trace={"hop_1": _summarize_results(hop_1_results), "hop_2": []},
            ).model_dump()

        return AttractionRagResult(
            status="ok",
            city=normalized_city,
            results=[self._result_to_attraction(result, interests or []) for result in hop_2_results],
            rag_trace={
                "hop_1": _summarize_results(hop_1_results),
                "hop_2": _summarize_results(hop_2_results),
            },
        ).model_dump()

    def _run_hop1(
        self, normalized_city: str, http_client: httpx.Client | None,
    ) -> tuple[list[VectorSearchResult], str] | dict[str, Any]:
        """Retrieve city-level context from the vector store.

        Returns a ``(hop_1_results, city_context)`` tuple on success, or an
        ``AttractionRagResult`` dict (early exit) when no context is found.
        """
        hop_1_query = f"{normalized_city} major neighborhoods attractions food museums nature travel overview"
        hop_1_results = self.vector_store.query(
            CITY_DOCS_COLLECTION,
            query_text=hop_1_query,
            n_results=2,
            where={"city": normalized_city},
        )

        # Purge legacy external data (e.g. external_wikipedia from before the
        # Wikivoyage migration) so stale disambiguation junk is never served as
        # city context or attractions. Re-query hop_1; if still empty, the
        # normal external-ingest path fetches fresh data from Wikivoyage.
        if hop_1_results and _hop_1_is_legacy_external(hop_1_results):
            self._purge_external_city_data(normalized_city)
            hop_1_results = self.vector_store.query(
                CITY_DOCS_COLLECTION,
                query_text=hop_1_query,
                n_results=2,
                where={"city": normalized_city},
            )

        if not hop_1_results:
            ingested = self._ingest_external_city_docs(normalized_city, http_client)
            if ingested:
                hop_1_results = self.vector_store.query(
                    CITY_DOCS_COLLECTION,
                    query_text=hop_1_query,
                    n_results=2,
                    where={"city": normalized_city},
                )

        if not hop_1_results:
            return AttractionRagResult(status="no_results", city=normalized_city).model_dump()

        city_context = " ".join(result.document for result in hop_1_results[:2])
        return hop_1_results, city_context

    def _run_hop2(
        self,
        normalized_city: str,
        interest_terms: str,
        city_context: str,
        hop_1_results: list[VectorSearchResult],
        limit: int,
        http_client: httpx.Client | None,
    ) -> list[VectorSearchResult]:
        """Retrieve interest-specific attractions using the city context.

        Handles stale external name detection, re-ingestion, and fallback
        ingest when hop_2 is empty but hop_1 came from external sources.
        Returns a list of results (may be empty).
        """
        hop_2_query = f"{normalized_city} {interest_terms} {city_context}"
        hop_2_results = self.vector_store.query(
            ATTRACTIONS_COLLECTION,
            query_text=hop_2_query,
            n_results=limit,
            where={"city": normalized_city},
        )

        # Detect stale external data (old format with text-chunk names) and re-ingest.
        if hop_2_results and _has_stale_external_names(hop_2_results):
            self._delete_stale_external_attractions(normalized_city)
            self._ingest_external_attractions(normalized_city, http_client)
            hop_2_results = self.vector_store.query(
                ATTRACTIONS_COLLECTION,
                query_text=hop_2_query,
                n_results=limit,
                where={"city": normalized_city},
            )

        # If hop_2 is still empty but hop_1 came from external sources, the
        # attractions were never ingested (or were cleared). Try to fetch them.
        if not hop_2_results and _hop_1_is_external(hop_1_results):
            self._ingest_external_attractions(normalized_city, http_client)
            hop_2_results = self.vector_store.query(
                ATTRACTIONS_COLLECTION,
                query_text=hop_2_query,
                n_results=limit,
                where={"city": normalized_city},
            )

        return hop_2_results

    def _ingest_external_city_docs(
        self,
        city: str,
        http_client: httpx.Client | None,
    ) -> bool:
        """Fetch external city docs and attractions, embedding both into ChromaDB.

        City overview text goes into the city_docs collection (hop_1).
        Parsed attractions from Wikivoyage's See/Do sections go into the
        attractions collection (hop_2) with real names instead of text chunks.
        Returns True if any docs were ingested, False otherwise.
        """
        docs = fetch_city_docs(city, client=http_client)
        if not docs:
            return False
        payload = external_docs_to_vectors(docs)
        self.vector_store.add_documents(
            CITY_DOCS_COLLECTION,
            documents=payload["documents"],
            metadatas=payload["metadatas"],
            ids=payload["ids"],
        )

        # Try to fetch real attractions from Wikivoyage's See/Do sections.
        attractions = fetch_city_attractions(city, client=http_client)
        if attractions:
            self.vector_store.add_documents(
                ATTRACTIONS_COLLECTION,
                documents=[
                    f"{a.name} in {a.city}. {a.description}" for a in attractions
                ],
                metadatas=[
                    {
                        "city": a.city,
                        "type": "attraction",
                        "name": a.name,
                        "source": f"external_{a.source}",
                        "categories": "",
                    }
                    for a in attractions
                ],
                ids=[
                    f"ext-{a.source}-{a.city.lower().replace(' ', '-')}-{a.name.lower().replace(' ', '-')}"
                    for a in attractions
                ],
            )
        else:
            # Fallback: add city docs to attractions collection without real names.
            attraction_metadatas = [
                {**meta, "type": "attraction", "name": f"{city} overview (section {i})"}
                for i, meta in enumerate(payload["metadatas"])
            ]
            attraction_ids = [f"{doc_id}-attr" for doc_id in payload["ids"]]
            self.vector_store.add_documents(
                ATTRACTIONS_COLLECTION,
                documents=payload["documents"],
                metadatas=attraction_metadatas,
                ids=attraction_ids,
            )
        return True

    def _ingest_external_attractions(
        self,
        city: str,
        http_client: httpx.Client | None,
    ) -> bool:
        """Fetch and embed real attractions for a city without touching city docs.

        Used when stale attraction data is detected but city docs are fine.
        Returns True if attractions were ingested, False otherwise.
        """
        attractions = fetch_city_attractions(city, client=http_client)
        if not attractions:
            return False
        self.vector_store.add_documents(
            ATTRACTIONS_COLLECTION,
            documents=[f"{a.name} in {a.city}. {a.description}" for a in attractions],
            metadatas=[
                {
                    "city": a.city,
                    "type": "attraction",
                    "name": a.name,
                    "source": f"external_{a.source}",
                    "categories": "",
                }
                for a in attractions
            ],
            ids=[
                f"ext-{a.source}-{a.city.lower().replace(' ', '-')}-{a.name.lower().replace(' ', '-')}"
                for a in attractions
            ],
        )
        return True

    def _delete_stale_external_attractions(self, city: str) -> None:
        """Delete all external attraction entries for a city from ChromaDB.

        Clears both legacy (external_wikipedia) and old-format Wikivoyage
        entries (text-chunk names) before re-ingesting with proper attraction
        names from Wikivoyage's See/Do sections. Curated entries are untouched.
        """
        self._delete_external_entries(ATTRACTIONS_COLLECTION, city)

    def _purge_external_city_data(self, city: str) -> None:
        """Delete all external (non-curated) docs and attractions for a city.

        Used when legacy external sources (e.g. external_wikipedia) are
        detected in hop_1 so the tool re-fetches fresh data from Wikivoyage
        instead of serving stale disambiguation text as attractions.
        """
        self._delete_external_entries(CITY_DOCS_COLLECTION, city)
        self._delete_external_entries(ATTRACTIONS_COLLECTION, city)

    def _delete_external_entries(self, collection_name: str, city: str) -> None:
        """Delete every entry tagged with an ``external_*`` source for a city."""
        collection = self.vector_store.get_or_create_collection(collection_name)
        existing = collection.get(where={"city": city})
        ids = [
            doc_id
            for doc_id, meta in zip(
                existing.get("ids") or [], existing.get("metadatas") or []
            )
            if (meta or {}).get("source", "").startswith("external_")
        ]
        if ids:
            collection.delete(ids=ids)

    def _seed_city_docs(self) -> None:
        docs = load_city_documents()
        self.vector_store.add_documents(
            CITY_DOCS_COLLECTION,
            documents=[doc["text"] for doc in docs],
            metadatas=[
                {"city": doc["city"], "type": "city_overview", "source": "curated_wikivoyage_style"} for doc in docs
            ],
            ids=[doc["id"] for doc in docs],
        )

    def _seed_attractions(self) -> None:
        attractions = load_attraction_documents()
        self.vector_store.add_documents(
            ATTRACTIONS_COLLECTION,
            documents=[_attraction_text(attraction) for attraction in attractions],
            metadatas=[_attraction_metadata(attraction) for attraction in attractions],
            ids=[attraction["id"] for attraction in attractions],
        )

    @staticmethod
    def _result_to_attraction(result: VectorSearchResult, interests: list[str]) -> dict[str, Any]:
        metadata = result.metadata
        categories = str(metadata.get("categories", "")).split(",") if metadata.get("categories") else []
        matched = sorted(set(categories) & set(interests))
        reason = "Matched broad city context."
        if matched:
            reason = f"Matched interests: {', '.join(matched)}."

        return {
            "name": metadata.get("name", result.id),
            "description": result.document,
            "categories": categories,
            "indoor": metadata.get("indoor"),
            "budget_level": metadata.get("budget_level"),
            "reason": reason,
        }


def _attraction_text(attraction: dict[str, Any]) -> str:
    categories = ", ".join(attraction["category"])
    return f"{attraction['name']} in {attraction['city']}. Categories: {categories}. {attraction['description']}"


def _attraction_metadata(attraction: dict[str, Any]) -> dict[str, str | int | float | bool]:
    return {
        "city": attraction["city"],
        "type": "attraction",
        "name": attraction["name"],
        "categories": ",".join(attraction["category"]),
        "indoor": attraction["indoor"],
        "budget_level": attraction["budget_level"],
        "estimated_time_hours": attraction["estimated_time_hours"],
    }


def _summarize_results(results: list[VectorSearchResult]) -> list[dict[str, Any]]:
    return [
        {
            "id": result.id,
            "summary": result.document[:240],
            "metadata": result.metadata,
        }
        for result in results
    ]


def _normalize_city(city: str) -> str:
    return city.strip().replace("_", " ").title()


def _is_legacy_external_source(source: str) -> bool:
    """True for external sources that predate the Wikivoyage migration."""
    return source.startswith("external_") and source != _VALID_EXTERNAL_SOURCE


def _hop_1_is_legacy_external(results: list[VectorSearchResult]) -> bool:
    """Check if hop_1 results contain legacy (non-Wikivoyage) external data."""
    return any(
        _is_legacy_external_source(result.metadata.get("source", ""))
        for result in results
    )


def _has_stale_external_names(results: list[VectorSearchResult]) -> bool:
    """Detect external attraction entries that should be re-ingested before use.

    Flags two cases: (1) legacy external sources (e.g. external_wikipedia)
    from before the Wikivoyage migration; (2) old-format Wikivoyage entries
    whose 'name' is a text-chunk fragment (len > 40, e.g.
    'Liverpool is a big city in Merseyside...'). The 'City overview
    (section N)' name is NOT flagged here: it is the legitimate fallback used
    when Wikivoyage has no parseable See/Do section, and is reached via the
    hop_1 legacy purge instead.
    """
    for result in results:
        source = result.metadata.get("source", "")
        name = result.metadata.get("name", "")
        if not source.startswith("external_"):
            continue
        if _is_legacy_external_source(source):
            return True
        if len(name) > 40:
            return True
    return False


def _hop_1_is_external(results: list[VectorSearchResult]) -> bool:
    """Check if hop_1 results came from external ingestion (not curated data)."""
    for result in results:
        source = result.metadata.get("source", "")
        if source.startswith("external_"):
            return True
    return False
