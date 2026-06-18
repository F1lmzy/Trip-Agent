import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.memory.vector_store import VectorSearchResult, VectorStore


CITY_DOCS_COLLECTION = "travel_city_docs"
ATTRACTIONS_COLLECTION = "travel_attractions"
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

    def seed(self) -> None:
        self._seed_city_docs()
        self._seed_attractions()

    def run(self, city: str, interests: list[str] | None = None, limit: int = 5) -> dict[str, Any]:
        normalized_city = _normalize_city(city)
        interest_terms = " ".join(interests or ["highlights"])

        hop_1_query = f"{normalized_city} major neighborhoods attractions food museums nature travel overview"
        hop_1_results = self.vector_store.query(
            CITY_DOCS_COLLECTION,
            query_text=hop_1_query,
            n_results=2,
            where={"city": normalized_city},
        )

        if not hop_1_results:
            return AttractionRagResult(status="no_results", city=normalized_city).model_dump()

        city_context = " ".join(result.document for result in hop_1_results[:2])
        hop_2_query = f"{normalized_city} {interest_terms} {city_context}"
        hop_2_results = self.vector_store.query(
            ATTRACTIONS_COLLECTION,
            query_text=hop_2_query,
            n_results=limit,
            where={"city": normalized_city},
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
