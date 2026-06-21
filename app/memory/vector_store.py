import logging
from typing import Any, Protocol
from uuid import uuid4

logger = logging.getLogger(__name__)

import chromadb
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel, Field

from app.config import get_settings


logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

Metadata = dict[str, str | int | float | bool]


class Embedder(Protocol):
    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> Any:
        ...


class VectorSearchResult(BaseModel):
    id: str
    document: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    distance: float | None = None


class VectorStore:
    def __init__(
        self,
        path: str | None = None,
        embedding_model_name: str | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.path = path or get_settings().chroma_path
        self.embedding_model_name = embedding_model_name or "openrouter"
        self._embedder = embedder
        self._client = chromadb.PersistentClient(
            path=self.path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def get_or_create_collection(self, name: str):
        return self._client.get_or_create_collection(name=name, embedding_function=None)

    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        metadatas: list[Metadata] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        if not documents:
            return []

        document_ids = ids or [str(uuid4()) for _ in documents]
        self._validate_lengths(documents, metadatas, document_ids)

        upsert_payload: dict[str, Any] = {
            "ids": document_ids,
            "documents": documents,
            "embeddings": self._embed(documents),
        }
        if metadatas is not None:
            upsert_payload["metadatas"] = metadatas

        collection = self.get_or_create_collection(collection_name)
        collection.upsert(**upsert_payload)
        return document_ids

    def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 3,
        where: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        collection = self.get_or_create_collection(collection_name)
        raw_results = collection.query(
            query_embeddings=self._embed([query_text]),
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return self._format_results(raw_results)

    def delete_collection(self, collection_name: str) -> None:
        try:
            self._client.delete_collection(name=collection_name)
        except ValueError:
            logger.debug("delete_collection: collection %r not found (ignored)", collection_name)
            return

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._get_embedder().encode(texts, normalize_embeddings=True)
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [list(embedding) for embedding in embeddings]

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            from app.memory.openrouter_embedder import OpenRouterEmbedder

            self._embedder = OpenRouterEmbedder(
                model=self.embedding_model_name
                if self.embedding_model_name != "openrouter"
                else None
            )
        return self._embedder

    @staticmethod
    def _validate_lengths(documents: list[str], metadatas: list[Metadata] | None, ids: list[str]) -> None:
        if len(documents) != len(ids):
            raise ValueError("documents, metadatas, and ids must have the same length")
        if metadatas is not None and len(documents) != len(metadatas):
            raise ValueError("documents, metadatas, and ids must have the same length")

    @staticmethod
    def _format_results(raw_results: dict[str, Any]) -> list[VectorSearchResult]:
        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        results: list[VectorSearchResult] = []
        for index, result_id in enumerate(ids):
            results.append(
                VectorSearchResult(
                    id=result_id,
                    document=documents[index],
                    metadata=metadatas[index] or {},
                    distance=distances[index] if index < len(distances) else None,
                )
            )
        return results
