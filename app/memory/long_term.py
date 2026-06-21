from uuid import uuid4

from app.memory.vector_store import VectorStore


PREFERENCES_COLLECTION = "user_preferences"


class LongTermMemory:
    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self.vector_store = vector_store or VectorStore()

    def add_preference(self, user_id: str, preference: str) -> str:
        memory_id = str(uuid4())
        self.vector_store.add_documents(
            PREFERENCES_COLLECTION,
            documents=[preference],
            metadatas=[{"user_id": user_id, "type": "preference"}],
            ids=[memory_id],
        )
        return memory_id

    def get_preferences(self, user_id: str) -> list[str]:
        collection = self.vector_store.get_or_create_collection(PREFERENCES_COLLECTION)
        results = collection.get(where={"user_id": user_id}, include=["documents"])
        return list(results.get("documents") or [])

    def search_preferences(self, user_id: str, query: str, limit: int = 3) -> list[str]:
        results = self.vector_store.query(
            PREFERENCES_COLLECTION,
            query_text=query,
            n_results=limit,
            where={"user_id": user_id},
        )
        return [result.document for result in results]

    def delete_preferences_with_prefix(self, user_id: str, prefix: str, except_value: str | None = None) -> None:
        """Delete stored preferences in one category for a user.

        Long-term preferences are stored as add-only vector documents. For
        mutually exclusive categories such as budget, replacing stale values
        prevents old vectors like "Budget preference: low" from bleeding into
        later requests that explicitly say "unlimited budget".
        """
        collection = self.vector_store.get_or_create_collection(PREFERENCES_COLLECTION)
        existing = collection.get(where={"user_id": user_id}, include=["documents"])
        ids = existing.get("ids") or []
        documents = existing.get("documents") or []
        ids_to_delete = [
            doc_id
            for doc_id, document in zip(ids, documents, strict=False)
            if isinstance(document, str)
            and document.startswith(prefix)
            and (except_value is None or document != except_value)
        ]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)

    def clear_preferences(self, user_id: str) -> None:
        collection = self.vector_store.get_or_create_collection(PREFERENCES_COLLECTION)
        existing = collection.get(where={"user_id": user_id})
        ids = existing.get("ids") or []
        if ids:
            collection.delete(ids=ids)


long_term_memory = LongTermMemory()
