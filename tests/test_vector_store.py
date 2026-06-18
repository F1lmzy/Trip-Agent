import hashlib

from app.memory.vector_store import VectorStore


class FakeEmbedder:
    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = text.lower()
        return [
            float(any(word in normalized for word in ["anime", "manga", "gaming", "akihabara"])),
            float(any(word in normalized for word in ["museum", "museums", "ueno", "gallery"])),
            float(any(word in normalized for word in ["paris", "louvre"])),
            float(int(hashlib.sha1(normalized.encode()).hexdigest()[:2], 16)) / 255.0,
        ]


def make_store(tmp_path) -> VectorStore:
    return VectorStore(path=str(tmp_path), embedder=FakeEmbedder())


def test_vector_store_can_create_collection(tmp_path):
    store = make_store(tmp_path)

    collection = store.get_or_create_collection("test_collection")

    assert collection.name == "test_collection"


def test_vector_store_adds_and_queries_documents(tmp_path):
    store = make_store(tmp_path)
    store.add_documents(
        "attractions",
        documents=[
            "Akihabara is known for anime, manga, gaming, and electronics.",
            "Ueno Park has museums, gardens, and cultural attractions.",
        ],
        metadatas=[{"city": "Tokyo"}, {"city": "Tokyo"}],
        ids=["akihabara", "ueno"],
    )

    results = store.query("attractions", "anime gaming Tokyo", n_results=1)

    assert len(results) == 1
    assert results[0].id == "akihabara"
    assert "Akihabara" in results[0].document
    assert results[0].metadata == {"city": "Tokyo"}


def test_vector_store_supports_metadata_filter(tmp_path):
    store = make_store(tmp_path)
    store.add_documents(
        "attractions",
        documents=[
            "Akihabara is a Tokyo district for anime and gaming.",
            "The Louvre is a Paris museum with major art collections.",
        ],
        metadatas=[{"city": "Tokyo"}, {"city": "Paris"}],
        ids=["tokyo-akihabara", "paris-louvre"],
    )

    results = store.query("attractions", "museum art", n_results=2, where={"city": "Paris"})

    assert results
    assert {result.metadata["city"] for result in results} == {"Paris"}


def test_vector_store_can_delete_collection(tmp_path):
    store = make_store(tmp_path)
    store.get_or_create_collection("temporary_collection")

    store.delete_collection("temporary_collection")
    recreated = store.get_or_create_collection("temporary_collection")

    assert recreated.name == "temporary_collection"


def test_vector_store_validates_input_lengths(tmp_path):
    store = make_store(tmp_path)

    try:
        store.add_documents("bad", documents=["one"], metadatas=[], ids=["one"])
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched input lengths")
