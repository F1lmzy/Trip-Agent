import hashlib


class FakeEmbedder:
    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = text.lower()
        return [
            float(any(word in normalized for word in ["food", "vegetarian", "vegan", "restaurant"])),
            float(any(word in normalized for word in ["museum", "museums", "gallery", "art"])),
            float(any(word in normalized for word in ["nature", "park", "garden"])),
            float(any(word in normalized for word in ["budget", "cheap", "affordable", "low"])),
            float(int(hashlib.sha1(normalized.encode()).hexdigest()[:2], 16)) / 255.0,
        ]
