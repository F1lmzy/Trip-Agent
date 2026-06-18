import hashlib


class FakeEmbedder:
    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = text.lower()
        return [
            float(any(word in normalized for word in ["food", "vegetarian", "vegan", "restaurant", "market", "hawker"])),
            float(any(word in normalized for word in ["museum", "museums", "gallery", "art", "louvre", "metropolitan"])),
            float(any(word in normalized for word in ["nature", "park", "garden", "gardens", "outdoor"])),
            float(any(word in normalized for word in ["budget", "cheap", "affordable", "low"])),
            float(any(word in normalized for word in ["anime", "manga", "gaming", "akihabara", "arcade"])),
            float(any(word in normalized for word in ["photo", "photos", "photography", "view", "views", "sky", "skyline", "shibuya"])),
            float(any(word in normalized for word in ["culture", "temple", "history", "historic", "asakusa", "senso"])),
            float("tokyo" in normalized),
            float("singapore" in normalized),
            float("paris" in normalized),
            float("new york" in normalized or "brooklyn" in normalized or "manhattan" in normalized),
            float("mumbai" in normalized or "colaba" in normalized or "bandra" in normalized),
            float(int(hashlib.sha1(normalized.encode()).hexdigest()[:2], 16)) / 255.0,
        ]
