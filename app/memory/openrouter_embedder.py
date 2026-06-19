"""Embedder that calls the OpenRouter embeddings API.

Replaces the heavy sentence-transformers + PyTorch dependency with a
lightweight HTTP call to OpenRouter's OpenAI-compatible /embeddings endpoint.
The embedder uses the existing OPENROUTER_API_KEY and falls back to a
deterministic hash-based embedding when the API is unavailable.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

import httpx

from app.config import get_settings

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
_FALLBACK_DIMENSION = 2048

logger = logging.getLogger(__name__)


class OpenRouterEmbedder:
    """Encodes texts via the OpenRouter embeddings API with in-memory caching.

    Implements the ``Embedder`` protocol expected by ``VectorStore``:

        def encode(self, texts: list[str], normalize_embeddings: bool = True) -> Any:
            ...

    When the API is unavailable (missing key, network error, timeout) the
    embedder falls back to a deterministic hash-based 2048-dimensional
    vector so the application keeps working with degraded retrieval quality.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or getattr(settings, "openrouter_embedding_model", None) or DEFAULT_EMBEDDING_MODEL
        self.api_key = api_key or settings.openrouter_api_key
        self._client = client
        # In-memory cache: text -> embedding vector
        self._cache: dict[str, list[float]] = {}

    def encode(
        self, texts: list[str], normalize_embeddings: bool = True
    ) -> list[list[float]]:
        """Encode a list of texts into embedding vectors.

        Uses the OpenRouter embeddings API with an in-memory cache.
        Falls back to deterministic hash-based embeddings on failure.
        """
        # Separate cached from uncached
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        for i, text in enumerate(texts):
            if text in self._cache:
                continue
            uncached_indices.append(i)
            uncached_texts.append(text)

        # Fetch uncached texts from the API
        if uncached_texts:
            try:
                api_embeddings = self._call_api(uncached_texts)
                for idx, emb in zip(uncached_indices, api_embeddings):
                    self._cache[texts[idx]] = emb
            except Exception:
                logger.warning(
                    "OpenRouter embeddings API failed, using hash-based fallback",
                    exc_info=True,
                )
                for idx in uncached_indices:
                    self._cache[texts[idx]] = _fallback_embed(texts[idx])

        # Build result in original order
        result = [self._cache[text] for text in texts]

        if normalize_embeddings:
            result = [_l2_normalize(emb) for emb in result]

        return result

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenRouter embeddings API and return embedding vectors."""
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not configured – cannot call embeddings API"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            # The API accepts a string or an array of strings
            "input": texts if len(texts) > 1 else texts[0],
        }

        if self._client is not None:
            response = self._client.post(
                OPENROUTER_EMBEDDINGS_URL, headers=headers, json=payload
            )
        else:
            timeout = httpx.Timeout(30.0, connect=10.0)
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    OPENROUTER_EMBEDDINGS_URL, headers=headers, json=payload
                )

        response.raise_for_status()
        body = response.json()
        return [item["embedding"] for item in body["data"]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _l2_normalize(embedding: list[float]) -> list[float]:
    """L2-normalize a vector in-place."""
    norm = math.sqrt(sum(v * v for v in embedding))
    if norm == 0:
        return embedding
    return [v / norm for v in embedding]


def _fallback_embed(text: str) -> list[float]:
    """Deterministic hash-based fallback embedding.

    Produces a 2048-dimensional vector from a seeded PRNG so that ChromaDB
    queries still return results (though with degraded relevance) when the
    OpenRouter embeddings API is unavailable.
    """
    normalized = text.lower()
    digest = hashlib.sha256(normalized.encode()).digest()
    # Use first 8 bytes as the PRNG seed
    rng_state = int.from_bytes(digest[:8], "big")
    vec: list[float] = []
    for _ in range(_FALLBACK_DIMENSION):
        rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        v = (rng_state & 0xFFFF) / 65535.0  # range [0, 1)
        vec.append(v)
    return _l2_normalize(vec)
