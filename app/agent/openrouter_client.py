from typing import Any

import httpx

from app.config import get_settings

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra"


def call_openrouter(
    messages: list[dict[str, str]],
    api_key: str | None = None,
    model: str | None = None,
    client: httpx.Client | None = None,
    temperature: float = 0.4,
) -> dict[str, Any]:
    settings = get_settings()
    resolved_api_key = api_key or settings.openrouter_api_key
    resolved_model = model or settings.openrouter_model or DEFAULT_OPENROUTER_MODEL

    if not resolved_api_key:
        return _fallback_result(
            status="fallback_missing_api_key",
            model=resolved_model,
            message="OpenRouter unavailable because OPENROUTER_API_KEY is not configured.",
        )

    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": resolved_model,
        "messages": messages,
        "temperature": temperature,
    }

    try:
        if client is None:
            with httpx.Client(timeout=30) as owned_client:
                response = owned_client.post(OPENROUTER_CHAT_URL, headers=headers, json=payload)
        else:
            response = client.post(OPENROUTER_CHAT_URL, headers=headers, json=payload)

        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        return _fallback_result(
            status="fallback_api_error",
            model=resolved_model,
            message="OpenRouter failed or returned an invalid response.",
        )

    return {
        "status": "ok",
        "source": "openrouter",
        "model": resolved_model,
        "content": str(content),
    }


def _fallback_result(status: str, model: str, message: str) -> dict[str, Any]:
    return {
        "status": status,
        "source": "fallback",
        "model": model,
        "content": None,
        "message": message,
    }
