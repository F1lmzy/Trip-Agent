from collections import Counter, defaultdict
from typing import Any

import httpx

OPENWEATHER_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


def run_weather_tool(
    city: str,
    api_key: str | None = None,
    units: str = "metric",
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    normalized_city = city.strip().replace("_", " ").title()

    if not api_key:
        return _fallback_result(
            city=normalized_city,
            status="fallback_missing_api_key",
            summary="Weather unavailable because OPENWEATHER_API_KEY is not configured.",
        )

    params = {"q": normalized_city, "appid": api_key, "units": units}

    try:
        if client is None:
            with httpx.Client(timeout=10) as owned_client:
                response = owned_client.get(OPENWEATHER_FORECAST_URL, params=params)
        else:
            response = client.get(OPENWEATHER_FORECAST_URL, params=params)

        response.raise_for_status()
        payload = response.json()
        forecast = _normalize_forecast(payload.get("list", []), units=units)
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return _fallback_result(
            city=normalized_city,
            status="fallback_api_error",
            summary="Weather unavailable because OpenWeatherMap could not be reached or returned an invalid response.",
        )

    if not forecast:
        return _fallback_result(
            city=normalized_city,
            status="fallback_no_forecast",
            summary="Weather unavailable because OpenWeatherMap returned no forecast entries.",
        )

    return {
        "tool_name": "weather_tool",
        "status": "ok",
        "city": normalized_city,
        "source": "openweathermap",
        "forecast": forecast,
    }


def _normalize_forecast(entries: list[dict[str, Any]], units: str = "metric", max_days: int = 5) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        dt_txt = entry.get("dt_txt")
        if not dt_txt:
            continue
        grouped[dt_txt[:10]].append(entry)

    forecast: list[dict[str, Any]] = []
    for date in sorted(grouped)[:max_days]:
        day_entries = grouped[date]
        summaries = [
            entry.get("weather", [{}])[0].get("main", "Unknown")
            for entry in day_entries
            if entry.get("weather")
        ]
        summary = Counter(summaries).most_common(1)[0][0] if summaries else "Unknown"
        temperatures = [_number(entry.get("main", {}).get("temp")) for entry in day_entries]
        feels_like = [_number(entry.get("main", {}).get("feels_like")) for entry in day_entries]
        humidity = [_number(entry.get("main", {}).get("humidity")) for entry in day_entries]
        wind_speed = [_number(entry.get("wind", {}).get("speed")) for entry in day_entries]

        day = {
            "date": date,
            "summary": summary,
            "humidity": round(_average(humidity), 1),
            "wind_speed": round(_average(wind_speed), 1),
            "outdoor_suitability": _outdoor_suitability(summary, _average(temperatures), _average(wind_speed)),
        }
        temp_unit = "c" if units == "metric" else "f"
        day[f"temperature_{temp_unit}"] = round(_average(temperatures), 1)
        day[f"feels_like_{temp_unit}"] = round(_average(feels_like), 1)
        forecast.append(day)

    return forecast


def _outdoor_suitability(summary: str, temperature: float, wind_speed: float) -> str:
    normalized = summary.lower()
    if any(term in normalized for term in ["rain", "thunderstorm", "snow", "drizzle"]) or wind_speed >= 10:
        return "poor"
    if any(term in normalized for term in ["cloud", "mist", "fog"]) or temperature <= 5 or temperature >= 32:
        return "fair"
    return "good"


def _fallback_result(city: str, status: str, summary: str) -> dict[str, Any]:
    return {
        "tool_name": "weather_tool",
        "status": status,
        "city": city,
        "source": "fallback",
        "forecast": [
            {
                "date": None,
                "summary": summary,
                "outdoor_suitability": "unknown",
            }
        ],
    }


def _number(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
