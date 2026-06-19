"""Repro: resolve real Wikimedia Commons image URLs for sample places.

Hits the LIVE Commons API (free, no key, no quota). NOT run by pytest.
Use this to confirm the resolver returns real, HTTP-200-able image URLs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from app.tools.wikimedia_image_tool import resolve_place_image

_PLACES = ["Eiffel Tower", "British Museum", "Senso-ji Temple", "Museum of Liverpool", "Trevi Fountain"]


def main() -> int:
    print("Resolving Wikimedia Commons images (LIVE, free API)...")
    with httpx.Client(timeout=15.0) as client:
        for place in _PLACES:
            url = resolve_place_image(place, client=client)
            status = "n/a"
            if url:
                try:
                    head = httpx.head(
                        url,
                        timeout=15.0,
                        follow_redirects=True,
                        headers={"User-Agent": "TripAgent/1.0 (https://github.com/srirajkavin/Trip-Agent; educational project)"},
                    )
                    status = str(head.status_code)
                except Exception as error:  # noqa: BLE001
                    status = f"HEAD-error: {error}"
            print(f"  {place!r:24} -> {status:>5}  {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
