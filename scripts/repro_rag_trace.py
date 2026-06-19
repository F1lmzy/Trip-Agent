"""Reproduce the Wikivoyage RAG trace with the REAL embedder + live Wikivoyage.

Run: .venv/bin/python scripts/repro_rag_trace.py [CITY]

Prints hop_1 / hop_2 contents and reports whether the embedder used the
OpenRouter API or silently fell back to deterministic hash vectors.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure repo root on path when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from app.memory.openrouter_embedder import OpenRouterEmbedder, _fallback_embed
from app.memory.vector_store import VectorStore
from app.tools.attraction_rag_tool import AttractionRagTool

CITY = sys.argv[1] if len(sys.argv) > 1 else "Liverpool"
INTERESTS = ["music", "food", "history"]

# Capture embedder mode by wrapping _call_api and _fallback_embed.
counts = {"api_calls": 0, "fallback_calls": 0, "api_errors": 0}
api_warnings: list[str] = []

_orig_call_api = OpenRouterEmbedder._call_api


def _wrapped_call_api(self, texts):
    counts["api_calls"] += 1
    try:
        return _orig_call_api(self, texts)
    except Exception as e:  # noqa: BLE001
        counts["api_errors"] += 1
        api_warnings.append(f"{type(e).__name__}: {e}")
        raise


def _wrapped_fallback(text):
    counts["fallback_calls"] += 1
    return _fallback_embed(text)


OpenRouterEmbedder._call_api = _wrapped_call_api
import app.memory.openrouter_embedder as _emb_mod
_emb_mod._fallback_embed = _wrapped_fallback

# Capture the "API failed" warning log.
logging.basicConfig(level=logging.WARNING)
_warn_records: list[str] = []


class _Collector(logging.Handler):
    def emit(self, record):
        if "OpenRouter embeddings API failed" in record.getMessage():
            _warn_records.append(record.getMessage())


logging.getLogger("app.memory.openrouter_embedder").addHandler(_Collector())


def dump_hop(label, entries):
    print(f"\n=== {label} ({len(entries)} entries) ===")
    for i, e in enumerate(entries):
        meta = e.get("metadata", {})
        doc = (e.get("summary", "") or "").replace("\n", " ")
        print(
            f"[{i}] id={e.get('id')!r}\n"
            f"    source={meta.get('source')!r} name={meta.get('name')!r} "
            f"type={meta.get('type')!r} city={meta.get('city')!r}\n"
            f"    doc[:240]={doc!r}"
        )


def main() -> int:
    # Fresh isolated Chroma path so we don't pollute ./chroma_db and so the
    # tool is forced to fetch externally for a non-curated city.
    tmp_path = Path("./.repro_chroma").resolve()
    if tmp_path.exists():
        import shutil
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    print(f"Reproducing RAG trace for city={CITY!r} interests={INTERESTS}")
    print(f"Using REAL OpenRouterEmbedder (model={OpenRouterEmbedder().model})")
    print(f"Using REAL httpx.Client -> live en.wikivoyage.org")
    print(f"Chroma path: {tmp_path}")

    vector_store = VectorStore(path=str(tmp_path))  # real OpenRouterEmbedder
    tool = AttractionRagTool(vector_store=vector_store)
    # Seed curated docs first (matches normal app startup), then run a
    # NON-curated city to force the external Wikivoyage path.
    tool.seed()

    with httpx.Client(timeout=20.0) as client:
        result = tool.run(city=CITY, interests=INTERESTS, http_client=client)

    print("\n--- RESULT ---")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2)[:4000])
    if result.get("results"):
        print("\n--- RESULTS (attractions) ---")
        for r in result["results"]:
            print(f"- {r.get('name')!r}: {r.get('description', '')[:120]!r}")

    trace = result.get("rag_trace", {}) or {}
    dump_hop("hop_1", trace.get("hop_1", []))
    dump_hop("hop_2", trace.get("hop_2", []))

    print("\n--- EMBEDDER MODE ---")
    print(f"_call_api invocations: {counts['api_calls']}")
    print(f"hash fallback invocations: {counts['fallback_calls']}")
    print(f"api errors captured: {counts['api_errors']}")
    for w in api_warnings:
        print(f"  api error: {w}")
    for w in _warn_records:
        print(f"  warn log: {w}")
    mode = "API" if counts["api_calls"] and not counts["fallback_calls"] else (
        "FALLBACK" if counts["fallback_calls"] else "NONE"
    )
    print(f"=> embedder mode: {mode}")

    # Also probe the Wikivoyage parse API directly to see real section shapes.
    print("\n--- WIKIVOYAGE PROBE (parse sections) ---")
    try:
        with httpx.Client(timeout=20.0) as client:
            r = client.get(
                "https://en.wikivoyage.org/w/api.php",
                params={
                    "action": "parse",
                    "format": "json",
                    "page": CITY.title(),
                    "prop": "sections",
                },
                headers={"User-Agent": "TripAgent/1.0 (educational project)"},
            )
            data = r.json()
            secs = data.get("parse", {}).get("sections", [])
            print(f"HTTP {r.status_code}, {len(secs)} sections")
            see_do = [s for s in secs if s.get("line", "").strip().lower() in ("see", "do")]
            print(f"See/Do sections matched by line==lower: {see_do}")
            print("First 8 sections:")
            for s in secs[:8]:
                print(f"  index={s.get('index')!r} line={s.get('line')!r} level={s.get('level')!r}")
            if not see_do and secs:
                print("ALL section lines:", [s.get("line") for s in secs])
    except Exception as e:  # noqa: BLE001
        print(f"Wikivoyage sections probe failed: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
