"""Reproduce the bad RAG trace against the REAL ./chroma_db (what the app uses)."""
from __future__ import annotations
import json, sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from app.memory.openrouter_embedder import OpenRouterEmbedder, _fallback_embed
from app.memory.vector_store import VectorStore
from app.tools.attraction_rag_tool import AttractionRagTool

counts = {"api": 0, "fallback": 0}
_orig = OpenRouterEmbedder._call_api
def _w_api(self, t):
    counts["api"] += 1
    return _orig(self, t)
def _w_fb(t):
    counts["fallback"] += 1
    return _fallback_embed(t)
OpenRouterEmbedder._call_api = _w_api
import app.memory.openrouter_embedder as _m
_m._fallback_embed = _w_fb

def dump(label, entries):
    print(f"\n=== {label} ({len(entries)}) ===")
    for i, e in enumerate(entries):
        m = e.get("metadata", {})
        print(f"[{i}] id={e.get('id')!r} src={m.get('source')!r} name={m.get('name')!r} city={m.get('city')!r}")
        print(f"    doc[:160]={(e.get('summary','') or '')[:160]!r}")

for city in sys.argv[1:] or ["Newcastle", "Liverpool"]:
    print(f"\n############## CITY={city!r} against REAL ./chroma_db ##############")
    vs = VectorStore(path="./chroma_db")  # REAL app DB, not isolated
    tool = AttractionRagTool(vector_store=vs)
    counts["api"]=0; counts["fallback"]=0
    with httpx.Client(timeout=20.0) as client:
        result = tool.run(city=city, interests=["music","food","history"], http_client=client)
    print("status:", result["status"])
    for r in result.get("results",[]):
        print(f"  RESULT name={r.get('name')!r} desc[:100]={r.get('description','')[:100]!r}")
    dump("hop_1", (result.get("rag_trace") or {}).get("hop_1",[]))
    dump("hop_2", (result.get("rag_trace") or {}).get("hop_2",[]))
    print(f"  embedder: api={counts['api']} fallback={counts['fallback']}")
