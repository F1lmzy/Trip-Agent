# Implementation Plan: Intelligent Travel Planning AI Agent

## Overview

Build the project in vertical slices so each phase leaves a runnable FastAPI application. The implementation starts with API contracts and a health endpoint, then adds parsing/planning, memory, tools, RAG, OpenRouter generation, frontend, tests, and deployment documentation. The goal is to satisfy the assessment requirements without overbuilding.

## Architecture Decisions

- **FastAPI as the application boundary:** exposes `/health`, `/chat`, `/memory/{user_id}`, and `/` for the minimal HTML frontend.
- **Custom planner instead of LangChain:** easier to explain, test, and debug for assessment. The planner returns visible sub-tasks and routes tools dynamically.
- **ChromaDB + sentence-transformers:** used both for attraction RAG and long-term preference memory.
- **OpenRouter for final response generation:** default model is `nvidia/nemotron-3-ultra`, with graceful fallback if the API is unavailable.
- **OpenWeatherMap for real weather:** external calls are isolated in `weather_tool.py` and mocked in tests.
- **Brave Search tool as optional dynamic enrichment:** called only for current/recent information or weak local RAG coverage.
- **Minimal static frontend:** plain HTML/CSS/JS served from FastAPI, no frontend build step.

## Dependency Graph

```text
Project config + schemas
    │
    ├── FastAPI app routes
    │       ├── Health endpoint
    │       ├── Chat endpoint
    │       ├── Memory endpoints
    │       └── Static frontend
    │
    ├── Parser + planner
    │       └── Orchestrator
    │             ├── Short-term memory
    │             ├── Long-term ChromaDB memory
    │             ├── Attraction RAG tool
    │             ├── Weather tool
    │             ├── Web search tool
    │             ├── Budget tool
    │             ├── Hotel tool
    │             └── OpenRouter response generator
    │
    └── Tests + docs + deployment files
```

## Task List

### Phase 1: Foundation

## Task 1: Create project scaffold, dependencies, and configuration

**Description:** Create the initial Python/FastAPI project layout, dependency files, environment examples, and settings loader.

**Acceptance criteria:**
- [ ] Project directories match the spec.
- [ ] `requirements.txt` includes FastAPI, Uvicorn, Pydantic, requests/httpx, ChromaDB, sentence-transformers, pytest, and related test dependencies.
- [ ] `.env.example` documents `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENWEATHER_API_KEY`, `BRAVE_SEARCH_API_KEY`, and `CHROMA_PATH`.

**Verification:**
- [ ] Run: `python -m compileall app`
- [ ] Run: `pip install -r requirements.txt`

**Dependencies:** None

**Files likely touched:**
- `requirements.txt`
- `.env.example`
- `app/config.py`
- package `__init__.py` files

**Estimated scope:** Medium: 3-5 files

---

## Task 2: Add API schemas and basic FastAPI app

**Description:** Define request/response models and implement `/health`, `/chat` placeholder behavior, memory endpoint placeholders, and static index serving placeholder.

**Acceptance criteria:**
- [ ] `GET /health` returns `{ "status": "ok" }`.
- [ ] `POST /chat` accepts `user_id` and `message` and returns a valid `ChatResponse` shape.
- [ ] `GET /` returns HTML content.

**Verification:**
- [ ] Run: `uvicorn app.main:app --reload`
- [ ] Manual check: open `http://127.0.0.1:8000/docs`
- [ ] Run endpoint smoke tests after tests are added.

**Dependencies:** Task 1

**Files likely touched:**
- `app/main.py`
- `app/schemas.py`
- `app/static/index.html`
- `tests/test_api.py`

**Estimated scope:** Medium: 3-5 files

### Checkpoint: Foundation

- [ ] App starts with `uvicorn app.main:app --reload`.
- [ ] `/health` works.
- [ ] Swagger UI loads.
- [ ] No secrets are committed.

---

### Phase 2: Core Agent Flow

## Task 3: Implement parser and planner

**Description:** Implement deterministic parsing for city, duration, interests, budget, hotel intent, dates, and follow-up hints. Add a custom planner that returns visible sub-tasks and dynamic tool routing decisions.

**Acceptance criteria:**
- [ ] Parser extracts common cities, interests, budget levels, and duration from representative prompts.
- [ ] Missing city produces a clarification path.
- [ ] Planner returns a `plan` array and selected tool names based on request context.
- [ ] Hotel tool is selected only when lodging is requested.
- [ ] Web search tool is selected for current/recent/event/closure requests or weak RAG fallback flags.

**Verification:**
- [ ] Run: `pytest -q tests/test_parser.py tests/test_planner.py`

**Dependencies:** Task 2

**Files likely touched:**
- `app/agent/parser.py`
- `app/agent/planner.py`
- `tests/test_parser.py`
- `tests/test_planner.py`

**Estimated scope:** Medium: 4 files

---

## Task 4: Implement short-term memory and basic orchestration

**Description:** Add per-user in-process conversation history and wire parser/planner into the chat endpoint through an orchestrator. This creates the first usable chat flow with clarification behavior.

**Acceptance criteria:**
- [ ] Conversation history is stored per `user_id`.
- [ ] Follow-up messages can access recent context.
- [ ] Missing-city requests return a clarifying question without calling planning tools.
- [ ] `/chat` response includes `plan`, `tools_used`, `needs_clarification`, and `clarifying_question`.

**Verification:**
- [ ] Run: `pytest -q tests/test_agent.py tests/test_memory.py`
- [ ] Manual check: send “Plan me a trip” and verify clarification.

**Dependencies:** Task 3

**Files likely touched:**
- `app/agent/orchestrator.py`
- `app/memory/short_term.py`
- `app/main.py`
- `tests/test_agent.py`
- `tests/test_memory.py`

**Estimated scope:** Medium: 5 files

### Checkpoint: Basic Agent

- [ ] `/chat` handles normal and vague requests.
- [ ] Planner output is visible in API responses.
- [ ] Short-term memory works for one running process.

---

### Phase 3: ChromaDB Memory and RAG

## Task 5: Implement ChromaDB vector store wrapper

**Description:** Add a small wrapper around ChromaDB collections using sentence-transformers embeddings, configurable persistent path, and test-safe temporary storage.

**Acceptance criteria:**
- [ ] Can create/get named collections.
- [ ] Can add documents with metadata.
- [ ] Can query similar documents.
- [ ] Tests can use a temporary Chroma path without touching production data.

**Verification:**
- [ ] Run: `pytest -q tests/test_vector_store.py`

**Dependencies:** Task 1

**Files likely touched:**
- `app/memory/vector_store.py`
- `tests/test_vector_store.py`

**Estimated scope:** Small: 2 files

---

## Task 6: Implement long-term user preference memory

**Description:** Store and retrieve stable user preferences in ChromaDB with `user_id` metadata. Add memory endpoints for get/add/reset.

**Acceptance criteria:**
- [ ] `POST /memory/{user_id}` stores a preference.
- [ ] `GET /memory/{user_id}` returns that user’s preferences only.
- [ ] `DELETE /memory/{user_id}` clears that user’s preferences.
- [ ] Orchestrator retrieves relevant memory during `/chat`.

**Verification:**
- [ ] Run: `pytest -q tests/test_memory.py tests/test_api.py`
- [ ] Manual check: add a preference, then ask for a new trip and verify `memory_used` includes it.

**Dependencies:** Task 5

**Files likely touched:**
- `app/memory/long_term.py`
- `app/main.py`
- `app/agent/orchestrator.py`
- `tests/test_memory.py`
- `tests/test_api.py`

**Estimated scope:** Medium: 5 files

---

## Task 7: Seed attraction data and implement multi-hop RAG tool

**Description:** Add curated city/attraction documents and implement `attraction_rag_tool` with two retrieval hops: city overview then interest-specific retrieval.

**Acceptance criteria:**
- [ ] Seed data includes at least Tokyo, Singapore, Paris, and New York or similar major demo cities.
- [ ] Hop 1 retrieves broad city context.
- [ ] Hop 2 uses user interests plus hop 1 context to retrieve personalized matches.
- [ ] Tool returns `rag_trace` with `hop_1` and `hop_2` summaries.

**Verification:**
- [ ] Run: `pytest -q tests/test_rag.py`
- [ ] Manual check: Tokyo anime/food/photography returns relevant matches like Akihabara, food markets, or photo viewpoints.

**Dependencies:** Task 5

**Files likely touched:**
- `app/tools/attraction_rag_tool.py`
- `app/data/attractions.json`
- `app/data/city_docs/*.md`
- `tests/test_rag.py`

**Estimated scope:** Medium: 4-5 files

### Checkpoint: Memory + RAG

- [ ] ChromaDB works locally.
- [ ] Long-term memory persists preferences.
- [ ] Multi-hop RAG trace appears in chat responses.

---

### Phase 4: External and Rule-Based Tools

## Task 8: Implement budget and hotel tools

**Description:** Add deterministic budget guidance and optional mock hotel lookup from local JSON data.

**Acceptance criteria:**
- [ ] Budget tool returns low/medium/luxury guidance.
- [ ] Missing budget defaults to medium and marks it as assumed.
- [ ] Hotel tool returns city/budget-matched mock hotels only when requested.

**Verification:**
- [ ] Run: `pytest -q tests/test_tools.py`

**Dependencies:** Task 3

**Files likely touched:**
- `app/tools/budget_tool.py`
- `app/tools/hotel_tool.py`
- `app/data/hotels.json`
- `tests/test_tools.py`

**Estimated scope:** Medium: 4 files

---

## Task 9: Implement OpenWeatherMap weather tool

**Description:** Add real weather integration through OpenWeatherMap, with graceful fallback when the API key is absent or the API fails. Tests must mock network calls.

**Acceptance criteria:**
- [ ] Tool calls OpenWeatherMap when `OPENWEATHER_API_KEY` is configured.
- [ ] Tool returns normalized day-level weather summaries.
- [ ] Missing API key returns a fallback status instead of crashing.
- [ ] Tests do not make real network calls.

**Verification:**
- [ ] Run: `pytest -q tests/test_tools.py`
- [ ] Manual check with a real `.env`: ask for Tokyo and verify weather appears in response/tool output.

**Dependencies:** Task 1

**Files likely touched:**
- `app/tools/weather_tool.py`
- `app/config.py`
- `tests/test_tools.py`

**Estimated scope:** Small: 3 files

---

## Task 10: Implement Brave Search web search tool

**Description:** Add a Brave Search API wrapper for fresh travel context. Keep it isolated with a safe fallback when no API key is available.

**Acceptance criteria:**
- [ ] Tool accepts city and query intent, returning compact Brave Search result summaries.
- [ ] Tool reads `BRAVE_SEARCH_API_KEY` from environment/config.
- [ ] Tool is called only when planner selects it.
- [ ] Missing API credentials or Brave API failure returns graceful fallback.
- [ ] Tests mock Brave Search and verify routing behavior.

**Verification:**
- [ ] Run: `pytest -q tests/test_tools.py tests/test_planner.py`

**Dependencies:** Task 3

**Files likely touched:**
- `app/tools/web_search_tool.py`
- `app/config.py`
- `app/agent/orchestrator.py`
- `tests/test_tools.py`

**Estimated scope:** Medium: 4 files

### Checkpoint: Tools

- [ ] Weather, attraction RAG, web search, budget, and hotel tools return structured outputs.
- [ ] Dynamic tool routing is test-covered.
- [ ] External API failures do not crash `/chat`.

---

### Phase 5: OpenRouter Generation and End-to-End Itinerary

## Task 11: Implement OpenRouter client and response generator

**Description:** Add OpenRouter chat completion integration using `nvidia/nemotron-3-ultra` by default. Generate final responses from parsed request, memory, plan, tool outputs, and RAG trace.

**Acceptance criteria:**
- [ ] OpenRouter API key/model are read from environment.
- [ ] Response generator asks for structured JSON-like itinerary with day/morning/afternoon/evening sections.
- [ ] If OpenRouter fails, fallback generator still returns a usable itinerary.
- [ ] Tests mock OpenRouter responses.

**Verification:**
- [ ] Run: `pytest -q tests/test_agent.py`
- [ ] Manual check with `OPENROUTER_API_KEY`: request a Tokyo itinerary and verify natural language response.

**Dependencies:** Tasks 4, 7, 8, 9, 10

**Files likely touched:**
- `app/agent/openrouter_client.py`
- `app/agent/response_generator.py`
- `app/agent/orchestrator.py`
- `tests/test_agent.py`

**Estimated scope:** Medium: 4 files

---

## Task 12: Wire full `/chat` orchestration

**Description:** Connect parser, memory, planner, tools, RAG, OpenRouter, and memory updates into the final chat flow.

**Acceptance criteria:**
- [ ] Normal request calls at least attraction RAG, weather, and budget tools.
- [ ] Current-info requests also call web search.
- [ ] Hotel requests call hotel tool.
- [ ] Response includes `message`, `itinerary`, `memory_used`, `tools_used`, `plan`, `rag_trace`, and clarification fields.
- [ ] Stable preferences are saved to long-term memory.

**Verification:**
- [ ] Run: `pytest -q tests/test_agent.py tests/test_api.py`
- [ ] Manual check: demo flow from spec works in Swagger UI.

**Dependencies:** Task 11

**Files likely touched:**
- `app/agent/orchestrator.py`
- `app/main.py`
- `tests/test_agent.py`
- `tests/test_api.py`

**Estimated scope:** Medium: 4 files

### Checkpoint: End-to-End API

- [ ] `/chat` produces complete 2-day itinerary output.
- [ ] `tools_used` proves dynamic tool selection.
- [ ] `plan` proves planning mechanism.
- [ ] `rag_trace` proves multi-hop RAG.
- [ ] Memory affects later requests.

---

### Phase 6: Frontend, Deployment, and Documentation

## Task 13: Build minimal HTML frontend

**Description:** Replace placeholder frontend with a simple chat page that calls `/chat`, displays responses, tools used, plan, and RAG trace.

**Acceptance criteria:**
- [ ] User can enter `user_id` and message.
- [ ] Frontend sends `POST /chat`.
- [ ] Frontend displays assistant message, itinerary, tools used, plan, and RAG trace.
- [ ] No frontend framework or build step is required.

**Verification:**
- [ ] Run: `uvicorn app.main:app --reload`
- [ ] Manual check: open `http://127.0.0.1:8000/` and complete one chat request.

**Dependencies:** Task 12

**Files likely touched:**
- `app/static/index.html`
- `app/main.py`

**Estimated scope:** Small: 2 files

---

## Task 14: Add deployment files and README

**Description:** Add Dockerfile, Render instructions, local setup instructions, and demo script.

**Acceptance criteria:**
- [ ] Dockerfile runs the FastAPI app.
- [ ] README documents local setup, env vars, tests, and deployment.
- [ ] `.env.example` is complete.
- [ ] Render start command is documented.

**Verification:**
- [ ] Run: `docker build -t travel-agent .`
- [ ] Run: `docker run -p 8000:8000 --env-file .env travel-agent`
- [ ] Manual check: `/health` works in Docker.

**Dependencies:** Task 12

**Files likely touched:**
- `Dockerfile`
- `README.md`
- `.env.example`

**Estimated scope:** Small: 3 files

---

## Task 15: Write technical documentation and final assessment mapping

**Description:** Create `docs/technical_doc.md` explaining architecture, tools, memory, planning, multi-hop RAG, API, deployment, limitations, and demo flow.

**Acceptance criteria:**
- [ ] Technical doc maps every assessment requirement to implementation evidence.
- [ ] Includes architecture diagram or text architecture.
- [ ] Includes demo flow for video/interview.
- [ ] Includes known limitations and future improvements.

**Verification:**
- [ ] Manual review against `docs/spec.md` success criteria.

**Dependencies:** Task 12

**Files likely touched:**
- `docs/technical_doc.md`

**Estimated scope:** Small: 1 file

### Final Checkpoint: Complete

- [ ] `pytest -q` passes.
- [ ] `uvicorn app.main:app --reload` runs locally.
- [ ] `/health`, `/`, `/chat`, and memory endpoints work.
- [ ] Normal trip request demonstrates at least 3 tools.
- [ ] Current-info trip request demonstrates web search tool.
- [ ] Vague request demonstrates clarification.
- [ ] Follow-up request demonstrates short-term memory.
- [ ] Preference reuse demonstrates long-term ChromaDB memory.
- [ ] Technical doc is complete.
- [ ] App is ready for Render deployment.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---:|---|
| ChromaDB/sentence-transformers install is heavy on Render free tier | Medium | Keep dependencies minimal; document expected cold start; use small embedding model such as `all-MiniLM-L6-v2`. |
| OpenRouter model name/provider availability changes | Medium | Make model configurable via `OPENROUTER_MODEL`; include fallback response generator. |
| OpenWeatherMap free tier/date forecast limitations | Low | Use current/near-term forecast and document limitation. |
| Brave Search API key may be missing or rate-limited | Medium | Isolate behind `web_search_tool`; support graceful fallback and mocked tests. |
| LLM output may be unstructured | Medium | Prompt for JSON-like structure, validate lightly, and fallback to deterministic itinerary format. |
| Parser may miss unusual city names | Medium | Start with regex/rule extraction; fallback to LLM generalization in response generator if needed. |
| Long-term memory may store one-off details | Medium | Implement stable-preference filters and avoid storing dates/hotel names. |

## Parallelization Opportunities

Safe to parallelize after Task 2:

- Parser/planner tests and implementation.
- Budget/hotel tools.
- Weather tool.
- Frontend mockup.
- Documentation drafts.

Must be sequential:

- Vector store before long-term memory and RAG.
- Tool implementations before full orchestration.
- Full orchestration before final frontend/demo documentation.

Needs coordination:

- API schema changes, because frontend, tests, orchestrator, and docs depend on response fields.

## Open Questions

None currently. Web search provider is confirmed as Brave Search API using `BRAVE_SEARCH_API_KEY`.
