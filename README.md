# Intelligent Travel Planning AI Agent

Trip-Agent is a FastAPI travel-planning agent for personalized city itineraries. It combines deterministic parsing and planning, multi-agent routing, tool execution, short-term conversation memory, long-term ChromaDB preference memory, multi-hop RAG, OpenRouter generation, MCP tool exposure, Server-Sent Events streaming, and a small browser UI.

The default itinerary length is 2 days, but prompts can request other durations, hotels, flights, current information, dietary needs, accessibility constraints, pet-friendly options, nightlife, and budget levels.

## Current capabilities

- Minimal HTML frontend at `/`
- REST chat API at `/chat`
- streaming chat API at `/chat/stream` via Server-Sent Events
- browser-friendly `GET /chat/stream?user_id=...&message=...` EventSource endpoint
- `/api/tools` endpoint listing the registered tool metadata
- `/health` endpoint with tool count and core configuration status
- memory endpoints for long-term preferences:
  - `GET /memory/{user_id}`
  - `POST /memory/{user_id}`
  - `DELETE /memory/{user_id}`
- MCP server mounted at `/mcp`
- standalone MCP runner with `python -m app.mcp_server`
- deterministic parser and planner with visible `plan` steps in every response
- multi-agent routing:
  - itinerary planning when a destination is known
  - destination recommendations when the user has not chosen a city
  - customer-query / clarification behavior for underspecified requests
  - follow-up handling from short-term memory
- OpenRouter itinerary generation with a bounded thread pool and deterministic fallback when unavailable
- ChromaDB-backed long-term memory for user preferences across sessions
- ChromaDB-backed multi-hop RAG for attractions and city context
- SerpAPI Google Flights and Google Hotels integrations when `SERPAPI_API_KEY` is configured
- DuckDuckGo web search through LangChain Community / `ddgs` for current travel context
- Wikimedia Commons image lookup for attractions and hotels
- Dockerfile and Render deployment configuration
- `llms.txt` for LLM-friendly project context

## Tooling overview

The runtime planner currently uses the core planning tool names below:

| Planner tool | Purpose |
|---|---|
| `attraction_rag_tool` | Multi-hop attraction and city-context retrieval from ChromaDB, curated data, and external content ingestion |
| `weather_tool` | Near-term OpenWeatherMap forecast with graceful fallback |
| `budget_tool` | Low, medium, and luxury budget guidance |
| `web_search_tool` | Fresh web context for current events, latest food spots, closures, and weak RAG cases |
| `hotel_tool` | Hotel planning path; uses SerpAPI hotels when configured and local fallback data otherwise |
| `flight_tool` | Flight planning path; uses SerpAPI flights when configured and local fallback only when no SerpAPI key is configured |

The `/api/tools` registry exposes 8 implemented tool capabilities:

| Registry id | Description |
|---|---|
| `attraction_rag_tool` | Multi-hop ChromaDB attraction retrieval |
| `weather_tool` | OpenWeatherMap forecast |
| `budget_tool` | Budget guidance |
| `serpapi_hotel_tool` | Real-time Google Hotels search via SerpAPI |
| `serpapi_flight_tool` | Real-time Google Flights search via SerpAPI |
| `web_search_tool` | Fresh travel web search |
| `destination_search_tool` | Destination discovery search |
| `wikimedia_image_tool` | Wikimedia Commons image resolution |

## MCP server

All implemented tool capabilities are exposed through MCP at `/mcp` using FastMCP streamable HTTP.

MCP tool names:

| MCP tool | Backing capability |
|---|---|
| `search_attractions` | Multi-hop attraction RAG |
| `get_weather` | OpenWeatherMap weather forecast |
| `apply_budget` | Budget guidance |
| `search_hotels` | SerpAPI Google Hotels directly |
| `search_flights` | SerpAPI Google Flights directly |
| `web_search` | DuckDuckGo web search |
| `search_destinations` | DuckDuckGo destination discovery |
| `lookup_place_image` | Wikimedia Commons image lookup |

Run the MCP server standalone:

```bash
python -m app.mcp_server
```

Debug with MCP Inspector:

```bash
mcp dev app/mcp_server.py
```

Connect MCP-compatible clients to:

```text
http://127.0.0.1:8000/mcp
```

## Architecture

```text
FastAPI app (`app/main.py`)
  ├── Static frontend (`app/static/index.html`)
  ├── REST + SSE API routes
  ├── Memory routes
  ├── Tool metadata route (`/api/tools`)
  ├── MCP server mount (`app/mcp_server.py`) — 8 MCP tools at /mcp
  ├── Schemas (`app/schemas.py`)
  ├── Multi-agent layer (`app/agents/*`)
  │   ├── Supervisor routing
  │   ├── ItineraryAgent
  │   ├── DestinationRecommendationAgent
  │   └── CustomerQueryAgent
  ├── Agent orchestration (`app/agent/*`)
  │   ├── Parser (`parser.py`)
  │   ├── Planner (`planner.py`)
  │   ├── Tool executor (`tool_executor.py`)
  │   ├── Response builders (`response_builders.py`)
  │   ├── OpenRouter client + response generator
  │   └── Streaming SSE generator (`streaming.py`)
  ├── Memory (`app/memory/*`)
  │   ├── Short-term in-process conversation memory
  │   ├── Long-term preference memory
  │   ├── ChromaDB vector store
  │   └── OpenRouter embeddings with deterministic fallback
  ├── Tools (`app/tools/*`)
  │   ├── Attraction RAG and external content ingestion
  │   ├── Weather, budget, hotels, flights, search, images
  │   └── Tool registry metadata
  └── Seed data (`app/data/*`)
```

## Response shape

`POST /chat` returns a `ChatResponse` like:

```json
{
  "message": "Markdown itinerary or fallback explanation",
  "itinerary": {
    "status": "generated_with_openrouter",
    "city": "Osaka",
    "duration_days": 3,
    "day_1": {
      "morning": "...",
      "afternoon": "...",
      "evening": "..."
    },
    "flights": {
      "status": "ok",
      "departure_flights": [],
      "return_flights": []
    }
  },
  "memory_used": [],
  "tools_used": ["attraction_rag_tool", "weather_tool", "budget_tool"],
  "plan": ["Parse destination, dates, preferences, and constraints", "..."],
  "rag_trace": {"hop_1": [], "hop_2": []},
  "needs_clarification": false,
  "clarifying_question": null
}
```

When required context is missing, such as destination or budget, the planner can return `needs_clarification: true` before running tools.

## Multi-hop RAG and external content ingestion

`attraction_rag_tool` performs two-hop retrieval over ChromaDB:

1. **Hop 1:** retrieve broad city overview context.
2. **Hop 2:** retrieve interest-specific attraction context using hop-1 context to expand the query.

For curated cities, the tool uses local data in `app/data/`.

For other cities, the tool can fetch travel content dynamically:

1. Fetch the city page from Wikivoyage.
2. Fall back to Wikipedia when needed.
3. Parse travel sections and `{{see}}` / `{{do}}` listing templates.
4. Strip wiki and external-link markup.
5. Chunk and embed content into ChromaDB.
6. Retry retrieval over the newly-ingested city content.

External content is cached in ChromaDB, so repeated requests do not need to re-fetch the same city content.

## Flights and hotels

### Flights

When `SERPAPI_API_KEY` is configured, flight requests use SerpAPI Google Flights.

Current flight behavior:

- accepts cities or airport IATA codes, e.g. `Singapore`, `Osaka`, `SIN`, `KIX`
- uses `airportsdata` plus curated overrides to resolve cities to SerpAPI-compatible airport IATA codes
- avoids unsupported broad city codes such as `NYC`, `LON`, `TYO`, and `CHI`
- uses `type=2` for one-way flights
- uses `type=1` and `return_date` for round trips
- parses SerpAPI `best_flights` and `other_flights`
- supports SerpAPI's `departure_token` second-step lookup for return-flight options
- if a return-token lookup fails, keeps the valid outbound flights instead of replacing them with mock data
- when a SerpAPI key is configured, SerpAPI errors/no-results are surfaced instead of hidden behind fake mock flights
- when no SerpAPI key is configured, the local mock flight tool keeps demos functional

### Hotels

When `SERPAPI_API_KEY` is configured, hotel requests use SerpAPI Google Hotels with live pricing, ratings, links, and images when available. If SerpAPI is unavailable or no key is configured, the local hotel fallback keeps the itinerary usable.

## Memory behavior

### Short-term memory

`ShortTermMemory` stores recent in-process conversation context per user. This enables follow-ups such as:

```text
Make it cheaper.
Add more nightlife.
```

### Long-term memory

Long-term preferences are stored in ChromaDB and retrieved by user id. Examples:

```bash
curl -X POST http://127.0.0.1:8000/memory/kavin \
  -H 'Content-Type: application/json' \
  -d '{"preference":"I prefer vegetarian food"}'
```

Then ask:

```text
Plan a 2-day trip to Singapore with food. Medium budget.
```

The response can include saved preferences in `memory_used`.

Budget preferences are treated as mutually exclusive: when a new current budget is parsed, stale retrieved budget memories are filtered out and conflicting stored budget preferences can be deleted before saving the new one.

## Requirements

- Python 3.11 recommended
- Optional API keys:
  - OpenRouter for LLM itinerary generation and embeddings
  - OpenWeatherMap for live weather
  - SerpAPI for live Google Flights and Google Hotels
- DuckDuckGo search does not require an API key, but can still fail because of network or rate limits.

The app still runs without API keys. Missing external credentials produce graceful fallback responses.

## Environment variables

Copy `.env.example` to `.env` for local development.

| Variable | Purpose | Required to run? |
|---|---|---:|
| `OPENROUTER_API_KEY` | OpenRouter LLM generation and embeddings | No |
| `OPENROUTER_MODEL` | OpenRouter chat model | No, defaults to `nvidia/nemotron-3-ultra` |
| `OPENROUTER_TIMEOUT_SECONDS` | Max OpenRouter wait before fallback | No, defaults to `120` |
| `OPENROUTER_EMBEDDING_MODEL` | OpenRouter embedding model for RAG and memory | No, defaults to `nvidia/llama-nemotron-embed-vl-1b-v2:free` |
| `OPENWEATHER_API_KEY` | OpenWeatherMap weather tool | No |
| `SERPAPI_API_KEY` | SerpAPI Google Flights and Google Hotels | No |
| `CHROMA_PATH` | ChromaDB persistence path | No, defaults to `./chroma_db` |

## Local setup

Using `venv` and `pip`:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

If you use `uv`, the test commands in this repository also work with:

```bash
uv run python3 -m pytest -q
```

Edit `.env` and add API keys if available.

## Run locally

```bash
. .venv/bin/activate
uvicorn app.main:app --reload
```

Open:

- Frontend: <http://127.0.0.1:8000/>
- Swagger UI: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>
- MCP endpoint: <http://127.0.0.1:8000/mcp>

## API quick examples

Health:

```bash
curl http://127.0.0.1:8000/health
```

List tools:

```bash
curl http://127.0.0.1:8000/api/tools
```

Chat:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"kavin","message":"Plan a 2-day trip to Tokyo with anime and food. Medium budget."}'
```

Streaming chat:

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"kavin","message":"Plan a 3-day trip to Osaka from Singapore. I like local bars. Luxury budget. With flights."}'
```

Browser EventSource-style streaming:

```bash
curl -N "http://127.0.0.1:8000/chat/stream?user_id=kavin&message=Plan%20a%202-day%20trip%20to%20Paris.%20Medium%20budget."
```

Memory:

```bash
curl -X POST http://127.0.0.1:8000/memory/kavin \
  -H 'Content-Type: application/json' \
  -d '{"preference":"I prefer halal restaurants and museums"}'

curl http://127.0.0.1:8000/memory/kavin

curl -X DELETE http://127.0.0.1:8000/memory/kavin
```

## Example prompts

Normal itinerary:

```text
Plan a 2-day trip to Tokyo. I like anime, food, and photography. Medium budget.
```

Flight request:

```text
Plan a 3-day trip to Osaka from Singapore. I like bars, luxury budget, with return flights.
```

Hotel request:

```text
Plan a Paris trip with museums and suggest hotels. Luxury budget.
```

Current-info request:

```text
Plan a Singapore trip with current events, latest food spots, and vegetarian options. Medium budget.
```

Destination discovery:

```text
Plan me a beach trip in Asia. Medium budget.
```

Clarification flow:

```text
Plan a trip to Edinburgh.
```

Expected: because budget is missing, `needs_clarification` should be `true` and the assistant should ask for low, medium, or luxury budget.

Follow-up flow:

```text
Make it cheaper.
```

Expected: with prior conversation history, the agent uses short-term memory and budget tooling.

## Tests

Tests mock external APIs and use temporary ChromaDB paths.

```bash
. .venv/bin/activate
pytest -q
```

Current verified suite size after the latest update:

```text
264 passed
```

The standard verification command used during development is:

```bash
uv run python3 -m pytest --tb=short -q
uv run python3 -m compileall app tests
```

## Docker

Build:

```bash
docker build -t travel-agent .
```

Run with local env vars:

```bash
docker run -p 8000:8000 --env-file .env travel-agent
```

Then open <http://127.0.0.1:8000/>.

## Render deployment

### Option 1: Render Python web service

Use these settings:

```text
Environment: Python
Build command: pip install -r requirements.txt
Start command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Add environment variables in Render:

```text
PYTHON_VERSION=3.11.11
OPENROUTER_API_KEY=<your OpenRouter key>
OPENROUTER_MODEL=nvidia/nemotron-3-ultra
OPENROUTER_EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free
OPENROUTER_TIMEOUT_SECONDS=120
OPENWEATHER_API_KEY=<your OpenWeatherMap key>
SERPAPI_API_KEY=<your SerpAPI key>
CHROMA_PATH=/opt/render/project/src/chroma_db
```

### Option 2: Render Blueprint

This repo includes `render.yaml`. In Render, create a new Blueprint from the repository and set the secret environment variables when prompted.

### ChromaDB persistence on Render

The default Render filesystem can be ephemeral. For a demo, this is acceptable: attraction RAG is re-seeded by the app and user memory can be recreated.

For persistent long-term user memory, add a Render persistent disk and set:

```text
CHROMA_PATH=/var/data/chroma_db
```

Mount the disk at:

```text
/var/data
```

## Demo video

https://github.com/user-attachments/assets/6b0498d7-7e9d-4b6d-9266-6d2a58ff708e

## Known limitations

- OpenWeatherMap forecast is near-term, not a full future travel-date forecast.
- Render's default filesystem is ephemeral unless a persistent disk is attached.
- DuckDuckGo/DDGS can fail due to DNS, network, or rate limiting; the app logs failures and degrades gracefully.
- SerpAPI flights are live third-party data. Some routes or departure-token return lookups can fail; when that happens, the app preserves valid outbound results and surfaces structured error/no-result details rather than inventing fake live flights.
- SerpAPI hotels and flights require a valid `SERPAPI_API_KEY` for live data.
- OpenRouter failures, empty responses, timeouts, or missing API keys trigger deterministic fallback itinerary generation.
- Without `OPENROUTER_API_KEY`, embeddings fall back to deterministic hash-based vectors, which keeps the app functional but degrades retrieval quality.
- The parser is deterministic and can still miss unusual phrasing or ambiguous locations.
- Long-term memory is vector-search based, so very old or loosely related preferences may need deletion through the memory API if they are no longer desired.
- `chromadb` adds dependency weight, but the app avoids `sentence-transformers` / PyTorch by using OpenRouter embeddings when configured.

## Project structure

```text
app/
  agent/          parser, planner, orchestrator, tool execution, streaming, OpenRouter response generation
  agents/         supervisor, itinerary, destination recommendation, customer query agents
  data/           curated city, attraction, and hotel data
  memory/         short-term memory, long-term memory, ChromaDB vector store, embedding adapter
  static/         minimal HTML frontend
  tools/          RAG, external content, weather, budget, SerpAPI hotels/flights, web search, images, registry
  mcp_server.py   FastMCP server exposing 8 tools at /mcp
docs/             spec and implementation plan
scripts/          local reproduction/debug scripts
tests/            unit, API, MCP, streaming, RAG, and tool tests
Dockerfile        container build
render.yaml       Render blueprint
llms.txt          LLM-friendly project information
```
