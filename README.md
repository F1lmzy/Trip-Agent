# Intelligent Travel Planning AI Agent

A FastAPI travel-planning agent that creates personalized 2-day itineraries with visible planning steps, tool use, short-term memory, long-term ChromaDB memory, multi-hop RAG, MCP-exposed tools, and SSE streaming.

## Features

- Minimal HTML frontend at `/`
- Chat API at `/chat` and streaming chat at `/chat/stream`
- All tools exposed as an MCP (Model Context Protocol) server at `/mcp`
- Custom parser and planner with visible reasoning steps
- Tool use:
  - `attraction_rag_tool` for multi-hop city and attraction retrieval
  - `weather_tool` using OpenWeatherMap
  - `budget_tool` for low/medium/luxury guidance
  - `web_search_tool` using LangChain's DuckDuckGo search integration for current or recent context
  - `hotel_tool` for lodging requests
  - `flight_tool` for mock flight suggestions between two locations and dates
- Short-term in-process conversation memory
- Long-term ChromaDB user preference memory
- OpenRouter response generation with deterministic fallback when unavailable
- `/api/tools` endpoint listing all available tools
- Enriched `/health` endpoint with tool count and configuration status
- `llms.txt` for LLM-friendly project information

## Architecture

```text
FastAPI app (`app/main.py`)
  ├── Static frontend (`app/static/index.html`)
  ├── Schemas (`app/schemas.py`)
  ├── MCP server (`app/mcp_server.py`) — exposes 6 tools at /mcp
  ├── Agent orchestration (`app/agent/orchestrator.py`)
  │   ├── Parser (`app/agent/parser.py`)
  │   ├── Planner (`app/agent/planner.py`)
  │   ├── Streaming (`app/agent/streaming.py`) — SSE event generation
  │   ├── OpenRouter client + response generator
  │   ├── Short-term memory
  │   ├── Long-term ChromaDB memory
  │   └── Tools (`app/tools/*`)
  │       └── Registry (`app/tools/registry.py`) — tool metadata
  └── Seed data (`app/data/*`)
```

## Requirements

- Python 3.11 recommended
- Optional API keys for best results:
  - OpenRouter
  - OpenWeatherMap
- DuckDuckGo search through LangChain Community does not require an API key.

The app still runs without API keys. Missing external credentials or search/network failures produce graceful fallback responses.

## Environment variables

Copy `.env.example` to `.env` for local development.

| Variable | Purpose | Required to run? |
|---|---|---:|
| `OPENROUTER_API_KEY` | OpenRouter LLM generation | No, fallback itinerary is used |
| `OPENROUTER_MODEL` | OpenRouter model name | No, defaults to `nvidia/nemotron-3-ultra` |
| `OPENROUTER_TIMEOUT_SECONDS` | Max time to wait for OpenRouter before using fallback | No, defaults to `45` |
| `OPENWEATHER_API_KEY` | OpenWeatherMap forecast tool | No, weather fallback is used |
| `CHROMA_PATH` | Local ChromaDB persistence path | No, defaults to `./chroma_db` |

## Local setup

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add API keys if you have them.

## Run locally

```bash
. .venv/bin/activate
uvicorn app.main:app --reload
```

Open:

- Frontend: <http://127.0.0.1:8000/>
- Swagger UI: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

## Example prompts

### Normal trip

```text
Plan a 2-day trip to Tokyo. I like anime, food, and photography. Medium budget.
```

Expected:

- `tools_used` includes `attraction_rag_tool`, `weather_tool`, and `budget_tool`
- `plan` shows the planning steps
- `rag_trace` includes `hop_1` and `hop_2`

### Flight request

```text
Plan a 2-day trip to Tokyo flying from London. Medium budget.
```

Expected:

- `tools_used` includes `flight_tool`
- `itinerary.notes` includes a flight summary

### Current-info trip

```text
Plan a Singapore trip with current events, latest food spots, and vegetarian options.
```

Expected:

- `tools_used` includes `web_search_tool`

### Hotel request

```text
Plan a Paris trip with museums and suggest hotels. Luxury budget.
```

Expected:

- `tools_used` includes `hotel_tool`

### Clarification flow

```text
Plan me a trip
```

Expected:

- `needs_clarification` is `true`
- The assistant asks which city to visit

### Long-term memory flow

Add a preference:

```bash
curl -X POST http://127.0.0.1:8000/memory/kavin \
  -H 'Content-Type: application/json' \
  -d '{"preference":"I prefer vegetarian food"}'
```

Then ask:

```text
Plan a 2-day trip to Singapore with food.
```

Expected:

- `memory_used` includes the saved preference

## API quick examples

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl http://127.0.0.1:8000/api/tools
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"kavin","message":"Plan a 2-day trip to Tokyo with anime and food. Medium budget."}'
```

### Streaming chat (Server-Sent Events)

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"kavin","message":"Plan a 2-day trip to Tokyo with anime and food."}'
```

The stream emits `plan`, `tool_start`, `tool_end`, `message`, and `result` events as SSE frames, ending with the full `ChatResponse` payload.

## MCP server

All six tools are exposed as an MCP (Model Context Protocol) server at `/mcp` using the streamable-http transport. Any MCP-compatible client (Claude Desktop, Cursor, VS Code, etc.) can connect to `http://127.0.0.1:8000/mcp` and call:

- `search_attractions` — multi-hop RAG attraction retrieval
- `get_weather` — weather forecast
- `apply_budget` — budget guidance
- `suggest_hotels` — hotel suggestions
- `suggest_flights` — mock flight suggestions
- `web_search` — fresh web search context

Run the MCP server standalone:

```bash
python -m app.mcp_server
```

Debug with the MCP Inspector:

```bash
mcp dev app/mcp_server.py
```

## Tests

Tests mock external APIs and use temporary ChromaDB paths.

```bash
. .venv/bin/activate
pytest -q
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
OPENROUTER_API_KEY=<your OpenRouter key>
OPENROUTER_MODEL=nvidia/nemotron-3-ultra
OPENWEATHER_API_KEY=<your OpenWeatherMap key>
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

## Known limitations

- OpenWeatherMap forecast is near-term, not a full future travel-date forecast.
- Local ChromaDB data is not committed and may reset on cloud redeploys without a persistent disk.
- The parser is deterministic and may miss unusual phrasing or uncommon cities.
- Missing API keys produce fallback output for OpenRouter and OpenWeatherMap. DuckDuckGo search does not require an API key, but it can still fail due to network or rate-limit issues.
- `sentence-transformers` and `chromadb` make installs and cold starts heavier than a basic FastAPI app.

## Project structure

```text
app/
  agent/       parser, planner, orchestrator, streaming, OpenRouter response generation
  data/        curated city and hotel/attraction data
  memory/      short-term memory, ChromaDB vector store, long-term memory
  static/      minimal HTML frontend
  tools/       RAG, weather, budget, hotel, flight, web search tools + registry
  mcp_server.py  MCP server exposing all tools at /mcp
docs/          spec and implementation plan
tests/         unit and API tests
llms.txt       LLM-friendly project information
```
