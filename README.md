# Intelligent Travel Planning AI Agent

A FastAPI travel-planning agent that creates personalized 2-day itineraries with visible planning steps, tool use, short-term memory, long-term ChromaDB memory, and multi-hop RAG.

## Features

- Minimal HTML frontend at `/`
- Chat API at `/chat`
- Custom parser and planner with visible reasoning steps
- Tool use:
  - `attraction_rag_tool` for multi-hop city and attraction retrieval
  - `weather_tool` using OpenWeatherMap
  - `budget_tool` for low/medium/luxury guidance
  - `web_search_tool` using Brave Search for current or recent context
  - `hotel_tool` for lodging requests
- Short-term in-process conversation memory
- Long-term ChromaDB user preference memory
- OpenRouter response generation with deterministic fallback when unavailable

## Architecture

```text
FastAPI app (`app/main.py`)
  ├── Static frontend (`app/static/index.html`)
  ├── Schemas (`app/schemas.py`)
  ├── Agent orchestration (`app/agent/orchestrator.py`)
  │   ├── Parser (`app/agent/parser.py`)
  │   ├── Planner (`app/agent/planner.py`)
  │   ├── OpenRouter client + response generator
  │   ├── Short-term memory
  │   ├── Long-term ChromaDB memory
  │   └── Tools (`app/tools/*`)
  └── Seed data (`app/data/*`)
```

## Requirements

- Python 3.11 recommended
- Optional API keys for best results:
  - OpenRouter
  - OpenWeatherMap
  - Brave Search

The app still runs without API keys. Missing external credentials produce graceful fallback responses.

## Environment variables

Copy `.env.example` to `.env` for local development.

| Variable | Purpose | Required to run? |
|---|---|---:|
| `OPENROUTER_API_KEY` | OpenRouter LLM generation | No, fallback itinerary is used |
| `OPENROUTER_MODEL` | OpenRouter model name | No, defaults to `nvidia/nemotron-3-ultra` |
| `OPENWEATHER_API_KEY` | OpenWeatherMap forecast tool | No, weather fallback is used |
| `BRAVE_SEARCH_API_KEY` | Brave Search web search tool | No, search fallback is used |
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
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"kavin","message":"Plan a 2-day trip to Tokyo with anime and food. Medium budget."}'
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
BRAVE_SEARCH_API_KEY=<your Brave Search key>
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
- Missing API keys produce fallback output, which is useful for demos but less rich than live API output.
- `sentence-transformers` and `chromadb` make installs and cold starts heavier than a basic FastAPI app.

## Project structure

```text
app/
  agent/       parser, planner, orchestrator, OpenRouter response generation
  data/        curated city and hotel/attraction data
  memory/      short-term memory, ChromaDB vector store, long-term memory
  static/      minimal HTML frontend
  tools/       RAG, weather, budget, hotel, and web search tools
docs/          spec and implementation plan
tests/         unit and API tests
```
