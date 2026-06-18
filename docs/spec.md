# Spec: Intelligent Travel Planning AI Agent

## Objective

Build a small, deployable travel planning AI agent that creates personalized 2-day trip itineraries through a FastAPI chat API and a minimal HTML frontend.

The agent should satisfy the assessment requirements by demonstrating:

- Personalized trip planning
- At least 4 dynamically used tools: weather, attraction RAG, web search, and budget
- Short-term conversation memory
- Long-term memory using ChromaDB
- A visible planning/reasoning mechanism
- Multi-hop RAG for attraction suggestions
- A deployed API/UI suitable for demo

Primary user story:

> As a traveler, I can ask for a 2-day trip plan for a city with interests and budget preferences, and receive a structured weather-aware itinerary with recommendations adapted to my preferences and remembered travel style.

Example request:

```json
{
  "user_id": "kavin",
  "message": "Plan a 2-day trip to Tokyo. I like food, anime, and photography. My budget is moderate."
}
```

Expected behavior:

- If the city is missing, ask a clarifying question.
- If interests are missing, use remembered preferences if available; otherwise ask or use general highlights.
- If budget is missing, use remembered budget if available; otherwise default to medium.
- New request details override long-term memory for the current trip.
- Stable preferences are saved to long-term memory.

## Tech Stack

- Python 3.11+
- FastAPI
- Uvicorn
- Pydantic
- OpenRouter API for LLM generation
- Default OpenRouter model: `nvidia/nemotron-3-ultra`
- OpenWeatherMap API for weather
- ChromaDB for attraction RAG and long-term memory
- sentence-transformers for local embeddings
- pytest for tests
- Minimal HTML/CSS/JavaScript frontend served by FastAPI
- Render for deployment

Environment variables:

```bash
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=nvidia/nemotron-3-ultra
OPENWEATHER_API_KEY=...
CHROMA_PATH=./chroma_db
```

## Commands

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload

# Run tests
pytest -q

# Run tests with coverage if pytest-cov is installed
pytest --cov=app tests/

# Build Docker image
docker build -t travel-agent .

# Run Docker image locally
docker run -p 8000:8000 --env-file .env travel-agent

# Render production start command
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Project Structure

```text
app/
  main.py                      # FastAPI app, API routes, static frontend serving
  schemas.py                   # Pydantic request/response models
  config.py                    # Environment variable loading and app settings

  agent/
    orchestrator.py            # Main agent flow
    parser.py                  # Extract city, days, interests, budget, dates, constraints
    planner.py                 # Custom planning mechanism and tool routing plan
    openrouter_client.py       # OpenRouter API wrapper
    response_generator.py      # LLM-backed final itinerary generation

  tools/
    weather_tool.py            # OpenWeatherMap integration
    attraction_rag_tool.py     # Multi-hop ChromaDB attraction retrieval
    web_search_tool.py         # Fresh web search context for current travel info
    budget_tool.py             # Budget rules and filtering hints
    hotel_tool.py              # Mock hotel recommendations

  memory/
    short_term.py              # Per-user conversation history
    long_term.py               # Stable preference storage/retrieval
    vector_store.py            # ChromaDB collection wrapper

  data/
    attractions.json           # Curated attraction documents
    hotels.json                # Mock hotel inventory
    city_docs/                 # Wikipedia/Wikivoyage/blog-style city documents

  static/
    index.html                 # Minimal frontend

tests/
  test_parser.py
  test_agent.py
  test_memory.py
  test_rag.py
  test_tools.py

docs/
  spec.md
  technical_doc.md

requirements.txt
Dockerfile
README.md
.env.example
```

## Code Style

Use typed, small Python functions with structured Pydantic models at API boundaries.

Example:

```python
from pydantic import BaseModel


class ChatRequest(BaseModel):
    user_id: str
    message: str


async def handle_chat(request: ChatRequest) -> ChatResponse:
    parsed = parse_user_request(request.message)

    if not parsed.city:
        return ChatResponse(
            message="Which city would you like to visit?",
            needs_clarification=True,
            tools_used=[],
            plan=[],
        )

    plan = create_trip_plan(parsed)
    return await orchestrator.create_itinerary(
        user_id=request.user_id,
        parsed_request=parsed,
        plan=plan,
    )
```

Conventions:

- Use snake_case for variables, functions, and filenames.
- Use PascalCase for Pydantic models and classes.
- Keep tools deterministic and structured.
- Each tool returns a dictionary with `tool_name`, `status`, and relevant payload.
- API responses include `tools_used`, `plan`, and `needs_clarification`.
- Do not hardcode API keys or secrets.
- LLM failures should degrade gracefully with a structured fallback response.

## Functional Requirements

### Trip Planning

The system generates a 2-day itinerary by default.

Supported inputs:

| Field | Required | Example |
|---|---:|---|
| City | Yes | Tokyo |
| Duration | No, default 2 | 2 days |
| Interests | No | food, anime, museums |
| Budget | No | low, medium, luxury |
| Travel style | No | relaxed, packed, family-friendly |
| Dates | No | July 15-16 |
| Dietary needs | No | vegetarian |
| Constraints | No | wheelchair-friendly |

### Tools

Minimum required dynamic tools:

1. `weather_tool`
   - Uses OpenWeatherMap.
   - Retrieves forecast for the destination.
   - Helps decide indoor/outdoor balance.

2. `attraction_rag_tool`
   - Uses ChromaDB and sentence-transformers.
   - Performs multi-hop RAG over curated city documents and attraction data.

3. `web_search_tool`
   - Uses LangChain DuckDuckGo search tool to retrieve fresh destination context, travel tips, events, closures, or recent recommendations.
   - Called when the local RAG knowledge base has weak coverage, when the user asks for current information, or when the itinerary would benefit from up-to-date external context.

4. `budget_tool`
   - Applies low/medium/luxury budget guidance.
   - Adds cost-aware recommendation constraints.

Optional fifth tool:

5. `hotel_tool`
   - Uses local mock hotel data.
   - Called only when user asks for hotels/accommodation or when useful for demo.

### Tool Use and Dynamic Decision-Making

The agent must not call every tool blindly for every request. It should decide which tools to call based on parsed user intent, available inputs, remembered preferences, and follow-up context.

Tool routing rules:

- If the city is missing, do not call trip-planning tools; ask a clarifying question first.
- Always call `attraction_rag_tool` when a city is available and the user asks for an itinerary.
- Call `weather_tool` when a city is available. If dates are provided, use date-aware forecast behavior when supported; otherwise use current or near-term forecast.
- Call `web_search_tool` when the user asks for current information, events, closures, new attractions, recent food recommendations, or when local ChromaDB retrieval returns insufficient context.
- Call `budget_tool` when the user provides a budget or when budget exists in long-term memory. If no budget is available, use medium as the default and mark it as assumed.
- Call `hotel_tool` only when the user asks for hotels, accommodation, places to stay, or lodging suggestions.
- For follow-up requests like “make it cheaper,” reuse short-term memory and call `budget_tool` plus itinerary regeneration.
- For follow-up requests like “add more indoor activities,” reuse short-term memory and call `attraction_rag_tool`, and use `weather_tool` if weather context is relevant.

The `/chat` response must include `tools_used` so the demo clearly shows which tools were selected dynamically.

### Planning and Reasoning

Use custom planning logic rather than LangChain Plan-and-Execute.

The planner must break the user goal into visible sub-tasks. Example plan:

```json
[
  "Parse destination, duration, interests, budget, dates, and constraints",
  "Retrieve long-term user preferences from ChromaDB",
  "Run city-level RAG retrieval for broad destination context",
  "Run interest-specific RAG retrieval using city context and user preferences",
  "Call weather tool for destination forecast",
  "Apply budget rules and constraints",
  "Optionally retrieve hotels if requested",
  "Generate a structured 2-day itinerary",
  "Save stable user preferences to long-term memory"
]
```

The `/chat` response must include the generated `plan` array so the demo can show the planning mechanism.

### Multi-hop RAG

Use ChromaDB with sentence-transformers embeddings.

RAG hop 1: city overview retrieval.

Example query:

```text
Tokyo major neighborhoods, attractions, food areas, travel overview
```

RAG hop 2: interest-specific retrieval using hop 1 context.

Example query:

```text
Tokyo anime food photography attractions near Akihabara Shibuya Asakusa
```

The final `/chat` response should include a compact `rag_context` or `rag_trace` field with hop summaries for demo/debugging.

External-source style data may come from curated Wikipedia/Wikivoyage/blog-style local documents stored in `app/data/city_docs/`. The MVP may ship with several supported city documents, while still accepting any major city using LLM generalization and weather lookup.

### Memory

Short-term memory:

- Store recent conversation history per `user_id` in process memory.
- Used for follow-ups like “make it cheaper” or “add more museums.”

Long-term memory:

- Store stable user preferences in ChromaDB.
- Retrieve preferences by `user_id` during planning.
- Save stable preferences only.

Save examples:

- “User likes museums.”
- “User prefers vegetarian food.”
- “User prefers low-budget trips.”

Do not save examples:

- “User is going tomorrow.”
- “User is staying at Hotel X this weekend.”
- One-off trip dates.

## API Specification

### `GET /health`

Response:

```json
{
  "status": "ok"
}
```

### `GET /`

Returns minimal HTML chat frontend.

### `POST /chat`

Request:

```json
{
  "user_id": "kavin",
  "message": "Plan a 2-day trip to Tokyo. I like food, anime, and photography. Medium budget."
}
```

Response:

```json
{
  "message": "Here is your personalized 2-day Tokyo itinerary...",
  "itinerary": {},
  "memory_used": ["User prefers budget-friendly food"],
  "tools_used": ["weather_tool", "attraction_rag_tool", "budget_tool"],
  "plan": ["Parse destination and constraints", "Run multi-hop RAG", "Check weather"],
  "rag_trace": {
    "hop_1": [],
    "hop_2": []
  },
  "needs_clarification": false,
  "clarifying_question": null
}
```

### `GET /memory/{user_id}`

Returns stored long-term memories for the user.

### `POST /memory/{user_id}`

Request:

```json
{
  "preference": "I prefer vegetarian food and relaxed itineraries."
}
```

Response:

```json
{
  "status": "saved"
}
```

### `DELETE /memory/{user_id}`

Response:

```json
{
  "status": "memory cleared"
}
```

## Testing Strategy

Use `pytest`.

Test levels:

- Unit tests for parser, planner, budget tool, and memory filtering.
- Integration-style tests for `/chat`, `/health`, and memory endpoints.
- Mock OpenWeatherMap and OpenRouter in tests.
- Use temporary ChromaDB directories during tests.

Minimum acceptance tests:

- Parser extracts city, duration, interests, and budget from a normal trip request.
- Missing city returns a clarifying question.
- Normal trip planning calls weather, attraction RAG, and budget tools.
- RAG tool performs two retrieval hops and returns a trace.
- Long-term memory can save, retrieve, and reset preferences.
- Stable preferences influence a later trip request.
- `/` returns HTML.
- `/health` returns status ok.

## Boundaries

Always:

- Use environment variables for API keys.
- Include `tools_used` in chat responses.
- Include visible `plan` output in chat responses.
- Include multi-hop RAG trace or summaries.
- Persist ChromaDB under configurable `CHROMA_PATH`.
- Gracefully handle missing OpenRouter/OpenWeatherMap keys.
- Mock external APIs in tests.

Ask first:

- Adding paid APIs or services.
- Adding a full frontend framework.
- Adding a separate database such as Postgres.
- Changing deployment target from Render.
- Expanding scope beyond the assessment MVP.

Never:

- Commit `.env` or API keys.
- Hardcode secrets.
- Store sensitive travel details as long-term memory.
- Remove failing tests to pass CI.
- Make real API calls from tests.

## Success Criteria

The feature is complete when:

- `GET /health` returns `{ "status": "ok" }`.
- `GET /` serves a minimal working HTML chat page.
- `POST /chat` returns a 2-day itinerary with morning/afternoon/evening sections.
- The chat response includes `tools_used` with at least weather, attraction RAG, web search, and budget tools when the request requires fresh context.
- The agent demonstrates dynamic tool use by skipping unnecessary tools, such as not calling hotel search unless lodging is requested.
- The chat response includes a visible planning `plan`.
- The chat response includes multi-hop RAG trace/summaries.
- OpenWeatherMap is used for real weather when `OPENWEATHER_API_KEY` is configured.
- LangChain DuckDuckGo search tool is used for fresh web context and requires no API key.
- ChromaDB and sentence-transformers are used for RAG and long-term memory.
- User preferences persist across sessions through ChromaDB.
- Vague requests trigger clarifying questions.
- `pytest -q` passes.
- The app runs locally with `uvicorn app.main:app --reload`.
- The app can be deployed to Render with the documented start command.
- `docs/technical_doc.md` maps the implementation to assessment requirements.

## Open Questions

None currently. Key decisions confirmed:

- OpenRouter model: `nvidia/nemotron-3-ultra`
- Weather API: OpenWeatherMap
- Web search API: LangChain DuckDuckGo search tool
- Vector database: ChromaDB
- Embeddings: sentence-transformers
- Frontend: minimal HTML served by FastAPI
