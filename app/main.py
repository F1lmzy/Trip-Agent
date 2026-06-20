import contextlib
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from app.agent.orchestrator import AgentServices, handle_chat
from app.agent.streaming import stream_chat
from app.config import get_settings
from app.memory.long_term import long_term_memory
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MemoryAddRequest,
    MemoryResponse,
    StatusResponse,
    ToolInfoResponse,
    ToolsListResponse,
)
from app.tools.registry import list_tools as list_tool_metadata
from app.tools.registry import tools_count

_STATIC_DIR = Path(__file__).parent / "static"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the MCP session manager lifecycle alongside the FastAPI app."""
    try:
        from app.mcp_server import mcp

        async with mcp.session_manager.run():
            yield
    except Exception:
        # If the MCP server cannot start, keep the REST API working.
        yield


app = FastAPI(title="Intelligent Travel Planning AI Agent", lifespan=lifespan)
agent_services = AgentServices()


def _mount_mcp() -> None:
    """Mount the MCP streamable-http app at /mcp if available."""
    try:
        from app.mcp_server import create_mcp_app

        app.mount("/mcp", create_mcp_app())
    except Exception:
        # MCP is optional; the REST API still works without it.
        pass


_mount_mcp()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        tools_available=tools_count(),
        mcp_endpoint="/mcp",
        openrouter_configured=bool(settings.openrouter_api_key),
        openweather_configured=bool(settings.openweather_api_key),
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = _STATIC_DIR / "index.html"
    return HTMLResponse(
        index_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return handle_chat(request, user_memory=long_term_memory, services=agent_services)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream the chat flow as Server-Sent Events.

    Emits plan, tool, message, and result events, ending with the full
    ChatResponse payload. Used by API clients via POST; the browser uses the
    GET /chat/stream endpoint below with EventSource (EventSource is GET-only).
    """
    return StreamingResponse(
        stream_chat(request, user_memory=long_term_memory, services=agent_services),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/chat/stream")
def chat_stream_get(user_id: str, message: str) -> StreamingResponse:
    """GET variant of /chat/stream for browser EventSource consumption.

    EventSource only supports GET with no request body, so the chat request is
    passed as query params. The SSE stream is identical to the POST endpoint.
    """
    request = ChatRequest(user_id=user_id, message=message)
    return StreamingResponse(
        stream_chat(request, user_memory=long_term_memory, services=agent_services),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/tools", response_model=ToolsListResponse)
def api_tools() -> ToolsListResponse:
    """List all available tools with metadata."""
    tools = [
        ToolInfoResponse(
            id=tool.id,
            name=tool.name,
            description=tool.description,
            type=tool.type,
            selected=tool.selected,
        )
        for tool in list_tool_metadata()
    ]
    return ToolsListResponse(tools=tools, total=len(tools))


@app.get("/memory/{user_id}", response_model=MemoryResponse)
def get_memory(user_id: str) -> MemoryResponse:
    return MemoryResponse(user_id=user_id, memories=long_term_memory.get_preferences(user_id))


@app.post("/memory/{user_id}", response_model=StatusResponse)
def add_memory(user_id: str, request: MemoryAddRequest) -> StatusResponse:
    long_term_memory.add_preference(user_id, request.preference)
    return StatusResponse(status="saved")


@app.delete("/memory/{user_id}", response_model=StatusResponse)
def reset_memory(user_id: str) -> StatusResponse:
    long_term_memory.clear_preferences(user_id)
    return StatusResponse(status="memory cleared")
