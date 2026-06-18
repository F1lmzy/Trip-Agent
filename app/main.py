from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.agent.orchestrator import handle_chat
from app.schemas import ChatRequest, ChatResponse, MemoryAddRequest, MemoryResponse, StatusResponse

app = FastAPI(title="Intelligent Travel Planning AI Agent")

_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/health", response_model=StatusResponse)
def health() -> StatusResponse:
    return StatusResponse(status="ok")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = _STATIC_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return handle_chat(request)


@app.get("/memory/{user_id}", response_model=MemoryResponse)
def get_memory(user_id: str) -> MemoryResponse:
    return MemoryResponse(user_id=user_id, memories=[])


@app.post("/memory/{user_id}", response_model=StatusResponse)
def add_memory(user_id: str, request: MemoryAddRequest) -> StatusResponse:
    return StatusResponse(status="memory endpoint scaffolded")


@app.delete("/memory/{user_id}", response_model=StatusResponse)
def reset_memory(user_id: str) -> StatusResponse:
    return StatusResponse(status="memory cleared")
