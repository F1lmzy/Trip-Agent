from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    message: str
    itinerary: dict[str, Any] = Field(default_factory=dict)
    memory_used: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list)
    rag_trace: dict[str, Any] = Field(default_factory=dict)
    needs_clarification: bool = False
    clarifying_question: str | None = None


class MemoryAddRequest(BaseModel):
    preference: str = Field(min_length=1)


class MemoryResponse(BaseModel):
    user_id: str
    memories: list[str] = Field(default_factory=list)


class StatusResponse(BaseModel):
    status: str
