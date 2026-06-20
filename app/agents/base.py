"""Base classes for the multi-agent system.

Defines the Agent interface, the AgentContext carried through a run, and a
small event type for streaming (used by later iterations; the emitter is
optional and unused in the structural scaffold).

Import-cycle note: AgentServices lives in app.agent.orchestrator, which in
turn imports this package's supervisor. To avoid an import-time cycle we
reference AgentServices only under TYPE_CHECKING and use postponed
annotations. Agents that need orchestrator helpers import them lazily inside
their run() methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from app.agent.parser import ParsedRequest
from app.agent.planner import PlanningResult
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.schemas import ChatResponse

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycle
    from app.agent.orchestrator import AgentServices

# An event emitter is a callback agents call with a dict event payload. The
# streaming layer (later iteration) supplies one that serializes each event
# as an SSE frame as it happens. None means "not streaming".
EventEmitter = Callable[[dict[str, Any]], None]


@dataclass
class AgentContext:
    """Everything an agent needs to handle one chat turn.

    ``parsed`` and ``plan`` are computed by the orchestrator before routing so
    every agent shares the same understanding of the request (and parsing
    happens exactly once). ``event_emitter`` is optional: when provided,
    agents emit progressive events as they work (used by /chat/stream).
    """

    parsed: ParsedRequest
    plan: PlanningResult
    services: "AgentServices"
    memory: ShortTermMemory
    user_memory: LongTermMemory
    user_id: str
    event_emitter: EventEmitter | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, event: dict[str, Any]) -> None:
        """Record an event and forward it to the emitter if streaming."""
        self.events.append(event)
        if self.event_emitter is not None:
            self.event_emitter(event)


class Agent:
    """Base class for specialized agents.

    Subclasses set ``name`` and implement ``run``. The name is surfaced in
    streaming events so the UI can show which agent is working.
    """

    name: str = "Agent"

    def run(self, ctx: AgentContext) -> ChatResponse:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
