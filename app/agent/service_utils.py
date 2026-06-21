from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.config import get_settings

if TYPE_CHECKING:
    from app.agent.orchestrator import AgentServices


def service_value(services: AgentServices, field_name: str, settings_name: str) -> Any:
    value = getattr(services, field_name)
    if value is not None:
        return value
    if not services.use_environment:
        return "" if field_name.endswith("_api_key") else None
    return getattr(get_settings(), settings_name)
