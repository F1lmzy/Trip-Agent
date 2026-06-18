from collections import defaultdict, deque
from typing import Literal

from pydantic import BaseModel


Role = Literal["user", "assistant"]


class ConversationMessage(BaseModel):
    role: Role
    content: str


class ShortTermMemory:
    def __init__(self, max_messages_per_user: int = 10) -> None:
        self.max_messages_per_user = max_messages_per_user
        self._messages: dict[str, deque[ConversationMessage]] = defaultdict(
            lambda: deque(maxlen=self.max_messages_per_user)
        )

    def add_message(self, user_id: str, role: Role, content: str) -> None:
        self._messages[user_id].append(ConversationMessage(role=role, content=content))

    def get_history(self, user_id: str) -> list[ConversationMessage]:
        return list(self._messages[user_id])

    def get_last_user_message(self, user_id: str) -> str | None:
        for message in reversed(self._messages[user_id]):
            if message.role == "user":
                return message.content
        return None

    def has_history(self, user_id: str) -> bool:
        return bool(self._messages[user_id])

    def clear(self, user_id: str) -> None:
        self._messages.pop(user_id, None)


short_term_memory = ShortTermMemory()
