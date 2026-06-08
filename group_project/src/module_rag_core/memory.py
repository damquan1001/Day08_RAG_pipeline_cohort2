from typing import List
from system_contracts import ChatMessage


class MemoryManager:
    """Manages conversation memory and history for multiple chat sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, List[ChatMessage]] = {}

    def get_history(self, session_id: str) -> List[ChatMessage]:
        """Retrieve chat history for a session."""
        return self._sessions.get(session_id, [])

    def add_message(self, session_id: str, message: ChatMessage) -> None:
        """Add a ChatMessage to the session history."""
        self._sessions.setdefault(session_id, []).append(message)

    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for the session."""
        self._sessions.pop(session_id, None)
