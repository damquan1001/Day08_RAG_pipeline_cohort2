from typing import List
from system_contracts import ChatMessage
from src.module_rag_core.ports.outbound import MemoryPort


class InMemoryMemoryAdapter(MemoryPort):
    """Outbound adapter storing chat session histories in memory."""

    def __init__(self) -> None:
        self._sessions: dict[str, List[ChatMessage]] = {}

    def get_history(self, session_id: str) -> List[ChatMessage]:
        """Get the full history for a session."""
        return self._sessions.get(session_id, [])

    def add_message(self, session_id: str, message: ChatMessage) -> None:
        """Store a ChatMessage inside a session."""
        self._sessions.setdefault(session_id, []).append(message)

    def clear(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self._sessions.pop(session_id, None)
