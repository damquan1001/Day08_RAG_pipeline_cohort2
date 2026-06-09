from abc import ABC, abstractmethod
from typing import List, Optional
from system_contracts import ChatMessage, Document


class VectorStorePort(ABC):
    """Port for communicating with vector and keyword databases."""

    @abstractmethod
    def connect(self, url: str, api_key: Optional[str] = None) -> None:
        """Establish connection to the database instance."""
        pass

    @abstractmethod
    def hybrid_search(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> List[Document]:
        """Retrieve candidate documents using vector & lexical similarity."""
        pass


class LLMServicePort(ABC):
    """Port for interacting with Google Gemini API."""

    @abstractmethod
    def configure(
        self,
        api_key: str,
        model_name: str,
        temperature: float,
    ) -> None:
        """Configure model parameters dynamically."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a query text into a vector representation."""
        pass

    @abstractmethod
    def condense_query(
        self,
        chat_history: List[ChatMessage],
        latest_query: str,
    ) -> str:
        """Condense a multi-turn conversation and follow-up query into a standalone query."""
        pass

    @abstractmethod
    def generate_answer(
        self,
        system_prompt: str,
        context: str,
        query: str,
    ) -> str:
        """Synthesize a response with inline citations based on the provided context."""
        pass


class MemoryPort(ABC):
    """Port for managing chat history across sessions."""

    @abstractmethod
    def get_history(self, session_id: str) -> List[ChatMessage]:
        """Get the full history for a session."""
        pass

    @abstractmethod
    def add_message(self, session_id: str, message: ChatMessage) -> None:
        """Store a ChatMessage inside a session."""
        pass

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        pass


class RerankerPort(ABC):
    """Port for document re-ranking."""

    @abstractmethod
    def rerank(
        self,
        query_vector: List[float],
        candidates: List[Document],
        lambda_param: float = 0.7,
        top_k: int = 5,
    ) -> List[Document]:
        """Re-rank candidate documents to balance relevance and diversity."""
        pass

    @abstractmethod
    def reorder_for_llm(self, documents: List[Document]) -> List[Document]:
        """Reorder documents to place important elements at start and end."""
        pass
