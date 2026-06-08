from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

# ==========================================
# 1. Configuration Schema (RAG Configuration)
# ==========================================
class RAGConfig(BaseModel):
    """Schema for configuring RAG models and parameters from .env and UI"""
    gemini_api_key: str
    llm_model_name: str = "gemini-1.5-flash"        # e.g., gemini-1.5-flash, gemini-1.5-pro
    embedding_model_name: str = "text-embedding-004"
    temperature: float = 0.2
    top_k: int = 5
    similarity_threshold: float = 0.5
    use_reranker: bool = True
    custom_params: Dict[str, Any] = Field(default_factory=dict)

# ==========================================
# 2. Data Models (Domain Entities)
# ==========================================
class Document(BaseModel):
    """Document representation retrieved from knowledge base"""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None

class ChatMessage(BaseModel):
    """A single chat turn message representation"""
    role: str  # "user" or "assistant"
    content: str

class RAGAnswer(BaseModel):
    """Final answer returned by the RAG engine"""
    answer: str
    sources: List[Document]
    standalone_query: str

# ==========================================
# 3. RAG Core Module Port (Interface)
# ==========================================
class RAGCoreInterface(ABC):
    """
    Interface/Contract for Module 1 (RAG Core).
    Member A implements this.
    Member B (UI) and Member D (Evaluation) consume this.
    """
    @abstractmethod
    def configure(self, config: RAGConfig) -> None:
        """Configure the active RAG engine parameters and API Keys dynamically"""
        pass

    @abstractmethod
    def generate_answer(self, session_id: str, user_query: str, chat_history: List[ChatMessage]) -> RAGAnswer:
        """
        Accept session ID, query and chat history to:
        1. Condense the query using Gemini LLM.
        2. Perform Hybrid Search (Vector + BM25).
        3. Rerank retrieved candidates (optional).
        4. Synthesize final answer with citations.
        """
        pass

    @abstractmethod
    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for the session"""
        pass
