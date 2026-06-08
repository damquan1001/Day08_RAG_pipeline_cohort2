from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

try:
    from pydantic import BaseModel, Field
except ImportError:
    class _FieldDefault:
        def __init__(self, default=None, default_factory=None):
            self.default = default
            self.default_factory = default_factory

    def Field(default=None, default_factory=None):
        return _FieldDefault(default=default, default_factory=default_factory)

    class BaseModel:
        """Small fallback used when pydantic is not installed in the demo env."""

        def __init__(self, **data):
            annotations = {}
            for cls in reversed(self.__class__.mro()):
                annotations.update(getattr(cls, "__annotations__", {}))

            for name in annotations:
                if name in data:
                    value = data.pop(name)
                else:
                    default = getattr(self.__class__, name, None)
                    if isinstance(default, _FieldDefault):
                        if default.default_factory is not None:
                            value = default.default_factory()
                        else:
                            value = default.default
                    elif hasattr(self.__class__, name):
                        value = default
                    else:
                        raise TypeError(f"Missing required field: {name}")
                setattr(self, name, value)

            for name, value in data.items():
                setattr(self, name, value)

        def model_dump(self):
            return dict(self.__dict__)

# ==========================================
# 1. Configuration Schema (RAG Configuration)
# ==========================================
class RAGConfig(BaseModel):
    """Schema for configuring RAG models and parameters from .env and UI"""
    gemini_api_key: str
    llm_model_name: str = "gemini-3.5-flash-lite"
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
