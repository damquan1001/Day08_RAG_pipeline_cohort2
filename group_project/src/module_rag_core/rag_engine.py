import os
from typing import List

from system_contracts import (
    ChatMessage,
    RAGAnswer,
    RAGConfig,
    RAGCoreInterface,
)
from src.module_rag_core.adapters.in_memory_memory import InMemoryMemoryAdapter
from src.module_rag_core.adapters.weaviate_adapter import WeaviateDockerAdapter
from src.module_rag_core.adapters.gemini_adapter import GeminiGenerativeAIAdapter
from src.module_rag_core.adapters.mmr_reranker import MMRRerankerAdapter
from src.module_rag_core.usecases.rag_orchestrator import RAGOrchestrator


class RAGCoreEngine(RAGCoreInterface):
    """
    Inbound Adapter wrapping the RAG Core module.
    Conforms to RAGCoreInterface, instantiated and consumed by Streamlit UI and tests.
    """

    def __init__(self) -> None:
        self.memory_adapter = InMemoryMemoryAdapter()
        self.vector_store_adapter = WeaviateDockerAdapter()
        self.llm_adapter = GeminiGenerativeAIAdapter()
        self.reranker_adapter = MMRRerankerAdapter()

        self.orchestrator = RAGOrchestrator(
            vector_store=self.vector_store_adapter,
            llm=self.llm_adapter,
            memory=self.memory_adapter,
            reranker=self.reranker_adapter,
        )
        self.config = RAGConfig(gemini_api_key="")

    def configure(self, config: RAGConfig) -> None:
        """Configure LLM client and connect to Weaviate local database."""
        self.config = config

        self.llm_adapter.configure(
            api_key=config.gemini_api_key,
            model_name=config.llm_model_name,
            temperature=config.temperature,
        )

        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        weaviate_key = os.getenv("WEAVIATE_API_KEY")
        self.vector_store_adapter.connect(url=weaviate_url, api_key=weaviate_key)

    def generate_answer(
        self,
        session_id: str,
        user_query: str,
        chat_history: List[ChatMessage],
    ) -> RAGAnswer:
        """Delegates query execution to the core RAGOrchestrator interactor."""
        return self.orchestrator.execute_rag(
            session_id=session_id,
            user_query=user_query,
            chat_history=chat_history,
            top_k=self.config.top_k,
            use_reranker=self.config.use_reranker,
        )

    def clear_session(self, session_id: str) -> None:
        """Delegates session clearing to the core RAGOrchestrator interactor."""
        self.orchestrator.clear_session(session_id)
