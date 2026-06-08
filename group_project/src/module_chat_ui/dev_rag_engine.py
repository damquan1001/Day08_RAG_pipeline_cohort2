from typing import List

from system_contracts import ChatMessage, Document, RAGAnswer, RAGConfig, RAGCoreInterface


class DevelopmentRAGCore(RAGCoreInterface):
    """Contract-compatible fallback until the real RAGCoreEngine is available."""

    def __init__(self) -> None:
        self.config = RAGConfig(gemini_api_key="")
        self.sessions: dict[str, list[ChatMessage]] = {}

    def configure(self, config: RAGConfig) -> None:
        self.config = config

    def generate_answer(
        self,
        session_id: str,
        user_query: str,
        chat_history: List[ChatMessage],
    ) -> RAGAnswer:
        self.sessions.setdefault(session_id, [])
        answer = (
            "RAG core is not implemented yet, so this is a UI development response. "
            f"Your question was: {user_query}\n\n"
            "When `src.module_rag_core.rag_engine.RAGCoreEngine` is added, this screen "
            "will call the real retrieval and generation pipeline through `RAGCoreInterface`."
        )
        source = Document(
            id="ui-dev-placeholder",
            content=(
                "Placeholder source emitted by DevelopmentRAGCore. Replace by implementing "
                "RAGCoreEngine in the RAG core module."
            ),
            metadata={
                "title": "Development fallback",
                "model": self.config.llm_model_name,
                "top_k": self.config.top_k,
                "reranker": self.config.use_reranker,
                "history_turns": len(chat_history),
            },
            score=1.0,
        )
        self.sessions[session_id].extend(
            [
                ChatMessage(role="user", content=user_query),
                ChatMessage(role="assistant", content=answer),
            ]
        )
        return RAGAnswer(answer=answer, sources=[source], standalone_query=user_query)

    def clear_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
