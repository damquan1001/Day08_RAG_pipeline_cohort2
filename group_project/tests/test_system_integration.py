from typing import List

from system_contracts import (
    ChatMessage,
    Document,
    RAGAnswer,
    RAGConfig,
    RAGCoreInterface,
)


class FakeRAGCore(RAGCoreInterface):
    """Fake RAG core used to verify integration contracts."""

    def __init__(self):
        self.config = None
        self.sessions = {}

    def configure(self, config: RAGConfig) -> None:
        self.config = config

    def generate_answer(
        self,
        session_id: str,
        user_query: str,
        chat_history: List[ChatMessage],
    ) -> RAGAnswer:
        self.sessions.setdefault(session_id, [])

        doc = Document(
            id="doc_fake_1",
            content="Noi dung luat ma tuy gia lap",
            score=0.95,
        )
        model = self.config.llm_model_name if self.config else "default"
        answer = f"Tra loi cho: '{user_query}' su dung mo hinh {model}"

        self.sessions[session_id].append(ChatMessage(role="user", content=user_query))
        self.sessions[session_id].append(ChatMessage(role="assistant", content=answer))

        return RAGAnswer(
            answer=answer,
            sources=[doc],
            standalone_query=user_query,
        )

    def clear_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


def test_system_contract_integration():
    rag_engine: RAGCoreInterface = FakeRAGCore()

    config = RAGConfig(
        gemini_api_key="fake-key",
        llm_model_name="gemini-1.5-pro",
        temperature=0.7,
        use_reranker=True,
    )
    rag_engine.configure(config)

    session_id = "user-session-999"
    history = [ChatMessage(role="user", content="Xin chao")]

    result = rag_engine.generate_answer(
        session_id,
        "Hinh phat ma tuy the nao?",
        history,
    )

    assert isinstance(result, RAGAnswer)
    assert "Hinh phat ma tuy the nao?" in result.answer
    assert result.standalone_query == "Hinh phat ma tuy the nao?"
    assert len(result.sources) == 1
    assert result.sources[0].id == "doc_fake_1"
    assert result.sources[0].score == 0.95


def test_real_rag_core_contract_offline():
    from src.module_rag_core.rag_engine import RAGCoreEngine

    engine = RAGCoreEngine()
    engine.configure(
        RAGConfig(
            gemini_api_key="",
            llm_model_name="gemini-2.5-flash",
            temperature=0.2,
            top_k=3,
            use_reranker=False,
        )
    )

    result = engine.generate_answer(
        "test-session",
        "Hinh phat cho toi tang tru trai phep chat ma tuy la gi?",
        [],
    )

    assert isinstance(result, RAGAnswer)
    assert result.standalone_query == "Hinh phat cho toi tang tru trai phep chat ma tuy la gi?"
    assert len(result.sources) > 0
    assert isinstance(result.sources[0], Document)
    assert "[" in result.answer and "]" in result.answer
