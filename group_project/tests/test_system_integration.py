import pytest
from typing import List
from system_contracts import RAGCoreInterface, RAGConfig, RAGAnswer, Document, ChatMessage

class FakeRAGCore(RAGCoreInterface):
    """A Fake RAG core class to simulate RAG Core behavior for testing integration contracts."""
    def __init__(self):
        self.config = None
        self.sessions = {}

    def configure(self, config: RAGConfig) -> None:
        self.config = config

    def generate_answer(self, session_id: str, user_query: str, chat_history: List[ChatMessage]) -> RAGAnswer:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            
        doc = Document(id="doc_fake_1", content="Nội dung luật ma tuý giả lập", score=0.95)
        answer = f"Trả lời cho: '{user_query}' sử dụng mô hình {self.config.llm_model_name if self.config else 'default'}"
        
        self.sessions[session_id].append(ChatMessage(role="user", content=user_query))
        self.sessions[session_id].append(ChatMessage(role="assistant", content=answer))
        
        return RAGAnswer(
            answer=answer,
            sources=[doc],
            standalone_query=user_query
        )

    def clear_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            del self.sessions[session_id]

def test_system_contract_integration():
    # Instantiate fake engine
    rag_engine: RAGCoreInterface = FakeRAGCore()
    
    # 1. Test passing UI configurations
    config = RAGConfig(
        gemini_api_key="fake-key",
        llm_model_name="gemini-1.5-pro",
        temperature=0.7,
        use_reranker=True
    )
    rag_engine.configure(config)
    
    # 2. Test generation through the contract
    session_id = "user-session-999"
    history = [ChatMessage(role="user", content="Xin chào")]
    
    result = rag_engine.generate_answer(session_id, "Hình phạt ma tuý thế nào?", history)
    
    # Verify outputs match contract schema
    assert isinstance(result, RAGAnswer)
    assert "Hình phạt ma tuý thế nào?" in result.answer
    assert result.standalone_query == "Hình phạt ma tuý thế nào?"
    assert len(result.sources) == 1
    assert result.sources[0].id == "doc_fake_1"
    assert result.sources[0].score == 0.95
