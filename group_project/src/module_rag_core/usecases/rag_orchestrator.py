from __future__ import annotations

from typing import List

from system_contracts import ChatMessage, RAGAnswer
from src.module_rag_core.ports.outbound import (
    LLMServicePort,
    MemoryPort,
    RerankerPort,
    VectorStorePort,
)


class RAGOrchestrator:
    """Pure use case that coordinates memory, retrieval, reranking, and LLM."""

    def __init__(
        self,
        vector_store: VectorStorePort,
        llm: LLMServicePort,
        memory: MemoryPort,
        reranker: RerankerPort,
    ) -> None:
        self.vector_store = vector_store
        self.llm = llm
        self.memory = memory
        self.reranker = reranker

    def execute_rag(
        self,
        session_id: str,
        user_query: str,
        chat_history: List[ChatMessage],
        top_k: int = 5,
        use_reranker: bool = True,
    ) -> RAGAnswer:
        standalone_query = self.llm.condense_query(chat_history, user_query)
        query_vector = self.llm.embed_query(standalone_query)

        docs = self.vector_store.hybrid_search(
            query=standalone_query,
            vector=query_vector,
            top_k=top_k * 2 if use_reranker else top_k,
        )

        if use_reranker and docs:
            docs = self.reranker.rerank(
                query_vector=query_vector,
                candidates=docs,
                lambda_param=0.7,
                top_k=top_k,
            )
        else:
            docs = docs[:top_k]

        reordered_docs = self.reranker.reorder_for_llm(docs)
        context_str = self._build_context(reordered_docs)
        answer = self.llm.generate_answer(
            system_prompt=self._system_prompt(),
            context=context_str,
            query=standalone_query,
        )

        self.memory.add_message(
            session_id, ChatMessage(role="user", content=user_query)
        )
        self.memory.add_message(
            session_id, ChatMessage(role="assistant", content=answer)
        )

        return RAGAnswer(
            answer=answer,
            sources=docs,
            standalone_query=standalone_query,
        )

    def clear_session(self, session_id: str) -> None:
        self.memory.clear(session_id)

    @staticmethod
    def _build_context(docs) -> str:
        context_parts = []
        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", f"doc_{index}")
            context_parts.append(f"Tai lieu [{source}]:\n{doc.content}")
        return "\n\n".join(context_parts)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a Vietnamese RAG assistant for drug prevention law and "
            "related legal news. If the user only greets, thanks, or makes "
            "small talk, answer briefly and invite them to ask about drug law. "
            "For legal lookup questions, answer only from the provided context. "
            "Every factual claim must include an inline citation in square "
            "brackets. If the context is insufficient, say that the information "
            "cannot be verified from the available documents."
        )
