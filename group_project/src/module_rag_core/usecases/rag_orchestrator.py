from typing import List
from system_contracts import ChatMessage, RAGAnswer, Document
from src.module_rag_core.ports.outbound import (
    VectorStorePort,
    LLMServicePort,
    MemoryPort,
    RerankerPort,
)


class RAGOrchestrator:
    """
    Pure Domain Use Case / Interactor orchestrating the RAG flow.
    Depends only on outbound ports, inverting dependencies (Clean Architecture).
    """

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
        """
        Orchestrates the entire RAG pipeline from retrieval to response synthesis.
        """
        # Step 1: Condense query based on history to support multi-turn conversation
        standalone_query = self.llm.condense_query(chat_history, user_query)

        # Step 2: Embed query to get query vector
        query_vector = self.llm.embed_query(standalone_query)

        # Step 3: Retrieve candidates from Vector Database
        docs = self.vector_store.hybrid_search(
            query=standalone_query,
            vector=query_vector,
            top_k=top_k * 2 if use_reranker else top_k,
        )

        # Step 4: Apply MMR reranking if enabled
        if use_reranker and docs:
            docs = self.reranker.rerank(
                query_vector=query_vector,
                candidates=docs,
                lambda_param=0.7,
                top_k=top_k,
            )
        else:
            docs = docs[:top_k]

        # Step 5: Reorder docs to mitigate "lost in the middle"
        reordered_docs = self.reranker.reorder_for_llm(docs)

        # Step 6: Construct Prompt Context
        context_parts = []
        for index, doc in enumerate(reordered_docs, start=1):
            source = doc.metadata.get("source", f"doc_{index}")
            context_parts.append(f"Tài liệu [{source}]:\n{doc.content}")
        context_str = "\n\n".join(context_parts)

        # Step 7: Call LLM with friendly yet grounded System Prompt
        system_prompt = (
            "Bạn là một trợ lý ảo chuyên nghiệp am hiểu sâu sắc về Luật Phòng chống ma túy Việt Nam và các tin tức pháp lý liên quan.\n\n"
            "Quy tắc phản hồi:\n"
            "1. Nếu người dùng chào hỏi, cảm ơn hoặc trò chuyện xã giao, hãy phản hồi một cách thân thiện, lịch sự và ngắn gọn, đồng thời hướng dẫn họ cách đặt câu hỏi liên quan đến Luật Phòng chống ma túy.\n"
            "2. Đối với các câu hỏi tra cứu thông tin hoặc kiến thức pháp lý:\n"
            "   a. Chỉ sử dụng thông tin có trong phần Ngữ cảnh (Context) được cung cấp dưới đây. Tuyệt đối không tự suy diễn hoặc sử dụng kiến thức bên ngoài nếu không có trong ngữ cảnh.\n"
            "   b. Với mọi tuyên bố hoặc thông tin đưa ra, bạn PHẢI trích dẫn nguồn bằng cách chèn nhãn tài liệu tương ứng ở định dạng [Nguồn] (ví dụ: nếu nhãn tài liệu là Tài liệu [luat_2021.md] thì trích dẫn là [luat_2021.md]).\n"
            "   c. Nếu thông tin không có trong ngữ cảnh hoặc ngữ cảnh không đủ để trả lời câu hỏi, hãy phản hồi trung thực và lịch sự: \"Tôi không thể xác minh thông tin này từ các tài liệu hiện có.\" Tuyệt đối không tự bịa câu trả lời.\n"
            "3. Trả lời bằng tiếng Việt, cấu trúc mạch lạc, phân chia các đoạn rõ ràng và dễ đọc."
        )

        answer = self.llm.generate_answer(
            system_prompt=system_prompt,
            context=context_str,
            query=standalone_query,
        )

        # Step 8: Save to history memory
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
        """Clear conversation history for the session."""
        self.memory.clear(session_id)
