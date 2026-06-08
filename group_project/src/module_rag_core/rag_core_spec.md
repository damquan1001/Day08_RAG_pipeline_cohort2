# Module RAG Core (Thành Viên A)

## 1. Vai Trò
Module này chịu trách nhiệm cho toàn bộ phần "bộ não" của hệ thống RAG Chatbot:
*   Truy vấn kết hợp (Hybrid Search: Vector Database & Lexical BM25).
*   Sắp xếp lại độ phù hợp của tài liệu (Reranker).
*   Giao tiếp mô hình ngôn ngữ lớn (Gemini LLM) để viết lại câu hỏi nối tiếp và sinh câu trả lời kèm citation (trích dẫn nguồn).
*   Lưu trữ lịch sử hội thoại nội bộ (Conversation Memory).

## 2. Đặc Tả Usecases Cần Thực Hiện
1.  **Configure Engine:** Nhận cấu hình `RAGConfig` (bao gồm `gemini_api_key`, `llm_model_name`, `temperature`, `top_k`, v.v.) và khởi tạo/cập nhật client kết nối đến Gemini và cơ sở dữ liệu.
2.  **Query Condensation (Viết lại câu hỏi):** Đọc lịch sử hội thoại và viết lại câu hỏi follow-up của người dùng thành câu độc lập.
3.  **Hybrid Retrieval:** Lấy tài liệu từ VectorDB (Weaviate) sử dụng Gemini Embedding và Lexical search (BM25). Hợp nhất kết quả trùng lặp.
4.  **Rerank (Sắp xếp lại):** Lọc top-k tài liệu tốt nhất bằng Cross-Encoder (nếu tham số `use_reranker=True`).
5.  **LLM Generation:** Gọi Gemini LLM tạo câu trả lời, đảm bảo có trích dẫn nguồn (ví dụ: `[1]`, `[2]`).

## 3. Quy Định Đầu Vào & Đầu Ra (Input/Output Interface)
Thành viên A phải xuất ra một Class kế thừa `RAGCoreInterface` từ file `system_contracts.py`.

```python
from system_contracts import RAGCoreInterface, RAGConfig, RAGAnswer, ChatMessage

class RAGCoreEngine(RAGCoreInterface):
    def configure(self, config: RAGConfig) -> None:
        # Triển khai logic cấu hình mô hình
        pass

    def generate_answer(self, session_id: str, user_query: str, chat_history: List[ChatMessage]) -> RAGAnswer:
        # Triển khai logic RAG & Memory
        # Trả về đối tượng kiểu RAGAnswer
        pass

    def clear_session(self, session_id: str) -> None:
        # Triển khai logic xoá lịch sử chat
        pass
```

## 4. Cách Tổ Chức Code Bên Trong
Thành viên A có toàn quyền thiết kế cấu trúc thư mục bên trong `src/module_rag_core/`. Khuyên dùng:
*   `vector_store.py`: Tương tác với cơ sở dữ liệu Vector (Weaviate).
*   `lexical_store.py`: Tương tác với BM25 index.
*   `gemini_client.py`: Khởi tạo và gọi Google Gemini API.
*   `memory.py`: Quản lý lưu lịch sử chat.
