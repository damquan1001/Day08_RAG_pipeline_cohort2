# Module Chat UI (Thành Viên B)

## 1. Vai Trò
Module này phụ trách toàn bộ trải nghiệm người dùng cuối (Frontend) của ứng dụng bằng thư viện Streamlit:
*   Giao diện hiển thị cuộc đối thoại (Chat interface).
*   Giao diện thay đổi tham số động (Settings sidebar).
*   Giao diện hiển thị nguồn tài liệu trích dẫn (Citations display).

## 2. Đặc Tả Usecases Cần Thực Hiện
1.  **Render Sidebar Settings:** 
    *   Đọc các biến môi trường mặc định từ `.env`.
    *   Hiển thị các input/select box để tuỳ chỉnh tham số: Lựa chọn Model Gemini (`gemini-1.5-flash`, `gemini-1.5-pro`), Nhiệt độ sáng tạo (`Temperature`), Số tài liệu lấy về (`Top K`), Nút bật/tắt `Reranker`.
    *   Bất kỳ khi nào người dùng nhấn lưu hoặc thay đổi cấu hình, UI phải gọi phương thức `.configure(config)` của RAG Engine.
2.  **Render Chat Message Stream:**
    *   Hiển thị lịch sử hội thoại dưới dạng bong bóng chat người dùng và trợ lý.
    *   Lưu lịch sử tin nhắn trong `st.session_state` của Streamlit để duy trì trạng thái khi trang web tự động re-render.
3.  **Render Citations / Sources:**
    *   Khi nhận kết quả `RAGAnswer` từ RAG Core, giao diện cần hiển thị câu trả lời và tạo ra một vùng hiển thị dẫn chứng rõ ràng (Ví dụ: `st.expander` chứa danh sách tiêu đề luật, nội dung cụ thể và độ tương đồng `similarity score`).

## 3. Quy Định Giao Tiếp Với RAG Core
Thành viên B không được tự viết logic tìm kiếm hay sinh câu trả lời, mà phải import và sử dụng interface `RAGCoreInterface`.

```python
from system_contracts import RAGConfig, ChatMessage
# Cách kết nối:
# 1. Thu thập dữ liệu từ sidebar
config = RAGConfig(
    gemini_api_key=st.session_state.api_key,
    llm_model_name=selected_model,
    temperature=selected_temp,
    top_k=selected_top_k,
    use_reranker=use_reranker
)

# 2. Gửi cấu hình sang cho RAG Core
rag_engine.configure(config)

# 3. Gửi câu hỏi của người dùng và nhận kết quả
rag_answer = rag_engine.generate_answer(
    session_id=st.session_state.session_id,
    user_query=user_input,
    chat_history=get_current_chat_history()
)
```

## 4. Cách Tổ Chức Code Bên Trong
Thành viên B tự thiết kế cấu trúc code bên trong `src/module_chat_ui/`. Khuyên dùng:
*   `components/`: Các module nhỏ vẽ giao diện chat, vẽ sidebar, vẽ citations.
*   `app_state.py`: Quản lý logic lưu/đọc `st.session_state`.
*   Tệp [app.py](file:///c:/Users/ntddatj/github-ntddatj/github-classroom/Day08_RAG_pipeline_cohort2/group_project/app.py) ở thư mục ngoài sẽ gọi khởi tạo giao diện từ module này.
