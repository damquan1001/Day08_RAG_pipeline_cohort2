# Đặc Tả Kiến Trúc Hệ Thống Toàn Cục (Full System Specification)

Tài liệu này đặc tả kiến trúc tổng thể, mô hình dữ liệu (schemas), hợp đồng kết nối (contracts) và cách phối hợp giữa 4 module của hệ thống Search Engine / RAG Chatbot.

---

## 1. Sơ Đồ Kiến Trúc Tổng Thể

Hệ thống được thiết kế theo mô hình phi tập trung (decentralized). Các module tương tác với nhau thông qua một hợp đồng chung duy nhất tại `system_contracts.py`.

```mermaid
graph TD
    subgraph UI Layer (Thành viên B)
        UI[app.py / module_chat_ui] -->|Đọc/Ghi tham số| Config[RAGConfig]
        UI -->|Hiển thị tin nhắn & Citation| User([Người dùng])
    end

    subgraph Interface Agreement (System Contracts)
        Interface[system_contracts.py]
    end

    subgraph Core Layer (Thành viên A)
        Engine[src/module_rag_core] -.->|Cài đặt/Implement| Interface
        UI -->|Gọi API qua Contract| Engine
        Engine -->|Gọi LLM & Embedding| Gemini[Google Gemini API]
    end

    subgraph Dataset & QA (Thành viên C)
        Dataset[src/module_dataset_creator/golden_dataset.json]
    end

    subgraph Evaluation (Thành viên D)
        Eval[src/module_evaluation/eval_pipeline.py] -->|Đọc| Dataset
        Eval -->|Import & Test Configs| Engine
        Eval -->|Xuất báo cáo| Report[results.md]
    end
```

---

## 2. Cấu Trúc Thư Mục Phân Rã (Decentralized Folder Tree)

Dự án được chia làm 4 module tương ứng với 4 thư mục trong `src/` để 4 thành viên code song song mà không xung đột:

```text
group_project/
├── .env                        # Chứa Gemini API Key và các cấu hình tham số hệ thống
├── system_contracts.py         # Định nghĩa các thực thể và interface giao tiếp chung
├── full_spec.md                # Tài liệu đặc tả hệ thống này
├── app.py                      # Điểm chạy Streamlit UI (quản lý giao diện cấu hình & chat)
├── README.md                   # Mẫu README gốc của bài tập nhóm
├── src/
│   ├── module_rag_core/        # THÀNH VIÊN A: RAG Engine & Conversation Memory
│   │   ├── rag_core_spec.md    # Tài liệu đặc tả riêng cho module RAG Core
│   │   └── (code tự triển khai)
│   ├── module_chat_ui/         # THÀNH VIÊN B: Streamlit UI
│   │   ├── chat_ui_spec.md     # Tài liệu đặc tả riêng cho module Chat UI
│   │   └── (code tự triển khai)
│   ├── module_dataset_creator/ # THÀNH VIÊN C: Quản lý Golden Dataset
│   │   ├── dataset_creator_spec.md # Tài liệu đặc tả riêng cho module tạo dataset
│   │   └── golden_dataset.json # Bộ dữ liệu test 15+ câu hỏi mẫu
│   └── module_evaluation/      # THÀNH VIÊN D: Đánh giá RAG (DeepEval/Ragas)
│       ├── evaluation_spec.md  # Tài liệu đặc tả riêng cho module đánh giá
│       └── (code tự triển khai)
└── tests/
    └── test_system_integration.py # Bộ kiểm thử tích hợp (System Integration Test)
```

---

## 3. Giao Diện & Tham Số Cấu Hình Động (Dynamic Configuration)

*   **Google Gemini LLM & Embedding:** Toàn bộ mô hình nhúng và sinh câu trả lời được cấu hình sử dụng Google Gemini.
*   **File cấu hình `.env`:** Quản lý các cấu hình mặc định (xem tại [.env.example](file:///c:/Users/ntddatj/github-ntddatj/github-classroom/Day08_RAG_pipeline_cohort2/group_project/.env.example)).
*   **Tham số trên giao diện UI:** Cho phép người dùng cấu hình động qua Streamlit Sidebar:
    *   *LLM Model:* `gemini-1.5-flash` hoặc `gemini-1.5-pro`
    *   *Temperature:* Lựa chọn từ `0.0` đến `1.0`
    *   *Top K:* Số lượng tài liệu truy xuất (1 đến 10)
    *   *Reranker:* Hộp checkbox bật/tắt reranking.

---

## 4. Hợp Đồng Kết Nối Giữa Các Module (`system_contracts.py`)

Các module kết nối với nhau thông qua file [system_contracts.py](file:///c:/Users/ntddatj/github-ntddatj/github-classroom/Day08_RAG_pipeline_cohort2/group_project/system_contracts.py):

*   `RAGConfig`: Gói tất cả tham số truyền từ UI sang RAG Engine.
*   `RAGAnswer`: Định dạng kết quả trả về từ RAG Engine (bao gồm câu trả lời, nguồn tài liệu `Document` có điểm số, câu hỏi độc lập sau khi viết lại).
*   `RAGCoreInterface`: Lớp abstract định nghĩa các phương thức:
    *   `configure(config: RAGConfig)`: Nhận cấu hình từ UI.
    *   `generate_answer(session_id: str, user_query: str, chat_history: List[ChatMessage])`: Nhận câu hỏi mới & lịch sử, trả về `RAGAnswer`.
    *   `clear_session(session_id: str)`: Xoá lịch sử hội thoại của session.

---

## 5. Hướng Dẫn Chạy & Kiểm Thử Tích Hợp

### 5.1. Khởi Chạy Streamlit UI
```bash
# Cài đặt thư viện
pip install -r requirements.txt

# Chạy app
streamlit run app.py
```

### 5.2. Chạy Kiểm Thử Tích Hợp Hệ Thống
Để chạy bộ test tự động xác thực sự ăn khớp về giao diện hợp đồng giữa các module:
```bash
pytest group_project/tests/test_system_integration.py
```
