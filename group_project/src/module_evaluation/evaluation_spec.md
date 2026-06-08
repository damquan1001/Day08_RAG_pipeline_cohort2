# Module Evaluation (Thành Viên D)

## 1. Vai Trò
Module này chịu trách nhiệm chạy các đánh giá tự động chất lượng đầu ra của RAG chatbot, so sánh các cấu hình khác nhau (A/B testing) và xuất báo cáo kết quả:
*   Đọc và xử lý bộ dữ liệu `golden_dataset.json`.
*   Tích hợp framework đánh giá (DeepEval hoặc RAGAS).
*   Chạy đo đạc 4 chỉ số chính: Faithfulness, Answer Relevance, Context Recall, Context Precision.
*   Xuất báo cáo dạng markdown.

## 2. Đặc Tả Usecases Cần Thực Hiện
1.  **Load Test Cases:** Đọc file dữ liệu từ `src/module_dataset_creator/golden_dataset.json`.
2.  **Generate Test Outputs:** Gọi đến `RAGCoreInterface.generate_answer` để lấy câu trả lời thực tế và các nguồn tài liệu trích dẫn tương ứng cho từng câu hỏi.
3.  **Compute Metrics:** Sử dụng các LLM-as-a-judge (được cấu hình bằng Gemini) để tính điểm số cho 4 chỉ số.
4.  **A/B Comparison:**
    *   Cấu hình A (Ví dụ: `use_reranker=False`, `search_mode="dense"`).
    *   Cấu hình B (Ví dụ: `use_reranker=True`, `search_mode="hybrid"`).
    *   Chạy evaluation trên cả 2 cấu hình và lập bảng so sánh điểm.
5.  **Generate Markdown Report:** Ghi điểm số trung bình, bảng so sánh A/B và liệt kê 3-5 câu hỏi bị điểm thấp nhất kèm phân tích nguyên nhân lỗi vào tệp `results.md` ở ngoài.

## 3. Quy Định Kết Nối Với RAG Core
Thành viên D sử dụng `RAGCoreInterface` từ file `system_contracts.py` để thay đổi cấu hình và gọi sinh câu trả lời trong quá trình kiểm thử:

```python
from system_contracts import RAGCoreInterface, RAGConfig
import json

# Lấy instance của RAG Core và cấu hình để chạy test
rag_engine.configure(RAGConfig(
    gemini_api_key=api_key,
    use_reranker=True,
    top_k=5
))

# Lấy câu trả lời thực tế phục vụ eval
result = rag_engine.generate_answer("eval-session", test_case["question"], [])
actual_output = result.answer
retrieval_context = [doc.content for doc in result.sources]
```

## 4. Cách Tổ Chức Code Bên Trong
Thành viên D tự thiết kế cấu trúc code bên trong `src/module_evaluation/`. Khuyên dùng:
*   `eval_pipeline.py`: Script chính để chạy vòng lặp đánh giá.
*   `results_formatter.py`: Định dạng kết quả kiểm thử và xuất file markdown.
