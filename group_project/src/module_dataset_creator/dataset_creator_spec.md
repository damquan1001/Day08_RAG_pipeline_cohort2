# Module Golden Dataset Creator (Thành Viên C)

## 1. Vai Trò
Module này chịu trách nhiệm chuẩn bị và quản lý bộ dữ liệu câu hỏi kiểm thử chuẩn (Golden Dataset). Đây là bộ dữ liệu làm cơ sở đánh giá chất lượng hệ thống RAG độc lập và khách quan.

## 2. Yêu Cầu Golden Dataset
*   Phải có tối thiểu **15 câu hỏi/đáp**.
*   Các câu hỏi phải đa dạng, xoay quanh tài liệu luật phòng chống ma túy và tin tức liên quan đã crawl được.
*   Bao gồm cả câu hỏi trực tiếp (chỉ cần tra cứu 1 điều luật) và câu hỏi suy luận (cần thông tin từ nhiều điều luật/tin tức khác nhau).

## 3. Quy Định Định Dạng Schema
Bộ dữ liệu phải được xuất ra file JSON đặt tên là `golden_dataset.json` nằm trong thư mục `src/module_dataset_creator/` với định dạng chính xác sau:

```json
[
  {
    "question": "Câu hỏi của người dùng thực tế?",
    "expected_answer": "Câu trả lời chuẩn mực dựa trên tài liệu pháp luật làm bằng chứng.",
    "expected_context": [
      "Đoạn văn bản trích dẫn số 1 từ luật ma tuý làm bằng chứng",
      "Đoạn văn bản trích dẫn số 2 từ tin tức liên quan làm bằng chứng"
    ]
  },
  ...
]
```

## 4. Công Cụ Hỗ Trợ (Tuỳ Chọn)
Thành viên C có thể viết các script python trong module này để:
*   `generate_dataset.py`: Tự động sinh câu hỏi thô từ tài liệu (sử dụng Gemini LLM sinh Q&A). Sau đó thành viên C tiến hành chỉnh sửa thủ công để đảm bảo tính chuẩn xác.
*   `validate_schema.py`: Script kiểm tra tính hợp lệ của cấu trúc file JSON trước khi đẩy lên Github.
