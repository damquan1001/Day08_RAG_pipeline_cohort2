from typing import List, Optional
import google.generativeai as genai
from system_contracts import ChatMessage
from src.module_rag_core.ports.outbound import LLMServicePort


class GeminiGenerativeAIAdapter(LLMServicePort):
    """Outbound adapter connecting to Google Gemini API using google-generativeai SDK."""

    def __init__(self) -> None:
        self.api_key: Optional[str] = None
        self.model_name: str = "gemini-2.5-flash"
        self.temperature: float = 0.2

    def configure(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.2,
    ) -> None:
        """Set Gemini API Key and model configurations dynamically."""
        self.api_key = api_key
        
        # Strip 'models/' prefix if present
        model_clean = model_name
        if model_clean.startswith("models/"):
            model_clean = model_clean[len("models/"):]
            
        # Map unsupported 'gemini-3.5-flash-lite' (hardcoded in friend's UI) to a valid model
        if model_clean == "gemini-3.5-flash-lite":
            self.model_name = "gemini-3.5-flash"
        else:
            self.model_name = model_clean


        self.temperature = temperature
        if api_key:
            genai.configure(api_key=api_key)


    def embed_query(self, query: str) -> List[float]:
        """Embed a query text into a vector representation using text-embedding-004 with fallback."""
        if not self.api_key:
            return [0.0] * 1024  # Default length for text-embedding-004

        # Try text-embedding-004
        try:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query",
            )
            if isinstance(response, dict) and "embedding" in response:
                return response["embedding"]
            elif hasattr(response, "embedding"):
                return response.embedding
            return [0.0] * 1024
        except Exception as e:
            # Try falling back to embedding-001
            try:
                response = genai.embed_content(
                    model="models/embedding-001",
                    content=query,
                    task_type="retrieval_query",
                )
                if isinstance(response, dict) and "embedding" in response:
                    return response["embedding"]
                elif hasattr(response, "embedding"):
                    return response.embedding
                return [0.0] * 1024
            except Exception as e2:
                print(f"Gemini embed_content error with both models: {e2}")
                # Return a dummy vector so the flow does not crash
                return [0.0] * 1024

    def condense_query(
        self,
        chat_history: List[ChatMessage],
        latest_query: str,
    ) -> str:
        """Condense a multi-turn conversation and follow-up query into a standalone query."""
        if not chat_history or not self.api_key:
            return latest_query

        history_str = ""
        for msg in chat_history:
            role = "Người dùng" if msg.role == "user" else "Trợ lý"
            history_str += f"{role}: {msg.content}\n"

        prompt = (
            "Dựa trên lịch sử hội thoại sau đây:\n"
            f"{history_str}\n"
            f"Hãy phân tích câu hỏi tiếp theo của người dùng: '{latest_query}'\n\n"
            "Yêu cầu:\n"
            "1. Nếu câu hỏi mới là câu chào hỏi (ví dụ: chào bạn, hello, hi), lời cảm ơn (ví dụ: cảm ơn, thank you), hoặc xã giao, hãy trả về nguyên văn câu hỏi đó.\n"
            "2. Đối với các câu hỏi tra cứu, hãy chuyển đổi nó thành một câu hỏi độc lập (standalone query) bằng tiếng Việt. "
            "Để tối ưu hóa tìm kiếm từ khóa (Lexical Search), bạn hãy TỰ ĐỘNG MỞ RỘNG truy vấn bằng cách bổ sung thêm các thuật ngữ pháp lý đồng nghĩa hoặc liên quan mật thiết trong Luật Việt Nam "
            "(ví dụ: 'hít heroin/đập đá' -> thêm cụm từ 'sử dụng trái phép chất ma túy'; 'trồng cần sa/thuốc phiện' -> thêm cụm từ 'trồng cây có chứa chất ma túy'; 'bị phạt thế nào' -> thêm cụm từ 'xử lý vi phạm, xử phạt, hình phạt').\n\n"
            "Chỉ trả về câu hỏi độc lập đã được mở rộng, tuyệt đối không kèm giải thích hay thêm bất kỳ thông tin ngoài yêu cầu."
        )

        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature
                ),
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini condense_query error: {e}")
            return latest_query

    def generate_answer(
        self,
        system_prompt: str,
        context: str,
        query: str,
    ) -> str:
        """Synthesize a grounded answer using Gemini LLM."""
        if not self.api_key:
            return (
                f"Lỗi: Chưa nhập Gemini API Key. Đây là câu trả lời giả lập cho câu hỏi '{query}' "
                "để xác thực kết nối. Vui lòng nhập API Key ở sidebar."
            )

        prompt = (
            f"Hệ thống: {system_prompt}\n\n"
            f"Ngữ cảnh (Context):\n{context}\n\n"
            f"Câu hỏi (Question): {query}\n\n"
            "Trả lời:"
        )

        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature
                ),
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini generate_answer error: {e}")
            return (
                f"Lỗi gọi Gemini API ({e}). Đây là câu trả lời giả lập cho câu hỏi '{query}'. "
                "Theo quy định pháp luật ma túy, các hành vi tàng trữ trái phép đều có thể bị truy cứu trách nhiệm hình sự."
            )
