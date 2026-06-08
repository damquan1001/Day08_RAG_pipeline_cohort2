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
        """Embed a query text into a vector representation using text-embedding-004."""
        if not self.api_key:
            return [0.0] * 1024  # Default length for text-embedding-004

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
            print(f"Gemini embed_content error: {e}")
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
            "1. Nếu câu hỏi mới là một câu chào hỏi (ví dụ: chào bạn, hello, hi), lời cảm ơn (ví dụ: cảm ơn, thank you), hoặc là câu hỏi độc lập đã rõ nghĩa và không cần thông tin từ lịch sử hội thoại trước đó, hãy trả về nguyên văn câu hỏi mới đó.\n"
            "2. Nếu câu hỏi mới chứa các từ thay thế hoặc tham chiếu cần ngữ cảnh từ lịch sử hội thoại (ví dụ: 'nó', 'hành vi này', 'điều đó', 'ở trên'), hãy chuyển đổi nó thành một câu hỏi độc lập (standalone query) bằng tiếng Việt, chứa đầy đủ ngữ cảnh để có thể hiểu được mục đích truy vấn mà không cần đọc lại lịch sử.\n\n"
            "Chỉ trả về câu hỏi cuối cùng thu được, tuyệt đối không kèm giải thích hay thêm bớt thông tin ngoài yêu cầu."
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
