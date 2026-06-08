from typing import List, Optional
import google.generativeai as genai
from system_contracts import ChatMessage


class GeminiClient:
    """Manages LLM interaction with the Google Gemini API for RAG operations."""

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
        """Configure Gemini API connection and model options."""
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        if api_key:
            genai.configure(api_key=api_key)

    def condense_query(
        self,
        chat_history: List[ChatMessage],
        latest_query: str,
    ) -> str:
        """
        Rewrite a follow-up query based on chat history to make it standalone.
        """
        # Skeleton fallback: returns latest_query unmodified
        return latest_query

    def generate_answer_with_citation(
        self,
        system_prompt: str,
        context: str,
        query: str,
    ) -> str:
        """
        Generate answer grounding on context documents with clear citations.
        """
        # Skeleton default answer
        return (
            f"Đây là câu trả lời giả lập cho câu hỏi '{query}' dựa trên tài liệu. "
            "Theo Luật phòng chống ma tuý, các hành vi tàng trữ bị nghiêm cấm [doc_weaviate_1]."
        )
