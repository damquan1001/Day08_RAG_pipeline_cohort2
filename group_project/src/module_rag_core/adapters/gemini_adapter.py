from __future__ import annotations

import re
from typing import List, Optional

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from system_contracts import ChatMessage
from src.module_rag_core.ports.outbound import LLMServicePort


class GeminiGenerativeAIAdapter(LLMServicePort):
    """Outbound adapter for Gemini, with an offline fallback for local demos."""

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
        self.api_key = api_key
        self.model_name = self._normalize_model_name(model_name)
        self.temperature = temperature

        if self.api_key and genai is not None:
            genai.configure(api_key=self.api_key)

    def embed_query(self, query: str) -> List[float]:
        if not self.api_key or genai is None:
            return [0.0] * 1024

        for model_name in ["models/text-embedding-004", "models/embedding-001"]:
            try:
                response = genai.embed_content(
                    model=model_name,
                    content=query,
                    task_type="retrieval_query",
                )
                if isinstance(response, dict) and "embedding" in response:
                    return response["embedding"]
                if hasattr(response, "embedding"):
                    return response.embedding
            except Exception as exc:
                print(f"Gemini embed_content error with {model_name}: {exc}")
        return [0.0] * 1024

    def condense_query(
        self,
        chat_history: List[ChatMessage],
        latest_query: str,
    ) -> str:
        if not chat_history or not self.api_key or genai is None:
            return latest_query

        history_text = "\n".join(
            f"{message.role}: {message.content}" for message in chat_history
        )
        prompt = (
            "Rewrite the latest Vietnamese message into a standalone query.\n"
            "Rules:\n"
            "1. If the message is a greeting, thanks, or small talk, return it unchanged.\n"
            "2. For legal lookup questions, expand the query with close Vietnamese "
            "legal terms to improve lexical search. Examples: 'hit heroin' should "
            "include 'su dung trai phep chat ma tuy'; 'trong can sa' should include "
            "'trong cay co chua chat ma tuy'; 'bi phat the nao' should include "
            "'xu ly vi pham, xu phat, hinh phat'.\n"
            "3. Return only the rewritten query, with no explanation.\n\n"
            f"History:\n{history_text}\n\nLatest message: {latest_query}"
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
        except Exception as exc:
            print(f"Gemini condense_query error: {exc}")
            return latest_query

    def generate_answer(
        self,
        system_prompt: str,
        context: str,
        query: str,
    ) -> str:
        if not self.api_key or genai is None:
            return self._offline_answer(context=context, query=query)

        prompt = (
            f"System: {system_prompt}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer in Vietnamese with inline citations:"
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
        except Exception as exc:
            print(f"Gemini generate_answer error: {exc}")
            return self._offline_answer(context=context, query=query)

    def _offline_answer(self, context: str, query: str) -> str:
        blocks = self._parse_context_blocks(context)
        if not blocks:
            return (
                "Toi khong the xac minh thong tin nay tu cac tai lieu hien co."
            )

        bullets = []
        for source, content in blocks[:2]:
            snippet = self._first_sentence(content)
            bullets.append(f"- {snippet} [{source}]")

        return (
            "Tra loi du thao offline cho cau hoi: "
            f"{query}\n\n"
            + "\n".join(bullets)
        )

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        model = model_name.removeprefix("models/")
        if model == "gemini-3.5-flash-lite":
            return "gemini-3.5-flash"
        return model

    @staticmethod
    def _parse_context_blocks(context: str) -> list[tuple[str, str]]:
        blocks: list[tuple[str, str]] = []
        pattern = re.compile(r"Tai lieu \[(?P<source>[^\]]+)\]:\n(?P<body>.*?)(?=\n\nTai lieu \[|\Z)", re.S)
        for match in pattern.finditer(context):
            source = match.group("source").strip()
            body = match.group("body").strip()
            if body:
                blocks.append((source, body))
        return blocks

    @staticmethod
    def _first_sentence(text: str, limit: int = 320) -> str:
        cleaned = " ".join(text.split())
        for separator in [". ", "; ", "\n"]:
            if separator in cleaned:
                cleaned = cleaned.split(separator, 1)[0]
                break
        if len(cleaned) > limit:
            cleaned = cleaned[:limit].rstrip() + "..."
        return cleaned
