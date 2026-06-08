"""
Task 10 - Generation Có Citation.

Uses OpenAI Chat Completions when OPENAI_API_KEY has usable quota. If the API is
unavailable, it falls back to deterministic local citation generation.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.task9_retrieval_pipeline import retrieve
else:
    from .task9_retrieval_pipeline import retrieve

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """Trả lời bằng tiếng Việt, chỉ dùng context được cung cấp.
Mỗi nhận định thực tế cần có citation theo nguồn trong context, ví dụ [article_01.md].
Nếu không đủ bằng chứng, nói rõ: "Tôi không thể xác minh thông tin này từ nguồn hiện có".
Không bịa nguồn, không dùng kiến thức ngoài context."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Put the strongest chunk first and move high-value chunks toward the end to
    reduce lost-in-the-middle.
    """
    if len(chunks) <= 2:
        return chunks

    reordered = []
    for index in range(0, len(chunks), 2):
        reordered.append(chunks[index])
    start = len(chunks) - 1 if len(chunks) % 2 == 0 else len(chunks) - 2
    for index in range(start, 0, -2):
        reordered.append(chunks[index])
    return reordered


def format_context(chunks: list[dict]) -> str:
    context_parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"source_{index}")
        doc_type = metadata.get("type", "unknown")
        score = float(chunk.get("score", 0.0))
        context_parts.append(
            f"[Document {index} | Source: {source} | Type: {doc_type} | Score: {score:.3f}]\n"
            f"{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(context_parts)


def _first_sentence(text: str) -> str:
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip()
        if len(line) < 30:
            continue
        if line.startswith("|") or set(line) <= {"|", "-", " "}:
            continue
        if not any(ch.isalpha() for ch in line):
            continue
        normalized = line
        break
    else:
        normalized = " ".join(text.split())

    for separator in [". ", "? ", "! ", "\n"]:
        if separator in normalized:
            return normalized.split(separator, 1)[0].strip() + separator.strip()
    return normalized[:300].strip()


def _local_generate(query: str, chunks: list[dict], reordered: list[dict]) -> str:
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    answer_parts = []
    for chunk in reordered:
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "nguồn không rõ")
        sentence = _first_sentence(chunk.get("content", ""))
        if sentence and any(ch.isalpha() for ch in sentence):
            answer_parts.append(f"{sentence} [{source}]")
        if len(answer_parts) >= 3:
            break

    return " ".join(answer_parts).strip() or "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def _openai_generate(query: str, context: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    user_message = f"""Context:
{context}

---

Question: {query}"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=700,
    )
    return response.choices[0].message.content or ""


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
            "generation_source": "none",
        }

    try:
        answer = _openai_generate(query, context)
        generation_source = f"openai:{OPENAI_MODEL}"
        if not answer.strip():
            raise RuntimeError("OpenAI returned an empty answer")
    except Exception:
        answer = _local_generate(query, chunks, reordered)
        generation_source = "local_fallback"

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "generation_source": generation_source,
    }


if __name__ == "__main__":
    result = generate_with_citation("Hình phạt cho tội tàng trữ trái phép chất ma túy?")
    print(f"[generation_source={result['generation_source']}]")
    print(result["answer"])
