"""Task 10 - OpenAI generation with citations."""

import os
from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve
from .text_utils import (
    markdown_field,
    platform_from_source,
    platform_from_url,
    year_from_text,
)


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """Answer in Vietnamese using only the provided context.
Every factual claim should include a citation in this exact format:
[Author/Platform Name, Year]."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to reduce lost-in-the-middle effects.

    The highest-scoring chunk remains first; remaining chunks alternate toward
    the end so strong evidence appears near prompt edges.
    """
    if len(chunks) <= 2:
        return chunks[:]

    reordered = [chunks[0]]
    middle = []
    tail = []
    for index, chunk in enumerate(chunks[1:], start=1):
        if index % 2:
            tail.append(chunk)
        else:
            middle.append(chunk)
    return reordered + middle + list(reversed(tail))


def _citation_label(chunk: dict, index: int) -> str:
    """Return citation text in [Author/Platform Name, Year] format."""
    metadata = chunk.get("metadata", {})
    content = chunk.get("content", "")
    source_url = metadata.get("source_url") or markdown_field(content, "Source")
    crawled = metadata.get("crawled_at") or markdown_field(content, "Crawled")
    source = metadata.get("source") or metadata.get("path") or f"Source {index}"

    platform = platform_from_url(source_url) or platform_from_source(str(source))
    year = year_from_text(crawled, source_url, str(source), content)
    return f"{platform}, {year}"


def format_context(chunks: list[dict]) -> str:
    """Format chunks with source labels for citation-aware prompting."""
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        label = _citation_label(chunk, index)
        raw_source = (
            chunk.get("metadata", {}).get("source")
            or chunk.get("metadata", {}).get("path")
            or f"Source {index}"
        )
        parts.append(
            f"[Document {index} | Source: {raw_source} | Citation: [{label}] | "
            f"Score: {chunk.get('score', 0):.3f}]\n"
            f"{chunk.get('content', '').strip()}"
        )
    return "\n\n---\n\n".join(parts)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Retrieve context and call OpenAI to generate a cited answer."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for Task 10 generation.")

    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    if not reordered:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "context": context,
            "retrieval_source": "none",
        }

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for Task 10 generation.") from exc

    client = OpenAI(api_key=api_key)
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    answer = response.choices[0].message.content or ""

    return {
        "answer": answer,
        "sources": reordered,
        "context": context,
        "retrieval_source": reordered[0].get("source", "none") if reordered else "none",
    }


if __name__ == "__main__":
    result = generate_with_citation("Hinh phat tang tru ma tuy?")
    print(result["answer"])
