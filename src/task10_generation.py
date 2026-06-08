"""Task 10 - Generation with citations from retrieved chunks."""

import re
from urllib.parse import urlparse

from .task9_retrieval_pipeline import retrieve


TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer in Vietnamese using only the provided context.
Every factual claim should include a citation in this exact format:
[Author/Platform Name, Year]."""


PLATFORM_NAMES = {
    "znews.vn": "ZNews",
    "lifestyle.znews.vn": "ZNews",
    "tienphong.vn": "Tien Phong",
    "tpo": "Tien Phong",
    "vnexpress.net": "VnExpress",
    "tuoitre.vn": "Tuoi Tre",
    "thanhnien.vn": "Thanh Nien",
    "dantri.com.vn": "Dan Tri",
    "plo.vn": "PLO",
    "congan.com.vn": "Cong An",
}


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


def _extract_field(content: str, field_name: str) -> str:
    pattern = rf"^\*\*{re.escape(field_name)}:\*\*\s*(.+)$"
    match = re.search(pattern, content, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def _platform_from_url(url: str) -> str:
    if not url:
        return ""
    host = urlparse(url).netloc.lower().removeprefix("www.")
    if host in PLATFORM_NAMES:
        return PLATFORM_NAMES[host]
    domain = ".".join(host.split(".")[-2:])
    if domain in PLATFORM_NAMES:
        return PLATFORM_NAMES[domain]
    if host:
        return host.split(".")[0].replace("-", " ").title()
    return ""


def _platform_from_source(source: str) -> str:
    name = re.sub(r"\.[a-z0-9]+$", "", source, flags=re.IGNORECASE)
    name = name.replace("-", " ").replace("_", " ").strip()
    return name.title() if name else "Unknown Source"


def _year_from_text(*values: str) -> str:
    for value in values:
        match = re.search(r"\b(19|20)\d{2}\b", value or "")
        if match:
            return match.group(0)
    return "n.d."


def _citation_label(chunk: dict, index: int) -> str:
    """Return citation text in [Author/Platform Name, Year] format."""
    metadata = chunk.get("metadata", {})
    content = chunk.get("content", "")
    source_url = metadata.get("source_url") or _extract_field(content, "Source")
    crawled = metadata.get("crawled_at") or _extract_field(content, "Crawled")
    source = metadata.get("source") or metadata.get("path") or f"Source {index}"

    platform = _platform_from_url(source_url) or _platform_from_source(str(source))
    year = _year_from_text(crawled, source_url, str(source), content)
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


def _extractive_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Toi khong the xac minh thong tin nay tu nguon hien co."

    sentences = []
    for index, chunk in enumerate(chunks[:3], start=1):
        content = " ".join(chunk.get("content", "").split())
        snippet = content[:260].strip()
        if len(content) > 260:
            snippet = snippet.rsplit(" ", 1)[0] + "..."
        citation = _citation_label(chunk, index)
        sentences.append(f"{snippet} [{citation}]")

    return (
        f"Duoi day la cau tra loi dua tren cac nguon truy xuat cho cau hoi "
        f"'{query}':\n\n" + "\n\n".join(sentences)
    )


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    Retrieve evidence, reorder it, format context, and return a cited answer.
    """
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    answer = _extractive_answer(query, reordered)

    return {
        "answer": answer,
        "sources": reordered,
        "context": context,
        "retrieval_source": reordered[0].get("source", "none") if reordered else "none",
    }


if __name__ == "__main__":
    result = generate_with_citation("Hinh phat tang tru ma tuy?")
    print(result["answer"])
