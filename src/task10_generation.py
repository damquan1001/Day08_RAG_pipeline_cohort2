"""Task 10 - Generation with citation."""

from __future__ import annotations

from .local_retrieval import citation_label
from .task9_retrieval_pipeline import retrieve


TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer in Vietnamese using only the provided context.
Every factual claim should include a citation in [Source, Year] format.
If the context is insufficient, say that the information cannot be verified."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to reduce lost-in-the-middle.

    For scores sorted descending [1, 2, 3, 4, 5], return [1, 3, 5, 4, 2].
    """
    if len(chunks) <= 2:
        return list(chunks)

    reordered: list[dict] = []
    reordered.extend(chunks[0::2])
    reordered.extend(reversed(chunks[1::2]))
    return reordered


def format_context(chunks: list[dict]) -> str:
    """Format chunks with source labels for citation."""
    parts: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {index}")
        doc_type = metadata.get("doc_type") or metadata.get("type", "unknown")
        parts.append(
            f"[Document {index} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk.get('content', '')}"
        )
    return "\n---\n".join(parts)


def _snippet(text: str, limit: int = 420) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rsplit(" ", 1)[0] + "..."


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    Offline RAG generation with citations.

    Returns:
        {'answer': str, 'sources': list[dict], 'retrieval_source': str}
    """
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)

    if not reordered:
        return {
            "answer": "Toi khong the xac minh thong tin nay tu nguon hien co.",
            "sources": [],
            "retrieval_source": "none",
        }

    primary = reordered[0]
    label = citation_label(primary.get("metadata", {}))
    answer = (
        "Dua tren nguon duoc truy xuat, noi dung lien quan nhat cho cau hoi "
        f"'{query}' la: {_snippet(primary.get('content', ''))} [{label}]."
    )

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


if __name__ == "__main__":
    print(generate_with_citation("Hinh phat tang tru ma tuy?")["answer"])
