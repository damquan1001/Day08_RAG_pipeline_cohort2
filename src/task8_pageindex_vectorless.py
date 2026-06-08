"""Task 8 - PageIndex vectorless RAG integration."""

import os
from pathlib import Path

from dotenv import load_dotenv

from .task6_lexical_search import lexical_search


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _require_pageindex():
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is required for Task 8.")
    try:
        from pageindex import PageIndex
    except ImportError as exc:
        raise RuntimeError("pageindex package is required for Task 8.") from exc
    return PageIndex(api_key=PAGEINDEX_API_KEY)


def upload_documents():
    """Upload all standardized Markdown documents to PageIndex."""
    client = _require_pageindex()
    uploaded = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        metadata = {
            "filename": md_file.name,
            "path": str(md_file.relative_to(STANDARDIZED_DIR)).replace("\\", "/"),
            "type": md_file.parent.name,
        }
        result = client.upload(content=content, metadata=metadata)
        uploaded.append(result)
        print(f"Uploaded: {md_file.name}")

    return uploaded


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Query PageIndex and mark results as source='pageindex'."""
    client = _require_pageindex()
    raw_results = client.query(query=query, top_k=top_k)

    results = []
    for result in raw_results:
        content = getattr(result, "text", None) or getattr(result, "content", None) or ""
        score = getattr(result, "score", 0.0)
        metadata = getattr(result, "metadata", {}) or {}
        results.append(
            {
                "content": content,
                "score": float(score),
                "metadata": metadata,
                "source": "pageindex",
            }
        )

    return results[:top_k]


if __name__ == "__main__":
    for item in pageindex_search("ma tuy", top_k=3):
        print(f"[{item['score']:.3f}] {item['content'][:100]}...")
