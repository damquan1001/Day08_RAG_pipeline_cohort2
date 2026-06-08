"""Task 8 - PageIndex-compatible vectorless fallback."""

import os
from pathlib import Path

from dotenv import load_dotenv

from .task6_lexical_search import lexical_search


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Return local upload metadata.

    A real PageIndex deployment would upload files with PAGEINDEX_API_KEY. The
    offline version exposes the same step without requiring an external account.
    """
    return [
        {"filename": path.name, "path": str(path.relative_to(STANDARDIZED_DIR))}
        for path in sorted(STANDARDIZED_DIR.rglob("*.md"))
    ]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless fallback search marked with source='pageindex'.
    """
    results = lexical_search(query, top_k=top_k)
    pageindex_results = []

    for item in results:
        pageindex_results.append(
            {
                "content": item["content"],
                "score": float(item.get("score", 0.0)),
                "metadata": item.get("metadata", {}),
                "source": "pageindex",
            }
        )

    return pageindex_results[:top_k]


if __name__ == "__main__":
    print(pageindex_search("ma tuy", top_k=3))
