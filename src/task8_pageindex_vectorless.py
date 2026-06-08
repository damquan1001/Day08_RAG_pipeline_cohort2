"""Task 8 - PageIndex vectorless fallback.

If PAGEINDEX_API_KEY and the SDK are available, this file can be extended to
call the real service.  For the individual assignment we provide a local
vectorless fallback over the markdown corpus with the required return shape.
"""

from __future__ import annotations

from .local_retrieval import bm25_rank, get_chunks, semantic_rank
from .task4_chunking_indexing import CHUNK_OVERLAP, CHUNK_SIZE


def upload_documents():
    """Return local document count for offline mode."""
    return len(get_chunks(CHUNK_SIZE, CHUNK_OVERLAP))


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval fallback.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict,
        'source': 'pageindex'}
    """
    chunks = get_chunks(CHUNK_SIZE, CHUNK_OVERLAP)
    results = bm25_rank(query, chunks, top_k, include_zero=False)
    if not results:
        results = semantic_rank(query, chunks, top_k)

    for item in results:
        item["source"] = "pageindex"
    return results[:top_k]


if __name__ == "__main__":
    for item in pageindex_search("ma tuy", top_k=3):
        print(f"[{item['score']:.3f}] {item['metadata'].get('source')}")
