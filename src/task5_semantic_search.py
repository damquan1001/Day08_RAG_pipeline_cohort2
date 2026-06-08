"""Task 5 - Semantic search.

The production version can swap this module for ChromaDB +
sentence-transformers.  For the individual pipeline we use local TF-IDF cosine
similarity so tests and demos run without model downloads.
"""

from __future__ import annotations

from .local_retrieval import get_chunks, semantic_rank
from .task4_chunking_indexing import CHUNK_OVERLAP, CHUNK_SIZE


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Return semantic-style search results sorted by score descending.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
    """
    chunks = get_chunks(CHUNK_SIZE, CHUNK_OVERLAP)
    return semantic_rank(query, chunks, top_k)


if __name__ == "__main__":
    for item in semantic_search("hinh phat ma tuy", top_k=5):
        print(f"[{item['score']:.3f}] {item['metadata'].get('source')}")
