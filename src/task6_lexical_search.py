"""Task 6 - Lexical search with a pure-Python BM25 implementation."""

from __future__ import annotations

from .local_retrieval import bm25_rank, get_chunks
from .task4_chunking_indexing import CHUNK_OVERLAP, CHUNK_SIZE


def build_bm25_index(corpus: list[dict]) -> list[dict]:
    """Return corpus as the lightweight reusable BM25 index."""
    return corpus


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Return BM25 lexical search results sorted by score descending.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
    """
    chunks = get_chunks(CHUNK_SIZE, CHUNK_OVERLAP)
    return bm25_rank(query, chunks, top_k, include_zero=False)


if __name__ == "__main__":
    for item in lexical_search("Dieu 248 ma tuy", top_k=5):
        print(f"[{item['score']:.3f}] {item['metadata'].get('source')}")
