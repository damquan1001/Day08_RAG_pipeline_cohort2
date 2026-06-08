"""Task 5 - Local semantic search over Task 4 chunks."""

from collections import Counter

from .task4_chunking_indexing import chunk_documents, load_documents
from .text_utils import cosine_from_counters, tokenize


def _load_corpus() -> list[dict]:
    return chunk_documents(load_documents())


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search chunks with normalized token-vector cosine similarity.

    This is a local stand-in for dense retrieval: it is deterministic, offline,
    sorted by descending score, and uses the same Task 4 chunks.
    """
    if top_k <= 0:
        return []

    query_vector = Counter(tokenize(query))
    if not query_vector:
        return []

    results = []
    for chunk in _load_corpus():
        chunk_vector = Counter(tokenize(chunk.get("content", "")))
        score = cosine_from_counters(query_vector, chunk_vector)
        if score <= 0:
            continue
        results.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("hinh phat ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
