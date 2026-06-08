"""Task 9 - Complete retrieval pipeline."""

from __future__ import annotations

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "heuristic"


def _with_source(results: list[dict], source: str) -> list[dict]:
    output: list[dict] = []
    for item in results:
        copied = {
            "content": item.get("content", ""),
            "score": float(item.get("score", 0.0)),
            "metadata": dict(item.get("metadata", {})),
            "source": source,
        }
        output.append(copied)
    return output


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Run semantic + lexical retrieval, merge with RRF, rerank, then fallback.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict,
        'source': 'hybrid'|'pageindex'}
    """
    if top_k <= 0:
        return []

    try:
        dense_results = semantic_search(query, top_k=top_k * 2)
    except Exception:
        dense_results = []

    try:
        sparse_results = lexical_search(query, top_k=top_k * 2)
    except Exception:
        sparse_results = []

    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    merged = _with_source(merged, "hybrid")

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        final_results = _with_source(final_results, "hybrid")
    else:
        final_results = merged[:top_k]

    best_score = final_results[0]["score"] if final_results else 0.0
    if best_score < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        return _with_source(fallback, "pageindex")[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    for item in retrieve("hinh phat ma tuy", top_k=3):
        print(f"[{item['score']:.3f}] [{item['source']}] {item['metadata'].get('source')}")
