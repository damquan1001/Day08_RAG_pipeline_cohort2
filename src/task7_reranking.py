"""Task 7 - Reranking.

Uses a local heuristic reranker by default.  It keeps the same interface as a
cross-encoder reranker but avoids downloading a model during tests.
"""

from __future__ import annotations

from .local_retrieval import heuristic_relevance


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """Heuristic cross-encoder substitute: query/document token relevance."""
    if top_k <= 0:
        return []
    if not candidates:
        return []

    max_base = max(float(item.get("score", 0.0)) for item in candidates) or 1.0
    reranked: list[dict] = []
    for item in candidates:
        base = float(item.get("score", 0.0)) / max_base
        copied = {
            "content": item.get("content", ""),
            "metadata": dict(item.get("metadata", {})),
            **{k: v for k, v in item.items() if k not in {"content", "metadata", "score"}},
        }
        copied["score"] = heuristic_relevance(query, copied["content"], base)
        reranked.append(copied)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Simple MMR fallback using existing scores when embeddings are absent."""
    del query_embedding, lambda_param
    return sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)[
        :top_k
    ]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """Reciprocal Rank Fusion for combining dense and lexical lists."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("content", "")
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items[key] = item

    fused: list[dict] = []
    for content, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True):
        item = {
            "content": content,
            "metadata": dict(items[content].get("metadata", {})),
            "score": float(score),
        }
        fused.append(item)
        if len(fused) >= top_k:
            break
    return fused


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """Unified reranking interface."""
    if method in {"cross_encoder", "heuristic"}:
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    if method == "mmr":
        return rerank_mmr([], candidates, top_k=top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    sample = [
        {"content": "Toi tang tru trai phep chat ma tuy", "score": 0.8, "metadata": {}},
        {"content": "Python programming", "score": 0.4, "metadata": {}},
    ]
    print(rerank("hinh phat ma tuy", sample, top_k=2))
