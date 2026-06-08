"""Task 7 - Reranking utilities."""

from collections import Counter

from .text_utils import content_key, cosine_from_counters, tokenize


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank candidates with a deterministic local relevance scorer."""
    if top_k <= 0:
        return []

    query_terms = set(tokenize(query))
    query_vector = Counter(tokenize(query))
    reranked = []

    for candidate in candidates:
        doc_tokens = tokenize(candidate.get("content", ""))
        doc_terms = set(doc_tokens)
        overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
        cosine = cosine_from_counters(query_vector, Counter(doc_tokens))
        prior = float(candidate.get("score", 0.0))
        score = 0.55 * overlap + 0.35 * cosine + 0.10 * prior
        item = candidate.copy()
        item["score"] = float(score)
        reranked.append(item)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Select relevant and diverse results using MMR over text similarity."""
    selected = []
    remaining = candidates[:]

    while remaining and len(selected) < top_k:
        best_item = None
        best_score = float("-inf")
        for item in remaining:
            relevance = float(item.get("score", 0.0))
            item_terms = Counter(tokenize(item.get("content", "")))
            diversity_penalty = 0.0
            for chosen in selected:
                chosen_terms = Counter(tokenize(chosen.get("content", "")))
                diversity_penalty = max(
                    diversity_penalty, cosine_from_counters(item_terms, chosen_terms)
                )
            score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if score > best_score:
                best_item = item
                best_score = score
        remaining.remove(best_item)
        selected.append({**best_item, "score": float(best_score)})

    return selected


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion."""
    scores = {}
    items = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = content_key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items[key] = item

    merged = []
    for key, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True):
        item = items[key].copy()
        item["score"] = float(score)
        merged.append(item)
    return merged[:top_k]


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """Unified reranking interface."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        return rerank_mmr([], candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy = [
        {"content": "Dieu 248 ma tuy", "score": 0.8, "metadata": {}},
        {"content": "Python programming", "score": 0.4, "metadata": {}},
    ]
    print(rerank("hinh phat ma tuy", dummy, top_k=2))
