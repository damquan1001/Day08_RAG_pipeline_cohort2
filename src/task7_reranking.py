"""
Task 7 - Reranking Module.

Dùng reranker local dựa trên lexical overlap + score ban đầu để không cần API
key. Module vẫn cung cấp MMR và RRF theo yêu cầu bài.
"""

import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _cosine_counter(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(left[t] * right[t] for t in common)
    left_norm = math.sqrt(sum(v * v for v in left.values()))
    right_norm = math.sqrt(sum(v * v for v in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _text_similarity(query: str, content: str) -> float:
    return _cosine_counter(Counter(_tokenize(query)), Counter(_tokenize(content)))


def _cosine_vector(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Local reranker: kết hợp độ khớp query-content và score retrieval ban đầu.
    """
    scored = []
    for candidate in candidates:
        base_score = float(candidate.get("score", 0.0))
        lexical_score = _text_similarity(query, candidate.get("content", ""))
        item = candidate.copy()
        item["score"] = float(0.7 * lexical_score + 0.3 * base_score)
        scored.append(item)

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[: max(top_k, 0)]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            candidate_embedding = candidates[idx].get("embedding", [])
            relevance = _cosine_vector(query_embedding, candidate_embedding)
            diversity_penalty = 0.0
            for selected_idx in selected:
                diversity_penalty = max(
                    diversity_penalty,
                    _cosine_vector(candidate_embedding, candidates[selected_idx].get("embedding", [])),
                )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break
        item = candidates[best_idx].copy()
        item["score"] = float(best_score)
        candidates[best_idx] = item
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("content", "")
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)
    results = []
    for content, score in sorted_items[: max(top_k, 0)]:
        item = content_map[content].copy()
        item["score"] = float(score)
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    if not candidates or top_k <= 0:
        return []
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    if method == "mmr":
        return rerank_cross_encoder(query, candidates, top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma túy", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ bị bắt vì sử dụng ma túy", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    for r in rerank("hình phạt tàng trữ ma túy", dummy_candidates, top_k=2):
        print(f"[{r['score']:.3f}] {r['content']}")
