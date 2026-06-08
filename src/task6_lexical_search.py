"""Task 6 - BM25 lexical search."""

import math
from collections import Counter

from .task4_chunking_indexing import chunk_documents, load_documents
from .text_utils import tokenize


CORPUS: list[dict] = []


def _load_corpus() -> list[dict]:
    global CORPUS
    if not CORPUS:
        CORPUS = chunk_documents(load_documents())
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """Build a BM25 index using rank-bm25 when available."""
    tokenized = [tokenize(doc.get("content", "")) for doc in corpus]
    try:
        from rank_bm25 import BM25Okapi

        return {"backend": "rank_bm25", "bm25": BM25Okapi(tokenized), "tokenized": tokenized}
    except ImportError:
        doc_freq = Counter()
        for tokens in tokenized:
            doc_freq.update(set(tokens))
        avg_len = sum(len(tokens) for tokens in tokenized) / len(tokenized) if tokenized else 0
        return {
            "backend": "local",
            "tokenized": tokenized,
            "doc_freq": doc_freq,
            "doc_count": len(tokenized),
            "avg_len": avg_len,
        }


def _local_score(query_tokens: list[str], doc_tokens: list[str], index: dict) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    k1 = 1.5
    b = 0.75
    frequencies = Counter(doc_tokens)
    avg_len = index["avg_len"] or 1
    score = 0.0
    for token in query_tokens:
        tf = frequencies.get(token, 0)
        if tf == 0:
            continue
        df = index["doc_freq"].get(token, 0)
        idf = math.log(1 + (index["doc_count"] - df + 0.5) / (df + 0.5))
        denom = tf + k1 * (1 - b + b * len(doc_tokens) / avg_len)
        score += idf * (tf * (k1 + 1)) / denom
    return score


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Search chunks with BM25 and return sorted results."""
    if top_k <= 0:
        return []

    corpus = _load_corpus()
    index = build_bm25_index(corpus)
    query_tokens = tokenize(query)

    if index["backend"] == "rank_bm25":
        scores = index["bm25"].get_scores(query_tokens)
    else:
        scores = [
            _local_score(query_tokens, doc_tokens, index)
            for doc_tokens in index["tokenized"]
        ]

    results = []
    for idx, score in enumerate(scores):
        if score <= 0:
            continue
        results.append(
            {
                "content": corpus[idx]["content"],
                "score": float(score),
                "metadata": corpus[idx].get("metadata", {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in lexical_search("Dieu 248 ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
