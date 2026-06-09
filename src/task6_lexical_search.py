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
    """Build the token statistics needed for BM25 scoring."""
    tokenized = [tokenize(doc.get("content", "")) for doc in corpus]
    doc_freq = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))

    avg_doc_len = (
        sum(len(tokens) for tokens in tokenized) / len(tokenized) if tokenized else 0
    )
    return {
        "tokenized": tokenized,
        "doc_freq": doc_freq,
        "avg_doc_len": avg_doc_len,
        "doc_count": len(tokenized),
    }


def _bm25_score(query_tokens: list[str], doc_tokens: list[str], index: dict) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    k1 = 1.5
    b = 0.75
    doc_count = index["doc_count"]
    avg_doc_len = index["avg_doc_len"] or 1
    frequencies = Counter(doc_tokens)
    score = 0.0

    for token in query_tokens:
        term_freq = frequencies.get(token, 0)
        if term_freq == 0:
            continue
        doc_freq = index["doc_freq"].get(token, 0)
        idf = math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
        denominator = term_freq + k1 * (1 - b + b * len(doc_tokens) / avg_doc_len)
        score += idf * (term_freq * (k1 + 1)) / denominator

    return score


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Search chunks with BM25 and return sorted results."""
    if top_k <= 0:
        return []

    corpus = _load_corpus()
    index = build_bm25_index(corpus)
    query_tokens = tokenize(query)

    scores = [
        _bm25_score(query_tokens, doc_tokens, index)
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
