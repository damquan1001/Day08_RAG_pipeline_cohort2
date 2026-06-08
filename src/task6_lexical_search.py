"""
Task 6 - Lexical Search Module (BM25).
"""

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.task4_chunking_indexing import chunk_documents, load_documents
else:
    from .task4_chunking_indexing import chunk_documents, load_documents


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


@lru_cache(maxsize=1)
def _bm25_index():
    corpus = chunk_documents(load_documents())
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    if not corpus:
        return [], None
    return corpus, BM25Okapi(tokenized_corpus)


def build_bm25_index(corpus: list[dict]):
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa bằng BM25.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sorted by score descending.
    """
    if top_k <= 0:
        return []

    corpus, bm25 = _bm25_index()
    if not corpus or bm25 is None:
        return []

    scores = bm25.get_scores(_tokenize(query))
    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        doc = corpus[int(idx)]
        results.append(
            {
                "content": doc["content"],
                "score": float(scores[idx]),
                "metadata": doc.get("metadata", {}),
            }
        )
    return results


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma túy", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
