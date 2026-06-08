"""
Task 5 - Semantic Search Module.

Triển khai local, không cần API key. Nếu chưa có vector store embedding từ
Task 4, module dùng TF-IDF cosine similarity như một dense-search fallback ổn
định cho môi trường bài tập.
"""

from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .task4_chunking_indexing import chunk_documents, load_documents


@lru_cache(maxsize=1)
def _semantic_index():
    docs = load_documents()
    chunks = chunk_documents(docs)
    if not chunks:
        return [], None, None

    texts = [chunk["content"] for chunk in chunks]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=20000,
    )
    matrix = vectorizer.fit_transform(texts)
    return chunks, vectorizer, matrix


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa/fallback dense retrieval trên corpus markdown.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sorted by score descending.
    """
    if top_k <= 0:
        return []

    chunks, vectorizer, matrix = _semantic_index()
    if not chunks or vectorizer is None or matrix is None:
        return []

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        chunk = chunks[int(idx)]
        results.append(
            {
                "content": chunk["content"],
                "score": float(scores[idx]),
                "metadata": chunk.get("metadata", {}),
            }
        )
    return results


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma túy", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
