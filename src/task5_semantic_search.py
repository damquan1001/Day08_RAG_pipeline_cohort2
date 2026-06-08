"""
Task 5 - Semantic Search Module.

Triển khai local, không cần API key. Nếu chưa có vector store embedding từ
Task 4, module dùng TF-IDF cosine similarity như một dense-search fallback ổn
định cho môi trường bài tập.
"""

import sys
from functools import lru_cache
import os
from pathlib import Path
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.task4_chunking_indexing import chunk_documents, load_documents, EMBEDDING_MODEL
else:
    from .task4_chunking_indexing import chunk_documents, load_documents, EMBEDDING_MODEL

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")


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


def _local_semantic_search_fallback(query: str, top_k: int = 10) -> list[dict]:
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


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa trên Weaviate Cloud (với fallback local TF-IDF).

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sorted by score descending.
    """
    if top_k <= 0:
        return []

    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_key = os.getenv("WEAVIATE_API_KEY")

    if not weaviate_url or not weaviate_key:
        print("[Warning] Weaviate Cloud credentials missing. Using local TF-IDF search fallback.")
        return _local_semantic_search_fallback(query, top_k)

    import weaviate
    from weaviate.classes.init import Auth
    from weaviate.classes.query import MetadataQuery
    from sentence_transformers import SentenceTransformer

    try:
        # Load embedding model to embed query
        # Since EMBEDDING_MODEL is configured, we load it. SentenceTransformer is cached.
        model = SentenceTransformer(EMBEDDING_MODEL)
        query_vector = model.encode(query).tolist()

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_key)
        )
    except Exception as e:
        print(f"[Warning] Failed to initialize Weaviate/Embedding client: {e}. Falling back to local TF-IDF.")
        return _local_semantic_search_fallback(query, top_k)

    try:
        class_name = "DrugLawDocs"
        if not client.collections.exists(class_name):
            print(f"[Warning] Collection '{class_name}' does not exist on Weaviate Cloud. Falling back to local TF-IDF.")
            return _local_semantic_search_fallback(query, top_k)

        collection = client.collections.get(class_name)
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True, certainty=True)
        )

        results = []
        for obj in response.objects:
            # Weaviate distance to similarity score mapping
            # similarity = 1 - distance (since distance = 1 - cosine_similarity for cosine)
            score = 1.0
            if obj.metadata:
                if obj.metadata.certainty is not None:
                    score = float(obj.metadata.certainty)
                elif obj.metadata.distance is not None:
                    score = float(1.0 - obj.metadata.distance)

            results.append({
                "content": obj.properties.get("content", ""),
                "score": score,
                "metadata": {
                    "source": obj.properties.get("source", "Unknown"),
                    "type": obj.properties.get("doc_type", "Unknown")
                }
            })
        
        # Weaviate already returns near_vector sorted by score descending, but let's make sure
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    except Exception as e:
        print(f"[Warning] Weaviate query failed: {e}. Falling back to local TF-IDF.")
        return _local_semantic_search_fallback(query, top_k)
    finally:
        try:
            client.close()
        except:
            pass


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma túy", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
