import math
from typing import List
from system_contracts import Document
from src.module_rag_core.ports.outbound import RerankerPort


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate the cosine similarity between two vector embeddings."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_prod = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_prod / (mag1 * mag2)


class MMRRerankerAdapter(RerankerPort):
    """Outbound adapter implementing MMR reranking and document reordering."""

    def rerank(
        self,
        query_vector: List[float],
        candidates: List[Document],
        lambda_param: float = 0.7,
        top_k: int = 5,
    ) -> List[Document]:
        """
        Rerank document candidates using Maximal Marginal Relevance (MMR)
        to optimize relevance and diversify information.
        """
        if not candidates:
            return []

        selected_docs: List[Document] = []
        remaining_indices = list(range(len(candidates)))

        for _ in range(min(top_k, len(candidates))):
            best_score = float("-inf")
            best_idx = -1

            for idx in remaining_indices:
                doc = candidates[idx]
                doc_vector = doc.metadata.get("vector")

                # Relevance term
                if doc_vector and query_vector:
                    relevance = cosine_similarity(query_vector, doc_vector)
                else:
                    relevance = doc.score if doc.score is not None else 0.0

                # Diversity/Redundancy term
                max_sim_to_selected = 0.0
                if selected_docs:
                    similarities = []
                    for sel_doc in selected_docs:
                        sel_vector = sel_doc.metadata.get("vector")
                        if doc_vector and sel_vector:
                            similarities.append(
                                cosine_similarity(doc_vector, sel_vector)
                            )
                        else:
                            similarities.append(0.0)
                    max_sim_to_selected = max(similarities)

                # MMR score formula
                mmr_score = (
                    lambda_param * relevance
                    - (1.0 - lambda_param) * max_sim_to_selected
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx == -1:
                break

            selected_docs.append(candidates[best_idx])
            remaining_indices.remove(best_idx)

        return selected_docs

    def reorder_for_llm(self, documents: List[Document]) -> List[Document]:
        """
        Reorder documents to prevent "lost in the middle" effect.
        Places most important documents at the beginning and the end.
        """
        if len(documents) <= 2:
            return documents

        reordered: List[Document] = []
        # Odd indexed elements (higher rank) go first
        for i in range(0, len(documents), 2):
            reordered.append(documents[i])
        # Even indexed elements go last in reverse order
        for i in range(len(documents) - 1 - (len(documents) % 2 == 0), 0, -2):
            reordered.append(documents[i])
        return reordered
