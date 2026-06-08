from typing import List
from system_contracts import Document


class Reranker:
    """Handles different re-ranking strategies to prioritize key evidence."""

    @staticmethod
    def rerank(
        query: str,
        candidates: List[Document],
        method: str = "mmr",
        top_k: int = 5,
    ) -> List[Document]:
        """
        Rerank document candidates using selected strategy (MMR, Cross-Encoder, or RRF).

        Args:
            query: The standalone query string.
            candidates: List of Document retrieved from search database.
            method: Re-ranking strategy name.
            top_k: Top K results to select.
        """
        # Skeleton returns first top_k elements from candidates
        return candidates[:top_k]

    @staticmethod
    def reorder_for_llm(documents: List[Document]) -> List[Document]:
        """
        Reorder documents to place most relevant at start and end of prompt.
        Avoids the 'Lost in the middle' effect.
        """
        if len(documents) <= 2:
            return documents

        reordered: List[Document] = []
        for i in range(0, len(documents), 2):
            reordered.append(documents[i])
        for i in range(len(documents) - 1 - (len(documents) % 2 == 0), 0, -2):
            reordered.append(documents[i])
        return reordered
