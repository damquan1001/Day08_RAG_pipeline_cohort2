import urllib.parse
from typing import List, Optional
import weaviate
from weaviate.classes.query import MetadataQuery
from system_contracts import Document
from src.module_rag_core.ports.outbound import VectorStorePort


class WeaviateDockerAdapter(VectorStorePort):
    """Outbound adapter connecting to local Weaviate running on Docker."""

    def __init__(self) -> None:
        self.client: Optional[weaviate.WeaviateClient] = None
        self.class_name = "DrugLawDocs"

    def connect(self, url: str, api_key: Optional[str] = None) -> None:
        """Establish connection to local or Cloud Weaviate instance."""
        try:
            if "weaviate.cloud" in url or "weaviate.network" in url:
                from weaviate.classes.init import Auth
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=url,
                    auth_credentials=Auth.api_key(api_key) if api_key else None
                )
            else:
                parsed = urllib.parse.urlparse(url)
                host = parsed.hostname or "localhost"
                port = parsed.port or 8080
                # connect_to_local connects to HTTP port and automatically determines gRPC port
                self.client = weaviate.connect_to_local(host=host, port=port)
        except Exception as e:
            print(
                f"Warning: Could not connect to Weaviate at {url} ({e}). "
                "Will use fallback mock data for testing."
            )
            self.client = None

    def hybrid_search(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> List[Document]:
        """
        Query Weaviate using Hybrid Search (BM25 + Vector).
        If Weaviate is not connected or queries fail, falls back to mock results.
        """
        if not self.client:
            return self._get_fallback_docs()

        try:
            collection = self.client.collections.get(self.class_name)
            response = collection.query.hybrid(
                query=query,
                vector=vector,
                alpha=alpha,
                limit=top_k,
                return_metadata=MetadataQuery(score=True),
                include_vector=True,
            )

            docs = []
            for obj in response.objects:
                doc_vector = None
                if obj.vector:
                    if isinstance(obj.vector, dict):
                        doc_vector = obj.vector.get("default")
                    else:
                        doc_vector = obj.vector

                metadata = {
                    "source": obj.properties.get("source", "Unknown"),
                    "type": obj.properties.get("doc_type", "Unknown"),
                    "vector": doc_vector,
                }

                # Copy any other metadata properties
                for key, val in obj.properties.items():
                    if key not in ["content", "source", "doc_type"]:
                        metadata[key] = val

                docs.append(
                    Document(
                        id=str(obj.uuid),
                        content=obj.properties.get("content", ""),
                        metadata=metadata,
                        score=obj.metadata.score if obj.metadata else None,
                    )
                )
            return docs

        except Exception as e:
            print(f"Weaviate query error: {e}. Falling back to mock data.")
            return self._get_fallback_docs()

    def _get_fallback_docs(self) -> List[Document]:
        """Returns mock documents for testing purposes."""
        return [
            Document(
                id="doc_weaviate_mock_1",
                content=(
                    "Điều 5. Các hành vi bị nghiêm cấm theo Luật phòng chống ma túy:\n"
                    "1. Trồng cây chứa chất ma túy.\n"
                    "2. Sản xuất, tàng trữ, vận chuyển, mua bán trái phép chất ma túy.\n"
                    "3. Sử dụng, tổ chức sử dụng trái phép chất ma túy (bao gồm hít, tiêm chích heroin hoặc các chất ma túy khác)."
                ),
                metadata={
                    "source": "luat_phong_chong_ma_tuy_2021.md",
                    "type": "legal",
                },
                score=0.9,
            ),
            Document(
                id="doc_weaviate_mock_2",
                content=(
                    "Tin tức 2024: Cơ quan công an vừa phát hiện và thu giữ một lượng lớn "
                    "chất ma túy tổng hợp được vận chuyển trái phép qua đường biên giới."
                ),
                metadata={"source": "tin_tuc_ma_tuy_2024.md", "type": "news"},
                score=0.8,
            ),
        ]
