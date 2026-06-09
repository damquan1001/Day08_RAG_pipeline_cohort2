from __future__ import annotations

import re
import urllib.parse
from typing import List, Optional

try:
    import weaviate
    from weaviate.classes.query import MetadataQuery
    from weaviate.classes.init import Auth
except ImportError:
    weaviate = None
    MetadataQuery = None
    Auth = None

from system_contracts import Document
from src.module_rag_core.ports.outbound import VectorStorePort


class WeaviateDockerAdapter(VectorStorePort):
    """Weaviate adapter with deterministic fallback documents for demos/tests."""

    def __init__(self) -> None:
        self.client = None
        self.class_name = "DrugLawDocs"

    def connect(self, url: str, api_key: Optional[str] = None) -> None:
        if weaviate is None:
            self.client = None
            return

        try:
            if self._is_cloud_url(url):
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=url,
                    auth_credentials=Auth.api_key(api_key) if api_key and Auth else None,
                )
            else:
                parsed = urllib.parse.urlparse(url)
                host = parsed.hostname or "localhost"
                port = parsed.port or 8080
                self.client = weaviate.connect_to_local(host=host, port=port)
        except Exception as exc:
            print(
                f"Warning: could not connect to Weaviate at {url} ({exc}). "
                "Using fallback documents."
            )
            self.client = None

    def hybrid_search(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> List[Document]:
        if not self.client or MetadataQuery is None:
            return self._fallback_docs(query=query, top_k=top_k)

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
                    doc_vector = (
                        obj.vector.get("default")
                        if isinstance(obj.vector, dict)
                        else obj.vector
                    )

                metadata = {
                    "source": obj.properties.get("source", "unknown"),
                    "type": obj.properties.get("doc_type", "unknown"),
                    "vector": doc_vector,
                }
                for key, value in obj.properties.items():
                    if key not in ["content", "source", "doc_type"]:
                        metadata[key] = value

                docs.append(
                    Document(
                        id=str(obj.uuid),
                        content=obj.properties.get("content", ""),
                        metadata=metadata,
                        score=obj.metadata.score if obj.metadata else None,
                    )
                )
            return docs
        except Exception as exc:
            print(f"Weaviate query error: {exc}. Using fallback documents.")
            return self._fallback_docs(query=query, top_k=top_k)

    def _fallback_docs(self, query: str, top_k: int) -> List[Document]:
        scored_docs = []
        query_tokens = self._tokens(query)
        for doc in self._offline_corpus():
            doc_tokens = self._tokens(doc.content + " " + str(doc.metadata))
            overlap = len(query_tokens & doc_tokens)
            score = overlap / max(len(query_tokens), 1)
            if doc.score is not None:
                score += doc.score * 0.1
            doc.score = round(score, 3)
            scored_docs.append(doc)

        scored_docs.sort(key=lambda item: item.score or 0.0, reverse=True)
        return scored_docs[:top_k]

    @staticmethod
    def _offline_corpus() -> List[Document]:
        return [
            Document(
                id="offline_blhs_249",
                content=(
                    "Dieu 249 Bo luat Hinh su: nguoi nao tang tru trai phep "
                    "chat ma tuy ma khong nham muc dich mua ban, van chuyen, "
                    "san xuat trai phep chat ma tuy thi co the bi phat tu tu "
                    "01 nam den 05 nam o khoan 1; muc phat tang theo khoi "
                    "luong chat ma tuy va tinh tiet tang nang."
                ),
                metadata={
                    "source": "bo-luat-hinh-su-ma-tuy.md",
                    "type": "legal",
                    "title": "Dieu 249 - tang tru trai phep chat ma tuy",
                },
                score=0.9,
            ),
            Document(
                id="offline_luat_28_29",
                content=(
                    "Luat Phong, chong ma tuy 2021 quy dinh bien phap cai "
                    "nghien gom cai nghien tu nguyen va cai nghien bat buoc. "
                    "Quy trinh cai nghien gom tiep nhan phan loai, dieu tri "
                    "cat con giai doc, giao duc tu van phuc hoi hanh vi, lao "
                    "dong tri lieu hoc nghe va chuan bi tai hoa nhap cong dong."
                ),
                metadata={
                    "source": "luat-phong-chong-ma-tuy-2021.md",
                    "type": "legal",
                    "title": "Dieu 28-29 - cai nghien ma tuy",
                },
                score=0.85,
            ),
            Document(
                id="offline_luat_5",
                content=(
                    "Dieu 5 Luat Phong, chong ma tuy 2021 nghiem cam trong "
                    "cay co chua chat ma tuy; san xuat, tang tru, van chuyen, "
                    "mua ban trai phep chat ma tuy; su dung, to chuc su dung, "
                    "cuong buc hoac loi keo nguoi khac su dung trai phep, bao "
                    "gom hit, tiem chich heroin hoac cac chat ma tuy khac."
                ),
                metadata={
                    "source": "luat-phong-chong-ma-tuy-2021.md",
                    "type": "legal",
                    "title": "Dieu 5 - hanh vi bi nghiem cam",
                },
                score=0.82,
            ),
            Document(
                id="offline_definition",
                content=(
                    "Chat ma tuy la chat gay nghien, chat huong than duoc quy "
                    "dinh trong danh muc chat ma tuy do Chinh phu ban hanh."
                ),
                metadata={
                    "source": "luat-phong-chong-ma-tuy-2021.md",
                    "type": "legal",
                    "title": "Dinh nghia chat ma tuy",
                },
                score=0.78,
            ),
            Document(
                id="offline_news",
                content=(
                    "Nhom tin tuc trong corpus dung de doi chieu cac vu viec "
                    "lien quan den ma tuy, bat giu, van chuyen va xu ly theo "
                    "quy dinh phap luat."
                ),
                metadata={
                    "source": "news_corpus.md",
                    "type": "news",
                    "title": "Tin tuc lien quan ma tuy",
                },
                score=0.7,
            ),
        ]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower()))

    @staticmethod
    def _is_cloud_url(url: str) -> bool:
        return "weaviate.cloud" in url or "weaviate.network" in url
