"""
Task 8 - PageIndex Vectorless RAG.

Fallback local mô phỏng vectorless retrieval bằng cách đọc cấu trúc markdown và
chấm điểm theo từ khóa. Không cần PAGEINDEX_API_KEY khi chạy bài cá nhân.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
import sys
import json
import hashlib
import requests
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_DIR / "data" / "pageindex_manifest.json"
PAGEINDEX_MARKDOWN_URL = "https://api.pageindex.ai/v1/markdown"

if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_DIR))
    from src.task4_chunking_indexing import chunk_documents, load_documents
else:
    from .task4_chunking_indexing import chunk_documents, load_documents

load_dotenv(PROJECT_DIR / ".env")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"


def _file_fingerprint(path: Path) -> str:
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except Exception:
        return str(path.stat().st_mtime)


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"backend": "pageindex_markdown_api", "documents": []}


def _save_manifest(manifest: dict):
    try:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[Warning] Failed to save PageIndex manifest: {e}")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def upload_documents():
    """
    Local no-op upload: trả về danh sách file markdown đã sẵn sàng query.
    """
    return [str(path) for path in STANDARDIZED_DIR.rglob("*.md")]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    query_terms = set(_tokenize(query))
    chunks = chunk_documents(load_documents())
    if not chunks or top_k <= 0:
        return []

    results = []
    for chunk in chunks:
        tokens = _tokenize(chunk["content"])
        if not tokens:
            continue
        overlap = sum(1 for token in tokens if token in query_terms)
        unique_overlap = len(query_terms & set(tokens))
        score = unique_overlap + overlap / max(len(tokens), 1)
        if score <= 0:
            continue
        results.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
                "source": "pageindex",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _extract_records_from_tree(nodes: list, source: str, doc_type: str, parent_title: str = "") -> list[dict]:
    records = []
    for node in nodes or []:
        title = str(node.get("title") or "").strip()
        node_id = str(node.get("node_id") or node.get("id") or "")
        summary = str(node.get("summary") or "").strip()
        text = str(
            node.get("text")
            or node.get("node_text")
            or node.get("content")
            or node.get("relevant_content")
            or ""
        ).strip()

        heading = " > ".join(part for part in [parent_title, title] if part)
        content_parts = [part for part in [heading, summary, text] if part]
        content = "\n\n".join(content_parts).strip()
        if content:
            records.append(
                {
                    "content": content,
                    "metadata": {
                        "source": source,
                        "type": doc_type,
                        "node_id": node_id,
                        "title": title,
                        "retrieval_backend": "pageindex_api",
                    },
                }
            )

        child_nodes = node.get("nodes") or node.get("children") or []
        records.extend(_extract_records_from_tree(child_nodes, source, doc_type, heading))
    return records


def _convert_markdown_with_pageindex(path: Path) -> dict[str, Any]:
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is not configured")

    with path.open("rb") as file_obj:
        response = requests.post(
            PAGEINDEX_MARKDOWN_URL,
            headers={"api_key": PAGEINDEX_API_KEY},
            files={"file": file_obj},
            data={
                "if_add_node_id": "yes",
                "if_add_node_summary": "yes",
                "if_add_node_text": "yes",
                "if_add_doc_description": "no",
            },
            timeout=120,
        )

    if response.status_code != 200:
        raise RuntimeError(f"PageIndex markdown API failed: HTTP {response.status_code}")
    return response.json()


def upload_documents() -> dict:
    """
    Convert markdown files through PageIndex API and cache their tree records.

    Returns:
        Manifest dict saved at data/pageindex_manifest.json.
    """
    manifest = _load_manifest()
    existing_by_path = {doc.get("path"): doc for doc in manifest.get("documents", [])}
    documents = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        rel_path = str(md_file.relative_to(PROJECT_DIR))
        fingerprint = _file_fingerprint(md_file)
        cached = existing_by_path.get(rel_path)
        if cached and cached.get("fingerprint") == fingerprint and cached.get("records"):
            documents.append(cached)
            continue

        result = _convert_markdown_with_pageindex(md_file)
        structure = result.get("structure") or result.get("result") or []
        doc_type = md_file.parent.name
        records = _extract_records_from_tree(structure, md_file.name, doc_type)
        if not records:
            content = md_file.read_text(encoding="utf-8")
            records = [
                {
                    "content": content,
                    "metadata": {
                        "source": md_file.name,
                        "type": doc_type,
                        "node_id": "raw_markdown",
                        "title": md_file.stem,
                        "retrieval_backend": "pageindex_api_raw_fallback",
                    },
                }
            ]

        documents.append(
            {
                "path": rel_path,
                "source": md_file.name,
                "type": doc_type,
                "fingerprint": fingerprint,
                "api_success": bool(result.get("success", True)),
                "records": records,
            }
        )

    manifest = {
        "backend": "pageindex_markdown_api",
        "documents": documents,
    }
    _save_manifest(manifest)
    return manifest


def _score_records(records: list[dict], query: str, top_k: int) -> list[dict]:
    query_terms = set(_tokenize(query))
    if not query_terms or top_k <= 0:
        return []

    results = []
    for record in records:
        tokens = _tokenize(record.get("content", ""))
        if not tokens:
            continue
        token_set = set(tokens)
        unique_overlap = len(query_terms & token_set)
        overlap = sum(1 for token in tokens if token in query_terms)
        score = unique_overlap + overlap / max(len(tokens), 1)
        if score <= 0:
            continue
        results.append(
            {
                "content": record["content"],
                "score": float(score),
                "metadata": record.get("metadata", {}),
                "source": "pageindex",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _local_pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    chunks = chunk_documents(load_documents())
    records = [
        {
            "content": chunk["content"],
            "metadata": {
                **chunk.get("metadata", {}),
                "retrieval_backend": "local_pageindex_fallback",
            },
        }
        for chunk in chunks
    ]
    return _score_records(records, query, top_k)


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using cached PageIndex tree records when available.

    The function intentionally avoids uploading documents automatically. Run
    upload_documents() or this module as a script to refresh the API cache.
    """
    if top_k <= 0:
        return []

    manifest = _load_manifest()
    records = []
    for document in manifest.get("documents", []):
        records.extend(document.get("records", []))

    if records:
        api_results = _score_records(records, query, top_k)
        if api_results:
            return api_results

    return _local_pageindex_search(query, top_k)


if __name__ == "__main__":
    print(f"PAGEINDEX_API_KEY: {'present' if PAGEINDEX_API_KEY else 'missing'}")
    if PAGEINDEX_API_KEY:
        try:
            manifest = upload_documents()
            record_count = sum(len(doc.get("records", [])) for doc in manifest.get("documents", []))
            print(f"[Task 8] Cached PageIndex records: {record_count}")
        except Exception as exc:
            print(f"[Task 8] PageIndex API unavailable, using local fallback: {type(exc).__name__}")

    results = pageindex_search("hình phạt sử dụng ma túy", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
