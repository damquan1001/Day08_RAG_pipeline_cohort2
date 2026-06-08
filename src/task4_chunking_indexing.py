"""
Task 4 - Load standardized Markdown, chunk it, and provide local indexing hooks.

The project README recommends production vector stores such as Weaviate. For this
training repo we keep a deterministic local implementation so later tasks and
tests can run without external services or model downloads.
"""

import re
from pathlib import Path


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Recursive character chunking is robust for mixed legal/news Markdown. A
# 500-character window keeps chunks small enough for focused retrieval while a
# 50-character overlap preserves short facts that cross a boundary.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Production recommendation: BAAI/bge-m3 (1024 dim) + Weaviate. This implementation
# also exposes a local token embedding so tests and BM25/hybrid retrieval can run
# without model downloads.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
VECTOR_STORE = "weaviate"


def _document_type(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "legal" in parts:
        return "legal"
    if "news" in parts:
        return "news"
    return "unknown"


def _extract_markdown_field(content: str, field_name: str) -> str:
    pattern = rf"^\*\*{re.escape(field_name)}:\*\*\s*(.+)$"
    match = re.search(pattern, content, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def load_documents() -> list[dict]:
    """
    Read all Markdown files from data/standardized/.

    Returns:
        List of {"content": str, "metadata": {"source": str, "type": str, ...}}
    """
    if not STANDARDIZED_DIR.exists():
        return []

    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name == ".gitkeep":
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        rel_path = md_file.relative_to(STANDARDIZED_DIR)
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "source_url": _extract_markdown_field(content, "Source"),
                    "crawled_at": _extract_markdown_field(content, "Crawled"),
                    "path": str(rel_path).replace("\\", "/"),
                    "type": _document_type(md_file),
                },
            }
        )
    return documents


def _split_text(text: str) -> list[str]:
    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        hard_end = min(start + CHUNK_SIZE, len(text))
        end = hard_end

        if hard_end < len(text):
            boundary_candidates = [
                text.rfind("\n\n", start, hard_end),
                text.rfind("\n", start, hard_end),
                text.rfind(". ", start, hard_end),
                text.rfind(" ", start, hard_end),
            ]
            boundary = max(boundary_candidates)
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - CHUNK_OVERLAP)

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split documents with recursive character boundaries.

    Returns:
        List of {"content": str, "metadata": dict}
    """
    chunks = []
    for doc_index, doc in enumerate(documents):
        metadata = doc.get("metadata", {})
        for chunk_index, chunk_text in enumerate(_split_text(doc.get("content", ""))):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "doc_index": doc_index,
                        "chunk_index": chunk_index,
                    },
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Attach a lightweight token set as the local embedding representation.
    """
    from .text_utils import tokenize

    embedded = []
    for chunk in chunks:
        item = chunk.copy()
        item["embedding"] = sorted(set(tokenize(item.get("content", ""))))
        embedded.append(item)
    return embedded


def index_to_vectorstore(chunks: list[dict]):
    """Return an in-memory index object for local retrieval."""
    return chunks


def run_pipeline():
    """Run the local load -> chunk -> embed -> index pipeline."""
    docs = load_documents()
    chunks = chunk_documents(docs)
    embedded = embed_chunks(chunks)
    return index_to_vectorstore(embedded)


if __name__ == "__main__":
    indexed = run_pipeline()
    print(f"Indexed {len(indexed)} chunks from {STANDARDIZED_DIR}")
