"""Task 4 - Load standardized Markdown, chunk documents, and index locally."""

from pathlib import Path

from .text_utils import markdown_field, tokenize


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Recursive character chunking works well for mixed legal/news Markdown. A
# 500-character window keeps retrieval focused; 50 chars of overlap preserves
# facts split across boundaries.
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


def load_documents() -> list[dict]:
    """Read all Markdown files from data/standardized/."""
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
                    "path": str(rel_path).replace("\\", "/"),
                    "type": _document_type(md_file),
                    "source_url": markdown_field(content, "Source"),
                    "crawled_at": markdown_field(content, "Crawled"),
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
            boundary = max(
                text.rfind("\n\n", start, hard_end),
                text.rfind("\n", start, hard_end),
                text.rfind(". ", start, hard_end),
                text.rfind(" ", start, hard_end),
            )
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
    """Split documents into recursive character chunks."""
    chunks = []
    for doc_index, doc in enumerate(documents):
        for chunk_index, content in enumerate(_split_text(doc.get("content", ""))):
            chunks.append(
                {
                    "content": content,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "doc_index": doc_index,
                        "chunk_index": chunk_index,
                    },
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Attach deterministic local token embeddings to chunks."""
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
    """Run load -> chunk -> embed -> index."""
    docs = load_documents()
    chunks = chunk_documents(docs)
    return index_to_vectorstore(embed_chunks(chunks))


if __name__ == "__main__":
    print(f"Indexed {len(run_pipeline())} chunks")
