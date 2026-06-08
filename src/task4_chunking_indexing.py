"""Task 4 - Chunking and local indexing.

This implementation keeps the required public API while using a deterministic
local index.  It is suitable for the individual assignment tests and for a
small offline demo when heavyweight vector DB packages are not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

from .local_retrieval import chunk_loaded_documents, load_markdown_documents, tokenize


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LOCAL_INDEX_PATH = STANDARDIZED_DIR.parent / "local_index.json"

# Recursive character chunking style: simple, robust for legal text and news.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNKING_METHOD = "recursive-character-local"

# Local hashed bag-of-words embedding.  The dimension mirrors all-MiniLM-L6-v2.
EMBEDDING_MODEL = "local-tfidf-hash"
EMBEDDING_DIM = 384
VECTOR_STORE = "local-json"


def load_documents() -> list[dict]:
    """Load markdown files from data/standardized/."""
    return load_markdown_documents()


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Create bounded-size chunks from loaded documents."""
    return chunk_loaded_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Attach deterministic normalized hash embeddings to chunks."""
    for chunk in chunks:
        vector = [0.0] * EMBEDDING_DIM
        for token in tokenize(chunk["content"]):
            vector[hash(token) % EMBEDDING_DIM] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        chunk["embedding"] = [value / norm for value in vector]
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> Path:
    """Persist the local index as JSON."""
    LOCAL_INDEX_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return LOCAL_INDEX_PATH


def run_pipeline() -> Path:
    """Run load -> chunk -> embed -> local index."""
    docs = load_documents()
    chunks = chunk_documents(docs)
    chunks = embed_chunks(chunks)
    return index_to_vectorstore(chunks)


if __name__ == "__main__":
    path = run_pipeline()
    print(f"Indexed local chunks to {path}")
