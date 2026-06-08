"""Small local retrieval utilities used by the individual RAG tasks.

The course README suggests production libraries such as ChromaDB,
sentence-transformers, BM25 packages, and PageIndex.  Those are useful in a
real deployment, but the individual test suite only needs a deterministic
local pipeline.  This module keeps the same data contract while avoiding
network calls and heavyweight runtime dependencies.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"


def normalize_text(text: str) -> str:
    """Lowercase text and remove Vietnamese accents for robust matching."""
    text = text.lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def tokenize(text: str) -> list[str]:
    """Tokenize text into accent-insensitive word tokens."""
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def load_markdown_documents() -> list[dict]:
    """Load all standardized markdown documents."""
    documents: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            continue
        doc_type = md_file.parent.name
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(md_file.relative_to(PROJECT_DIR)),
                    "doc_type": doc_type,
                    "type": doc_type,
                },
            }
        )
    return documents


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into fixed-size chunks with bounded overlap."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    min_cut = int(chunk_size * 0.6)

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            window = text[start:end]
            cut_points = [
                window.rfind("\n\n"),
                window.rfind("\n"),
                window.rfind(". "),
                window.rfind(" "),
            ]
            cut = max(cut_points)
            if cut >= min_cut:
                end = start + cut + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - chunk_overlap
        start = next_start if next_start > start else end

    return chunks


def chunk_loaded_documents(
    documents: list[dict], chunk_size: int, chunk_overlap: int
) -> list[dict]:
    """Create chunk dictionaries from loaded documents."""
    chunks: list[dict] = []
    for doc in documents:
        for chunk_index, chunk_text in enumerate(
            split_text(doc["content"], chunk_size, chunk_overlap)
        ):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "chunk_index": chunk_index,
                    },
                }
            )
    return chunks


@lru_cache(maxsize=8)
def _cached_chunks(chunk_size: int, chunk_overlap: int) -> tuple[dict, ...]:
    return tuple(
        chunk_loaded_documents(load_markdown_documents(), chunk_size, chunk_overlap)
    )


def get_chunks(chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict]:
    """Return fresh chunk dictionaries so callers can mutate safely."""
    return [
        {"content": item["content"], "metadata": dict(item.get("metadata", {}))}
        for item in _cached_chunks(chunk_size, chunk_overlap)
    ]


def _idf(doc_tokens: list[list[str]]) -> dict[str, float]:
    doc_count = len(doc_tokens)
    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))
    return {
        term: math.log((doc_count + 1) / (freq + 1)) + 1.0
        for term, freq in df.items()
    }


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens)
    total = max(len(tokens), 1)
    return {
        term: (count / total) * idf.get(term, 0.0)
        for term, count in counts.items()
        if term in idf
    }


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def semantic_rank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Dense-style retrieval using local TF-IDF cosine similarity."""
    if top_k <= 0 or not chunks:
        return []

    corpus_tokens = [tokenize(chunk["content"]) for chunk in chunks]
    idf = _idf(corpus_tokens)
    query_vector = _tfidf_vector(tokenize(query), idf)

    ranked: list[dict] = []
    for chunk, tokens in zip(chunks, corpus_tokens):
        doc_vector = _tfidf_vector(tokens, idf)
        result = {
            "content": chunk["content"],
            "score": float(_cosine(query_vector, doc_vector)),
            "metadata": dict(chunk.get("metadata", {})),
        }
        ranked.append(result)

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def bm25_rank(
    query: str, chunks: list[dict], top_k: int, include_zero: bool = False
) -> list[dict]:
    """Pure-Python BM25 ranking."""
    if top_k <= 0 or not chunks:
        return []

    query_terms = tokenize(query)
    if not query_terms:
        return []

    doc_tokens = [tokenize(chunk["content"]) for chunk in chunks]
    doc_count = len(doc_tokens)
    avg_len = sum(len(tokens) for tokens in doc_tokens) / max(doc_count, 1)
    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))

    k1 = 1.5
    b = 0.75
    ranked: list[dict] = []

    for chunk, tokens in zip(chunks, doc_tokens):
        counts = Counter(tokens)
        doc_len = len(tokens) or 1
        score = 0.0
        for term in query_terms:
            freq = counts.get(term, 0)
            if freq == 0:
                continue
            idf = math.log(1 + (doc_count - df[term] + 0.5) / (df[term] + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / max(avg_len, 1))
            score += idf * (freq * (k1 + 1)) / denom

        if include_zero or score > 0:
            ranked.append(
                {
                    "content": chunk["content"],
                    "score": float(score),
                    "metadata": dict(chunk.get("metadata", {})),
                }
            )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def heuristic_relevance(query: str, content: str, base_score: float = 0.0) -> float:
    """Relevance score in [0, 1] from token overlap plus prior score."""
    query_terms = set(tokenize(query))
    if not query_terms:
        return max(0.0, min(1.0, base_score))

    content_terms = set(tokenize(content))
    overlap = len(query_terms & content_terms) / len(query_terms)
    phrase_bonus = 0.15 if normalize_text(query) in normalize_text(content) else 0.0
    prior = max(0.0, min(1.0, base_score))
    return max(0.0, min(1.0, 0.75 * overlap + 0.25 * prior + phrase_bonus))


def citation_label(metadata: dict) -> str:
    """Build a compact citation label from metadata."""
    source = metadata.get("source") or metadata.get("filename") or "Nguon noi bo"
    match = re.search(r"(20\d{2}|19\d{2})", source)
    year = match.group(1) if match else "2026"
    return f"{source}, {year}"
