"""Small local text helpers shared by retrieval tasks."""

import math
import re
import unicodedata
from collections import Counter


TOKEN_RE = re.compile(r"[0-9a-zA-Z_]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Lowercase and strip accents so Vietnamese queries match ASCII fallbacks."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def cosine_from_counters(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[token] * right.get(token, 0) for token in left)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def content_key(item: dict) -> str:
    metadata = item.get("metadata", {})
    return "|".join(
        [
            str(metadata.get("path", "")),
            str(metadata.get("chunk_index", "")),
            item.get("content", "")[:80],
        ]
    )
