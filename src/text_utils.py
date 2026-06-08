"""Shared text and citation helpers for the RAG tasks."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from urllib.parse import urlparse


TOKEN_RE = re.compile(r"[0-9a-zA-Z_]+", re.UNICODE)

PLATFORM_NAMES = {
    "znews.vn": "ZNews",
    "lifestyle.znews.vn": "ZNews",
    "tienphong.vn": "Tien Phong",
    "tpo": "Tien Phong",
    "tuoitre.vn": "Tuoi Tre",
    "vnexpress.net": "VnExpress",
    "thanhnien.vn": "Thanh Nien",
    "dantri.com.vn": "Dan Tri",
    "plo.vn": "PLO",
    "congan.com.vn": "Cong An",
}


def normalize_text(text: str) -> str:
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


def markdown_field(content: str, field_name: str) -> str:
    pattern = rf"^\*\*{re.escape(field_name)}:\*\*\s*(.+)$"
    match = re.search(pattern, content, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def platform_from_url(url: str) -> str:
    if not url:
        return ""
    host = urlparse(url).netloc.lower().removeprefix("www.")
    if host in PLATFORM_NAMES:
        return PLATFORM_NAMES[host]
    domain = ".".join(host.split(".")[-2:])
    if domain in PLATFORM_NAMES:
        return PLATFORM_NAMES[domain]
    return host.split(".")[0].replace("-", " ").title() if host else ""


def platform_from_source(source: str) -> str:
    name = re.sub(r"\.[a-z0-9]+$", "", source, flags=re.IGNORECASE)
    name = name.replace("-", " ").replace("_", " ").strip()
    return name.title() if name else "Unknown Source"


def year_from_text(*values: str) -> str:
    for value in values:
        match = re.search(r"\b(19|20)\d{2}\b", value or "")
        if match:
            return match.group(0)
    return "n.d."


def content_key(item: dict) -> str:
    metadata = item.get("metadata", {})
    return "|".join(
        [
            str(metadata.get("path", "")),
            str(metadata.get("chunk_index", "")),
            item.get("content", "")[:80],
        ]
    )
