"""Task 3 - Convert landing files to standardized Markdown."""

import json
from pathlib import Path


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_EXTENSIONS = {".pdf", ".docx", ".doc"}


def _safe_text(value) -> str:
    return "" if value is None else str(value).strip()


def convert_legal_docs() -> list[Path]:
    """Convert legal PDF/DOC/DOCX files with MarkItDown."""
    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise RuntimeError("markitdown is required for Task 3 legal conversion.") from exc

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        return []

    converter = MarkItDown()
    saved_paths = []
    for filepath in sorted(legal_dir.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() not in LEGAL_EXTENSIONS:
            continue
        result = converter.convert(str(filepath))
        content = _safe_text(getattr(result, "text_content", ""))
        if not content:
            continue
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content + "\n", encoding="utf-8")
        saved_paths.append(output_path)
        print(f"Saved: {output_path}")

    return saved_paths


def convert_news_articles() -> list[Path]:
    """Convert crawled news JSON files to Markdown with metadata headers."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        return []

    saved_paths = []
    for filepath in sorted(news_dir.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() != ".json":
            continue
        data = json.loads(filepath.read_text(encoding="utf-8"))
        title = _safe_text(data.get("title")) or filepath.stem
        source = _safe_text(data.get("url")) or "N/A"
        crawled = _safe_text(
            data.get("date_crawled") or data.get("crawl_date") or data.get("crawled_at")
        ) or "N/A"
        body = _safe_text(
            data.get("content_markdown")
            or data.get("markdown")
            or data.get("content")
            or data.get("text")
        )
        if not body:
            continue

        content = (
            f"# {title}\n\n"
            f"**Source:** {source}\n"
            f"**Crawled:** {crawled}\n\n"
            "---\n\n"
            f"{body}\n"
        )
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content, encoding="utf-8")
        saved_paths.append(output_path)
        print(f"Saved: {output_path}")

    return saved_paths


def convert_all() -> list[Path]:
    """Convert all supported landing files while preserving subdirectories."""
    legal_files = convert_legal_docs()
    news_files = convert_news_articles()
    converted = legal_files + news_files
    print(f"Converted {len(converted)} files to {OUTPUT_DIR}")
    return converted


if __name__ == "__main__":
    convert_all()
