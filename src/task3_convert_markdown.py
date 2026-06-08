"""
Task 3 - Convert files in data/landing/ to Markdown.

Legal documents are converted with MarkItDown. News crawl JSON files are
normalized into Markdown using their metadata and content fields.
"""

import json
from pathlib import Path


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
SUPPORTED_LEGAL_EXTENSIONS = {".pdf", ".docx", ".doc"}


def _safe_text(value):
    """Return a stripped string for optional metadata fields."""
    if value is None:
        return ""
    return str(value).strip()


def convert_legal_docs():
    """Convert PDF/DOC/DOCX files in data/landing/legal/ to Markdown."""
    try:
        from markitdown import MarkItDown
    except ImportError:
        print("Skipping legal docs: install markitdown to convert PDF/DOC/DOCX files")
        return []

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print(f"Skipping: {legal_dir} does not exist")
        return []

    md = MarkItDown()
    converted_files = []

    for filepath in sorted(legal_dir.iterdir()):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in SUPPORTED_LEGAL_EXTENSIONS:
            continue

        print(f"Converting: {filepath.name}")
        result = md.convert(str(filepath))
        content = _safe_text(result.text_content)

        if not content:
            print(f"  Skipped: no text extracted from {filepath.name}")
            continue

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content + "\n", encoding="utf-8")
        converted_files.append(output_path)
        print(f"  Saved: {output_path}")

    return converted_files


def convert_news_articles():
    """Convert crawled JSON articles in data/landing/news/ to Markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print(f"Skipping: {news_dir} does not exist")
        return []

    converted_files = []

    for filepath in sorted(news_dir.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() != ".json":
            continue

        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))

        title = _safe_text(data.get("title")) or filepath.stem
        source = _safe_text(data.get("url")) or "N/A"
        crawled = _safe_text(
            data.get("date_crawled")
            or data.get("crawl_date")
            or data.get("crawled_at")
        ) or "N/A"
        body = _safe_text(
            data.get("content_markdown")
            or data.get("markdown")
            or data.get("content")
            or data.get("text")
        )

        if not body:
            print(f"  Skipped: no content field in {filepath.name}")
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
        converted_files.append(output_path)
        print(f"  Saved: {output_path}")

    return converted_files


def convert_all():
    """Convert all supported landing files into data/standardized/."""
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_files = convert_legal_docs()

    print("\n--- News Articles ---")
    news_files = convert_news_articles()

    total = len(legal_files) + len(news_files)
    print(f"\nDone! Converted {total} files. Output at: {OUTPUT_DIR}")
    return legal_files + news_files


if __name__ == "__main__":
    convert_all()
