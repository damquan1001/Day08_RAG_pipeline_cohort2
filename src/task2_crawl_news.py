"""Task 2 - Crawl news articles about Vietnamese artists and drug cases."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://lifestyle.znews.vn/chi-dan-an-tay-la-nhung-mat-xich-cuoi-trong-duong-day-ma-tuy-post1510988.html",
    "https://tienphong.vn/an-tay-khoc-nuc-no-khi-bi-bat-em-mat-het-su-nghiep-roi-post1691646.tpo",
    "https://tuoitre.vn/nguoi-mau-nhikolai-dinh-bi-bat-trong-chuyen-an-ma-tuy-o-khu-ma-lang-quan-1-20240625230004986.htm",
    "https://thanhnien.vn/ca-si-chi-dan-nguoi-mau-an-tay-bi-bat-vi-lien-quan-ma-tuy-185241114173112269.htm",
    "https://dantri.com.vn/phap-luat/nhieu-nghe-si-nguoi-mau-lien-quan-duong-day-ma-tuy-20241115103812345.htm",
]


def setup_directory():
    """Create data/landing/news/ if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


async def crawl_article(url: str) -> dict:
    """
    Crawl one article using Crawl4AI.

    Returns a dict with url, title, date_crawled, and content_markdown.
    """
    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError as exc:
        raise RuntimeError("crawl4ai is required for Task 2. Install requirements.txt.") from exc

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    metadata = getattr(result, "metadata", {}) or {}
    title = metadata.get("title") or metadata.get("og:title") or url
    markdown = (
        getattr(result, "markdown", None)
        or getattr(result, "fit_markdown", None)
        or getattr(result, "cleaned_html", None)
        or ""
    )

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": str(markdown).strip(),
    }


async def crawl_all(urls: list[str] | None = None) -> list[Path]:
    """Crawl all configured articles and save them as JSON."""
    setup_directory()
    urls = urls or ARTICLE_URLS
    saved_paths = []

    for index, url in enumerate(urls, start=1):
        print(f"[{index}/{len(urls)}] Crawling: {url}")
        article = await crawl_article(url)
        filename = f"article_{index:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        saved_paths.append(filepath)
        print(f"  Saved: {filepath}")

    return saved_paths


if __name__ == "__main__":
    asyncio.run(crawl_all())
