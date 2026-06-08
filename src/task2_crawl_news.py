"""
Task 2 - Crawl news articles about Vietnamese artists related to drug cases.

Requirements:
    1. Crawl at least 5 articles from Vietnamese news sites.
    2. Prefer Crawl4AI, with a dependency-free fallback for constrained setups.
    3. Save one JSON file per article in data/landing/news/.
    4. Include metadata: source URL, crawl date, title, and content.
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen


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

    for index, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        filepath = DATA_DIR / f"article_{index:02d}.json"
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved: {filepath}")


def _crawl_with_stdlib(url: str) -> dict:
    """Small fallback crawler for environments where Crawl4AI is not installed."""
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        raw = response.read()

    html = raw.decode("utf-8", errors="ignore")
    title = _extract_title(html, url)
    content = _clean_html(html)
    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": f"# {title}\n\n{content}",
    }


def _extract_title(html: str, url: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    if not match:
        return _title_from_url(url)

    title = _clean_html(match.group(1))
    return title.split("|")[0].split(" - ")[0].strip() or _title_from_url(url)


def _title_from_url(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1].rsplit(".", 1)[0]
    slug = re.sub(r"-post\d+$", "", slug)
    return re.sub(r"[-_]+", " ", slug).strip().title() or "Unknown"


def _clean_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>|</p>|</h[1-6]>", "\n", html)
    text = re.sub(r"(?is)<[^>]+>", " ", html)

    entities = {
        "&nbsp;": " ",
        "&amp;": "&",
        "&quot;": '"',
        "&#39;": "'",
        "&lt;": "<",
        "&gt;": ">",
    }
    for entity, value in entities.items():
        text = text.replace(entity, value)

    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("Please add ARTICLE_URLS before running.")
    else:
        asyncio.run(crawl_all())
