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


def setup_directory():
    """Create data/landing/news/ if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://lifestyle.znews.vn/chi-dan-an-tay-la-nhung-mat-xich-cuoi-trong-duong-day-ma-tuy-post1510988.html",
    "https://tienphong.vn/an-tay-khoc-nuc-no-khi-bi-bat-em-mat-het-su-nghiep-roi-post1691646.tpo",
    "https://tuoitre.vn/nguoi-mau-nhikolai-dinh-bi-bat-trong-chuyen-an-ma-tuy-o-khu-ma-lang-quan-1-20240625230004986.htm",
    "https://vietnamnet.vn/loi-khai-cua-dien-vien-huu-tin-sau-khi-bi-bat-vi-choi-ma-tuy-2029765.html",
    "https://nld.com.vn/phap-luat/dien-vien-hai-huu-tin-khai-dung-ma-tuy-vi-to-mo-20230428112443007.htm",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl one article and return metadata plus markdown-like text.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str,
            "content_markdown": str
        }
    """
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            metadata = getattr(result, "metadata", {}) or {}
            return {
                "url": url,
                "title": metadata.get("title", _title_from_url(url)),
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": getattr(result, "markdown", "") or "",
            }
    except ImportError:
        return await asyncio.to_thread(_crawl_with_stdlib, url)


async def crawl_all():
    """Crawl all URLs in ARTICLE_URLS and save each one as JSON."""
    setup_directory()

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
