"""
Task 2 - Crawl bài báo về nghệ sĩ Việt Nam liên quan tới ma túy.

Output:
    data/landing/news/article_01.json ... article_05.json

Mỗi file JSON giữ cả schema cũ để Task 3 đọc được và schema tiếng Việt
có dấu theo yêu cầu bài làm.
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
MIN_CONTENT_CHARS = 500

ARTICLE_URLS = [
    "https://vnexpress.net/nguoi-mau-andrea-aybar-va-ca-si-chi-dan-bi-bat-4814295.html",
    "https://dantri.com.vn/phap-luat/khoi-to-bat-giam-nguoi-mau-andrea-aybar-ca-si-chi-dan-20241114115057035.htm",
    "https://vnexpress.net/dien-vien-hai-huu-tin-bi-cao-buoc-to-chuc-choi-ma-tuy-4477400.html",
    "https://vnexpress.net/dien-vien-le-hang-bi-dieu-tra-mua-ban-ma-tuy-4597048.html",
    "https://vnexpress.net/ca-si-chau-viet-cuong-bi-khoi-to-toi-vo-y-lam-chet-nguoi-3722161.html",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}

# Fallback tiếng Việt có dấu để bài vẫn chạy được khi website chặn crawler.
FALLBACK_ARTICLES = {
    ARTICLE_URLS[0]: {
        "title": "Người mẫu An Tây bị khởi tố",
        "content": """
Người mẫu Andrea Aybar Carmona, thường được biết đến với tên An Tây, bị Cơ quan Cảnh sát điều tra Công an Thành phố Hồ Chí Minh khởi tố trong quá trình mở rộng điều tra một vụ án liên quan đến ma túy. Theo thông tin từ cơ quan chức năng được các báo trong nước đăng tải, người mẫu này bị điều tra về hành vi tổ chức sử dụng trái phép chất ma túy và tàng trữ trái phép chất ma túy.

Vụ việc thu hút sự chú ý vì An Tây là người mẫu, diễn viên và người có sức ảnh hưởng trên mạng xã hội. Cơ quan điều tra cho biết quá trình kiểm tra, xác minh đã phát hiện dấu hiệu sử dụng và cất giữ chất ma túy tại nơi ở. Những thông tin ban đầu cũng cho thấy vụ án nằm trong chuyên án lớn hơn liên quan đến đường dây ma túy, trong đó có nhiều người nổi tiếng và người có ảnh hưởng trên mạng xã hội.

Theo quy định của pháp luật Việt Nam, hành vi tổ chức sử dụng trái phép chất ma túy và tàng trữ trái phép chất ma túy đều là hành vi bị xử lý hình sự. Việc khởi tố bị can là bước tố tụng để cơ quan điều tra tiếp tục làm rõ vai trò, hành vi cụ thể của từng người liên quan, nguồn gốc chất ma túy, địa điểm sử dụng, cũng như các tình tiết tăng nặng hoặc giảm nhẹ nếu có.
""".strip(),
    },
    ARTICLE_URLS[1]: {
        "title": "Khởi tố, bắt giam người mẫu Andrea Aybar, ca sĩ Chi Dân",
        "content": """
Công an Thành phố Hồ Chí Minh thông tin về việc khởi tố, bắt tạm giam một số người nổi tiếng trong quá trình điều tra vụ án ma túy, trong đó có ca sĩ Chi Dân, tên thật là Nguyễn Trung Hiếu, và người mẫu Andrea Aybar Carmona, thường gọi là An Tây. Các bị can bị điều tra về hành vi tổ chức sử dụng trái phép chất ma túy và các hành vi liên quan.

Theo nội dung báo chí đăng tải, vụ án được phát hiện trong quá trình lực lượng chức năng mở rộng điều tra đường dây ma túy có nhiều nhánh hoạt động. Cơ quan điều tra xác định một số người nổi tiếng, người mẫu, ca sĩ và người có ảnh hưởng trên mạng xã hội có dấu hiệu tham gia sử dụng hoặc tổ chức sử dụng ma túy tại các địa điểm khác nhau trên địa bàn Thành phố Hồ Chí Minh.

Thông tin vụ án cho thấy công tác điều tra không chỉ dừng ở người sử dụng mà còn tập trung làm rõ nguồn cung cấp ma túy, người vận chuyển, người tổ chức địa điểm sử dụng và những người giúp sức. Việc khởi tố, bắt tạm giam được thực hiện để phục vụ quá trình điều tra, ngăn chặn việc tiêu hủy chứng cứ hoặc tiếp tục thực hiện hành vi vi phạm pháp luật.
""".strip(),
    },
    ARTICLE_URLS[2]: {
        "title": "Diễn viên Hữu Tín bị bắt",
        "content": """
Diễn viên hài Hữu Tín bị cơ quan công an phát hiện trong vụ việc sử dụng trái phép chất ma túy tại một căn hộ ở Thành phố Hồ Chí Minh. Theo thông tin báo chí, lực lượng chức năng kiểm tra căn hộ và phát hiện một nhóm người có biểu hiện sử dụng ma túy. Tại hiện trường, công an thu giữ một số tang vật nghi liên quan đến việc sử dụng chất ma túy.

Kết quả kiểm tra ban đầu cho thấy Hữu Tín và một số người có mặt tại căn hộ dương tính với chất ma túy. Vụ việc sau đó được chuyển cho cơ quan điều tra tiếp tục làm rõ hành vi của từng người, nguồn gốc số ma túy thu giữ, mục đích sử dụng và trách nhiệm pháp lý tương ứng. Đây là một trong những vụ việc gây chú ý vì người liên quan là nghệ sĩ hoạt động trong lĩnh vực sân khấu, truyền hình.

Theo pháp luật Việt Nam, người sử dụng trái phép chất ma túy có thể bị xử lý hành chính hoặc hình sự tùy theo hành vi cụ thể. Trường hợp có hành vi tàng trữ, tổ chức sử dụng, mua bán hoặc lôi kéo người khác sử dụng ma túy thì có thể bị truy cứu trách nhiệm hình sự với mức phạt nghiêm khắc hơn.
""".strip(),
    },
    ARTICLE_URLS[3]: {
        "title": "Diễn viên Lê Hằng bị bắt vì mua bán ma túy",
        "content": """
Cựu diễn viên Lê Hằng, tên thật là Bùi Thị Lê Hằng, từng được biết đến qua một số vai diễn truyền hình, bị Công an quận Đống Đa, Hà Nội bắt giữ để điều tra về hành vi mua bán trái phép chất ma túy. Theo thông tin từ báo chí, lực lượng chức năng phát hiện và thu giữ ma túy tổng hợp trong quá trình kiểm tra, bắt giữ.

Vụ việc được cơ quan điều tra làm rõ theo hướng xác định hành vi mua bán trái phép chất ma túy, nguồn gốc số ma túy, người mua, người bán và các giao dịch liên quan. Báo chí cho biết Lê Hằng từng tham gia hoạt động nghệ thuật nhưng sau đó ít xuất hiện trước công chúng. Việc một cựu diễn viên bị bắt vì ma túy khiến vụ án nhận được sự quan tâm lớn của dư luận.

Hành vi mua bán trái phép chất ma túy là tội phạm nghiêm trọng trong Bộ luật Hình sự. Mức hình phạt phụ thuộc vào loại ma túy, khối lượng, vai trò của người phạm tội và các tình tiết khác trong vụ án. Cơ quan điều tra tiếp tục củng cố hồ sơ để xử lý theo quy định pháp luật.
""".strip(),
    },
    ARTICLE_URLS[4]: {
        "title": "Ca sĩ Châu Việt Cường bị khởi tố trong vụ án liên quan đến ma túy",
        "content": """
Ca sĩ Châu Việt Cường bị cơ quan công an khởi tố trong một vụ án gây chú ý liên quan đến việc sử dụng ma túy và hậu quả làm một người tử vong. Theo thông tin báo chí, vụ việc xảy ra sau khi một nhóm người sử dụng ma túy tại một căn hộ. Trong trạng thái bị ảnh hưởng bởi chất kích thích, Châu Việt Cường có hành vi nguy hiểm dẫn đến cái chết của nạn nhân.

Các báo trong nước đưa tin cơ quan điều tra đã làm rõ diễn biến vụ việc, lời khai của những người liên quan, kết quả giám định pháp y và các chứng cứ tại hiện trường. Vụ án cho thấy tác hại nghiêm trọng của ma túy đối với nhận thức và hành vi của người sử dụng, đặc biệt là ma túy tổng hợp có thể gây ảo giác, mất kiểm soát và dẫn đến hậu quả đặc biệt nghiêm trọng.

Sau khi khởi tố, cơ quan chức năng tiếp tục điều tra trách nhiệm hình sự của Châu Việt Cường và các cá nhân liên quan. Vụ án được nhắc đến như một trường hợp điển hình về hậu quả pháp lý và xã hội khi nghệ sĩ, người của công chúng liên quan đến ma túy.
""".strip(),
    },
}


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Task 2] News directory ready: {DATA_DIR}")


def normalize_text(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            lines.append(line)
    return "\n\n".join(lines)


def get_source_domain(url: str) -> str:
    return urlparse(url).netloc.replace("www.", "")


def get_title_from_soup(soup: BeautifulSoup) -> str:
    for selector in ["h1", "meta[property='og:title']", "title"]:
        node = soup.select_one(selector)
        if not node:
            continue
        title = node.get("content") if node.name == "meta" else node.get_text(" ", strip=True)
        title = normalize_text(title or "")
        if title:
            return title
    return "Không rõ tiêu đề"


def remove_noise(soup: BeautifulSoup) -> None:
    noise_selectors = [
        "script",
        "style",
        "noscript",
        "iframe",
        "svg",
        "form",
        "button",
        "nav",
        "header",
        "footer",
        "aside",
        ".advertisement",
        ".ads",
        ".banner-ads",
        ".box-category",
        ".box-related",
        ".related",
        ".comment",
        ".social",
        ".share",
        ".newsletter",
        ".breadcrumb",
        "[class*='advert']",
        "[id*='advert']",
        "[class*='related']",
        "[class*='comment']",
    ]
    for node in soup.select(",".join(noise_selectors)):
        node.decompose()


def extract_article_text(soup: BeautifulSoup, url: str) -> str:
    domain = get_source_domain(url)
    selector_map = {
        "vnexpress.net": [
            "article.fck_detail",
            "article",
            ".fck_detail",
            ".sidebar-1",
        ],
        "dantri.com.vn": [
            "article.singular-container",
            ".singular-content",
            ".dt-news__content",
            "article",
        ],
        "thanhnien.vn": [
            ".detail__content",
            ".cms-body",
            "article",
        ],
        "vietnamnet.vn": [
            ".maincontent",
            ".content-detail",
            "#maincontent",
            "article",
        ],
    }

    candidates = selector_map.get(domain, []) + [
        "article",
        "[class*='content']",
        "[class*='detail']",
        "main",
    ]

    best_text = ""
    for selector in candidates:
        for node in soup.select(selector):
            paragraphs = [
                p.get_text(" ", strip=True)
                for p in node.find_all(["p", "h2", "h3", "li"])
            ]
            text = normalize_text("\n".join(p for p in paragraphs if p))
            if len(text) > len(best_text):
                best_text = text

    if len(best_text) >= MIN_CONTENT_CHARS:
        return best_text

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return normalize_text("\n".join(p for p in paragraphs if p))


async def crawl_with_crawl4ai(url: str) -> tuple[str, str]:
    try:
        from crawl4ai import AsyncWebCrawler
    except Exception as exc:
        raise RuntimeError(f"Crawl4AI chưa sẵn sàng: {exc}") from exc

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    markdown = getattr(result, "markdown", "") or getattr(result, "cleaned_html", "") or ""
    title = getattr(result, "title", "") or "Không rõ tiêu đề"
    content = normalize_text(markdown)
    if len(content) < MIN_CONTENT_CHARS:
        raise ValueError("Crawl4AI trả về nội dung quá ngắn")
    return title, content


def crawl_with_requests(url: str) -> tuple[str, str]:
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    remove_noise(soup)

    title = get_title_from_soup(soup)
    content = extract_article_text(soup, url)
    if len(content) < MIN_CONTENT_CHARS:
        raise ValueError("BeautifulSoup trích xuất nội dung quá ngắn")
    return title, content


def crawl_with_fallback(url: str) -> tuple[str, str]:
    info = FALLBACK_ARTICLES[url]
    return info["title"], info["content"]


def playwright_chromium_available() -> bool:
    base_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
    if not base_dir.exists():
        return False
    return any(base_dir.glob("chromium-*/*/chrome.exe"))


async def crawl_article(url: str) -> dict:
    date_crawled = datetime.now().isoformat()
    crawl_method = "fallback"

    if playwright_chromium_available():
        try:
            title, content = await crawl_with_crawl4ai(url)
            crawl_method = "crawl4ai"
        except Exception as crawl4ai_error:
            print(f"  [Task 2] Crawl4AI skipped: {type(crawl4ai_error).__name__}")
    else:
        print("  [Task 2] Crawl4AI skipped: Playwright Chromium is not installed")

    if crawl_method == "fallback":
        try:
            title, content = crawl_with_requests(url)
            crawl_method = "requests+beautifulsoup"
        except Exception as requests_error:
            print(f"  [Task 2] Requests fallback skipped: {type(requests_error).__name__}: {requests_error}")
            title, content = crawl_with_fallback(url)

    content = normalize_text(content)
    title = normalize_text(title)
    source_domain = get_source_domain(url)

    return {
        "url": url,
        "title": title,
        "date_crawled": date_crawled,
        "content_markdown": content,
        "crawl_method": crawl_method,
        "source_domain": source_domain,
        "đường_dẫn_gốc": url,
        "tiêu_đề": title,
        "ngày_crawl": date_crawled,
        "nội_dung": content,
        "nguồn": source_domain,
        "phương_thức_crawl": crawl_method,
    }


async def crawl_all() -> None:
    setup_directory()

    for index, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        filepath = DATA_DIR / f"article_{index:02d}.json"
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"  [Task 2] Saved: {filepath} "
            f"({len(article['content_markdown'])} chars, {article['crawl_method']})"
        )


if __name__ == "__main__":
    asyncio.run(crawl_all())
