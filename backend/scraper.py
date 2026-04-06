"""
News Scraping Module
Extracts financial news from mubasher.info
"""
import logging
from datetime import datetime
from scrapling import Fetcher
from config import JUNK_MARKERS, NEWS_SOURCE_URL

logger = logging.getLogger(__name__)


def clean_content(text: str, title: str) -> str:
    """
    Clean article content from junk markers and ads
    
    Args:
        text: Raw article text
        title: Article title (to remove from body)
    
    Returns:
        Cleaned article text or None if too short
    """
    if not text or len(text) < 25:
        return None

    # Remove title from body
    text = text.replace(title, "").strip()

    # Remove junk markers
    for marker in JUNK_MARKERS:
        if marker in text:
            text = text.split(marker)[0]

    # Final cleanup
    cleaned = " ".join(text.split()).strip()
    return cleaned if len(cleaned) > 50 else None


def scrape_news(stock_code: str, company_name: str, max_news: int = 20) -> list:
    """
    Scrape latest news for a company from mubasher.info
    
    Args:
        stock_code: Stock ticker symbol (e.g., "COMI")
        company_name: Company name in Arabic
        max_news: Maximum number of articles to retrieve
    
    Returns:
        List of news articles with metadata
    """
    stock_code = stock_code.upper().strip()
    url = NEWS_SOURCE_URL.format(ticker=stock_code)

    logger.info(f"Scraping news for {company_name} ({stock_code}) from {url}")

    try:
        fetcher = Fetcher()
        page = fetcher.get(url)
        links_elements = page.css('a[href*="/news/"]')

        news_data = []
        seen_urls = set()
        count = 0

        for el in links_elements:
            if count >= max_news:
                break

            title = el.css('::text').get("").strip()
            link = el.attrib.get('href', '')

            # Validate title and check for duplicates
            if len(title) < 25 or link in seen_urls:
                continue

            full_url = "https://www.mubasher.info" + link if link.startswith('/') else link
            seen_urls.add(link)

            logger.debug(f"Processing article: {title[:50]}...")

            try:
                detail = fetcher.get(full_url)

                # Extract date
                date = detail.css('.news-details__date::text, time::text, .date::text').get("غير محدد").strip()

                # Extract article body
                raw_body = ""
                for selector in ['.news-details__content', '.article-content', '#article-body', 'article']:
                    parts = detail.css(selector + '::text').getall()
                    if parts:
                        combined = " ".join(parts).strip()
                        if len(combined) > 100:
                            raw_body = combined
                            break

                # Fallback to paragraph extraction
                if not raw_body:
                    raw_body = " ".join(detail.css('p::text').getall())

                # Clean content
                body = clean_content(raw_body, title)
                if not body:
                    logger.debug(f"Skipped article (too short): {title[:30]}")
                    continue

                # Create news item
                news_item = {
                    "headline": title,
                    "date": date,
                    "body": body,
                    "link": full_url,
                    "ticker": stock_code,
                    "company": company_name,
                    "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                news_data.append(news_item)
                count += 1
                logger.info(f"✓ Scraped article {count}: {title[:40]}...")

            except Exception as e:
                logger.warning(f"Error scraping article: {e}")
                continue

        logger.info(f"Successfully scraped {len(news_data)} articles for {stock_code}")
        return news_data

    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        return []


def validate_news_articles(articles: list) -> list:
    """
    Validate and filter articles
    
    Args:
        articles: List of news articles
    
    Returns:
        Validated articles
    """
    validated = []
    for article in articles:
        required_fields = ["headline", "body", "link", "ticker", "company", "date"]
        if all(field in article and article[field] for field in required_fields):
            validated.append(article)
        else:
            logger.warning(f"Skipped invalid article: {article.get('headline', 'Unknown')}")
    
    return validated
