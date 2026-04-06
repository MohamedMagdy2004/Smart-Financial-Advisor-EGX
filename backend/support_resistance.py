"""
Support/Resistance scraper for Mubasher EGX pages.
"""
import logging
import re
from typing import Optional, Dict

from scrapling import Fetcher


logger = logging.getLogger(__name__)


SR_SOURCE_URL = "https://www.mubasher.info/markets/EGX/stocks/{ticker}/support-resistance"


def _to_float(value: str) -> Optional[float]:
    cleaned = value.replace(",", "").strip()
    try:
        return float(cleaned)
    except Exception:
        return None


def _extract_levels_from_text(text: str) -> Dict[str, Optional[float]]:
    normalized = " ".join(text.split())

    support_patterns = [
        r"(?:الدعم|Support)\s*(?:الأول|1)?\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)",
        r"(?:Support 1|S1)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    resistance_patterns = [
        r"(?:المقاومة|Resistance)\s*(?:الأولى|1)?\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)",
        r"(?:Resistance 1|R1)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)",
    ]

    support = None
    resistance = None

    for pattern in support_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            support = _to_float(match.group(1))
            if support is not None:
                break

    for pattern in resistance_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            resistance = _to_float(match.group(1))
            if resistance is not None:
                break

    return {"support": support, "resistance": resistance}


def fetch_support_resistance(ticker: str) -> Dict:
    """
    Scrape support/resistance from Mubasher page for a ticker.

    Returns a normalized structure with values and metadata.
    """
    stock = ticker.upper().strip()
    url = SR_SOURCE_URL.format(ticker=stock)

    try:
        fetcher = Fetcher()
        page = fetcher.get(url)
        text = " ".join(page.css("body *::text").getall())
        levels = _extract_levels_from_text(text)

        support = levels.get("support")
        resistance = levels.get("resistance")

        if support is None and resistance is None:
            logger.warning(f"Support/resistance not found for {stock} at {url}")

        return {
            "ticker": stock,
            "support": support,
            "resistance": resistance,
            "source_url": url,
            "source": "mubasher",
            "status": "ok" if (support is not None or resistance is not None) else "not_found",
        }
    except Exception as exc:
        logger.error(f"Failed to fetch support/resistance for {stock}: {exc}")
        return {
            "ticker": stock,
            "support": None,
            "resistance": None,
            "source_url": url,
            "source": "mubasher",
            "status": "error",
            "error": str(exc),
        }
