"""
Support/Resistance scraper for Mubasher EGX pages.
Extracts all 5 levels: R2, R1, Pivot, S1, S2
"""
import logging
import re
from typing import Optional, Dict

from scrapling import Fetcher


logger = logging.getLogger(__name__)


SR_SOURCE_URL = "https://www.mubasher.info/markets/EGX/stocks/{ticker}/support-resistance"


def _normalize_arabic_numerals(text: str) -> str:
    """Convert Arabic numerals to English"""
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    english_digits = "0123456789"
    trans = str.maketrans(arabic_digits, english_digits)
    return text.translate(trans)


def _to_float(value: str) -> Optional[float]:
    """Safe float conversion with Arabic numeral support"""
    if not value:
        return None
    
    # Normalize Arabic numerals first
    value = _normalize_arabic_numerals(value)
    
    # Handle European decimal format (comma as decimal): 19,54 → 19.54
    # But preserve thousands separators like 1,000.50
    # Strategy: if comma appears, check if there's a period after it
    # If yes, comma is thousands separator; if no, comma is decimal
    
    if ',' in value:
        # European format likely: swap comma and period if they conflict
        parts = value.split('.')
        if len(parts) == 1:
            # No period found, comma might be decimal separator
            if value.count(',') == 1:
                last_comma_idx = value.rfind(',')
                after_comma = value[last_comma_idx+1:]
                # If 2-3 digits after comma, it's likely decimal (European format)
                if len(after_comma) in (1, 2, 3):
                    value = value.replace(',', '.')
                # Otherwise keep comma as thousands separator and remove it
                else:
                    value = value.replace(',', '')
        # If period exists, comma is thousands separator - just remove it
        else:
            value = value.replace(',', '')
    
    try:
        result = float(value)
        # Sanity check: EGX prices are typically between 0.1 and 500 EGP
        if result < 0.1 or result > 500:
            return result  # Still return it but with warning in status
        return result
    except Exception as e:
        return None


def _extract_all_levels_from_text(text: str) -> Dict[str, Optional[float]]:
    """
    Extract all 5 support/resistance levels from Mubasher page text.
    First tries to extract from JavaScript variable (most reliable).
    Falls back to HTML text parsing if JS extraction fails.
    """
    normalized = " ".join(text.split())
    normalized = _normalize_arabic_numerals(normalized)
    
    levels = {
        "resistance_2": None,
        "resistance_1": None,
        "pivot": None,
        "support_1": None,
        "support_2": None,
    }
    
    print(f"\n{'='*80}")
    print("PARSING ATTEMPT:")
    print(f"{'='*80}")
    print(f"Text length: {len(normalized)} chars\n")
    
    # FIRST: Try to extract from JavaScript variable (most reliable)
    js_pattern = r"window\.midata\.supportResistance\s*=\s*\{([^}]+)\}"
    js_match = re.search(js_pattern, text)
    
    if js_match:
        js_content = js_match.group(1)
        
        # Extract values from JavaScript object
        # Format: 'l1Resistance':'19.61', 'l2Resistance':'19.96', 'pivot':'19.40', etc.
        js_extractions = {
            "resistance_2": r"'l2Resistance':'([0-9.]+)'",
            "resistance_1": r"'l1Resistance':'([0-9.]+)'",
            "pivot": r"'pivot':'([0-9.]+)'",
            "support_1": r"'l1Support':'([0-9.]+)'",
            "support_2": r"'l2Support':'([0-9.]+)'",
        }
        
        for key, pattern in js_extractions.items():
            match = re.search(pattern, js_content)
            if match:
                levels[key] = _to_float(match.group(1))
        
        # If we found all 5 values from JavaScript, return them
        if all(v is not None for v in levels.values()):
            return levels
    
    # FALLBACK: Parse HTML text (for cases where JS parsing fails)
    
    # Patterns that handle multi-line format where number comes after label
    patterns = {
        "resistance_2": [
            # Try to match: "مستوى مقاومة ثان (م ٢)" followed by a number (possibly on next line)
            r"مستوى\s+مقاومة\s+ثان[^0-9]*([0-9]+(?:\.[0-9]+)?)",
            r"م\s*2[^0-9]*([0-9]+(?:\.[0-9]+)?)",
            r"resistance\s*2[^0-9]*([0-9]+(?:\.[0-9]+)?)",
            r"R2[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        ],
        "resistance_1": [
            r"مستوى\s+مقاومة\s+أول[^0-9]*([0-9]+(?:\.[0-9]+)?)",
            r"م\s*1[^0-9]*([0-9]+(?:\.[0-9]+)?)",
            r"resistance\s*1[^0-9]*([0-9]+(?:\.[0-9]+)?)",
            r"R1[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        ],
        "pivot": [
            r"نقطة\s+(?:إ?رتكاز|ارتكاز)[^0-9]*([0-9]+(?:\.[0-9]+)?)",
            r"pivot[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        ],
        "support_1": [
            r"مستوى\s+دعم\s+أول[^0-9]*([0-9]+(?:\.[0-9]+)?)",
            r"د\s*1[^0-9]*([0-9]+(?:\.[0-9]+)?)",
            r"support\s*1[^0-9]*([0-9]+(?:\.[0-9]+)?)",
            r"S1[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        ],
        "support_2": [
            r"مستوى\s+دعم\s+ثان[^0-9]*([0-9]+(?:\.[0-9]+)?)",
            r"د\s*2[^0-9]*([0-9]+(?:\.[0-9]+)?)",
            r"support\s*2[^0-9]*([0-9]+(?:\.[0-9]+)?)",
            r"S2[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        ],
    }
    
    for level_name, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                levels[level_name] = _to_float(match.group(1))
                break
    
    return levels


def fetch_support_resistance(ticker: str) -> Dict:
    """
    Scrape support/resistance from Mubasher page for a ticker.

    Returns all 5 levels (R2, R1, Pivot, S1, S2) with validation.
    """
    stock = ticker.upper().strip()
    url = SR_SOURCE_URL.format(ticker=stock)

    try:
        fetcher = Fetcher()
        page = fetcher.get(url)
        text = " ".join(page.css("body *::text").getall())
        
        levels = _extract_all_levels_from_text(text)

        # Check if any levels were found
        found_any = any(v is not None for v in levels.values())
        
        if not found_any:
            logger.warning(f"Support/resistance levels not found for {stock} at {url}")
        else:
            logger.info(f"Extracted support/resistance for {stock}: {levels}")

        result = {
            "ticker": stock,
            "resistance_2_EGP": levels["resistance_2"],
            "resistance_1_EGP": levels["resistance_1"],
            "pivot_EGP": levels["pivot"],
            "support_1_EGP": levels["support_1"],
            "support_2_EGP": levels["support_2"],
            "source_url": url,
            "source": "mubasher",
            "status": "ok" if found_any else "not_found",
        }
        
        return result
        
    except Exception as exc:
        logger.error(f"Failed to fetch support/resistance for {stock}: {exc}")
        return {
            "ticker": stock,
            "resistance_2_EGP": None,
            "resistance_1_EGP": None,
            "pivot_EGP": None,
            "support_1_EGP": None,
            "support_2_EGP": None,
            "source_url": url,
            "source": "mubasher",
            "status": "error",
            "error": str(exc),
        }
