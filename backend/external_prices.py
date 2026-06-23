import re
from urllib.request import urlopen, Request
from urllib.error import URLError


def fetch_stock_price(ticker: str, timeout: int = 20):
    """Fetch the current price for an EGX stock ticker from Mubasher."""
    symbol = ticker.upper().strip()
    if not symbol:
        return None

    urls = [
        f"https://www.mubasher.info/markets/EGX/stocks/{symbol}/support-resistance",
        f"https://www.mubasher.info/markets/EGX/stocks/{symbol}",
    ]

    for url in urls:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(req, timeout=timeout) as response:
                raw = response.read().decode("utf-8", "ignore")
        except (URLError, TimeoutError, ValueError):
            continue

        preferred_patterns = [
            r"lastPrice\s*[:=]\s*['\"]?([0-9]+(?:\.[0-9]+)?)",
            r"price\s*[:=]\s*['\"]?([0-9]+(?:\.[0-9]+)?)",
            r"currentPrice\s*[:=]\s*['\"]?([0-9]+(?:\.[0-9]+)?)",
            r'\bprice\s*(?:up|down)?(?:-with-icon|-only)?">([0-9]+(?:\.[0-9]+)?)',
            r'\bprice\s*(?:up|down)?(?:-with-icon|-only)?">([0-9]+(?:\.[0-9]+)?)',
            r'\b([0-9]+(?:\.[0-9]+)?)\s*(?:EGP|جنيه)\b',
        ]

        for pattern in preferred_patterns:
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        tokens = re.findall(r'\d+(?:,\d+)?(?:\.\d+)?', raw)
        for token in tokens:
            numeric = token.replace(',', '')
            try:
                value = float(numeric)
            except ValueError:
                continue
            if 1 <= value <= 500:
                return value

    return None
