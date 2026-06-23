import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from external_prices import fetch_stock_price


class DummyResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.payload.encode("utf-8")


def test_fetch_stock_price_parses_current_price_from_yahoo():
    html = '<script>"regularMarketPrice":{"raw":82.45}</script>'

    with patch("external_prices.urlopen", return_value=DummyResponse(html)):
        assert fetch_stock_price("COMI") == 82.45


def test_fetch_stock_price_prefers_last_price_from_mubasher_support_page():
    html = '<div>previous price 135.97</div><script>var lastPrice = "135.80";</script>'

    with patch("external_prices.urlopen", return_value=DummyResponse(html)):
        assert fetch_stock_price("COMI") == 135.80
