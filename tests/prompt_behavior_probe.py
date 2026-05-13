"""Local probe for prompt composition and query routing.

Run from the backend directory:
/bin/python3 prompt_behavior_probe.py
"""

from __future__ import annotations

import re

import chat_orchestrator as co
from decision_engine import _build_investor_profile_context, build_groq_prompt


class _FakeRow(dict):
    pass


class _FakeIloc:
    def __init__(self, row):
        self._row = row

    def __getitem__(self, index):
        return self._row


class _FakeFrame:
    def __init__(self, row):
        self._row = row
        self.empty = False

    @property
    def iloc(self):
        return _FakeIloc(self._row)


def _sentence_count(text: str) -> int:
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DECIMAL>", text)
    fragments = [part.strip() for part in re.split(r"(?<=[.!?])\s+", protected) if part.strip()]
    return len(fragments)


def main() -> None:
    risk_answers = {
        "investment_horizon": "medium",
        "max_drawdown_tolerance": "medium",
        "style": "balanced",
        "rebalancing_frequency": "monthly",
    }
    fake_company = {"price": {"current_EGP": 100.0}}
    risk_context = _build_investor_profile_context("moderate", risk_answers, fake_company)
    payload = {
        "ticker": "EAST",
        "query_type": "FULL_ANALYSIS",
        "query_text": "analyze EAST for me",
        "response_format_rule": "Return the full structured analysis with fundamentals and technicals, keeping JSON valid.",
        "user_risk_profile": "moderate",
        "investor_profile": risk_context,
        "news_summary": {"articles": 5},
        "news_items_total": 5,
        "news_items_used": 5,
        "news_items": [{"headline": "Sample headline"}],
        "financial_company": fake_company,
        "output_schema": {
            "stock_analysis": "string",
            "decision_translator": {
                "buy_or_not": "BUY|HOLD|SELL",
                "simple_reason": "string",
                "clear_recommendations": ["string"],
            },
            "scenario_analysis": [],
            "simplified_explanation": "string",
            "advanced_explanation": "string",
            "risk_warning": "string",
        },
    }

    prompt_bundle = build_groq_prompt(payload)
    print("=== SYSTEM PROMPT ===")
    print(prompt_bundle["system_prompt"])
    print()
    print("=== USER PROMPT ===")
    print(prompt_bundle["user_prompt"])
    print()
    print("Investor profile block present:", "---INVESTOR PROFILE---" in prompt_bundle["user_prompt"])
    print("Query type present:", "QUERY_TYPE: FULL_ANALYSIS" in prompt_bundle["user_prompt"])
    print("Stop loss present:", "Stop Loss target:" in prompt_bundle["user_prompt"])
    print()

    original_fetch = co._fetch_from_yfinance
    try:
        co._fetch_from_yfinance = lambda ticker, from_date="2024-01-01": _FakeFrame({"Close": 12.34, "Volume": 567890})
        simple_fact_message = "what is the price of EAST today?"
        simple_fact_reply = co._format_simple_fact_reply("EAST", simple_fact_message)
        print("=== SIMPLE FACT REPLY ===")
        print(simple_fact_reply)
        print("Query type:", co.classify_query_type(simple_fact_message))
        print("Sentence count:", _sentence_count(simple_fact_reply))
        print()

        full_analysis_message = "analyze EAST for me"
        fake_decision = {
            "result": {
                "stock_analysis": "RSI is 55 and SMA20 is above SMA50. Recommendation: BUY.",
                "decision_translator": {
                    "buy_or_not": "BUY",
                    "simple_reason": "Recommendation: BUY because RSI and SMA structure are supportive.",
                    "clear_recommendations": ["Watch RSI", "Respect SMA support"],
                },
                "advanced_explanation": "Advanced view: SMA20 stays above SMA50, reinforcing the trend.",
                "scenario_analysis": [],
                "risk_warning": "Risk remains moderate.",
            }
        }
        analysis_reply = co._format_full_analysis_reply("EAST", fake_decision, chat_history=None)
        print("=== FULL ANALYSIS REPLY ===")
        print(analysis_reply)
        print("Query type:", co.classify_query_type(full_analysis_message))
        print("Contains RSI:", "RSI" in analysis_reply)
        print("Contains SMA:", "SMA" in analysis_reply)
        print("Contains recommendation:", "Recommendation" in analysis_reply or "BUY" in analysis_reply)
    finally:
        co._fetch_from_yfinance = original_fetch


if __name__ == "__main__":
    main()
