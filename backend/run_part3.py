"""
CLI runner for Part 3 (no frontend needed).

Usage:
python run_part3.py --ticker COMI --news output/COMI_news.json --financial /path/to/part2.json
"""
import argparse
import json

from decision_engine import generate_final_decision


def ask_risk_answers() -> dict:
    print("Risk profile questions:")
    print("1) Investment horizon? [short/medium/long]")
    horizon = input("   > ").strip().lower() or "medium"

    print("2) Max drawdown tolerance? [low/medium/high]")
    drawdown = input("   > ").strip().lower() or "medium"

    print("3) Trading style? [defensive/balanced/aggressive]")
    style = input("   > ").strip().lower() or "balanced"

    return {
        "investment_horizon": horizon,
        "max_drawdown_tolerance": drawdown,
        "style": style,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Part 3 decision engine")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--news", required=True, help="Path to Part 1 JSON output")
    parser.add_argument("--financial", required=True, help="Path to Part 2 JSON output")
    parser.add_argument("--risk-profile", default=None, choices=["conservative", "moderate", "aggressive"])
    parser.add_argument("--ask-risk-questions", action="store_true")
    args = parser.parse_args()

    risk_answers = ask_risk_answers() if args.ask_risk_questions else None

    result = generate_final_decision(
        ticker=args.ticker,
        news_json_path=args.news,
        financial_json_path=args.financial,
        user_risk_profile=args.risk_profile,
        risk_answers=risk_answers,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
