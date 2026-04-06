"""
Generate Part 2 financial JSON (technical/trading context) for one EGX ticker.
"""
import json
import os
from datetime import datetime
from typing import Any, Dict
import logging
import numpy as np
import pandas as pd
import yfinance as yf

from config import OUTPUT_DIR
import requests  # أضف هذا السطر
logger = logging.getLogger(__name__)
# إعداد جلسة عمل لتجنب الحظر (Rate Limit)
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
})

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _trend_from_sma(close: float, sma20: float, sma50: float) -> str:
    if np.isnan(close) or np.isnan(sma20):
        return "UNKNOWN"
    if close > sma20 and (np.isnan(sma50) or sma20 > sma50):
        return "UPTREND"
    if close < sma20 and (np.isnan(sma50) or sma20 < sma50):
        return "DOWNTREND"
    return "SIDEWAYS"


def _build_company_payload(symbol: str, risk_profile: str, df: pd.DataFrame) -> Dict[str, Any]:
    latest = df.iloc[-1]
    close = float(latest["Close"])
    sma20 = float(latest.get("SMA20", np.nan))
    sma50 = float(latest.get("SMA50", np.nan))
    rsi = float(latest.get("RSI", np.nan))
    atr = float(latest.get("ATR", np.nan))

    trend = _trend_from_sma(close, sma20, sma50)

    rsi_score = 1 if rsi < 35 else (-1 if rsi > 70 else 0)
    trend_score = 1 if trend == "UPTREND" else (-1 if trend == "DOWNTREND" else 0)
    total_score = rsi_score + trend_score

    if total_score >= 2:
        signal = "STRONG BUY"
    elif total_score >= 1:
        signal = "BUY"
    elif total_score <= -2:
        signal = "STRONG SELL"
    elif total_score <= -1:
        signal = "SELL"
    else:
        signal = "HOLD"

    if signal in ("BUY", "STRONG BUY"):
        action_existing = "HOLD / ADD"
        action_new = "CONSIDER ENTRY — wait for confirmation"
    elif signal in ("SELL", "STRONG SELL"):
        action_existing = "REDUCE / EXIT"
        action_new = "DO NOT ENTER"
    else:
        action_existing = "HOLD"
        action_new = "WAIT"

    confidence = 55.0
    if signal in ("STRONG BUY", "STRONG SELL"):
        confidence = 65.0
    elif signal == "HOLD":
        confidence = 50.0

    return {
        "symbol": symbol,
        "exchange": "EGX (Egyptian Exchange)",
        "analysis_date": df.index[-1].strftime("%Y-%m-%d"),
        "price": {
            "current_EGP": round(close, 2),
            "sma20_EGP": round(sma20, 2) if not np.isnan(sma20) else None,
            "sma50_EGP": round(sma50, 2) if not np.isnan(sma50) else None,
            "support_EGP": None,
            "resistance_EGP": None,
            "sr_source": "mubasher_enrichment_pending",
        },
        "trend": trend,
        "signal": signal,
        "action_existing_holders": action_existing,
        "action_new_capital": action_new,
        "confidence_pct": confidence,
        "confidence_note": "Baseline technical confidence from RSI + trend model.",
        "total_score": total_score,
        "max_score": 2,
        "sub_scores": {
            "rsi": rsi_score,
            "trend": trend_score,
        },
        "risk_profile": risk_profile,
        "indicators": {
            "RSI_14": round(rsi, 1) if not np.isnan(rsi) else None,
            "ATR_14_EGP": round(atr, 2) if not np.isnan(atr) else None,
            "ATR_pct_of_price": round((atr / close) * 100, 2) if close and not np.isnan(atr) else None,
        },
        "position_sizing": {
            "applicable": signal in ("BUY", "STRONG BUY"),
            "suggested_shares": None,
            "position_cost_EGP": None,
            "stop_loss_EGP": None,
            "take_profit_EGP": None,
            "capital_at_risk_EGP": None,
            "risk_pct_of_capital": None,
            "within_capital_limit": None,
        },
        "backtest": {
            "signal_validation": "UNVALIDATED",
            "validation_note": "Lightweight API mode without notebook backtesting.",
        },
        "llm_prompt_summary": (
            f"{symbol} trend is {trend}, RSI is {round(rsi, 1) if not np.isnan(rsi) else 'N/A'}, "
            f"signal is {signal}, risk profile is {risk_profile}."
        ),
    }


def generate_part2_financial_json(
    ticker: str,
    user_risk_profile: str,
    from_date: str = "2024-01-01",
) -> Dict[str, Any]:
    symbol = ticker.upper().strip()
    yf_symbol = f"{symbol}.CA"
    ticker_obj = yf.Ticker(yf_symbol)
    try:
        raw = ticker_obj.history(
            start=from_date,
            end=datetime.today().strftime("%Y-%m-%d"),
            raise_errors=True
        )
    except Exception as e:
        logger.error(f"Yahoo Finance error for {yf_symbol}: {e}")
        raise RuntimeError(f"Could not fetch data for {yf_symbol}. Yahoo Finance might be rate-limiting your IP.")
    if raw.empty:
        raise RuntimeError(f"No market data returned for {yf_symbol}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50, min_periods=10).mean()
    df["RSI"] = _compute_rsi(df["Close"])
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df = df.dropna(subset=["Close", "RSI", "SMA20"]) 

    if df.empty:
        raise RuntimeError("Insufficient indicator rows after preprocessing")

    company_payload = _build_company_payload(symbol, user_risk_profile, df)
    part2_json = {
        "part": "financial_analysis",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "period": f"{from_date} to {datetime.today().strftime('%Y-%m-%d')}",
        "user_risk_profile": user_risk_profile,
        "symbols_requested": [symbol],
        "symbols_processed": [symbol],
        "companies": [company_payload],
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_part2_financial_{ts}.json"
    full_path = os.path.join(OUTPUT_DIR, filename)
    with open(full_path, "w", encoding="utf-8") as handle:
        json.dump(part2_json, handle, ensure_ascii=False, indent=2)

    return {
        "output_file": full_path,
        "payload": part2_json,
    }
