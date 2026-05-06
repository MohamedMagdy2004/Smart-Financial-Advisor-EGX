"""
Generate Part 2 financial JSON (technical/trading context) for one EGX ticker.
"""
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import time

from config import OUTPUT_DIR, ALPHA_VANTAGE_API_KEY
import requests
logger = logging.getLogger(__name__)

# إعداد جلسة عمل لتجنب الحظر (Rate Limit)
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
})


def _fetch_from_alpha_vantage(symbol: str, from_date: str) -> Optional[pd.DataFrame]:
    """
    Fallback: Fetch daily stock data from Alpha Vantage.
    NOTE: Alpha Vantage's free tier does NOT support EGX stocks.
    This function is kept for future premium upgrades or different symbols.
    Returns a DataFrame with OHLCV data, or None if fetch fails.
    """
    if not ALPHA_VANTAGE_API_KEY:
        logger.warning("Alpha Vantage API key not configured, cannot fallback")
        return None
    
    try:
        logger.info(f"Fetching data from Alpha Vantage for {symbol}")
        
        # Alpha Vantage endpoint
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,  # EGX tickers may not be supported by free tier
            "apikey": ALPHA_VANTAGE_API_KEY,
            "outputsize": "compact"  # Free tier: last 100 days. Use "full" for premium
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Log response keys for debugging
        logger.debug(f"Alpha Vantage response keys: {list(data.keys())}")
        
        # Check for API errors
        if "Error Message" in data:
            logger.error(f"Alpha Vantage error: {data['Error Message']}")
            if "Invalid API call" in data["Error Message"]:
                logger.error(f"Symbol {symbol} may not be supported by Alpha Vantage free tier (EGX stocks require premium)")
            return None
        
        if "Note" in data:
            logger.warning(f"Alpha Vantage note: {data['Note']}")
            return None
        
        if "Information" in data:
            logger.error(f"Alpha Vantage quota/feature limit: {data['Information']}")
            return None
        
        if "Time Series (Daily)" not in data:
            logger.error(f"No time series data in Alpha Vantage response. Keys present: {list(data.keys())}")
            logger.debug(f"Alpha Vantage response preview: {str(data)[:500]}")
            return None
        
        ts = data["Time Series (Daily)"]
        
        # Parse into DataFrame
        records = []
        for date_str, values in ts.items():
            records.append({
                'Date': pd.to_datetime(date_str),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['5. volume'])
            })
        
        df = pd.DataFrame(records)
        if df.empty:
            logger.error("Alpha Vantage returned empty DataFrame")
            return None
        
        df = df.sort_values('Date').reset_index(drop=True)
        df.set_index('Date', inplace=True)
        
        # Filter by from_date
        from_dt = pd.to_datetime(from_date)
        df = df[df.index >= from_dt]
        
        if df.empty:
            logger.error(f"No data from {from_date} in Alpha Vantage response")
            return None
        
        logger.info(f"Successfully fetched {len(df)} rows from Alpha Vantage for {symbol}")
        return df
    
    except Exception as e:
        logger.error(f"Alpha Vantage fetch failed: {e}")
        return None


def _fetch_from_yfinance(symbol: str, from_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch daily stock data from Yahoo Finance.
    Tries multiple symbol formats (SYMBOL, SYMBOL.CA, etc).
    Returns a DataFrame with OHLCV data, or None if all attempts fail.
    """
    # Try multiple symbol formats for EGX stocks
    symbols_to_try = [
        symbol.upper().strip(),  # COMI
        f"{symbol.upper().strip()}.CA",  # COMI.CA
        f"{symbol.upper().strip()}.EGX",  # COMI.EGX
    ]
    
    for yf_symbol in symbols_to_try:
        try:
            logger.info(f"Trying Yahoo Finance with symbol: {yf_symbol}")
            ticker_obj = yf.Ticker(yf_symbol)
            raw = ticker_obj.history(
                start=from_date,
                end=datetime.today().strftime("%Y-%m-%d"),
                raise_errors=True
            )
            
            if raw.empty:
                logger.warning(f"Yahoo Finance returned empty data for {yf_symbol}")
                continue
            
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            
            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            logger.info(f"Successfully fetched {len(df)} rows from Yahoo Finance ({yf_symbol})")
            return df
        
        except Exception as e:
            logger.debug(f"Yahoo Finance failed for {yf_symbol}: {e}")
            continue
    
    logger.warning(f"All Yahoo Finance attempts failed for {symbol}")
    return None


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# Scoring and indicator helpers (from EGX_Trading_System_4.ipynb)
RISK_PROFILES = {
    "conservative": {
        "strong_buy_threshold": 10,
        "buy_threshold": 6,
        "sell_threshold": -6,
        "strong_sell_threshold": -10,
    },
    "moderate": {
        "strong_buy_threshold": 8,
        "buy_threshold": 4,
        "sell_threshold": -4,
        "strong_sell_threshold": -8,
    },
    "aggressive": {
        "strong_buy_threshold": 6,
        "buy_threshold": 2,
        "sell_threshold": -2,
        "strong_sell_threshold": -6,
    },
}

CONFIDENCE_PRIORS = {
    'STRONG BUY': 65.0,
    'BUY': 55.0,
    'SELL': 55.0,
    'STRONG SELL': 65.0,
    'HOLD': 0.0,
}


def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm <= minus_dm)] = 0
    minus_dm[(minus_dm < plus_dm)] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr14 = tr.rolling(period).mean()
    df['ADX_Plus'] = (plus_dm.rolling(period).mean() / atr14 * 100).fillna(0)
    df['ADX_Minus'] = (minus_dm.rolling(period).mean() / atr14 * 100).fillna(0)

    dx = ((df['ADX_Plus'] - df['ADX_Minus']).abs() / (df['ADX_Plus'] + df['ADX_Minus']).replace(0, np.nan) * 100)
    df['ADX'] = dx.ewm(span=period, adjust=False).mean().fillna(0)
    return df


def _calculate_bollinger(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
    df['BB_Mid'] = df['Close'].rolling(period).mean()
    std = df['Close'].rolling(period).std()
    df['BB_Upper'] = df['BB_Mid'] + (std * std_dev)
    df['BB_Lower'] = df['BB_Mid'] - (std * std_dev)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid'].replace(0, np.nan)
    df['BB_Position'] = ((df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])).clip(0, 1)
    return df


def _calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    direction = np.sign(df['Close'].diff())
    df['OBV'] = (direction * df['Volume']).cumsum()
    df['OBV_MA10'] = df['OBV'].rolling(10).mean()
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    return df


def _detect_trend(df: pd.DataFrame) -> str:
    if len(df) < 50:
        return "UNKNOWN"
    close = df['Close'].iloc[-1]
    sma20 = df['SMA20'].iloc[-1]
    sma50 = df['SMA50'].iloc[-1]
    if np.isnan(close) or np.isnan(sma20):
        return "UNKNOWN"
    if close > sma20 and sma20 > sma50:
        return "UPTREND"
    elif close < sma20 and sma20 < sma50:
        return "DOWNTREND"
    elif abs(close - sma20) / sma20 < 0.02:
        return "CONSOLIDATING"
    else:
        return "SIDEWAYS"


def _safe_iloc(series: pd.Series, idx: int) -> float:
    try:
        val = series.iloc[idx]
        return float(val) if pd.notna(val) else np.nan
    except (IndexError, ValueError):
        return np.nan


def _score_rsi(rsi: float) -> int:
    if np.isnan(rsi):
        return 0
    if rsi < 30:
        return 2
    elif rsi < 45:
        return 1
    elif rsi < 55:
        return 0
    elif rsi < 70:
        return -1
    else:
        return -2


def _score_adx(adx: float, adx_plus: float, adx_minus: float) -> int:
    if np.isnan(adx) or np.isnan(adx_plus) or np.isnan(adx_minus):
        return 0
    if adx_plus > adx_minus:
        if adx > 25:
            return 2
        elif adx > 15:
            return 1
        else:
            return 0
    else:
        if adx > 25:
            return -2
        elif adx > 15:
            return -1
        else:
            return 0


def _score_bollinger(bb_pos: float, regime_multiplier: float) -> int:
    if np.isnan(bb_pos):
        return 0
    if bb_pos < 0.15:
        raw = 2
    elif bb_pos > 0.85:
        raw = -2
    else:
        raw = 0
    return int(round(raw * regime_multiplier))


def _score_volume_spike(vol_now: float, vol_ma20: float) -> int:
    if np.isnan(vol_ma20) or vol_ma20 == 0:
        return 0
    vol_ratio = vol_now / vol_ma20
    if vol_ratio > 2.0:
        return 2
    elif vol_ratio > 1.3:
        return 1
    else:
        return 0


def _score_obv(obv_now: float, obv_ma10: float) -> int:
    if np.isnan(obv_ma10) or obv_ma10 == 0:
        return 0
    if obv_now > obv_ma10:
        return 1
    elif obv_now < obv_ma10:
        return -1
    else:
        return 0


def _score_trend(trend: str) -> int:
    if trend == "UPTREND":
        return 2
    elif trend == "DOWNTREND":
        return -2
    else:
        return 0


def _build_company_payload(symbol: str, risk_profile: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive 7-signal scoring matching EGX_Trading_System_4.ipynb."""
    if len(df) < 5:
        raise ValueError(f"Insufficient data: {len(df)} rows")
    
    if risk_profile not in RISK_PROFILES:
        risk_profile = "moderate"
    profile = RISK_PROFILES[risk_profile]
    
    latest_idx = -1
    price = _safe_iloc(df['Close'], latest_idx)
    rsi = _safe_iloc(df['RSI'], latest_idx)
    adx = _safe_iloc(df['ADX'], latest_idx)
    adx_plus = _safe_iloc(df['ADX_Plus'], latest_idx)
    adx_minus = _safe_iloc(df['ADX_Minus'], latest_idx)
    bb_pos = _safe_iloc(df['BB_Position'], latest_idx)
    obv_now = _safe_iloc(df['OBV'], latest_idx)
    obv_ma10 = _safe_iloc(df['OBV_MA10'], latest_idx)
    vol_now = _safe_iloc(df['Volume'], latest_idx)
    vol_ma20 = _safe_iloc(df['Vol_MA20'], latest_idx)
    sma20 = _safe_iloc(df['SMA20'], latest_idx)
    sma50 = _safe_iloc(df['SMA50'], latest_idx)
    atr = _safe_iloc(df['ATR'], latest_idx)
    
    trend = _detect_trend(df)
    bb_multiplier = 1.0 if trend in ('UPTREND', 'DOWNTREND') else (0.5 if trend == 'CONSOLIDATING' else 0.75)
    
    rsi_score = _score_rsi(rsi)
    adx_score = _score_adx(adx, adx_plus, adx_minus)
    bb_score = _score_bollinger(bb_pos, bb_multiplier)
    vol_score = _score_volume_spike(vol_now, vol_ma20)
    obv_score = _score_obv(obv_now, obv_ma10)
    trend_score = _score_trend(trend)
    
    total_score = rsi_score + adx_score + bb_score + vol_score + obv_score + trend_score
    
    sbt = profile['strong_buy_threshold']
    bt = profile['buy_threshold']
    sst = profile['strong_sell_threshold']
    st = profile['sell_threshold']
    
    if total_score >= sbt:
        decision = 'STRONG BUY'
    elif total_score >= bt:
        decision = 'BUY'
    elif total_score <= sst:
        decision = 'STRONG SELL'
    elif total_score <= st:
        decision = 'SELL'
    else:
        decision = 'HOLD'
    
    if decision in ('BUY', 'STRONG BUY'):
        action_existing = 'HOLD / ADD'
        action_new = 'CONSIDER ENTRY — see position sizing'
    elif decision in ('SELL', 'STRONG SELL'):
        action_existing = 'EXIT position'
        action_new = 'DO NOT ENTER — unfavourable conditions'
    else:
        action_existing = 'HOLD — no change needed'
        action_new = 'WAIT — no clear edge yet'
    
    confidence = CONFIDENCE_PRIORS.get(decision, 50.0)
    confidence_note = f"{decision} signal based on {abs(total_score)}-point composite score from 7 indicators."
    
    return {
        "symbol": symbol,
        "exchange": "EGX (Egyptian Exchange)",
        "analysis_date": df.index[-1].strftime("%Y-%m-%d"),
        "price": {
            "current_EGP": round(price, 2),
            "sma20_EGP": round(sma20, 2) if not np.isnan(sma20) else None,
            "sma50_EGP": round(sma50, 2) if not np.isnan(sma50) else None,
        },
        "trend": trend,
        "signal": decision,
        "action_existing_holders": action_existing,
        "action_new_capital": action_new,
        "confidence_pct": confidence,
        "confidence_note": confidence_note,
        "total_score": int(total_score),
        "max_score": 14,
        "sub_scores": {
            "rsi": rsi_score,
            "adx": adx_score,
            "bollinger": bb_score,
            "volume_spike": vol_score,
            "obv": obv_score,
            "trend": trend_score,
        },
        "indicators": {
            "RSI_14": round(rsi, 1) if not np.isnan(rsi) else None,
            "ADX_14": round(adx, 1) if not np.isnan(adx) else None,
            "ADX_Plus": round(adx_plus, 1) if not np.isnan(adx_plus) else None,
            "ADX_Minus": round(adx_minus, 1) if not np.isnan(adx_minus) else None,
            "BB_Position": round(bb_pos, 2) if not np.isnan(bb_pos) else None,
            "OBV": round(obv_now, 0) if not np.isnan(obv_now) else None,
            "ATR_14_EGP": round(atr, 2) if not np.isnan(atr) else None,
            "ATR_pct_of_price": round((atr / price) * 100, 2) if price and not np.isnan(atr) else None,
        },
        "risk_profile": risk_profile,
        "regime": trend,
        "regime_multipliers": {"bollinger": bb_multiplier},
        "thresholds_used": {
            "strong_buy": sbt,
            "buy": bt,
            "sell": st,
            "strong_sell": sst,
        },
        "position_sizing": {
            "applicable": decision in ("BUY", "STRONG BUY"),
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
            "validation_note": "Lightweight API mode — run EGX_Trading_System_4.ipynb for backtesting.",
        },
        "llm_prompt_summary": (
            f"{symbol}: {decision} ({total_score}/14 points). "
            f"Trend={trend} | RSI={round(rsi, 0) if not np.isnan(rsi) else 'N/A'} | "
            f"ADX={round(adx, 0) if not np.isnan(adx) else 'N/A'} | Risk={risk_profile}"
        ),
    }


def generate_part2_financial_json(
    ticker: str,
    user_risk_profile: str,
    from_date: str = "2024-01-01",
) -> Dict[str, Any]:
    symbol = ticker.upper().strip()
    
    # Try Yahoo Finance first
    logger.info(f"Starting data fetch for {symbol}")
    df = _fetch_from_yfinance(symbol, from_date)
    
    # Fallback to Alpha Vantage if yfinance fails
    if df is None or df.empty:
        logger.warning(f"Yahoo Finance failed, attempting Alpha Vantage fallback for {symbol}")
        df = _fetch_from_alpha_vantage(symbol, from_date)
    
    # If both sources fail, raise error with helpful guidance
    if df is None or df.empty:
        error_msg = (
            f"❌ Could not fetch data for {symbol} from any source.\n"
            f"  • Yahoo Finance: Currently rate-limited (429 Too Many Requests). "
            f"Please wait a few minutes and try again.\n"
            f"  • Alpha Vantage: Free tier does not support EGX stocks. "
            f"(Premium required: https://www.alphavantage.co/premium/)\n"
            f"ℹ️  Recommended: Wait 5-10 minutes and retry."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Calculate indicators
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50, min_periods=10).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["RSI"] = _compute_rsi(df["Close"])
    df = _calculate_adx(df, period=14)
    df = _calculate_bollinger(df, period=20, std_dev=2)
    df = _calculate_obv(df)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - df["Close"].shift(1)).abs(), (df["Low"] - df["Close"].shift(1)).abs()], axis=1).max(axis=1)
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
