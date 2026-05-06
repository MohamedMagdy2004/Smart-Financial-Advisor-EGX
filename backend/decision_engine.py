"""
Part 3: Decision engine that merges news + financial outputs and asks Groq LLM
for actionable recommendations.
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from config import OUTPUT_DIR, GROQ_API_KEY, GROQ_MODEL
from support_resistance import fetch_support_resistance


logger = logging.getLogger(__name__)


def _safe_read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(data: Any, filename: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_path = os.path.join(OUTPUT_DIR, filename)
    with open(full_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return full_path


def _derive_risk_profile_from_answers(answers: Optional[Dict[str, str]]) -> str:
    if not answers:
        return "moderate"

    score = 0
    horizon = answers.get("investment_horizon", "medium")
    drawdown = answers.get("max_drawdown_tolerance", "medium")
    style = answers.get("style", "balanced")

    if horizon == "short":
        score += 2
    elif horizon == "long":
        score -= 1

    if drawdown == "high":
        score += 2
    elif drawdown == "low":
        score -= 2

    if style == "aggressive":
        score += 2
    elif style == "defensive":
        score -= 2

    if score >= 2:
        return "aggressive"
    if score <= -2:
        return "conservative"
    return "moderate"


HORIZON_ANALYSIS_WEIGHTS = {
    "short": {"technical_weight": 0.70, "fundamental_weight": 0.30, "analysis_lead": "technicals"},
    "medium": {"technical_weight": 0.40, "fundamental_weight": 0.60, "analysis_lead": "fundamentals"},
    "long": {"technical_weight": 0.20, "fundamental_weight": 0.80, "analysis_lead": "fundamentals"},
}

DRAWDOWN_MIDPOINTS = {
    "low": 0.05,
    "medium": 0.15,
    "high": 0.25,
}

REBALANCING_FREQUENCY_BY_HORIZON = {
    "short": "weekly",
    "medium": "monthly",
    "long": "quarterly",
}

QUERY_RESPONSE_RULES = {
    "SIMPLE_FACT": "Answer in one sentence only. No indicators, no report, no extra commentary.",
    "QUICK_SUMMARY": "Answer in 3-5 lines maximum. Include price, trend direction, and one key signal only.",
    "FULL_ANALYSIS": "Return the full structured analysis with fundamentals and technicals, keeping JSON valid.",
    "COMPARISON": "Return side-by-side key metrics only. No long narrative.",
    "NEWS_ONLY": "Return bullet points of recent news items only. No technical analysis.",
}


def _normalize_choice(value: Optional[str], default: str) -> str:
    normalized = str(value or default).strip().lower()
    return normalized or default


def _analysis_weights_for_horizon(horizon: Optional[str]) -> Dict[str, Any]:
    normalized = _normalize_choice(horizon, "medium")
    return HORIZON_ANALYSIS_WEIGHTS.get(normalized, HORIZON_ANALYSIS_WEIGHTS["medium"])


def _drawdown_midpoint(drawdown: Optional[str]) -> float:
    normalized = _normalize_choice(drawdown, "medium")
    return DRAWDOWN_MIDPOINTS.get(normalized, DRAWDOWN_MIDPOINTS["medium"])


def _derive_rebalancing_frequency(horizon: Optional[str], trading_style: Optional[str]) -> str:
    normalized_horizon = _normalize_choice(horizon, "medium")
    if normalized_horizon in REBALANCING_FREQUENCY_BY_HORIZON:
        return REBALANCING_FREQUENCY_BY_HORIZON[normalized_horizon]

    normalized_style = _normalize_choice(trading_style, "balanced")
    if normalized_style == "aggressive":
        return "weekly"
    if normalized_style == "defensive":
        return "quarterly"
    return "monthly"


def _derive_stop_loss(entry_price: Optional[float], drawdown_midpoint: float) -> Optional[float]:
    if entry_price is None:
        return None
    try:
        return round(float(entry_price) * (1 - drawdown_midpoint), 2)
    except (TypeError, ValueError):
        return None


def _build_investor_profile_context(
    user_risk_profile: str,
    risk_answers: Optional[Dict[str, str]],
    target_company: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    answers = risk_answers or {}
    horizon = _normalize_choice(answers.get("investment_horizon"), "medium")
    drawdown = _normalize_choice(answers.get("max_drawdown_tolerance"), "medium")
    trading_style = _normalize_choice(answers.get("style"), "balanced")
    rebalancing = _normalize_choice(answers.get("rebalancing_frequency"), "") or _derive_rebalancing_frequency(horizon, trading_style)
    weights = _analysis_weights_for_horizon(horizon)
    drawdown_midpoint = _drawdown_midpoint(drawdown)

    entry_price = None
    if target_company:
        entry_price = (target_company.get("price") or {}).get("current_EGP")

    stop_loss = _derive_stop_loss(entry_price, drawdown_midpoint)
    analysis_focus = "fundamentals" if weights["analysis_lead"] == "fundamentals" else "technicals"

    block_lines = [
        "---INVESTOR PROFILE---",
        f"Risk Profile: {user_risk_profile}",
        f"Investment Horizon: {horizon} → Favor {analysis_focus} analysis",
        f"Technical Analysis Weight: {weights['technical_weight']:.2f}",
        f"Fundamental Analysis Weight: {weights['fundamental_weight']:.2f}",
        f"Drawdown Tolerance: {drawdown} → Stop Loss target: {stop_loss if stop_loss is not None else 'N/A'} EGP",
        f"Trading Style: {trading_style}",
        f"Rebalancing Frequency: {rebalancing}",
        "----------------------",
    ]

    return {
        "risk_profile": user_risk_profile,
        "investment_horizon": horizon,
        "drawdown_tolerance": drawdown,
        "trading_style": trading_style,
        "rebalancing_frequency": rebalancing,
        "technical_weight": weights["technical_weight"],
        "fundamental_weight": weights["fundamental_weight"],
        "analysis_focus": analysis_focus,
        "analysis_lead": weights["analysis_lead"],
        "drawdown_midpoint": drawdown_midpoint,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "block": "\n".join(block_lines),
    }


def _summarize_news(news_items: List[Dict]) -> Dict[str, Any]:
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    impacts = {"high": 0, "medium": 0, "low": 0}

    for item in news_items:
        sentiment = str(item.get("sentiment", "neutral")).lower()
        impact = str(item.get("impact_level", "low")).lower()
        if sentiment in counts:
            counts[sentiment] += 1
        if impact in impacts:
            impacts[impact] += 1

    return {
        "articles": len(news_items),
        "sentiment_counts": counts,
        "impact_counts": impacts,
    }


def _truncate_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _impact_rank(item: Dict[str, Any]) -> int:
    impact = str(item.get("impact_level", "low")).lower()
    if impact == "high":
        return 3
    if impact == "medium":
        return 2
    return 1


def _compact_news_items(news_items: List[Dict[str, Any]], max_items: int, max_text_length: int) -> List[Dict[str, Any]]:
    sorted_items = sorted(news_items, key=_impact_rank, reverse=True)
    selected = sorted_items[:max_items]
    compact = []
    for row in selected:
        compact.append(
            {
                "date": row.get("news_date"),
                "headline": _truncate_text(row.get("headline"), max_text_length),
                "event_type": row.get("event_type"),
                "sentiment": row.get("sentiment"),
                "impact_level": row.get("impact_level"),
                "short_summary": _truncate_text(row.get("short_summary"), max_text_length),
            }
        )
    return compact


def _compact_financial_company(company: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not company:
        return None

    indicators = company.get("indicators", {})
    price = company.get("price", {})
    return {
        "symbol": company.get("symbol"),
        "analysis_date": company.get("analysis_date"),
        "trend": company.get("trend"),
        "signal": company.get("signal"),
        "confidence_pct": company.get("confidence_pct"),
        "risk_profile": company.get("risk_profile"),
        "action_existing_holders": company.get("action_existing_holders"),
        "action_new_capital": company.get("action_new_capital"),
        "price": {
            "current_EGP": price.get("current_EGP"),
            "sma20_EGP": price.get("sma20_EGP"),
            "sma50_EGP": price.get("sma50_EGP"),
            "support_EGP": price.get("support_EGP"),
            "resistance_EGP": price.get("resistance_EGP"),
        },
        "indicators": {
            "RSI_14": indicators.get("RSI_14"),
            "ATR_14_EGP": indicators.get("ATR_14_EGP"),
            "ATR_pct_of_price": indicators.get("ATR_pct_of_price"),
        },
        "llm_prompt_summary": _truncate_text(company.get("llm_prompt_summary"), 320),
    }


def enrich_financial_with_mubasher_levels(financial_data: Dict[str, Any]) -> Dict[str, Any]:
    companies = financial_data.get("companies", [])
    for company in companies:
        symbol = company.get("symbol")
        if not symbol:
            continue
        levels = fetch_support_resistance(symbol)

        company.setdefault("price", {})
        if levels.get("support") is not None:
            company["price"]["support_EGP"] = levels["support"]
        if levels.get("resistance") is not None:
            company["price"]["resistance_EGP"] = levels["resistance"]

        company["price"]["sr_source"] = levels.get("source", "mubasher")
        company["price"]["sr_source_url"] = levels.get("source_url")
        company["price"]["sr_status"] = levels.get("status")

    financial_data["sr_enriched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return financial_data


def _build_prompt_payload(
    ticker: str,
    news_data: List[Dict[str, Any]],
    financial_data: Dict[str, Any],
    user_risk_profile: str,
    risk_context: Dict[str, Any],
    query_type: str = "FULL_ANALYSIS",
    query_text: Optional[str] = None,
    max_news_items: int = 8,
    max_text_length: int = 320,
) -> Dict[str, Any]:
    target_company = None
    for company in financial_data.get("companies", []):
        if company.get("symbol", "").upper() == ticker.upper():
            target_company = company
            break

    company_news = [row for row in news_data if row.get("ticker", "").upper() == ticker.upper()]
    compact_news = _compact_news_items(
        news_items=company_news,
        max_items=max_news_items,
        max_text_length=max_text_length,
    )

    return {
        "ticker": ticker.upper(),
        "query_type": query_type,
        "query_text": query_text,
        "response_format_rule": QUERY_RESPONSE_RULES.get(query_type, QUERY_RESPONSE_RULES["FULL_ANALYSIS"]),
        "user_risk_profile": user_risk_profile,
        "investor_profile": risk_context,
        "news_summary": _summarize_news(company_news),
        "news_items_total": len(company_news),
        "news_items_used": len(compact_news),
        "news_items": compact_news,
        "financial_company": _compact_financial_company(target_company),
        "output_schema": {
            "stock_analysis": "string",
            "decision_translator": {
                "buy_or_not": "BUY|HOLD|SELL",
                "simple_reason": "string",
                "clear_recommendations": ["string"],
            },
            "scenario_analysis": [
                {
                    "scenario": "If price breaks resistance",
                    "action": "BUY|HOLD|SELL",
                    "reason": "string",
                }
            ],
            "simplified_explanation": "string",
            "advanced_explanation": "string",
            "risk_warning": "string",
        },
    }


def build_groq_prompt(payload: Dict[str, Any]) -> Dict[str, Any]:
    investor_profile = payload.get("investor_profile", {})
    system_prompt = (
        "أنت خبير مالي رفيع المستوى ومحلل فني وأساسي متخصص في البورصة المصرية (EGX). "
        "مهمتك ليست مجرد سرد أرقام، بل تقديم رؤية استراتيجية شاملة ومستفيضة. "
        "يجب أن تجمع بين الأخبار والمؤشرات الفنية والبيانات الأساسية، لكنك يجب أن تلتزم بحرفية بقيود نوع السؤال. "
        "إذا كان الاستثمار متوسط أو طويل الأجل، فابدأ بالأساسيات ولا تضع المؤشرات الفنية في المقدمة."
    )

    user_prompt = (
        f"QUERY_TYPE: {payload.get('query_type', 'FULL_ANALYSIS')}\n"
        f"RESPONSE FORMAT RULE: {payload.get('response_format_rule', QUERY_RESPONSE_RULES['FULL_ANALYSIS'])}\n\n"
        f"{investor_profile.get('block', '')}\n\n"
        f"The model MUST use this investor profile block to shape its recommendation. "
        f"If horizon is medium or long, do NOT lead with technical indicators; lead with fundamentals first.\n\n"
        f"حلل سهم {payload['ticker']} وفق نوع السؤال التالي: {payload.get('query_text') or 'طلب تحليل استثماري'}\n\n"
        "1) في stock_analysis: استخدم مزيج المؤشرات مع وزن يتوافق مع أفق الاستثمار.\n"
        "2) إذا كان الأفق متوسطاً أو طويلاً، ابدأ بالتحليل الأساسي ثم ادعم بالفني.\n"
        "3) في advanced_explanation: ركز على المستويات الرقمية والدعم والمقاومة وحجم السيولة.\n"
        "4) ممنوع منعاً باتاً كتابة أي أكواد JSON أو علامات ``` داخل القيم النصية. اكتب نصاً عادياً فقط.\n"
        "5) الرد يجب أن يكون JSON صالح (Valid JSON) فقط، بدون أي مقدمات أو خاتمة خارج الأقواس.\n\n"
        f"بيانات الدخل: {json.dumps(payload, ensure_ascii=False)}"
    )

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }


def _call_groq(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY. Set it in environment/.env.")
    prompt_bundle = build_groq_prompt(payload)

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "temperature": 0.1,
            "messages": prompt_bundle["messages"],
        },
        timeout=120,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:].strip()
    content = content.strip("`").strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Groq response was not valid JSON; wrapping raw text.")
        return {
            "stock_analysis": content,
            "decision_translator": {
                "buy_or_not": "ANALYSIS READY",
                "simple_reason": content[:500],
                "clear_recommendations": ["Review the detailed analysis below."],
            },
            "scenario_analysis": [],
            "simplified_explanation": content,
            "advanced_explanation": content,
            "risk_warning": "Warning: Analysis generated but JSON structure was invalid.",
        }


def generate_final_decision(
    ticker: str,
    news_json_path: str,
    financial_json_path: str,
    user_risk_profile: Optional[str] = None,
    risk_answers: Optional[Dict[str, str]] = None,
    query_type: str = "FULL_ANALYSIS",
    query_text: Optional[str] = None,
) -> Dict[str, Any]:
    ticker = ticker.upper().strip()

    news_data = _safe_read_json(news_json_path)
    financial_data = _safe_read_json(financial_json_path)

    if not isinstance(news_data, list):
        raise ValueError("Part 1 news JSON must be a list of analyzed news objects.")
    if not isinstance(financial_data, dict):
        raise ValueError("Part 2 financial JSON must be an object with companies list.")

    profile = user_risk_profile or _derive_risk_profile_from_answers(risk_answers)

    enriched_financial = enrich_financial_with_mubasher_levels(financial_data)
    target_company = None
    for company in enriched_financial.get("companies", []):
        if company.get("symbol", "").upper() == ticker.upper():
            target_company = company
            break

    risk_context = _build_investor_profile_context(profile, risk_answers, target_company)
    llm_result = None
    attempt_sizes = [(8, 320), (5, 220), (3, 160)]
    last_http_error = None

    for max_items, text_len in attempt_sizes:
        payload = _build_prompt_payload(
            ticker=ticker,
            news_data=news_data,
            financial_data=enriched_financial,
            user_risk_profile=profile,
            risk_context=risk_context,
            query_type=query_type,
            query_text=query_text,
            max_news_items=max_items,
            max_text_length=text_len,
        )
        try:
            llm_result = _call_groq(payload)
            break
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 413:
                logger.warning(
                    "Groq payload too large (413). Retrying with smaller payload size: max_items=%s, text_len=%s",
                    max_items,
                    text_len,
                )
                last_http_error = exc
                continue
            raise

    if llm_result is None:
        raise RuntimeError(
            "Groq request still exceeded payload size after retries"
        ) from last_http_error

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    financial_filename = f"{ticker}_financial_enriched_{timestamp}.json"
    decision_filename = f"{ticker}_final_decision_{timestamp}.json"

    enriched_financial_path = _save_json(enriched_financial, financial_filename)

    final_output = {
        "part": "final_decision",
        "ticker": ticker,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_risk_profile": profile,
        "query_type": query_type,
        "inputs": {
            "news_json": news_json_path,
            "financial_json": financial_json_path,
            "financial_json_enriched": enriched_financial_path,
        },
        "llm_model": GROQ_MODEL,
        "prompt_debug": build_groq_prompt(payload),
        "result": llm_result,
    }

    output_file = _save_json(final_output, decision_filename)
    final_output["output_file"] = output_file

    return final_output
