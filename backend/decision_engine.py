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
        "user_risk_profile": user_risk_profile,
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


def _call_groq(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY. Set it in environment/.env.")
    system_prompt = (
        "أنت خبير مالي رفيع المستوى ومحلل فني أساسي متخصص في البورصة المصرية (EGX). "
        "مهمتك ليست مجرد سرد أرقام، بل تقديم رؤية استراتيجية شاملة ومستفيضة. "
        "يجب أن يجمع تحليلك بين 'نبض السوق' المستمد من الأخبار (Sentiment Analysis) "
        "وبين 'لغة الأرقام' من المؤشرات الفنية (Technical Indicators). "
        "تحدث بلهجة مهنية واثقة، فسر العلاقات بين المؤشرات، ولا تختصر الإجابة أبداً."
    )

    user_prompt = (
        f"حلل سهم {payload['ticker']} بعمق. اتبع الآتي:\n\n"
        "1) في 'stock_analysis': ركز على شرح السعر الحالي مقارنة بالمتوسطات SMA20 و SMA50 ونظرة عامة على الاتجاه.\n"
        "2) في 'advanced_explanation': ركز **حصرياً** على المستويات الرقمية (فيبوناتشي، دعم، مقاومة) وحجم السيولة. **لا تكرر الكلام المذكور أعلاه**.\n"
        "3) ممنوع منعاً باتاً كتابة أي أكواد JSON أو علامات ``` داخل القيم النصية. اكتب نصاً عادياً فقط.\n"
        "4) الرد يجب أن يكون JSON صالح (Valid JSON) فقط، بدون أي مقدمات أو خاتمة خارج الأقواس.\n\n"
        f"بيانات الدخل: {json.dumps(payload, ensure_ascii=False)}"
    )

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
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
    llm_result = None
    attempt_sizes = [(8, 320), (5, 220), (3, 160)]
    last_http_error = None

    for max_items, text_len in attempt_sizes:
        payload = _build_prompt_payload(
            ticker=ticker,
            news_data=news_data,
            financial_data=enriched_financial,
            user_risk_profile=profile,
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
        "inputs": {
            "news_json": news_json_path,
            "financial_json": financial_json_path,
            "financial_json_enriched": enriched_financial_path,
        },
        "llm_model": GROQ_MODEL,
        "result": llm_result,
    }

    output_file = _save_json(final_output, decision_filename)
    final_output["output_file"] = output_file

    return final_output
