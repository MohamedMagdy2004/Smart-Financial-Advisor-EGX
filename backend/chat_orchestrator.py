"""
Chat orchestration for end-to-end pipeline:
user message -> infer ticker -> part1 -> part2 -> final decision.
Updated to prioritize professional persona in chat reply.
FIXED: Added confidence threshold + vague message detection to prevent random ticker guessing.
"""
import json
import logging
import os
import re
from typing import Any, Dict, Optional

import requests

from analyzer import analyze_news_batch, save_results
from config import COMPANIES, GROQ_API_KEY, GROQ_MODEL
from decision_engine import QUERY_RESPONSE_RULES, _build_investor_profile_context, generate_final_decision
from part2_generator import generate_part2_financial_json, _fetch_from_yfinance
from scraper import scrape_news, validate_news_articles

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MIN_CONFIDENCE_THRESHOLD = 0.65  # Reject LLM inference below this confidence


def _known_tickers() -> set:
    return {value[1] for value in COMPANIES.values()}


def _load_latest_part2_snapshot(ticker: str) -> Optional[Dict[str, Any]]:
    output_dir = os.environ.get("OUTPUT_DIR", "output")
    prefix = f"{ticker.upper()}_part2_financial_"
    try:
        candidates = [
            os.path.join(output_dir, name)
            for name in os.listdir(output_dir)
            if name.startswith(prefix) and name.endswith(".json")
        ]
    except Exception:
        return None

    if not candidates:
        return None

    latest = max(candidates, key=os.path.getmtime)
    try:
        with open(latest, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {"output_file": latest, "payload": payload}
    except Exception:
        return None


def _extract_tickers_from_message(user_message: str) -> list:
    text = user_message.upper()
    tickers = []
    for ticker in sorted(_known_tickers()):
        if re.search(rf"\b{re.escape(ticker)}\b", text):
            tickers.append(ticker)
    return tickers


def classify_query_type(user_message: str) -> str:
    text = user_message.strip().lower()
    tickers = _extract_tickers_from_message(user_message)

    if not tickers:
        return "GENERAL_CHAT"

    comparison_markers = ["compare", "comparison", "vs", "versus", "مقارنة", "قارن", "بين"]
    if len(tickers) >= 2 or any(marker in text for marker in comparison_markers):
        return "COMPARISON"

    quick_summary_priority_markers = ["إيه أخبار", "ايه اخبار", "what's up with", "how is"]
    if any(marker in text for marker in quick_summary_priority_markers):
        return "QUICK_SUMMARY"

    news_markers = ["news", "news on", "any news", "أخبار", "خبر", "اخبار", "what's new", "what is new"]
    if any(marker in text for marker in news_markers):
        return "NEWS_ONLY"

    simple_fact_markers = [
        "price", "current price", "volume", "market open", "market close",
        "when does market open", "when is market open", "opening time", "closing time",
        "سعر", "حجم التداول", "افتتاح", "إغلاق", "يفتح", "يغلق", "كم السعر", "كم حجم"
    ]
    if any(marker in text for marker in simple_fact_markers):
        return "SIMPLE_FACT"

    quick_summary_markers = [
        "how is", "how's", "is it up", "is it down", "up or down", "doing today",
        "summary", "quick summary", "short summary", "أداء", "عامل", "ما الوضع", "ملخص"
    ]
    if any(marker in text for marker in quick_summary_markers):
        return "QUICK_SUMMARY"

    full_analysis_markers = [
        "analyze", "analyse", "analysis", "report", "should i buy", "should i sell",
        "buy", "sell", "recommend", "recommendation", "technical", "fundamental",
        "حلل", "تحليل", "تقرير", "أنصح", "أشتري", "اشتري", "بيع", "توصية"
    ]
    if any(marker in text for marker in full_analysis_markers):
        return "FULL_ANALYSIS"

    return "FULL_ANALYSIS"


def _format_simple_fact_reply(ticker: str, user_message: str) -> str:
    text = user_message.lower()
    if any(marker in text for marker in ["market open", "when does market open", "when is market open", "افتتاح", "يفتح"]):
        return "The EGX market usually opens in the morning session; check the exchange schedule for the exact session time today."

    df = _fetch_from_yfinance(ticker, from_date="2024-01-01")
    if df is None or df.empty:
        return f"I couldn't fetch a live quote for {ticker} right now."

    latest = df.iloc[-1]
    price = float(latest.get("Close", 0.0))
    volume = int(latest.get("Volume", 0) or 0)
    return f"{ticker} is trading at {price:.2f} EGP today with volume around {volume:,} shares."


def _format_quick_summary_reply(decision_result: Dict[str, Any], ticker: str) -> str:
    result = decision_result.get("result", decision_result)
    translator = result.get("decision_translator", {})
    stock_analysis = str(result.get("stock_analysis", "")).strip()
    simple_reason = str(translator.get("simple_reason", "")).strip()
    recommendation = str(translator.get("buy_or_not", "HOLD")).strip()
    lines = [
        f"{ticker} quick summary:",
        stock_analysis[:180] if stock_analysis else f"Recommendation: {recommendation}",
        f"Signal: {recommendation}",
    ]
    if simple_reason:
        lines.append(simple_reason[:180])
    return "\n".join(lines[:4])


def _format_news_only_reply(articles: list, ticker: str) -> str:
    if not articles:
        return f"No recent news items were found for {ticker}."

    bullets = []
    for item in articles[:5]:
        headline = item.get("headline") or item.get("title") or item.get("short_summary") or "News item"
        bullets.append(f"- {headline}")
    return f"Recent news for {ticker}:\n" + "\n".join(bullets)


def _format_comparison_reply(first_result: Dict[str, Any], second_result: Dict[str, Any], first_ticker: str, second_ticker: str) -> str:
    first_company = (first_result.get("payload") or {}).get("companies", [{}])[0]
    second_company = (second_result.get("payload") or {}).get("companies", [{}])[0]
    first_price = (first_company.get("price") or {}).get("current_EGP")
    second_price = (second_company.get("price") or {}).get("current_EGP")
    first_signal = first_company.get("signal", "N/A")
    second_signal = second_company.get("signal", "N/A")
    return (
        f"{first_ticker} vs {second_ticker}\n"
        f"- {first_ticker}: price={first_price}, signal={first_signal}\n"
        f"- {second_ticker}: price={second_price}, signal={second_signal}"
    )


def _format_full_analysis_reply(ticker: str, decision_result: Dict[str, Any], chat_history: Optional[list] = None) -> str:
    result = decision_result.get("result", {})
    translator = result.get("decision_translator", {})

    def clean_llm_text(text):
        if not text:
            return ""
        text = re.sub(r'`json.*?`', '', text, flags=re.DOTALL)
        text = re.sub(r'{.*?"stock_analysis".*?}', '', text, flags=re.DOTALL)
        tags_to_remove = ["### stock_analysis", "### advanced_explanation", "### اربط الأخبار بالواقع"]
        for tag in tags_to_remove:
            text = text.replace(tag, "")
        return text.strip()

    analysis = clean_llm_text(result.get("stock_analysis", ""))
    advanced = clean_llm_text(result.get("advanced_explanation", ""))
    scenarios = result.get("scenario_analysis", [])
    recommendations = translator.get("clear_recommendations", [])
    warning = clean_llm_text(result.get("risk_warning", ""))

    detailed_content = (
        f"📊 **التحليل الفني والأساسي:**\n{analysis}\n\n"
    )

    if advanced and advanced[:100] != analysis[:100]:
        detailed_content += f"💡 **رؤية الخبراء والمستويات الرقمية:**\n{advanced}\n\n"

    if scenarios and isinstance(scenarios, list):
        detailed_content += "🎯 **السيناريوهات المتوقعة:**\n"
        for s in scenarios:
            if isinstance(s, dict):
                detailed_content += (
                    f"- {s.get('scenario', '')}: **{s.get('action', '')}** ({s.get('reason', '')})\n"
                )

    if recommendations and isinstance(recommendations, list):
        detailed_content += "\n✅ **توصيات إضافية:**\n"
        filtered_recs = [r for r in recommendations if "detailed analysis" not in r.lower()]
        if filtered_recs:
            detailed_content += "\n".join([f"- {r}" for r in filtered_recs])

    if warning:
        detailed_content += f"\n\n⚠️ **تحذير المخاطر:**\n{warning}"

    is_first_interaction = not chat_history or len(chat_history) < 2

    if is_first_interaction:
        identity_intro = (
            "أنا المستشار المالي الذكي للبورصة المصرية، مشروع تخرج تم تطويره "
            "بواسطة طلاب كلية الذكاء الاصطناعي.\n\n"
        )
        return f"{identity_intro}🔍 **نتائج تحليل سهم {ticker}:**\n\n{detailed_content}"

    return f"🔍 **نتائج تحليل سهم {ticker}:**\n\n{detailed_content}"


def _companies_for_prompt() -> list:
    return [
        {"name_ar": value[0], "ticker": value[1]}
        for value in COMPANIES.values()
    ]


def _fallback_match_ticker(user_message: str) -> Optional[str]:
    text = user_message.upper()
    for _, (name_ar, ticker) in COMPANIES.items():
        if ticker in text or name_ar in user_message:
            return ticker
    return None


def _is_vague_message(user_message: str) -> bool:
    """
    Detects vague messages that don't specify any company/ticker.
    Examples: 'حلل', 'analyze', 'give me analysis', 'عايز توصية'
    Returns True if the message is too vague to infer a specific ticker.
    """
    msg = user_message.lower().strip()
    words = msg.split()
    
    # Single-word financial verbs without context
    vague_single_words = [
        "حلل", "تحليل", "analyze", "analysis", "analyse",
        "توصية", "توصيات", "recommend", "recommendation",
        "تقرير", "report", "أنصح", "نصيحة", "advice",
        "سهم", "stock", "أسهم", "stocks",
        "بورصة", "market", "سوق",
        "اشتري", "أشتري", "buy",
        "بيع", "sell",
        "أداء", "performance"
    ]
    
    # If it's a single vague word → definitely vague
    if len(words) == 1 and msg in vague_single_words:
        return True
    
    # If message is very short (<=2 words) and contains only vague terms
    if len(words) <= 2:
        has_specific = any(
            word in msg for word in ["comi", "fwry", "etel", "hrho", "swdy", "oras", "abuk", "tmgh"]
        )
        if not has_specific:
            return True
    
    return False


def infer_ticker_from_message(user_message: str) -> Dict[str, Any]:
    """
    Infer ticker from user message with confidence validation.
    FIXED: Rejects low-confidence inferences and vague messages.
    """
    # Step 1: Check if message is too vague
    if _is_vague_message(user_message):
        raise RuntimeError(
            "لم تحدد شركة معينة للتحليل. "
            "يرجى ذكر اسم الشركة أو رمز التداول (مثل: COMI، فوري، البنك التجاري الدولي)."
        )
    
    # Step 2: Try fallback matching (exact ticker or company name in message)
    fallback = _fallback_match_ticker(user_message)
    if fallback:
        return {
            "ticker": fallback,
            "reason": "fallback_match",
            "confidence": 0.85
        }
    
    # Step 3: If no Groq key, can't do LLM inference
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is required to infer ticker from free text. "
            "Please mention a specific company name or ticker symbol."
        )
    
    # Step 4: LLM inference with strict confidence threshold
    companies = _companies_for_prompt()
    system_prompt = (
        "You are a ticker inference assistant for the Egyptian Stock Exchange (EGX). "
        "Your job is to map user requests to the correct ticker symbol. "
        "Rules:\n"
        "1. ONLY pick from the provided company list\n"
        "2. If the user mentions a specific company name or ticker, return it with high confidence\n"
        "3. If the message is vague or doesn't clearly mention a company, set confidence LOW (<0.5)\n"
        "4. NEVER guess randomly - if uncertain, set confidence below 0.5\n"
        "Return strict JSON only."
    )
    user_prompt = (
        "Analyze the user message and determine which EGX ticker they are asking about.\n"
        "If the message does NOT clearly mention a specific company or ticker, "
        "set confidence below 0.5 and explain why.\n\n"
        f"Company list: {json.dumps(companies, ensure_ascii=False)}\n"
        f"User message: {user_message}\n\n"
        'Output schema: {"ticker": "COMI", "confidence": 0.0-1.0, "reason": "..."}'
    )

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=60,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        raise RuntimeError(
            "Ticker inference model returned invalid JSON. "
            "Please mention a specific company name or ticker symbol."
        )

    ticker = str(parsed.get("ticker", "")).upper().strip()
    confidence = float(parsed.get("confidence", 0))
    reason = str(parsed.get("reason", "model_inference"))
    
    # Step 5: Validate confidence threshold
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        raise RuntimeError(
            f"لم أتمكن من تحديد الشركة المقصودة بدقة كافية (confidence: {confidence:.2f}). "
            f"السبب: {reason}. "
            "يرجى ذكر اسم الشركة أو رمز التداول بوضوح (مثل: COMI، فوري، البنك التجاري الدولي)."
        )
    
    # Step 6: Validate ticker exists in our database
    valid_tickers = {value[1] for value in COMPANIES.values()}
    if ticker not in valid_tickers:
        raise RuntimeError(
            f"الرمز '{ticker}' غير موجود في قائمة الشركات المدعومة. "
            "يرجى التأكد من صحة الاسم أو الرمز."
        )

    return {
        "ticker": ticker,
        "confidence": confidence,
        "reason": reason,
    }


def _company_name_from_ticker(ticker: str) -> str:
    for _, (name_ar, symbol) in COMPANIES.items():
        if symbol == ticker.upper():
            return name_ar
    raise RuntimeError(f"Unknown ticker: {ticker}")


def _is_general_chat(user_message: str) -> bool:
    """
    Advanced detection: distinguishes general chat from stock analysis requests.
    Returns True if it's a greeting, help request, or general inquiry.
    Returns False only if the user explicitly asks for stock analysis or mentions a ticker.
    """
    msg = user_message.lower().strip()
    
    # 1. Explicit ticker mention (4-letter code in caps) = NOT general chat
    if re.search(r'\b[A-Z]{4}\b', user_message):
        return False
    
    # 2. Explicit financial action verbs = NOT general chat
    financial_actions = [
        "حلل", "سهم", "stock", "analyze", "بورصة", "أشتري", "اشتري", 
        "بيع", "سعر", "قيمة", "أداء", "توقعات", "اتجاه", "شراء", "بيع",
        "predict", "forecast", "technical", "fundamental", "buy", "sell"
    ]
    if any(word in msg for word in financial_actions):
        return False
    
    # 3. GENERAL/HELP KEYWORDS - these ARE general chat
    help_keywords = [
        "مين", "اسمك", "أهلا", "اهلا", "صباح", "مساء", "ازيك", "أزيك",
        "hello", "hi", "who are you", "بتعمل ايه", "بتعمل إيه", "وظيفتك",
        "كيف", "كيف يمكن", "كيف تستطيع", "كيفك", "مساعدة", "تساعد",
        "how can you help", "what can you do", "what do you do", "capabilities",
        "خدمات", "خدمة", "تقدم", "تقديم", "يمكنك", "يمكنك أن", "يمكنك ما",
        "ماذا", "ما الذي", "شنو", "شنو اللي", "ليش", "ليه"
    ]
    
    # If message contains help/greeting keywords, it IS general chat
    if any(word in msg for word in help_keywords):
        return True
    
    # 4. Short messages without specific tickers = general chat
    if len(msg.split()) <= 2:
        return True
    
    # 5. Default: if no explicit financial action, treat as general
    return True


def run_general_chat(user_message: str, chat_history: Optional[list] = None) -> str:
    system_prompt = (
        "You are the 'EGX Smart Financial Advisor', a cutting-edge graduation project "
        "developed by 4th-year students at the Faculty of Artificial Intelligence. "
        "- Your goal is to be a friendly, professional financial assistant. "
        "- If greeted, reply warmly in the same language. "
        "- If asked what you do, explain that you analyze EGX stocks using AI, news scraping (Mubasher), "
        "and financial data (yfinance). "
        "- Always encourage the user to provide a stock ticker (e.g., FWRY, COMI) to start the deep analysis."
    )

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history[-3:])  # إضافة آخر 3 رسائل للسياق
    messages.append({"role": "user", "content": user_message})

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": GROQ_MODEL, "messages": messages, "temperature": 0.7},
        timeout=90,
    )
    response.raise_for_status()  # Raise HTTPError if status code is not 2xx
    return response.json()["choices"][0]["message"]["content"]


def run_chat_pipeline(
    user_message: str,
    risk_answers: Optional[Dict[str, str]] = None,
    user_risk_profile: Optional[str] = None,
    max_news: int = 20,
    chat_history: Optional[list] = None,
    last_analysis_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    query_type = classify_query_type(user_message)
    if _is_general_chat(user_message):
        general_reply = run_general_chat(user_message, chat_history)
        return {
            "ticker": "GENERAL",
            "query_type": "GENERAL_CHAT",
            "chat_reply": general_reply,
            "final_result": {"status": "chat_only"},
            # بنبعت دول فاضيين عشان الـ Frontend ميعرضش "جاري تشغيل الـ Pipeline"
            "part1_news_output": None,
            "part2_financial_output": None,
            "metadata": last_analysis_context,
        }
    
    # FIXED: Try to infer ticker, but handle vague messages gracefully
    try:
        inferred = infer_ticker_from_message(user_message)
    except RuntimeError as exc:
        # If inference failed due to vague message, return helpful error
        return {
            "ticker": "UNKNOWN",
            "query_type": "ERROR",
            "chat_reply": str(exc),
            "final_result": {"status": "error", "message": str(exc)},
            "part1_news_output": None,
            "part2_financial_output": None
        }
    
    ticker = inferred["ticker"]
    company_name = _company_name_from_ticker(ticker)

    if query_type == "SIMPLE_FACT":
        fact_reply = _format_simple_fact_reply(ticker, user_message)
        return {
            "ticker_inference": inferred,
            "ticker": ticker,
            "query_type": query_type,
            "chat_reply": fact_reply,
            "final_result": {"status": "simple_fact", "query_type": query_type},
            "part1_news_output": None,
            "part2_financial_output": None,
            "metadata": {"ticker": ticker, "type": "simple_fact"},
        }

    if query_type == "NEWS_ONLY":
        raw_articles = scrape_news(ticker, company_name, max_news=max_news)
        articles = validate_news_articles(raw_articles)
        if not articles:
            raise RuntimeError(f"No news scraped for ticker {ticker}")
        analyzed_news = analyze_news_batch(articles)
        news_path = save_results(analyzed_news, ticker)
        return {
            "ticker_inference": inferred,
            "ticker": ticker,
            "query_type": query_type,
            "chat_reply": _format_news_only_reply(analyzed_news, ticker),
            "final_result": {"status": "news_only", "query_type": query_type},
            "part1_news_output": news_path,
            "part2_financial_output": None,
            "metadata": {"ticker": ticker, "part1_news_output": news_path},
        }

    if query_type == "QUICK_SUMMARY":
        try:
            part2 = generate_part2_financial_json(
                ticker=ticker,
                user_risk_profile=user_risk_profile or "moderate",
                drawdown_tolerance=(risk_answers or {}).get("max_drawdown_tolerance"),
            )
        except Exception:
            part2 = _load_latest_part2_snapshot(ticker)
            if not part2:
                raise
        return {
            "ticker_inference": inferred,
            "ticker": ticker,
            "query_type": query_type,
            "chat_reply": _format_quick_summary_reply(part2, ticker),
            "final_result": {"status": "quick_summary", "query_type": query_type, "result": part2.get("payload")},
            "part1_news_output": None,
            "part2_financial_output": part2.get("output_file"),
            "metadata": part2.get("payload") or {"ticker": ticker},
        }

    if query_type == "COMPARISON":
        tickers = _extract_tickers_from_message(user_message)
        if len(tickers) < 2:
            tickers = [ticker, ticker]
        first_ticker, second_ticker = tickers[:2]
        first_result = generate_part2_financial_json(
            ticker=first_ticker,
            user_risk_profile=user_risk_profile or "moderate",
            drawdown_tolerance=(risk_answers or {}).get("max_drawdown_tolerance"),
        )
        second_result = generate_part2_financial_json(
            ticker=second_ticker,
            user_risk_profile=user_risk_profile or "moderate",
            drawdown_tolerance=(risk_answers or {}).get("max_drawdown_tolerance"),
        )
        return {
            "ticker_inference": inferred,
            "ticker": ticker,
            "query_type": query_type,
            "chat_reply": _format_comparison_reply(first_result, second_result, first_ticker, second_ticker),
            "final_result": {"status": "comparison", "query_type": query_type},
            "part1_news_output": None,
            "part2_financial_output": [first_result.get("output_file"), second_result.get("output_file")],
            "metadata": {
                "tickers": [first_ticker, second_ticker],
                "files": [first_result.get("output_file"), second_result.get("output_file")]
            },
        }

    raw_articles = scrape_news(ticker, company_name, max_news=max_news)
    articles = validate_news_articles(raw_articles)
    if not articles:
        raise RuntimeError(f"No news scraped for ticker {ticker}")

    analyzed_news = analyze_news_batch(articles)
    news_path = save_results(analyzed_news, ticker)

    try:
        part2 = generate_part2_financial_json(
            ticker=ticker,
            user_risk_profile=user_risk_profile or "moderate",
            drawdown_tolerance=(risk_answers or {}).get("max_drawdown_tolerance"),
        )
    except Exception:
        part2 = _load_latest_part2_snapshot(ticker)
        if not part2:
            raise
    financial_path = part2["output_file"]

    final_decision = generate_final_decision(
        ticker=ticker,
        news_json_path=news_path,
        financial_json_path=financial_path,
        user_risk_profile=user_risk_profile,
        risk_answers=risk_answers,
        query_type=query_type,
        query_text=user_message,
    )
    chat_reply = _format_full_analysis_reply(ticker, final_decision, chat_history)

    decision_result = final_decision.get("result", {})
    if isinstance(decision_result, dict):
        investor_profile_block = ((final_decision.get("prompt_debug") or {}).get("investor_profile") or {}).get("block")
        if not investor_profile_block:
            financial_company = (part2.get("payload") or {}).get("companies", [{}])[0]
            investor_profile_block = _build_investor_profile_context(
                user_risk_profile or "moderate",
                risk_answers,
                financial_company,
            ).get("block")
        if investor_profile_block:
            decision_result.setdefault("investor_profile_block", investor_profile_block)

    return {
        "ticker_inference": inferred,
        "ticker": ticker,
        "query_type": query_type,
        "chat_reply": chat_reply,  # الرد المفلتر والمنظم
        "final_result": decision_result,
        "part1_news_output": news_path,
        "part2_financial_output": financial_path,
        "metadata": final_decision.get("metadata") or {"ticker": ticker, "news": news_path, "financial": financial_path},
    }