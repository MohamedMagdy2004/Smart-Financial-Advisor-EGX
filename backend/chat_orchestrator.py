"""
Chat orchestration for end-to-end pipeline:
user message -> infer ticker -> part1 -> part2 -> final decision.
Updated to prioritize professional persona in chat reply.
"""
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

from analyzer import analyze_news_batch, save_results
from config import COMPANIES, GROQ_API_KEY, GROQ_MODEL
from decision_engine import QUERY_RESPONSE_RULES, _build_investor_profile_context, generate_final_decision
from part2_generator import generate_part2_financial_json, _fetch_from_yfinance
from scraper import scrape_news, validate_news_articles



logger = logging.getLogger(__name__)




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

    # "What's the latest on X today?" should be QUICK_SUMMARY, not NEWS_ONLY.
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


def is_follow_up_question(user_message: str) -> bool:
    """
    Detect if message is a follow-up question (not a new analysis request).
    
    Follow-ups: "أشتريه؟", "ليه؟", "should I buy?", etc.
    New requests: "حلل سهم", "تحليل جديد", "new analysis", etc.
    
    Returns:
        True if message looks like follow-up, False otherwise
    """
    text = user_message.lower().strip()
    
    # New analysis request markers - return False
    new_analysis_markers = [
        "حلل", "تحليل", "analyze", "analyse", "new analysis",
        "تحليل جديد", "جديد", "حدث", "refresh", "reanalyze",
        "آخر أخبار", "latest news", "update", "تحديث"
    ]
    if any(marker in text for marker in new_analysis_markers):
        return False
    
    # Follow-up question markers - return True
    followup_markers = [
        # Arabic follow-ups
        "أشتري", "اشتري", "أبيع", "ابيع", "أدخل", "ادخل",
        "أستنى", "استنى", "انتظر", "ليه", "ليش", "لما",
        "مناسب", "المخاطرة", "الدعم", "المقاومة", "الأخبار",
        "يعني", "طب", "إيه", "ايه", "كام", "كام؟",
        "عامل", "ما الوضع", "اعمل ايه", "أستنى ولا",
        # English follow-ups
        "should i buy", "should i sell", "why", "is it", "what is",
        "support and resistance", "risk", "entry", "exit", "what should i do",
        "enter now", "hold", "sell now", "buy now"
    ]
    
    if any(marker in text for marker in followup_markers):
        return True
    
    # Short vague questions with no ticker = likely follow-up
    if len(text) < 20 and not _extract_tickers_from_message(user_message):
        # Single word or short phrase without ticker = follow-up
        if len(text.split()) <= 3:
            return True
    
    return False


def is_refresh_request(user_message: str) -> bool:
    """
    Detect if user explicitly requested a fresh analysis (not using old cache).
    
    Returns:
        True if user said "refresh", "new analysis", "update", etc.
    """
    text = user_message.lower()
    refresh_markers = [
        "refresh", "تحديث", "حدث", "جديد", "جديدة",
        "new analysis", "تحليل جديد", "reanalyze", "اعادة",
        "تحليل من جديد", "من الأول"
    ]
    return any(marker in text for marker in refresh_markers)


def get_context_age_minutes(analysis_time_iso: str) -> float:
    """
    Calculate how many minutes have passed since analysis_time.
    
    Args:
        analysis_time_iso: ISO format timestamp string
        
    Returns:
        Age in minutes (float), or None if parsing fails
    """
    try:
        analysis_dt = datetime.fromisoformat(analysis_time_iso)
        current_dt = datetime.now(timezone.utc)
        # Handle naive datetime from fromisoformat
        if analysis_dt.tzinfo is None:
            analysis_dt = analysis_dt.replace(tzinfo=timezone.utc)
        age_seconds = (current_dt - analysis_dt).total_seconds()
        return age_seconds / 60.0
    except Exception:
        return None


def answer_follow_up_from_analysis_files(
    user_message: str,
    analysis_context: dict,
    user_risk_profile: Optional[str] = None
) -> Dict[str, Any]:
    """
    Answer a follow-up question using previously generated analysis files.
    
    Does NOT:
    - Scrape news
    - Call Modal
    - Fetch new financial data
    - Generate new files
    
    Only reads and interprets existing analysis files through Groq.
    
    Args:
        user_message: User's follow-up question
        analysis_context: Dict with ticker, company_name, analysis_time, files
        user_risk_profile: Optional risk profile for context
        
    Returns:
        Response dict with chat_reply and metadata
    """
    ticker = analysis_context.get("ticker")
    company_name = analysis_context.get("company_name")
    files = analysis_context.get("files", {})
    analysis_time = analysis_context.get("analysis_time")
    
    logger.info(f"Answering follow-up from files for ticker={ticker}")
    
    # Read the most important file: final decision
    final_decision_file = files.get("final_decision")
    if not final_decision_file or not os.path.exists(final_decision_file):
        logger.warning(f"Final decision file not found: {final_decision_file}")
        return {
            "ticker": ticker,
            "query_type": "FOLLOW_UP_FROM_FILES",
            "chat_reply": "لا أملك معلومة كافية من آخر تحليل للإجابة بدقة. هل تريد أن أجري تحليلًا جديدًا؟",
            "final_result": {"status": "context_incomplete"},
            "part1_news_output": None,
            "part2_financial_output": None,
        }
    
    # Read final decision JSON
    try:
        with open(final_decision_file, "r", encoding="utf-8") as f:
            final_decision_data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read final decision file: {e}")
        return {
            "ticker": ticker,
            "query_type": "FOLLOW_UP_FROM_FILES",
            "chat_reply": "لا أملك معلومة كافية من آخر تحليل للإجابة بدقة. هل تريد أن أجري تحليلًا جديدًا؟",
            "final_result": {"status": "context_read_error"},
            "part1_news_output": None,
            "part2_financial_output": None,
        }
    
    # Financial data is now consolidated: read it directly from financial file
    # (it includes support/resistance levels merged from Mubasher)
    financial_data = None
    financial_file = files.get("financial")
    if financial_file and os.path.exists(financial_file):
        try:
            with open(financial_file, "r", encoding="utf-8") as f:
                financial_data = json.load(f)
                logger.info(f"Loaded consolidated financial_analysis data: {financial_file}")
        except Exception as e:
            logger.warning(f"Could not read consolidated financial file: {e}")
    
    # Build Groq prompt
    system_prompt = (
        "You are a financial analyst answering follow-up questions about a previous stock analysis. "
        "Answer ONLY based on the provided analysis files. Do NOT invent new market data or claim real-time updates. "
        "If the files don't contain enough information to answer, say so clearly. "
        "Keep answers concise and practical. Respond in the same language as the question (Arabic preferred if possible)."
    )
    
    # Prepare context for LLM
    final_decision_result = final_decision_data.get("result", {})
    analysis_summary = {
        "ticker": ticker,
        "company_name": company_name,
        "analysis_time": analysis_time,
        "recommendation": final_decision_result.get("decision_translator", {}).get("buy_or_not", "N/A"),
        "reasoning": final_decision_result.get("stock_analysis", "")[:500],
        "risk_warnings": final_decision_result.get("risk_warning", "")[:300],
        "technical_summary": final_decision_result.get("technical_summary", "")[:300],
    }
    
    user_prompt = (
        f"Stock: {ticker} ({company_name})\n"
        f"Analysis done: {analysis_time}\n"
        f"Risk profile: {user_risk_profile or 'moderate'}\n\n"
        f"Previous Analysis Summary:\n"
        f"{json.dumps(analysis_summary, ensure_ascii=False, indent=2)}\n\n"
        f"Full final decision data:\n"
        f"{json.dumps(final_decision_result, ensure_ascii=False, indent=2)}\n\n"
        f"User follow-up question:\n"
        f"{user_message}"
    )
    
    # Call Groq with file context
    try:
        groq_response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "temperature": 0.3,  # Lower for consistency with past analysis
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=30,
        )
        groq_response.raise_for_status()
        answer = groq_response.json()["choices"][0]["message"]["content"]
        logger.info(f"Follow-up answer generated from files")
    except Exception as e:
        logger.error(f"Groq follow-up failed: {e}")
        answer = f"لا أملك معلومة كافية من آخر تحليل للإجابة بدقة. هل تريد أن أجري تحليلًا جديدًا؟ (خطأ: {str(e)[:50]})"
    
    return {
        "ticker": ticker,
        "query_type": "FOLLOW_UP_FROM_FILES",
        "chat_reply": answer,
        "final_result": {
            "status": "follow_up_from_files",
            "based_on_files": files
        },
        "part1_news_output": None,  # No new files generated
        "part2_financial_output": None,  # No new files generated
    }


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
        text = re.sub(r'```json.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'\{.*?"stock_analysis".*?\}', '', text, flags=re.DOTALL)
        tags_to_remove = ["### stock_analysis", "### advanced_explanation", "### اربط الأخبار بالواقع"]
        for tag in tags_to_remove:
            text = text.replace(tag, "")
        return text.strip()

    analysis = clean_llm_text(result.get("stock_analysis", ""))
    advanced = clean_llm_text(result.get("advanced_explanation", ""))
    scenarios = result.get("scenario_analysis", [])
    recommendations = translator.get("clear_recommendations", [])
    warning = clean_llm_text(result.get("risk_warning", ""))

    detailed_content = f"📊 **التحليل الفني والأساسي:**\n{analysis}\n\n"
    if advanced and advanced[:100] != analysis[:100]:
        detailed_content += f"💡 **رؤية الخبراء والمستويات الرقمية:**\n{advanced}\n\n"

    if scenarios and isinstance(scenarios, list):
        detailed_content += "🎯 **السيناريوهات المتوقعة:**\n"
        for s in scenarios:
            if isinstance(s, dict):
                detailed_content += f"- {s.get('scenario', '')}: **{s.get('action', '')}** ({s.get('reason', '')})\n"

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


def infer_ticker_from_message(user_message: str) -> Dict[str, Any]:
    fallback = _fallback_match_ticker(user_message)
    if not GROQ_API_KEY:
        if fallback:
            return {"ticker": fallback, "reason": "fallback_without_groq", "confidence": 0.6}
        raise RuntimeError("GROQ_API_KEY is required to infer ticker from free text.")

    companies = _companies_for_prompt()
    system_prompt = (
        "You map Arabic/English user stock requests to EGX ticker symbols. "
        "Return strict JSON only."
    )
    user_prompt = (
        "Pick exactly one ticker from the provided company list.\n"
        "If uncertain, return the closest valid ticker with lower confidence.\n"
        "Output schema: {\"ticker\": \"COMI\", \"confidence\": 0.0-1.0, \"reason\": \"...\"}\n\n"
        f"Company list: {json.dumps(companies, ensure_ascii=False)}\n"
        f"User message: {user_message}"
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
        if fallback:
            return {"ticker": fallback, "reason": "fallback_after_bad_json", "confidence": 0.55}


        raise RuntimeError("Ticker inference model output was not valid JSON")

    ticker = str(parsed.get("ticker", "")).upper().strip()
    valid_tickers = {value[1] for value in COMPANIES.values()}
    if ticker not in valid_tickers:
        if fallback:
            return {"ticker": fallback, "reason": "fallback_after_invalid_ticker", "confidence": 0.55}
        raise RuntimeError(f"Inferred invalid ticker: {ticker}")

    return {
        "ticker": ticker,
        "confidence": float(parsed.get("confidence", 0.7)),
        "reason": parsed.get("reason", "model_inference"),
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
    """
    Run the full chat pipeline with follow-up detection and file-based caching.
    
    Args:
        user_message: User input
        risk_answers: Risk profile answers
        user_risk_profile: User's risk profile
        max_news: Max news articles to scrape
        chat_history: Previous conversation history
        last_analysis_context: Cached analysis context for follow-ups (from get_last_file_based_analysis_context)
    """
    logger.info(f"Pipeline start: message='{user_message[:50]}...' followup_context={last_analysis_context is not None}")
    
    # ✅ FOLLOW-UP DETECTION BEFORE TICKER INFERENCE (CRITICAL)
    is_followup = is_follow_up_question(user_message)
    is_refresh = is_refresh_request(user_message)
    
    logger.info(f"Follow-up detected: {is_followup}, Refresh requested: {is_refresh}")
    
    if is_followup and not is_refresh:
        # User asked a follow-up question
        if last_analysis_context:
            # We have cached analysis
            age_minutes = last_analysis_context.get("age_minutes")
            logger.info(f"Context available: ticker={last_analysis_context.get('ticker')} age={age_minutes:.1f}m")
            
            if age_minutes is not None and age_minutes <= 5.0:
                # Context is fresh (within 5 minutes) - answer from files
                logger.info(f"Context fresh (<5m), answering from files")
                return answer_follow_up_from_analysis_files(
                    user_message,
                    last_analysis_context,
                    user_risk_profile=user_risk_profile
                )
            elif age_minutes is not None and age_minutes > 5.0:
                # Context is too old - ask user
                logger.info(f"Context stale (>{age_minutes:.0f}m), asking user")
                ticker = last_analysis_context.get("ticker")
                return {
                    "ticker": ticker,
                    "query_type": "CONTEXT_FRESHNESS_CHECK",
                    "chat_reply": (
                        f"آخر تحليل لدي لسهم {ticker} كان منذ {int(age_minutes)} دقيقة.\n"
                        f"هل تريد أن أجيب بناءً على هذا التحليل، أم أجري تحليلًا جديدًا؟"
                    ),
                    "final_result": {"status": "context_stale", "age_minutes": age_minutes},
                    "part1_news_output": None,
                    "part2_financial_output": None,
                }
        else:
            # Follow-up but no context - ask for clarification
            logger.info(f"Follow-up detected but no context, asking clarification")
            return {
                "ticker": None,
                "query_type": "CLARIFICATION_NEEDED",
                "chat_reply": "تقصد أي سهم؟ اكتب اسم الشركة أو رمز السهم.",
                "final_result": {"status": "clarification_needed"},
                "part1_news_output": None,
                "part2_financial_output": None,
            }
    
    # NOT a follow-up (or explicit refresh) - run full pipeline
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
            "part2_financial_output": None
        }
    inferred = infer_ticker_from_message(user_message)
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

    # Extract file paths for follow-up context
    # Note: financial_json_enriched no longer exists separately (consolidated into financial_json)
    final_decision_path = final_decision.get("output_file")
    
    # Build metadata for follow-up questions (5-minute cache)
    analysis_metadata = {
        "context_type": "file_based_stock_analysis",
        "ticker": ticker,
        "company_name": company_name,
        "analysis_time": datetime.now(timezone.utc).isoformat(),
        "files": {
            "news": news_path,
            "financial": financial_path,
            "final_decision": final_decision_path,
        }
    }

    return {
        "ticker_inference": inferred,
        "ticker": ticker,
        "query_type": query_type,
        "chat_reply": chat_reply,  # الرد المفلتر والمنظم
        "final_result": decision_result,
        "part1_news_output": news_path,
        "part2_financial_output": financial_path,
        "metadata": analysis_metadata,  # For follow-up context
    }