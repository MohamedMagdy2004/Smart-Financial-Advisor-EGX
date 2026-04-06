"""
Chat orchestration for end-to-end pipeline:
user message -> infer ticker -> part1 -> part2 -> final decision.
Updated to prioritize professional persona in chat reply.
"""
import json
import logging
from typing import Any, Dict, Optional

import requests

from analyzer import analyze_news_batch, save_results
from config import COMPANIES, GROQ_API_KEY, GROQ_MODEL
from decision_engine import generate_final_decision
from part2_generator import generate_part2_financial_json
from scraper import scrape_news, validate_news_articles


logger = logging.getLogger(__name__)


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
    """نسخة مطورة لتمييز الدردشة العامة عن طلبات البورصة"""
    msg = user_message.lower().strip()

    # 1. لو الرسالة فيها كود سهم صريح (4 حروف كابيتال)، دي مش دردشة عامة
    import re
    if re.search(r'\b[A-Z]{4}\b', user_message.upper()):
        return False

    # 2. لو فيها كلمات "أكشن" مالية، نعتبرها طلب تحليل فوراً
    financial_actions = ["حلل", "سهم", "stock", "analyze", "بورصة", "أشتري", "اشتري", "بيع", "سعر"]
    if any(word in msg for word in financial_actions):
        return False

    # 3. كلمات الترحيب والاستفسار عن الهوية
    general_keywords = [
        "مين", "اسمك", "أهلا", "اهلا", "صباح", "مساء", "ازيك", "أزيك",
        "hello", "hi", "who are you", "بتعمل ايه", "بتعمل إيه", "وظيفتك"
    ]

    # لو الرسالة قصيرة جداً أو فيها كلمة ترحيب، نعتبرها دردشة عامة
    return len(msg.split()) < 3 or any(word in msg for word in general_keywords)


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
    return response.json()["choices"][0]["message"]["content"]


def run_chat_pipeline(
    user_message: str,
    risk_answers: Optional[Dict[str, str]] = None,
    user_risk_profile: Optional[str] = None,
    max_news: int = 20,
    chat_history: Optional[list] = None,
) -> Dict[str, Any]:
    if _is_general_chat(user_message):
        general_reply = run_general_chat(user_message, chat_history)
        return {
            "ticker": "GENERAL",
            "chat_reply": general_reply,
            "final_result": {"status": "chat_only"},
            # بنبعت دول فاضيين عشان الـ Frontend ميعرضش "جاري تشغيل الـ Pipeline"
            "part1_news_output": None,
            "part2_financial_output": None
        }
    inferred = infer_ticker_from_message(user_message)
    ticker = inferred["ticker"]
    company_name = _company_name_from_ticker(ticker)

    raw_articles = scrape_news(ticker, company_name, max_news=max_news)
    articles = validate_news_articles(raw_articles)
    if not articles:
        raise RuntimeError(f"No news scraped for ticker {ticker}")

    analyzed_news = analyze_news_batch(articles)
    news_path = save_results(analyzed_news, ticker)

    part2 = generate_part2_financial_json(
        ticker=ticker,
        user_risk_profile=user_risk_profile or "moderate",
    )
    financial_path = part2["output_file"]

    final_decision = generate_final_decision(
        ticker=ticker,
        news_json_path=news_path,
        financial_json_path=financial_path,
        user_risk_profile=user_risk_profile,
        risk_answers=risk_answers,
    )

    decision_result = final_decision.get("result", {})
    translator = decision_result.get("decision_translator", {})



    # 1. استخراج كل الحقول من نتيجة الـ AI
    # --- بداية التعديل المطور للتنظيف ومنع التكرار ---

    # دالة داخلية لتنظيف النص من أي مخلفات تقنية أو JSON بيغلط فيها الـ AI
    def clean_llm_text(text):
        if not text: return ""
        # حذف أي بلوكات كود JSON لو الموديل كتبها غلط جوه النص
        text = re.sub(r'```json.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'\{.*?"stock_analysis".*?\}', '', text, flags=re.DOTALL)
        # حذف العناوين الجانبية اللي الموديل بيكررها
        tags_to_remove = ["### stock_analysis", "### advanced_explanation", "### اربط الأخبار بالواقع"]
        for tag in tags_to_remove:
            text = text.replace(tag, "")
        return text.strip()

    import re
    analysis = clean_llm_text(decision_result.get("stock_analysis", ""))
    advanced = clean_llm_text(decision_result.get("advanced_explanation", ""))
    scenarios = decision_result.get("scenario_analysis", [])
    recommendations = translator.get("clear_recommendations", [])
    warning = clean_llm_text(decision_result.get("risk_warning", ""))

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



    is_first_interaction = not chat_history or len(chat_history) < 2

    if is_first_interaction:
        identity_intro = (
            "أنا المستشار المالي الذكي للبورصة المصرية، مشروع تخرج تم تطويره "
            "بواسطة طلاب كلية الذكاء الاصطناعي.\n\n"
        )
        chat_reply = f"{identity_intro}🔍 **نتائج تحليل سهم {ticker}:**\n\n{detailed_content}"
    else:
        chat_reply = f"🔍 **نتائج تحليل سهم {ticker}:**\n\n{detailed_content}"


    return {
        "ticker_inference": inferred,
        "ticker": ticker,
        "chat_reply": chat_reply,  # الرد المفلتر والمنظم
        "final_result": decision_result,
        "part1_news_output": news_path,
        "part2_financial_output": financial_path,
    }