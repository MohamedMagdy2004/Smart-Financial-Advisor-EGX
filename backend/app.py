"""
FastAPI Backend for Egyptian Stock Exchange News Pipeline
Provides REST API for scraping, analyzing, and retrieving financial news
"""
import logging
import json
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import os
from datetime import datetime
from pathlib import Path

from config import (
    API_HOST, API_PORT, CORS_ORIGINS, COMPANIES, OUTPUT_SCHEME, DEBUG
)
from scraper import scrape_news, validate_news_articles
from analyzer import analyze_news_batch, save_results
from decision_engine import generate_final_decision
from chat_orchestrator import run_chat_pipeline

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Silence noisy third-party debug streams (can include transport internals)
logging.getLogger("hpack").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP SETUP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Egyptian Stock Exchange News Pipeline API",
    description="Scrape and analyze financial news with AI",
    version="1.0.0",
    redirect_slashes=False ######
)
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]#########
if isinstance(CORS_ORIGINS, list):
    origins.extend([o for o in CORS_ORIGINS if o not in origins])
    ########
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] ,#######
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

class CompanyListResponse:
    """Response model for available companies"""
    pass


class NewsArticle:
    """News article data model"""
    pass


class AnalyzeRequest(BaseModel):
    ticker: str
    articles: List[dict]


class DecisionRequest(BaseModel):
    ticker: str
    news_json_path: str
    financial_json_path: str
    user_risk_profile: Optional[str] = None
    risk_answers: Optional[dict] = None


class ChatMessageRequest(BaseModel):
    message: str
    max_news: Optional[int] = 10
    user_risk_profile: Optional[str] = None
    risk_answers: Optional[dict] = None
    history: Optional[List[dict]] = []

def _resolve_json_path(path_like: str) -> str:
    candidate = Path(path_like)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    output_candidate = Path(__file__).resolve().parent / "output" / path_like
    if output_candidate.exists():
        return str(output_candidate)

    cwd_candidate = Path.cwd() / path_like
    if cwd_candidate.exists():
        return str(cwd_candidate)

    raise FileNotFoundError(f"JSON file not found: {path_like}")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    """API health check"""
    return {
        "status": "online",
        "service": "Egyptian Stock Exchange News Pipeline API",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "news-pipeline-api"
    }


@app.get("/companies", tags=["Companies"])
async def list_companies():
    """
    Get list of available Egyptian Stock Exchange companies
    
    Returns:
        List of companies with their symbols
    """
    return {
        "count": len(COMPANIES),
        "companies": [
            {
                "id": k,
                "name": v[0],
                "symbol": v[1]
            }
            for k, v in COMPANIES.items()
        ]
    }


@app.get("/companies/{ticker}", tags=["Companies"])
async def get_company(ticker: str):
    """
    Get company info by ticker symbol
    
    Args:
        ticker: Stock symbol (e.g., COMI)
    
    Returns:
        Company details
    """
    ticker = ticker.upper()
    for company_name, symbol in COMPANIES.values():
        if symbol == ticker:
            return {
                "symbol": ticker,
                "name": company_name
            }
    raise HTTPException(status_code=404, detail=f"Company {ticker} not found")


@app.get("/pipeline/scrape", tags=["Pipeline"])
async def start_scraping(
    ticker: str = Query(..., description="Stock symbol e.g., COMI"),
    max_news: int = Query(20, description="Maximum news articles to scrape")
):
    """
    Step 1: Scrape news articles
    
    Args:
        ticker: Stock symbol
        max_news: Max articles to fetch (default: 20)
    
    Returns:
        List of scraped news articles
    """
    ticker = ticker.upper()
    
    # Find company name
    company_name = None
    for _, (name, symbol) in COMPANIES.items():
        if symbol == ticker:
            company_name = name
            break
    
    if not company_name:
        raise HTTPException(status_code=400, detail=f"Company {ticker} not found")
    
    logger.info(f"Starting scrape for {ticker} ({company_name})")
    
    try:
        articles = scrape_news(ticker, company_name, max_news)
        validated = validate_news_articles(articles)
        
        return {
            "status": "success",
            "ticker": ticker,
            "company": company_name,
            "count": len(validated),
            "articles": validated
        }
    except Exception as e:
        logger.error(f"Scraping error: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {e}")


@app.post("/pipeline/analyze", tags=["Pipeline"])
async def analyze_articles(
    payload: AnalyzeRequest
):
    """
    Step 2: Analyze articles with AI
    
    Args:
        ticker: Stock symbol
        articles: List of article dicts from scrape endpoint
    
    Returns:
        Analyzed articles with extracted financial info
    """
    ticker = payload.ticker.upper()
    articles = payload.articles

    if not articles or len(articles) == 0:
        raise HTTPException(status_code=400, detail="No articles provided")
    
    logger.info(f"Analyzing {len(articles)} articles for {ticker}")
    
    try:
        results = analyze_news_batch(articles)
        
        # Save to JSON file
        filepath = save_results(results, ticker)
        
        return {
            "status": "success",
            "ticker": ticker,
            "count": len(results),
            "results": results,
            "output_file": filepath,
            "output_scheme": json.loads(OUTPUT_SCHEME)
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.post("/pipeline/full", tags=["Pipeline"])
async def full_pipeline(
    ticker: str = Query(..., description="Stock symbol"),
    max_news: int = Query(20, description="Max articles to scrape")
):
    """
    Complete pipeline: Scrape + Analyze in one call
    
    Args:
        ticker: Stock symbol
        max_news: Max articles to fetch
    
    Returns:
        Analyzed articles with extracted info
    """
    ticker = ticker.upper()
    
    # Step 1: Scrape
    scrape_response = await start_scraping(ticker, max_news)
    if scrape_response["count"] == 0:
        raise HTTPException(status_code=404, detail="No articles found")
    
    # Step 2: Analyze
    articles = scrape_response["articles"]
    analyze_response = await analyze_articles(
        AnalyzeRequest(ticker=ticker, articles=articles)
    )
    
    return {
        "status": "complete",
        "pipeline": "scrape → analyze",
        "ticker": ticker,
        "company": scrape_response["company"],
        "articles_scraped": scrape_response["count"],
        "articles_analyzed": analyze_response["count"],
        "results": analyze_response["results"],
        "output_file": analyze_response["output_file"],
        "output_scheme": analyze_response["output_scheme"]
    }


@app.post("/pipeline/decision", tags=["Pipeline"])
async def run_decision_engine(payload: DecisionRequest):
    """
    Part 3 endpoint:
    Merge Part 1 (news JSON) + Part 2 (financial JSON), enrich S/R,
    then ask Groq LLM for final recommendation output.
    """
    try:
        news_path = _resolve_json_path(payload.news_json_path)
        financial_path = _resolve_json_path(payload.financial_json_path)

        result = generate_final_decision(
            ticker=payload.ticker,
            news_json_path=news_path,
            financial_json_path=financial_path,
            user_risk_profile=payload.user_risk_profile,
            risk_answers=payload.risk_answers,
        )
        return {
            "status": "success",
            "ticker": payload.ticker.upper(),
            "user_risk_profile": result.get("user_risk_profile"),
            "output_file": result.get("output_file"),
            "result": result.get("result"),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(f"Decision engine failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Decision engine failed: {exc}")


@app.post("/chat/message", tags=["Chat"])
async def chat_message(payload: ChatMessageRequest):
    """
    End-to-end chat endpoint:
    1) infer ticker from user message (LLM)
    2) run Part 1 news scrape+analyze
    3) run Part 2 financial generation
    4) run Part 3 final decision
    """
    try:
        result = run_chat_pipeline(
            user_message=payload.message,
            risk_answers=payload.risk_answers,
            user_risk_profile=payload.user_risk_profile,
            max_news=int(payload.max_news or 20),
            chat_history=payload.history,
        )
        return {"status": "success", **result}
    except Exception as exc:
        logger.error(f"Chat pipeline failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Chat pipeline failed: {exc}")


@app.get("/output/{filename}", tags=["Output"])
async def download_output(filename: str):
    """
    Download output JSON file
    
    Args:
        filename: Filename to download (e.g., COMI_20260325_143022.json)
    
    Returns:
        JSON file
    """
    from config import OUTPUT_DIR
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(filepath, media_type="application/json")


@app.get("/output-schema", tags=["Documentation"])
async def get_output_schema():
    """
    Get the output JSON schema for analysis results
    
    Returns:
        Schema describing the analysis output format
    """
    return {
        "description": "Financial news analysis output schema",
        "schema": json.loads(OUTPUT_SCHEME),
        "example_result": {
            "company_name": "البنك التجاري الدولي",
            "ticker": "COMI",
            "news_date": "25/03/2026",
            "headline": "تحديث على أرباح البنك",
            "link": "https://www.mubasher.info/news/...",
            "event_type": "earnings",
            "sentiment": "positive",
            "impact_level": "high",
            "financials": {
                "net_profit_current": 1000000000,
                "net_profit_previous": 900000000,
                "percentage_change": 11.1
            },
            "currency": "EGP",
            "short_summary": "أعلنت الشركة عن أرباح قياسية..."
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status": "error"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Generic error handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status": "error"
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP/SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Run on app startup"""
    logger.info("=" * 70)
    logger.info("🚀 Egyptian Stock Exchange News Pipeline API Starting")
    logger.info(f"📍 Host: {API_HOST}:{API_PORT}")
    logger.info(f"🌍 CORS Origins: {CORS_ORIGINS}")
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on app shutdown"""
    logger.info("🛑 API shutting down")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG,
        log_level="debug" if DEBUG else "info"
    )
