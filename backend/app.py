"""
FastAPI Backend for Egyptian Stock Exchange News Pipeline
Provides REST API for scraping, analyzing, and retrieving financial news
"""
import logging
import json
from typing import Optional, List
from uuid import UUID
from fastapi import FastAPI, HTTPException, Query , Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import os
from datetime import datetime
from pathlib import Path
from db import get_db
from sqlalchemy.orm import Session
import models
import services
from schemas import (
    User, UserCreate, UserUpdate, UserList, LoginRequest, Message, MessageCreate, MessageList,
    WatchlistCreate, WatchlistResponse, PortfolioHoldingCreate, PortfolioHoldingUpdate, PortfolioHoldingResponse
)

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
    redirect_slashes=False 
)
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
if isinstance(CORS_ORIGINS, list):
    origins.extend([o for o in CORS_ORIGINS if o not in origins])

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    user_id: Optional[UUID] = None  # Optional: for chat memory persistence
    max_news: Optional[int] = 10
    user_risk_profile: Optional[str] = None
    risk_answers: Optional[dict] = None
    history: Optional[List[dict]] = []


class ChatAliasRequest(BaseModel):
    message: str
    risk_profile: Optional[str] = None
    horizon: Optional[str] = None
    drawdown: Optional[str] = None
    style: Optional[str] = None
    max_news: Optional[int] = 10
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
    """Get list of available Egyptian Stock Exchange companies"""
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
    """Get company info by ticker symbol"""
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
    ticker = ticker.upper()
    company_name = None
    for _, (name, symbol) in COMPANIES.items():
        if symbol == ticker:
            company_name = name
            break
    
    if not company_name:
        raise HTTPException(status_code=400, detail=f"Company {ticker} not found")
    
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
async def analyze_articles(payload: AnalyzeRequest):
    ticker = payload.ticker.upper()
    articles = payload.articles

    if not articles or len(articles) == 0:
        raise HTTPException(status_code=400, detail="No articles provided")
    
    try:
        results = analyze_news_batch(articles)
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
    ticker = ticker.upper()
    scrape_response = await start_scraping(ticker, max_news)
    if scrape_response["count"] == 0:
        raise HTTPException(status_code=404, detail="No articles found")
    
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
async def chat_message(payload: ChatMessageRequest, db: Session = Depends(get_db)):
    """Chat with the AI about EGX stocks"""
    try:
        # Load last analysis context for follow-ups
        last_context = None
        if payload.user_id:
            # User authenticated - try to load cached analysis
            try:
                last_context = services.get_last_file_based_analysis_context(db, payload.user_id)
                if last_context:
                    logger.info(f"Loaded last context: ticker={last_context.get('ticker')}")
            except Exception as e:
                logger.warning(f"Could not load analysis context: {e}")
        
        # Run pipeline with optional context
        result = run_chat_pipeline(
            user_message=payload.message,
            risk_answers=payload.risk_answers or {},
            user_risk_profile=payload.user_risk_profile,
            max_news=payload.max_news or 20,
            chat_history=payload.history or [],
            last_analysis_context=last_context,
        )
        
        # Save analysis context for follow-ups
        logger.info(f"DEBUG: Checking save conditions: user_id={payload.user_id}, has_metadata={bool(result.get('metadata'))}")
        if payload.user_id and result.get("metadata"):
            logger.info(f"DEBUG: user_id={payload.user_id}")
            logger.info(f"DEBUG: result metadata keys={result.get('metadata', {}).keys()}")
            logger.info(f"DEBUG: result metadata ticker={result.get('metadata', {}).get('ticker')}")
            try:
                saved = services.save_file_based_analysis_context(
                    db,
                    payload.user_id,
                    result.get("metadata")
                )
                logger.info(f"DEBUG: save_file_based_analysis_context returned={saved}")
                logger.info(f"Saved analysis context for user {payload.user_id}")
            except Exception as e:
                logger.error(f"Could not save analysis context: {e}", exc_info=True)
        else:
            logger.info(f"DEBUG: Skipping context save: user_id={payload.user_id}, metadata_exists={bool(result.get('metadata'))}")
        
        return {"status": "success", **result}
    except Exception as exc:
        logger.error(f"Chat pipeline failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat pipeline failed: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# WATCHLIST & PORTFOLIO ENDPOINTS (COMPATIBLE WITH BOTH FRONTENDS)
# ─────────────────────────────────────────────────────────────────────────────

# تعديل لدعم جلب الـ Watchlist بالطريقتين (?user_id= أو /watchlist/id)
@app.get("/watchlist", response_model=List[WatchlistResponse], tags=["Watchlist"])
async def get_user_watchlist_query(user_id: Optional[UUID] = Query(None), db: Session = Depends(get_db)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id query parameter is required")
    try:
        return services.get_watchlist(db, user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/watchlist/{user_id}", response_model=List[WatchlistResponse], tags=["Watchlist"])
async def get_user_watchlist_path(user_id: UUID, db: Session = Depends(get_db)):
    try:
        return services.get_watchlist(db, user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/watchlist", response_model=WatchlistResponse, tags=["Watchlist"])
async def add_to_watchlist(data: WatchlistCreate, db: Session = Depends(get_db)):
    try:
        return services.add_watchlist_item(db, data)
    except Exception as exc:
        logger.error(f"Failed to add watchlist item: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add watchlist item: {str(exc)}")

@app.delete("/watchlist/{item_id}", tags=["Watchlist"])
async def remove_from_watchlist(item_id: UUID, db: Session = Depends(get_db)):
    try:
        item = services.remove_watchlist_item(db, item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# تعديل لدعم جلب الـ Portfolio بالطريقتين (?user_id= أو /portfolio/id)
@app.get("/portfolio", response_model=List[PortfolioHoldingResponse], tags=["Portfolio"])
async def get_user_portfolio_query(user_id: Optional[UUID] = Query(None), db: Session = Depends(get_db)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id query parameter is required")
    try:
        return services.get_portfolio(db, user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/portfolio/{user_id}", response_model=List[PortfolioHoldingResponse], tags=["Portfolio"])
async def get_user_portfolio_path(user_id: UUID, db: Session = Depends(get_db)):
    try:
        return services.get_portfolio(db, user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/portfolio", response_model=PortfolioHoldingResponse, tags=["Portfolio"])
async def add_to_portfolio(data: PortfolioHoldingCreate, db: Session = Depends(get_db)):
    try:
        return services.add_portfolio_holding(db, data)
    except Exception as exc:
        logger.error(f"Failed to add portfolio holding: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add portfolio holding: {str(exc)}")

@app.patch("/portfolio/{holding_id}", response_model=PortfolioHoldingResponse, tags=["Portfolio"])
async def update_portfolio(holding_id: UUID, data: PortfolioHoldingUpdate, db: Session = Depends(get_db)):
    try:
        holding = services.update_portfolio_holding(db, holding_id, data)
        if not holding:
            raise HTTPException(status_code=404, detail="Holding not found")
        return holding
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.delete("/portfolio/{holding_id}", tags=["Portfolio"])
async def delete_from_portfolio(holding_id: UUID, db: Session = Depends(get_db)):
    try:
        holding = services.delete_portfolio_holding(db, holding_id)
        if not holding:
            raise HTTPException(status_code=404, detail="Holding not found")
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat", tags=["Chat"])
async def chat_message_alias(payload: ChatAliasRequest):
    risk_answers = {
        "investment_horizon": payload.horizon or "medium",
        "max_drawdown_tolerance": payload.drawdown or "medium",
        "style": payload.style or "balanced",
    }
    try:
        result = run_chat_pipeline(
            user_message=payload.message,
            risk_answers=risk_answers,
            user_risk_profile=payload.risk_profile,
            max_news=int(payload.max_news or 20),
            chat_history=payload.history,
        )
        return {"status": "success", **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/output/{filename}", tags=["Output"])
async def download_output(filename: str):
    from config import OUTPUT_DIR
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath, media_type="application/json")


@app.get("/output-schema", tags=["Documentation"])
async def get_output_schema():
    return {
        "description": "Financial news analysis output schema",
        "schema": json.loads(OUTPUT_SCHEME)
    }

# ─────────────────────────────────────────────────────────────────────────────
# USERS & MESSAGES ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/users", tags=["Users"], response_model=UserList)
async def list_users(db: Session = Depends(get_db)):
    users = services.get_users(db)
    return UserList(count=len(users), users=[User.from_orm(u) for u in users])

@app.get("/users/{email}", tags=["Users"], response_model=User)
async def get_user(email: str, db: Session = Depends(get_db)):
    user = services.get_user(db, email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return User.from_orm(user)

@app.post("/users", tags=["Users"], response_model=User)
async def create_user(user_data: UserCreate, db: Session = Depends(get_db)):
    existing_user = services.get_user(db, user_data.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = services.create_user(db, user_data)
    return User.from_orm(user)

@app.put("/users/{user_id}", tags=["Users"], response_model=User)
async def update_user(user_id: UUID, user_data: UserUpdate, db: Session = Depends(get_db)):
    updated_user = services.update_user(db, user_id, user_data)
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")
    return User.from_orm(updated_user)

@app.delete("/users/{user_id}", tags=["Users"], response_model=User)
async def delete_user(user_id: UUID, db: Session = Depends(get_db)):
    deleted_user = services.delete_user(db, user_id)
    if not deleted_user:
        raise HTTPException(status_code=404, detail="User not found")
    return User.from_orm(deleted_user)

@app.post("/login", tags=["Users"])
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = services.authenticate_user(db, request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"message": "Login successful", "user": User.from_orm(user)}

@app.get("/messages/{user_id}", tags=["Messages"])
async def list_message(user_id: UUID, db: Session = Depends(get_db)):
    messages = services.get_message(db, user_id=user_id)
    return [Message.from_orm(m) for m in messages]

@app.post("/messages", tags=["Messages"], response_model=Message)
async def create_message(message_data: MessageCreate, db: Session = Depends(get_db)):
    message = services.create_message(db, message_data)
    return Message.from_orm(message)

@app.delete("/messages/{message_id}", tags=["Messages"], response_model=Message)
async def delete_message(message_id: UUID, db: Session = Depends(get_db)):
    deleted_message = services.delete_message(db, message_id)
    if not deleted_message:
        raise HTTPException(status_code=404, detail="Message not found")
    return Message.from_orm(deleted_message)

@app.delete("/messages/user/{user_id}", tags=["Messages"])
async def delete_messages_by_user(user_id: UUID, db: Session = Depends(get_db)):
    deleted_messages = services.delete_messages_by_user(db, user_id)
    return {"deleted_count": len(deleted_messages), "messages": [Message.from_orm(m) for m in deleted_messages]}

# ─────────────────────────────────────────────────────────────────────────────
# ERROR HANDLERS & STARTUP
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status": "error"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status": "error"}
    )

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 70)
    logger.info("🚀 Egyptian Stock Exchange News Pipeline API Starting")
    logger.info("=" * 70)
    try:
        from db import create_tables
        create_tables()
        logger.info("✅ Database tables initialized")
    except Exception as exc:
        logger.error(f"Failed to initialize database tables: {exc}", exc_info=True)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 API shutting down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",  # تم تعديلها لتشير للملف الحالي بشكل صحيح عند الاستدعاء المباشر
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG,
        log_level="debug" if DEBUG else "info"
    )
