"""
Configuration file for Egyptian Stock Exchange News Pipeline
"""
import json
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Model ID on HuggingFace
MODEL_ID = "tegana/qwen2.5-arabic-finance-news-parser"

# System message for the model
SYSTEM_MESSAGE = "\n".join([
    "You are a professional Arabic financial news parser.",
    "You will be given a news article (`## Article`) and an output schema (`## Output Scheme`).",
    "Extract the required structured information and return ONLY a valid JSON object.",
    "Do not add any introduction, explanation, or markdown fences."
])

# Output schema - defines what data to extract from each article
OUTPUT_SCHEME = json.dumps({
    "event_type": "earnings|capital_increase|capital_decrease|dividends|acquisition|sale_of_stake|financing|project|board_decision|regulatory_approval|analysis_financial|stock_exchange_decision|other",
    "sentiment": "positive|negative|neutral",
    "impact_level": "high|medium|low",
    "financials": {
        "net_profit_current": None,
        "net_profit_previous": None,
        "revenue_current": None,
        "revenue_previous": None,
        "capital_before": None,
        "capital_after": None,
        "capital_increase": None,
        "dividend_per_share": None,
        "financing_amount": None,
        "project_cost": None,
        "percentage_change": None
    },
    "currency": "EGP",
    "short_summary": "ملخص من 3 إلى 5 جمل يغطي طبيعة الحدث والأرقام الرئيسية والأثر المتوقع"
}, ensure_ascii=False)

# ─────────────────────────────────────────────────────────────────────────────
# COMPANIES LIST
# ─────────────────────────────────────────────────────────────────────────────

COMPANIES = {
    "1":  ("البنك التجاري الدولي",       "COMI"),
    "2":  ("المصرية للاتصالات",          "ETEL"),
    "3":  ("فوري",                       "FWRY"),
    "4":  ("هيرميس",                     "HRHO"),
    "5":  ("السويدي",                    "SWDY"),
    "6":  ("أوراسكوم كونستراكشون",       "ORAS"),
    "7":  ("أبو قير للأسمدة",            "ABUK"),
    "8":  ("طلعت مصطفى",                 "TMGH"),
    "9":  ("حديد عز",                    "ESRS"),
    "10": ("سوديك",                      "OCDI"),
    "11": ("أورانج",                      "ORAS"),
    "12": ("مدينة مصر",                  "MASR"),
    "13": ("جهينة للصناعات الغذائية",                       "JUFO"),
    "14": ("دومتي",                     "DOMT"),
    "15": ("عبور لاند",                    "OLFI"),
    "16": ("إيسترن كومباني",        "EAST"),
    "17": ("النساجون الشرقيون",              "ORWE"),
    "18": ("غاز مصر",                    "EGAS"),
    "19": ("إيديتا للصناعات الغذائية",                       "EFID"),
    "20": ("موبكو للأسمدة",                          "MFPC"),
    "21": ("أوراسكوم للإنشاء",           "ORAS"),
    "22": ("بالم هيلز للتعمير",                  "PHDC"),
    "23": ("بنك كيو إن بي الأهلي",                               "QNBA"),
    "24": ("إي فاينانس للاستثمارات",                             "EFIH"),
    "25": ("بنك فيصل الإسلامي",                             "FAIT"),
    "26": ("عامر جروب",              "AMER"),
    "27": ("بورتو جروب",                "PORT"),
    "28": ("مصر للألومنيوم",                    "EALU"),
    "29": ("العربية للأدوية",                       "ADPC"),
    "30": ("دايس للملابس",                          "DSCW"),
}

# ─────────────────────────────────────────────────────────────────────────────
# SCRAPING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Base URL for news source
NEWS_SOURCE_URL = "https://www.mubasher.info/markets/EGX/stocks/{ticker}/news"

# Default number of news articles to scrape
DEFAULT_MAX_NEWS = 20

# Junk markers to filter out from articles
JUNK_MARKERS = [
    "هذا الخبر خاص بخدمة الاخبار المدفوعة",
    "حمل تطبيق معلومات مباشر",
    "تنويه مهم:",
    "القرارات الاستثمارية مسؤولية كاملة",
    "للمتابعة قناتنا الرسمية",
    "تابعوا آخر أخبار البورصة",
    "لمتابعة آخر أخبار البنوك",
    "ترشيحات"
]

# ─────────────────────────────────────────────────────────────────────────────
# PATHS AND OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

# Output directory for JSON files
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logs directory
LOGS_DIR = os.environ.get("LOGS_DIR", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# API CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Backend API host and port
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))

# CORS allowed origins
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")

# ─────────────────────────────────────────────────────────────────────────────
# MODAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Modal API token (set via environment variable)
MODAL_API_TOKEN = os.environ.get("MODAL_API_TOKEN", "")

# Use Modal for inference? (True for remote, False for local)
USE_MODAL = True

# Groq configuration for Part 3 decision engine
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE FLAGS
# ─────────────────────────────────────────────────────────────────────────────

# Enable caching of news articles
ENABLE_CACHE = os.environ.get("ENABLE_CACHE", "True").lower() == "true"

# Cache TTL in seconds (24 hours)
CACHE_TTL = 86400

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

ENV = os.environ.get("ENV", "development")
DEBUG = ENV == "development"
