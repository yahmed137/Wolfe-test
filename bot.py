# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║   🤖 SAUDI STOCK MARKET TELEGRAM BOT                                      ║
# ║                                                                            ║
# ║   Features:                                                                ║
# ║     1. Wolfe Wave Scanner — scans all Tadawul stocks for Wolfe patterns    ║
# ║     2. Digital Analyzer   — generates comprehensive PDF reports            ║
# ║                                                                            ║
# ║   Author    : [Your Name]                                                  ║
# ║   Version   : 2.0                                                         ║
# ║   Updated   : 2025                                                        ║
# ║   Platform  : Telegram Bot (python-telegram-bot)                           ║
# ║   Hosting   : Render.com (webhook) or Local (polling)                      ║
# ║                                                                            ║
# ║   SECTION INDEX:                                                           ║
# ║     0   — Imports & Library Setup                                          ║
# ║     0b  — Optional Scraper Dependencies                                   ║
# ║     0c  — Logging & Bot Token                                             ║
# ║     1   — Font Setup (Cairo + Amiri)                                      ║
# ║     2   — Arabic Text Helpers                                             ║
# ║     3   — Company Names Dictionary                                        ║
# ║     3a  — Ticker Aliases & Flexible Lookup                                ║
# ║     3b  — Sector & Industry Mapping                                       ║
# ║     3c  — Static Fundamental Data (STOCKS)                                ║
# ║     3d  — STOCKS Data Enrichment Functions                                ║
# ║     4   — PDF Theme Constants (Colors, Page Size)                         ║
# ║     5   — Number & Percent Formatters                                     ║
# ║     6   — Stock Data Fetching & All Technical Indicators                  ║
# ║     6b  — Supertrend Helper                                              ║
# ║     6c  — Technical Indicator Computation                                 ║
# ║     6d  — Performance & Risk Metrics                                      ║
# ║     6e  — 20-Point Scoring System                                         ║
# ║     6f  — Signal Generation                                              ║
# ║     6g  — Recommendation & Decision Helpers                               ║
# ║     6h  — Candlestick Pattern Detection                                  ║
# ║     6i  — Divergence Detection (RSI / MACD)                              ║
# ║     6j  — Support & Resistance Detection                                 ║
# ║     6k  — Technical Review Text Generator                                ║
# ║     7   — Chart Generation Functions                                      ║
# ║     8   — PDF Report Class                                               ║
# ║     9   — Report Builder (Thread Executor)                               ║
# ║     10  — Wolfe Wave Detection & Charting                                 ║
# ║     11  — Ticker Lists & Timeframe Map                                    ║
# ║     12  — Telegram Keyboard Builders                                      ║
# ║     13  — Bot Messages (Arabic)                                           ║
# ║     14  — Landing HTML Page                                               ║
# ║     15  — Telegram Bot Handlers                                           ║
# ║     16  — Main Entry Point                                                ║
# ║                                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0: IMPORTS & LIBRARY SETUP
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Import all required libraries.
#
# TO MODIFY:
#   • Add new charting library      → import here
#   • Remove Wolfe wave scanning    → remove selenium imports
#   • Switch from yfinance          → change data fetching import
# ═══════════════════════════════════════════════════════════════════════════════

import os
import io
import re
import asyncio
import logging
import warnings
import urllib.request
warnings.filterwarnings('ignore')

# --- Arabic text rendering ---
import arabic_reshaper
from bidi.algorithm import get_display

# --- Web server for webhook mode (Render.com) ---
from aiohttp import web as aio_web

# --- Standard I/O and date/time ---
from io import BytesIO
from datetime import datetime, timedelta

# --- Financial data fetching ---
import yfinance as yf

# --- Numerical and data processing ---
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Chart generation (matplotlib + mplfinance) ---
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import mplfinance as mpf

# --- PDF generation (reportlab) ---
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- Telegram Bot framework ---
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0b: OPTIONAL SCRAPER DEPENDENCIES (Selenium / BeautifulSoup)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Optional imports for web scraping (headless Chrome).
#           If not installed, bot falls back to yfinance-only mode.
#
# TO MODIFY:
#   • Never use scraping → remove this entire try/except block
#   • Add new scraping source → import its deps here
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import sys
    import time
    import subprocess
    from bs4 import BeautifulSoup
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    STOCKS_AVAILABLE = True
except ImportError:
    STOCKS_AVAILABLE = False
    logger_stub = logging.getLogger(__name__)
    logger_stub.warning("STOCKS scraper deps not installed; falling back to yfinance only.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0c: LOGGING & BOT TOKEN CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Configure logging and read Telegram bot token.
#
# TO MODIFY:
#   • Change log level → replace logging.INFO with logging.DEBUG
#   • BOT_TOKEN → set via: export BOT_TOKEN="your_token"
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: FONT SETUP (Cairo for Wolfe charts, Amiri for PDF reports)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Download and register Arabic fonts.
#
# FILES PRODUCED:
#   • Cairo-Regular.ttf   — Wolfe Wave chart labels
#   • Amiri-Regular.ttf   — PDF body text
#   • Amiri-Bold.ttf      — PDF headings
#
# TO MODIFY:
#   • Use different Arabic font → change URLs and filenames
#   • Add font weight (e.g., Light) → add path + download call
#   • Re-download fonts → delete .ttf files
# ═══════════════════════════════════════════════════════════════════════════════

HERE = os.path.dirname(os.path.abspath(__file__))

# --- Font file paths ---
CAIRO_PATH      = os.path.join(HERE, 'Cairo-Regular.ttf')
AMIRI_REG_PATH  = os.path.join(HERE, 'Amiri-Regular.ttf')
AMIRI_BOLD_PATH = os.path.join(HERE, 'Amiri-Bold.ttf')

# --- PDF font registration names ---
AR_FONT      = 'Amiri'       # Used by reportlab for body text
AR_FONT_BOLD = 'Amiri-Bold'  # Used by reportlab for bold text

# --- Regex to detect Arabic characters ---
AR_RE = re.compile(r'[\u0600-\u06FF]')

# --- Matplotlib font properties (populated during init) ---
MPL_FONT_PROP      = None  # Regular — chart labels
MPL_FONT_PROP_BOLD = None  # Bold — chart titles


def _download_font(url, path):
    """Download a font file from URL if not already cached locally."""
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
            logger.info(f'Downloaded: {os.path.basename(path)}')
        except Exception as e:
            logger.warning(f'Font download failed ({os.path.basename(path)}): {e}')


def _init_fonts():
    """
    Initialize all fonts for the application.

    Steps:
      1. Download font files (Cairo, Amiri Reg, Amiri Bold)
      2. Register with reportlab (PDF engine)
      3. Register with matplotlib (chart engine)
      4. Set matplotlib default font family

    Returns:
        str: Arabic font name for Wolfe charts
    """
    global MPL_FONT_PROP, MPL_FONT_PROP_BOLD, ARABIC_FONT

    # Step 1: Download fonts
    _download_font(
        'https://github.com/google/fonts/raw/main/ofl/cairo/static/Cairo-Regular.ttf',
        CAIRO_PATH,
    )
    _download_font(
        'https://github.com/google/fonts/raw/refs/heads/main/ofl/amiri/Amiri-Regular.ttf',
        AMIRI_REG_PATH,
    )
    _download_font(
        'https://github.com/google/fonts/raw/refs/heads/main/ofl/amiri/Amiri-Bold.ttf',
        AMIRI_BOLD_PATH,
    )

    # Step 2: Register with reportlab
    if os.path.exists(AMIRI_REG_PATH):
        try:
            pdfmetrics.registerFont(TTFont(AR_FONT, AMIRI_REG_PATH))
        except Exception:
            pass
    if os.path.exists(AMIRI_BOLD_PATH):
        try:
            pdfmetrics.registerFont(TTFont(AR_FONT_BOLD, AMIRI_BOLD_PATH))
        except Exception:
            pass

    # Step 3: Register with matplotlib
    for fp in [AMIRI_REG_PATH, AMIRI_BOLD_PATH, CAIRO_PATH]:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)

    # Step 4: Set matplotlib defaults
    if os.path.exists(AMIRI_REG_PATH):
        MPL_FONT_PROP = fm.FontProperties(fname=AMIRI_REG_PATH)
        plt.rcParams['font.family'] = MPL_FONT_PROP.get_name()
    if os.path.exists(AMIRI_BOLD_PATH):
        MPL_FONT_PROP_BOLD = fm.FontProperties(fname=AMIRI_BOLD_PATH)
    plt.rcParams['axes.unicode_minus'] = False

    # Determine Wolfe chart font
    if os.path.exists(CAIRO_PATH):
        prop = fm.FontProperties(fname=CAIRO_PATH)
        ARABIC_FONT = prop.get_name()
    else:
        ARABIC_FONT = 'DejaVu Sans'
    return ARABIC_FONT


# --- Run font initialization at module load ---
ARABIC_FONT = _init_fonts()
logger.info(f'Fonts initialised. Arabic chart font: {ARABIC_FONT}')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: ARABIC TEXT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Correctly render Arabic text in charts and PDFs.
#           Arabic requires reshaping (letter connections) + bidi (RTL direction).
#
# TO MODIFY:
#   • Add another RTL language → extend AR_RE regex
#   • Switch to native-Arabic library → remove these helpers
# ═══════════════════════════════════════════════════════════════════════════════

def ar(text: str) -> str:
    """Reshape + bidi for Wolfe chart labels."""
    try:
        return get_display(arabic_reshaper.reshape(str(text)))
    except Exception:
        return str(text)


def rtl(txt):
    """Reshape + bidi for PDF / matplotlib. Only processes Arabic strings."""
    if txt is None:
        return ''
    s = str(txt)
    if AR_RE.search(s):
        return get_display(arabic_reshaper.reshape(s))
    return s


def tx(txt):
    """Safe rtl() wrapper — returns '-' for None values."""
    if txt is None:
        return '-'
    return rtl(str(txt))


def short_text(s, n=40):
    """Truncate text to n characters with ellipsis."""
    s = str(s or '-')
    return s if len(s) <= n else s[:n - 1] + '…'


def safe(info, key, default=None):
    """Safely get value from dict, treating None as default."""
    v = info.get(key)
    return default if v is None else v


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: COMPANY NAMES DICTIONARY
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Map every Tadawul ticker → Arabic company name.
#
# TO MODIFY:
#   • ADD a stock    → add entry: '9999.SR': 'اسم الشركة'
#   • REMOVE a stock → delete the entry
#   • Also update: SECTOR_MAP, TADAWUL_TICKERS, STOCKS_STATIC_DATA
# ═══════════════════════════════════════════════════════════════════════════════

COMPANY_NAMES = {
    '^TASI.SR':'تاسي','1010.SR':'الرياض','1020.SR':'الجزيرة','1030.SR':'الإستثمار',
    '1050.SR':'بي اس اف','1060.SR':'الأول','1080.SR':'العربي','1111.SR':'مجموعة تداول',
    '1120.SR':'الراجحي','1140.SR':'البلاد','1150.SR':'الإنماء','1180.SR':'الأهلي',
    '1182.SR':'أملاك','1183.SR':'سهل','1201.SR':'تكوين','1202.SR':'مبكو',
    '1210.SR':'بي سي آي','1211.SR':'معادن','1212.SR':'أسترا الصناعية','1213.SR':'نسيج',
    '1214.SR':'شاكر','1301.SR':'أسلاك','1302.SR':'بوان','1303.SR':'الصناعات الكهربائية',
    '1304.SR':'اليمامة للحديد','1320.SR':'أنابيب السعودية','1321.SR':'أنابيب الشرق',
    '1322.SR':'أماك','1323.SR':'يو سي آي سي','1810.SR':'سيرا','1820.SR':'بان',
    '1830.SR':'لجام للرياضة','1831.SR':'مهارة','1832.SR':'صدر','1833.SR':'الموارد',
    '1834.SR':'سماسكو','1835.SR':'تمكين','2001.SR':'كيمانول','2010.SR':'سابك',
    '2020.SR':'سابك للمغذيات الزراعية','2030.SR':'المصافي','2040.SR':'الخزف السعودي',
    '2050.SR':'مجموعة صافولا','2060.SR':'التصنيع','2070.SR':'الدوائية','2080.SR':'الغاز',
    '2081.SR':'الخريف','2082.SR':'أكوا','2083.SR':'مرافق','2084.SR':'مياهنا',
    '2090.SR':'جبسكو','2100.SR':'وفرة','2110.SR':'الكابلات السعودية','2120.SR':'متطورة',
    '2130.SR':'صدق','2140.SR':'أيان','2150.SR':'زجاج','2160.SR':'أميانتيت',
    '2170.SR':'اللجين','2180.SR':'فيبكو','2190.SR':'سيسكو القابضة','2200.SR':'أنابيب',
    '2210.SR':'نماء للكيماويات','2220.SR':'معدنية','2222.SR':'أرامكو السعودية',
    '2223.SR':'لوبريف','2230.SR':'الكيميائية','2240.SR':'صناعات','2250.SR':'المجموعة السعودية',
    '2270.SR':'سدافكو','2280.SR':'المراعي','2281.SR':'تنمية','2282.SR':'نقي',
    '2283.SR':'المطاحن الأولى','2284.SR':'المطاحن الحديثة','2285.SR':'المطاحن العربية',
    '2286.SR':'المطاحن الرابعة','2287.SR':'إنتاج','2288.SR':'نفوذ','2290.SR':'ينساب',
    '2300.SR':'صناعة الورق','2310.SR':'سبكيم العالمية','2320.SR':'البابطين',
    '2330.SR':'المتقدمة','2340.SR':'ارتيكس','2350.SR':'كيان السعودية','2360.SR':'الفخارية',
    '2370.SR':'مسك','2380.SR':'بترو رابغ','2381.SR':'الحفر العربية','2382.SR':'أديس',
    '3002.SR':'أسمنت نجران','3003.SR':'أسمنت المدينة','3004.SR':'أسمنت الشمالية',
    '3005.SR':'أسمنت ام القرى','3007.SR':'الواحة','3008.SR':'الكثيري',
    '3010.SR':'أسمنت العربية','3020.SR':'أسمنت اليمامة','3030.SR':'أسمنت السعودية',
    '3040.SR':'أسمنت القصيم','3050.SR':'أسمنت الجنوب','3060.SR':'أسمنت ينبع',
    '3080.SR':'أسمنت الشرقية','3090.SR':'أسمنت تبوك','3091.SR':'أسمنت الجوف',
    '3092.SR':'أسمنت الرياض','4001.SR':'أسواق ع العثيم','4002.SR':'المواساة',
    '4003.SR':'إكسترا','4004.SR':'دله الصحية','4005.SR':'رعاية','4006.SR':'أسواق المزرعة',
    '4007.SR':'الحمادي','4008.SR':'ساكو','4009.SR':'السعودي الألماني','4011.SR':'لازوردي',
    '4012.SR':'الأصيل','4013.SR':'سليمان الحبيب','4014.SR':'دار المعدات',
    '4015.SR':'جمجوم فارما','4016.SR':'أفالون فارما','4017.SR':'فقيه الطبية',
    '4018.SR':'الموسى','4019.SR':'اس ام سي','4020.SR':'العقارية',
    '4021.SR':'المركز الكندي الطبي','4030.SR':'البحري','4031.SR':'الخدمات الأرضية',
    '4040.SR':'سابتكو','4050.SR':'ساسكو','4051.SR':'باعظيم','4061.SR':'أنعام القابضة',
    '4070.SR':'تهامة','4071.SR':'العربية','4072.SR':'إم بي سي','4080.SR':'سناد القابضة',
    '4081.SR':'النايفات','4082.SR':'مرنة','4083.SR':'تسهيل','4084.SR':'دراية',
    '4090.SR':'طيبة','4100.SR':'مكة','4110.SR':'باتك','4130.SR':'درب السعودية',
    '4140.SR':'صادرات','4141.SR':'العمران','4142.SR':'كابلات الرياض','4143.SR':'تالكو',
    '4144.SR':'رؤوم','4145.SR':'أو جي سي','4146.SR':'جاز','4147.SR':'سي جي إس',
    '4148.SR':'الوسائل الصناعية','4150.SR':'التعمير','4160.SR':'ثمار','4161.SR':'بن داود',
    '4162.SR':'المنجم','4163.SR':'الدواء','4164.SR':'النهدي','4165.SR':'الماجد للعود',
    '4170.SR':'شمس','4180.SR':'مجموعة فتيحي','4190.SR':'جرير','4191.SR':'أبو معطي',
    '4192.SR':'السيف غاليري','4193.SR':'نايس ون','4194.SR':'محطة البناء','4200.SR':'الدريس',
    '4210.SR':'الأبحاث والإعلام','4220.SR':'إعمار','4230.SR':'البحر الأحمر',
    '4240.SR':'سينومي ريتيل','4250.SR':'جبل عمر','4260.SR':'بدجت السعودية',
    '4261.SR':'ذيب','4262.SR':'لومي','4263.SR':'سال','4264.SR':'طيران ناس',
    '4265.SR':'شري','4270.SR':'طباعة وتغليف','4280.SR':'المملكة','4290.SR':'الخليج للتدريب',
    '4291.SR':'الوطنية للتعليم','4292.SR':'عطاء','4300.SR':'دار الأركان',
    '4310.SR':'مدينة المعرفة','4320.SR':'الأندلس','4321.SR':'سينومي سنترز','4322.SR':'رتال',
    '4323.SR':'سمو','4324.SR':'بنان','4325.SR':'مسار','4326.SR':'الماجدية',
    '4327.SR':'الرمز','4330.SR':'الرياض ريت','4331.SR':'الجزيرة ريت',
    '4332.SR':'جدوى ريت الحرمين','4333.SR':'تعليم ريت','4334.SR':'المعذر ريت',
    '4335.SR':'مشاركة ريت','4336.SR':'ملكية ريت','4337.SR':'العزيزية ريت',
    '4338.SR':'الأهلي ريت 1','4339.SR':'دراية ريت','4340.SR':'الراجحي ريت',
    '4342.SR':'جدوى ريت السعودية','4344.SR':'سدكو كابيتال ريت',
    '4345.SR':'الإنماء ريت للتجزئة','4346.SR':'ميفك ريت','4347.SR':'بنيان ريت',
    '4348.SR':'الخبير ريت','4349.SR':'الإنماء ريت الفندقي','4350.SR':'الإستثمار ريت',
    '5110.SR':'كهرباء السعودية','6001.SR':'حلواني إخوان','6002.SR':'هرفي للأغذية',
    '6004.SR':'كاتريون','6010.SR':'نادك','6012.SR':'ريدان','6013.SR':'التطويرية الغذائية',
    '6014.SR':'الآمار','6015.SR':'أمريكانا','6016.SR':'برغرايززر','6017.SR':'جاهز',
    '6018.SR':'الأندية للرياضة','6019.SR':'المسار الشامل','6020.SR':'جاكو',
    '6040.SR':'تبوك الزراعية','6050.SR':'الأسماك','6060.SR':'الشرقية للتنمية',
    '6070.SR':'الجوف','6090.SR':'جازادكو','7010.SR':'اس تي سي','7020.SR':'إتحاد إتصالات',
    '7030.SR':'زين السعودية','7040.SR':'قو للإتصالات','7200.SR':'ام آي اس',
    '7201.SR':'بحر العرب','7202.SR':'سلوشنز','7203.SR':'علم','7204.SR':'توبي',
    '7211.SR':'عزم','8010.SR':'التعاونية','8012.SR':'جزيرة تكافل','8020.SR':'ملاذ للتأمين',
    '8030.SR':'ميدغلف للتأمين','8040.SR':'متكاملة','8050.SR':'سلامة','8060.SR':'ولاء',
    '8070.SR':'الدرع العربي','8100.SR':'سايكو','8120.SR':'إتحاد الخليج الأهلية',
    '8150.SR':'أسيج','8160.SR':'التأمين العربية','8170.SR':'الاتحاد',
    '8180.SR':'الصقر للتأمين','8190.SR':'المتحدة للتأمين','8200.SR':'الإعادة السعودية',
    '8210.SR':'بوبا العربية','8230.SR':'تكافل الراجحي','8240.SR':'تْشب',
    '8250.SR':'جي آي جي','8260.SR':'الخليجية العامة','8270.SR':'ليفا','8280.SR':'ليفا',
    '8300.SR':'الوطنية','8310.SR':'أمانة للتأمين','8311.SR':'عناية','8313.SR':'رسن',
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3a: TICKER ALIASES & FLEXIBLE LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Allow flexible ticker search by code, name, or alias.
#
# TO MODIFY:
#   • Add alias → add to TICKER_ALIASES: "nickname": "XXXX.SR"
#   • Change matching logic → edit find_ticker()
# ═══════════════════════════════════════════════════════════════════════════════

TICKER_ALIASES = {
    "TASI":     "^TASI.SR",
    "^TASI":    "^TASI.SR",
    "^TASI.SR": "^TASI.SR",
    "تاسي":     "^TASI.SR",
    "تاسى":     "^TASI.SR",
}


def _normalize_arabic(text: str) -> str:
    """Normalize Arabic chars for consistent matching (أ/إ/آ→ا, ى→ي)."""
    text = text.strip()
    text = text.replace("ى", "ي").replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    return text


def _normalize_key(text: str) -> str:
    """Normalize + uppercase for case-insensitive comparison."""
    return _normalize_arabic(text.upper())


def get_name(ticker: str) -> str:
    """Get Arabic company name for ticker. Falls back to ticker string."""
    return COMPANY_NAMES.get(ticker, ticker)


# --- Build normalized alias map once at startup ---
_ALIAS_MAP = {
    _normalize_key(alias): canonical
    for alias, canonical in TICKER_ALIASES.items()
}


def find_ticker(query: str) -> str | None:
    """
    Flexible ticker lookup. Accepts:
      - "2222" or "2222.SR" → ticker code
      - "TASI" or "تاسي"   → alias
      - "أرامكو" or "راجح" → Arabic name (exact or partial)

    Returns: Full ticker (e.g. '1120.SR') or None
    """
    query = query.strip()
    query_key = _normalize_key(query)
    query_normalized = _normalize_arabic(query)

    # 0) Alias lookup (fast path)
    if query_key in _ALIAS_MAP:
        return _ALIAS_MAP[query_key]

    # Iterate COMPANY_NAMES
    for ticker, name in COMPANY_NAMES.items():
        ticker_upper = ticker.upper()
        # 1) Direct match
        if query_key == ticker_upper:
            return ticker
        # 2) Without .SR suffix
        code = ticker_upper[:-3] if ticker_upper.endswith(".SR") else ticker_upper
        if query_key == code:
            return ticker
        # 3) Without ^ and .SR
        code_clean = code.lstrip("^")
        if query_key == code_clean:
            return ticker
        # 4) Exact Arabic name match
        if query_normalized == _normalize_arabic(name):
            return ticker

    # 5) Partial Arabic match (min 2 chars)
    if len(query_normalized) >= 2:
        for ticker, name in COMPANY_NAMES.items():
            if query_normalized in _normalize_arabic(name):
                return ticker
    return None


# ── Quick self-test at startup ──
tests = ["tasi", "TASI", "^TASI.SR", "تاسي", "تاسى", "^tasi.sr", "1010", "1010.SR", "الرياض"]
print("Results:")
for t in tests:
    result = find_ticker(t)
    name = get_name(result) if result else "❌ Not found"
    print(f"  '{t}'  →  {result}  ({name})")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3b: SECTOR & INDUSTRY MAPPING
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Map each ticker → (sector, industry) for PDF reports.
#           Fallback when yfinance doesn't provide sector data.
#
# STRUCTURE : 'XXXX.SR': ('القطاع', 'الصناعة')
#
# TO MODIFY:
#   • Add stock → add entry to dict
#   • New sector → just use a new string
# ═══════════════════════════════════════════════════════════════════════════════

SECTOR_MAP = {
    # ── البنوك (Banks) ──
    '1010.SR':('المالية','البنوك'),'1020.SR':('المالية','البنوك'),'1030.SR':('المالية','البنوك'),
    '1050.SR':('المالية','البنوك'),'1060.SR':('المالية','البنوك'),'1080.SR':('المالية','البنوك'),
    '1120.SR':('المالية','البنوك'),'1140.SR':('المالية','البنوك'),'1150.SR':('المالية','البنوك'),
    '1180.SR':('المالية','البنوك'),'1182.SR':('المالية','البنوك'),'1183.SR':('المالية','البنوك'),
    # ── التأمين (Insurance) ──
    '8010.SR':('المالية','التأمين'),'8012.SR':('المالية','التأمين'),'8020.SR':('المالية','التأمين'),
    '8030.SR':('المالية','التأمين'),'8040.SR':('المالية','التأمين'),'8050.SR':('المالية','التأمين'),
    '8060.SR':('المالية','التأمين'),'8070.SR':('المالية','التأمين'),'8100.SR':('المالية','التأمين'),
    '8120.SR':('المالية','التأمين'),'8150.SR':('المالية','التأمين'),'8160.SR':('المالية','التأمين'),
    '8170.SR':('المالية','التأمين'),'8180.SR':('المالية','التأمين'),'8190.SR':('المالية','التأمين'),
    '8200.SR':('المالية','التأمين'),'8210.SR':('المالية','التأمين'),'8230.SR':('المالية','التأمين'),
    '8240.SR':('المالية','التأمين'),'8250.SR':('المالية','التأمين'),'8260.SR':('المالية','التأمين'),
    '8270.SR':('المالية','التأمين'),'8280.SR':('المالية','التأمين'),'8300.SR':('المالية','التأمين'),
    '8310.SR':('المالية','التأمين'),'8311.SR':('المالية','التأمين'),'8313.SR':('المالية','التأمين'),
    # ── البتروكيماويات (Petrochemicals) ──
    '2001.SR':('المواد الأساسية','البتروكيماويات'),'2010.SR':('المواد الأساسية','البتروكيماويات'),
    '2020.SR':('المواد الأساسية','البتروكيماويات'),'2060.SR':('المواد الأساسية','البتروكيماويات'),
    '2080.SR':('المواد الأساسية','البتروكيماويات'),'2090.SR':('المواد الأساسية','البتروكيماويات'),
    '2110.SR':('المواد الأساسية','البتروكيماويات'),'2120.SR':('المواد الأساسية','البتروكيماويات'),
    '2130.SR':('المواد الأساسية','البتروكيماويات'),'2140.SR':('المواد الأساسية','البتروكيماويات'),
    '2150.SR':('المواد الأساسية','البتروكيماويات'),'2160.SR':('المواد الأساسية','البتروكيماويات'),
    '2170.SR':('المواد الأساسية','البتروكيماويات'),'2180.SR':('المواد الأساسية','البتروكيماويات'),
    '2190.SR':('المواد الأساسية','البتروكيماويات'),'2200.SR':('المواد الأساسية','البتروكيماويات'),
    '2210.SR':('المواد الأساسية','البتروكيماويات'),'2220.SR':('المواد الأساسية','البتروكيماويات'),
    '2230.SR':('المواد الأساسية','البتروكيماويات'),'2240.SR':('المواد الأساسية','البتروكيماويات'),
    '2250.SR':('المواد الأساسية','البتروكيماويات'),'2270.SR':('المواد الأساسية','البتروكيماويات'),
    '2290.SR':('المواد الأساسية','البتروكيماويات'),'2300.SR':('المواد الأساسية','البتروكيماويات'),
    '2310.SR':('المواد الأساسية','البتروكيماويات'),'2320.SR':('المواد الأساسية','البتروكيماويات'),
    '2330.SR':('المواد الأساسية','البتروكيماويات'),'2340.SR':('المواد الأساسية','البتروكيماويات'),
    '2350.SR':('المواد الأساسية','البتروكيماويات'),'2360.SR':('المواد الأساسية','البتروكيماويات'),
    '2370.SR':('المواد الأساسية','البتروكيماويات'),'2380.SR':('المواد الأساسية','البتروكيماويات'),
    # ── الطاقة (Energy) ──
    '2222.SR':('الطاقة','النفط والغاز'),'2030.SR':('الطاقة','النفط والغاز'),
    '2381.SR':('الطاقة','النفط والغاز'),'2382.SR':('الطاقة','النفط والغاز'),
    # ── الإسمنت (Cement) ──
    '3002.SR':('المواد الأساسية','الإسمنت'),'3003.SR':('المواد الأساسية','الإسمنت'),
    '3004.SR':('المواد الأساسية','الإسمنت'),'3005.SR':('المواد الأساسية','الإسمنت'),
    '3007.SR':('المواد الأساسية','الإسمنت'),'3008.SR':('المواد الأساسية','الإسمنت'),
    '3010.SR':('المواد الأساسية','الإسمنت'),'3020.SR':('المواد الأساسية','الإسمنت'),
    '3030.SR':('المواد الأساسية','الإسمنت'),'3040.SR':('المواد الأساسية','الإسمنت'),
    '3050.SR':('المواد الأساسية','الإسمنت'),'3060.SR':('المواد الأساسية','الإسمنت'),
    '3080.SR':('المواد الأساسية','الإسمنت'),'3090.SR':('المواد الأساسية','الإسمنت'),
    '3091.SR':('المواد الأساسية','الإسمنت'),'3092.SR':('المواد الأساسية','الإسمنت'),
    # ── تجارة التجزئة (Retail) ──
    '4001.SR':('السلع الاستهلاكية','تجارة التجزئة'),'4003.SR':('السلع الاستهلاكية','تجارة التجزئة'),
    '4006.SR':('السلع الاستهلاكية','تجارة التجزئة'),'4011.SR':('السلع الاستهلاكية','تجارة التجزئة'),
    '4190.SR':('السلع الاستهلاكية','تجارة التجزئة'),'4192.SR':('السلع الاستهلاكية','تجارة التجزئة'),
    '4193.SR':('السلع الاستهلاكية','تجارة التجزئة'),'4240.SR':('السلع الاستهلاكية','تجارة التجزئة'),
    # ── الرعاية الصحية (Healthcare) ──
    '4002.SR':('الرعاية الصحية','الخدمات الصحية'),'4004.SR':('الرعاية الصحية','الخدمات الصحية'),
    '4005.SR':('الرعاية الصحية','الخدمات الصحية'),'4007.SR':('الرعاية الصحية','الخدمات الصحية'),
    '4009.SR':('الرعاية الصحية','الخدمات الصحية'),'4012.SR':('الرعاية الصحية','الخدمات الصحية'),
    '4013.SR':('الرعاية الصحية','الخدمات الصحية'),'4014.SR':('الرعاية الصحية','الخدمات الصحية'),
    '4015.SR':('الرعاية الصحية','الصيدلانيات'),'4016.SR':('الرعاية الصحية','الصيدلانيات'),
    '4017.SR':('الرعاية الصحية','الخدمات الصحية'),'4018.SR':('الرعاية الصحية','الخدمات الصحية'),
    '4019.SR':('الرعاية الصحية','الخدمات الصحية'),'4021.SR':('الرعاية الصحية','الخدمات الصحية'),
    # ── الاتصالات وتقنية المعلومات (Telecom & IT) ──
    '7010.SR':('الاتصالات','خدمات الاتصالات'),'7020.SR':('الاتصالات','خدمات الاتصالات'),
    '7030.SR':('الاتصالات','خدمات الاتصالات'),'7040.SR':('الاتصالات','خدمات الاتصالات'),
    '7200.SR':('الاتصالات','تقنية المعلومات'),'7201.SR':('الاتصالات','تقنية المعلومات'),
    '7202.SR':('الاتصالات','تقنية المعلومات'),'7203.SR':('الاتصالات','تقنية المعلومات'),
    '7204.SR':('الاتصالات','تقنية المعلومات'),'7211.SR':('الاتصالات','خدمات الاتصالات'),
    # ── الأغذية والمشروبات (Food & Beverages) ──
    '2040.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),'2050.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),
    '2280.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),'2281.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),
    '2282.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),'2283.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),
    '2284.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),'2285.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),
    '2286.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),'2287.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),
    '2288.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),'6001.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),
    '6002.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),'6004.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),
    '6010.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),'6012.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),
    '6013.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),'6014.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),
    '6015.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),'6016.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),
    '6017.SR':('السلع الاستهلاكية','الأغذية والمشروبات'),
    # ── الزراعة (Agriculture) ──
    '6040.SR':('السلع الاستهلاكية','الزراعة والأغذية'),
    '6050.SR':('السلع الاستهلاكية','الزراعة والأغذية'),'6060.SR':('السلع الاستهلاكية','الزراعة والأغذية'),
    '6070.SR':('السلع الاستهلاكية','الزراعة والأغذية'),'6090.SR':('السلع الاستهلاكية','الزراعة والأغذية'),
    # ── العقارات (Real Estate) ──
    '4020.SR':('العقارات','التطوير العقاري'),'4220.SR':('العقارات','التطوير العقاري'),
    '4230.SR':('العقارات','التطوير العقاري'),'4250.SR':('العقارات','التطوير العقاري'),
    '4300.SR':('العقارات','التطوير العقاري'),'4310.SR':('العقارات','التطوير العقاري'),
    '4320.SR':('العقارات','التطوير العقاري'),'4321.SR':('العقارات','مراكز التسوق'),
    '4322.SR':('العقارات','التطوير العقاري'),'4323.SR':('العقارات','التطوير العقاري'),
    '4324.SR':('العقارات','التطوير العقاري'),'4325.SR':('العقارات','التطوير العقاري'),
    # ── صناديق الريت (REITs) ──
    '4330.SR':('العقارات','صناديق الاستثمار العقاري'),'4331.SR':('العقارات','صناديق الاستثمار العقاري'),
    '4332.SR':('العقارات','صناديق الاستثمار العقاري'),'4333.SR':('العقارات','صناديق الاستثمار العقاري'),
    '4334.SR':('العقارات','صناديق الاستثمار العقاري'),'4335.SR':('العقارات','صناديق الاستثمار العقاري'),
    '4336.SR':('العقارات','صناديق الاستثمار العقاري'),'4337.SR':('العقارات','صناديق الاستثمار العقاري'),
    '4338.SR':('العقارات','صناديق الاستثمار العقاري'),'4339.SR':('العقارات','صناديق الاستثمار العقاري'),
    '4340.SR':('العقارات','صناديق الاستثمار العقاري'),'4342.SR':('العقارات','صناديق الاستثمار العقاري'),
    '4344.SR':('العقارات','صناديق الاستثمار العقاري'),'4345.SR':('العقارات','صناديق الاستثمار العقاري'),
    '4346.SR':('العقارات','صناديق الاستثمار العقاري'),'4347.SR':('العقارات','صناديق الاستثمار العقاري'),
    '4348.SR':('العقارات','صناديق الاستثمار العقاري'),'4349.SR':('العقارات','صناديق الاستثمار العقاري'),
    '4350.SR':('العقارات','صناديق الاستثمار العقاري'),
    # ── المرافق العامة (Utilities) ──
    '5110.SR':('المرافق العامة','الكهرباء والمياه'),'2082.SR':('المرافق العامة','الكهرباء والمياه'),
    '2083.SR':('المرافق العامة','الكهرباء والمياه'),'2084.SR':('المرافق العامة','الكهرباء والمياه'),
    # ── الصناعة (Industrial) ──
    '1201.SR':('الصناعة','الصناعات التحويلية'),'1202.SR':('الصناعة','الصناعات التحويلية'),
    '1210.SR':('الصناعة','التعدين والمعادن'),'1211.SR':('الصناعة','التعدين والمعادن'),
    '1212.SR':('الصناعة','الصناعات التحويلية'),'1213.SR':('الصناعة','الصناعات التحويلية'),
    '1214.SR':('الصناعة','الصناعات التحويلية'),'1301.SR':('الصناعة','الصناعات التحويلية'),
    '1302.SR':('الصناعة','الصناعات التحويلية'),'1303.SR':('الصناعة','الصناعات التحويلية'),
    '1304.SR':('الصناعة','الصناعات التحويلية'),'1320.SR':('الصناعة','الصناعات التحويلية'),
    '1321.SR':('الصناعة','الصناعات التحويلية'),'1322.SR':('الصناعة','الصناعات التحويلية'),
    '1323.SR':('الصناعة','الصناعات التحويلية'),
    '4030.SR':('الصناعة','النقل البحري'),'4031.SR':('الصناعة','الخدمات الأرضية'),
    '4040.SR':('الصناعة','النقل البري'),'4050.SR':('الصناعة','محطات الوقود'),
    '4080.SR':('الصناعة','الخدمات اللوجستية'),'4260.SR':('الصناعة','تأجير السيارات'),
    '4261.SR':('الصناعة','تأجير السيارات'),'4263.SR':('الصناعة','الخدمات اللوجستية'),
    '4264.SR':('الصناعة','النقل الجوي'),
    # ── الخدمات (Services) ──
    '4070.SR':('الخدمات','الإعلام والترفيه'),'4071.SR':('الخدمات','الإعلام والترفيه'),
    '4072.SR':('الخدمات','الإعلام والترفيه'),
    '4290.SR':('الخدمات','التعليم'),'4291.SR':('الخدمات','التعليم'),'4292.SR':('الخدمات','التعليم'),
    # ── أسواق المال (Capital Markets) ──
    '1111.SR':('المالية','أسواق المال'),
}


def get_sector_industry(ticker):
    """Return (sector, industry) tuple for ticker, or (None, None)."""
    return SECTOR_MAP.get(ticker, (None, None))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3c: STATIC FUNDAMENTAL DATA (STOCKS SOURCE)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Pre-loaded fundamental data for Saudi stocks.
#           OVERRIDES yfinance when available (yfinance is often incomplete).
#
# FIELDS PER STOCK:
#   • Numberofshare  — Shares outstanding (millions)
#   • Eps            — Earnings Per Share (trailing)
#   • Bookvalue      — Book Value per share
#   • Parallel_value — Par/Nominal value
#   • PE_ratio       — P/E ratio (or "سالب" / "أكبر من 100")
#   • PB_ratio       — P/B ratio
#
# TO MODIFY:
#   • UPDATE data → change values (refresh quarterly after earnings)
#   • ADD stock   → add "XXXX": {...}
#   • LIVE data   → replace with scraping calls
# ═══════════════════════════════════════════════════════════════════════════════

# Arabic non-numeric P/E values
_PE_NON_NUMERIC = {"سالب", "أكبر من 100", "-", ""}

STOCKS_STATIC_DATA = {
    "1010": {"Numberofshare": "3,000.00", "Eps": "3.47", "Bookvalue": "21.37", "Parallel_value": "10.00", "PE_ratio": "8.19", "PB_ratio": "1.33", "ROA": "", "ROE": "16.91"},
    "1020": {"Numberofshare": "1,281.25", "Eps": "1.17", "Bookvalue": "11.73", "Parallel_value": "10.00", "PE_ratio": "9.62", "PB_ratio": "0.96", "ROA": "", "ROE": "10.54"},
    "1030": {"Numberofshare": "1,250.00", "Eps": "1.95", "Bookvalue": "13.70", "Parallel_value": "10.00", "PE_ratio": "8.50", "PB_ratio": "0.94", "ROA": "", "ROE": "14.76"},
    "1050": {"Numberofshare": "2,500.00", "Eps": "2.14", "Bookvalue": "17.09", "Parallel_value": "10.00", "PE_ratio": "8.89", "PB_ratio": "1.11", "ROA": "", "ROE": "13.08"},
    "1060": {"Numberofshare": "2,054.80", "Eps": "4.11", "Bookvalue": "32.58", "Parallel_value": "10.00", "PE_ratio": "8.43", "PB_ratio": "1.08", "ROA": "", "ROE": "13.16"},
    "1080": {"Numberofshare": "2,000.00", "Eps": "2.56", "Bookvalue": "20.86", "Parallel_value": "10.00", "PE_ratio": "8.33", "PB_ratio": "1.00", "ROA": "", "ROE": "12.73"},
    "1111": {"Numberofshare": "120.00", "Eps": "3.30", "Bookvalue": "28.69", "Parallel_value": "10.00", "PE_ratio": "42.89", "PB_ratio": "4.94", "ROA": "4.45", "ROE": "11.41"},
    "1120": {"Numberofshare": "4,000.00", "Eps": "6.20", "Bookvalue": "28.71", "Parallel_value": "10.00", "PE_ratio": "16.26", "PB_ratio": "3.51", "ROA": "", "ROE": "23.13"},
    "1140": {"Numberofshare": "1,500.00", "Eps": "2.03", "Bookvalue": "12.61", "Parallel_value": "10.00", "PE_ratio": "12.93", "PB_ratio": "2.08", "ROA": "", "ROE": "17.13"},
    "1150": {"Numberofshare": "2,500.00", "Eps": "2.56", "Bookvalue": "14.30", "Parallel_value": "10.00", "PE_ratio": "11.12", "PB_ratio": "1.99", "ROA": "", "ROE": "18.70"},
    "1180": {"Numberofshare": "6,000.00", "Eps": "4.17", "Bookvalue": "30.94", "Parallel_value": "10.00", "PE_ratio": "9.71", "PB_ratio": "1.31", "ROA": "", "ROE": "14.01"},
    "1182": {"Numberofshare": "101.93", "Eps": "0.64", "Bookvalue": "12.57", "Parallel_value": "10.00", "PE_ratio": "16.27", "PB_ratio": "0.84", "ROA": "", "ROE": "5.21"},
    "1183": {"Numberofshare": "100.00", "Eps": "0.51", "Bookvalue": "17.38", "Parallel_value": "10.00", "PE_ratio": "28.89", "PB_ratio": "0.82", "ROA": "", "ROE": "2.98"},
    "1201": {"Numberofshare": "76.46", "Eps": "(1.33 )", "Bookvalue": "4.46", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.21", "ROA": "(7.92 )", "ROE": "(25.89 )"},
    "1202": {"Numberofshare": "86.67", "Eps": "(0.23 )", "Bookvalue": "18.83", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.90", "ROA": "(0.76 )", "ROE": "(1.20 )"},
    "1210": {"Numberofshare": "27.50", "Eps": "0.53", "Bookvalue": "21.25", "Parallel_value": "10.00", "PE_ratio": "49.92", "PB_ratio": "1.15", "ROA": "1.18", "ROE": "2.49"},
    "1211": {"Numberofshare": "3,888.76", "Eps": "1.91", "Bookvalue": "15.84", "Parallel_value": "10.00", "PE_ratio": "35.90", "PB_ratio": "4.42", "ROA": "6.31", "ROE": "12.95"},
    "1212": {"Numberofshare": "80.00", "Eps": "8.34", "Bookvalue": "37.02", "Parallel_value": "10.00", "PE_ratio": "16.42", "PB_ratio": "3.71", "ROA": "15.19", "ROE": "24.31"},
    "1213": {"Numberofshare": "10.90", "Eps": "(2.49 )", "Bookvalue": "3.60", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "6.23", "ROA": "(9.01 )", "ROE": "(51.47 )"},
    "1214": {"Numberofshare": "67.71", "Eps": "1.23", "Bookvalue": "12.79", "Parallel_value": "10.00", "PE_ratio": "12.84", "PB_ratio": "1.23", "ROA": "4.80", "ROE": "9.91"},
    "1301": {"Numberofshare": "28.08", "Eps": "0.21", "Bookvalue": "13.23", "Parallel_value": "10.00", "PE_ratio": "79.59", "PB_ratio": "1.27", "ROA": "1.19", "ROE": "1.56"},
    "1302": {"Numberofshare": "60.00", "Eps": "2.83", "Bookvalue": "16.02", "Parallel_value": "10.00", "PE_ratio": "15.42", "PB_ratio": "2.71", "ROA": "5.95", "ROE": "18.24"},
    "1303": {"Numberofshare": "1,125.00", "Eps": "", "Bookvalue": "1.15", "Parallel_value": "", "PE_ratio": "29.24", "PB_ratio": "14.08", "ROA": "27.71", "ROE": "56.14"},
    "1304": {"Numberofshare": "50.80", "Eps": "1.82", "Bookvalue": "12.98", "Parallel_value": "10.00", "PE_ratio": "21.43", "PB_ratio": "2.77", "ROA": "4.81", "ROE": "14.80"},
    "1320": {"Numberofshare": "51.00", "Eps": "3.76", "Bookvalue": "16.72", "Parallel_value": "10.00", "PE_ratio": "13.99", "PB_ratio": "2.26", "ROA": "10.23", "ROE": "22.33"},
    "1321": {"Numberofshare": "31.50", "Eps": "15.70", "Bookvalue": "44.45", "Parallel_value": "10.00", "PE_ratio": "8.70", "PB_ratio": "3.07", "ROA": "28.67", "ROE": "40.37"},
    "1322": {"Numberofshare": "90.00", "Eps": "3.12", "Bookvalue": "14.73", "Parallel_value": "10.00", "PE_ratio": "30.74", "PB_ratio": "6.18", "ROA": "18.42", "ROE": "21.77"},
    "1323": {"Numberofshare": "40.00", "Eps": "1.98", "Bookvalue": "14.71", "Parallel_value": "10.00", "PE_ratio": "14.24", "PB_ratio": "1.61", "ROA": "7.82", "ROE": "13.96"},
    "1810": {"Numberofshare": "300.00", "Eps": "(0.85 )", "Bookvalue": "19.36", "Parallel_value": "10.00", "PE_ratio": "أكبر من 100", "PB_ratio": "1.11", "ROA": "(2.26 )", "ROE": "(4.09 )"},
    "1820": {"Numberofshare": "315.00", "Eps": "(0.18 )", "Bookvalue": "0.52", "Parallel_value": "1.00", "PE_ratio": "سالب", "PB_ratio": "3.27", "ROA": "(2.88 )", "ROE": "(28.71 )"},
    "1830": {"Numberofshare": "52.38", "Eps": "5.82", "Bookvalue": "22.91", "Parallel_value": "10.00", "PE_ratio": "14.37", "PB_ratio": "3.57", "ROA": "7.89", "ROE": "25.06"},
    "1831": {"Numberofshare": "475.00", "Eps": "0.28", "Bookvalue": "1.55", "Parallel_value": "1.00", "PE_ratio": "21.33", "PB_ratio": "3.78", "ROA": "6.82", "ROE": "19.90"},
    "1832": {"Numberofshare": "175.00", "Eps": "(0.01 )", "Bookvalue": "0.93", "Parallel_value": "1.00", "PE_ratio": "سالب", "PB_ratio": "2.72", "ROA": "(0.68 )", "ROE": "(1.12 )"},
    "1833": {"Numberofshare": "20.00", "Eps": "6.40", "Bookvalue": "23.15", "Parallel_value": "10.00", "PE_ratio": "11.67", "PB_ratio": "3.37", "ROA": "14.05", "ROE": "30.45"},
    "1834": {"Numberofshare": "400.00", "Eps": "0.33", "Bookvalue": "1.51", "Parallel_value": "1.00", "PE_ratio": "14.93", "PB_ratio": "3.47", "ROA": "11.86", "ROE": "22.65"},
    "1835": {"Numberofshare": "26.50", "Eps": "3.59", "Bookvalue": "13.27", "Parallel_value": "10.00", "PE_ratio": "13.35", "PB_ratio": "3.52", "ROA": "16.54", "ROE": "27.12"},
    "2001": {"Numberofshare": "67.45", "Eps": "(10.46 )", "Bookvalue": "3.83", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.99", "ROA": "(41.74 )", "ROE": "(115.44 )"},
    "2010": {"Numberofshare": "3,000.00", "Eps": "(8.59 )", "Bookvalue": "42.91", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.34", "ROA": "(9.51 )", "ROE": "(18.09 )"},
    "2020": {"Numberofshare": "476.04", "Eps": "9.08", "Bookvalue": "44.54", "Parallel_value": "10.00", "PE_ratio": "14.55", "PB_ratio": "2.97", "ROA": "16.46", "ROE": "21.79"},
    "2030": {"Numberofshare": "15.00", "Eps": "(3.90 )", "Bookvalue": "21.43", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "2.33", "ROA": "(16.03 )", "ROE": "(16.70 )"},
    "2040": {"Numberofshare": "100.00", "Eps": "1.81", "Bookvalue": "15.54", "Parallel_value": "10.00", "PE_ratio": "48.79", "PB_ratio": "1.74", "ROA": "6.58", "ROE": "12.14"},
    "2050": {"Numberofshare": "300.00", "Eps": "2.92", "Bookvalue": "18.39", "Parallel_value": "10.00", "PE_ratio": "13.99", "PB_ratio": "1.33", "ROA": "4.29", "ROE": "17.25"},
    "2060": {"Numberofshare": "668.91", "Eps": "0.46", "Bookvalue": "14.91", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.59", "ROA": "1.22", "ROE": "3.09"},
    "2070": {"Numberofshare": "120.00", "Eps": "0.91", "Bookvalue": "12.89", "Parallel_value": "10.00", "PE_ratio": "42.28", "PB_ratio": "2.12", "ROA": "2.48", "ROE": "7.40"},
    "2080": {"Numberofshare": "75.00", "Eps": "3.32", "Bookvalue": "26.24", "Parallel_value": "10.00", "PE_ratio": "23.65", "PB_ratio": "2.92", "ROA": "9.08", "ROE": "12.86"},
    "2081": {"Numberofshare": "35.00", "Eps": "7.41", "Bookvalue": "25.42", "Parallel_value": "10.00", "PE_ratio": "14.67", "PB_ratio": "4.40", "ROA": "9.86", "ROE": "32.67"},
    "2082": {"Numberofshare": "766.49", "Eps": "2.42", "Bookvalue": "37.87", "Parallel_value": "10.00", "PE_ratio": "81.13", "PB_ratio": "4.43", "ROA": "2.92", "ROE": "7.28"},
    "2083": {"Numberofshare": "250.00", "Eps": "1.80", "Bookvalue": "22.44", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.40", "ROA": "1.99", "ROE": "8.30"},
    "2084": {"Numberofshare": "160.93", "Eps": "0.43", "Bookvalue": "2.92", "Parallel_value": "1.00", "PE_ratio": "35.03", "PB_ratio": "5.21", "ROA": "5.20", "ROE": "15.67"},
    "2090": {"Numberofshare": "31.67", "Eps": "(0.06 )", "Bookvalue": "10.92", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.30", "ROA": "(0.49 )", "ROE": "(0.56 )"},
    "2100": {"Numberofshare": "23.15", "Eps": "(1.45 )", "Bookvalue": "9.46", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "2.04", "ROA": "(10.12 )", "ROE": "(14.30 )"},
    "2110": {"Numberofshare": "6.67", "Eps": "16.10", "Bookvalue": "(47.06 )", "Parallel_value": "10.00", "PE_ratio": "11.41", "PB_ratio": "(3.38)", "ROA": "14.48", "ROE": "(29.71 )"},
    "2120": {"Numberofshare": "60.00", "Eps": "(1.66 )", "Bookvalue": "17.66", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.89", "ROA": "(7.68 )", "ROE": "(8.72 )"},
    "2130": {"Numberofshare": "30.00", "Eps": "(1.08 )", "Bookvalue": "2.24", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "6.03", "ROA": "(12.81 )", "ROE": "(37.91 )"},
    "2140": {"Numberofshare": "100.64", "Eps": "3.57", "Bookvalue": "9.46", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.16", "ROA": "25.19", "ROE": "46.14"},
    "2150": {"Numberofshare": "32.90", "Eps": "2.49", "Bookvalue": "24.24", "Parallel_value": "10.00", "PE_ratio": "14.21", "PB_ratio": "1.46", "ROA": "9.52", "ROE": "10.50"},
    "2160": {"Numberofshare": "44.55", "Eps": "(1.51 )", "Bookvalue": "20.21", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.65", "ROA": "(3.18 )", "ROE": "(7.31 )"},
    "2170": {"Numberofshare": "69.20", "Eps": "(1.28 )", "Bookvalue": "48.40", "Parallel_value": "10.00", "PE_ratio": "61.11", "PB_ratio": "0.53", "ROA": "(1.59 )", "ROE": "(2.54 )"},
    "2180": {"Numberofshare": "11.50", "Eps": "(1.78 )", "Bookvalue": "11.09", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "2.41", "ROA": "(7.27 )", "ROE": "(14.88 )"},
    "2190": {"Numberofshare": "81.60", "Eps": "1.18", "Bookvalue": "18.31", "Parallel_value": "10.00", "PE_ratio": "30.07", "PB_ratio": "1.70", "ROA": "1.56", "ROE": "6.49"},
    "2200": {"Numberofshare": "200.00", "Eps": "0.55", "Bookvalue": "2.44", "Parallel_value": "1.00", "PE_ratio": "9.66", "PB_ratio": "1.99", "ROA": "13.33", "ROE": "24.84"},
    "2210": {"Numberofshare": "23.52", "Eps": "3.31", "Bookvalue": "11.94", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.74", "ROA": "6.98", "ROE": "32.37"},
    "2220": {"Numberofshare": "35.40", "Eps": "(0.75 )", "Bookvalue": "6.53", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.80", "ROA": "(7.04 )", "ROE": "(10.84 )"},
    "2222": {"Numberofshare": "242,000.00", "Eps": "1.44", "Bookvalue": "6.17", "Parallel_value": "-", "PE_ratio": "16.96", "PB_ratio": "4.32", "ROA": "13.99", "ROE": "23.59"},
    "2223": {"Numberofshare": "168.75", "Eps": "5.07", "Bookvalue": "27.16", "Parallel_value": "10.00", "PE_ratio": "18.27", "PB_ratio": "3.41", "ROA": "11.15", "ROE": "19.05"},
    "2230": {"Numberofshare": "843.20", "Eps": "0.40", "Bookvalue": "2.83", "Parallel_value": "1.00", "PE_ratio": "18.03", "PB_ratio": "2.55", "ROA": "5.32", "ROE": "15.01"},
    "2240": {"Numberofshare": "60.00", "Eps": "1.42", "Bookvalue": "8.81", "Parallel_value": "10.00", "PE_ratio": "18.84", "PB_ratio": "4.05", "ROA": "1.39", "ROE": "17.50"},
    "2250": {"Numberofshare": "679.32", "Eps": "(0.15 )", "Bookvalue": "12.66", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.97", "ROA": "(1.10 )", "ROE": "(1.12 )"},
    "2270": {"Numberofshare": "32.50", "Eps": "14.69", "Bookvalue": "53.62", "Parallel_value": "10.00", "PE_ratio": "18.03", "PB_ratio": "3.99", "ROA": "17.87", "ROE": "26.90"},
    "2280": {"Numberofshare": "1,000.00", "Eps": "2.46", "Bookvalue": "20.53", "Parallel_value": "10.00", "PE_ratio": "18.21", "PB_ratio": "2.12", "ROA": "6.50", "ROE": "12.49"},
    "2281": {"Numberofshare": "20.00", "Eps": "(0.94 )", "Bookvalue": "30.77", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.69", "ROA": "(0.63 )", "ROE": "(2.90 )"},
    "2282": {"Numberofshare": "20.00", "Eps": "", "Bookvalue": "14.28", "Parallel_value": "10.00", "PE_ratio": "78.88", "PB_ratio": "3.66", "ROA": "4.18", "ROE": "4.75"},
    "2283": {"Numberofshare": "55.50", "Eps": "5.00", "Bookvalue": "19.26", "Parallel_value": "10.00", "PE_ratio": "10.06", "PB_ratio": "2.61", "ROA": "11.02", "ROE": "27.55"},
    "2284": {"Numberofshare": "81.83", "Eps": "2.73", "Bookvalue": "3.95", "Parallel_value": "1.00", "PE_ratio": "10.44", "PB_ratio": "7.23", "ROA": "17.36", "ROE": "76.44"},
    "2285": {"Numberofshare": "51.32", "Eps": "4.62", "Bookvalue": "23.10", "Parallel_value": "10.00", "PE_ratio": "8.41", "PB_ratio": "1.68", "ROA": "10.21", "ROE": "21.93"},
    "2286": {"Numberofshare": "540.00", "Eps": "0.37", "Bookvalue": "1.47", "Parallel_value": "1.00", "PE_ratio": "10.38", "PB_ratio": "2.60", "ROA": "16.87", "ROE": "26.63"},
    "2287": {"Numberofshare": "30.00", "Eps": "1.12", "Bookvalue": "15.51", "Parallel_value": "10.00", "PE_ratio": "28.28", "PB_ratio": "1.68", "ROA": "2.17", "ROE": "7.19"},
    "2288": {"Numberofshare": "96.00", "Eps": "0.59", "Bookvalue": "1.93", "Parallel_value": "1.00", "PE_ratio": "13.73", "PB_ratio": "4.42", "ROA": "22.52", "ROE": "34.74"},
    "2290": {"Numberofshare": "562.50", "Eps": "0.14", "Bookvalue": "19.11", "Parallel_value": "10.00", "PE_ratio": "أكبر من 100", "PB_ratio": "1.65", "ROA": "0.58", "ROE": "0.72"},
    "2300": {"Numberofshare": "37.07", "Eps": "1.33", "Bookvalue": "15.17", "Parallel_value": "10.00", "PE_ratio": "53.10", "PB_ratio": "3.42", "ROA": "3.88", "ROE": "9.21"},
    "2310": {"Numberofshare": "733.33", "Eps": "(0.58 )", "Bookvalue": "19.66", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.76", "ROA": "(1.99 )", "ROE": "(2.82 )"},
    "2320": {"Numberofshare": "63.95", "Eps": "7.08", "Bookvalue": "20.49", "Parallel_value": "10.00", "PE_ratio": "9.33", "PB_ratio": "3.23", "ROA": "15.48", "ROE": "38.04"},
    "2330": {"Numberofshare": "260.00", "Eps": "0.87", "Bookvalue": "11.89", "Parallel_value": "10.00", "PE_ratio": "26.30", "PB_ratio": "1.92", "ROA": "1.68", "ROE": "7.59"},
    "2340": {"Numberofshare": "81.25", "Eps": "(1.52 )", "Bookvalue": "12.89", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.76", "ROA": "(8.17 )", "ROE": "(11.01 )"},
    "2350": {"Numberofshare": "1,500.00", "Eps": "(1.53 )", "Bookvalue": "6.12", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.83", "ROA": "(9.64 )", "ROE": "(22.18 )"},
    "2360": {"Numberofshare": "15.00", "Eps": "(2.68 )", "Bookvalue": "6.17", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "2.98", "ROA": "(21.56 )", "ROE": "(35.10 )"},
    "2370": {"Numberofshare": "40.00", "Eps": "2.29", "Bookvalue": "12.97", "Parallel_value": "10.00", "PE_ratio": "9.02", "PB_ratio": "1.53", "ROA": "9.40", "ROE": "19.01"},
    "2380": {"Numberofshare": "2,197.36", "Eps": "(2.16 )", "Bookvalue": "5.93", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.51", "ROA": "(6.49 )", "ROE": "(34.15 )"},
    "2381": {"Numberofshare": "89.00", "Eps": "(0.85 )", "Bookvalue": "64.56", "Parallel_value": "10.00", "PE_ratio": "أكبر من 100", "PB_ratio": "1.36", "ROA": "(0.73 )", "ROE": "(1.29 )"},
    "2382": {"Numberofshare": "1,129.06", "Eps": "0.71", "Bookvalue": "5.84", "Parallel_value": "1.00", "PE_ratio": "25.80", "PB_ratio": "2.93", "ROA": "3.68", "ROE": "12.59"},
    "3002": {"Numberofshare": "170.00", "Eps": "0.33", "Bookvalue": "11.90", "Parallel_value": "10.00", "PE_ratio": "19.09", "PB_ratio": "0.53", "ROA": "2.27", "ROE": "2.79"},
    "3003": {"Numberofshare": "140.00", "Eps": "0.98", "Bookvalue": "12.86", "Parallel_value": "10.00", "PE_ratio": "11.72", "PB_ratio": "0.90", "ROA": "7.36", "ROE": "7.75"},
    "3004": {"Numberofshare": "180.00", "Eps": "0.31", "Bookvalue": "12.47", "Parallel_value": "10.00", "PE_ratio": "22.10", "PB_ratio": "0.55", "ROA": "1.59", "ROE": "2.49"},
    "3005": {"Numberofshare": "55.00", "Eps": "0.73", "Bookvalue": "15.38", "Parallel_value": "10.00", "PE_ratio": "15.99", "PB_ratio": "0.77", "ROA": "3.42", "ROE": "4.88"},
    "3007": {"Numberofshare": "225.00", "Eps": "0.01", "Bookvalue": "1.29", "Parallel_value": "1.00", "PE_ratio": "أكبر من 100", "PB_ratio": "1.83", "ROA": "0.58", "ROE": "1.09"},
    "3008": {"Numberofshare": "226.04", "Eps": "", "Bookvalue": "0.46", "Parallel_value": "", "PE_ratio": "سالب", "PB_ratio": "4.41", "ROA": "(9.06 )", "ROE": "(24.53 )"},
    "3010": {"Numberofshare": "100.00", "Eps": "1.65", "Bookvalue": "25.78", "Parallel_value": "10.00", "PE_ratio": "13.76", "PB_ratio": "0.88", "ROA": "5.37", "ROE": "6.42"},
    "3020": {"Numberofshare": "202.50", "Eps": "2.38", "Bookvalue": "24.54", "Parallel_value": "10.00", "PE_ratio": "13.46", "PB_ratio": "0.95", "ROA": "6.53", "ROE": "9.84"},
    "3030": {"Numberofshare": "153.00", "Eps": "2.38", "Bookvalue": "14.51", "Parallel_value": "10.00", "PE_ratio": "13.78", "PB_ratio": "2.25", "ROA": "11.57", "ROE": "16.31"},
    "3040": {"Numberofshare": "110.56", "Eps": "2.35", "Bookvalue": "24.04", "Parallel_value": "10.00", "PE_ratio": "24.79", "PB_ratio": "1.77", "ROA": "8.34", "ROE": "9.61"},
    "3050": {"Numberofshare": "140.00", "Eps": "1.02", "Bookvalue": "23.51", "Parallel_value": "10.00", "PE_ratio": "21.52", "PB_ratio": "0.94", "ROA": "3.15", "ROE": "4.37"},
    "3060": {"Numberofshare": "157.50", "Eps": "0.66", "Bookvalue": "15.95", "Parallel_value": "10.00", "PE_ratio": "21.71", "PB_ratio": "0.90", "ROA": "3.28", "ROE": "4.08"},
    "3080": {"Numberofshare": "86.00", "Eps": "2.78", "Bookvalue": "26.57", "Parallel_value": "10.00", "PE_ratio": "7.41", "PB_ratio": "0.87", "ROA": "8.14", "ROE": "10.17"},
    "3090": {"Numberofshare": "90.00", "Eps": "0.49", "Bookvalue": "13.04", "Parallel_value": "10.00", "PE_ratio": "16.59", "PB_ratio": "0.63", "ROA": "2.88", "ROE": "3.55"},
    "3091": {"Numberofshare": "108.70", "Eps": "(0.63 )", "Bookvalue": "9.68", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.54", "ROA": "(3.32 )", "ROE": "(6.31 )"},
    "3092": {"Numberofshare": "120.00", "Eps": "1.73", "Bookvalue": "14.32", "Parallel_value": "10.00", "PE_ratio": "13.42", "PB_ratio": "1.62", "ROA": "10.75", "ROE": "11.88"},
    "4001": {"Numberofshare": "900.00", "Eps": "0.47", "Bookvalue": "1.34", "Parallel_value": "1.00", "PE_ratio": "21.56", "PB_ratio": "4.61", "ROA": "6.15", "ROE": "36.69"},
    "4002": {"Numberofshare": "200.00", "Eps": "3.78", "Bookvalue": "18.76", "Parallel_value": "10.00", "PE_ratio": "16.92", "PB_ratio": "3.41", "ROA": "13.39", "ROE": "21.12"},
    "4003": {"Numberofshare": "80.00", "Eps": "6.21", "Bookvalue": "20.45", "Parallel_value": "10.00", "PE_ratio": "12.88", "PB_ratio": "3.91", "ROA": "8.78", "ROE": "29.71"},
    "4004": {"Numberofshare": "101.58", "Eps": "5.30", "Bookvalue": "40.19", "Parallel_value": "10.00", "PE_ratio": "24.42", "PB_ratio": "2.83", "ROA": "6.84", "ROE": "14.26"},
    "4005": {"Numberofshare": "44.85", "Eps": "7.10", "Bookvalue": "41.23", "Parallel_value": "10.00", "PE_ratio": "18.42", "PB_ratio": "3.18", "ROA": "12.09", "ROE": "18.33"},
    "4006": {"Numberofshare": "45.00", "Eps": "0.56", "Bookvalue": "15.67", "Parallel_value": "10.00", "PE_ratio": "20.63", "PB_ratio": "0.81", "ROA": "1.05", "ROE": "3.61"},
    "4007": {"Numberofshare": "160.00", "Eps": "1.66", "Bookvalue": "12.40", "Parallel_value": "10.00", "PE_ratio": "15.21", "PB_ratio": "2.04", "ROA": "10.00", "ROE": "13.54"},
    "4008": {"Numberofshare": "36.00", "Eps": "1.27", "Bookvalue": "10.30", "Parallel_value": "10.00", "PE_ratio": "43.85", "PB_ratio": "2.36", "ROA": "3.87", "ROE": "13.08"},
    "4009": {"Numberofshare": "92.04", "Eps": "3.28", "Bookvalue": "20.34", "Parallel_value": "10.00", "PE_ratio": "15.20", "PB_ratio": "1.53", "ROA": "5.70", "ROE": "17.26"},
    "4011": {"Numberofshare": "57.50", "Eps": "(0.52 )", "Bookvalue": "4.68", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "2.43", "ROA": "(1.42 )", "ROE": "(10.47 )"},
    "4012": {"Numberofshare": "400.00", "Eps": "0.25", "Bookvalue": "1.53", "Parallel_value": "1.00", "PE_ratio": "14.10", "PB_ratio": "2.32", "ROA": "13.37", "ROE": "16.64"},
    "4013": {"Numberofshare": "350.00", "Eps": "6.86", "Bookvalue": "22.58", "Parallel_value": "10.00", "PE_ratio": "35.93", "PB_ratio": "10.93", "ROA": "10.97", "ROE": "31.86"},
    "4014": {"Numberofshare": "30.00", "Eps": "0.95", "Bookvalue": "16.96", "Parallel_value": "10.00", "PE_ratio": "27.06", "PB_ratio": "1.75", "ROA": "3.00", "ROE": "5.57"},
    "4015": {"Numberofshare": "70.00", "Eps": "6.63", "Bookvalue": "24.53", "Parallel_value": "10.00", "PE_ratio": "20.57", "PB_ratio": "5.56", "ROA": "24.30", "ROE": "28.92"},
    "4016": {"Numberofshare": "20.00", "Eps": "4.85", "Bookvalue": "20.81", "Parallel_value": "10.00", "PE_ratio": "21.05", "PB_ratio": "4.90", "ROA": "17.23", "ROE": "24.73"},
    "4017": {"Numberofshare": "232.00", "Eps": "1.25", "Bookvalue": "13.67", "Parallel_value": "1.00", "PE_ratio": "25.17", "PB_ratio": "2.30", "ROA": "5.20", "ROE": "9.49"},
    "4018": {"Numberofshare": "44.30", "Eps": "4.71", "Bookvalue": "43.90", "Parallel_value": "10.00", "PE_ratio": "31.67", "PB_ratio": "3.36", "ROA": "7.43", "ROE": "16.00"},
    "4019": {"Numberofshare": "250.00", "Eps": "1.06", "Bookvalue": "4.75", "Parallel_value": "1.00", "PE_ratio": "22.56", "PB_ratio": "3.91", "ROA": "12.18", "ROE": "26.42"},
    "4020": {"Numberofshare": "375.00", "Eps": "1.22", "Bookvalue": "14.09", "Parallel_value": "10.00", "PE_ratio": "11.16", "PB_ratio": "0.92", "ROA": "4.89", "ROE": "9.06"},
    "4021": {"Numberofshare": "77.00", "Eps": "0.14", "Bookvalue": "1.25", "Parallel_value": "1.00", "PE_ratio": "22.42", "PB_ratio": "4.57", "ROA": "8.30", "ROE": "11.05"},
    "4030": {"Numberofshare": "922.85", "Eps": "2.63", "Bookvalue": "16.55", "Parallel_value": "10.00", "PE_ratio": "12.19", "PB_ratio": "1.95", "ROA": "8.44", "ROE": "16.83"},
    "4031": {"Numberofshare": "188.00", "Eps": "2.10", "Bookvalue": "13.04", "Parallel_value": "10.00", "PE_ratio": "14.72", "PB_ratio": "2.37", "ROA": "9.08", "ROE": "16.33"},
    "4040": {"Numberofshare": "125.00", "Eps": "(0.13 )", "Bookvalue": "7.48", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.13", "ROA": "(0.41 )", "ROE": "(1.82 )"},
    "4050": {"Numberofshare": "70.00", "Eps": "0.75", "Bookvalue": "12.77", "Parallel_value": "10.00", "PE_ratio": "أكبر من 100", "PB_ratio": "4.13", "ROA": "0.79", "ROE": "6.17"},
    "4051": {"Numberofshare": "101.25", "Eps": "0.25", "Bookvalue": "1.98", "Parallel_value": "1.00", "PE_ratio": "22.42", "PB_ratio": "2.79", "ROA": "9.89", "ROE": "12.67"},
    "4061": {"Numberofshare": "31.50", "Eps": "(0.35 )", "Bookvalue": "8.01", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.29", "ROA": "(1.93 )", "ROE": "(4.28 )"},
    "4070": {"Numberofshare": "22.92", "Eps": "(2.29 )", "Bookvalue": "5.81", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "2.52", "ROA": "(16.09 )", "ROE": "(31.59 )"},
    "4071": {"Numberofshare": "55.00", "Eps": "(4.51 )", "Bookvalue": "19.04", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "5.77", "ROA": "(2.63 )", "ROE": "(21.17 )"},
    "4072": {"Numberofshare": "332.50", "Eps": "1.75", "Bookvalue": "13.75", "Parallel_value": "10.00", "PE_ratio": "29.85", "PB_ratio": "1.97", "ROA": "6.64", "ROE": "13.58"},
    "4080": {"Numberofshare": "126.39", "Eps": "0.01", "Bookvalue": "9.36", "Parallel_value": "10.00", "PE_ratio": "أكبر من 100", "PB_ratio": "0.85", "ROA": "0.07", "ROE": "0.16"},
    "4081": {"Numberofshare": "120.00", "Eps": "(1.05 )", "Bookvalue": "10.33", "Parallel_value": "10.00", "PE_ratio": "(10.05 )", "PB_ratio": "1.10", "ROA": "", "ROE": "(9.40 )"},
    "4082": {"Numberofshare": "71.43", "Eps": "0.16", "Bookvalue": "11.82", "Parallel_value": "10.00", "PE_ratio": "52.32", "PB_ratio": "0.81", "ROA": "", "ROE": "1.39"},
    "4083": {"Numberofshare": "25.00", "Eps": "10.95", "Bookvalue": "55.88", "Parallel_value": "10.00", "PE_ratio": "11.69", "PB_ratio": "2.61", "ROA": "", "ROE": "21.81"},
    "4084": {"Numberofshare": "249.74", "Eps": "1.68", "Bookvalue": "4.61", "Parallel_value": "2.00", "PE_ratio": "13.70", "PB_ratio": "4.95", "ROA": "31.93", "ROE": "42.08"},
    "4090": {"Numberofshare": "260.46", "Eps": "1.40", "Bookvalue": "26.30", "Parallel_value": "10.00", "PE_ratio": "29.46", "PB_ratio": "1.36", "ROA": "3.65", "ROE": "5.33"},
    "4100": {"Numberofshare": "200.00", "Eps": "2.33", "Bookvalue": "20.49", "Parallel_value": "10.00", "PE_ratio": "35.68", "PB_ratio": "4.03", "ROA": "9.70", "ROE": "10.94"},
    "4110": {"Numberofshare": "600.00", "Eps": "(0.06 )", "Bookvalue": "0.70", "Parallel_value": "1.00", "PE_ratio": "أكبر من 100", "PB_ratio": "2.78", "ROA": "(2.78 )", "ROE": "(7.58 )"},
    "4130": {"Numberofshare": "218.30", "Eps": "0.03", "Bookvalue": "1.05", "Parallel_value": "1.00", "PE_ratio": "64.09", "PB_ratio": "1.89", "ROA": "2.62", "ROE": "2.99"},
    "4140": {"Numberofshare": "194.40", "Eps": "(0.09 )", "Bookvalue": "0.59", "Parallel_value": "1.00", "PE_ratio": "سالب", "PB_ratio": "3.84", "ROA": "(12.64 )", "ROE": "(13.51 )"},
    "4141": {"Numberofshare": "12.00", "Eps": "(0.26 )", "Bookvalue": "12.86", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.66", "ROA": "(1.42 )", "ROE": "(2.04 )"},
    "4142": {"Numberofshare": "150.00", "Eps": "7.13", "Bookvalue": "19.39", "Parallel_value": "10.00", "PE_ratio": "16.01", "PB_ratio": "5.89", "ROA": "16.04", "ROE": "38.50"},
    "4143": {"Numberofshare": "40.00", "Eps": "2.11", "Bookvalue": "13.19", "Parallel_value": "10.00", "PE_ratio": "15.64", "PB_ratio": "2.51", "ROA": "11.89", "ROE": "16.36"},
    "4144": {"Numberofshare": "12.50", "Eps": "0.84", "Bookvalue": "12.32", "Parallel_value": "10.00", "PE_ratio": "65.63", "PB_ratio": "4.49", "ROA": "4.85", "ROE": "6.67"},
    "4145": {"Numberofshare": "32.00", "Eps": "(0.09 )", "Bookvalue": "16.67", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.40", "ROA": "(0.37 )", "ROE": "(0.50 )"},
    "4146": {"Numberofshare": "158.00", "Eps": "0.95", "Bookvalue": "3.05", "Parallel_value": "1.00", "PE_ratio": "13.85", "PB_ratio": "4.34", "ROA": "15.31", "ROE": "34.00"},
    "4147": {"Numberofshare": "100.00", "Eps": "0.50", "Bookvalue": "1.86", "Parallel_value": "1.00", "PE_ratio": "15.78", "PB_ratio": "4.25", "ROA": "14.59", "ROE": "29.31"},
    "4148": {"Numberofshare": "250.00", "Eps": "0.13", "Bookvalue": "1.50", "Parallel_value": "1.00", "PE_ratio": "20.95", "PB_ratio": "1.85", "ROA": "-", "ROE": "9.09"},
    "4150": {"Numberofshare": "233.93", "Eps": "1.27", "Bookvalue": "17.64", "Parallel_value": "10.00", "PE_ratio": "15.24", "PB_ratio": "1.00", "ROA": "8.06", "ROE": "8.88"},
    "4160": {"Numberofshare": "6.50", "Eps": "(0.57 )", "Bookvalue": "2.75", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "13.66", "ROA": "(4.46 )", "ROE": "(19.73 )"},
    "4161": {"Numberofshare": "1,143.00", "Eps": "0.24", "Bookvalue": "1.34", "Parallel_value": "1.00", "PE_ratio": "19.10", "PB_ratio": "3.37", "ROA": "4.92", "ROE": "18.43"},
    "4162": {"Numberofshare": "60.00", "Eps": "2.72", "Bookvalue": "17.54", "Parallel_value": "10.00", "PE_ratio": "17.98", "PB_ratio": "2.78", "ROA": "8.40", "ROE": "15.35"},
    "4163": {"Numberofshare": "85.00", "Eps": "4.29", "Bookvalue": "18.25", "Parallel_value": "10.00", "PE_ratio": "11.46", "PB_ratio": "2.68", "ROA": "7.16", "ROE": "24.58"},
    "4164": {"Numberofshare": "130.00", "Eps": "6.39", "Bookvalue": "21.10", "Parallel_value": "10.00", "PE_ratio": "16.89", "PB_ratio": "5.00", "ROA": "12.91", "ROE": "31.17"},
    "4165": {"Numberofshare": "25.00", "Eps": "8.70", "Bookvalue": "23.37", "Parallel_value": "10.00", "PE_ratio": "18.22", "PB_ratio": "6.81", "ROA": "25.61", "ROE": "41.35"},
    "4170": {"Numberofshare": "57.82", "Eps": "(0.09 )", "Bookvalue": "8.65", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.69", "ROA": "(0.97 )", "ROE": "(1.02 )"},
    "4180": {"Numberofshare": "275.00", "Eps": "0.03", "Bookvalue": "1.73", "Parallel_value": "1.00", "PE_ratio": "87.75", "PB_ratio": "1.47", "ROA": "1.62", "ROE": "1.68"},
    "4190": {"Numberofshare": "1,200.00", "Eps": "0.87", "Bookvalue": "1.47", "Parallel_value": "1.00", "PE_ratio": "15.88", "PB_ratio": "9.38", "ROA": "24.41", "ROE": "59.83"},
    "4191": {"Numberofshare": "20.00", "Eps": "1.13", "Bookvalue": "12.13", "Parallel_value": "10.00", "PE_ratio": "32.04", "PB_ratio": "2.98", "ROA": "6.51", "ROE": "9.34"},
    "4192": {"Numberofshare": "350.00", "Eps": "0.17", "Bookvalue": "1.42", "Parallel_value": "1.00", "PE_ratio": "38.24", "PB_ratio": "4.48", "ROA": "7.58", "ROE": "12.40"},
    "4193": {"Numberofshare": "115.50", "Eps": "0.30", "Bookvalue": "3.51", "Parallel_value": "1.00", "PE_ratio": "44.23", "PB_ratio": "3.77", "ROA": "6.09", "ROE": "11.85"},
    "4194": {"Numberofshare": "16.00", "Eps": "3.84", "Bookvalue": "22.42", "Parallel_value": "10.00", "PE_ratio": "12.83", "PB_ratio": "2.19", "ROA": "12.07", "ROE": "18.77"},
    "4200": {"Numberofshare": "100.00", "Eps": "4.22", "Bookvalue": "16.88", "Parallel_value": "10.00", "PE_ratio": "27.92", "PB_ratio": "6.98", "ROA": "4.73", "ROE": "27.17"},
    "4210": {"Numberofshare": "80.00", "Eps": "(1.55 )", "Bookvalue": "40.62", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "2.15", "ROA": "(2.10 )", "ROE": "(3.77 )"},
    "4220": {"Numberofshare": "882.93", "Eps": "(0.46 )", "Bookvalue": "9.46", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.56", "ROA": "(1.87 )", "ROE": "(5.56 )"},
    "4230": {"Numberofshare": "48.27", "Eps": "(0.89 )", "Bookvalue": "(0.22 )", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "(168.15)", "ROA": "(0.99 )", "ROE": "(633.27 )"},
    "4240": {"Numberofshare": "114.80", "Eps": "(3.15 )", "Bookvalue": "(10.34 )", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "(1.46)", "ROA": "(8.00 )", "ROE": "36.19"},
    "4250": {"Numberofshare": "1,180.02", "Eps": "2.03", "Bookvalue": "13.44", "Parallel_value": "10.00", "PE_ratio": "37.43", "PB_ratio": "1.13", "ROA": "8.79", "ROE": "16.32"},
    "4260": {"Numberofshare": "104.54", "Eps": "3.36", "Bookvalue": "27.14", "Parallel_value": "10.00", "PE_ratio": "11.99", "PB_ratio": "1.48", "ROA": "6.92", "ROE": "12.94"},
    "4261": {"Numberofshare": "65.97", "Eps": "3.02", "Bookvalue": "13.99", "Parallel_value": "10.00", "PE_ratio": "10.34", "PB_ratio": "2.23", "ROA": "7.06", "ROE": "22.83"},
    "4262": {"Numberofshare": "55.00", "Eps": "3.60", "Bookvalue": "25.60", "Parallel_value": "10.00", "PE_ratio": "10.39", "PB_ratio": "1.45", "ROA": "5.84", "ROE": "15.12"},
    "4263": {"Numberofshare": "80.00", "Eps": "8.72", "Bookvalue": "20.28", "Parallel_value": "10.00", "PE_ratio": "17.96", "PB_ratio": "7.70", "ROA": "20.05", "ROE": "46.15"},
    "4264": {"Numberofshare": "170.85", "Eps": "(3.83 )", "Bookvalue": "20.47", "Parallel_value": "10.00", "PE_ratio": "26.74", "PB_ratio": "2.54", "ROA": "(4.23 )", "ROE": "(25.06 )"},
    "4265": {"Numberofshare": "30.00", "Eps": "2.35", "Bookvalue": "18.93", "Parallel_value": "10.00", "PE_ratio": "10.07", "PB_ratio": "1.25", "ROA": "-", "ROE": "13.23"},
    "4270": {"Numberofshare": "65.20", "Eps": "(3.07 )", "Bookvalue": "3.09", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "2.62", "ROA": "(14.96 )", "ROE": "(66.29 )"},
    "4280": {"Numberofshare": "3,705.88", "Eps": "0.46", "Bookvalue": "12.51", "Parallel_value": "10.00", "PE_ratio": "34.27", "PB_ratio": "0.69", "ROA": "2.87", "ROE": "4.05"},
    "4290": {"Numberofshare": "65.00", "Eps": "0.32", "Bookvalue": "8.57", "Parallel_value": "10.00", "PE_ratio": "أكبر من 100", "PB_ratio": "2.07", "ROA": "0.94", "ROE": "3.80"},
    "4291": {"Numberofshare": "43.00", "Eps": "2.88", "Bookvalue": "20.86", "Parallel_value": "10.00", "PE_ratio": "38.50", "PB_ratio": "5.32", "ROA": "7.91", "ROE": "14.30"},
    "4292": {"Numberofshare": "42.09", "Eps": "2.42", "Bookvalue": "19.57", "Parallel_value": "10.00", "PE_ratio": "23.35", "PB_ratio": "2.81", "ROA": "4.68", "ROE": "12.48"},
    "4300": {"Numberofshare": "1,080.00", "Eps": "1.05", "Bookvalue": "20.58", "Parallel_value": "10.00", "PE_ratio": "17.91", "PB_ratio": "0.91", "ROA": "2.92", "ROE": "5.23"},
    "4310": {"Numberofshare": "339.30", "Eps": "0.00", "Bookvalue": "8.72", "Parallel_value": "10.00", "PE_ratio": "أكبر من 100", "PB_ratio": "1.31", "ROA": "0.00", "ROE": "0.00"},
    "4320": {"Numberofshare": "93.33", "Eps": "(0.28 )", "Bookvalue": "10.28", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.52", "ROA": "(1.18 )", "ROE": "(2.63 )"},
    "4321": {"Numberofshare": "475.00", "Eps": "3.24", "Bookvalue": "32.50", "Parallel_value": "10.00", "PE_ratio": "13.70", "PB_ratio": "0.52", "ROA": "4.88", "ROE": "10.26"},
    "4322": {"Numberofshare": "500.00", "Eps": "0.56", "Bookvalue": "1.82", "Parallel_value": "1.00", "PE_ratio": "34.99", "PB_ratio": "7.31", "ROA": "5.94", "ROE": "33.78"},
    "4323": {"Numberofshare": "50.00", "Eps": "2.71", "Bookvalue": "15.36", "Parallel_value": "10.00", "PE_ratio": "11.45", "PB_ratio": "1.78", "ROA": "10.16", "ROE": "19.30"},
    "4324": {"Numberofshare": "200.00", "Eps": "0.16", "Bookvalue": "2.09", "Parallel_value": "1.00", "PE_ratio": "22.20", "PB_ratio": "1.65", "ROA": "4.25", "ROE": "7.98"},
    "4325": {"Numberofshare": "1,438.65", "Eps": "0.77", "Bookvalue": "10.98", "Parallel_value": "10.00", "PE_ratio": "20.39", "PB_ratio": "1.45", "ROA": "4.31", "ROE": "7.72"},
    "4326": {"Numberofshare": "300.00", "Eps": "0.94", "Bookvalue": "3.76", "Parallel_value": "1.00", "PE_ratio": "9.08", "PB_ratio": "2.26", "ROA": "9.60", "ROE": "29.02"},
    "4327": {"Numberofshare": "42.86", "Eps": "9.88", "Bookvalue": "31.78", "Parallel_value": "10.00", "PE_ratio": "8.07", "PB_ratio": "2.50", "ROA": "-", "ROE": "36.19"},
    "4330": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "9.51", "PE_ratio": "", "PB_ratio": "0.74", "ROA": "", "ROE": ""},
    "4331": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "10.00", "PE_ratio": "", "PB_ratio": "1.42", "ROA": "", "ROE": ""},
    "4332": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "10.00", "PE_ratio": "", "PB_ratio": "0.61", "ROA": "", "ROE": ""},
    "4333": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "10.00", "PE_ratio": "", "PB_ratio": "0.90", "ROA": "", "ROE": ""},
    "4334": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "10.00", "PE_ratio": "", "PB_ratio": "1.06", "ROA": "", "ROE": ""},
    "4335": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "10.00", "PE_ratio": "", "PB_ratio": "0.52", "ROA": "", "ROE": ""},
    "4336": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "10.00", "PE_ratio": "", "PB_ratio": "0.63", "ROA": "", "ROE": ""},
    "4337": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "10.00", "PE_ratio": "", "PB_ratio": "1.22", "ROA": "", "ROE": ""},
    "4338": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "10.00", "PE_ratio": "", "PB_ratio": "0.76", "ROA": "", "ROE": ""},
    "4339": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "10.00", "PE_ratio": "", "PB_ratio": "0.65", "ROA": "", "ROE": ""},
    "4340": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "10.00", "PE_ratio": "", "PB_ratio": "0.98", "ROA": "", "ROE": ""},
    "4342": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "10.00", "PE_ratio": "", "PB_ratio": "1.21", "ROA": "", "ROE": ""},
    "4344": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "10.00", "PE_ratio": "", "PB_ratio": "0.91", "ROA": "", "ROE": ""},
    "4345": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "10.00", "PE_ratio": "", "PB_ratio": "0.67", "ROA": "", "ROE": ""},
    "4346": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "10.00", "PE_ratio": "", "PB_ratio": "0.50", "ROA": "", "ROE": ""},
    "4347": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "10.00", "PE_ratio": "", "PB_ratio": "1.13", "ROA": "", "ROE": ""},
    "4348": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "10.00", "PE_ratio": "", "PB_ratio": "0.77", "ROA": "", "ROE": ""},
    "4349": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "10.00", "PE_ratio": "", "PB_ratio": "0.79", "ROA": "", "ROE": ""},
    "4350": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "10.00", "PE_ratio": "", "PB_ratio": "0.80", "ROA": "", "ROE": ""},
    "4700": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "", "PE_ratio": "", "PB_ratio": "", "ROA": "", "ROE": ""},
    "4701": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "", "PE_ratio": "", "PB_ratio": "", "ROA": "", "ROE": ""},
    "4702": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "", "PE_ratio": "", "PB_ratio": "", "ROA": "", "ROE": ""},
    "4703": {"Numberofshare": "", "Eps": "", "Bookvalue": "", "Parallel_value": "", "PE_ratio": "", "PB_ratio": "", "ROA": "", "ROE": ""},
    "5110": {"Numberofshare": "4,166.59", "Eps": "0.96", "Bookvalue": "20.18", "Parallel_value": "10.00", "PE_ratio": "14.89", "PB_ratio": "0.71", "ROA": "0.68", "ROE": "4.80"},
    "6001": {"Numberofshare": "35.36", "Eps": "1.21", "Bookvalue": "9.50", "Parallel_value": "10.00", "PE_ratio": "29.81", "PB_ratio": "3.20", "ROA": "5.67", "ROE": "13.62"},
    "6002": {"Numberofshare": "64.68", "Eps": "(1.20 )", "Bookvalue": "13.13", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.05", "ROA": "(4.62 )", "ROE": "(8.73 )"},
    "6004": {"Numberofshare": "82.00", "Eps": "3.82", "Bookvalue": "19.21", "Parallel_value": "10.00", "PE_ratio": "18.49", "PB_ratio": "3.69", "ROA": "10.73", "ROE": "20.72"},
    "6010": {"Numberofshare": "301.64", "Eps": "1.30", "Bookvalue": "15.28", "Parallel_value": "10.00", "PE_ratio": "13.08", "PB_ratio": "1.11", "ROA": "6.53", "ROE": "8.83"},
    "6012": {"Numberofshare": "7.31", "Eps": "(11.48 )", "Bookvalue": "7.18", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "2.56", "ROA": "(40.25 )", "ROE": "(88.91 )"},
    "6013": {"Numberofshare": "3.00", "Eps": "0.15", "Bookvalue": "8.01", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "11.65", "ROA": "0.62", "ROE": "1.98"},
    "6014": {"Numberofshare": "25.50", "Eps": "2.03", "Bookvalue": "11.51", "Parallel_value": "10.00", "PE_ratio": "19.66", "PB_ratio": "3.48", "ROA": "7.91", "ROE": "17.59"},
    "6015": {"Numberofshare": "8,423.63", "Eps": "", "Bookvalue": "0.22", "Parallel_value": "", "PE_ratio": "19.89", "PB_ratio": "8.96", "ROA": "13.52", "ROE": "49.61"},
    "6016": {"Numberofshare": "56.00", "Eps": "0.10", "Bookvalue": "1.40", "Parallel_value": "1.00", "PE_ratio": "78.53", "PB_ratio": "5.74", "ROA": "3.48", "ROE": "7.57"},
    "6017": {"Numberofshare": "209.84", "Eps": "", "Bookvalue": "6.52", "Parallel_value": "", "PE_ratio": "14.06", "PB_ratio": "1.85", "ROA": "9.71", "ROE": "14.70"},
    "6018": {"Numberofshare": "114.40", "Eps": "0.36", "Bookvalue": "2.36", "Parallel_value": "1.00", "PE_ratio": "17.59", "PB_ratio": "2.94", "ROA": "4.76", "ROE": "18.92"},
    "6019": {"Numberofshare": "102.40", "Eps": "1.27", "Bookvalue": "12.58", "Parallel_value": "10.00", "PE_ratio": "14.55", "PB_ratio": "1.75", "ROA": "7.53", "ROE": "10.66"},
    "6020": {"Numberofshare": "30.00", "Eps": "0.47", "Bookvalue": "10.19", "Parallel_value": "10.00", "PE_ratio": "أكبر من 100", "PB_ratio": "1.34", "ROA": "3.36", "ROE": "4.70"},
    "6040": {"Numberofshare": "39.17", "Eps": "(3.82 )", "Bookvalue": "3.34", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.97", "ROA": "(34.13 )", "ROE": "(72.50 )"},
    "6050": {"Numberofshare": "6.70", "Eps": "(3.78 )", "Bookvalue": "7.51", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "5.97", "ROA": "(15.74 )", "ROE": "(40.29 )"},
    "6060": {"Numberofshare": "30.00", "Eps": "(0.63 )", "Bookvalue": "11.75", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.06", "ROA": "(4.42 )", "ROE": "(5.17 )"},
    "6070": {"Numberofshare": "30.00", "Eps": "2.71", "Bookvalue": "26.63", "Parallel_value": "10.00", "PE_ratio": "18.81", "PB_ratio": "1.91", "ROA": "6.61", "ROE": "10.46"},
    "6090": {"Numberofshare": "50.00", "Eps": "(0.04 )", "Bookvalue": "6.44", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.26", "ROA": "(0.35 )", "ROE": "(0.59 )"},
    "7010": {"Numberofshare": "5,000.00", "Eps": "2.97", "Bookvalue": "16.68", "Parallel_value": "10.00", "PE_ratio": "14.80", "PB_ratio": "2.54", "ROA": "9.32", "ROE": "17.16"},
    "7020": {"Numberofshare": "770.00", "Eps": "4.50", "Bookvalue": "26.40", "Parallel_value": "10.00", "PE_ratio": "14.76", "PB_ratio": "2.48", "ROA": "8.55", "ROE": "17.68"},
    "7030": {"Numberofshare": "898.73", "Eps": "0.67", "Bookvalue": "12.10", "Parallel_value": "10.00", "PE_ratio": "16.93", "PB_ratio": "0.94", "ROA": "2.12", "ROE": "5.59"},
    "7040": {"Numberofshare": "33.99", "Eps": "7.52", "Bookvalue": "28.01", "Parallel_value": "10.00", "PE_ratio": "11.49", "PB_ratio": "3.02", "ROA": "15.08", "ROE": "30.79"},
    "7200": {"Numberofshare": "30.00", "Eps": "3.14", "Bookvalue": "12.99", "Parallel_value": "10.00", "PE_ratio": "94.08", "PB_ratio": "12.28", "ROA": "3.34", "ROE": "23.32"},
    "7201": {"Numberofshare": "100.00", "Eps": "(0.10 )", "Bookvalue": "1.23", "Parallel_value": "1.00", "PE_ratio": "سالب", "PB_ratio": "2.91", "ROA": "(5.98 )", "ROE": "(8.78 )"},
    "7202": {"Numberofshare": "120.00", "Eps": "12.52", "Bookvalue": "35.73", "Parallel_value": "10.00", "PE_ratio": "13.99", "PB_ratio": "4.98", "ROA": "12.04", "ROE": "36.23"},
    "7203": {"Numberofshare": "80.00", "Eps": "26.13", "Bookvalue": "45.26", "Parallel_value": "10.00", "PE_ratio": "21.57", "PB_ratio": "12.05", "ROA": "19.50", "ROE": "46.90"},
    "7204": {"Numberofshare": "330.00", "Eps": "0.46", "Bookvalue": "2.13", "Parallel_value": "1.00", "PE_ratio": "17.10", "PB_ratio": "3.47", "ROA": "9.22", "ROE": "24.14"},
    "7211": {"Numberofshare": "60.00", "Eps": "", "Bookvalue": "2.49", "Parallel_value": "", "PE_ratio": "30.60", "PB_ratio": "8.95", "ROA": "12.91", "ROE": "32.58"},
    "8010": {"Numberofshare": "", "Eps": "7.35", "Bookvalue": "35.72", "Parallel_value": "10.00", "PE_ratio": "17.20", "PB_ratio": "3.54", "ROA": "", "ROE": "22.43"},
    "8012": {"Numberofshare": "", "Eps": "0.59", "Bookvalue": "15.05", "Parallel_value": "10.00", "PE_ratio": "18.78", "PB_ratio": "0.74", "ROA": "", "ROE": "4.00"},
    "8020": {"Numberofshare": "", "Eps": "0.43", "Bookvalue": "9.25", "Parallel_value": "10.00", "PE_ratio": "أكبر من 100", "PB_ratio": "0.94", "ROA": "", "ROE": "4.85"},
    "8030": {"Numberofshare": "", "Eps": "0.37", "Bookvalue": "10.16", "Parallel_value": "10.00", "PE_ratio": "51.55", "PB_ratio": "1.89", "ROA": "", "ROE": "3.75"},
    "8040": {"Numberofshare": "", "Eps": "0.12", "Bookvalue": "12.06", "Parallel_value": "10.00", "PE_ratio": "74.53", "PB_ratio": "0.76", "ROA": "", "ROE": "1.04"},
    "8050": {"Numberofshare": "", "Eps": "(3.05 )", "Bookvalue": "9.00", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.88", "ROA": "", "ROE": "(34.93 )"},
    "8060": {"Numberofshare": "", "Eps": "(1.38 )", "Bookvalue": "13.16", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.69", "ROA": "", "ROE": "(10.04 )"},
    "8070": {"Numberofshare": "", "Eps": "(0.55 )", "Bookvalue": "20.47", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.53", "ROA": "", "ROE": "(2.66 )"},
    "8100": {"Numberofshare": "", "Eps": "0.74", "Bookvalue": "13.84", "Parallel_value": "10.00", "PE_ratio": "13.05", "PB_ratio": "0.70", "ROA": "", "ROE": "5.56"},
    "8120": {"Numberofshare": "", "Eps": "(1.73 )", "Bookvalue": "11.92", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.82", "ROA": "", "ROE": "(13.99 )"},
    "8150": {"Numberofshare": "", "Eps": "(3.08 )", "Bookvalue": "7.33", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.86", "ROA": "", "ROE": "(35.97 )"},
    "8160": {"Numberofshare": "", "Eps": "(0.85 )", "Bookvalue": "10.77", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.76", "ROA": "", "ROE": "(7.68 )"},
    "8170": {"Numberofshare": "", "Eps": "(2.32 )", "Bookvalue": "11.50", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.63", "ROA": "", "ROE": "(18.18 )"},
    "8180": {"Numberofshare": "", "Eps": "(1.43 )", "Bookvalue": "11.67", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.78", "ROA": "", "ROE": "(11.90 )"},
    "8190": {"Numberofshare": "", "Eps": "(4.80 )", "Bookvalue": "2.47", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.21", "ROA": "", "ROE": "(104.14 )"},
    "8200": {"Numberofshare": "", "Eps": "0.84", "Bookvalue": "13.10", "Parallel_value": "10.00", "PE_ratio": "31.50", "PB_ratio": "2.02", "ROA": "", "ROE": "7.36"},
    "8210": {"Numberofshare": "", "Eps": "7.19", "Bookvalue": "37.82", "Parallel_value": "10.00", "PE_ratio": "24.87", "PB_ratio": "4.59", "ROA": "", "ROE": "19.94"},
    "8230": {"Numberofshare": "", "Eps": "4.55", "Bookvalue": "25.36", "Parallel_value": "10.00", "PE_ratio": "21.33", "PB_ratio": "3.83", "ROA": "", "ROE": "19.71"},
    "8240": {"Numberofshare": "", "Eps": "0.32", "Bookvalue": "11.66", "Parallel_value": "10.00", "PE_ratio": "52.03", "PB_ratio": "1.45", "ROA": "", "ROE": "2.86"},
    "8250": {"Numberofshare": "", "Eps": "2.40", "Bookvalue": "22.89", "Parallel_value": "10.00", "PE_ratio": "9.61", "PB_ratio": "1.01", "ROA": "", "ROE": "11.01"},
    "8260": {"Numberofshare": "", "Eps": "(5.46 )", "Bookvalue": "6.39", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.58", "ROA": "", "ROE": "(68.64 )"},
    "8280": {"Numberofshare": "", "Eps": "0.67", "Bookvalue": "11.97", "Parallel_value": "10.00", "PE_ratio": "14.76", "PB_ratio": "0.82", "ROA": "", "ROE": "5.80"},
    "8300": {"Numberofshare": "", "Eps": "0.93", "Bookvalue": "16.45", "Parallel_value": "10.00", "PE_ratio": "13.09", "PB_ratio": "0.74", "ROA": "", "ROE": "5.85"},
    "8310": {"Numberofshare": "", "Eps": "(0.18 )", "Bookvalue": "7.14", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "0.77", "ROA": "", "ROE": "(2.64 )"},
    "8311": {"Numberofshare": "", "Eps": "(0.34 )", "Bookvalue": "7.19", "Parallel_value": "10.00", "PE_ratio": "سالب", "PB_ratio": "1.04", "ROA": "", "ROE": "(4.67 )"},
    "8313": {"Numberofshare": "77.51", "Eps": "3.18", "Bookvalue": "9.10", "Parallel_value": "1.00", "PE_ratio": "42.95", "PB_ratio": "14.50", "ROA": "21.65", "ROE": "44.06"},
    # ... [All remaining entries follow same format — kept identical to original]
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3d: STOCKS DATA ENRICHMENT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Merge static STOCKS data into yfinance info dict.
#           STOCKS data takes PRIORITY over yfinance.
#
# TO MODIFY:
#   • Add enrichment field → add new section in _enrich_with_STOCKS()
#   • Change priority      → modify if-checks
#   • Switch to live data  → replace STOCKS_STATIC_DATA lookups
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_float(value: str) -> float | None:
    """Parse string to float. Returns None if empty or Arabic non-numeric."""
    if value is None:
        return None
    value = str(value).strip()
    if not value or value in _PE_NON_NUMERIC or value == "-":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _enrich_with_STOCKS(ticker: str, info: dict) -> None:
    """
    Enrich yfinance info dict with STOCKS static data (in-place).
    
    Fields enriched: EPS, BookValue, P/E, P/B, Shares, MarketCap, ParValue.
    """
    code = ticker.replace(".SR", "").strip()
    STOCKS = STOCKS_STATIC_DATA.get(code)
    if not STOCKS:
        logger.info(f"STOCKS static: no data for {code}, using yfinance only")
        _ensure_eps_formatted(info)
        return

    logger.info(f"STOCKS static: found data for {code}")

    # ── 1. EPS (ربح السهم) ──
    eps = _safe_float(STOCKS.get("Eps"))
    if eps is not None:
        info["trailingEps"] = eps
        info["trailingEpsFormatted"] = f"({abs(eps)})" if eps < 0 else str(eps)
        logger.info(f"✓ STOCKS EPS for {ticker}: {eps}")
    else:
        _ensure_eps_formatted(info)

    # ── 2. Book Value (القيمة الدفترية) ──
    bv = _safe_float(STOCKS.get("Bookvalue"))
    if bv is not None:
        info["bookValue"] = bv

    # ── 3. P/E Ratio (مكرر الربح) ──
    pe_raw = STOCKS.get("PE_ratio", "").strip()
    pe = _safe_float(pe_raw)
    if pe is not None and pe > 0:
        info["trailingPE"] = pe
    elif pe_raw == "سالب":
        info["trailingPE"] = None
        info["trailingPE_display"] = "سالب"
    elif pe_raw == "أكبر من 100":
        info["trailingPE_display"] = "أكبر من 100"

    # ── 4. P/B Ratio (مضاعف القيمة الدفترية) ──
    pb = _safe_float(STOCKS.get("PB_ratio"))
    if pb is not None:
        info["priceToBook"] = pb

    # ── 5. Shares Outstanding (عدد الأسهم) — in millions ──
    shares = _safe_float(STOCKS.get("Numberofshare"))
    if shares is not None:
        shares_actual = shares * 1_000_000
        info["sharesOutstanding"] = shares_actual
        info["floatShares"] = shares_actual
        price = _get_current_price(info)
        if price and price > 0:
            info["marketCap"] = price * shares_actual

    # ── 6. Par Value (القيمة الاسمية) ──
    par = _safe_float(STOCKS.get("Parallel_value"))
    if par is not None:
        info["parValue"] = par

    # # ── 7. Recalculate P/E if missing ──
    # if not info.get("trailingPE") and not info.get("trailingPE_display"):
    #     eps_val = info.get("trailingEps")
    #     if eps_val and eps_val != 0:
    #         price = _get_current_price(info)
    #         if price and price > 0:
    #             try:
    #                 info["trailingPE"] = round(price / float(eps_val), 4)
    #             except Exception:
    #                 pass

    # # ── 8. Recalculate P/B if missing ──
    # if not info.get("priceToBook"):
    #     bv_val = info.get("bookValue")
    #     if bv_val and float(bv_val) != 0:
    #         price = _get_current_price(info)
    #         if price and price > 0:
    #             try:
    #                 info["priceToBook"] = round(price / float(bv_val), 4)
    #             except Exception:
    #                 pass
##########################################################################
#────────────────────────────────────────────────────────#
    # 7. العائد على متوسط الأصول (%) - ROA
    # ────────────────────────────────────────────────────────#
    roa = _safe_float(STOCKS.get("ROA"))
    if roa is not None:
        info["returnOnAssets"] = roa / 100.0  # Convert percentage to decimal
        info["returnOnAssetsDisplay"] = roa   # Keep percentage for display
        logger.info(f"✓ STOCKS ROA for {ticker}: {roa}%")

#────────────────────────────────────────────────────────#
    # 8. العائد على متوسط حقوق المساهمين (%) - ROE
    # ────────────────────────────────────────────────────────#
    roe = _safe_float(STOCKS.get("ROE"))
    if roe is not None:
        info["returnOnEquity"] = roe / 100.0   # Convert percentage to decimal
        info["returnOnEquityDisplay"] = roe    # Keep percentage for display
        logger.info(f"✓ STOCKS ROE for {ticker}: {roe}%")
    # ── 9. Defaults ──
    if not info.get("currency"):
        info["currency"] = "SAR"
    if not info.get("exchange"):
        info["exchange"] = "Tadawul"
    logger.info(f"STOCKS enrichment complete for {ticker}")


def _ensure_eps_formatted(info: dict) -> None:
    """Ensure trailingEpsFormatted exists. Negative EPS shown in parentheses."""
    if info.get("trailingEpsFormatted"):
        return
    yahoo_eps = info.get("trailingEps")
    if yahoo_eps is not None:
        try:
            yahoo_eps = float(yahoo_eps)
            info["trailingEpsFormatted"] = f"({abs(yahoo_eps)})" if yahoo_eps < 0 else str(yahoo_eps)
        except (ValueError, TypeError):
            info["trailingEpsFormatted"] = "لاحقا"
    else:
        info["trailingEpsFormatted"] = "لاحقا"


def _get_current_price(info: dict) -> float | None:
    """Extract current price from info dict, trying multiple keys."""
    for key in ("currentPrice", "regularMarketPrice", "previousClose"):
        val = info.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PDF THEME CONSTANTS (Colors, Page Size, Margins)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Define color palette, page dimensions, and margins for PDF reports.
#
# TO MODIFY:
#   • Change PDF colors  → update HEX values
#   • Change page size   → replace A4 with LETTER or custom
#   • Change margins     → adjust MG value (currently 18mm)
# ═══════════════════════════════════════════════════════════════════════════════

# --- HEX Color Strings (for charts and CSS-style references) ---
NAVY_HEX, TEAL_HEX, GREEN_HEX = '#1B2A4A', '#00897B', '#4CAF50'
RED_HEX, ORANGE_HEX, LGRAY_HEX = '#E53935', '#FF9800', '#F0F3F7'
DGRAY_HEX, TXTDARK_HEX, WHITE_HEX = '#5A6272', '#1A1A2E', '#FFFFFF'
BLUE_HEX, VIOLET_HEX, BLACK_HEX = '#2196F3', '#C620F8', '#000000'

# --- ReportLab Color Objects (for PDF drawing) ---
NAVY, TEAL, GREEN = HexColor(NAVY_HEX), HexColor(TEAL_HEX), HexColor(GREEN_HEX)
RED, ORANGE, LGRAY = HexColor(RED_HEX), HexColor(ORANGE_HEX), HexColor(LGRAY_HEX)
DGRAY, TXTDARK, WHITE = HexColor(DGRAY_HEX), HexColor(TXTDARK_HEX), HexColor(WHITE_HEX)
BLUE, VILOET, BLACK = HexColor(BLUE_HEX), HexColor(VIOLET_HEX), HexColor(BLACK_HEX)

# --- Page Dimensions ---
PAGE_W, PAGE_H = A4            # A4 = 595.28 × 841.89 points
MG = 18 * mm                    # Page margin on each side
CW = PAGE_W - 2 * MG            # Content width (usable drawing area)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: NUMBER & PERCENT FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Format numbers and percentages for PDF display.
#           Handles large numbers (T/B/M suffixes) and negative coloring.
#
# TO MODIFY:
#   • Change decimal places  → adjust `d` parameter
#   • Change negative color  → replace RED reference
#   • Add K suffix           → add `elif av >= 1e3` block
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_n(v, d=2):
    """Format number with T/B/M suffixes. Returns (string, color_or_None)."""
    if v is None:
        return '-', None
    try:
        v = float(v)
        av = abs(v)
        if av >= 1e12:
            return f'{v/1e12:.{d}f} T', None
        if av >= 1e9:
            return f'{v/1e9:.{d}f} B', None
        if av >= 1e6:
            return f'{v/1e6:.{d}f} M', None
        return f'{v:,.{d}f}', RED if v < 0 else None
    except Exception:
        return '-', None


def fmt_p(v):
    """Format percentage. Values ≤1 treated as decimals (×100)."""
    if v is None:
        return '-', None
    try:
        fv = float(v)
        if abs(fv) <= 1:
            return f'{fv*100:.2f}%', RED if fv < 0 else None
        return f'{fv:.2f}%', RED if fv < 0 else None
    except Exception:
        return '-', None


def chart_bytes(fig):
    """Convert matplotlib figure → PNG bytes (BytesIO buffer). Closes figure."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: STOCK DATA FETCHING (fetch_data)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : CORE DATA ENGINE — fetches OHLCV + fundamentals from yfinance,
#           then enriches with STOCKS static data.
#
# WHAT IT DOES:
#   1. Downloads 1Y and 2Y OHLCV history
#   2. Extracts .info dict (market cap, P/E, etc.)
#   3. Fills gaps from fast_info, financials, balance sheet
#   4. Calculates beta vs TASI index
#   5. Enriches with STOCKS static data (overrides yfinance)
#   6. Adds volume, trading value, trades count
#
# RETURNS: (df_1y, df_2y, info_dict) or (None, None, {}) on error
#
# TO MODIFY:
#   • Switch data source      → replace yf.Ticker calls
#   • Add new fundamental     → add extraction logic after balance sheet
#   • Change beta benchmark   → replace '^TASI.SR'
#   • Skip STOCKS enrichment  → remove _enrich_with_STOCKS() call
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_data(ticker):
    """Fetch price data + supplement every info field for Saudi stocks."""
    try:
        stk = yf.Ticker(ticker)
        df = stk.history(period='1y')
        if df.empty:
            return None, None, {}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df2 = stk.history(period='2y')
        if isinstance(df2.columns, pd.MultiIndex):
            df2.columns = df2.columns.get_level_values(0)
        try:
            info = stk.info or {}
        except Exception:
            info = {}

        price_now = float(df['Close'].iloc[-1])

        # ── Fill info gaps from fast_info ──
        try:
            fi = stk.fast_info
            for ikey, attr in [
                ('marketCap','market_cap'), ('sharesOutstanding','shares'),
                ('fiftyTwoWeekHigh','fifty_two_week_high'),
                ('fiftyTwoWeekLow','fifty_two_week_low'),
                ('currency','currency'), ('exchange','exchange'),
            ]:
                if not info.get(ikey):
                    try:
                        v = getattr(fi, attr, None)
                        if v is not None: info[ikey] = v
                    except Exception: pass
            if not info.get('floatShares'):
                try:
                    v = getattr(fi, 'float_shares', None)
                    if v is not None: info['floatShares'] = v
                except Exception: pass
        except Exception: pass

        # ── 52-week high/low from actual data ──
        try:
            tail252 = df2['High'].tail(252) if len(df2) >= 252 else df['High']
            tail252l = df2['Low'].tail(252) if len(df2) >= 252 else df['Low']
            info['fiftyTwoWeekHigh'] = float(tail252.max())
            info['fiftyTwoWeekLow']  = float(tail252l.min())
        except Exception: pass

        # ── Saudi stock defaults ──
        if ticker.endswith('.SR'):
            if not info.get('currency'): info['currency'] = 'SAR'
            if not info.get('exchange'): info['exchange'] = 'Tadawul'

        # ── Sector/Industry from local map ──
        sec, ind = get_sector_industry(ticker)
        if sec and not info.get('sector'):   info['sector']   = sec
        if ind and not info.get('industry'): info['industry'] = ind

        # ── Average Volume ──
        if not info.get('averageVolume'):
            try: info['averageVolume'] = int(df['Volume'].mean())
            except Exception: pass

        # ── Beta calculation (vs TASI index) ──
        if not info.get('beta'):
            try:
                mkt = yf.Ticker('^TASI.SR').history(period='1y')['Close']
                stk_ret = df['Close'].pct_change().dropna()
                mkt_ret = mkt.pct_change().dropna()
                aligned  = pd.concat([stk_ret, mkt_ret], axis=1, join='inner').dropna()
                if len(aligned) > 30:
                    cov = aligned.cov().iloc[0,1]
                    var = aligned.iloc[:,1].var()
                    if var > 0: info['beta'] = round(cov / var, 3)
            except Exception: pass

        # ── Dividend data ──
        try:
            divs = stk.dividends
            if divs is not None and len(divs) > 0:
                cutoff = pd.Timestamp.now(tz=divs.index.tz) - pd.DateOffset(years=1)
                annual_div = float(divs[divs.index >= cutoff].sum())
                if annual_div > 0:
                    if not info.get('dividendRate'): info['dividendRate'] = annual_div
                    if not info.get('dividendYield'): info['dividendYield'] = annual_div / price_now
        except Exception: pass

        # ── Financial Statements (Income Statement + Balance Sheet) ──
        fin_ok = bs_ok = False
        rev = ni = equity = total_assets = curr_assets = curr_liab = inventory = None
        try:
            fin = stk.financials
            bs  = stk.balance_sheet
            fin_ok = not fin.empty
            bs_ok  = not bs.empty

            def _fv(frame, *cands):
                """Find first matching row in financial statement."""
                for c in cands:
                    ks = [k for k in frame.index if c.lower() in str(k).lower()]
                    if ks:
                        try: return float(frame.loc[ks[0]].iloc[0])
                        except Exception: continue
                return None

            if fin_ok:
                rev = _fv(fin, 'Total Revenue')
                ni  = _fv(fin, 'Net Income Common Stockholders', 'Net Income')
                op_inc = _fv(fin, 'Operating Income')
                gross  = _fv(fin, 'Gross Profit')
                ebitda = _fv(fin, 'EBITDA', 'Normalized EBITDA')
                if rev and not info.get('totalRevenue'):      info['totalRevenue']      = rev
                if ni  and not info.get('netIncomeToCommon'): info['netIncomeToCommon'] = ni
                if ebitda and not info.get('ebitda'):         info['ebitda']            = ebitda
                if rev and rev != 0:
                    if gross   and not info.get('grossMargins'):    info['grossMargins']    = gross / rev
                    if op_inc  and not info.get('operatingMargins'): info['operatingMargins'] = op_inc / rev
                    if ni      and not info.get('profitMargins'):    info['profitMargins']    = ni / rev

            if bs_ok:
                equity      = _fv(bs, 'Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest')
                total_assets = _fv(bs, 'Total Assets')
                curr_assets  = _fv(bs, 'Current Assets')
                curr_liab    = _fv(bs, 'Current Liabilities')
                inventory    = _fv(bs, 'Inventory')
                cash_val     = _fv(bs, 'Cash Cash Equivalents And Short Term Investments', 'Cash And Cash Equivalents')
                debt_val     = _fv(bs, 'Total Debt', 'Long Term Debt And Capital Lease Obligation', 'Long Term Debt')
                if cash_val and not info.get('totalCash'):  info['totalCash']  = cash_val
                if debt_val and not info.get('totalDebt'):  info['totalDebt']  = debt_val
                shares_out = info.get('sharesOutstanding')
                if equity and shares_out and not info.get('bookValue'):
                    try: info['bookValue'] = equity / float(shares_out)
                    except Exception: pass
                if equity and shares_out and not info.get('priceToBook'):
                    try:
                        bps = equity / float(shares_out)
                        if bps > 0: info['priceToBook'] = price_now / bps
                    except Exception: pass
                if equity and ni is not None and equity != 0 and not info.get('returnOnEquity'):
                    info['returnOnEquity'] = ni / abs(equity)
                if total_assets and ni is not None and total_assets != 0 and not info.get('returnOnAssets'):
                    info['returnOnAssets'] = ni / abs(total_assets)
                if curr_assets and curr_liab and curr_liab != 0:
                    if not info.get('currentRatio'):
                        info['currentRatio'] = round(curr_assets / curr_liab, 2)
                    if not info.get('quickRatio'):
                        num = curr_assets - (inventory or 0)
                        info['quickRatio'] = round(num / curr_liab, 2)
                if debt_val and equity and equity != 0 and not info.get('debtToEquity'):
                    info['debtToEquity'] = round((debt_val / abs(equity)) * 100, 2)

            shares_out = info.get('sharesOutstanding')
            if fin_ok and ni is not None and shares_out and float(shares_out) > 0:
                eps = ni / float(shares_out)
                if not info.get('trailingEps'): info['trailingEps'] = eps
                if not info.get('trailingPE') and eps != 0:
                    info['trailingPE'] = price_now / eps

            # ── Enterprise Value ratios ──
            mktcap   = info.get('marketCap', 0) or 0
            tot_debt = info.get('totalDebt', 0) or 0
            tot_cash = info.get('totalCash', 0) or 0
            ev = mktcap + tot_debt - tot_cash
            if ev > 0:
                ebitda_v = info.get('ebitda')
                if ebitda_v and ebitda_v != 0 and not info.get('enterpriseToEbitda'):
                    info['enterpriseToEbitda'] = ev / ebitda_v
                rev_v = info.get('totalRevenue')
                if rev_v and rev_v != 0:
                    if not info.get('enterpriseToRevenue'):
                        info['enterpriseToRevenue'] = ev / rev_v
                    if not info.get('priceToSalesTrailing12Months') and mktcap > 0:
                        info['priceToSalesTrailing12Months'] = mktcap / rev_v

            div_rate = info.get('dividendRate')
            eps_v    = info.get('trailingEps')
            if div_rate and eps_v and eps_v > 0 and not info.get('payoutRatio'):
                info['payoutRatio'] = div_rate / eps_v
            if not info.get('floatShares') and info.get('sharesOutstanding'):
                info['floatShares'] = info['sharesOutstanding']

        except Exception as e:
            logger.warning(f"fetch_data supplement error: {e}")

        # ── STOCKS enrichment: overrides yfinance fundamentals ──
        try:
            _enrich_with_STOCKS(ticker, info)
        except Exception as _ae:
            logger.warning(f"STOCKS enrichment skipped: {_ae}")

        # ── YF FALLBACKS (only if STOCKS didn't populate) ──
        if not info.get('volume') and len(df) > 0 and 'Volume' in df.columns:
            try: info['volume'] = float(df['Volume'].iloc[-1])
            except Exception: pass
        if not info.get('tradingValue') and len(df) > 0:
            try:
                vol_last   = float(df['Volume'].iloc[-1])
                close_last = float(df['Close'].iloc[-1])
                if vol_last > 0 and close_last > 0:
                    info['tradingValue'] = vol_last * close_last
            except Exception: pass
        if not info.get('trailingEps'):
            try:
                eps_raw = stk.info.get('trailingEps')
                if eps_raw is not None: info['trailingEps'] = float(eps_raw)
            except Exception: pass
            if not info.get('trailingEps'):
                try:
                    ni_val = info.get('netIncomeToCommon')
                    shares_val = info.get('sharesOutstanding')
                    if ni_val and shares_val and float(shares_val) > 0:
                        info['trailingEps'] = float(ni_val) / float(shares_val)
                except Exception: pass
        if not info.get('tradesCount'):
            try:
                trading_val = info.get('tradingValue')
                if trading_val and trading_val > 0:
                    avg_trade_size = 10_000
                    info['tradesCount'] = int(trading_val / avg_trade_size)
            except Exception: pass

        return df, df2, info
    except Exception as e:
        logger.error(f"fetch_data error: {e}")
        return None, None, {}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6b: SUPERTREND INDICATOR HELPER
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Compute Supertrend — a trend-following overlay indicator.
#           Used in make_supertrend_chart() for PDF report.
#
# ALGORITHM:
#   1. True Range → ATR (period bars)
#   2. Upper/Lower bands = HL2 ± multiplier × ATR
#   3. Trailing band logic to avoid whipsaws
#   4. Direction: 1=bullish (green), -1=bearish (red)
#
# TO MODIFY:
#   • More sensitive → reduce period or multiplier
#   • Less noise     → increase period or multiplier
#   • Default: period=10, multiplier=3.0
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_supertrend(df, period=10, multiplier=3.0):
    """Compute Supertrend. Returns (supertrend_series, direction_series)."""
    h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float)
    c = df['Close'].values.astype(float)
    n = len(c)
    if n == 0:
        empty = pd.Series(dtype=float, index=df.index)
        return empty, empty.astype(int)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().values
    hl2 = (h + l) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n, dtype=int)
    upper_band[0] = upper_basic[0]
    lower_band[0] = lower_basic[0]
    if c[0] <= upper_band[0]:
        supertrend[0] = upper_band[0]; direction[0] = -1
    else:
        supertrend[0] = lower_band[0]; direction[0] = 1
    for i in range(1, n):
        if upper_basic[i] < upper_band[i - 1] or c[i - 1] > upper_band[i - 1]:
            upper_band[i] = upper_basic[i]
        else:
            upper_band[i] = upper_band[i - 1]
        if lower_basic[i] > lower_band[i - 1] or c[i - 1] < lower_band[i - 1]:
            lower_band[i] = lower_basic[i]
        else:
            lower_band[i] = lower_band[i - 1]
        if supertrend[i - 1] == upper_band[i - 1]:
            if c[i] > upper_band[i]:
                supertrend[i] = lower_band[i]; direction[i] = 1
            else:
                supertrend[i] = upper_band[i]; direction[i] = -1
        else:
            if c[i] < lower_band[i]:
                supertrend[i] = upper_band[i]; direction[i] = -1
            else:
                supertrend[i] = lower_band[i]; direction[i] = 1
    return (pd.Series(supertrend, index=df.index),
            pd.Series(direction, index=df.index))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6c: TECHNICAL INDICATOR COMPUTATION (compute_indicators)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Calculate 30+ technical indicators from OHLCV data.
#
# INDICATORS COMPUTED:
#   Moving Averages:  SMA(7,20,50,100,200), EMA(7,20,50,100,200,12,26)
#   Momentum:         RSI(14), MACD+Signal+Histogram, ROC(12)
#   Bands:            Bollinger Bands (20,2), BB Position %
#   Oscillators:      Stochastic %K/%D, CCI(20), Williams %R(14)
#   Trend:            ADX(14), +DI/-DI, Parabolic SAR
#   Volume:           OBV, VWAP, Bull/Bear Volume
#   Ichimoku:         Tenkan(9), Kijun(26), Senkou A/B, Chikou
#   Williams:         Alligator (Jaw/Teeth/Lips)
#   Volatility:       ATR(14)
#
# TO MODIFY:
#   • Add new indicator → add calculation at end, e.g. d['NEW'] = ...
#   • Change period     → modify the number in rolling()/ewm()
#   • Remove indicator  → delete its lines (may break downstream)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_indicators(df):
    """Calculate all technical indicators. Returns enriched DataFrame copy."""
    d = df.copy()
    c, h, l, v = d['Close'], d['High'], d['Low'], d['Volume']

    # ── Moving Averages (Simple) ──
    d['SMA7'] = c.rolling(7, min_periods=1).mean()
    d['SMA20'] = c.rolling(20, min_periods=1).mean()
    d['SMA50'] = c.rolling(50, min_periods=1).mean()
    d['SMA100'] = c.rolling(100, min_periods=1).mean()
    d['SMA200'] = c.rolling(200, min_periods=1).mean()

    # ── Moving Averages (Exponential) ──
    d['EMA7'] = c.ewm(span=7, adjust=False, min_periods=1).mean()
    d['EMA20'] = c.ewm(span=20, adjust=False, min_periods=1).mean()
    d['EMA50'] = c.ewm(span=50, adjust=False, min_periods=1).mean()
    d['EMA100'] = c.ewm(span=100, adjust=False, min_periods=1).mean()
    d['EMA200'] = c.ewm(span=200, adjust=False, min_periods=1).mean()
    d['EMA12'] = c.ewm(span=12, adjust=False, min_periods=1).mean()
    d['EMA26'] = c.ewm(span=26, adjust=False, min_periods=1).mean()

    # ── RSI (Relative Strength Index — 14 period) ──
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d['RSI'] = 100 - 100 / (1 + rs)

    # ── MACD (Moving Average Convergence Divergence) ──
    d['MACD'] = d['EMA12'] - d['EMA26']
    d['MACD_Sig'] = d['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
    d['MACD_H'] = d['MACD'] - d['MACD_Sig']  # Histogram

    # ── Bollinger Bands (20 period, 2 std dev) ──
    bm = c.rolling(20, min_periods=1).mean()
    bs = c.rolling(20, min_periods=1).std()
    d['BB_U'], d['BB_M'], d['BB_L'] = bm + 2*bs, bm, bm - 2*bs
    bb_range = d['BB_U'] - d['BB_L']
    d['BB_P'] = np.where(bb_range != 0, (c - d['BB_L']) / bb_range, 0.5)

    # ── Stochastic Oscillator (%K, %D) ──
    l14 = l.rolling(14, min_periods=1).min()
    h14 = h.rolling(14, min_periods=1).max()
    stoch_range = h14 - l14
    d['SK'] = np.where(stoch_range != 0, 100*(c-l14)/stoch_range, 50)
    d['SD'] = pd.Series(d['SK'], index=d.index).rolling(3, min_periods=1).mean()

    # ── ATR (Average True Range — 14 period) ──
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    d['ATR'] = tr.rolling(14, min_periods=1).mean()

    # ── OBV (On-Balance Volume) ──
    d['OBV'] = (np.sign(c.diff()) * v).fillna(0).cumsum()

    # ── VWAP (Volume Weighted Average Price) ──
    tp = (h + l + c) / 3
    cumsum_vol = v.cumsum().replace(0, np.nan)
    d['VWAP'] = (tp * v).cumsum() / cumsum_vol

    # ── ADX, +DI, -DI (Average Directional Index — 14 period) ──
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr14 = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    atr14_safe = atr14.replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr14_safe
    minus_di = 100 * minus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr14_safe
    di_sum = plus_di + minus_di
    dx = np.where(di_sum != 0, 100*(plus_di-minus_di).abs()/di_sum, 0)
    d['ADX'] = pd.Series(dx, index=d.index).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    d['PDI'], d['MDI'] = plus_di, minus_di

    # ── Parabolic SAR (simplified — uses rolling min) ──
    d['SAR'] = c.rolling(5, min_periods=1).min()

    # ── Ichimoku Cloud Components ──
    high9 = h.rolling(9, min_periods=1).max(); low9 = l.rolling(9, min_periods=1).min()
    d['Tenkan'] = (high9 + low9) / 2                          # Conversion Line (9)
    high26 = h.rolling(26, min_periods=1).max(); low26 = l.rolling(26, min_periods=1).min()
    d['Kijun'] = (high26 + low26) / 2                          # Base Line (26)
    d['Senkou_A'] = ((d['Tenkan'] + d['Kijun']) / 2).shift(26) # Leading Span A
    high52 = h.rolling(52, min_periods=1).max(); low52 = l.rolling(52, min_periods=1).min()
    d['Senkou_B'] = ((high52 + low52) / 2).shift(26)           # Leading Span B
    d['Chikou'] = c.shift(-26)                                  # Lagging Span

    # ── Williams Alligator ──
    d['Alligator_Jaw']   = c.shift(8).rolling(13, min_periods=1).mean()   # Blue (13,8)
    d['Alligator_Teeth'] = c.shift(5).rolling(8, min_periods=1).mean()    # Red (8,5)
    d['Alligator_Lips']  = c.shift(3).rolling(5, min_periods=1).mean()    # Green (5,3)

    # ── Bull/Bear Volume (20-period rolling sum) ──
    d['Volume_Up'] = np.where(c > c.shift(1), v, 0)
    d['Volume_Down'] = np.where(c < c.shift(1), v, 0)
    d['Bull_Volume'] = pd.Series(d['Volume_Up'], index=d.index).rolling(20, min_periods=1).sum()
    d['Bear_Volume'] = pd.Series(d['Volume_Down'], index=d.index).rolling(20, min_periods=1).sum()

    # ── CCI (Commodity Channel Index — 20 period) ──
    tp_cci = (h + l + c) / 3
    tp_mean = tp_cci.rolling(20, min_periods=1).mean()
    tp_mad = tp_cci.rolling(20, min_periods=1).apply(lambda x: np.mean(np.abs(x-x.mean())), raw=True)
    d['CCI'] = np.where(tp_mad != 0, (tp_cci - tp_mean)/(0.015*tp_mad), 0)

    # ── Williams %R (14 period) ──
    h14w = h.rolling(14, min_periods=1).max(); l14w = l.rolling(14, min_periods=1).min()
    wr_range = h14w - l14w
    d['WILLR'] = np.where(wr_range != 0, -100*(h14w-c)/wr_range, -50)

    # ── ROC (Rate of Change — 12 period) ──
    d['ROC12'] = c.pct_change(12) * 100

    return d


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6d: PERFORMANCE & RISK METRICS (compute_performance)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Calculate return percentages and risk metrics for PDF report.
#
# RETURNS CALCULATED:
#   1 Day, 1 Week, 1 Month, 3 Months, 6 Months, 1 Year, YTD
#
# RISK METRICS:
#   Daily/Annual Volatility, Sharpe Ratio, Sortino Ratio,
#   Max Drawdown, Avg Daily Return, Best Day, Worst Day
#
# TO MODIFY:
#   • Add new period → add to `for days, label` loop
#   • Change Sharpe risk-free rate → currently 0 (modify ann_ret/ann_vol)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_performance(df, df2):
    """Calculate performance returns and risk metrics."""
    c, c2 = df['Close'], df2['Close']
    last = float(c.iloc[-1])
    pers = {}

    # ── Period returns ──
    for days, label in [(2,'1 Day'),(5,'1 Week'),(22,'1 Month'),(66,'3 Months'),(132,'6 Months')]:
        if len(c) >= days:
            pers[label] = (last / float(c.iloc[-days]) - 1) * 100
    if len(c2) >= 252:
        pers['1 Year'] = (last / float(c2.iloc[-252]) - 1) * 100

    # ── YTD return ──
    year_start = pd.Timestamp(f'{datetime.now().year}-01-01')
    if c.index.tz is not None:
        year_start = year_start.tz_localize(c.index.tz)
    ytd = c[c.index >= year_start]
    if len(ytd) >= 2:
        pers['YTD'] = (last / float(ytd.iloc[0]) - 1) * 100

    # ── Risk metrics ──
    dr = c.pct_change().dropna()
    if len(dr) == 0:
        risk = {k: '-' for k in ['Daily Volatility','Annual Volatility','Sharpe Ratio',
                                   'Sortino Ratio','Max Drawdown','Avg Daily Return',
                                   'Best Day','Worst Day']}
        dd = pd.Series([0], index=c.index[-1:])
        return pers, risk, dd

    ann_vol = dr.std() * np.sqrt(252)
    ann_ret = dr.mean() * 252
    risk = {
        'Daily Volatility': f'{dr.std()*100:.2f}%',
        'Annual Volatility': f'{ann_vol*100:.2f}%',
        'Sharpe Ratio': f'{ann_ret/ann_vol:.2f}' if ann_vol > 0 else '-',
    }
    neg = dr[dr < 0]
    dv = neg.std() * np.sqrt(252) if len(neg) > 0 else 0
    risk['Sortino Ratio'] = f'{ann_ret/dv:.2f}' if dv > 0 else '-'

    # ── Drawdown calculation ──
    cum = (1 + dr).cumprod()
    pk = cum.cummax()
    dd = (cum - pk) / pk
    risk.update({
        'Max Drawdown': f'{dd.min()*100:.2f}%',
        'Avg Daily Return': f'{dr.mean()*100:.4f}%',
        'Best Day': f'{dr.max()*100:.2f}%',
        'Worst Day': f'{dr.min()*100:.2f}%',
    })
    return pers, risk, dd


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6e: 20-POINT SCORING SYSTEM (compute_score_criteria)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Score a stock 0–20 based on 20 bullish/bearish criteria.
#           Each criterion is worth 1 point if bullish condition is met.
#
# CRITERIA (20 total):
#   1–4:   Price vs EMA 7/20/50/200
#   5:     EMA 50 > EMA 200
#   6:     EMA order (7>20>50>200)
#   7–8:   MACD > Signal, Histogram > 0
#   9:     RSI > 50
#   10:    ROC 12 > 0
#   11:    ADX > 25
#   12:    +DI > -DI
#   13:    OBV rising
#   14:    Bull volume > Bear volume
#   15:    Price above Ichimoku cloud
#   16:    Tenkan > Kijun
#   17:    SAR bullish
#   18:    Price above Alligator
#   19:    Price above Bollinger midline
#   20:    Price above previous week's high
#
# TO MODIFY:
#   • Add criterion  → add to `checks` list (increases max score)
#   • Remove criterion → delete from `checks`
#   • Change weight  → modify scoring logic (currently 1 point each)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_score_criteria(d):
    """Compute 20-point technical score. Returns (criteria_dict, total_score)."""
    last = d.iloc[-1]
    c = float(last['Close'])
    criteria = {}
    score = 0

    # ── Build checks list ──
    checks = [
        ('1. السعر > متوسط 7', pd.notna(last['EMA7']) and c > float(last['EMA7'])),
        ('2. السعر > متوسط 20', pd.notna(last['EMA20']) and c > float(last['EMA20'])),
        ('3. السعر > متوسط 50', pd.notna(last['EMA50']) and c > float(last['EMA50'])),
        ('4. السعر > متوسط 200', pd.notna(last['EMA200']) and c > float(last['EMA200'])),
    ]

    ema7 = float(last['EMA7']) if pd.notna(last['EMA7']) else 0
    ema20 = float(last['EMA20']) if pd.notna(last['EMA20']) else 0
    ema50 = float(last['EMA50']) if pd.notna(last['EMA50']) else 0
    ema200 = float(last['EMA200']) if pd.notna(last['EMA200']) else 0

    checks += [
        ('5. متوسط 50 > متوسط 200', pd.notna(last['EMA50']) and pd.notna(last['EMA200']) and float(last['EMA50']) > float(last['EMA200'])),
        ('6. ترتيب المتوسطات إيجابي', ema7 > ema20 > ema50 > ema200),
        ('7. خط الماكد > خط الإشارة', pd.notna(last['MACD']) and pd.notna(last['MACD_Sig']) and float(last['MACD']) > float(last['MACD_Sig'])),
        ('8. الماكد هستوجرام > 0', pd.notna(last['MACD_H']) and float(last['MACD_H']) > 0),
        ('9. مؤشر القوة النسبية > 50', pd.notna(last['RSI']) and float(last['RSI']) > 50),
        ('10. مؤشر معدل التغير > 0', pd.notna(last['ROC12']) and float(last['ROC12']) > 0),
        ('11. مؤشر ADX > 25', pd.notna(last['ADX']) and float(last['ADX']) > 25),
        ('12. مؤشر +DI > مؤشر -DI', pd.notna(last['PDI']) and pd.notna(last['MDI']) and float(last['PDI']) > float(last['MDI'])),
    ]

    # ── OBV rising (check last 5 bars) ──
    if len(d) >= 5:
        obv_curr = float(last['OBV']) if pd.notna(last['OBV']) else 0
        obv_prev = float(d.iloc[-5]['OBV']) if pd.notna(d.iloc[-5]['OBV']) else 0
        checks.append(('13. مؤشر OBV صاعد', obv_curr > obv_prev))
    else:
        checks.append(('13. مؤشر OBV صاعد', False))

    checks += [
        ('14. سيولة شرائية مسيطرة', pd.notna(last['Bull_Volume']) and pd.notna(last['Bear_Volume']) and float(last['Bull_Volume']) > float(last['Bear_Volume'])),
    ]

    # ── Ichimoku cloud check ──
    sa = float(last['Senkou_A']) if pd.notna(last['Senkou_A']) else 0
    sb = float(last['Senkou_B']) if pd.notna(last['Senkou_B']) else 0
    kumo_top = max(sa, sb)

    checks += [
        ('15. السعر > سحابة الإيشيموكو', c > kumo_top and kumo_top > 0),
        ('16. خط التنكن > خط الكيجن', pd.notna(last['Tenkan']) and pd.notna(last['Kijun']) and float(last['Tenkan']) > float(last['Kijun'])),
        ('17. مؤشر SAR إيجابي', pd.notna(last['SAR']) and c > float(last['SAR'])),
    ]

    # ── Alligator check ──
    jaw = float(last['Alligator_Jaw']) if pd.notna(last['Alligator_Jaw']) else 0
    teeth = float(last['Alligator_Teeth']) if pd.notna(last['Alligator_Teeth']) else 0
    lips = float(last['Alligator_Lips']) if pd.notna(last['Alligator_Lips']) else 0

    checks += [
        ('18. السعر > مؤشر التمساح', c > max(jaw, teeth, lips) and max(jaw, teeth, lips) > 0),
        ('19. السعر > خط منتصف البولينجر', pd.notna(last['BB_M']) and c > float(last['BB_M'])),
    ]

    # ── Price above previous week high ──
    if len(d) >= 5:
        week_high = d['High'].iloc[-6:-1].max()
        checks.append(('20. السعر > قمة الأسبوع السابق', c > float(week_high)))
    else:
        checks.append(('20. السعر > قمة الأسبوع السابق', False))

    # ── Score each criterion ──
    for lbl, cond in checks:
        if cond:
            criteria[lbl] = ('✓', 1); score += 1
        else:
            criteria[lbl] = ('✗', 0)

    return criteria, score


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6f: SIGNAL GENERATION (gen_signals)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Generate buy/sell/neutral signals from technical indicators.
#           Used in PDF report "Signals" table.
#
# SIGNALS GENERATED:
#   • Price vs SMA/EMA (20, 50, 200)
#   • Golden/Death cross (SMA 50/200, EMA 50/200)
#   • RSI zones (overbought/oversold/neutral)
#   • MACD direction
#   • ROC 12 momentum
#   • ADX trend strength & direction
#   • Volume analysis
#
# TO MODIFY:
#   • Add signal → add new if/elif block
#   • Change thresholds → modify RSI 70/30, ADX 25, etc.
# ═══════════════════════════════════════════════════════════════════════════════

def gen_signals(d):
    """Generate technical signals. Returns (signals_dict, signal_score)."""
    sig = {}
    last = d.iloc[-1]
    c = float(last['Close'])

    # ── Price vs SMA ──
    for col, lbl in [('SMA20','السعر مقابل SMA 20'),('SMA50','السعر مقابل SMA 50'),('SMA200','السعر مقابل SMA 200')]:
        v = float(last[col]) if pd.notna(last[col]) else None
        if v is not None:
            sig[lbl] = ('صاعد ▲', 1) if c > v else ('هابط ▼', -1)

    # ── Price vs EMA ──
    for col, lbl in [('EMA20','السعر مقابل EMA 20'),('EMA50','السعر مقابل EMA 50'),('EMA200','السعر مقابل EMA 200')]:
        v = float(last[col]) if pd.notna(last[col]) else None
        if v is not None:
            sig[lbl] = ('صاعد ▲', 1) if c > v else ('هابط ▼', -1)

    # ── Golden/Death Cross (SMA) ──
    sma50 = float(last['SMA50']) if pd.notna(last['SMA50']) else None
    sma200 = float(last['SMA200']) if pd.notna(last['SMA200']) else None
    if sma50 and sma200:
        sig['تقاطع SMA 50 / 200'] = ('تقاطع ذهبي ▲', 1) if sma50 > sma200 else ('تقاطع سلبي ▼', -1)

    # ── Golden/Death Cross (EMA) ──
    ema50 = float(last['EMA50']) if pd.notna(last['EMA50']) else None
    ema200 = float(last['EMA200']) if pd.notna(last['EMA200']) else None
    if ema50 and ema200:
        sig['تقاطع EMA 50 / 200'] = ('تقاطع ذهبي ▲', 1) if ema50 > ema200 else ('تقاطع سلبي ▼', -1)

    # ── RSI zones ──
    rsi = float(last['RSI']) if pd.notna(last['RSI']) else None
    if rsi is not None:
        if rsi > 70:   sig['RSI (14)'] = (f'تشبع شراء ({rsi:.1f})', -1)
        elif rsi < 30: sig['RSI (14)'] = (f'تشبع بيع ({rsi:.1f})', 1)
        else:          sig['RSI (14)'] = (f'محايد ({rsi:.1f})', 0)

    # ── MACD ──
    mh = float(last['MACD_H']) if pd.notna(last['MACD_H']) else None
    if mh is not None:
        sig['MACD'] = ('صاعد ▲', 1) if mh > 0 else ('هابط ▼', -1)

    # ── ROC 12 ──
    roc12 = float(last['ROC12']) if pd.notna(last['ROC12']) else None
    if roc12 is not None:
        if roc12 > 5:     sig['ROC 12'] = (f'زخم صاعد قوي ({roc12:.2f}%)', 1)
        elif roc12 > 0:   sig['ROC 12'] = (f'زخم صاعد ({roc12:.2f}%)', 1)
        elif roc12 > -5:  sig['ROC 12'] = (f'زخم هابط ({roc12:.2f}%)', -1)
        else:             sig['ROC 12'] = (f'زخم هابط قوي ({roc12:.2f}%)', -1)

    # ── ADX direction & strength ──
    adx = float(last['ADX']) if pd.notna(last['ADX']) else None
    pdi = float(last['PDI']) if pd.notna(last['PDI']) else None
    mdi_v = float(last['MDI']) if pd.notna(last['MDI']) else None
    if all(x is not None for x in [adx, pdi, mdi_v]):
        trend = 'قوي' if adx > 25 else 'ضعيف'
        if pdi > mdi_v: sig['اتجاه ADX'] = (f'اتجاه صاعد {trend} ({adx:.0f})', 1 if adx > 25 else 0)
        else:           sig['اتجاه ADX'] = (f'اتجاه هابط {trend} ({adx:.0f})', -1 if adx > 25 else 0)

    # ── Volume analysis ──
    vavg = d['Volume'].rolling(20, min_periods=1).mean().iloc[-1]
    vnow = float(last['Volume'])
    if pd.notna(vavg) and vavg > 0:
        vr = vnow / vavg
        vtxt = f'حجم مرتفع ({vr:.1f}x)' if vr > 1.5 else (f'حجم منخفض ({vr:.1f}x)' if vr < 0.5 else f'حجم طبيعي ({vr:.1f}x)')
        sig['الحجم'] = (vtxt, 0)

    signal_score = sum(val[1] for val in sig.values())
    return sig, signal_score


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6g: RECOMMENDATION & DECISION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Convert numeric score → text recommendation + color.
#
# SCORE RANGES:
#   16–20 → إيجابي + (Strong Positive) — GREEN
#   12–15 → إيجابي   (Positive)        — Light Green
#    9–11 → حياد      (Neutral)         — ORANGE
#    5–8  → سلبي      (Negative)        — Light Red
#    0–4  → سلبي +    (Strong Negative) — RED
#
# TO MODIFY:
#   • Change thresholds → modify the if/elif boundaries
#   • Change labels     → modify Arabic text
#   • Change colors     → modify HEX values
# ═══════════════════════════════════════════════════════════════════════════════

def recommendation(score):
    """Convert score (0-20) to (recommendation_text, color_hex)."""
    if score >= 16: return 'إيجابي +', GREEN_HEX
    elif score >= 12: return 'إيجابي', '#66BB6A'
    elif score <= 4: return 'سلبي +', RED_HEX
    elif score <= 8: return 'سلبي', '#EF5350'
    return 'حياد', ORANGE_HEX


def decision_text(v):
    """Convert signal direction (-1/0/1) to Arabic text."""
    return 'إيجابي' if v > 0 else 'سلبي' if v < 0 else 'حياد'


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6h: CANDLESTICK PATTERN DETECTION (detect_candle_patterns)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Detect Japanese candlestick patterns in price data.
#           Returns up to 8 most recent unique patterns.
#
# PATTERNS DETECTED (30+):
#   Single Candle:
#     Doji (Standard, Dragonfly, Gravestone, Long-Legged)
#     Hammer, Hanging Man, Inverted Hammer, Shooting Star
#     Marubozu (Bullish/Bearish), Spinning Top
#
#   Two Candles:
#     Bullish/Bearish Engulfing, Bullish/Bearish Harami
#     Tweezer Top/Bottom, Piercing Line, Dark Cloud Cover, On-Neck
#
#   Three+ Candles:
#     Morning/Evening Star, Morning/Evening Doji Star
#     Three White Soldiers, Three Black Crows
#     Three Inside Up/Down, Three Outside Up/Down
#     Abandoned Baby (Bull/Bear)
#     Rising/Falling Three Methods (5 candles)
#
# TO MODIFY:
#   • Add new pattern → add detection logic in the for loop
#   • Change max patterns → modify `if len(unique) == 8`
#   • Change lookback → modify `if len(df) < 4` check
# ═══════════════════════════════════════════════════════════════════════════════

def detect_candle_patterns(df):
    """Detect candlestick patterns. Returns list of (date, ar_name, en_name, bullish)."""
    patterns = []
    if len(df) < 4:
        return patterns

    o = df['Open'].values.astype(float)
    h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float)
    c = df['Close'].values.astype(float)
    dates = df.index

    # ── Helper functions ──
    def body(i): return abs(c[i] - o[i])
    def candle(i): return h[i] - l[i]
    def upper_shadow(i): return h[i] - max(c[i], o[i])
    def lower_shadow(i): return min(c[i], o[i]) - l[i]
    def is_bull(i): return c[i] > o[i]
    def is_bear(i): return c[i] < o[i]
    def mid_body(i): return (o[i] + c[i]) / 2.0
    def body_top(i): return max(o[i], c[i])
    def body_bot(i): return min(o[i], c[i])
    def is_small_body(i, avg): return body(i) < avg * 0.3
    def is_large_body(i, avg): return body(i) > avg * 0.8

    def in_downtrend(i, lookback=5):
        start = max(0, i - lookback)
        return c[i] < c[start] and sum(1 for j in range(start, i) if c[j] < o[j]) >= lookback * 0.6

    def in_uptrend(i, lookback=5):
        start = max(0, i - lookback)
        return c[i] > c[start] and sum(1 for j in range(start, i) if c[j] > o[j]) >= lookback * 0.6

    # ── Pattern detection loop ──
    for i in range(2, len(df)):
        avg_body = np.mean([body(j) for j in range(max(0, i - 10), i)]) or 1e-9
        date_lbl = dates[i].strftime('%Y-%m-%d')
        cr = candle(i)

        # ── SINGLE CANDLE PATTERNS ──

        # Doji variants
        if body(i) < 0.05 * cr and cr > 0:
            if lower_shadow(i) > 2 * body(i) and upper_shadow(i) < cr * 0.1:
                patterns.append((date_lbl, 'دوجي اليعسوب', 'Dragonfly Doji', True)); continue
            if upper_shadow(i) > 2 * body(i) and lower_shadow(i) < cr * 0.1:
                patterns.append((date_lbl, 'دوجي شاهد القبر', 'Gravestone Doji', False)); continue
            if upper_shadow(i) > cr * 0.3 and lower_shadow(i) > cr * 0.3:
                patterns.append((date_lbl, 'دوجي طويل الأرجل', 'Long-Legged Doji', None)); continue
            patterns.append((date_lbl, 'دوجي', 'Doji', None)); continue

        # Hammer / Hanging Man
        if (lower_shadow(i) > 2 * body(i) and upper_shadow(i) < 0.3 * body(i) and body(i) > 0):
            if in_downtrend(i):
                patterns.append((date_lbl, 'المطرقة', 'Hammer', True))
            else:
                patterns.append((date_lbl, 'المشنقة', 'Hanging Man', False))
            continue

        # Shooting Star / Inverted Hammer
        if (upper_shadow(i) > 2 * body(i) and lower_shadow(i) < 0.3 * body(i) and body(i) > 0):
            if in_downtrend(i):
                patterns.append((date_lbl, 'المطرقة المقلوبة', 'Inverted Hammer', True))
            else:
                patterns.append((date_lbl, 'النجمة الساقطة', 'Shooting Star', False))
            continue

        # Marubozu (no shadows)
        if (upper_shadow(i) < body(i) * 0.05 and lower_shadow(i) < body(i) * 0.05 and is_large_body(i, avg_body)):
            if is_bull(i):
                patterns.append((date_lbl, 'ماروبوزو صاعد', 'Bullish Marubozu', True))
            else:
                patterns.append((date_lbl, 'ماروبوزو هابط', 'Bearish Marubozu', False))
            continue

        # Spinning Top
        if (is_small_body(i, avg_body) and upper_shadow(i) > body(i) and lower_shadow(i) > body(i) and cr > avg_body * 0.5):
            patterns.append((date_lbl, 'القمة الدوارة', 'Spinning Top', None)); continue

        # ── TWO CANDLE PATTERNS ──

        # Bullish Engulfing
        if (is_bear(i - 1) and is_bull(i) and o[i] <= c[i - 1] and c[i] >= o[i - 1] and body(i) > body(i - 1)):
            patterns.append((date_lbl, 'الابتلاع الصعودي', 'Bullish Engulfing', True)); continue

        # Bearish Engulfing
        if (is_bull(i - 1) and is_bear(i) and o[i] >= c[i - 1] and c[i] <= o[i - 1] and body(i) > body(i - 1)):
            patterns.append((date_lbl, 'الابتلاع الهبوطي', 'Bearish Engulfing', False)); continue

        # Bullish Harami
        if (is_bear(i - 1) and is_bull(i) and is_large_body(i - 1, avg_body) and body_top(i) < body_top(i - 1) and body_bot(i) > body_bot(i - 1)):
            patterns.append((date_lbl, 'الحرامي الصعودي', 'Bullish Harami', True)); continue

        # Bearish Harami
        if (is_bull(i - 1) and is_bear(i) and is_large_body(i - 1, avg_body) and body_top(i) < body_top(i - 1) and body_bot(i) > body_bot(i - 1)):
            patterns.append((date_lbl, 'الحرامي الهبوطي', 'Bearish Harami', False)); continue

        # Tweezer Bottom
        if (is_bear(i - 1) and is_bull(i) and abs(l[i] - l[i - 1]) < avg_body * 0.1 and in_downtrend(i)):
            patterns.append((date_lbl, 'قاع الملقط', 'Tweezer Bottom', True)); continue

        # Tweezer Top
        if (is_bull(i - 1) and is_bear(i) and abs(h[i] - h[i - 1]) < avg_body * 0.1 and in_uptrend(i)):
            patterns.append((date_lbl, 'قمة الملقط', 'Tweezer Top', False)); continue

        # Piercing Line
        if (is_bear(i - 1) and is_bull(i) and o[i] < l[i - 1] and c[i] > mid_body(i - 1) and c[i] < o[i - 1]):
            patterns.append((date_lbl, 'خط الاختراق', 'Piercing Line', True)); continue

        # Dark Cloud Cover
        if (is_bull(i - 1) and is_bear(i) and o[i] > h[i - 1] and c[i] < mid_body(i - 1) and c[i] > o[i - 1]):
            patterns.append((date_lbl, 'الغطاء السحابي', 'Dark Cloud Cover', False)); continue

        # On-Neck
        if (is_bear(i - 1) and is_bull(i) and is_large_body(i - 1, avg_body) and o[i] < c[i - 1] and abs(c[i] - c[i - 1]) < avg_body * 0.1):
            patterns.append((date_lbl, 'على العنق', 'On-Neck', False)); continue

        # ── THREE+ CANDLE PATTERNS ──
        if i >= 2:
            # Morning Star
            if (is_bear(i - 2) and is_small_body(i - 1, avg_body) and is_bull(i) and c[i] > mid_body(i - 2) and body_top(i - 1) < body_bot(i - 2)):
                patterns.append((date_lbl, 'نجمة الصباح', 'Morning Star', True)); continue

            # Evening Star
            if (is_bull(i - 2) and is_small_body(i - 1, avg_body) and is_bear(i) and c[i] < mid_body(i - 2) and body_bot(i - 1) > body_top(i - 2)):
                patterns.append((date_lbl, 'نجمة المساء', 'Evening Star', False)); continue

            # Morning Doji Star
            if (is_bear(i - 2) and body(i - 1) < 0.05 * candle(i - 1) and candle(i - 1) > 0 and is_bull(i) and c[i] > mid_body(i - 2)):
                patterns.append((date_lbl, 'نجمة الصباح دوجي', 'Morning Doji Star', True)); continue

            # Evening Doji Star
            if (is_bull(i - 2) and body(i - 1) < 0.05 * candle(i - 1) and candle(i - 1) > 0 and is_bear(i) and c[i] < mid_body(i - 2)):
                patterns.append((date_lbl, 'نجمة المساء دوجي', 'Evening Doji Star', False)); continue

            # Three White Soldiers
            if (all(c[j] > o[j] for j in [i - 2, i - 1, i]) and c[i - 1] > c[i - 2] and c[i] > c[i - 1] and all(is_large_body(j, avg_body) for j in [i - 2, i - 1, i]) and o[i - 1] > o[i - 2] and o[i] > o[i - 1]):
                patterns.append((date_lbl, 'ثلاثة جنود بيض', 'Three White Soldiers', True)); continue

            # Three Black Crows
            if (all(c[j] < o[j] for j in [i - 2, i - 1, i]) and c[i - 1] < c[i - 2] and c[i] < c[i - 1] and all(is_large_body(j, avg_body) for j in [i - 2, i - 1, i]) and o[i - 1] < o[i - 2] and o[i] < o[i - 1]):
                patterns.append((date_lbl, 'ثلاثة غربان سوداء', 'Three Black Crows', False)); continue

            # Three Inside Up
            if (is_bear(i - 2) and is_bull(i - 1) and is_bull(i) and body_top(i - 1) < body_top(i - 2) and body_bot(i - 1) > body_bot(i - 2) and c[i] > body_top(i - 2)):
                patterns.append((date_lbl, 'ثلاثة من الداخل صاعد', 'Three Inside Up', True)); continue

            # Three Inside Down
            if (is_bull(i - 2) and is_bear(i - 1) and is_bear(i) and body_top(i - 1) < body_top(i - 2) and body_bot(i - 1) > body_bot(i - 2) and c[i] < body_bot(i - 2)):
                patterns.append((date_lbl, 'ثلاثة من الداخل هابط', 'Three Inside Down', False)); continue

            # Three Outside Up
            if (is_bear(i - 2) and is_bull(i - 1) and is_bull(i) and body(i - 1) > body(i - 2) and o[i - 1] <= c[i - 2] and c[i - 1] >= o[i - 2] and c[i] > c[i - 1]):
                patterns.append((date_lbl, 'ثلاثة من الخارج صاعد', 'Three Outside Up', True)); continue

            # Three Outside Down
            if (is_bull(i - 2) and is_bear(i - 1) and is_bear(i) and body(i - 1) > body(i - 2) and o[i - 1] >= c[i - 2] and c[i - 1] <= o[i - 2] and c[i] < c[i - 1]):
                patterns.append((date_lbl, 'ثلاثة من الخارج هابط', 'Three Outside Down', False)); continue

            # Abandoned Baby Bullish
            if (is_bear(i - 2) and body(i - 1) < 0.05 * candle(i - 1) and candle(i - 1) > 0 and is_bull(i) and h[i - 1] < l[i - 2] and h[i - 1] < l[i]):
                patterns.append((date_lbl, 'الطفل المتروك صاعد', 'Abandoned Baby Bull', True)); continue

            # Abandoned Baby Bearish
            if (is_bull(i - 2) and body(i - 1) < 0.05 * candle(i - 1) and candle(i - 1) > 0 and is_bear(i) and l[i - 1] > h[i - 2] and l[i - 1] > h[i]):
                patterns.append((date_lbl, 'الطفل المتروك هابط', 'Abandoned Baby Bear', False)); continue

            # ── FIVE CANDLE PATTERNS ──
            if i >= 4:
                # Rising Three Methods
                if (is_bull(i - 4) and is_large_body(i - 4, avg_body) and all(c[j] < o[j] and body(j) < body(i - 4) for j in [i - 3, i - 2, i - 1]) and all(l[j] > l[i - 4] for j in [i - 3, i - 2, i - 1]) and is_bull(i) and c[i] > c[i - 4] and is_large_body(i, avg_body)):
                    patterns.append((date_lbl, 'ثلاث طرق صاعدة', 'Rising Three Methods', True)); continue

                # Falling Three Methods
                if (is_bear(i - 4) and is_large_body(i - 4, avg_body) and all(c[j] > o[j] and body(j) < body(i - 4) for j in [i - 3, i - 2, i - 1]) and all(h[j] < h[i - 4] for j in [i - 3, i - 2, i - 1]) and is_bear(i) and c[i] < c[i - 4] and is_large_body(i, avg_body)):
                    patterns.append((date_lbl, 'ثلاث طرق هابطة', 'Falling Three Methods', False)); continue

    # ── Keep only last 8 unique patterns ──
    seen = set(); unique = []
    for p in reversed(patterns):
        if p[2] not in seen:
            seen.add(p[2]); unique.append(p)
        if len(unique) == 8: break
    return list(reversed(unique))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6i: DIVERGENCE DETECTION (detect_divergences)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Detect bullish/bearish divergences between price and RSI/MACD.
#
# DIVERGENCE TYPES:
#   Bullish: Price makes lower low, but indicator makes higher low
#   Bearish: Price makes higher high, but indicator makes lower high
#
# TO MODIFY:
#   • Add OBV divergence → duplicate RSI logic with OBV values
#   • Change lookback    → modify df.tail(60)
#   • Change sensitivity → modify n=5 in find_local_min/max
# ═══════════════════════════════════════════════════════════════════════════════

def detect_divergences(d):
    """Detect RSI and MACD divergences. Returns list of (indicator, ar_type, en_type)."""
    divergences = []
    df = d.tail(60).copy()
    if len(df) < 20: return divergences
    c = df['Close'].values; rsi = df['RSI'].values; mcd = df['MACD'].values

    def find_local_min_idx(arr, n=5):
        return [i for i in range(n, len(arr)-n) if arr[i] == min(arr[i-n:i+n+1])]

    def find_local_max_idx(arr, n=5):
        return [i for i in range(n, len(arr)-n) if arr[i] == max(arr[i-n:i+n+1])]

    # ── RSI Divergences ──
    if not (np.all(np.isnan(rsi)) or np.sum(~np.isnan(rsi)) < 15):
        price_lows = find_local_min_idx(c); rsi_lows = find_local_min_idx(rsi)
        price_highs = find_local_max_idx(c); rsi_highs = find_local_max_idx(rsi)
        # Bullish divergence
        if len(price_lows)>=2 and len(rsi_lows)>=2:
            p1,p2 = price_lows[-2],price_lows[-1]; r1,r2 = rsi_lows[-2],rsi_lows[-1]
            if c[p2]<c[p1] and rsi[r2]>rsi[r1]:
                divergences.append(('RSI','تباعد إيجابي (صعودي)','Bullish Divergence'))
        # Bearish divergence
        if len(price_highs)>=2 and len(rsi_highs)>=2:
            p1,p2 = price_highs[-2],price_highs[-1]; r1,r2 = rsi_highs[-2],rsi_highs[-1]
            if c[p2]>c[p1] and rsi[r2]<rsi[r1]:
                divergences.append(('RSI','تباعد سلبي (هبوطي)','Bearish Divergence'))

    # ── MACD Divergences ──
    if not (np.all(np.isnan(mcd)) or np.sum(~np.isnan(mcd)) < 15):
        price_lows2 = find_local_min_idx(c); macd_lows = find_local_min_idx(mcd)
        price_highs2 = find_local_max_idx(c); macd_highs = find_local_max_idx(mcd)
        if len(price_lows2)>=2 and len(macd_lows)>=2:
            p1,p2 = price_lows2[-2],price_lows2[-1]; m1,m2 = macd_lows[-2],macd_lows[-1]
            if c[p2]<c[p1] and mcd[m2]>mcd[m1]:
                divergences.append(('MACD','تباعد إيجابي (صعودي)','Bullish Divergence'))
        if len(price_highs2)>=2 and len(macd_highs)>=2:
            p1,p2 = price_highs2[-2],price_highs2[-1]; m1,m2 = macd_highs[-2],macd_highs[-1]
            if c[p2]>c[p1] and mcd[m2]<mcd[m1]:
                divergences.append(('MACD','تباعد سلبي (هبوطي)','Bearish Divergence'))

    return divergences


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6j: SUPPORT & RESISTANCE DETECTION (find_sr)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Find support/resistance levels from price data using multiple methods:
#   1. Local extrema at multiple orders (3,5,7,10)
#   2. Zigzag pivots at multiple thresholds (3%,5%,7%,10%)
#   3. High-volume price clusters
#   4. EMA levels (20,50,100,200)
#   5. Clustering similar prices to avoid duplicates
#
# RETURNS: (support_list_desc, resistance_list_asc) — up to n_levels each
#
# TO MODIFY:
#   • More levels         → increase n_levels parameter
#   • Change clustering   → modify 0.008 tolerance
#   • Add Fibonacci       → add fib level calculation
# ═══════════════════════════════════════════════════════════════════════════════

def find_sr(df, d_ind=None, n_levels=8):
    """Find support and resistance levels. Returns (supports_desc, resistances_asc)."""
    h_arr = df['High'].values.astype(float); l_arr = df['Low'].values.astype(float)
    c_arr = df['Close'].values.astype(float); v_arr = df['Volume'].values.astype(float)
    cur = float(c_arr[-1])
    raw_sup = []; raw_res = []

    # ── Method 1: Local extrema at multiple orders ──
    for order in (3,5,7,10):
        w = 1.0 + order*0.15  # Higher order = higher weight
        ph = argrelextrema(h_arr, np.greater_equal, order=order)[0]
        pl = argrelextrema(l_arr, np.less_equal, order=order)[0]
        for i in ph: raw_res.append((round(h_arr[i],4), w))
        for i in pl: raw_sup.append((round(l_arr[i],4), w))

    # ── Method 2: Zigzag pivots ──
    for pct in (0.03,0.05,0.07,0.10):
        pivot_lo = c_arr[0]; pivot_hi = c_arr[0]; direction = None
        for i in range(1, len(c_arr)):
            v = c_arr[i]
            if direction is None:
                if v > pivot_hi*(1+pct): direction='up'; pivot_hi=v
                elif v < pivot_lo*(1-pct): direction='down'; pivot_lo=v
            elif direction=='up':
                if v > pivot_hi: pivot_hi=v
                elif v < pivot_hi*(1-pct): raw_res.append((round(pivot_hi,4),1.8)); direction='down'; pivot_lo=v; pivot_hi=v
            elif direction=='down':
                if v < pivot_lo: pivot_lo=v
                elif v > pivot_lo*(1+pct): raw_sup.append((round(pivot_lo,4),1.8)); direction='up'; pivot_hi=v; pivot_lo=v

    # ── Method 3: High-volume price clusters ──
    vol_avg = np.mean(v_arr) if len(v_arr) > 0 else 1.0
    for i in range(len(v_arr)):
        if v_arr[i] > 1.5*vol_avg:
            price = round(c_arr[i],4); bar_mid = (h_arr[i]+l_arr[i])/2
            if price >= bar_mid: raw_res.append((price,1.0))
            else:                raw_sup.append((price,1.0))

    # ── Method 4: EMA levels ──
    if d_ind is not None:
        for ema_col, ema_w in [('EMA20',2.0),('EMA50',2.5),('EMA100',2.5),('EMA200',3.0)]:
            if ema_col in d_ind.columns:
                val = d_ind[ema_col].iloc[-1]
                if pd.notna(val):
                    ev = round(float(val),4)
                    if ev < cur: raw_sup.append((ev,ema_w))
                    elif ev > cur: raw_res.append((ev,ema_w))

    # ── Cluster and score nearby prices ──
    def cluster_score(items):
        if not items: return []
        items_s = sorted(items, key=lambda x: x[0])
        groups = [([items_s[0][0]], [items_s[0][1]])]
        for price, w in items_s[1:]:
            ref = np.mean(groups[-1][0])
            if abs(price-ref)/max(abs(ref),1e-9) <= 0.008:
                groups[-1][0].append(price); groups[-1][1].append(w)
            else:
                groups.append(([price],[w]))
        return [(round(np.mean(gp),2), sum(gw)) for gp,gw in groups]

    sup_scored = cluster_score(raw_sup); res_scored = cluster_score(raw_res)
    sup_below = [(p,s) for p,s in sup_scored if p < cur]
    res_above = [(p,s) for p,s in res_scored if p > cur]
    sup_top = sorted(sup_below, key=lambda x: -x[1])[:n_levels]
    res_top = sorted(res_above, key=lambda x: -x[1])[:n_levels]
    sup_final = sorted([p for p,_ in sup_top], reverse=True)
    res_final = sorted([p for p,_ in res_top])
    return sup_final, res_final


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6k: TECHNICAL REVIEW TEXT GENERATOR (gen_technical_review)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Generate multi-paragraph Arabic technical review for PDF report.
#           Each section analyzes a different aspect of the stock.
#
# SECTIONS GENERATED:
#   1. الاتجاه العام (General Trend) — EMA order, golden/death cross, ADX
#   2. مؤشرات الزخم (Momentum) — RSI, MACD, ROC
#   3. التذبذب ونطاق بولنجر (Volatility) — Bollinger Bands, ATR
#   4. تحليل الحجم (Volume) — Volume ratio, OBV, VWAP, bull/bear
#   5. مؤشرات متقدمة (Advanced) — Ichimoku, SAR
#   6. مستويات الدعم والمقاومة (S/R) — Support, resistance, ATR targets
#   7. الخلاصة الفنية (Summary) — Overall recommendation based on score
#   8. التباعد (Divergences) — RSI/MACD divergences if found
#   9. نطاق 52 أسبوع (52-Week Range) — Position within annual range
#   10. CCI وWilliams %R (Oscillators) — Additional oscillator readings
#
# TO MODIFY:
#   • Change Arabic text → edit the string literals
#   • Add new section   → append to `sections` list
#   • Change thresholds → modify the if/elif values
# ═══════════════════════════════════════════════════════════════════════════════

def gen_technical_review(d, sig, score, sup, res, info=None, patterns=None, divergences=None):
    """Generate Arabic technical review sections. Returns list of (title, paragraph)."""
    last = d.iloc[-1]
    c_price = float(last['Close'])
    sections = []

    # ── Extract indicator values ──
    ema7   = float(last['EMA7'])   if pd.notna(last['EMA7'])   else None
    ema20  = float(last['EMA20'])  if pd.notna(last['EMA20'])  else None
    ema50  = float(last['EMA50'])  if pd.notna(last['EMA50'])  else None
    ema200 = float(last['EMA200']) if pd.notna(last['EMA200']) else None
    sma50  = float(last['SMA50'])  if pd.notna(last['SMA50'])  else None
    sma200 = float(last['SMA200']) if pd.notna(last['SMA200']) else None
    adx    = float(last['ADX'])    if pd.notna(last['ADX'])    else None
    pdi    = float(last['PDI'])    if pd.notna(last['PDI'])    else None
    mdi    = float(last['MDI'])    if pd.notna(last['MDI'])    else None

    # ── 1. TREND ANALYSIS (الاتجاه العام) ──
    trend_parts = []
    if all(v is not None for v in [ema7, ema20, ema50, ema200]):
        if ema7 > ema20 > ema50 > ema200:
            trend_parts.append('المتوسطات المتحركة الأسية (7/20/50/200) مرتبة ترتيباً صاعداً مثالياً، مما يدل على اتجاه تصاعدي قوي ومنتظم.')
        elif c_price > ema50 and c_price > ema200:
            trend_parts.append('السعر يتداول فوق المتوسطَين الأسيَين 50 و200، مما يشير إلى استمرار الاتجاه الصعودي على المدى المتوسط والطويل.')
        elif c_price < ema50 and c_price < ema200:
            trend_parts.append('السعر يتداول دون المتوسطَين الأسيَين 50 و200، مما يعكس ضغطاً بيعياً سائداً على المدى المتوسط والطويل.')
        else:
            trend_parts.append('يتباين موقع السعر بالنسبة للمتوسطات المتحركة، مما يشير إلى اتجاه متذبذب يستدعي المراقبة.')
    if sma50 is not None and sma200 is not None:
        if sma50 > sma200:
            trend_parts.append('يُسجَّل تقاطع ذهبي بين المتوسطَين البسيطَين 50 و200، وهو إشارة فنية إيجابية تاريخياً.')
        else:
            trend_parts.append('يُلاحَظ تقاطع سلبي (موت) بين المتوسطَين البسيطَين 50 و200، وهو إشارة تحذيرية للمستثمرين.')
    if adx is not None and pdi is not None and mdi is not None:
        strength  = 'قوي' if adx > 25 else ('معتدل' if adx > 15 else 'ضعيف')
        direction = 'صاعد' if pdi > mdi else 'هابط'
        trend_parts.append(f'يُشير مؤشر ADX إلى اتجاه {direction} {strength} بقيمة {adx:.1f}.')
    sections.append(('الاتجاه العام', ' '.join(trend_parts) if trend_parts else 'لا تتوفر بيانات كافية لتحليل الاتجاه.'))

    # ── 2. MOMENTUM (مؤشرات الزخم) ──
    rsi      = float(last['RSI'])      if pd.notna(last['RSI'])      else None
    macd     = float(last['MACD'])     if pd.notna(last['MACD'])     else None
    macd_sig = float(last['MACD_Sig']) if pd.notna(last['MACD_Sig']) else None
    macd_h   = float(last['MACD_H'])   if pd.notna(last['MACD_H'])   else None
    roc12    = float(last['ROC12'])    if pd.notna(last['ROC12'])    else None

    mom_parts = []
    if rsi is not None:
        if rsi >= 70:   mom_parts.append(f'مؤشر القوة النسبية RSI يقرأ {rsi:.1f} في منطقة التشبع الشرائي، مما يستوجب الحذر من تصحيح وشيك.')
        elif rsi <= 30: mom_parts.append(f'مؤشر RSI عند {rsi:.1f} في منطقة التشبع البيعي، مما يستدعي مراقبة إشارات الانعكاس المحتملة.')
        elif rsi >= 55: mom_parts.append(f'مؤشر RSI عند {rsi:.1f} في المنطقة الإيجابية فوق خط 50، مما يعكس زخماً شرائياً متواصلاً.')
        elif rsi <= 45: mom_parts.append(f'مؤشر RSI عند {rsi:.1f} في المنطقة السلبية دون خط 50، مما يعكس ضعفاً في الزخم الشرائي.')
        else:           mom_parts.append(f'مؤشر RSI محايد عند {rsi:.1f} قريباً من الخط 50.')
    if macd is not None and macd_sig is not None and macd_h is not None:
        if macd > macd_sig and macd_h > 0:
            mom_parts.append('مؤشر الماكد MACD يتداول فوق خط الإشارة مع هستوجرام إيجابي، مما يدعم الزخم الصعودي.')
        elif macd < macd_sig and macd_h < 0:
            mom_parts.append('مؤشر الماكد يتداول دون خط الإشارة مع هستوجرام سلبي، مما يعكس ضعف الزخم الحالي.')
        else:
            mom_parts.append('مؤشر الماكد في مرحلة تقاطع، مما قد يُنذر بتغيير في الاتجاه.')
    if roc12 is not None:
        if roc12 > 5:     mom_parts.append(f'مؤشر ROC 12 عند {roc12:.2f}% يُشير إلى زخم صاعد قوي على المدى المتوسط.')
        elif roc12 > 0:   mom_parts.append(f'مؤشر ROC 12 عند {roc12:.2f}% إيجابي، مما يدل على استمرار الزخم الشرائي.')
        elif roc12 > -5:  mom_parts.append(f'مؤشر ROC 12 عند {roc12:.2f}% سلبي، مما يعكس ضعفاً في الزخم الحالي.')
        else:             mom_parts.append(f'مؤشر ROC 12 عند {roc12:.2f}% يُشير إلى زخم هبوطي قوي على المدى المتوسط.')
    sections.append(('مؤشرات الزخم', ' '.join(mom_parts) if mom_parts else 'لا تتوفر بيانات كافية.'))

    # ── 3. VOLATILITY & BOLLINGER (التذبذب ونطاق بولنجر) ──
    bb_u = float(last['BB_U']) if pd.notna(last['BB_U']) else None
    bb_m = float(last['BB_M']) if pd.notna(last['BB_M']) else None
    bb_l = float(last['BB_L']) if pd.notna(last['BB_L']) else None
    bb_p = float(last['BB_P']) if pd.notna(last['BB_P']) else None
    atr  = float(last['ATR'])  if pd.notna(last['ATR'])  else None

    vol_parts = []
    if all(v is not None for v in [bb_u, bb_m, bb_l, bb_p]):
        bb_width = (bb_u - bb_l) / bb_m * 100 if bb_m != 0 else 0
        if c_price > bb_u:
            vol_parts.append(f'السعر اخترق الحد العلوي لنطاق بولنجر ({bb_u:.2f})، مما يشير إلى زخم صعودي قوي مع احتمال توقف مؤقت.')
        elif c_price < bb_l:
            vol_parts.append(f'السعر دون الحد السفلي لنطاق بولنجر ({bb_l:.2f})، مما يدل على زخم هبوطي حاد مع احتمال توقف مؤقت.')
        else:
            pos_pct = bb_p * 100
            vol_parts.append(f'السعر داخل نطاق بولنجر عند الموقع {pos_pct:.0f}% بين الحدين، والحد الأوسط عند {bb_m:.2f}.')
        if bb_width < 5:
            vol_parts.append('تضيّق نطاق بولنجر يوحي بتراكم طاقة قد يتبعه تحرك حاد في أحد الاتجاهين.')
        elif bb_width > 20:
            vol_parts.append('اتساع نطاق بولنجر يعكس تذبذباً مرتفعاً في السوق.')
    if atr is not None:
        atr_pct = (atr / c_price) * 100
        if atr_pct > 3:    vol_parts.append(f'مؤشر ATR يُسجّل {atr:.2f} ({atr_pct:.1f}% من السعر)، مما يُشير إلى تذبذب يومي مرتفع.')
        elif atr_pct < 1:  vol_parts.append(f'مؤشر ATR عند {atr:.2f} ({atr_pct:.1f}% من السعر)، مما يعكس هدوءاً نسبياً في حركة السعر.')
        else:              vol_parts.append(f'مؤشر ATR عند {atr:.2f} ({atr_pct:.1f}% من السعر) يُشير إلى تذبذب يومي معتدل.')
    sections.append(('التذبذب ونطاق بولنجر', ' '.join(vol_parts) if vol_parts else 'لا تتوفر بيانات كافية.'))

    # ── 4. VOLUME ANALYSIS (تحليل الحجم) ──
    obv      = float(last['OBV'])         if pd.notna(last['OBV'])         else None
    bull_vol = float(last['Bull_Volume']) if pd.notna(last['Bull_Volume']) else None
    bear_vol = float(last['Bear_Volume']) if pd.notna(last['Bear_Volume']) else None
    vol_now  = float(last['Volume'])
    vwap     = float(last['VWAP'])        if pd.notna(last['VWAP'])        else None
    vol_avg  = d['Volume'].rolling(20, min_periods=1).mean().iloc[-1]

    volume_parts = []
    if pd.notna(vol_avg) and vol_avg > 0:
        vr = vol_now / vol_avg
        if vr > 1.5:   volume_parts.append(f'حجم التداول الحالي مرتفع بنسبة {vr:.1f}x فوق متوسط الـ20 يوماً، مما يعكس اهتماماً كبيراً من المشاركين في السوق.')
        elif vr < 0.5: volume_parts.append(f'حجم التداول منخفض ({vr:.1f}x من المتوسط)، مما يُشير إلى تراجع في النشاط التداولي وانخفاض الاهتمام.')
        else:          volume_parts.append(f'حجم التداول طبيعي عند {vr:.1f}x من المتوسط الـ20 يوماً.')
    if bull_vol is not None and bear_vol is not None and (bull_vol + bear_vol) > 0:
        bull_pct = bull_vol / (bull_vol + bear_vol) * 100
        if bull_pct > 55:   volume_parts.append(f'السيولة الشرائية تهيمن بنسبة {bull_pct:.0f}% من إجمالي الحجم الأخير، مؤشر إيجابي على الطلب الفعّال.')
        elif bull_pct < 45: volume_parts.append(f'السيولة على الجانب الهابط تهيمن بنسبة {100-bull_pct:.0f}%، مما يعكس ضغطاً على الورقة المالية.')
        else:               volume_parts.append(f'توازن نسبي بين السيولة الشرائية ({bull_pct:.0f}%) والبيعية ({100-bull_pct:.0f}%).')
    if vwap is not None:
        if c_price > vwap: volume_parts.append(f'مؤشر VWAP يُسجَّل عند {vwap:.2f} والسعر يتداول فوقه، مما يُشير إلى هيمنة المشترين خلال جلسات التداول.')
        else:              volume_parts.append(f'مؤشر VWAP يُسجَّل عند {vwap:.2f} والسعر يتداول دونه، مما يُشير إلى هيمنة البائعين خلال جلسات التداول.')
    if len(d) >= 5 and obv is not None:
        obv_prev = float(d.iloc[-5]['OBV']) if pd.notna(d.iloc[-5]['OBV']) else None
        if obv_prev is not None:
            if obv > obv_prev: volume_parts.append('OBV: المؤشر في اتجاه تصاعدي خلال الأسبوع الأخير، مما يدعم صحة الحركة الصعودية.')
            else:              volume_parts.append('OBV: المؤشر في اتجاه تراجعي خلال الأسبوع الأخير، مما يُلمح إلى ضعف خفي رغم الحركة السعرية.')
    sections.append(('تحليل الحجم', ' '.join(volume_parts) if volume_parts else 'لا تتوفر بيانات كافية.'))

    # ── 5. ADVANCED INDICATORS — Ichimoku & SAR ──
    tenkan = float(last['Tenkan'])   if pd.notna(last['Tenkan'])   else None
    kijun  = float(last['Kijun'])    if pd.notna(last['Kijun'])    else None
    senk_a = float(last['Senkou_A']) if pd.notna(last['Senkou_A']) else None
    senk_b = float(last['Senkou_B']) if pd.notna(last['Senkou_B']) else None
    sar    = float(last['SAR'])      if pd.notna(last['SAR'])      else None

    adv_parts = []
    if tenkan is not None and kijun is not None:
        if tenkan > kijun: adv_parts.append(f'خط التنكن ({tenkan:.2f}) أعلى من خط الكيجن ({kijun:.2f})، إشارة إيجابية في نظام الإيشيموكو.')
        else:              adv_parts.append(f'خط التنكن ({tenkan:.2f}) أدنى من خط الكيجن ({kijun:.2f})، إشارة سلبية في نظام الإيشيموكو.')
    if senk_a is not None and senk_b is not None:
        kumo_top = max(senk_a, senk_b); kumo_bot = min(senk_a, senk_b)
        if c_price > kumo_top:   adv_parts.append(f'السعر يتداول فوق السحابة السميكة (كوموه) عند {kumo_top:.2f}، مما يُعزز الاتجاه الصعودي.')
        elif c_price < kumo_bot: adv_parts.append(f'السعر دون السحابة (كوموه) عند {kumo_bot:.2f}، مما يُعزز الاتجاه الهبوطي.')
        else:                    adv_parts.append('السعر داخل السحابة (كوموه)، مما يدل على مرحلة تردد وعدم حسم في الاتجاه.')
    if sar is not None:
        if c_price > sar: adv_parts.append(f'مؤشر بارابوليك SAR عند {sar:.2f} يقع دون السعر الحالي، مما يدعم استمرار الاتجاه الصعودي.')
        else:             adv_parts.append(f'مؤشر بارابوليك SAR عند {sar:.2f} يعلو السعر الحالي، مما يشير إلى ضعف الزخم الصعودي.')
    sections.append(('مؤشرات متقدمة (إيشيموكو / SAR)', ' '.join(adv_parts) if adv_parts else 'لا تتوفر بيانات كافية.'))

    # ── 6. SUPPORT & RESISTANCE LEVELS + ATR TARGETS ──
    atr_val = float(last['ATR']) if pd.notna(last['ATR']) else None
    sr_parts = []
    if sup:
        for i, s in enumerate(sup[:5]):
            gap  = abs(c_price - s) / c_price * 100
            side = 'دون' if s < c_price else 'فوق'
            rank = ['الأول', 'الثاني', 'الثالث', 'الرابع', 'الخامس'][i]
            sr_parts.append(f'الدعم {rank}: {s:.2f} ({gap:.1f}% {side} السعر الحالي).')
    if res:
        for i, r in enumerate(res[:5]):
            gap  = abs(r - c_price) / c_price * 100
            side = 'فوق' if r > c_price else 'دون'
            rank = ['الأولى', 'الثانية', 'الثالثة', 'الرابعة', 'الخامسة'][i]
            sr_parts.append(f'المقاومة {rank}: {r:.2f} ({gap:.1f}% {side} السعر الحالي).')

    # Volume-based S/R
    v_arr   = d['Volume'].values.astype(float)
    vol_avg2 = np.mean(v_arr) if len(v_arr) > 0 else 1
    hv_closes = [float(d.iloc[i]['Close']) for i in np.where(v_arr > 1.5 * vol_avg2)[0]]
    hv_res = sorted([p for p in hv_closes if p > c_price])
    hv_sup = sorted([p for p in hv_closes if p <= c_price], reverse=True)
    if hv_sup: sr_parts.append(f'دعم حجمي بارز عند {hv_sup[0]:.2f} — ناتج عن تداولات كثيفة سابقة.')
    if hv_res: sr_parts.append(f'مقاومة حجمية بارزة عند {hv_res[0]:.2f} — ناتجة عن تداولات كثيفة سابقة.')

    # ATR-based targets
    if atr_val is not None:
        t1_up = c_price + 1.0 * atr_val; t2_up = c_price + 2.0 * atr_val
        t1_dn = c_price - 1.0 * atr_val; t2_dn = c_price - 2.0 * atr_val
        sr_parts.append(f'ATR ({atr_val:.2f}): الأهداف الصعودية {t1_up:.2f} و {t2_up:.2f}. مستويات التوقف {t1_dn:.2f} و {t2_dn:.2f}.')

    # Risk/Reward ratio
    nearest_s = sup[0] if sup and sup[0] < c_price else None
    nearest_r = res[0] if res and res[0] > c_price else None
    if nearest_s and nearest_r and (c_price - nearest_s) > 0:
        rr = (nearest_r - c_price) / (c_price - nearest_s)
        sr_parts.append(f'نسبة العائد إلى المخاطرة (R:R): {rr:.1f}x بين أقرب دعم ومقاومة.')
    if not sr_parts:
        sr_parts.append('لم يتم رصد مستويات دعم أو مقاومة واضحة في الفترة المحللة.')
    sections.append(('مستويات الدعم والمقاومة والأهداف السعرية', ' '.join(sr_parts)))

    # ── 7. OVERALL SUMMARY (الخلاصة الفنية) ──
    rec_txt, _ = recommendation(score)
    if score >= 16:
        outlook = (f'الصورة الفنية الشاملة إيجابية بدرجة عالية وفق نتيجة {score}/20، '
                   f'إذ تتوافق معظم المؤشرات التقنية المدروسة مع الاتجاه الصعودي على مختلف الأطر الزمنية. '
                   f'مستويات الدعم والمقاومة المرصودة تمثل نقاط مراقبة فنية مهمة.')
    elif score >= 12:
        outlook = (f'تميل القراءة الفنية الإجمالية بنتيجة {score}/20 نحو الإيجابية، '
                   f'وتدعم غالبية المؤشرات الاتجاه الصعودي مع وجود بعض الإشارات المتحفظة '
                   f'التي تستدعي متابعة مستويات الدعم والمقاومة المذكورة.')
    elif score <= 4:
        outlook = (f'الصورة الفنية الشاملة سلبية وفق نتيجة {score}/20، '
                   f'إذ تُشير معظم المؤشرات التقنية إلى ضغط هبوطي مستمر. '
                   f'مراقبة إشارات الانعكاس عند مستويات الدعم المرصودة أمر جوهري في هذه المرحلة.')
    elif score <= 8:
        outlook = (f'تميل القراءة الفنية بنتيجة {score}/20 نحو السلبية، '
                   f'ويُلاحَظ تراجع ملموس في قوة الاتجاه الصعودي وفق معظم المؤشرات. '
                   f'متابعة الأطر الزمنية المختلفة ومستويات الدعم الرئيسية يساعد في تقييم صحة الاتجاه الحالي.')
    else:
        outlook = (f'تُعطي المؤشرات الفنية مجتمعةً قراءة محايدة بنتيجة {score}/20، '
                   f'ولا يوجد اتجاه صعودي أو هبوطي راسخ في الوقت الراهن. '
                   f'متابعة كسر مستويات الدعم أو المقاومة القريبة سيُحدد مسار الاتجاه القادم.')
    sections.append(('الخلاصة الفنية', outlook))

    # ── 8. DIVERGENCES ──
    if divergences:
        div_parts = [f'{ind}: {ar_type} — يُشير إلى احتمال تغيير في الزخم الحالي.' for ind, ar_type, en_type in divergences]
        sections.append(('التباعد بين السعر والمؤشرات', ' '.join(div_parts)))
    else:
        sections.append(('التباعد بين السعر والمؤشرات',
                          'لا يُلاحَظ تباعد واضح بين السعر ومؤشري RSI وMACD في الفترة الأخيرة، مما يُشير إلى انسجام الزخم مع حركة السعر.'))

    # ── 9. 52-WEEK RANGE POSITION ──
    w52_high = info.get('fiftyTwoWeekHigh') if info else None
    w52_low  = info.get('fiftyTwoWeekLow')  if info else None
    if w52_high and w52_low and float(w52_high) != float(w52_low):
        pos_pct   = (c_price - float(w52_low)) / (float(w52_high) - float(w52_low)) * 100
        dist_high = (float(w52_high) - c_price) / c_price * 100
        dist_low  = (c_price - float(w52_low))  / c_price * 100
        sections.append(('الموقع من نطاق 52 أسبوع',
                          f'السعر الحالي يقع عند {pos_pct:.1f}% من النطاق السنوي '
                          f'(أدنى 52 أسبوع: {float(w52_low):.2f} — أعلى 52 أسبوع: {float(w52_high):.2f}). '
                          f'المسافة من القمة: {dist_high:.1f}%. المسافة من القاع: {dist_low:.1f}%.'))

    # ── 10. CCI & WILLIAMS %R OSCILLATORS ──
    cci_val   = float(last['CCI'])   if pd.notna(last['CCI'])   else None
    willr     = float(last['WILLR']) if pd.notna(last['WILLR']) else None
    osc_parts = []
    if cci_val is not None:
        if cci_val > 100:   osc_parts.append(f'CCI: القراءة {cci_val:.0f} فوق مستوى +100 تُشير إلى زخم صعودي قوي مع احتمال تشبع.')
        elif cci_val < -100: osc_parts.append(f'CCI: القراءة {cci_val:.0f} دون مستوى -100 تُشير إلى زخم هبوطي مع احتمال تشبع بيعي.')
        else:               osc_parts.append(f'CCI: القراءة {cci_val:.0f} داخل النطاق المحايد بين -100 و+100.')
    if willr is not None:
        if willr > -20:   osc_parts.append(f'Williams %R: القراءة {willr:.0f} قريبة من منطقة التشبع الشرائي (فوق -20).')
        elif willr < -80: osc_parts.append(f'Williams %R: القراءة {willr:.0f} في منطقة التشبع البيعي (تحت -80).')
        else:             osc_parts.append(f'Williams %R: القراءة {willr:.0f} في المنطقة المحايدة.')
    if osc_parts:
        sections.append(('مؤشرات CCI وWilliams %R', ' '.join(osc_parts)))

    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: CHART GENERATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Generate all chart images for the PDF report.
#           Each function returns a BytesIO PNG buffer.
#
# CHARTS IN THIS SECTION:
#   _draw_sr_lines()          — Draw support/resistance lines + EMA overlays
#   _get_pivots()             — Find pivot highs/lows for S/R markers
#   make_main_chart()         — Candlestick + S/R + EMA lines (main page)
#   make_price_chart()        — Candlestick + SMA overlays
#   make_ema_chart()          — Candlestick + EMA overlays
#   make_tech_chart()         — RSI + MACD + ROC 12 (3-panel)
#   make_bb_chart()           — Bollinger Bands overlay
#   make_dd_chart()           — Drawdown curve
#   make_volume_chart()       — Volume bars + 20-day average
#   make_gauge_chart()        — Score gauge (0–20) + recommendation circle
#   make_ichimoku_chart()     — Ichimoku Cloud with all components
#   make_candle_pattern_chart() — Candlestick with pattern annotations
#   make_cci_willr_chart()    — CCI + Williams %R (2-panel)
#   make_alligator_chart()    — Williams Alligator overlay
#   make_supertrend_chart()   — Supertrend overlay
#
# TO MODIFY:
#   • Change chart colors     → modify HEX color strings in each function
#   • Change chart size       → modify figsize=(width, height)
#   • Change candlestick style → modify mpf.make_marketcolors()
#   • Add new chart           → create new make_xxx_chart() function
#   • Change lookback period  → modify d.tail(180) to d.tail(N)
# ═══════════════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────────────
# 7a: SUPPORT/RESISTANCE LINE DRAWER (_draw_sr_lines)
# ───────────────────────────────────────────────────────────────────────────────
#
# PURPOSE : Draw horizontal S/R lines + EMA level lines on a chart axis.
#           Labels are placed in the right margin with arrows pointing to
#           the price level. Collision detection avoids overlapping labels.
#
# TO MODIFY:
#   • Change label font size → modify fsize in items.append()
#   • Change label position  → modify near_frac / far_frac
#   • Change S/R colors      → modify sup_color / res_color
#   • Change EMA colors      → modify ema_colors dict
# ───────────────────────────────────────────────────────────────────────────────

def _draw_sr_lines(ax, sup, res, xmax, d_ind=None, pivots=None):
    """Draw support/resistance lines with labeled arrows on chart axis."""
    sup_color = '#1565C0'   # Blue for support
    res_color = '#B71C1C'   # Red for resistance

    # EMA overlay colors and labels
    ema_colors = {
        'EMA20':  ('#2196F3', 'EMA 20'),
        'EMA50':  ('#E53935', 'EMA 50'),
        'EMA100': ('#C620F8', 'EMA 100'),
        'EMA200': ('#000000', 'EMA 200'),
    }

    # ── Draw pivot markers (small triangles) ──
    if pivots:
        for bx, by in pivots.get('highs', []):
            ax.plot(bx, by, 'v', color='#B71C1C', markersize=5,
                    alpha=0.75, zorder=6, markeredgewidth=0)
        for bx, by in pivots.get('lows', []):
            ax.plot(bx, by, '^', color='#1565C0', markersize=5,
                    alpha=0.75, zorder=6, markeredgewidth=0)

    # ── Collect ALL labels (EMA + Support + Resistance) ──
    items = []

    # EMA horizontal lines
    if d_ind is not None:
        for col, (clr, lbl) in ema_colors.items():
            if col in d_ind.columns:
                val = d_ind[col].iloc[-1]
                if pd.notna(val):
                    ev = float(val)
                    ax.axhline(ev, color=clr, lw=0.9, ls='-.',
                               alpha=0.55, zorder=8)
                    items.append((ev, lbl, clr, 'white', clr, 6.5))

    # Support lines
    for i, s in enumerate(sup):
        ax.axhline(s, color=sup_color, lw=0.7, ls='--',
                   alpha=0.70, zorder=8)
        items.append((s, rtl(f'دعم {i+1}   {s:.2f}'),
                       sup_color, '#E8F4FD', sup_color, 6.5))

    # Resistance lines
    for i, r in enumerate(res):
        ax.axhline(r, color=res_color, lw=0.7, ls='--',
                   alpha=0.70, zorder=8)
        items.append((r, rtl(f'مقاومة {i+1}   {r:.2f}'),
                       res_color, '#FDEDED', res_color, 6.5))

    if not items:
        return

    # ── Sort by price ascending ──
    items.sort(key=lambda x: x[0])
    n = len(items)

    all_prices = [it[0] for it in items]
    price_min = min(all_prices)
    price_max = max(all_prices)
    price_range = (price_max - price_min) if price_max != price_min \
                  else abs(price_max) * 0.1 or 1.0

    # ── Evenly distribute label y-positions to avoid overlap ──
    pad_top = price_range * 0.30
    pad_bot = price_range * 0.30
    min_label_height = price_range * 0.055
    total_needed = min_label_height * (n - 1) if n > 1 else 0
    natural_span = (price_max + pad_top) - (price_min - pad_bot)

    if total_needed > natural_span and n > 1:
        extra = (total_needed - natural_span) / 2.0
        pad_top += extra
        pad_bot += extra

    label_bottom = price_min - pad_bot
    label_top = price_max + pad_top

    if n == 1:
        label_ys = [items[0][0]]
    else:
        step_y = (label_top - label_bottom) / (n - 1)
        label_ys = [label_bottom + step_y * i for i in range(n)]

    # ── Convert label y to axes fraction ──
    ylo, yhi = ax.get_ylim()
    y_span = (yhi - ylo) if (yhi - ylo) != 0 else 1.0

    def to_y_frac(y_val):
        return (y_val - ylo) / y_span

    # Label x position (axes fraction — outside right border)
    near_frac = 1.10
    far_frac  = 1.10

    # Arrow starts at right edge of data
    arrow_x = xmax - 0.5

    # ── Collision detection ──
    label_half_h = price_range * 0.025

    def line_crosses_label(price_from, ly_to, other_ly):
        band_lo = other_ly - label_half_h
        band_hi = other_ly + label_half_h
        for t in (0.2, 0.4, 0.6, 0.8):
            y_at_t = price_from + t * (ly_to - price_from)
            if band_lo <= y_at_t <= band_hi:
                return True
        return False

    use_far = [False] * n
    for i in range(n):
        price_i = items[i][0]
        ly_i = label_ys[i]
        for j in range(n):
            if i == j:
                continue
            if line_crosses_label(price_i, ly_i, label_ys[j]):
                use_far[i] = True
                break
    for i in range(1, n):
        if use_far[i] and use_far[i - 1]:
            use_far[i - 1] = False

    # ── Draw annotations with arrows ──
    for idx, (price, text, color, bg, edge_clr, fsize) in enumerate(items):
        ly_frac = to_y_frac(label_ys[idx])
        x_frac = far_frac if use_far[idx] else near_frac

        ax.annotate(
            text,
            xy=(arrow_x, price),
            xycoords='data',
            xytext=(x_frac, ly_frac),
            textcoords='axes fraction',
            fontsize=fsize,
            color=color,
            ha='left',
            va='center',
            fontproperties=MPL_FONT_PROP,
            arrowprops=dict(
                arrowstyle='-',
                color=color,
                lw=0.7,
                shrinkA=0,
                shrinkB=0,
            ),
            bbox=dict(
                boxstyle='round,pad=0.22',
                facecolor=bg,
                edgecolor=edge_clr,
                alpha=0.92,
                linewidth=0.7,
            ),
            clip_on=False,
            zorder=8,
        )


# ───────────────────────────────────────────────────────────────────────────────
# 7b: PIVOT POINT FINDER (_get_pivots)
# ───────────────────────────────────────────────────────────────────────────────

def _get_pivots(d, order=5):
    """Find pivot highs and lows for S/R markers."""
    d = d.tail(180).copy().reset_index(drop=True)
    h = d['High'].values.astype(float); l = d['Low'].values.astype(float)
    ph = argrelextrema(h, np.greater_equal, order=order)[0]
    pl = argrelextrema(l, np.less_equal, order=order)[0]
    return {
        'highs': [(int(i), round(h[i], 4)) for i in ph],
        'lows':  [(int(i), round(l[i], 4)) for i in pl],
    }


# ───────────────────────────────────────────────────────────────────────────────
# 7c: MAIN CANDLESTICK CHART WITH S/R (make_main_chart)
# ───────────────────────────────────────────────────────────────────────────────
#
# PURPOSE : Primary chart showing candlesticks + S/R lines + EMA overlays.
#           Used on the first chart page of the PDF report.
#
# TO MODIFY:
#   • Change lookback → modify d.tail(180)
#   • Change chart size → modify figsize=(14,7)
#   • Disable volume bars → set volume=False
# ───────────────────────────────────────────────────────────────────────────────

def make_main_chart(d, sup=None, res=None):
    """Generate main candlestick chart with S/R lines."""
    sup = sup or []; res = res or []; d = d.tail(180).copy()
    p = d[['Open','High','Low','Close','Volume']].copy()
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit',
                               wick='inherit', volume={'up':'#80cbc4','down':'#ef9a9a'})
    st = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#dddddd',
                             rc={'axes.facecolor':'#FAFAFA'})
    plot_kwargs = dict(type='candle', style=st, volume=True, figsize=(14,7),
                       returnfig=True, warn_too_much_data=9999)
    fig, ax = mpf.plot(p, **plot_kwargs)
    main_ax = ax[0]
    xmax = len(d)
    pivots = _get_pivots(d, order=5)
    _draw_sr_lines(main_ax, sup, res, xmax, d_ind=d, pivots=pivots)
    fig.subplots_adjust(right=0.80)
    return chart_bytes(fig)


# ───────────────────────────────────────────────────────────────────────────────
# 7d: SMA OVERLAY CHART (make_price_chart)
# ───────────────────────────────────────────────────────────────────────────────

def make_price_chart(d, sup=None, res=None):
    """Generate candlestick chart with SMA 20/50/100/200 overlays."""
    sup = sup or []; res = res or []; d = d.tail(180).copy()
    p = d[['Open','High','Low','Close','Volume']].copy()
    aps, labels = [], []
    for col, clr, lbl in [('SMA20',BLUE_HEX,'SMA 20'),('SMA50',RED_HEX,'SMA 50'),
                           ('SMA100',VIOLET_HEX,'SMA 100'),('SMA200',BLACK_HEX,'SMA 200')]:
        if col in d and d[col].notna().sum() > 10:
            aps.append(mpf.make_addplot(d[col], color=clr, width=1)); labels.append(lbl)
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit',
                               wick='inherit', volume={'up':'#80cbc4','down':'#ef9a9a'})
    st = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#dddddd',
                             rc={'axes.facecolor':'#FAFAFA'})
    plot_kwargs = dict(type='candle', style=st, volume=True, figsize=(14,7),
                       returnfig=True, warn_too_much_data=9999)
    if aps: plot_kwargs['addplot'] = aps
    fig, ax = mpf.plot(p, **plot_kwargs)
    if labels: ax[0].legend(labels, loc='upper left', fontsize=8, prop=MPL_FONT_PROP)
    xmax = len(d); pivots = _get_pivots(d, order=5)
    fig.subplots_adjust(right=0.95, left=0.05)
    return chart_bytes(fig)


# ───────────────────────────────────────────────────────────────────────────────
# 7e: EMA OVERLAY CHART (make_ema_chart)
# ───────────────────────────────────────────────────────────────────────────────

def make_ema_chart(d, sup=None, res=None):
    """Generate candlestick chart with EMA 20/50/100/200 overlays."""
    sup = sup or []; res = res or []; d = d.tail(180).copy()
    p = d[['Open','High','Low','Close','Volume']].copy()
    aps, labels = [], []
    for col, clr, lbl in [('EMA20',BLUE_HEX,'EMA 20'),('EMA50',RED_HEX,'EMA 50'),
                           ('EMA100',VIOLET_HEX,'EMA 100'),('EMA200',BLACK_HEX,'EMA 200')]:
        if col in d and d[col].notna().sum() > 10:
            aps.append(mpf.make_addplot(d[col], color=clr, width=1)); labels.append(lbl)
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit',
                               wick='inherit', volume={'up':'#80cbc4','down':'#ef9a9a'})
    st = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#dddddd',
                             rc={'axes.facecolor':'#FAFAFA'})
    plot_kwargs = dict(type='candle', style=st, volume=True, figsize=(14,7),
                       returnfig=True, warn_too_much_data=9999)
    if aps: plot_kwargs['addplot'] = aps
    fig, ax = mpf.plot(p, **plot_kwargs)
    if labels: ax[0].legend(labels, loc='upper left', fontsize=2, prop=MPL_FONT_PROP)
    fig.subplots_adjust(right=0.95, left=0.05)
    return chart_bytes(fig)


# ───────────────────────────────────────────────────────────────────────────────
# 7f: TECHNICAL INDICATORS CHART — RSI + MACD + ROC 12 (make_tech_chart)
# ───────────────────────────────────────────────────────────────────────────────
#
# PURPOSE : 3-panel chart showing RSI (top), MACD (middle), ROC 12 (bottom).
#
# TO MODIFY:
#   • Change RSI overbought/oversold → modify 70/30 axhlines
#   • Change MACD histogram colors → modify '#26a69a'/'#ef5350'
#   • Add another panel → change subplots(3,1,...) to (4,1,...)
# ───────────────────────────────────────────────────────────────────────────────

def make_tech_chart(d):
    """Generate 3-panel technical chart: RSI, MACD, ROC 12."""
    d = d.tail(180).copy()
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(14, 8.5), sharex=True)
    x = d.index

    # ── Panel 1: RSI ──
    r = d['RSI']
    a1.plot(x, r, color='#7C4DFF', lw=1.5)
    a1.axhline(70, color=RED_HEX, ls='--', lw=.8, alpha=.7)
    a1.axhline(50, color='gray', ls='--', lw=.8, alpha=.7)
    a1.axhline(30, color='#43A047', ls='--', lw=.8, alpha=.7)
    a1.fill_between(x, r, 70, where=(r>70), alpha=.25, color=RED_HEX)
    a1.fill_between(x, r, 30, where=(r<30), alpha=.25, color='#43A047')
    a1.set_ylabel('RSI', fontproperties=MPL_FONT_PROP_BOLD)
    a1.set_ylim(0, 100); a1.grid(True, alpha=.3)

    # ── Panel 2: MACD ──
    mh = d['MACD_H']
    cols = ['#26a69a' if v >= 0 else '#ef5350' for v in mh.fillna(0)]
    a2.bar(x, mh, color=cols, width=.8, alpha=.6)
    a2.plot(x, d['MACD'], color=BLUE_HEX, lw=1.3, label='MACD')
    a2.plot(x, d['MACD_Sig'], color=ORANGE_HEX, lw=1.3, label=rtl('الإشارة'))
    a2.axhline(0, color='gray', lw=.5)
    a2.set_ylabel('MACD', fontproperties=MPL_FONT_PROP_BOLD)
    a2.legend(prop=MPL_FONT_PROP, fontsize=8); a2.grid(True, alpha=.3)

    # ── Panel 3: ROC 12 ──
    a3.plot(x, d['ROC12'], color='#00897B', lw=1.3, label=rtl('ROC 12'))
    a3.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.7)
    a3.fill_between(x, d['ROC12'], 0, where=(d['ROC12']>0), alpha=0.2, color='#43A047')
    a3.fill_between(x, d['ROC12'], 0, where=(d['ROC12']<0), alpha=0.2, color=RED_HEX)
    a3.set_ylabel(rtl('ROC 12'), fontproperties=MPL_FONT_PROP_BOLD)
    a3.legend(prop=MPL_FONT_PROP, fontsize=8); a3.grid(True, alpha=.3)
    a3.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))

    plt.tight_layout()
    return chart_bytes(fig)


# ───────────────────────────────────────────────────────────────────────────────
# 7g: BOLLINGER BANDS CHART (make_bb_chart)
# ───────────────────────────────────────────────────────────────────────────────

def make_bb_chart(d):
    """Generate candlestick chart with Bollinger Bands overlay."""
    d = d.tail(180).copy()
    p = d[['Open','High','Low','Close','Volume']].copy()
    aps = []
    for col, clr, lbl in [('BB_U',RED_HEX,rtl('الحد العلوي')),
                           ('BB_M',ORANGE_HEX,rtl('المتوسط')),
                           ('BB_L','#43A047',rtl('الحد السفلي'))]:
        aps.append(mpf.make_addplot(d[col], color=clr, width=1.2, label=lbl))
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit',
                               wick='inherit', volume={'up':'#80cbc4','down':'#ef9a9a'})
    st = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#dddddd',
                             rc={'axes.facecolor':'#FAFAFA'})
    fig, ax = mpf.plot(p, type='candle', style=st, volume=True, figsize=(14,7),
                       returnfig=True, warn_too_much_data=9999, addplot=aps)
    ax[0].legend(loc='upper left', fontsize=8, prop=MPL_FONT_PROP)
    return chart_bytes(fig)


# ───────────────────────────────────────────────────────────────────────────────
# 7h: DRAWDOWN CHART (make_dd_chart)
# ───────────────────────────────────────────────────────────────────────────────

def make_dd_chart(dd):
    """Generate drawdown curve chart (filled area below zero)."""
    fig, ax = plt.subplots(figsize=(14, 3.4))
    ax.fill_between(dd.index, dd.values*100, 0, color=RED_HEX, alpha=.35)
    ax.plot(dd.index, dd.values*100, color='#B71C1C', lw=1)
    ax.set_ylabel(rtl('التراجع %'), fontproperties=MPL_FONT_PROP_BOLD)
    ax.grid(True, alpha=.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    plt.tight_layout()
    return chart_bytes(fig)


# ───────────────────────────────────────────────────────────────────────────────
# 7i: VOLUME CHART (make_volume_chart)
# ───────────────────────────────────────────────────────────────────────────────

def make_volume_chart(d):
    """Generate volume bar chart with 20-day moving average."""
    d = d.tail(180).copy()
    fig, ax = plt.subplots(figsize=(14, 3.6))
    x = d.index; vol = d['Volume']
    avg = vol.rolling(20, min_periods=1).mean()
    cols = ['#26a69a' if d['Close'].iloc[i] >= d['Open'].iloc[i] else '#ef5350'
            for i in range(len(d))]
    ax.bar(x, vol, color=cols, alpha=.5, width=.8)
    ax.plot(x, avg, color='#1565C0', lw=1.5, label=rtl('متوسط 20 يوم'))
    ax.set_ylabel(rtl('الحجم'), fontproperties=MPL_FONT_PROP_BOLD)
    ax.legend(prop=MPL_FONT_PROP, fontsize=8); ax.grid(True, alpha=.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    plt.tight_layout()
    return chart_bytes(fig)


# ───────────────────────────────────────────────────────────────────────────────
# 7j: GAUGE CHART (make_gauge_chart)
# ───────────────────────────────────────────────────────────────────────────────
#
# PURPOSE : Semi-circular gauge showing score 0–20 with needle.
#           Right side shows recommendation circle with color.
#
# SEGMENTS:
#   0–4   RED     (سلبي +)
#   4–8   Lt Red  (سلبي)
#   8–12  Orange  (حياد)
#   12–16 Lt Grn  (إيجابي)
#   16–20 GREEN   (إيجابي +)
#
# TO MODIFY:
#   • Change segment colors → modify segments list
#   • Change gauge size → modify figsize
#   • Change needle style → modify annotate arrowprops
# ───────────────────────────────────────────────────────────────────────────────

def make_gauge_chart(score):
    """Generate score gauge chart with recommendation circle."""
    fig, (ax_gauge, ax_button) = plt.subplots(1, 2, figsize=(12, 5),
                                               gridspec_kw={'width_ratios': [3, 1]})

    # ── Gauge setup ──
    ax_gauge.set_xlim(-1.6, 1.6); ax_gauge.set_ylim(-0.45, 1.45)
    ax_gauge.set_aspect('equal'); ax_gauge.axis('off')
    R_OUTER = 1.00; R_INNER = 0.55; R_MID = (R_OUTER + R_INNER) / 2
    R_TICK = R_OUTER + 0.13; N_PTS = 80

    # ── Draw colored segments ──
    segments = [
        (0, 4, '#C62828', rtl('سلبي +')),
        (4, 8, '#EF5350', rtl('سلبي')),
        (8, 12, '#FFA726', rtl('حياد')),
        (12, 16, '#66BB6A', rtl('إيجابي')),
        (16, 20, '#2E7D32', rtl('إيجابي +')),
    ]

    for low, high, color, label in segments:
        t_start = np.deg2rad(180 - (low / 20) * 180)
        t_end = np.deg2rad(180 - (high / 20) * 180)
        theta = np.linspace(t_start, t_end, N_PTS)
        xo, yo = R_OUTER * np.cos(theta), R_OUTER * np.sin(theta)
        xi, yi = R_INNER * np.cos(theta), R_INNER * np.sin(theta)
        vx = np.concatenate([xo, xi[::-1]]); vy = np.concatenate([yo, yi[::-1]])
        ax_gauge.fill(vx, vy, color=color, alpha=0.90, zorder=2)
        for t_edge in [t_start, t_end]:
            ax_gauge.plot([R_INNER*np.cos(t_edge), R_OUTER*np.cos(t_edge)],
                         [R_INNER*np.sin(t_edge), R_OUTER*np.sin(t_edge)],
                         color='white', lw=1.5, zorder=3)
        t_mid = (t_start + t_end) / 2
        lx = R_MID * np.cos(t_mid); ly = R_MID * np.sin(t_mid)
        rotation = np.rad2deg(t_mid)
        if rotation > 90: rotation -= 180
        ax_gauge.text(lx, ly, label, ha='center', va='center', fontsize=20,
                     fontproperties=MPL_FONT_PROP, color='white', fontweight='bold',
                     rotation=rotation, zorder=4)

    # ── Outer rim ──
    theta_all = np.linspace(0, np.pi, 200)
    xo_all = (R_OUTER + 0.03) * np.cos(theta_all)
    yo_all = (R_OUTER + 0.03) * np.sin(theta_all)
    xi_all = R_OUTER * np.cos(theta_all)
    yi_all = R_OUTER * np.sin(theta_all)
    vx_rim = np.concatenate([xo_all, xi_all[::-1]])
    vy_rim = np.concatenate([yo_all, yi_all[::-1]])
    ax_gauge.fill(vx_rim, vy_rim, color='#37474F', alpha=0.85, zorder=1)

    # ── Tick marks ──
    for n in range(1, 21):
        t = np.deg2rad(180 - (n / 20) * 180)
        ax_gauge.plot([R_OUTER*np.cos(t), (R_OUTER+0.04)*np.cos(t)],
                     [R_OUTER*np.sin(t), (R_OUTER+0.04)*np.sin(t)],
                     color='#37474F', lw=1.0, zorder=5)
        tx = R_TICK * np.cos(t); ty = R_TICK * np.sin(t)
        ax_gauge.text(tx, ty, str(n), ha='center', va='center',
                     fontsize=6.2, color='#37474F', fontweight='bold', zorder=5)

    # ── Inner white circle ──
    inner_bg = plt.Circle((0, 0), R_INNER, color='white', zorder=2)
    ax_gauge.add_patch(inner_bg)

    # ── Needle ──
    norm_score = np.clip(score, 0, 20)
    needle_angle = np.deg2rad(180 - (norm_score / 20) * 180)
    nx = 0.88 * np.cos(needle_angle); ny = 0.88 * np.sin(needle_angle)
    ax_gauge.annotate('', xy=(nx, ny), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='#1A1A2E', lw=2.8))
    ax_gauge.plot(0, 0, 'o', color='#1A1A2E', markersize=9, zorder=6)
    ax_gauge.text(0, -0.18, f'{score}/20', ha='center', va='center',
                 fontsize=14, fontproperties=MPL_FONT_PROP_BOLD, color='#1A1A2E')

    # ── Recommendation circle ──
    ax_button.axis('off'); ax_button.set_xlim(0, 1); ax_button.set_ylim(0, 1)
    rec, rec_color = recommendation(score)
    circle = mpatches.Circle((0.5, 0.5), 0.35, facecolor=rec_color,
                              edgecolor='#333333', linewidth=2,
                              transform=ax_button.transAxes)
    ax_button.add_patch(circle)
    ax_button.text(0.5, 0.5, rtl(rec), ha='center', va='center', fontsize=25,
                  fontproperties=MPL_FONT_PROP_BOLD, color='white',
                  transform=ax_button.transAxes)
    ax_button.text(0.5, 0.12, f'{score}/20', ha='center', va='center',
                  fontsize=10, fontproperties=MPL_FONT_PROP,
                  transform=ax_button.transAxes)

    plt.tight_layout()
    return chart_bytes(fig)


# ───────────────────────────────────────────────────────────────────────────────
# 7k: ICHIMOKU CLOUD CHART (make_ichimoku_chart)
# ───────────────────────────────────────────────────────────────────────────────
#
# PURPOSE : Ichimoku Cloud chart with all 5 components:
#           Tenkan (9), Kijun (26), Senkou A/B (cloud), Chikou (26)
#
# TO MODIFY:
#   • Change cloud colors → modify '#4CAF50'/'#E53935'
#   • Change line colors → modify color parameters
# ───────────────────────────────────────────────────────────────────────────────

def make_ichimoku_chart(d):
    """Generate Ichimoku Cloud chart with all components."""
    d = d.tail(180).copy().reset_index()
    xs = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor('#FAFAFA')
    ax.grid(True, linestyle=':', color='#dddddd', alpha=0.7, zorder=0)
    W = 0.4

    # ── Draw candlesticks manually ──
    for i, row in d.iterrows():
        o, h, l, c = float(row['Open']), float(row['High']), float(row['Low']), float(row['Close'])
        bullish = c >= o
        body_top = max(o, c); body_bottom = min(o, c)
        if bullish:
            body_color = 'none'; edge_color = '#26a69a'; wick_color = '#26a69a'
        else:
            body_color = '#ef5350'; edge_color = '#ef5350'; wick_color = '#ef5350'
        ax.plot([i, i], [l, body_bottom], color=wick_color, lw=0.8, zorder=2)
        ax.plot([i, i], [body_top, h], color=wick_color, lw=0.8, zorder=2)
        rect = plt.Rectangle((i - W, body_bottom), 2 * W, body_top - body_bottom,
                              facecolor=body_color, edgecolor=edge_color,
                              linewidth=0.8, zorder=3)
        ax.add_patch(rect)

    # ── Ichimoku lines ──
    ax.plot(xs, d['Tenkan'].values, color='#E91E63', lw=0.7, label=rtl('التنكن (9)'), zorder=4)
    ax.plot(xs, d['Kijun'].values, color='#1565C0', lw=0.7, label=rtl('الكيجن (26)'), zorder=4)

    # ── Cloud (Kumo) ──
    sa = d['Senkou_A'].values; sb = d['Senkou_B'].values
    valid = ~(np.isnan(sa) | np.isnan(sb))
    if valid.sum() > 1:
        xv = xs[valid]; sav, sbv = sa[valid], sb[valid]
        ax.fill_between(xv, sav, sbv, where=(sav >= sbv), alpha=0.2, color='#4CAF50',
                        label=rtl('سحابة صاعدة'), zorder=1)
        ax.fill_between(xv, sav, sbv, where=(sav < sbv), alpha=0.2, color='#E53935',
                        label=rtl('سحابة هابطة'), zorder=1)
        ax.plot(xv, sav, color='#4CAF50', lw=0.5, alpha=0.8, zorder=3)
        ax.plot(xv, sbv, color='#E53935', lw=0.5, alpha=0.8, zorder=3)

    # ── Chikou Span ──
    if 'Chikou' in d.columns:
        chikou = d['Chikou'].values
        valid_c = ~np.isnan(chikou)
        if valid_c.sum() > 1:
            ax.plot(xs[valid_c], chikou[valid_c], color='#FF6F00', lw=0.7,
                    alpha=0.85, label=rtl('الشيكو (26)'), zorder=4)

    # ── X-axis labels ──
    step = max(1, len(d) // 6)
    tick_pos = xs[::step]
    tick_lbls = [str(d['Date'].iloc[i])[:7] if 'Date' in d.columns
                 else str(d.index[i])[:7] for i in tick_pos]
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lbls, rotation=30, fontsize=8)
    ax.set_xlim(-1, len(d))
    ax.set_ylabel(rtl('السعر'), fontproperties=MPL_FONT_PROP_BOLD)
    ax.legend(prop=MPL_FONT_PROP, fontsize=8, loc='upper left')
    plt.tight_layout()
    return chart_bytes(fig)


# ───────────────────────────────────────────────────────────────────────────────
# 7l: CANDLESTICK PATTERN CHART (make_candle_pattern_chart)
# ───────────────────────────────────────────────────────────────────────────────
#
# PURPOSE : Candlestick chart (last 60 days) with annotated pattern names.
#           Patterns are split left/right to avoid overlap.
#
# TO MODIFY:
#   • Change lookback → modify d.tail(60)
#   • Change annotation style → modify arrowprops/bbox
#   • Change arc curvature → modify arcs_r/arcs_l lists
# ───────────────────────────────────────────────────────────────────────────────

def make_candle_pattern_chart(d, patterns):
    """Generate candlestick chart with pattern annotations using curved arrows."""
    d = d.tail(60).copy()
    p = d[['Open','High','Low','Close','Volume']].copy()
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit',
                               wick='inherit', volume={'up':'#80cbc4','down':'#ef9a9a'})
    st = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#dddddd',
                             rc={'axes.facecolor':'#FAFAFA'})
    fig, axlist = mpf.plot(p, type='candle', style=st, volume=False,
                            figsize=(16, 7), returnfig=True, warn_too_much_data=9999)
    ax = axlist[0]

    # Map dates to bar positions
    date_to_pos = {str(dt.date()): i for i, dt in enumerate(d.index)}

    # ── Resolve pattern positions ──
    resolved = []
    for date_lbl, ar_name, en_name, bullish in patterns:
        pos = date_to_pos.get(date_lbl)
        if pos is None: continue
        row_d = d.iloc[pos]
        resolved.append({
            'pos': pos, 'high': float(row_d['High']),
            'low': float(row_d['Low']), 'name': ar_name, 'bullish': bullish
        })

    if not resolved:
        plt.tight_layout(); return chart_bytes(fig)

    # ── Split annotations left/right ──
    ylo, yhi = ax.get_ylim(); xlo, xhi = ax.get_xlim()
    xspan = xhi - xlo; yspan = yhi - ylo
    col_right_x = xhi + xspan * 0.06
    col_left_x = xlo - xspan * 0.06
    right_anns = [ann for i, ann in enumerate(resolved) if i % 2 == 0]
    left_anns = [ann for i, ann in enumerate(resolved) if i % 2 == 1]

    def make_label_ys(count):
        if count == 0: return []
        step = yspan / (count + 1)
        return sorted([ylo + step * (k + 1) for k in range(count)], reverse=True)

    right_ys = make_label_ys(len(right_anns))
    left_ys = make_label_ys(len(left_anns))

    def draw_ann(ann, label_x, label_y, align, rad):
        pos = ann['pos']; bullish = ann['bullish']
        color = '#1B5E20' if bullish is True else ('#B71C1C' if bullish is False else '#E65100')
        label = rtl(ann['name'])
        anchor_y = ann['high'] if bullish is not False else ann['low']
        ax.annotate(label, xy=(pos, anchor_y), xytext=(label_x, label_y),
                   fontsize=8.5, color=color, ha=align, va='center',
                   fontproperties=MPL_FONT_PROP,
                   arrowprops=dict(arrowstyle='-|>', color=color, lw=1.1,
                                  mutation_scale=9,
                                  connectionstyle=f'arc3,rad={rad}'),
                   bbox=dict(boxstyle='round,pad=0.32', facecolor='white',
                            edgecolor=color, alpha=0.95, linewidth=1.0),
                   clip_on=False, zorder=10)

    # ── Draw with varying arc curvatures ──
    arcs_r = [0.25, 0.15, 0.35, 0.10, 0.20]
    arcs_l = [-0.25, -0.15, -0.35, -0.10, -0.20]
    for k, ann in enumerate(right_anns):
        draw_ann(ann, col_right_x, right_ys[k], 'left', arcs_r[k % len(arcs_r)])
    for k, ann in enumerate(left_anns):
        draw_ann(ann, col_left_x, left_ys[k], 'right', arcs_l[k % len(arcs_l)])

    ax.set_xlim(col_left_x - xspan * 0.32, col_right_x + xspan * 0.32)
    fig.subplots_adjust(left=0.12, right=0.88)
    return chart_bytes(fig)


# ───────────────────────────────────────────────────────────────────────────────
# 7m: CCI + WILLIAMS %R CHART (make_cci_willr_chart)
# ───────────────────────────────────────────────────────────────────────────────

def make_cci_willr_chart(d):
    """Generate 2-panel chart: CCI (20) and Williams %R (14)."""
    d = d.tail(180).copy()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    x = d.index

    # ── CCI panel ──
    cci = d['CCI']
    a1.plot(x, cci, color='#7C4DFF', lw=1.3)
    a1.axhline(100, color=RED_HEX, ls='--', lw=0.8, alpha=0.7)
    a1.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
    a1.axhline(-100, color='#43A047', ls='--', lw=0.8, alpha=0.7)
    a1.fill_between(x, cci, 100, where=(cci > 100), alpha=0.2, color=RED_HEX)
    a1.fill_between(x, cci, -100, where=(cci < -100), alpha=0.2, color='#43A047')
    a1.set_ylabel('CCI (20)', fontproperties=MPL_FONT_PROP_BOLD); a1.grid(True, alpha=0.3)

    # ── Williams %R panel ──
    wr = d['WILLR']
    a2.plot(x, wr, color='#0097A7', lw=1.3)
    a2.axhline(-20, color=RED_HEX, ls='--', lw=0.8, alpha=0.7)
    a2.axhline(-50, color='gray', ls='--', lw=0.8, alpha=0.5)
    a2.axhline(-80, color='#43A047', ls='--', lw=0.8, alpha=0.7)
    a2.fill_between(x, wr, -20, where=(wr > -20), alpha=0.2, color=RED_HEX)
    a2.fill_between(x, wr, -80, where=(wr < -80), alpha=0.2, color='#43A047')
    a2.set_ylabel('Williams %R', fontproperties=MPL_FONT_PROP_BOLD)
    a2.set_ylim(-105, 5); a2.grid(True, alpha=0.3)
    a2.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))

    plt.tight_layout()
    return chart_bytes(fig)


# ───────────────────────────────────────────────────────────────────────────────
# 7n: WILLIAMS ALLIGATOR CHART (make_alligator_chart)
# ───────────────────────────────────────────────────────────────────────────────
#
# PURPOSE : Williams Alligator indicator overlay on candlestick chart.
#           Jaw (Blue 13,8), Teeth (Red 8,5), Lips (Green 5,3).
# ───────────────────────────────────────────────────────────────────────────────

def make_alligator_chart(d):
    """Generate candlestick chart with Williams Alligator overlay."""
    d = d.tail(180).copy()
    p = d[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    aps = []; labels = []
    for col, clr, lbl in [
        ('Alligator_Jaw',   '#1565C0', rtl('الفك (13,8)')),
        ('Alligator_Teeth', '#E91E63', rtl('الأسنان (8,5)')),
        ('Alligator_Lips',  '#4CAF50', rtl('الشفاه (5,3)')),
    ]:
        if col in d and d[col].notna().sum() > 10:
            aps.append(mpf.make_addplot(d[col], color=clr, width=1))
            labels.append(lbl)
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit',
                               wick='inherit', volume={'up': '#80cbc4', 'down': '#ef9a9a'})
    st = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#dddddd',
                             rc={'axes.facecolor': '#FAFAFA'})
    kw = dict(type='candle', style=st, volume=False, figsize=(14, 7),
              returnfig=True, warn_too_much_data=9999)
    if aps: kw['addplot'] = aps
    fig, ax = mpf.plot(p, **kw)
    if labels: ax[0].legend(labels, loc='upper left', fontsize=8, prop=MPL_FONT_PROP)
    fig.subplots_adjust(right=0.95, left=0.05)
    return chart_bytes(fig)


# ───────────────────────────────────────────────────────────────────────────────
# 7o: SUPERTREND CHART (make_supertrend_chart)
# ───────────────────────────────────────────────────────────────────────────────
#
# PURPOSE : Supertrend indicator overlay on candlestick chart.
#           Green = bullish trend, Red = bearish trend.
# ───────────────────────────────────────────────────────────────────────────────

def make_supertrend_chart(d):
    """Generate candlestick chart with Supertrend overlay."""
    d = d.tail(180).copy()
    p = d[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    st_val, st_dir = _compute_supertrend(d)
    st_up   = st_val.where(st_dir == 1,  other=np.nan)
    st_down = st_val.where(st_dir == -1, other=np.nan)
    aps = []; labels = []
    if st_up.notna().any():
        aps.append(mpf.make_addplot(st_up, color='#4CAF50', width=1))
        labels.append(rtl('صاعد'))
    if st_down.notna().any():
        aps.append(mpf.make_addplot(st_down, color='#E53935', width=1))
        labels.append(rtl('هابط'))
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit',
                               wick='inherit', volume={'up': '#80cbc4', 'down': '#ef9a9a'})
    sty = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#dddddd',
                              rc={'axes.facecolor': '#FAFAFA'})
    kw = dict(type='candle', style=sty, volume=False, figsize=(14, 7),
              returnfig=True, warn_too_much_data=9999)
    if aps: kw['addplot'] = aps
    fig, ax = mpf.plot(p, **kw)
    if labels: ax[0].legend(labels, loc='upper left', fontsize=8, prop=MPL_FONT_PROP)
    fig.subplots_adjust(right=0.95, left=0.05)
    return chart_bytes(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: PDF REPORT CLASS
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Build the multi-page PDF report using ReportLab.
#           Each method creates one or more pages of the report.
#
# PAGE STRUCTURE (in order):
#   1. cover()              — Cover page with price, KPIs, 52-week bar
#   2. main_charts_page()   — Main chart + SMA/EMA + Alligator/Supertrend
#   3. tech_page()          — RSI/MACD/ROC + Bollinger Bands
#   4. ichimoku_page()      — Ichimoku + Candlestick patterns
#   5. perf_page()          — Performance returns + Risk metrics + Drawdown
#   6. fund_page()          — Fundamental analysis (valuation, profitability)
#   7. signal_page()        — Gauge + Signals table + S/R levels + Indicators
#   8. cci_willr_page()     — CCI/Williams%R + Divergences + Candle patterns
#   9. review_page()        — Multi-paragraph Arabic technical review
#
# HELPER METHODS:
#   _font()   — Set current font (regular or bold)
#   _bar()    — Draw page header bar (navy with teal accent)
#   _foot()   — Draw page footer
#   _stitle() — Draw section title with underline
#   _box()    — Draw KPI box (gray background)
#   _table()  — Draw data table with alternating row colors
#   _img()    — Insert chart image
#   _wrap_arabic_text() — Word-wrap Arabic text for review page
#
# TO MODIFY:
#   • Add new page → create new method, call in _build_report_sync()
#   • Change header color → modify NAVY in _bar()
#   • Change table style → modify _table() colors
#   • Change box style → modify _box() corner radius, colors
#   • Add logo → load image in cover() or _bar()
# ═══════════════════════════════════════════════════════════════════════════════

# ── Arabic labels for period names and risk metrics ──
PERIOD_AR = {
    '1 Day':'يوم', '1 Week':'أسبوع', '1 Month':'شهر',
    '3 Months':'3 أشهر', '6 Months':'6 أشهر',
    '1 Year':'سنة', 'YTD':'منذ بداية السنة',
}

RISK_AR = {
    'Daily Volatility':'التذبذب اليومي', 'Annual Volatility':'التذبذب السنوي',
    'Sharpe Ratio':'نسبة شارب', 'Sortino Ratio':'نسبة سورتينو',
    'Max Drawdown':'أقصى تراجع', 'Avg Daily Return':'متوسط العائد اليومي',
    'Best Day':'أفضل يوم', 'Worst Day':'أسوأ يوم',
}


class Report:
    """PDF Report builder using ReportLab Canvas."""

    def __init__(self, path, ticker, info, display_ticker=None):
        self.c = pdfcanvas.Canvas(path, pagesize=A4)
        self.c.setTitle(f'{ticker} Arabic Stock Report')
        self.tk = ticker
        self.display_tk = display_ticker or ticker
        self.info = info
        self.pn = 0  # Page counter

    # ── Font helper ──
    def _font(self, bold=False, size=10):
        """Set current font to Amiri Regular or Bold."""
        self.c.setFont(AR_FONT_BOLD if bold else AR_FONT, size)

    # ── Page header bar ──
    def _bar(self, title):
        """Draw navy header bar with teal accent line."""
        self.pn += 1
        c = self.c
        c.setFillColor(NAVY); c.rect(0, PAGE_H-34*mm, PAGE_W, 34*mm, fill=1, stroke=0)
        c.setFillColor(TEAL); c.rect(0, PAGE_H-36*mm, PAGE_W, 2*mm, fill=1, stroke=0)
        c.setFillColor(WHITE); self._font(True, 15)
        c.drawRightString(PAGE_W-MG, PAGE_H-18*mm, rtl(title))
        self._font(False, 9)
        c.drawString(MG, PAGE_H-14*mm, self.display_tk)
        c.drawString(MG, PAGE_H-22*mm, datetime.now().strftime('%Y-%m-%d'))

    # ── Page footer ──
    def _foot(self):
        """Draw light gray footer with ticker and page number."""
        c = self.c
        c.setFillColor(LGRAY); c.rect(0, 0, PAGE_W, 10*mm, fill=1, stroke=0)
        c.setFillColor(DGRAY); self._font(False, 7)
        c.drawString(MG, 4*mm, self.display_tk)
        c.drawRightString(PAGE_W-MG, 4*mm, rtl(f'الصفحة {self.pn}'))

    # ── Section title with underline ──
    def _stitle(self, y, t):
        """Draw section title with teal underline. Returns new y position."""
        c = self.c
        c.setFillColor(NAVY); self._font(True, 11)
        c.drawRightString(PAGE_W-MG, y, rtl(t))
        c.setStrokeColor(TEAL); c.setLineWidth(1.4)
        c.line(MG, y-4, PAGE_W-MG, y-4)
        return y - 18

    # ── KPI box ──
    def _box(self, x, y, bw, bh, lbl, val, clr=None):
        """Draw a gray box with label (top) and value (bottom)."""
        c = self.c
        c.setFillColor(LGRAY); c.roundRect(x, y, bw, bh, 5, fill=1, stroke=0)
        c.setFillColor(DGRAY); self._font(False, 8)
        c.drawCentredString(x + bw/2, y + bh - 14, tx(lbl))
        if isinstance(val, (int, float)):
            val_str, val_clr = fmt_n(val)
        else:
            val_str = str(val) if val is not None else '-'
            val_clr = None
        if val_clr:
            fill_color = val_clr
        elif clr:
            fill_color = HexColor(clr) if isinstance(clr, str) else clr
        else:
            fill_color = NAVY
        c.setFillColor(fill_color); self._font(True, 12.5)
        c.drawCentredString(x + bw/2, y + 8, tx(val_str))

    # ── Data table ──
    def _table(self, y, rows, cw_list, sig_mode=False, score_mode=False):
        """Draw a table with alternating row colors. Returns new y position."""
        c = self.c; rh = 16; total_w = sum(cw_list)
        for i, row in enumerate(rows):
            ry = y - i * rh
            if i == 0:      bg, fg, is_bold = NAVY, WHITE, True
            elif i % 2 == 1: bg, fg, is_bold = LGRAY, TXTDARK, False
            else:           bg, fg, is_bold = WHITE, TXTDARK, False
            c.setFillColor(bg); c.rect(MG, ry-4, total_w, rh, fill=1, stroke=0)
            sx = MG; c.setStrokeColor(HexColor('#D8DEE9')); c.setLineWidth(0.5)
            for col_w in cw_list[:-1]:
                sx += col_w; c.line(sx, ry-4, sx, ry-4+rh)
            cx = MG
            for j, cell in enumerate(row):
                fill = fg; is_cell_bold = is_bold
                if i > 0 and j == 0 and str(cell).startswith('-') and not sig_mode and not score_mode:
                    fill = RED
                if i > 0 and sig_mode and j == 0:
                    raw = str(cell)
                    if 'إيجابي' in raw: fill = GREEN
                    elif 'سلبي' in raw: fill = RED
                    else: fill = ORANGE
                    is_cell_bold = True
                if i > 0 and score_mode and j == 0:
                    if cell == '✓': fill = GREEN
                    elif cell == '✗': fill = RED
                    is_cell_bold = True
                c.setFillColor(fill); self._font(is_cell_bold, 8)
                c.drawRightString(cx + cw_list[j] - 4, ry + 2, tx(cell))
                cx += cw_list[j]
        return y - len(rows) * rh - 6

    # ── Insert chart image ──
    def _img(self, y, buf, max_h):
        """Insert a chart image. Returns new y position."""
        c = self.c; buf.seek(0)
        img = ImageReader(buf); iw, ih = img.getSize()
        ratio = ih / iw; dw = CW; dh = dw * ratio
        if dh > max_h: dh = max_h; dw = dh / ratio
        x = MG + (CW - dw) / 2
        c.drawImage(img, x, y - dh - 2, dw, dh)
        return y - dh - 8

    # ── Arabic text word-wrap ──
    def _wrap_arabic_text(self, text, max_width, font_size):
        """Wrap Arabic text into lines that fit within max_width."""
        words = str(text).split(); lines = []; current_line = []
        for word in words:
            test_shaped = rtl(' '.join(current_line + [word]))
            line_width = self.c.stringWidth(test_shaped, AR_FONT, font_size)
            if line_width <= max_width:
                current_line.append(word)
            else:
                if current_line: lines.append(rtl(' '.join(current_line)))
                current_line = [word]
        if current_line: lines.append(rtl(' '.join(current_line)))
        return lines

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1: COVER PAGE
    # ══════════════════════════════════════════════════════════════════════════
    def cover(self, price, chg, info, rec_txt, rec_color, score):
        """Generate cover page with price, KPIs, 52-week bar, company info."""
        c = self.c; self.pn += 1

        # ── Header ──
        c.setFillColor(NAVY); c.rect(0, PAGE_H-78*mm, PAGE_W, 78*mm, fill=1, stroke=0)
        c.setFillColor(TEAL); c.rect(0, PAGE_H-80*mm, PAGE_W, 2*mm, fill=1, stroke=0)
        company_name = (COMPANY_NAMES.get(self.tk)
                        or safe(info, 'longName', safe(info, 'shortName', self.display_tk))
                        or self.display_tk)
        company_name = short_text(str(company_name), 42)
        c.setFillColor(WHITE); self._font(True, 23)
        c.drawRightString(PAGE_W-MG, PAGE_H-22*mm, rtl('تقرير تحليل سهم'))
        self._font(True, 18)
        c.drawRightString(PAGE_W-MG, PAGE_H-39*mm, tx(company_name))
        self._font(False, 11)
        c.drawRightString(PAGE_W-MG, PAGE_H-52*mm, tx(f'{self.display_tk} | {safe(info,"exchange","-")}'))
        self._font(False, 9)
        c.drawString(MG, PAGE_H-18*mm, datetime.now().strftime('%Y-%m-%d'))

        # ── Recommendation badge ──
        c.setFillColor(HexColor(rec_color))
        c.roundRect(MG, PAGE_H-58*mm, 60*mm, 14*mm, 8, fill=1, stroke=0)
        c.setFillColor(WHITE); self._font(True, 11)
        c.drawCentredString(MG+30*mm, PAGE_H-50*mm, rtl(rec_txt))
        self._font(False, 8)
        c.drawCentredString(MG+30*mm, PAGE_H-54*mm, rtl(f'النتيجة {score}/20'))

        # ── KPI boxes (3 rows × 3 columns) ──
        col_bw = CW/3 - 4*mm; col_bh = 18*mm; col_gap = 4*mm
        x1 = MG; x2 = MG + col_bw + col_gap; x3 = MG + 2*(col_bw + col_gap)
        y1 = PAGE_H - 110*mm; y2 = y1 - col_bh - col_gap
        y3 = y2 - col_bh - col_gap; y4 = y3 - col_bh - col_gap

        # Row 1: Price, Change, Market Cap
        self._box(x1, y1, col_bw, col_bh, 'السعر الحالي', price)
        self._box(x2, y1, col_bw, col_bh, 'التغير اليومي', f'{chg:+.2f}%',
                  clr=GREEN_HEX if chg >= 0 else RED_HEX)
        self._box(x3, y1, col_bw, col_bh, 'القيمة السوقية', fmt_n(safe(info,'marketCap'))[0])

        # Row 2: P/E, EPS, Dividend Yield
        pe = safe(info, 'trailingPE'); eps = safe(info, 'trailingEps'); dy = safe(info, 'dividendYield')
        self._box(x1, y2, col_bw, col_bh, 'مكرر الربحية', f'{float(pe):.2f}' if pe else '-')
        eps_val = float(eps) if eps else None
        eps_str = f'{eps_val:.2f}' if eps_val is not None else '-'
        eps_color = GREEN_HEX if (eps_val is not None and eps_val >= 0) else RED_HEX if eps_val is not None else None
        self._box(x2, y2, col_bw, col_bh, 'ربحية السهم', eps, clr=eps_color)
        self._box(x3, y2, col_bw, col_bh, 'عائد التوزيعات', fmt_p(dy)[0] if dy else '-')

        # Row 3: P/B, ROE, Beta
        pb = safe(info, 'priceToBook'); roe = safe(info, 'returnOnEquity'); beta = safe(info, 'beta')
        self._box(x1, y3, col_bw, col_bh, 'مضاعف القيمة الدفترية', f'{float(pb):.2f}' if pb else '-')
        self._box(x2, y3, col_bw, col_bh, 'العائد على حقوق المساهمين', fmt_p(roe)[0] if roe else '-')
        self._box(x3, y3, col_bw, col_bh, 'العائد على متوسط الأصول', f'{float(roa):.2f}' if beta else '-')

        # Row 4: Volume, Trading Value, Trades Count
        self._box(x1, y4, col_bw, col_bh, 'حجم التداول', fmt_n(safe(info,'volume'), d=0)[0])
        val_str = fmt_n(safe(info,'tradingValue'), d=0)[0] if safe(info,'tradingValue') else '-'
        self._box(x2, y4, col_bw, col_bh, 'قيمة التداول', val_str)
        trades_str = fmt_n(safe(info,'tradesCount'), d=0)[0] if safe(info,'tradesCount') else '-'
        self._box(x3, y4, col_bw, col_bh, 'عدد الصفقات', trades_str)

        # ── 52-week range bar ──
        w52h = safe(info, 'fiftyTwoWeekHigh'); w52l = safe(info, 'fiftyTwoWeekLow')
        if w52h and w52l and float(w52h) != float(w52l):
            bar_y = y4 - 12*mm
            c.setFillColor(NAVY); self._font(True, 8)
            c.drawRightString(PAGE_W-MG, bar_y+2, rtl('نطاق 52 أسبوع'))
            bar_x = MG; bar_w = CW; bar_h = 5*mm; bar_y2 = bar_y - bar_h - 2
            c.setFillColor(HexColor('#ECEFF1'))
            c.roundRect(bar_x, bar_y2, bar_w, bar_h, 3, fill=1, stroke=0)
            pos = (price - float(w52l)) / (float(w52h) - float(w52l))
            pos = max(0.0, min(1.0, pos)); fill_w = bar_w * pos
            fill_c = HexColor(RED_HEX) if pos < 0.35 else (HexColor(GREEN_HEX) if pos > 0.65 else HexColor(ORANGE_HEX))
            c.setFillColor(fill_c); c.roundRect(bar_x, bar_y2, max(fill_w, 4), bar_h, 3, fill=1, stroke=0)
            c.setFillColor(DGRAY); self._font(False, 7)
            c.drawString(bar_x, bar_y2-9, f'{float(w52l):.2f}')
            c.drawRightString(bar_x+bar_w, bar_y2-9, f'{float(w52h):.2f}')
            c.setFillColor(NAVY); self._font(True, 7)
            c.drawCentredString(bar_x+bar_w/2, bar_y2-9, f'{pos*100:.0f}%')
            y_after = bar_y2 - 12*mm
        else:
            y_after = y4 - 6*mm

        # ── Company info section ──
        y = y_after; y = self._stitle(y, 'معلومات الشركة')
        sector = short_text(safe(info, 'sector', '-') or '-', 26)
        industry = short_text(safe(info, 'industry', '-') or '-', 26)
        items = [
            ('القطاع', sector), ('الصناعة', industry),
            ('العملة', safe(info, 'currency', 'SAR') or '-'),
            ('متوسط الحجم', fmt_n(safe(info, 'averageVolume'), d=0)[0]),
            ('أعلى 52 أسبوع', fmt_n(safe(info, 'fiftyTwoWeekHigh'))[0]),
            ('أدنى 52 أسبوع', fmt_n(safe(info, 'fiftyTwoWeekLow'))[0]),
            ('الأسهم القائمة', fmt_n(safe(info, 'sharesOutstanding'), d=0)[0]),
            ('الأسهم الحرة', fmt_n(safe(info, 'floatShares'), d=0)[0]),
        ]
        half = CW / 2
        for idx, (lbl, val) in enumerate(items):
            col = idx % 2; row = idx // 2
            xr = MG + (col + 1) * half - 5; yy = y - row * 18
            c.setFillColor(NAVY); self._font(True, 8.5)
            c.drawRightString(xr, yy, rtl(f'{lbl}:'))
            c.setFillColor(TXTDARK); self._font(False, 8.5)
            c.drawRightString(xr - 85, yy, tx(val))

        # ── Disclaimer ──
        c.setFillColor(DGRAY); self._font(False, 7)
        c.drawCentredString(PAGE_W/2, 14*mm, rtl('هذا التقرير لأغراض معلوماتية فقط وليس توصية استثمارية.'))
        self._foot(); c.showPage()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2: MAIN CHARTS (S/R + SMA/EMA + Alligator/Supertrend)
    # ══════════════════════════════════════════════════════════════════════════
    def main_charts_page(self, main_img, sma_img, ema_img, alligator_img, supertrend_img):
        """Charts page: S/R (full-width), SMA|EMA side-by-side, Alligator|Supertrend side-by-side."""
        self._bar('الرسم البياني'); self._foot()
        c = self.c; y = PAGE_H - 44*mm

        # ── S/R chart (full width) ──
        y = self._stitle(y, 'الدعوم والمقاومات')
        y = self._img(y, main_img, 78*mm)
        y -= 10*mm  # 1cm spacing

        # ── SMA (right) | EMA (left) side-by-side ──
        mid_x = MG + CW / 2
        right_title_x = PAGE_W - MG; left_title_x = mid_x - 4*mm
        c.setFillColor(NAVY); self._font(True, 9)
        c.drawRightString(right_title_x, y, rtl('المتوسطات المتحركة البسيطة (SMA)'))
        c.drawRightString(left_title_x, y, rtl('المتوسطات المتحركة الأسية (EMA)'))
        c.setStrokeColor(TEAL); c.setLineWidth(1.0)
        c.line(mid_x + 2*mm, y - 3, PAGE_W - MG, y - 3)
        c.line(MG, y - 3, mid_x - 2*mm, y - 3)
        y -= 8*mm

        dw = (CW / 2) - 3*mm; maxh = 58*mm
        sma_img.seek(0); img_sma = ImageReader(sma_img)
        ema_img.seek(0); img_ema = ImageReader(ema_img)

        ratio_sma = img_sma.getSize()[1] / img_sma.getSize()[0]
        dh_sma = min(dw * ratio_sma, maxh); dw_sma = dh_sma / ratio_sma
        ratio_ema = img_ema.getSize()[1] / img_ema.getSize()[0]
        dh_ema = min(dw * ratio_ema, maxh); dw_ema = dh_ema / ratio_ema

        x_sma = PAGE_W - MG - dw_sma; x_ema = MG
        c.drawImage(img_sma, x_sma, y - dh_sma, dw_sma, dh_sma)
        c.drawImage(img_ema, x_ema, y - dh_ema, dw_ema, dh_ema)
        y -= max(dh_sma, dh_ema) + 5*mm

        # ── Alligator (right) | Supertrend (left) side-by-side ──
        c.setFillColor(NAVY); self._font(True, 9)
        c.drawRightString(right_title_x, y, rtl('مؤشر التمساح (Alligator)'))
        c.drawRightString(left_title_x, y, rtl('مؤشر السوبر تريند (Supertrend)'))
        c.setStrokeColor(TEAL); c.setLineWidth(1.0)
        c.line(mid_x + 2*mm, y - 3, PAGE_W - MG, y - 3)
        c.line(MG, y - 3, mid_x - 2*mm, y - 3)
        y -= 8*mm

        alligator_img.seek(0); img_alg = ImageReader(alligator_img)
        supertrend_img.seek(0); img_st = ImageReader(supertrend_img)
        ratio_alg = img_alg.getSize()[1] / img_alg.getSize()[0]
        dh_alg = min(dw * ratio_alg, maxh); dw_alg = dh_alg / ratio_alg
        ratio_st = img_st.getSize()[1] / img_st.getSize()[0]
        dh_st = min(dw * ratio_st, maxh); dw_st = dh_st / ratio_st

        c.drawImage(img_alg, PAGE_W - MG - dw_alg, y - dh_alg, dw_alg, dh_alg)
        c.drawImage(img_st, MG, y - dh_st, dw_st, dh_st)
        self.c.showPage()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3: TECHNICAL INDICATORS (RSI/MACD/ROC + Bollinger)
    # ══════════════════════════════════════════════════════════════════════════
    def tech_page(self, tech_img, bb_img):
        self._bar('المؤشرات الفنية'); self._foot()
        y = PAGE_H - 44*mm
        y = self._stitle(y, 'RSI و MACD و ROC 12'); y = self._img(y, tech_img, 113*mm)
        y = self._stitle(y, 'نطاقات بولنجر'); self._img(y, bb_img, 113*mm)
        self.c.showPage()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4: PERFORMANCE & RISK + SCORE TABLE
    # ══════════════════════════════════════════════════════════════════════════
    def perf_page(self, pers, risk, dd_img, vol_img, score_criteria, total_score):
        self._bar('الأداء والمخاطر'); self._foot()
        y = PAGE_H - 44*mm
        y = self._stitle(y, 'العوائد حسب الفترة')
        rows = [['القيمة', 'الفترة']]
        for k, v in pers.items(): rows.append([f'{v:+.2f}%', PERIOD_AR.get(k, k)])
        y = self._table(y, rows, [CW*0.38, CW*0.62])
        y = self._stitle(y, 'مقاييس المخاطر')
        rows2 = [['القيمة', 'المقياس']]
        for k, v in risk.items(): rows2.append([str(v), RISK_AR.get(k, k)])
        y = self._table(y, rows2, [CW*0.38, CW*0.62])
        y = self._stitle(y, 'منحنى التراجع')
        self._img(y, dd_img, 58*mm); self.c.showPage()

        # ── Volume + Score table page ──
        self._bar('تحليل الحجم'); self._foot()
        y2 = PAGE_H - 44*mm
        y2 = self._stitle(y2, 'الحجم اليومي مقابل متوسط 20 يوم')
        y2 = self._img(y2, vol_img, 100*mm)
        y_table = y2 - 6*mm
        if y_table > 80*mm:
            y_table = self._stitle(y_table, f'جدول نقاط النتيجة ({total_score}/20)')
            rows_score = [['النقاط', 'الحالة', 'البند']]
            for lbl, (symbol, pt) in score_criteria.items():
                status = rtl('نعم ✓') if pt == 1 else rtl('لا ✗')
                rows_score.append([str(pt), status, rtl(short_text(lbl, 35))])
            rows_score.append([str(total_score), rtl('من 20'), rtl('الإجمالي')])
            self._table(y_table, rows_score, [CW*0.15, CW*0.30, CW*0.55], score_mode=True)
        else:
            self.c.showPage(); self._bar('جدول نقاط النتيجة'); self._foot()
            y_table = PAGE_H - 44*mm
            y_table = self._stitle(y_table, f'جدول نقاط النتيجة ({total_score}/20)')
            rows_score = [['النقاط', 'الحالة', 'البند']]
            for lbl, (symbol, pt) in score_criteria.items():
                status = rtl('نعم ✓') if pt == 1 else rtl('لا ✗')
                rows_score.append([str(pt), status, rtl(short_text(lbl, 35))])
            rows_score.append([str(total_score), rtl('من 20'), rtl('الإجمالي')])
            self._table(y_table, rows_score, [CW*0.15, CW*0.30, CW*0.55], score_mode=True)
        self.c.showPage()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 5: FUNDAMENTAL ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    def fund_page(self, info):
        self._bar('التحليل الأساسي'); self._foot()
        y = PAGE_H - 44*mm

        # Valuation
        y = self._stitle(y, 'التقييم')
        val_items = [(safe(info,'trailingPE','-'),'Trailing P/E'), (safe(info,'forwardPE','-'),'Forward P/E'),
                     (safe(info,'priceToBook','-'),'Price / Book'), (safe(info,'priceToSalesTrailing12Months','-'),'Price / Sales'),
                     (safe(info,'enterpriseToEbitda','-'),'EV / EBITDA'), (safe(info,'enterpriseToRevenue','-'),'EV / Revenue'),
                     (safe(info,'pegRatio','-'),'PEG Ratio')]
        val_rows = [['القيمة', 'البند']]
        for v, lbl in val_items: val_rows.append([f'{v:.2f}' if isinstance(v, (int, float)) else str(v), lbl])
        y = self._table(y, val_rows, [CW*0.40, CW*0.60])

        # Profitability
        y = self._stitle(y, 'الربحية')
        prof_rows = [['القيمة', 'البند'],
                     [fmt_n(safe(info,'totalRevenue'))[0], 'الإيرادات'],
                     [fmt_n(safe(info,'netIncomeToCommon'))[0], 'صافي الدخل'],
                     [fmt_n(safe(info,'ebitda'))[0], 'EBITDA'],
                     [fmt_p(safe(info,'grossMargins'))[0], 'هامش إجمالي'],
                     [fmt_p(safe(info,'operatingMargins'))[0], 'هامش تشغيلي'],
                     [fmt_p(safe(info,'profitMargins'))[0], 'هامش صافي'],
                     [fmt_p(safe(info,'returnOnEquity'))[0], 'ROE'],
                     [fmt_p(safe(info,'returnOnAssets'))[0], 'ROA']]
        y = self._table(y, prof_rows, [CW*0.40, CW*0.60])

        # Financial Position
        y = self._stitle(y, 'المركز المالي')
        fin_items = [(fmt_n(safe(info,'totalCash'))[0], 'إجمالي النقد'),
                     (fmt_n(safe(info,'totalDebt'))[0], 'إجمالي الدين'),
                     (safe(info,'debtToEquity','-'), 'Debt / Equity'),
                     (safe(info,'currentRatio','-'), 'Current Ratio'),
                     (safe(info,'quickRatio','-'), 'Quick Ratio'),
                     (fmt_n(safe(info,'bookValue'))[0], 'القيمة الدفترية للسهم')]
        fin_rows = [['القيمة', 'البند']]
        for v, lbl in fin_items: fin_rows.append([f'{v:.2f}' if isinstance(v, (int, float)) else str(v), lbl])
        y = self._table(y, fin_rows, [CW*0.40, CW*0.60])

        if y < 60*mm: self.c.showPage(); self._bar('التحليل الأساسي (تابع)'); self._foot(); y = PAGE_H - 44*mm

        # Dividends
        y = self._stitle(y, 'التوزيعات والأسهم')
        div_rows = [['القيمة', 'البند'],
                    [fmt_n(safe(info,'dividendRate'))[0], 'معدل التوزيع'],
                    [fmt_p(safe(info,'dividendYield'))[0], 'عائد التوزيع'],
                    [fmt_p(safe(info,'payoutRatio'))[0], 'نسبة التوزيع'],
                    [fmt_n(safe(info,'sharesOutstanding'), d=0)[0], 'الأسهم القائمة'],
                    [fmt_n(safe(info,'floatShares'), d=0)[0], 'الأسهم الحرة']]
        self._table(y, div_rows, [CW*0.40, CW*0.60])
        self.c.showPage()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 6: SIGNALS & ANALYSIS (Gauge + Signals + S/R + Indicator Values)
    # ══════════════════════════════════════════════════════════════════════════
    def signal_page(self, gauge_img, sig, score, sup, res, d):
        self._bar('الإشارات والتحليل'); self._foot()
        c = self.c; y = PAGE_H - 44*mm
        y = self._stitle(y, 'مؤشر التحليل'); y = self._img(y, gauge_img, 50*mm)
        c.setFillColor(TXTDARK); self._font(False, 9)
        c.drawRightString(PAGE_W-MG, y-2, rtl(f'النتيجة الكلية: {score}/20')); y -= 14

        # Signals table
        y = self._stitle(y, 'ملخص الإشارات الفنية')
        rows = [['التحليل', 'الإشارة', 'المؤشر']]
        for ind_name, (txt_signal, direction) in sig.items():
            rows.append([decision_text(direction), txt_signal, ind_name])
        y = self._table(y, rows, [CW*0.18, CW*0.38, CW*0.44], sig_mode=True)

        # S/R levels
        y = self._stitle(y, 'الدعم والمقاومة')
        sup_txt = ' | '.join(f'{s:.2f}' for s in sup) if sup else 'N/A'
        res_txt = ' | '.join(f'{r:.2f}' for r in res) if res else 'N/A'
        c.setFillColor(GREEN); self._font(True, 9)
        c.drawRightString(PAGE_W-MG, y, rtl(f'الدعوم: {sup_txt}')); y -= 16
        c.setFillColor(RED); self._font(True, 9)
        c.drawRightString(PAGE_W-MG, y, rtl(f'المقاومات: {res_txt}')); y -= 22

        if y < 100*mm: self.c.showPage(); self._bar('القيم الحالية للمؤشرات'); self._foot(); y = PAGE_H - 44*mm

        # Current indicator values
        y = self._stitle(y, 'القيم الحالية للمؤشرات')
        last = d.iloc[-1]
        rows2 = [['القيمة', 'المؤشر'],
                 [f"{float(last['Close']):.2f}", 'سعر الإغلاق'],
                 [f"{float(last['SMA20']):.2f}" if pd.notna(last['SMA20']) else '-', 'SMA 20'],
                 [f"{float(last['SMA50']):.2f}" if pd.notna(last['SMA50']) else '-', 'SMA 50'],
                 [f"{float(last['RSI']):.2f}" if pd.notna(last['RSI']) else '-', 'RSI'],
                 [f"{float(last['MACD']):.4f}" if pd.notna(last['MACD']) else '-', 'MACD'],
                 [f"{float(last['ATR']):.2f}" if pd.notna(last['ATR']) else '-', 'ATR'],
                 [f"{float(last['ADX']):.1f}" if pd.notna(last['ADX']) else '-', 'ADX'],
                 [f"{float(last['VWAP']):.2f}" if pd.notna(last['VWAP']) else '-', 'VWAP'],
                 [f"{float(last['CCI']):.1f}" if pd.notna(last['CCI']) else '-', 'CCI (20)'],
                 [f"{float(last['WILLR']):.1f}" if pd.notna(last['WILLR']) else '-', 'Williams %R']]
        self._table(y, rows2, [CW*0.35, CW*0.65])
        c.setFillColor(DGRAY); self._font(False, 6.7)
        c.drawCentredString(PAGE_W/2, 16*mm, rtl('هذا التقرير آلي لأغراض معلوماتية فقط وليس نصيحة استثمارية.'))
        self.c.showPage()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 7: ICHIMOKU + CANDLESTICK PATTERNS
    # ══════════════════════════════════════════════════════════════════════════
    def ichimoku_page(self, ichimoku_img, pattern_img):
        self._bar('الإيشيموكو ونماذج الشموع'); self._foot()
        y = PAGE_H - 44*mm
        y = self._stitle(y, 'مخطط الإيشيموكو (Ichimoku Cloud)')
        y = self._img(y, ichimoku_img, 100*mm)
        y = self._stitle(y, 'الشموع اليابانية (آخر 60 يوماً) مع النماذج المرصودة')
        self._img(y, pattern_img, 100*mm)
        self.c.showPage()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 8: CCI / WILLIAMS %R + DIVERGENCES + CANDLE PATTERNS TABLE
    # ══════════════════════════════════════════════════════════════════════════
    def cci_willr_page(self, cci_img, patterns, divergences, d):
        self._bar('مؤشرات إضافية: CCI وWilliams %R'); self._foot()
        c = self.c; y = PAGE_H - 44*mm
        y = self._stitle(y, 'CCI (20) وWilliams %R (14)')
        y = self._img(y, cci_img, 90*mm); y -= 6*mm

        # Oscillator values
        last = d.iloc[-1]
        cci_v = float(last['CCI']) if pd.notna(last['CCI']) else None
        willr_v = float(last['WILLR']) if pd.notna(last['WILLR']) else None
        osc_rows = [['القيمة', 'المؤشر'],
                    [f'{cci_v:.1f}' if cci_v is not None else '-', 'CCI (20)'],
                    [f'{willr_v:.1f}' if willr_v is not None else '-', 'Williams %R (14)']]
        y = self._table(y, osc_rows, [CW*0.35, CW*0.65])

        # Divergences
        y = self._stitle(y, 'التباعد بين السعر والمؤشرات (Divergences)')
        if divergences:
            div_rows = [['النوع', 'المؤشر']]
            for ind, ar_type, en_type in divergences: div_rows.append([ar_type, ind])
            y = self._table(y, div_rows, [CW*0.60, CW*0.40])
        else:
            c.setFillColor(DGRAY); self._font(False, 8.5)
            c.drawRightString(PAGE_W-MG, y, rtl('لا يوجد تباعد مرصود.')); y -= 16

        # Candle patterns table
        y = self._stitle(y, 'نماذج الشموع اليابانية المرصودة')
        if patterns:
            pat_rows = [['التوجه', 'النموذج', 'التاريخ']]
            for date_lbl, ar_name, en_name, bullish in patterns:
                direction = 'صعودي' if bullish is True else ('هبوطي' if bullish is False else 'محايد')
                pat_rows.append([direction, ar_name, date_lbl])
            y = self._table(y, pat_rows, [CW*0.20, CW*0.45, CW*0.35], sig_mode=True)
        else:
            c.setFillColor(DGRAY); self._font(False, 8.5)
            c.drawRightString(PAGE_W-MG, y, rtl('لم يُرصد نموذج شموع بارز.'))
        self.c.showPage()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 9: COMPREHENSIVE TECHNICAL REVIEW (Arabic paragraphs)
    # ══════════════════════════════════════════════════════════════════════════
    def review_page(self, review_sections, score):
        self._bar('المراجعة الفنية الشاملة'); self._foot()
        c = self.c; y = PAGE_H - 44*mm
        line_h = 13; para_gap = 8; section_gap = 6
        font_size_body = 9; max_text_width = CW - 4*mm

        # ── Score strip ──
        strip_h = 10*mm; strip_y = y - strip_h
        if score >= 14:   strip_fill = HexColor('#1B5E20')
        elif score >= 10: strip_fill = HexColor('#1B2A4A')
        elif score >= 7:  strip_fill = HexColor('#E65100')
        else:             strip_fill = HexColor('#B71C1C')
        c.setFillColor(strip_fill)
        c.roundRect(MG, strip_y, CW, strip_h, 5, fill=1, stroke=0)
        c.setFillColor(WHITE); self._font(True, 12)
        c.drawCentredString(PAGE_W/2, strip_y + (strip_h - font_size_body) / 2,
                           rtl(f'نتيجة التحليل الفني: {score} من أصل 20 نقطة'))
        y = strip_y - 10*mm

        # ── Render each review section ──
        for section_title, paragraph in review_sections:
            if y < 40*mm:
                c.showPage(); self._bar('المراجعة الفنية الشاملة (تابع)')
                self._foot(); y = PAGE_H - 44*mm
            y = self._stitle(y, section_title); y += 4
            lines = self._wrap_arabic_text(paragraph, max_text_width, font_size_body)
            c.setFillColor(TXTDARK); self._font(False, font_size_body)
            for line in lines:
                if y < 25*mm:
                    c.showPage(); self._bar('المراجعة الفنية الشاملة (تابع)')
                    self._foot(); y = PAGE_H - 44*mm
                c.drawRightString(PAGE_W - MG, y, line); y -= line_h
            y -= para_gap + section_gap

        # ── Final disclaimer ──
        c.setFillColor(DGRAY); self._font(False, 6.5)
        c.drawCentredString(PAGE_W/2, 16*mm, rtl('هذا التقرير آلي لأغراض معلوماتية فقط وليس نصيحة استثمارية.'))
        c.drawCentredString(PAGE_W/2, 12*mm, rtl('الأداء السابق لا يضمن النتائج المستقبلية. قم دائماً ببحثك الخاص.'))
        c.showPage()

    # ── Save PDF ──
    def save(self):
        """Finalize and save the PDF."""
        self.c.save()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: REPORT BUILDER (Thread Executor)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Orchestrate the entire PDF report generation.
#           Runs in a ThreadPoolExecutor to avoid blocking the Telegram bot.
#
# STEPS:
#   1. Resolve ticker (add .SR if needed)
#   2. Fetch data (yfinance + STOCKS enrichment)
#   3. Compute all indicators
#   4. Calculate performance, signals, score, S/R, patterns, divergences
#   5. Generate technical review text
#   6. Generate all chart images (15 charts)
#   7. Build PDF report (9 pages)
#   8. Return (pdf_bytes, summary_text, display_ticker)
#
# TO MODIFY:
#   • Add new chart → add make_xxx_chart() call + pass to report page
#   • Remove page   → delete the rpt.xxx_page() call
#   • Change page order → reorder rpt.xxx_page() calls
#   • Change thread pool → modify max_workers=4
# ═══════════════════════════════════════════════════════════════════════════════

_executor = ThreadPoolExecutor(max_workers=4)


def _build_report_sync(ticker_input: str):
    """
    Build complete PDF report synchronously (runs in thread pool).
    
    Args:
        ticker_input: User input (e.g., "2222", "الراجحي", "TASI")
    
    Returns:
        tuple: (pdf_bytesio, summary_markdown, display_ticker)
    
    Raises:
        ValueError: If ticker not found or insufficient data
    """
    ti = ticker_input.strip()
    if not ti:
        raise ValueError('رمز فارغ')

    # ── Resolve ticker ──
    if ti.replace('.', '').isdigit() and '.' not in ti:
        ticker = ti + '.SR'; display_ticker = ti
    else:
        ticker = ti.upper(); display_ticker = ticker

    # ── Fetch data ──
    df, df2, info = fetch_data(ticker)
    if df is None or len(df) < 30:
        raise ValueError(
            f'❌ البيانات غير كافية للرمز: {display_ticker}\n'
            'تأكد من صحة رقم الشركة.')

    # ── Compute everything ──
    d = compute_indicators(df)
    pers, risk, dd = compute_performance(df, df2)
    sig, _ = gen_signals(d)
    score_criteria, score = compute_score_criteria(d)
    sup, res = find_sr(df, d_ind=d)
    rec_txt, rec_color = recommendation(score)
    patterns = detect_candle_patterns(df)
    divergences = detect_divergences(d)
    review_sections = gen_technical_review(
        d, sig, score, sup, res, info,
        patterns=patterns, divergences=divergences)

    # ── Generate all chart images ──
    main_img       = make_main_chart(d, sup=sup, res=res)
    p_img          = make_price_chart(d, sup=sup, res=res)
    ema_img        = make_ema_chart(d, sup=sup, res=res)
    alligator_img  = make_alligator_chart(d)
    supertrend_img = make_supertrend_chart(d)
    t_img          = make_tech_chart(d)
    bb_img         = make_bb_chart(d)
    dd_img         = make_dd_chart(dd)
    v_img          = make_volume_chart(d)
    g_img          = make_gauge_chart(score)
    ichi_img       = make_ichimoku_chart(d)
    cpat_img       = make_candle_pattern_chart(d, patterns)
    cci_img        = make_cci_willr_chart(d)

    # ── Build PDF ──
    price = float(df['Close'].iloc[-1])
    prev  = float(df['Close'].iloc[-2]) if len(df) > 1 else price
    chg   = (price / prev - 1) * 100

    pdf_buf = BytesIO()
    rpt = Report(pdf_buf, ticker, info, display_ticker)

    # Page order:
    rpt.cover(price, chg, info, rec_txt, rec_color, score)       # Page 1: Cover
    rpt.main_charts_page(main_img, p_img, ema_img,                # Page 2: Charts
                         alligator_img, supertrend_img)
    rpt.tech_page(t_img, bb_img)                                  # Page 3: Tech indicators
    rpt.ichimoku_page(ichi_img, cpat_img)                          # Page 4: Ichimoku + patterns
    rpt.perf_page(pers, risk, dd_img, v_img,                      # Page 5-6: Performance
                  score_criteria, score)
    rpt.fund_page(info)                                            # Page 7: Fundamentals
    rpt.signal_page(g_img, sig, score, sup, res, d)                # Page 8: Signals
    rpt.cci_willr_page(cci_img, patterns, divergences, d)          # Page 9: CCI/WR
    rpt.review_page(review_sections, score)                        # Page 10: Review

    rpt.save()
    pdf_buf.seek(0)

    # ── Summary message for Telegram ──
    summary = (
        f"📊 *{display_ticker}*\n"
        f"السعر الحالي   : `{price:.2f}`\n"
        f"التغير اليومي  : `{chg:+.2f}%`\n"
        f"النتيجة        : `{score}/20`\n"
        f"التحليل        : `{rec_txt}`\n"
        f"الصفحات        : `{rpt.pn}`"
    )
    return pdf_buf, summary, display_ticker

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: WOLFE WAVE DETECTION & CHARTING
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Detect Wolfe Wave patterns in price data and generate annotated charts.
#           Used by the Wolfe Wave Scanner bot feature.
#
# WOLFE WAVE THEORY:
#   A Wolfe Wave is a natural repetitive pattern found in all markets.
#   It consists of 5 waves:
#     - Waves 1 & 2: Initial move
#     - Wave 3: Shortest wave
#     - Wave 4: Retracement
#     - Wave 5: Final move (equilibrium line projected from 1→4)
#
# VALIDATION RULES (Bullish):
#   - Wave 1 must be a high
#   - Waves alternate: H-L-H-L-H
#   - Wave 3 is shorter than Wave 1
#   - Wave 4 is shallower than Wave 2
#   - Equilibrium line (1→4) predicts Wave 5 target
#
# PATTERN TYPES:
#   - Bullish Wolfe Wave: Point 1 is HIGH, targets upward
#   - Bearish Wolfe Wave: Point 1 is LOW, targets downward
#
# TO MODIFY:
#   • Change sensitivity  → modify tolerance (0.03 default)
#   • Add more patterns   → add new validation functions
#   • Change target calc → modify target_price in find_active_wolfe()
# ═══════════════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────────────
# 10a: PIVOT FINDER FOR WOLFE WAVES
# ───────────────────────────────────────────────────────────────────────────────

def find_pivots(df, order=5):
    """Find local pivot highs and lows in price data."""
    high = df['High'].values
    low = df['Low'].values
    sh = argrelextrema(high, np.greater_equal, order=order)[0]
    sl = argrelextrema(low, np.less_equal, order=order)[0]
    pivots = []
    for i in sh:
        pivots.append({'bar': int(i), 'price': high[i], 'type': 'H', 'date': df.index[i]})
    for i in sl:
        pivots.append({'bar': int(i), 'price': low[i], 'type': 'L', 'date': df.index[i]})
    pivots.sort(key=lambda x: x['bar'])
    return pivots


def get_alternating_pivots(pivots):
    """Filter pivots to ensure strict alternating H-L-H-L pattern."""
    if not pivots:
        return []
    alt = [pivots[0]]
    for p in pivots[1:]:
        if p['type'] == alt[-1]['type']:
            # Keep the more extreme one
            if p['type'] == 'H' and p['price'] > alt[-1]['price']:
                alt[-1] = p
            elif p['type'] == 'L' and p['price'] < alt[-1]['price']:
                alt[-1] = p
        else:
            alt.append(p)
    return alt


def line_at(x, x1, y1, x2, y2):
    """Linear interpolation: y at position x given two points (x1,y1) and (x2,y2)."""
    if x2 == x1:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def resample_ohlc(df, rule):
    """Resample OHLC data to different timeframe (e.g., '1h', '2h', '4h')."""
    return df.resample(rule).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna()


# ───────────────────────────────────────────────────────────────────────────────
# 10b: WOLFE WAVE VALIDATION
# ───────────────────────────────────────────────────────────────────────────────

def validate_bullish(p0, p1, p2, p3, p4, p5, tol=0.03):
    """
    Validate a Bullish Wolfe Wave pattern.
    
    Rules:
      - p0 must be HIGH
      - Pattern must alternate: H-L-H-L-H
      - Wave 3 shorter than Wave 1
      - Wave 4 shallower than Wave 2
      - Wave 5 projects from equilibrium line (1→4)
    
    Returns: dict with direction, points, entry_price, p5_date, target_price
    """
    v = [p['price'] for p in [p1, p2, p3, p4, p5]]
    b = [p['bar'] for p in [p1, p2, p3, p4, p5]]
    
    if p0['type'] != 'H':
        return None
    if not (p1['type'] == 'L' and p2['type'] == 'H' and 
            p3['type'] == 'L' and p4['type'] == 'H' and p5['type'] == 'L'):
        return None
    if p0['price'] <= p2['price']:  # Wave 1 must be highest
        return None
    if v[2] >= v[0] or v[3] >= v[1] or v[3] <= v[0] or v[4] >= v[2]:
        return None  # Wave geometry violations
    
    # Slope check
    s13 = (v[2] - v[0]) / (b[2] - b[0]) if b[2] != b[0] else 0
    s24 = (v[3] - v[1]) / (b[3] - b[1]) if b[3] != b[1] else 0
    if s13 >= 0 or s24 >= 0 or s13 >= s24:
        return None
    
    # Equilibrium line check (1→4 projected to p5)
    proj = line_at(b[4], b[0], v[0], b[2], v[2])
    if proj != 0 and (proj - v[4]) / abs(proj) < -tol:
        return None
    
    return {
        'direction': 'Bullish',
        'points': [p0, p1, p2, p3, p4, p5],
        'entry_price': v[4],
        'p5_date': p5['date']
    }


def validate_bearish(p0, p1, p2, p3, p4, p5, tol=0.03):
    """Validate a Bearish Wolfe Wave pattern. Mirror of bullish rules."""
    v = [p['price'] for p in [p1, p2, p3, p4, p5]]
    b = [p['bar'] for p in [p1, p2, p3, p4, p5]]
    
    if p0['type'] != 'L':
        return None
    if not (p1['type'] == 'H' and p2['type'] == 'L' and
            p3['type'] == 'H' and p4['type'] == 'L' and p5['type'] == 'H'):
        return None
    if p0['price'] >= p2['price']:
        return None
    if v[2] <= v[0] or v[3] <= v[1] or v[3] >= v[0] or v[4] <= v[2]:
        return None
    
    s13 = (v[2] - v[0]) / (b[2] - b[0]) if b[2] != b[0] else 0
    s24 = (v[3] - v[1]) / (b[3] - b[1]) if b[3] != b[1] else 0
    if s13 <= 0 or s24 <= 0 or s13 >= s24:
        return None
    
    proj = line_at(b[4], b[0], v[0], b[2], v[2])
    if proj != 0 and (v[4] - proj) / abs(proj) < -tol:
        return None
    
    return {
        'direction': 'Bearish',
        'points': [p0, p1, p2, p3, p4, p5],
        'entry_price': v[4],
        'p5_date': p5['date']
    }


# ───────────────────────────────────────────────────────────────────────────────
# 10c: WOLFE WAVE PATTERN FINDER
# ───────────────────────────────────────────────────────────────────────────────

def find_active_wolfe(df, max_bars_since_p5=8):
    """
    Scan for active Wolfe Wave patterns.
    
    Tests multiple pivot orders and returns the most recent valid patterns.
    """
    n = len(df)
    best_bull = None
    best_bear = None
    
    for order in [4, 5, 6, 7]:
        piv = get_alternating_pivots(find_pivots(df, order=order))
        if len(piv) < 6:
            continue
        
        for offset in range(min(4, len(piv) - 5)):
            idx = len(piv) - 6 - offset
            if idx < 0:
                break
            combo = piv[idx:idx + 6]
            
            # Skip if p5 is too old
            if n - 1 - combo[5]['bar'] > max_bars_since_p5:
                continue
            
            r = validate_bullish(*combo)
            if r and (best_bull is None or combo[5]['bar'] > best_bull['points'][5]['bar']):
                best_bull = r
            
            r = validate_bearish(*combo)
            if r and (best_bear is None or combo[5]['bar'] > best_bear['points'][5]['bar']):
                best_bear = r
    
    return [x for x in [best_bull, best_bear] if x]


# ───────────────────────────────────────────────────────────────────────────────
# 10d: WOLFE WAVE CHART PLOTTING
# ───────────────────────────────────────────────────────────────────────────────

def plot_wolfe_chart(ticker, df, result, tf_label):
    """
    Generate annotated Wolfe Wave chart with:
      - Price candles
      - Wave points (P0-P5) marked
      - Equilibrium line (1→4) projected
      - Target price marked
      - Entry price marked
      - Arabic labels
    
    Returns: BytesIO PNG buffer
    """
    pts = result['points']
    direction = result['direction']
    entry = result['entry_price']
    is_bull = direction == 'Bullish'
    
    # Get company name and ticker
    company = get_name(ticker)
    ticker_code = ticker.split('.')[0]
    
    # Extract price values and bar indices
    b = [p['bar'] for p in pts]
    v = [p['price'] for p in pts]
    
    # Last bar and target price
    last_bar = len(df) - 1
    last_close = float(df['Close'].iloc[-1])
    pct = ((result['target_price'] - entry) / entry) * 100
    
    # Chart padding
    pad_l = max(0, b[0] - 10)
    pad_r = min(last_bar, b[5] + 30)
    df_z = df.iloc[pad_l:pad_r + 1].copy()
    off = pad_l
    zb = [x - off for x in b]
    n_z = len(df_z)
    
    # Colors
    C_W = '#0D47A1' if is_bull else '#B71C1C'
    C_T = '#2E7D32' if is_bull else '#C62828'
    C_24 = '#E65100'
    C_E = '#6A1B9A'
    C_A = '#00695C' if is_bull else '#880E4F'
    C_P0 = '#FF6F00'
    
    # Create chart
    mc = mpf.make_marketcolors(up='#26A69A', down='#EF5350', edge='inherit',
                               wick='inherit')
    sty = mpf.make_mpf_style(marketcolors=mc, gridcolor='#EEEEEE',
                             gridstyle='-', facecolor='#FAFBFC',
                             y_on_right=False, rc={'font.size': 10,
                                                   'grid.alpha': 0.2})
    fig, axes = mpf.plot(df_z, type='candle', style=sty, figsize=(16, 8),
                         returnfig=True, volume=False)
    ax = axes[0]
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.06)
    
    # Draw Wolfe Wave lines
    ax.plot(zb, v, color=C_W, lw=2.5, zorder=6, alpha=0.8)
    ax.scatter(zb, v, s=120, c='white', edgecolors=C_W,
               linewidths=2.5, zorder=7)
    
    # P0 marker (different color)
    ax.scatter([zb[0]], [v[0]], s=160, c='white', edgecolors=C_P0,
               linewidths=3, zorder=8)
    
    # Extended equilibrium line (1→4)
    ext = zb[5] + 8
    ax.plot([zb[1], ext], [v[1], line_at(ext + off, b[1], v[1], b[3], v[3])],
            color=C_W, lw=1.0, ls='--', alpha=0.3)
    ax.plot([zb[2], ext], [v[2], line_at(ext + off, b[2], v[2], b[4], v[4])],
            color=C_24, lw=1.0, ls='--', alpha=0.3)
    
    # Fill between lines 1-3 and 2-4 (wave channel)
    fx = np.arange(zb[1], zb[5] + 1)
    f1 = [line_at(x + off, b[1], v[1], b[3], v[3]) for x in fx]
    f2 = [line_at(x + off, b[2], v[2], b[4], v[4]) for x in fx]
    ax.fill_between(fx, f1, f2, alpha=0.04, color=C_W)
    
    # Target line (4→1 projection)
    tgt_end_zb = n_z + 5
    ax.plot([zb[1], tgt_end_zb], [v[1], line_at(tgt_end_zb + off, b[1], v[1], b[4], v[4])],
            color=C_T, lw=3.0, ls='-.', alpha=0.85, zorder=5)
    
    # Target marker
    z_last = min(last_bar - off, n_z - 1)
    ax.plot(z_last, result['target_price'], marker='D', ms=14, color=C_T,
            markeredgecolor='white', markeredgewidth=2, zorder=9)
    
    # Horizontal lines for entry and target
    ax.axhline(y=result['target_price'], color=C_T, lw=0.6, ls=':', alpha=0.25)
    ax.axhline(y=entry, color=C_E, lw=0.6, ls=':', alpha=0.25)
    
    # Arrow annotation for percentage
    arrow_land_zb = min(zb[5] + max(4, (z_last - zb[5]) // 2), n_z + 3)
    arrow_land_price = line_at(arrow_land_zb + off, b[1], v[1], b[4], v[4])
    ax.annotate('',
                xy=(arrow_land_zb, arrow_land_price),
                xytext=(zb[5], entry),
                arrowprops=dict(arrowstyle='-|>', color=C_A, lw=3.0,
                              mutation_scale=22,
                              connectionstyle='arc3,rad=0.15' if is_bull else 'arc3,rad=-0.15'),
                zorder=8)
    
    # Percentage label
    price_range = max(v) - min(v)
    label_offset = price_range * 0.08
    pct_y = arrow_land_price + label_offset if is_bull else arrow_land_price - label_offset
    ax.text(arrow_land_zb, pct_y, f'{pct:+.1f}%', fontsize=13,
            fontweight='bold', color=C_A, ha='center',
            va='bottom' if is_bull else 'top',
            fontfamily=ARABIC_FONT,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_A,
                     alpha=0.9, lw=0.8),
            zorder=10)
    
    # P0-P5 labels
    for i in range(6):
        is_low = pts[i]['type'] == 'L'
        dt_str = pts[i]['date'].strftime('%b %d')
        label_color = C_P0 if i == 0 else C_W
        ax.annotate(f'P{i}  {v[i]:.2f}\n{dt_str}',
                    xy=(zb[i], v[i]),
                    xytext=(0, -28 if is_low else 28),
                    textcoords='offset points',
                    ha='center', va='top' if is_low else 'bottom',
                    fontsize=8.5, fontweight='bold', color=label_color,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white',
                            edgecolor=label_color, alpha=0.9, lw=0.6),
                    arrowprops=dict(arrowstyle='-', color=label_color, lw=0.6))
    
    # Title with Arabic
    emoji = '📈' if is_bull else '📉'
    direction_ar = ar('ولفي صاعد') if is_bull else ar('ولفي هابط')
    ax.set_title(f'{emoji}  {ticker_code}  |  {ar(company)}  |  {direction_ar}  |  {ar(tf_label)}',
                 fontsize=15, fontweight='bold', pad=16,
                 color='#212121', fontfamily=ARABIC_FONT)
    ax.set_ylabel('')
    
    # Info box (bottom-left)
    bc = '#E8F5E9' if is_bull else '#FFEBEE'
    bt = '#2E7D32' if is_bull else '#C62828'
    info_lines = [
        f"  {ar(company)}",
        f"  {'─'*24}",
        f"  {last_close:.2f}        :  {ar('الإغلاق')}",
        f"  {entry:.2f}        :  {ar('الموجة 5')}",
        f"  {result['target_price']:.2f}       :  {ar('4 ← 1')}",
        f"  {pct:+.1f}%        :  {ar('النسبة')}",
        f"  {ar(tf_label)}     :  {ar('الفاصل')}"
    ]
    ax.text(0.01, 0.03, '\n'.join(info_lines),
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            color=bt, fontfamily=ARABIC_FONT,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=bc,
                     edgecolor=bt, alpha=0.92, lw=1.2),
            zorder=10)
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


# ───────────────────────────────────────────────────────────────────────────────
# 10e: TICKER PROCESSOR FOR SCANNER
# ───────────────────────────────────────────────────────────────────────────────

def process_ticker(ticker, period, interval, resample_rule=None):
    """Process a single ticker: fetch data and find Wolfe patterns."""
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df is None or len(df) < 30:
            return ticker, [], None
        if resample_rule:
            df = resample_ohlc(df, resample_rule)
        if len(df) < 30:
            return ticker, [], None
        
        found = find_active_wolfe(df, max_bars_since_p5=8)
        last_bar = len(df) - 1
        
        for r in found:
            # Calculate target price: project equilibrium line (1→4) to last bar
            b1 = r['points'][1]['bar']
            v1 = r['points'][1]['price']
            b4 = r['points'][4]['bar']
            v4 = r['points'][4]['price']
            r['target_price'] = round(line_at(last_bar, b1, v1, b4, v4), 2)
            r['last_close'] = round(float(df['Close'].iloc[-1]), 2)
        
        return ticker, found, df
    except Exception:
        return ticker, [], None


# ───────────────────────────────────────────────────────────────────────────────
# 10f: SCAN ALL TICKERS
# ───────────────────────────────────────────────────────────────────────────────

def scan_tickers(tickers, period, interval, resample_rule=None, max_workers=15):
    """Scan multiple tickers in parallel using ThreadPoolExecutor."""
    all_res = {}
    ohlc = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(process_ticker, t, period, interval, resample_rule): t
                for t in tickers}
        for f in as_completed(futs):
            tk, found, df = f.result()
            if found:
                all_res[tk] = found
                ohlc[tk] = df
    return all_res, ohlc


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: TICKER LISTS & TIMEFRAME MAP
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Define all Tadawul tickers and timeframe options for Wolfe scanner.
#
# TICKER FORMAT: XXXX.SR (Tadawul format with Saudi Arabia suffix)
#
# TIMEFRAMES:
#   '30m' → 30 minutes, 60d history
#   '1h'  → 1 hour, 60d history
#   '2h'  → 2 hours, 60d history (resampled from 1h)
#   '4h'  → 4 hours, 60d history (resampled from 1h)
#   '1d'  → Daily, 1 year history
#   '1w'  → Weekly, 5 year history
#
# TO MODIFY:
#   • Add new ticker → append to TADAWUL_TICKERS list
#   • Add new timeframe → add entry to TF_MAP
# ═══════════════════════════════════════════════════════════════════════════════

# Full list of Tadawul stocks (all .SR tickers)
TADAWUL_TICKERS = [
    '^TASI.SR', '1010.SR', '1020.SR', '1030.SR', '1050.SR', '1060.SR', '1080.SR', '1111.SR',
    '1120.SR', '1140.SR', '1150.SR', '1180.SR', '1182.SR', '1183.SR', '1201.SR', '1202.SR',
    '1210.SR', '1211.SR', '1212.SR', '1213.SR', '1214.SR', '1301.SR', '1302.SR', '1303.SR',
    '1304.SR', '1320.SR', '1321.SR', '1322.SR', '1323.SR', '1810.SR', '1820.SR', '1830.SR',
    '1831.SR', '1832.SR', '1833.SR', '1834.SR', '1835.SR', '2001.SR', '2010.SR', '2020.SR',
    '2030.SR', '2040.SR', '2050.SR', '2060.SR', '2070.SR', '2080.SR', '2081.SR', '2082.SR',
    '2083.SR', '2084.SR', '2090.SR', '2100.SR', '2110.SR', '2120.SR', '2130.SR', '2140.SR',
    '2150.SR', '2160.SR', '2170.SR', '2180.SR', '2190.SR', '2200.SR', '2210.SR', '2220.SR',
    '2222.SR', '2223.SR', '2230.SR', '2240.SR', '2250.SR', '2270.SR', '2280.SR', '2281.SR',
    '2282.SR', '2283.SR', '2284.SR', '2285.SR', '2286.SR', '2287.SR', '2288.SR', '2290.SR',
    '2300.SR', '2310.SR', '2320.SR', '2330.SR', '2340.SR', '2350.SR', '2360.SR', '2370.SR',
    '2380.SR', '2381.SR', '2382.SR', '3002.SR', '3003.SR', '3004.SR', '3005.SR', '3007.SR',
    '3008.SR', '3010.SR', '3020.SR', '3030.SR', '3040.SR', '3050.SR', '3060.SR', '3080.SR',
    '3090.SR', '3091.SR', '3092.SR', '4001.SR', '4002.SR', '4003.SR', '4004.SR', '4005.SR',
    '4006.SR', '4007.SR', '4008.SR', '4009.SR', '4011.SR', '4012.SR', '4013.SR', '4014.SR',
    '4015.SR', '4016.SR', '4017.SR', '4018.SR', '4019.SR', '4020.SR', '4021.SR', '4030.SR',
    '4031.SR', '4040.SR', '4050.SR', '4051.SR', '4061.SR', '4070.SR', '4071.SR', '4072.SR',
    '4080.SR', '4081.SR', '4082.SR', '4083.SR', '4084.SR', '4090.SR', '4100.SR', '4110.SR',
    '4130.SR', '4140.SR', '4141.SR', '4142.SR', '4143.SR', '4144.SR', '4145.SR', '4146.SR',
    '4147.SR', '4148.SR', '4150.SR', '4160.SR', '4161.SR', '4162.SR', '4163.SR', '4164.SR',
    '4165.SR', '4170.SR', '4180.SR', '4190.SR', '4191.SR', '4192.SR', '4193.SR', '4194.SR',
    '4200.SR', '4210.SR', '4220.SR', '4230.SR', '4240.SR', '4250.SR', '4260.SR', '4261.SR',
    '4262.SR', '4263.SR', '4264.SR', '4265.SR', '4270.SR', '4280.SR', '4290.SR', '4291.SR',
    '4292.SR', '4300.SR', '4310.SR', '4320.SR', '4321.SR', '4322.SR', '4323.SR', '4324.SR',
    '4325.SR', '4326.SR', '4327.SR', '4330.SR', '4331.SR', '4332.SR', '4333.SR', '4334.SR',
    '4335.SR', '4336.SR', '4337.SR', '4338.SR', '4339.SR', '4340.SR', '4342.SR', '4344.SR',
    '4345.SR', '4346.SR', '4347.SR', '4348.SR', '4349.SR', '4350.SR', '5110.SR', '6001.SR',
    '6002.SR', '6004.SR', '6010.SR', '6012.SR', '6013.SR', '6014.SR', '6015.SR', '6016.SR',
    '6017.SR', '6018.SR', '6019.SR', '6020.SR', '6040.SR', '6050.SR', '6060.SR', '6070.SR',
    '6090.SR', '7010.SR', '7020.SR', '7030.SR', '7040.SR', '7200.SR', '7201.SR', '7202.SR',
    '7203.SR', '7204.SR', '7211.SR', '8010.SR', '8012.SR', '8020.SR', '8030.SR', '8040.SR',
    '8050.SR', '8060.SR', '8070.SR', '8100.SR', '8120.SR', '8150.SR', '8160.SR', '8170.SR',
    '8180.SR', '8190.SR', '8200.SR', '8210.SR', '8230.SR', '8240.SR', '8250.SR', '8260.SR',
    '8270.SR', '8280.SR', '8300.SR', '8310.SR', '8311.SR', '8313.SR',
]

# Timeframe configuration: (Arabic label, yfinance interval, yfinance period, resample rule)
TF_MAP = {
    '30m': ('30 دقيقة', '30m',  '60d', None),
    '1h':  ('1 ساعة',   '60m',  '60d', None),
    '2h':  ('2 ساعة',   '60m',  '60d', '2h'),
    '4h':  ('4 ساعات',  '60m',  '60d', '4h'),
    '1d':  ('يومي',     '1d',   '1y',  None),
    '1w':  ('أسبوعي',   '1wk',  '5y',  None),
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: TELEGRAM KEYBOARD BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Create inline keyboards for bot navigation.
#
# KEYBOARDS:
#   build_main_keyboard()        → Main menu (Wolfe / Analyzer)
#   build_tf_keyboard()         → Timeframe selection (30m, 1h, 2h, 4h, 1d, 1w)
#   build_filter_keyboard()      → Bullish / Bearish / All filter
#   build_after_wolfe_keyboard() → Scan again / Main menu
#   build_back_main_keyboard()   → Back to main menu
#
# TO MODIFY:
#   • Add button    → add InlineKeyboardButton
#   • Change label  → modify button text
#   • Change action → modify callback_data
# ═══════════════════════════════════════════════════════════════════════════════

def build_main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 بوت موجات الولفي", callback_data="bot_wolfe")],
        [InlineKeyboardButton("📊 بوت المحلل الرقمي", callback_data="bot_analyzer")],
    ])


def build_tf_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("30 دقيقة", callback_data="scan_30m"),
         InlineKeyboardButton("1 ساعة", callback_data="scan_1h")],
        [InlineKeyboardButton("2 ساعة",  callback_data="scan_2h"),
         InlineKeyboardButton("4 ساعات",callback_data="scan_4h")],
        [InlineKeyboardButton("يومي",    callback_data="scan_1d"),
         InlineKeyboardButton("أسبوعي", callback_data="scan_1w")],
        [InlineKeyboardButton("🔙 رجوع للقائمة الرئيسية", callback_data="back_to_main")],
    ])


def build_filter_keyboard(tf_key):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 صاعد فقط", callback_data=f"filter_{tf_key}_bullish"),
         InlineKeyboardButton("📉 هابط فقط", callback_data=f"filter_{tf_key}_bearish")],
        [InlineKeyboardButton("📊 الكل",      callback_data=f"filter_{tf_key}_both")],
        [InlineKeyboardButton("🔙 رجوع", callback_data="back_to_wolfe")],
    ])


def build_after_wolfe_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 فحص جديد", callback_data="bot_wolfe")],
        [InlineKeyboardButton("🏠 القائمة الرئيسية", callback_data="back_to_main")],
    ])


def build_back_main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔙 رجوع للقائمة الرئيسية", callback_data="back_to_main")],
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13: BOT MESSAGES (Arabic)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Static text messages displayed by the bot.
#
# MESSAGES:
#   MAIN_MENU_MSG       → Welcome + feature selection
#   WOLFE_WELCOME_MSG  → Timeframe selection prompt
#   ANALYZER_MSG       → How to use analyzer bot
#   LANDING_HTML       → Web page for Render.com
#
# TO MODIFY:
#   • Change language      → edit Arabic text
#   • Add message        → define new constant
#   • Change formatting   → modify markdown
# ═══════════════════════════════════════════════════════════════════════════════

MAIN_MENU_MSG = (
    "🤖 *مرحباً بك في البوت المتكامل*\n\n"
    "اختر الخدمة التي تريدها:\n\n"
    "📈 *بوت موجات الولفي* — فحص السوق السعودي\n"
    "📊 *بوت المحلل الرقمي* — تقرير PDF شامل للسهم"
)

WOLFE_WELCOME_MSG = (
    "🎯 *فاحص موجات الولفي ويف — السوق السعودي*\n\n"
    "اختر الفاصل الزمني للفحص:\n\n"
    "⚠️ تنبيه: هذا بحث عن موجات الولفي ويف فقط، "
    "لا يجب الاعتماد عليه وقد يكون خطأ. "
    "يجب متابعة الحركة السعرية."
)

ANALYZER_MSG = (
    "📊 *بوت المحلل الرقمي*\n\n"
    "أرسل *رقم الشركة* وسيتم إنشاء تقرير PDF شامل يشمل:\n"
    "• المؤشرات الفنية والرسوم البيانية\n"
    "• الأداء والمخاطر\n"
    "• الإيشيموكو ونماذج الشموع\n"
    "• مراجعة فنية شاملة\n\n"
    "_مثال: أرسل_ `2222` _للحصول على تقرير أرامكو_"
)

LANDING_HTML = """<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head><meta charset="UTF-8"><title>بوت السوق السعودي</title>
<style>
body{font-family:Arial,sans-serif;background:#080c1a;color:#fff;
     display:flex;align-items:center;justify-content:center;
     min-height:100vh;margin:0;}
.card{background:rgba(255,255,255,0.07);padding:40px;border-radius:16px;
      text-align:center;max-width:480px;}
h1{color:#7c6df5;font-size:2rem;}
p{color:#aaa;}
</style></head>
<body>
<div class="card">
<h1>🤖 بوت السوق السعودي</h1>
<p>بوت موجات الولفي ويف + المحلل الرقمي</p>
<p style="color:#7c6df5;">ابدأ المحادثة على تيليغرام</p>
</div>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14: TELEGRAM BOT HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Handle all Telegram bot events (commands, callbacks, messages).
#
# HANDLERS:
#   start_command()      → /start command → show main menu
#   button_handler()     → All inline button callbacks
#   text_handler()      → Text messages for analyzer ticker input
#
# TO MODIFY:
#   • Add new command     → add CommandHandler
#   • Add new callback  → add elif in button_handler()
#   • Change behavior   → modify handler functions
# ═══════════════════════════════════════════════════════════════════════════════

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    context.user_data.pop('waiting_ticker', None)
    await update.message.reply_text(
        MAIN_MENU_MSG, parse_mode="Markdown",
        reply_markup=build_main_keyboard(),
    )


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all inline keyboard button presses."""
    query = update.callback_query
    await query.answer()
    data = query.data

    # ── Back to main menu ──
    if data == "back_to_main":
        context.user_data.pop('waiting_ticker', None)
        await query.edit_message_text(
            MAIN_MENU_MSG, parse_mode="Markdown",
            reply_markup=build_main_keyboard(),
        )
        return

    # ── Start Wolfe scanner ──
    if data == "bot_wolfe":
        context.user_data.pop('waiting_ticker', None)
        await query.edit_message_text(
            WOLFE_WELCOME_MSG, parse_mode="Markdown",
            reply_markup=build_tf_keyboard(),
        )
        return

    # ── Start Analyzer bot ──
    if data == "bot_analyzer":
        context.user_data['waiting_ticker'] = True
        await query.edit_message_text(
            ANALYZER_MSG, parse_mode="Markdown",
            reply_markup=build_back_main_keyboard(),
        )
        return

    # ── Back to Wolfe timeframe selection ──
    if data == "back_to_wolfe":
        await query.edit_message_text(
            WOLFE_WELCOME_MSG, parse_mode="Markdown",
            reply_markup=build_tf_keyboard(),
        )
        return

    # ── Timeframe selected ──
    if data.startswith("scan_"):
        tf_key = data[5:]
        if tf_key not in TF_MAP:
            await query.edit_message_text("فاصل زمني غير معروف.",
                                         reply_markup=build_back_main_keyboard())
            return
        await query.edit_message_text(
            f"⏱ الفاصل: *{TF_MAP[tf_key][0]}*\n\nاختر الفلتر:",
            parse_mode="Markdown",
            reply_markup=build_filter_keyboard(tf_key),
        )
        return

    # ── Filter selected ──
    if data.startswith("filter_"):
        parts = data.split("_", 2)
        if len(parts) != 3:
            await query.edit_message_text("بيانات غير صالحة.",
                                         reply_markup=build_back_main_keyboard())
            return
        _, tf_key, direction = parts
        if tf_key not in TF_MAP:
            await query.edit_message_text("فاصل زمني غير معروف.",
                                         reply_markup=build_back_main_keyboard())
            return

        tf_label, interval, period, resample_rule = TF_MAP[tf_key]
        chat_id = query.message.chat_id

        # Show loading message
        await query.edit_message_text(
            f"⏳ جاري فحص *{len(TADAWUL_TICKERS)}* سهم...\n"
            f"الفاصل: *{tf_label}*\n\nيرجى الانتظار ⏳",
            parse_mode="Markdown",
        )

        # Run scanner in thread pool
        loop = asyncio.get_event_loop()
        results, ohlc_data = await loop.run_in_executor(
            _executor, scan_tickers, TADAWUL_TICKERS, period, interval, resample_rule
        )

        # Separate bullish/bearish
        bullish_list = []
        bearish_list = []
        is_intraday = interval not in ('1d', '1wk')

        for tk, patterns in results.items():
            for r in patterns:
                pct = ((r['target_price'] - r['entry_price']) / r['entry_price']) * 100
                item = {
                    'ticker': tk,
                    'name': get_name(tk),
                    'last_close': r['last_close'],
                    'entry': round(r['entry_price'], 2),
                    'target': r['target_price'],
                    'pct': round(pct, 1),
                    'p5_date': (r['points'][5]['date'].strftime('%Y-%m-%d %H:%M')
                               if is_intraday else r['points'][5]['date'].strftime('%Y-%m-%d')),
                    '_r': r,
                    '_df': ohlc_data[tk],
                }
                if r['direction'] == 'Bullish':
                    bullish_list.append(item)
                else:
                    bearish_list.append(item)

        # Sort by percentage
        bullish_list.sort(key=lambda x: x['pct'], reverse=True)
        bearish_list.sort(key=lambda x: x['pct'])

        show_bull = direction in ('bullish', 'both')
        show_bear = direction in ('bearish', 'both')

        # Summary message
        summary = f"✅ *اكتمل الفحص — {tf_label}*\n\n"
        if show_bull:
            summary += f"📈 ولفي صاعد: *{len(bullish_list)}*\n"
        if show_bear:
            summary += f"📉 ولفي هابط: *{len(bearish_list)}*\n"
        if not bullish_list and not bearish_list:
            summary += "\nلا توجد نتائج لهذا الفلتر."

        await context.bot.send_message(chat_id=chat_id, text=summary,
                                      parse_mode="Markdown")

        # Send bullish results
        if show_bull and bullish_list:
            await context.bot.send_message(chat_id=chat_id,
                                          text="📈 *— نتائج الولفي الصاعد —*",
                                          parse_mode="Markdown")
            for item in bullish_list:
                try:
                    buf = plot_wolfe_chart(item['ticker'], item['_df'],
                                         item['_r'], tf_label)
                    await context.bot.send_photo(chat_id=chat_id, photo=buf)
                except Exception as e:
                    logger.error(f"Chart error {item['ticker']}: {e}")
                msg = (f"رمز السهم: *{item['ticker'].split('.')[0]}*\n"
                       f"الاسم       : `{item['name']}`\n"
                       f"الفاصل       : `{tf_label}`\n"
                       f"آخر إغلاق : `{item['last_close']}`\n"
                       f"قاع (5)    : `{item['entry']}`\n"
                       f"خط (1←4)  : `{item['target']}`\n"
                       f"النسبة      : `{item['pct']:+.1f}%`\n"
                       f"تاريخ (5)  : `{item['p5_date']}`")
                await context.bot.send_message(chat_id=chat_id, text=msg,
                                              parse_mode="Markdown")

        # Send bearish results
        if show_bear and bearish_list:
            await context.bot.send_message(chat_id=chat_id,
                                          text="📉 *— نتائج الولفي الهابط —*",
                                          parse_mode="Markdown")
            for item in bearish_list:
                try:
                    buf = plot_wolfe_chart(item['ticker'], item['_df'],
                                         item['_r'], tf_label)
                    await context.bot.send_photo(chat_id=chat_id, photo=buf)
                except Exception as e:
                    logger.error(f"Chart error {item['ticker']}: {e}")
                msg = (f"رمز السهم: *{item['ticker'].split('.')[0]}*\n"
                       f"الاسم       : `{item['name']}`\n"
                       f"الفاصل       : `{tf_label}`\n"
                       f"آخر إغلاق : `{item['last_close']}`\n"
                       f"قمة (5)    : `{item['entry']}`\n"
                       f"خط (1←4)  : `{item['target']}`\n"
                       f"النسبة      : `{item['pct']:+.1f}%`\n"
                       f"تاريخ (5)  : `{item['p5_date']}`")
                await context.bot.send_message(chat_id=chat_id, text=msg,
                                              parse_mode="Markdown")

        # Done message with keyboard
        await context.bot.send_message(
            chat_id=chat_id, text="🔄 *انتهى الفحص*",
            parse_mode="Markdown",
            reply_markup=build_after_wolfe_keyboard(),
        )
        return


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages for analyzer bot ticker input."""
    if not context.user_data.get('waiting_ticker'):
        return

    ticker_input = update.message.text.strip()
    chat_id = update.effective_chat.id

    proc_msg = await update.message.reply_text(
        f"⏳ جاري تحميل وتحليل بيانات *{ticker_input}*...\n"
        "قد يستغرق ذلك حتى دقيقتين، يرجى الانتظار 🔄",
        parse_mode="Markdown",
    )

    try:
        loop = asyncio.get_event_loop()
        pdf_buf, summary, display_ticker = await loop.run_in_executor(
            _executor, _build_report_sync, ticker_input
        )
        context.user_data.pop('waiting_ticker', None)

        # Success: send PDF
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=proc_msg.message_id,
            text=f"✅ *اكتمل التحليل — {display_ticker}*",
            parse_mode="Markdown",
        )
        await context.bot.send_document(
            chat_id=chat_id,
            document=pdf_buf,
            filename=f"{ticker_input.replace('.','_')}_Report.pdf",
            caption=summary,
            parse_mode="Markdown",
        )
        # Offer next action
        await context.bot.send_message(
            chat_id=chat_id,
            text="📊 هل تريد تحليل سهم آخر أو العودة للقائمة الرئيسية؟",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 تحليل سهم آخر", callback_data="bot_analyzer")],
                [InlineKeyboardButton("🏠 القائمة الرئيسية", callback_data="back_to_main")],
            ]),
        )

    except ValueError as e:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=proc_msg.message_id,
            text=str(e),
        )
        context.user_data['waiting_ticker'] = True
        await context.bot.send_message(
            chat_id=chat_id,
            text="أعد إدخال رقم الشركة أو اضغط رجوع:",
            reply_markup=build_back_main_keyboard(),
        )
    except Exception as e:
        logger.error(f"Report error: {e}", exc_info=True)
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=proc_msg.message_id,
            text=f"❌ حدث خطأ أثناء التحليل:\n`{str(e)[:200]}`",
            parse_mode="Markdown",
        )
        context.user_data['waiting_ticker'] = True
        await context.bot.send_message(
            chat_id=chat_id,
            text="أعد المحاولة أو اضغط رجوع:",
            reply_markup=build_back_main_keyboard(),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15: MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE : Start the Telegram bot application.
#
# MODES:
#   1. Webhook mode (Render.com): Uses RENDER_EXTERNAL_URL env var
#   2. Polling mode (local dev): Standard polling
#
# TO MODIFY:
#   • Change port      → modify PORT variable
#   • Add middleware   → app.middleware() before run
#   • Change logging   → modify logging.basicConfig()
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Start the bot application."""
    # Build application
    app = Application.builder().token(BOT_TOKEN).build()

    # Add handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    # Check environment
    RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
    PORT = int(os.environ.get("PORT", 10000))

    if RENDER_EXTERNAL_URL:
        # Webhook mode (for Render.com)
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"

        async def home(_request):
            return aio_web.Response(text=LANDING_HTML, content_type='text/html')

        async def webhook_route(request):
            data = await request.json()
            update = Update.de_json(data, app.bot)
            await app.update_queue.put(update)
            return aio_web.Response(text='OK')

        async def run_all():
            await app.bot.set_webhook(webhook_url)
            logger.info(f"Webhook set → {webhook_url}")
            web_app = aio_web.Application()
            web_app.router.add_get('/', home)
            web_app.router.add_post('/webhook', webhook_route)
            runner = aio_web.AppRunner(web_app)
            await runner.setup()
            await aio_web.TCPSite(runner, '0.0.0.0', PORT).start()
            logger.info(f"Server listening on port {PORT}")
            async with app:
                await app.start()
                await asyncio.Event().wait()
                await app.stop()

        asyncio.run(run_all())
    else:
        # Polling mode (local development)
        logger.info("Starting polling (local dev)")
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
