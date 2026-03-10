# ═══════════════════════════════════════════════════════════════
# PART 1: IMPORTS, CONFIGURATION, FONT SETUP, ARABIC HELPERS
# ═══════════════════════════════════════════════════════════════
#
# EXPLANATION:
# - All library imports are grouped: stdlib → third-party → project-specific
# - Font initialization downloads Cairo (for Wolfe charts) and Amiri (for PDF)
# - Arabic text helpers handle reshaping + BiDi for correct RTL rendering
# - To change fonts: modify CAIRO_PATH, AMIRI_REG_PATH, AMIRI_BOLD_PATH
# - To add new font: add download URL in _init_fonts(), register with pdfmetrics
# ═══════════════════════════════════════════════════════════════

import os
import io
import re
import asyncio
import logging
import warnings
import urllib.request
warnings.filterwarnings('ignore')

import arabic_reshaper
from bidi.algorithm import get_display
from aiohttp import web as aio_web
from io import BytesIO
from datetime import datetime, timedelta

import yfinance as yf
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import mplfinance as mpf

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes,
)

# ── Optional scraper dependencies (graceful fallback) ──
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

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]

# ─────────────────────────────────────────────────────────────
# FONT SETUP
# ─────────────────────────────────────────────────────────────
# Cairo: used for Wolfe wave chart labels (matplotlib)
# Amiri: used for PDF report text (ReportLab + matplotlib)
#
# HOW TO CHANGE:
# - Replace font URLs below with your preferred Arabic font
# - Update AR_FONT / AR_FONT_BOLD names if registering different fonts
# ─────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))

CAIRO_PATH      = os.path.join(HERE, 'Cairo-Regular.ttf')
AMIRI_REG_PATH  = os.path.join(HERE, 'Amiri-Regular.ttf')
AMIRI_BOLD_PATH = os.path.join(HERE, 'Amiri-Bold.ttf')

AR_FONT      = 'Amiri'
AR_FONT_BOLD = 'Amiri-Bold'
AR_RE = re.compile(r'[\u0600-\u06FF]')
MPL_FONT_PROP      = None
MPL_FONT_PROP_BOLD = None


def _download_font(url, path):
    """Download a font file if it doesn't exist locally."""
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
            logger.info(f'Downloaded: {os.path.basename(path)}')
        except Exception as e:
            logger.warning(f'Font download failed ({os.path.basename(path)}): {e}')


def _init_fonts():
    """Initialize all fonts for matplotlib and ReportLab."""
    global MPL_FONT_PROP, MPL_FONT_PROP_BOLD, ARABIC_FONT

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

    # Register fonts with ReportLab for PDF generation
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

    # Register fonts with matplotlib
    for fp in [AMIRI_REG_PATH, AMIRI_BOLD_PATH, CAIRO_PATH]:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)

    if os.path.exists(AMIRI_REG_PATH):
        MPL_FONT_PROP = fm.FontProperties(fname=AMIRI_REG_PATH)
        plt.rcParams['font.family'] = MPL_FONT_PROP.get_name()
    if os.path.exists(AMIRI_BOLD_PATH):
        MPL_FONT_PROP_BOLD = fm.FontProperties(fname=AMIRI_BOLD_PATH)
    plt.rcParams['axes.unicode_minus'] = False

    if os.path.exists(CAIRO_PATH):
        prop = fm.FontProperties(fname=CAIRO_PATH)
        ARABIC_FONT = prop.get_name()
    else:
        ARABIC_FONT = 'DejaVu Sans'
    return ARABIC_FONT


ARABIC_FONT = _init_fonts()
logger.info(f'Fonts initialised. Arabic chart font: {ARABIC_FONT}')

# ─────────────────────────────────────────────────────────────
# ARABIC TEXT HELPERS
# ─────────────────────────────────────────────────────────────
# These functions handle Arabic text reshaping and BiDi reordering.
# Arabic text must be reshaped (letters connected) and reordered
# (right-to-left) before rendering in matplotlib or ReportLab.
#
# ar()  → for Wolfe chart labels
# rtl() → for PDF / matplotlib Arabic text
# tx()  → safe wrapper returning '-' for None
# ─────────────────────────────────────────────────────────────

def ar(text: str) -> str:
    """Reshape + bidi for Wolfe chart labels."""
    try:
        return get_display(arabic_reshaper.reshape(str(text)))
    except Exception:
        return str(text)


def rtl(txt):
    """Reshape + bidi for PDF / matplotlib Arabic text."""
    if txt is None:
        return ''
    s = str(txt)
    if AR_RE.search(s):
        return get_display(arabic_reshaper.reshape(s))
    return s


def tx(txt):
    """Safe text: returns reshaped Arabic or '-' for None."""
    if txt is None:
        return '-'
    return rtl(str(txt))


def short_text(s, n=40):
    """Truncate text to n characters with ellipsis."""
    s = str(s or '-')
    return s if len(s) <= n else s[:n - 1] + '…'


def safe(info, key, default=None):
    """Safely get a value from info dict, returning default if None."""
    v = info.get(key)
    return default if v is None else v


# ═══════════════════════════════════════════════════════════════
# PART 2: COMPANY NAMES, TICKER LOOKUP, SECTOR MAP
# ═══════════════════════════════════════════════════════════════
#
# EXPLANATION:
# - COMPANY_NAMES: maps Yahoo Finance ticker → Arabic company name
# - TICKER_ALIASES: common alternative inputs (e.g. "تاسي" → "^TASI.SR")
# - find_ticker(): resolves user input to canonical ticker
# - SECTOR_MAP: maps ticker → (sector, industry) for Saudi market
#
# HOW TO MODIFY:
# - To add a new company: add entry to COMPANY_NAMES, SECTOR_MAP,
#   STOCKS_STATIC_DATA, and TADAWUL_TICKERS
# - To add alias: add to TICKER_ALIASES
# ═══════════════════════════════════════════════════════════════

COMPANY_NAMES = {
    '^TASI.SR': 'تاسي',
    '1010.SR': 'الرياض',
    '1020.SR': 'الجزيرة',
    '1030.SR': 'الإستثمار',
    '1050.SR': 'بي اس اف',
    '1060.SR': 'الأول',
    '1080.SR': 'العربي',
    '1111.SR': 'مجموعة تداول',
    '1120.SR': 'الراجحي',
    '1140.SR': 'البلاد',
    '1150.SR': 'الإنماء',
    '1180.SR': 'الأهلي',
    '1182.SR': 'أملاك',
    '1183.SR': 'سهل',
    '1201.SR': 'تكوين',
    '1202.SR': 'مبكو',
    '1210.SR': 'بي سي آي',
    '1211.SR': 'معادن',
    '1212.SR': 'أسترا الصناعية',
    '1213.SR': 'نسيج',
    '1214.SR': 'شاكر',
    '1301.SR': 'أسلاك',
    '1302.SR': 'بوان',
    '1303.SR': 'الصناعات الكهربائية',
    '1304.SR': 'اليمامة للحديد',
    '1320.SR': 'أنابيب السعودية',
    '1321.SR': 'أنابيب الشرق',
    '1322.SR': 'أماك',
    '1323.SR': 'يو سي آي سي',
    '1810.SR': 'سيرا',
    '1820.SR': 'بان',
    '1830.SR': 'لجام للرياضة',
    '1831.SR': 'مهارة',
    '1832.SR': 'صدر',
    '1833.SR': 'الموارد',
    '1834.SR': 'سماسكو',
    '1835.SR': 'تمكين',
    '2001.SR': 'كيمانول',
    '2010.SR': 'سابك',
    '2020.SR': 'سابك للمغذيات الزراعية',
    '2030.SR': 'المصافي',
    '2040.SR': 'الخزف السعودي',
    '2050.SR': 'مجموعة صافولا',
    '2060.SR': 'التصنيع',
    '2070.SR': 'الدوائية',
    '2080.SR': 'الغاز',
    '2081.SR': 'الخريف',
    '2082.SR': 'أكوا',
    '2083.SR': 'مرافق',
    '2084.SR': 'مياهنا',
    '2090.SR': 'جبسكو',
    '2100.SR': 'وفرة',
    '2110.SR': 'الكابلات السعودية',
    '2120.SR': 'متطورة',
    '2130.SR': 'صدق',
    '2140.SR': 'أيان',
    '2150.SR': 'زجاج',
    '2160.SR': 'أميانتيت',
    '2170.SR': 'اللجين',
    '2180.SR': 'فيبكو',
    '2190.SR': 'سيسكو القابضة',
    '2200.SR': 'أنابيب',
    '2210.SR': 'نماء للكيماويات',
    '2220.SR': 'معدنية',
    '2222.SR': 'أرامكو السعودية',
    '2223.SR': 'لوبريف',
    '2230.SR': 'الكيميائية',
    '2240.SR': 'صناعات',
    '2250.SR': 'المجموعة السعودية',
    '2270.SR': 'سدافكو',
    '2280.SR': 'المراعي',
    '2281.SR': 'تنمية',
    '2282.SR': 'نقي',
    '2283.SR': 'المطاحن الأولى',
    '2284.SR': 'المطاحن الحديثة',
    '2285.SR': 'المطاحن العربية',
    '2286.SR': 'المطاحن الرابعة',
    '2287.SR': 'إنتاج',
    '2288.SR': 'نفوذ',
    '2290.SR': 'ينساب',
    '2300.SR': 'صناعة الورق',
    '2310.SR': 'سبكيم العالمية',
    '2320.SR': 'البابطين',
    '2330.SR': 'المتقدمة',
    '2340.SR': 'ارتيكس',
    '2350.SR': 'كيان السعودية',
    '2360.SR': 'الفخارية',
    '2370.SR': 'مسك',
    '2380.SR': 'بترو رابغ',
    '2381.SR': 'الحفر العربية',
    '2382.SR': 'أديس',
    '3002.SR': 'أسمنت نجران',
    '3003.SR': 'أسمنت المدينة',
    '3004.SR': 'أسمنت الشمالية',
    '3005.SR': 'أسمنت ام القرى',
    '3007.SR': 'الواحة',
    '3008.SR': 'الكثيري',
    '3010.SR': 'أسمنت العربية',
    '3020.SR': 'أسمنت اليمامة',
    '3030.SR': 'أسمنت السعودية',
    '3040.SR': 'أسمنت القصيم',
    '3050.SR': 'أسمنت الجنوب',
    '3060.SR': 'أسمنت ينبع',
    '3080.SR': 'أسمنت الشرقية',
    '3090.SR': 'أسمنت تبوك',
    '3091.SR': 'أسمنت الجوف',
    '3092.SR': 'أسمنت الرياض',
    '4001.SR': 'أسواق ع العثيم',
    '4002.SR': 'المواساة',
    '4003.SR': 'إكسترا',
    '4004.SR': 'دله الصحية',
    '4005.SR': 'رعاية',
    '4006.SR': 'أسواق المزرعة',
    '4007.SR': 'الحمادي',
    '4008.SR': 'ساكو',
    '4009.SR': 'السعودي الألماني',
    '4011.SR': 'لازوردي',
    '4012.SR': 'الأصيل',
    '4013.SR': 'سليمان الحبيب',
    '4014.SR': 'دار المعدات',
    '4015.SR': 'جمجوم فارما',
    '4016.SR': 'أفالون فارما',
    '4017.SR': 'فقيه الطبية',
    '4018.SR': 'الموسى',
    '4019.SR': 'اس ام سي',
    '4020.SR': 'العقارية',
    '4021.SR': 'المركز الكندي الطبي',
    '4030.SR': 'البحري',
    '4031.SR': 'الخدمات الأرضية',
    '4040.SR': 'سابتكو',
    '4050.SR': 'ساسكو',
    '4051.SR': 'باعظيم',
    '4061.SR': 'أنعام القابضة',
    '4070.SR': 'تهامة',
    '4071.SR': 'العربية',
    '4072.SR': 'إم بي سي',
    '4080.SR': 'سناد القابضة',
    '4081.SR': 'النايفات',
    '4082.SR': 'مرنة',
    '4083.SR': 'تسهيل',
    '4084.SR': 'دراية',
    '4090.SR': 'طيبة',
    '4100.SR': 'مكة',
    '4110.SR': 'باتك',
    '4130.SR': 'درب السعودية',
    '4140.SR': 'صادرات',
    '4141.SR': 'العمران',
    '4142.SR': 'كابلات الرياض',
    '4143.SR': 'تالكو',
    '4144.SR': 'رؤوم',
    '4145.SR': 'أو جي سي',
    '4146.SR': 'جاز',
    '4147.SR': 'سي جي إس',
    '4148.SR': 'الوسائل الصناعية',
    '4150.SR': 'التعمير',
    '4160.SR': 'ثمار',
    '4161.SR': 'بن داود',
    '4162.SR': 'المنجم',
    '4163.SR': 'الدواء',
    '4164.SR': 'النهدي',
    '4165.SR': 'الماجد للعود',
    '4170.SR': 'شمس',
    '4180.SR': 'مجموعة فتيحي',
    '4190.SR': 'جرير',
    '4191.SR': 'أبو معطي',
    '4192.SR': 'السيف غاليري',
    '4193.SR': 'نايس ون',
    '4194.SR': 'محطة البناء',
    '4200.SR': 'الدريس',
    '4210.SR': 'الأبحاث والإعلام',
    '4220.SR': 'إعمار',
    '4230.SR': 'البحر الأحمر',
    '4240.SR': 'سينومي ريتيل',
    '4250.SR': 'جبل عمر',
    '4260.SR': 'بدجت السعودية',
    '4261.SR': 'ذيب',
    '4262.SR': 'لومي',
    '4263.SR': 'سال',
    '4264.SR': 'طيران ناس',
    '4265.SR': 'شري',
    '4270.SR': 'طباعة وتغليف',
    '4280.SR': 'المملكة',
    '4290.SR': 'الخليج للتدريب',
    '4291.SR': 'الوطنية للتعليم',
    '4292.SR': 'عطاء',
    '4300.SR': 'دار الأركان',
    '4310.SR': 'مدينة المعرفة',
    '4320.SR': 'الأندلس',
    '4321.SR': 'سينومي سنترز',
    '4322.SR': 'رتال',
    '4323.SR': 'سمو',
    '4324.SR': 'بنان',
    '4325.SR': 'مسار',
    '4326.SR': 'الماجدية',
    '4327.SR': 'الرمز',
    '4330.SR': 'الرياض ريت',
    '4331.SR': 'الجزيرة ريت',
    '4332.SR': 'جدوى ريت الحرمين',
    '4333.SR': 'تعليم ريت',
    '4334.SR': 'المعذر ريت',
    '4335.SR': 'مشاركة ريت',
    '4336.SR': 'ملكية ريت',
    '4337.SR': 'العزيزية ريت',
    '4338.SR': 'الأهلي ريت 1',
    '4339.SR': 'دراية ريت',
    '4340.SR': 'الراجحي ريت',
    '4342.SR': 'جدوى ريت السعودية',
    '4344.SR': 'سدكو كابيتال ريت',
    '4345.SR': 'الإنماء ريت للتجزئة',
    '4346.SR': 'ميفك ريت',
    '4347.SR': 'بنيان ريت',
    '4348.SR': 'الخبير ريت',
    '4349.SR': 'الإنماء ريت الفندقي',
    '4350.SR': 'الإستثمار ريت',
    '5110.SR': 'كهرباء السعودية',
    '6001.SR': 'حلواني إخوان',
    '6002.SR': 'هرفي للأغذية',
    '6004.SR': 'كاتريون',
    '6010.SR': 'نادك',
    '6012.SR': 'ريدان',
    '6013.SR': 'التطويرية الغذائية',
    '6014.SR': 'الآمار',
    '6015.SR': 'أمريكانا',
    '6016.SR': 'برغرايززر',
    '6017.SR': 'جاهز',
    '6018.SR': 'الأندية للرياضة',
    '6019.SR': 'المسار الشامل',
    '6020.SR': 'جاكو',
    '6040.SR': 'تبوك الزراعية',
    '6050.SR': 'الأسماك',
    '6060.SR': 'الشرقية للتنمية',
    '6070.SR': 'الجوف',
    '6090.SR': 'جازادكو',
    '7010.SR': 'اس تي سي',
    '7020.SR': 'إتحاد إتصالات',
    '7030.SR': 'زين السعودية',
    '7040.SR': 'قو للإتصالات',
    '7200.SR': 'ام آي اس',
    '7201.SR': 'بحر العرب',
    '7202.SR': 'سلوشنز',
    '7203.SR': 'علم',
    '7204.SR': 'توبي',
    '7211.SR': 'عزم',
    '8010.SR': 'التعاونية',
    '8012.SR': 'جزيرة تكافل',
    '8020.SR': 'ملاذ للتأمين',
    '8030.SR': 'ميدغلف للتأمين',
    '8040.SR': 'متكاملة',
    '8050.SR': 'سلامة',
    '8060.SR': 'ولاء',
    '8070.SR': 'الدرع العربي',
    '8100.SR': 'سايكو',
    '8120.SR': 'إتحاد الخليج الأهلية',
    '8150.SR': 'أسيج',
    '8160.SR': 'التأمين العربية',
    '8170.SR': 'الاتحاد',
    '8180.SR': 'الصقر للتأمين',
    '8190.SR': 'المتحدة للتأمين',
    '8200.SR': 'الإعادة السعودية',
    '8210.SR': 'بوبا العربية',
    '8230.SR': 'تكافل الراجحي',
    '8240.SR': 'تْشب',
    '8250.SR': 'جي آي جي',
    '8260.SR': 'الخليجية العامة',
    '8270.SR': 'ليفا',
    '8280.SR': 'ليفا',
    '8300.SR': 'الوطنية',
    '8310.SR': 'أمانة للتأمين',
    '8311.SR': 'عناية',
    '8313.SR': 'رسن',
}

TICKER_ALIASES = {
    "TASI":     "^TASI.SR",
    "^TASI":    "^TASI.SR",
    "^TASI.SR": "^TASI.SR",
    "تاسي":     "^TASI.SR",
    "تاسى":     "^TASI.SR",
}

# ───────────────────── TICKER LOOKUP HELPERS ─────────────────

def _normalize_arabic(text: str) -> str:
    """Normalize Arabic text by unifying letter forms."""
    text = text.strip()
    text = text.replace("ى", "ي").replace("أ", "ا")
    text = text.replace("إ", "ا").replace("آ", "ا")
    return text


def _normalize_key(text: str) -> str:
    return _normalize_arabic(text.upper())


def get_name(ticker: str) -> str:
    """Get Arabic company name from ticker."""
    return COMPANY_NAMES.get(ticker, ticker)


# Build normalized alias map once at startup
_ALIAS_MAP = {
    _normalize_key(alias): canonical
    for alias, canonical in TICKER_ALIASES.items()
}


def find_ticker(query: str) -> str | None:
    """
    Resolve user input to canonical ticker.
    Tries: aliases → exact ticker → code without .SR → Arabic name → partial match.
    """
    query = query.strip()
    query_key = _normalize_key(query)
    query_normalized = _normalize_arabic(query)

    # 0) Alias lookup (fast path)
    if query_key in _ALIAS_MAP:
        return _ALIAS_MAP[query_key]

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

    # 5) Partial Arabic match (last resort)
    if len(query_normalized) >= 2:
        for ticker, name in COMPANY_NAMES.items():
            if query_normalized in _normalize_arabic(name):
                return ticker

    return None


# ─────────────────────────────────────────────────────────────
# SECTOR MAP
# ─────────────────────────────────────────────────────────────
# Maps ticker → (sector_name, industry_name)
# Used when yfinance doesn't return sector/industry for Saudi stocks.
#
# HOW TO MODIFY:
# - Add new entries: 'XXXX.SR': ('القطاع', 'الصناعة')
# ─────────────────────────────────────────────────────────────

SECTOR_MAP = {
    '1010.SR': ('المالية', 'البنوك'),
    '1020.SR': ('المالية', 'البنوك'),
    '1030.SR': ('المالية', 'البنوك'),
    '1050.SR': ('المالية', 'البنوك'),
    '1060.SR': ('المالية', 'البنوك'),
    '1080.SR': ('المالية', 'البنوك'),
    '1120.SR': ('المالية', 'البنوك'),
    '1140.SR': ('المالية', 'البنوك'),
    '1150.SR': ('المالية', 'البنوك'),
    '1180.SR': ('المالية', 'البنوك'),
    '1182.SR': ('المالية', 'البنوك'),
    '1183.SR': ('المالية', 'البنوك'),
    '8010.SR': ('المالية', 'التأمين'),
    '8012.SR': ('المالية', 'التأمين'),
    '8020.SR': ('المالية', 'التأمين'),
    '8030.SR': ('المالية', 'التأمين'),
    '8040.SR': ('المالية', 'التأمين'),
    '8050.SR': ('المالية', 'التأمين'),
    '8060.SR': ('المالية', 'التأمين'),
    '8070.SR': ('المالية', 'التأمين'),
    '8100.SR': ('المالية', 'التأمين'),
    '8120.SR': ('المالية', 'التأمين'),
    '8150.SR': ('المالية', 'التأمين'),
    '8160.SR': ('المالية', 'التأمين'),
    '8170.SR': ('المالية', 'التأمين'),
    '8180.SR': ('المالية', 'التأمين'),
    '8190.SR': ('المالية', 'التأمين'),
    '8200.SR': ('المالية', 'التأمين'),
    '8210.SR': ('المالية', 'التأمين'),
    '8230.SR': ('المالية', 'التأمين'),
    '8240.SR': ('المالية', 'التأمين'),
    '8250.SR': ('المالية', 'التأمين'),
    '8260.SR': ('المالية', 'التأمين'),
    '8270.SR': ('المالية', 'التأمين'),
    '8280.SR': ('المالية', 'التأمين'),
    '8300.SR': ('المالية', 'التأمين'),
    '8310.SR': ('المالية', 'التأمين'),
    '8311.SR': ('المالية', 'التأمين'),
    '8313.SR': ('المالية', 'التأمين'),
    '2001.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2010.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2020.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2060.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2080.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2090.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2110.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2120.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2130.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2140.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2150.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2160.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2170.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2180.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2190.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2200.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2210.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2220.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2230.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2240.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2250.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2270.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2290.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2300.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2310.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2320.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2330.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2340.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2350.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2360.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2370.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2380.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '2222.SR': ('الطاقة', 'النفط والغاز'),
    '2030.SR': ('الطاقة', 'النفط والغاز'),
    '2381.SR': ('الطاقة', 'النفط والغاز'),
    '2382.SR': ('الطاقة', 'النفط والغاز'),
    '3002.SR': ('المواد الأساسية', 'الإسمنت'),
    '3003.SR': ('المواد الأساسية', 'الإسمنت'),
    '3004.SR': ('المواد الأساسية', 'الإسمنت'),
    '3005.SR': ('المواد الأساسية', 'الإسمنت'),
    '3007.SR': ('المواد الأساسية', 'الإسمنت'),
    '3008.SR': ('المواد الأساسية', 'الإسمنت'),
    '3010.SR': ('المواد الأساسية', 'الإسمنت'),
    '3020.SR': ('المواد الأساسية', 'الإسمنت'),
    '3030.SR': ('المواد الأساسية', 'الإسمنت'),
    '3040.SR': ('المواد الأساسية', 'الإسمنت'),
    '3050.SR': ('المواد الأساسية', 'الإسمنت'),
    '3060.SR': ('المواد الأساسية', 'الإسمنت'),
    '3080.SR': ('المواد الأساسية', 'الإسمنت'),
    '3090.SR': ('المواد الأساسية', 'الإسمنت'),
    '3091.SR': ('المواد الأساسية', 'الإسمنت'),
    '3092.SR': ('المواد الأساسية', 'الإسمنت'),
    '4001.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4003.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4006.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4011.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4190.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4192.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4193.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4240.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4002.SR': ('الرعاية الصحية', 'الخدمات الصحية'),
    '4004.SR': ('الرعاية الصحية', 'الخدمات الصحية'),
    '4005.SR': ('الرعاية الصحية', 'الخدمات الصحية'),
    '4007.SR': ('الرعاية الصحية', 'الخدمات الصحية'),
    '4009.SR': ('الرعاية الصحية', 'الخدمات الصحية'),
    '4012.SR': ('الرعاية الصحية', 'الخدمات الصحية'),
    '4013.SR': ('الرعاية الصحية', 'الخدمات الصحية'),
    '4014.SR': ('الرعاية الصحية', 'الخدمات الصحية'),
    '4015.SR': ('الرعاية الصحية', 'الصيدلانيات'),
    '4016.SR': ('الرعاية الصحية', 'الصيدلانيات'),
    '4017.SR': ('الرعاية الصحية', 'الخدمات الصحية'),
    '4018.SR': ('الرعاية الصحية', 'الخدمات الصحية'),
    '4019.SR': ('الرعاية الصحية', 'الخدمات الصحية'),
    '4021.SR': ('الرعاية الصحية', 'الخدمات الصحية'),
    '7010.SR': ('الاتصالات', 'خدمات الاتصالات'),
    '7020.SR': ('الاتصالات', 'خدمات الاتصالات'),
    '7030.SR': ('الاتصالات', 'خدمات الاتصالات'),
    '7040.SR': ('الاتصالات', 'خدمات الاتصالات'),
    '7200.SR': ('الاتصالات', 'تقنية المعلومات'),
    '7201.SR': ('الاتصالات', 'تقنية المعلومات'),
    '7202.SR': ('الاتصالات', 'تقنية المعلومات'),
    '7203.SR': ('الاتصالات', 'تقنية المعلومات'),
    '7204.SR': ('الاتصالات', 'تقنية المعلومات'),
    '7211.SR': ('الاتصالات', 'خدمات الاتصالات'),
    '2040.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '2050.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '2280.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '2281.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '2282.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '2283.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '2284.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '2285.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '2286.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '2287.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '2288.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '6001.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '6002.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '6004.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '6010.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '6012.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '6013.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '6014.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '6015.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '6016.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '6017.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '6040.SR': ('السلع الاستهلاكية', 'الزراعة والأغذية'),
    '6050.SR': ('السلع الاستهلاكية', 'الزراعة والأغذية'),
    '6060.SR': ('السلع الاستهلاكية', 'الزراعة والأغذية'),
    '6070.SR': ('السلع الاستهلاكية', 'الزراعة والأغذية'),
    '6090.SR': ('السلع الاستهلاكية', 'الزراعة والأغذية'),
    '4020.SR': ('العقارات', 'التطوير العقاري'),
    '4220.SR': ('العقارات', 'التطوير العقاري'),
    '4230.SR': ('العقارات', 'التطوير العقاري'),
    '4250.SR': ('العقارات', 'التطوير العقاري'),
    '4300.SR': ('العقارات', 'التطوير العقاري'),
    '4310.SR': ('العقارات', 'التطوير العقاري'),
    '4320.SR': ('العقارات', 'التطوير العقاري'),
    '4321.SR': ('العقارات', 'مراكز التسوق'),
    '4322.SR': ('العقارات', 'التطوير العقاري'),
    '4323.SR': ('العقارات', 'التطوير العقاري'),
    '4324.SR': ('العقارات', 'التطوير العقاري'),
    '4325.SR': ('العقارات', 'التطوير العقاري'),
    '4326.SR': ('العقارات', 'التطوير العقاري'),
    '4327.SR': ('العقارات', 'التطوير العقاري'),
    '4330.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '4331.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '4332.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '4333.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '4334.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '4335.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '4336.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '4337.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '4338.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '4339.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '4340.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '4342.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '4344.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '4345.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '4346.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '4347.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '4348.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '4349.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '4350.SR': ('العقارات', 'صناديق الاستثمار العقاري'),
    '5110.SR': ('المرافق العامة', 'الكهرباء والمياه'),
    '2082.SR': ('المرافق العامة', 'الكهرباء والمياه'),
    '2083.SR': ('المرافق العامة', 'الكهرباء والمياه'),
    '2084.SR': ('المرافق العامة', 'الكهرباء والمياه'),
    '1201.SR': ('الصناعة', 'الصناعات التحويلية'),
    '1202.SR': ('الصناعة', 'الصناعات التحويلية'),
    '1210.SR': ('الصناعة', 'التعدين والمعادن'),
    '1211.SR': ('الصناعة', 'التعدين والمعادن'),
    '1212.SR': ('الصناعة', 'الصناعات التحويلية'),
    '1213.SR': ('الصناعة', 'الصناعات التحويلية'),
    '1214.SR': ('الصناعة', 'الصناعات التحويلية'),
    '1301.SR': ('الصناعة', 'الصناعات التحويلية'),
    '1302.SR': ('الصناعة', 'الصناعات التحويلية'),
    '1303.SR': ('الصناعة', 'الصناعات التحويلية'),
    '1304.SR': ('الصناعة', 'الصناعات التحويلية'),
    '1320.SR': ('الصناعة', 'الصناعات التحويلية'),
    '1321.SR': ('الصناعة', 'الصناعات التحويلية'),
    '1322.SR': ('الصناعة', 'الصناعات التحويلية'),
    '1323.SR': ('الصناعة', 'الصناعات التحويلية'),
    '4030.SR': ('الصناعة', 'النقل البحري'),
    '4031.SR': ('الصناعة', 'الخدمات الأرضية'),
    '4040.SR': ('الصناعة', 'النقل البري'),
    '4050.SR': ('الصناعة', 'محطات الوقود'),
    '4080.SR': ('الصناعة', 'الخدمات اللوجستية'),
    '4260.SR': ('الصناعة', 'تأجير السيارات'),
    '4261.SR': ('الصناعة', 'تأجير السيارات'),
    '4263.SR': ('الصناعة', 'الخدمات اللوجستية'),
    '4264.SR': ('الصناعة', 'النقل الجوي'),
    '4070.SR': ('الخدمات', 'الإعلام والترفيه'),
    '4071.SR': ('الخدمات', 'الإعلام والترفيه'),
    '4072.SR': ('الخدمات', 'الإعلام والترفيه'),
    '4290.SR': ('الخدمات', 'التعليم'),
    '4291.SR': ('الخدمات', 'التعليم'),
    '4292.SR': ('الخدمات', 'التعليم'),
    '1111.SR': ('المالية', 'أسواق المال'),
    '6018.SR': ('الخدمات', 'الترفيه'),
    '6019.SR': ('الخدمات', 'التغذية'),
    '1810.SR': ('الخدمات', 'الفندقة والسياحة'),
    '1820.SR': ('الخدمات', 'الفندقة والسياحة'),
    '1830.SR': ('الخدمات', 'الترفيه'),
    '1831.SR': ('الخدمات', 'خدمات الأعمال'),
    '1832.SR': ('الخدمات', 'خدمات الأعمال'),
    '1833.SR': ('الخدمات', 'خدمات الأعمال'),
    '1834.SR': ('الخدمات', 'خدمات الأعمال'),
    '1835.SR': ('الخدمات', 'خدمات الأعمال'),
    '2081.SR': ('الصناعة', 'المعدات الصناعية'),
    '4051.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4061.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4081.SR': ('المالية', 'التمويل'),
    '4082.SR': ('المالية', 'التمويل'),
    '4083.SR': ('المالية', 'التمويل'),
    '4084.SR': ('المالية', 'أسواق المال'),
    '4090.SR': ('العقارات', 'التطوير العقاري'),
    '4100.SR': ('العقارات', 'التطوير العقاري'),
    '4110.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4130.SR': ('الخدمات', 'خدمات الأعمال'),
    '4140.SR': ('الصناعة', 'التصدير'),
    '4141.SR': ('الصناعة', 'الصناعات التحويلية'),
    '4142.SR': ('الصناعة', 'الصناعات التحويلية'),
    '4143.SR': ('الصناعة', 'الصناعات التحويلية'),
    '4144.SR': ('الخدمات', 'خدمات الأعمال'),
    '4145.SR': ('الطاقة', 'النفط والغاز'),
    '4146.SR': ('الصناعة', 'الخدمات اللوجستية'),
    '4147.SR': ('الصناعة', 'الخدمات اللوجستية'),
    '4148.SR': ('الصناعة', 'الصناعات التحويلية'),
    '4150.SR': ('العقارات', 'التطوير العقاري'),
    '4160.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4161.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4162.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4163.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4164.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4165.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4170.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4180.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4191.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4194.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4200.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4210.SR': ('الخدمات', 'الإعلام والترفيه'),
    '4262.SR': ('الصناعة', 'تأجير السيارات'),
    '4265.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
    '4270.SR': ('الصناعة', 'الصناعات التحويلية'),
    '4280.SR': ('العقارات', 'التطوير العقاري'),
    '6020.SR': ('السلع الاستهلاكية', 'الأغذية والمشروبات'),
    '2070.SR': ('الرعاية الصحية', 'الصيدلانيات'),
    '2100.SR': ('المواد الأساسية', 'البتروكيماويات'),
    '7040.SR': ('الاتصالات', 'خدمات الاتصالات'),
    '4008.SR': ('السلع الاستهلاكية', 'تجارة التجزئة'),
}


def get_sector_industry(ticker):
    """Get (sector, industry) tuple for a ticker."""
    return SECTOR_MAP.get(ticker, (None, None))



# ═══════════════════════════════════════════════════════════════
# PART 3: STOCKS_STATIC_DATA & ENRICHMENT FUNCTIONS
# ═══════════════════════════════════════════════════════════════
#
# EXPLANATION:
# - STOCKS_STATIC_DATA: hardcoded fundamentals from Tadawul/STOCKS
# - Priority: STOCKS data FIRST → Yahoo Finance fallback
# - _enrich_with_STOCKS(): merges static data into yfinance info dict
# - All display values stored in info dict (not local variables)
#
# KEY BUG FIX:
# - Original code set `roaval` and `roa` as local vars in
#   _enrich_with_STOCKS() but used them in cover() → NameError
# - Now ROA/ROE display values are stored as info['roa_display']
#   and info['roe_display'] in the info dict
#
# HOW TO MODIFY:
# - To update fundamentals: edit STOCKS_STATIC_DATA entries
# - To add new field: add key to dict, handle in _enrich_with_STOCKS()
# ═══════════════════════════════════════════════════════════════

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
    "2222": {"Numberofshare": "242,000.00", "Eps": "1.44", "Bookvalue": "6.17", "Parallel_value": "-", "PE_ratio": "16.96", "PB_ratio": "4.32", "ROA": "13.99", "ROE": "23.59"},
    # ... (all remaining entries from the original STOCKS_STATIC_DATA - kept identical)
    # NOTE: The full dict is the same as your original. Due to space I show
    # representative entries. In your actual code, keep ALL entries.
}

# ──── IMPORTANT: Copy ALL remaining entries from your original
# ──── STOCKS_STATIC_DATA here. They are unchanged.
# ──── I'm showing the key entries above; the rest are identical to your code.

# For brevity, here's a programmatic way to include them all:
# (In your actual file, paste the full STOCKS_STATIC_DATA from your original code)


def _safe_float(value: str) -> float | None:
    """Parse string to float. Handles parenthesized negatives like '(0.23 )' → -0.23."""
    if value is None:
        return None
    value = str(value).strip()
    if not value or value in _PE_NON_NUMERIC or value == "-":
        return None
    if value.startswith("(") and value.endswith(")"):
        inner = value[1:-1].strip()
        try:
            return -abs(float(inner))
        except (ValueError, TypeError):
            return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_display(value: str) -> str | None:
    """Return cleaned display string, or None if empty/dash."""
    if value is None:
        return None
    value = str(value).strip()
    if not value or value == "-":
        return None
    return value


def _get_current_price(info: dict) -> float | None:
    """Extract current price from info dict."""
    for key in ("currentPrice", "regularMarketPrice", "previousClose"):
        val = info.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return None


def _format_roa_roe(value_str: str) -> tuple[float | None, str]:
    """
    Parse ROA/ROE from STOCKS format.
    Returns (numeric_value, display_string).
    """
    if value_str is None:
        return None, ""
    value_str = str(value_str).strip()
    if not value_str or value_str == "-":
        return None, ""
    numeric = _safe_float(value_str)
    if numeric is not None:
        if numeric < 0:
            display = f"({abs(numeric)}%)"
        else:
            display = f"{numeric}%"
        return numeric, display
    return None, ""


def _enrich_with_STOCKS(ticker: str, info: dict) -> None:
    """
    Enrich info dict with STOCKS static data.
    Priority: STOCKS data FIRST → fall back to yfinance.

    CRITICAL: Stores display values in info dict so cover() can access them:
      - info['roa_display']  → string for ROA display
      - info['roe_display']  → string for ROE display
      - info['bv_display']   → string for Book Value display
      - info['trailingPE_display'] → string for P/E display
      - info['trailingEpsFormatted'] → string for EPS display
    """
    code = ticker.replace(".SR", "").strip()
    stocks = STOCKS_STATIC_DATA.get(code)

    if not stocks:
        logger.info(f"STOCKS static: no data for {code}, using yfinance only")
        # Ensure display fields exist even without STOCKS
        _ensure_display_fields_from_yahoo(info)
        return

    logger.info(f"STOCKS static: found data for {code}")

    # ── 1. EPS (ربح السهم) ──
    eps = _safe_float(stocks.get("Eps"))
    if eps is not None:
        info["trailingEps"] = eps
        info["trailingEpsFormatted"] = f"({abs(eps)})" if eps < 0 else str(eps)
    elif info.get("trailingEps") is not None:
        try:
            yf_eps = float(info["trailingEps"])
            info["trailingEpsFormatted"] = f"({abs(yf_eps)})" if yf_eps < 0 else str(yf_eps)
        except (ValueError, TypeError):
            info["trailingEpsFormatted"] = "-"
    else:
        info["trailingEpsFormatted"] = "-"

    # ── 2. Book Value (القيمة الدفترية) ──
    bv = _safe_float(stocks.get("Bookvalue"))
    if bv is not None:
        info["bookValue"] = bv
        info["bv_display"] = f"{bv:.2f}" if bv >= 0 else f"({abs(bv):.2f})"
    elif info.get("bookValue") is not None:
        try:
            info["bv_display"] = f"{float(info['bookValue']):.2f}"
        except (ValueError, TypeError):
            info["bv_display"] = "-"
    else:
        info["bv_display"] = "-"

    # ── 3. P/E Ratio (مكرر الربح) ──
    pe_raw = stocks.get("PE_ratio", "").strip()
    pe = _safe_float(pe_raw)
    if pe is not None and pe > 0:
        info["trailingPE"] = pe
        info["trailingPE_display"] = str(pe)
    elif pe_raw == "سالب":
        info["trailingPE"] = None
        info["trailingPE_display"] = "سالب"
    elif pe_raw == "أكبر من 100":
        info["trailingPE"] = None
        info["trailingPE_display"] = "أكبر من 100"
    else:
        yf_pe = info.get("trailingPE")
        info["trailingPE_display"] = str(round(float(yf_pe), 2)) if yf_pe else "-"

    # ── 4. P/B Ratio (مضاعف القيمة الدفترية) ──
    pb = _safe_float(stocks.get("PB_ratio"))
    if pb is not None:
        info["priceToBook"] = pb

    # ── 5. Shares Outstanding (عدد الأسهم) ──
    shares_str = stocks.get("Numberofshare", "")
    shares_clean = shares_str.replace(",", "").strip() if shares_str else ""
    shares = _safe_float(shares_clean)
    if shares is not None:
        shares_actual = shares * 1_000_000
        info["sharesOutstanding"] = shares_actual
        info["floatShares"] = shares_actual
        price = _get_current_price(info)
        if price and price > 0:
            info["marketCap"] = price * shares_actual

    # ── 6. Par Value (القيمة الاسمية) ──
    par = _safe_float(stocks.get("Parallel_value"))
    if par is not None:
        info["parValue"] = par

    # ── 7. ROA (العائد على الأصول) ──
    roa_numeric, roa_display = _format_roa_roe(stocks.get("ROA"))
    if roa_numeric is not None:
        info["returnOnAssets"] = roa_numeric / 100.0
        info["roa_display"] = roa_display
    else:
        # Fallback to yfinance
        yf_roa = info.get("returnOnAssets")
        if yf_roa is not None:
            try:
                roa_pct = float(yf_roa) * 100
                info["roa_display"] = f"({abs(round(roa_pct, 2))}%)" if roa_pct < 0 else f"{round(roa_pct, 2)}%"
            except (ValueError, TypeError):
                info["roa_display"] = "-"
        else:
            info["roa_display"] = "-"

    # ── 8. ROE (العائد على حقوق المساهمين) ──
    roe_numeric, roe_display = _format_roa_roe(stocks.get("ROE"))
    if roe_numeric is not None:
        info["returnOnEquity"] = roe_numeric / 100.0
        info["roe_display"] = roe_display
    else:
        yf_roe = info.get("returnOnEquity")
        if yf_roe is not None:
            try:
                roe_pct = float(yf_roe) * 100
                info["roe_display"] = f"({abs(round(roe_pct, 2))}%)" if roe_pct < 0 else f"{round(roe_pct, 2)}%"
            except (ValueError, TypeError):
                info["roe_display"] = "-"
        else:
            info["roe_display"] = "-"

    # ── 9. Recalculate P/E if still missing ──
    if not info.get("trailingPE") and info.get("trailingPE_display") in (None, "-", ""):
        eps_val = info.get("trailingEps")
        price = _get_current_price(info)
        if eps_val and float(eps_val) != 0 and price and price > 0:
            try:
                calc_pe = round(price / float(eps_val), 2)
                info["trailingPE"] = calc_pe
                info["trailingPE_display"] = str(calc_pe)
            except Exception:
                pass

    # ── 10. Recalculate P/B if missing ──
    if not info.get("priceToBook"):
        bv_val = info.get("bookValue")
        price = _get_current_price(info)
        if bv_val and float(bv_val) != 0 and price and price > 0:
            try:
                info["priceToBook"] = round(price / float(bv_val), 4)
            except Exception:
                pass

    # ── 11. Defaults ──
    if not info.get("currency"):
        info["currency"] = "SAR"
    if not info.get("exchange"):
        info["exchange"] = "Tadawul"


def _ensure_display_fields_from_yahoo(info: dict) -> None:
    """Ensure all display fields exist using yfinance data when STOCKS unavailable."""
    # EPS
    if not info.get("trailingEpsFormatted"):
        yf_eps = info.get("trailingEps")
        if yf_eps is not None:
            try:
                val = float(yf_eps)
                info["trailingEpsFormatted"] = f"({abs(val)})" if val < 0 else str(val)
            except (ValueError, TypeError):
                info["trailingEpsFormatted"] = "-"
        else:
            info["trailingEpsFormatted"] = "-"

    # Book Value display
    if not info.get("bv_display"):
        bv = info.get("bookValue")
        if bv is not None:
            try:
                info["bv_display"] = f"{float(bv):.2f}"
            except (ValueError, TypeError):
                info["bv_display"] = "-"
        else:
            info["bv_display"] = "-"

    # P/E display
    if not info.get("trailingPE_display"):
        pe = info.get("trailingPE")
        info["trailingPE_display"] = str(round(float(pe), 2)) if pe else "-"

    # ROA display
    if not info.get("roa_display"):
        yf_roa = info.get("returnOnAssets")
        if yf_roa is not None:
            try:
                roa_pct = float(yf_roa) * 100
                info["roa_display"] = f"({abs(round(roa_pct, 2))}%)" if roa_pct < 0 else f"{round(roa_pct, 2)}%"
            except (ValueError, TypeError):
                info["roa_display"] = "-"
        else:
            info["roa_display"] = "-"

    # ROE display
    if not info.get("roe_display"):
        yf_roe = info.get("returnOnEquity")
        if yf_roe is not None:
            try:
                roe_pct = float(yf_roe) * 100
                info["roe_display"] = f"({abs(round(roe_pct, 2))}%)" if roe_pct < 0 else f"{round(roe_pct, 2)}%"
            except (ValueError, TypeError):
                info["roe_display"] = "-"
        else:
            info["roe_display"] = "-"



# ═══════════════════════════════════════════════════════════════
# PART 4: PDF COVER PAGE — FIXED LAYOUT
# ═══════════════════════════════════════════════════════════════
#
# CHANGES FROM ORIGINAL:
# 1. REMOVED: قيمة التداول (tradingValue), عدد الصفقات (tradesCount), بيتا (beta)
# 2. ADDED:   العائد على الأصول (ROA), القيمة الدفترية (bookValue)
# 3. FIXED:   roaval/roa NameError — now reads from info['roa_display']
# 4. Layout:  3 rows × 3 columns of metric boxes
#
# ROW 1: السعر الحالي | التغير اليومي | القيمة السوقية
# ROW 2: مكرر الربحية | ربحية السهم | عائد التوزيعات
# ROW 3: مضاعف القيمة الدفترية | العائد على حقوق المساهمين | العائد على الأصول
# ROW 4: حجم التداول | القيمة الدفترية | (empty or another metric)
# ═══════════════════════════════════════════════════════════════

# (This replaces the cover() method inside the Report class)
# Below is the corrected cover method. In your Report class, replace the
# existing cover() with this:

    def cover(self, price, chg, info, rec_txt, rec_color, score):
        c = self.c
        self.pn += 1

        # ── Header bar ──
        c.setFillColor(NAVY)
        c.rect(0, PAGE_H - 78 * mm, PAGE_W, 78 * mm, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.rect(0, PAGE_H - 80 * mm, PAGE_W, 2 * mm, fill=1, stroke=0)

        company_name = (COMPANY_NAMES.get(self.tk)
                        or safe(info, 'longName', safe(info, 'shortName', self.display_tk))
                        or self.display_tk)
        company_name = short_text(str(company_name), 42)

        c.setFillColor(WHITE)
        self._font(True, 23)
        c.drawRightString(PAGE_W - MG, PAGE_H - 22 * mm, rtl('تقرير تحليل سهم'))
        self._font(True, 18)
        c.drawRightString(PAGE_W - MG, PAGE_H - 39 * mm, tx(company_name))
        self._font(False, 11)
        c.drawRightString(PAGE_W - MG, PAGE_H - 52 * mm,
                          tx(f'{self.display_tk} | {safe(info, "exchange", "-")}'))
        self._font(False, 9)
        c.drawString(MG, PAGE_H - 18 * mm, datetime.now().strftime('%Y-%m-%d'))

        # ── Recommendation badge ──
        c.setFillColor(HexColor(rec_color))
        c.roundRect(MG, PAGE_H - 58 * mm, 60 * mm, 14 * mm, 8, fill=1, stroke=0)
        c.setFillColor(WHITE)
        self._font(True, 11)
        c.drawCentredString(MG + 30 * mm, PAGE_H - 50 * mm, rtl(rec_txt))
        self._font(False, 8)
        c.drawCentredString(MG + 30 * mm, PAGE_H - 54 * mm, rtl(f'النتيجة {score}/20'))

        # ── Metric boxes layout: 4 rows × 3 columns ──
        col_bw = CW / 3 - 4 * mm
        col_bh = 18 * mm
        col_gap = 4 * mm
        x1 = MG
        x2 = MG + col_bw + col_gap
        x3 = MG + 2 * (col_bw + col_gap)
        y1 = PAGE_H - 110 * mm
        y2 = y1 - col_bh - col_gap
        y3 = y2 - col_bh - col_gap
        y4 = y3 - col_bh - col_gap

        # ── ROW 1: Price, Change, Market Cap ──
        self._box(x1, y1, col_bw, col_bh, 'السعر الحالي', price)
        self._box(x2, y1, col_bw, col_bh, 'التغير اليومي',
                  f'{chg:+.2f}%', clr=GREEN_HEX if chg >= 0 else RED_HEX)
        self._box(x3, y1, col_bw, col_bh, 'القيمة السوقية',
                  fmt_n(safe(info, 'marketCap'))[0])

        # ── ROW 2: P/E, EPS, Dividend Yield ──
        pe_display = info.get('trailingPE_display', '-')
        self._box(x1, y2, col_bw, col_bh, 'مكرر الربحية', pe_display)

        eps_formatted = info.get('trailingEpsFormatted', '-')
        eps_val = _safe_float(eps_formatted) if eps_formatted != '-' else None
        eps_color = (GREEN_HEX if (eps_val is not None and eps_val >= 0)
                     else RED_HEX if eps_val is not None else None)
        self._box(x2, y2, col_bw, col_bh, 'ربحية السهم', eps_formatted, clr=eps_color)

        dy = safe(info, 'dividendYield')
        self._box(x3, y2, col_bw, col_bh, 'عائد التوزيعات',
                  fmt_p(dy)[0] if dy else '-')

        # ── ROW 3: P/B, ROE, ROA ──   (CHANGED: was P/B, ROE, Beta)
        pb = safe(info, 'priceToBook')
        self._box(x1, y3, col_bw, col_bh, 'مضاعف القيمة الدفترية',
                  f'{float(pb):.2f}' if pb else '-')

        roe_display = info.get('roe_display', '-')
        self._box(x2, y3, col_bw, col_bh, 'العائد على حقوق المساهمين',
                  roe_display)

        # ★ FIX: Use info['roa_display'] instead of broken local variable 'roaval'
        roa_display = info.get('roa_display', '-')
        self._box(x3, y3, col_bw, col_bh, 'العائد على الأصول', roa_display)

        # ── ROW 4: Volume, Book Value ──   (CHANGED: removed قيمة التداول, عدد الصفقات)
        self._box(x1, y4, col_bw, col_bh, 'حجم التداول',
                  fmt_n(safe(info, 'volume'), d=0)[0])

        # ★ NEW: القيمة الدفترية (Book Value per share)
        bv_display = info.get('bv_display', '-')
        self._box(x2, y4, col_bw, col_bh, 'القيمة الدفترية', bv_display)

        # Third box in row 4 — empty or could be used for another metric
        # If you want to add something later, use self._box(x3, y4, ...)
        # For now, leave it empty or add number of shares:
        shares_display = fmt_n(safe(info, 'sharesOutstanding'), d=0)[0]
        self._box(x3, y4, col_bw, col_bh, 'الأسهم القائمة', shares_display)

        # ── 52-week range bar ──
        w52h = safe(info, 'fiftyTwoWeekHigh')
        w52l = safe(info, 'fiftyTwoWeekLow')
        if w52h and w52l and float(w52h) != float(w52l):
            bar_y = y4 - 12 * mm
            c.setFillColor(NAVY)
            self._font(True, 8)
            c.drawRightString(PAGE_W - MG, bar_y + 2, rtl('نطاق 52 أسبوع'))
            bar_x = MG
            bar_w = CW
            bar_h = 5 * mm
            bar_y2 = bar_y - bar_h - 2
            c.setFillColor(HexColor('#ECEFF1'))
            c.roundRect(bar_x, bar_y2, bar_w, bar_h, 3, fill=1, stroke=0)
            pos = (price - float(w52l)) / (float(w52h) - float(w52l))
            pos = max(0.0, min(1.0, pos))
            fill_w = bar_w * pos
            fill_c = (HexColor(RED_HEX) if pos < 0.35
                      else HexColor(GREEN_HEX) if pos > 0.65
                      else HexColor(ORANGE_HEX))
            c.setFillColor(fill_c)
            c.roundRect(bar_x, bar_y2, max(fill_w, 4), bar_h, 3, fill=1, stroke=0)
            c.setFillColor(DGRAY)
            self._font(False, 7)
            c.drawString(bar_x, bar_y2 - 9, f'{float(w52l):.2f}')
            c.drawRightString(bar_x + bar_w, bar_y2 - 9, f'{float(w52h):.2f}')
            c.setFillColor(NAVY)
            self._font(True, 7)
            c.drawCentredString(bar_x + bar_w / 2, bar_y2 - 9, f'{pos * 100:.0f}%')
            y_after = bar_y2 - 12 * mm
        else:
            y_after = y4 - 6 * mm

        # ── Company info section ──
        y = y_after
        y = self._stitle(y, 'معلومات الشركة')
        sector = short_text(safe(info, 'sector', '-') or '-', 26)
        industry = short_text(safe(info, 'industry', '-') or '-', 26)
        items = [
            ('القطاع', sector),
            ('الصناعة', industry),
            ('العملة', safe(info, 'currency', 'SAR') or '-'),
            ('متوسط الحجم', fmt_n(safe(info, 'averageVolume'), d=0)[0]),
            ('أعلى 52 أسبوع', fmt_n(safe(info, 'fiftyTwoWeekHigh'))[0]),
            ('أدنى 52 أسبوع', fmt_n(safe(info, 'fiftyTwoWeekLow'))[0]),
            ('الأسهم الحرة', fmt_n(safe(info, 'floatShares'), d=0)[0]),
            ('القيمة الاسمية', f"{safe(info, 'parValue', '-')}"),
        ]
        half = CW / 2
        for idx, (lbl, val) in enumerate(items):
            col = idx % 2
            row = idx // 2
            xr = MG + (col + 1) * half - 5
            yy = y - row * 18
            c.setFillColor(NAVY)
            self._font(True, 8.5)
            c.drawRightString(xr, yy, rtl(f'{lbl}:'))
            c.setFillColor(TXTDARK)
            self._font(False, 8.5)
            c.drawRightString(xr - 85, yy, tx(val))

        c.setFillColor(DGRAY)
        self._font(False, 7)
        c.drawCentredString(PAGE_W / 2, 14 * mm,
                            rtl('هذا التقرير لأغراض معلوماتية فقط وليس توصية استثمارية.'))
        self._foot()
        c.showPage()



# ═══════════════════════════════════════════════════════════════
# PART 5: CHART FUNCTIONS WITH CORRECTED LEGENDS
# ═══════════════════════════════════════════════════════════════
#
# FIXES:
# 1. SMA chart: legends show "SMA 20", "SMA 50", etc.
# 2. EMA chart: legends show "EMA 20", "EMA 50", etc.
# 3. Alligator: legends show Arabic names with correct description
# 4. Supertrend: legends show Arabic صاعد/هابط
# 5. PDF page titles corrected with Arabic name + English abbreviation
# ═══════════════════════════════════════════════════════════════

def make_price_chart(d, sup=None, res=None):
    """SMA chart — المتوسطات المتحركة البسيطة (SMA)"""
    sup = sup or []; res = res or []
    d = d.tail(180).copy()
    p = d[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    aps = []; labels = []
    for col, clr, lbl in [
        ('SMA20',  BLUE_HEX,   'SMA 20'),
        ('SMA50',  RED_HEX,    'SMA 50'),
        ('SMA100', VIOLET_HEX, 'SMA 100'),
        ('SMA200', BLACK_HEX,  'SMA 200'),
    ]:
        if col in d and d[col].notna().sum() > 10:
            aps.append(mpf.make_addplot(d[col], color=clr, width=1))
            labels.append(lbl)
    mc = mpf.make_marketcolors(
        up='#26a69a', down='#ef5350', edge='inherit',
        wick='inherit', volume={'up': '#80cbc4', 'down': '#ef9a9a'})
    st = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=':', gridcolor='#dddddd',
        rc={'axes.facecolor': '#FAFAFA'})
    kw = dict(type='candle', style=st, volume=True, figsize=(14, 7),
              returnfig=True, warn_too_much_data=9999)
    if aps:
        kw['addplot'] = aps
    fig, ax = mpf.plot(p, **kw)
    if labels:
        ax[0].legend(labels, loc='upper left', fontsize=8, prop=MPL_FONT_PROP)
    fig.subplots_adjust(right=0.95, left=0.05)
    return chart_bytes(fig)


def make_ema_chart(d, sup=None, res=None):
    """EMA chart — المتوسطات المتحركة الأسية (EMA)"""
    sup = sup or []; res = res or []
    d = d.tail(180).copy()
    p = d[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    aps = []; labels = []
    for col, clr, lbl in [
        ('EMA20',  BLUE_HEX,   'EMA 20'),
        ('EMA50',  RED_HEX,    'EMA 50'),
        ('EMA100', VIOLET_HEX, 'EMA 100'),
        ('EMA200', BLACK_HEX,  'EMA 200'),
    ]:
        if col in d and d[col].notna().sum() > 10:
            aps.append(mpf.make_addplot(d[col], color=clr, width=1))
            labels.append(lbl)
    mc = mpf.make_marketcolors(
        up='#26a69a', down='#ef5350', edge='inherit',
        wick='inherit', volume={'up': '#80cbc4', 'down': '#ef9a9a'})
    st = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=':', gridcolor='#dddddd',
        rc={'axes.facecolor': '#FAFAFA'})
    kw = dict(type='candle', style=st, volume=True, figsize=(14, 7),
              returnfig=True, warn_too_much_data=9999)
    if aps:
        kw['addplot'] = aps
    fig, ax = mpf.plot(p, **kw)
    if labels:
        ax[0].legend(labels, loc='upper left', fontsize=8, prop=MPL_FONT_PROP)
    fig.subplots_adjust(right=0.95, left=0.05)
    return chart_bytes(fig)


def make_alligator_chart(d):
    """Alligator chart — مؤشر التمساح (Alligator)"""
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
    mc = mpf.make_marketcolors(
        up='#26a69a', down='#ef5350', edge='inherit',
        wick='inherit', volume={'up': '#80cbc4', 'down': '#ef9a9a'})
    st = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=':', gridcolor='#dddddd',
        rc={'axes.facecolor': '#FAFAFA'})
    kw = dict(type='candle', style=st, volume=False, figsize=(14, 7),
              returnfig=True, warn_too_much_data=9999)
    if aps:
        kw['addplot'] = aps
    fig, ax = mpf.plot(p, **kw)
    if labels:
        ax[0].legend(labels, loc='upper left', fontsize=8, prop=MPL_FONT_PROP)
    fig.subplots_adjust(right=0.95, left=0.05)
    return chart_bytes(fig)


def make_supertrend_chart(d):
    """Supertrend chart — مؤشر السوبر تريند (Supertrend)"""
    d = d.tail(180).copy()
    p = d[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    st_val, st_dir = _compute_supertrend(d)
    st_up   = st_val.where(st_dir == 1,  other=np.nan)
    st_down = st_val.where(st_dir == -1, other=np.nan)
    aps = []; labels = []
    if st_up.notna().any():
        aps.append(mpf.make_addplot(st_up, color='#4CAF50', width=1.5))
        labels.append(rtl('سوبر تريند صاعد'))
    if st_down.notna().any():
        aps.append(mpf.make_addplot(st_down, color='#E53935', width=1.5))
        labels.append(rtl('سوبر تريند هابط'))
    mc = mpf.make_marketcolors(
        up='#26a69a', down='#ef5350', edge='inherit',
        wick='inherit', volume={'up': '#80cbc4', 'down': '#ef9a9a'})
    sty = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=':', gridcolor='#dddddd',
        rc={'axes.facecolor': '#FAFAFA'})
    kw = dict(type='candle', style=sty, volume=False, figsize=(14, 7),
              returnfig=True, warn_too_much_data=9999)
    if aps:
        kw['addplot'] = aps
    fig, ax = mpf.plot(p, **kw)
    if labels:
        ax[0].legend(labels, loc='upper left', fontsize=8, prop=MPL_FONT_PROP)
    fig.subplots_adjust(right=0.95, left=0.05)
    return chart_bytes(fig)



# ═══════════════════════════════════════════════════════════════
# PART 6: main_charts_page WITH CORRECTED ARABIC TITLES
# ═══════════════════════════════════════════════════════════════
#
# FIXES: Titles now include Arabic name + English abbreviation in parentheses
# ═══════════════════════════════════════════════════════════════

    def main_charts_page(self, main_img, sma_img, ema_img, alligator_img, supertrend_img):
        self._bar('الرسم البياني')
        self._foot()
        c = self.c
        y = PAGE_H - 44 * mm

        # ── Section 1: Support & Resistance (full-width) ──
        y = self._stitle(y, 'الدعوم والمقاومات')
        y = self._img(y, main_img, 78 * mm)
        y -= 10 * mm  # 1cm spacing

        # ── Section 2: SMA (right) | EMA (left) side-by-side ──
        mid_x = MG + CW / 2
        right_title_x = PAGE_W - MG
        left_title_x  = mid_x - 4 * mm

        c.setFillColor(NAVY)
        self._font(True, 9)
        # ★ CORRECTED TITLES: Arabic name (English abbreviation)
        c.drawRightString(right_title_x, y,
                          rtl('المتوسطات المتحركة البسيطة (SMA)'))
        c.drawRightString(left_title_x, y,
                          rtl('المتوسطات المتحركة الأسية (EMA)'))
        c.setStrokeColor(TEAL)
        c.setLineWidth(1.0)
        c.line(mid_x + 2 * mm, y - 3, PAGE_W - MG, y - 3)
        c.line(MG,             y - 3, mid_x - 2 * mm, y - 3)
        y -= 8 * mm

        sma_img.seek(0); img_sma = ImageReader(sma_img)
        ema_img.seek(0); img_ema = ImageReader(ema_img)
        dw = (CW / 2) - 3 * mm
        maxh = 58 * mm

        ratio_sma = img_sma.getSize()[1] / img_sma.getSize()[0]
        dh_sma = min(dw * ratio_sma, maxh)
        dw_sma = dh_sma / ratio_sma

        ratio_ema = img_ema.getSize()[1] / img_ema.getSize()[0]
        dh_ema = min(dw * ratio_ema, maxh)
        dw_ema = dh_ema / ratio_ema

        x_sma = PAGE_W - MG - dw_sma  # right
        x_ema = MG                      # left
        c.drawImage(img_sma, x_sma, y - dh_sma, dw_sma, dh_sma)
        c.drawImage(img_ema, x_ema, y - dh_ema, dw_ema, dh_ema)
        y -= max(dh_sma, dh_ema) + 5 * mm

        # ── Section 3: Alligator (right) | Supertrend (left) side-by-side ──
        c.setFillColor(NAVY)
        self._font(True, 9)
        # ★ CORRECTED TITLES
        c.drawRightString(right_title_x, y,
                          rtl('مؤشر التمساح (Alligator)'))
        c.drawRightString(left_title_x, y,
                          rtl('مؤشر السوبر تريند (Supertrend)'))
        c.setStrokeColor(TEAL)
        c.setLineWidth(1.0)
        c.line(mid_x + 2 * mm, y - 3, PAGE_W - MG, y - 3)
        c.line(MG,             y - 3, mid_x - 2 * mm, y - 3)
        y -= 8 * mm

        alligator_img.seek(0);  img_alg = ImageReader(alligator_img)
        supertrend_img.seek(0); img_st  = ImageReader(supertrend_img)

        ratio_alg = img_alg.getSize()[1] / img_alg.getSize()[0]
        dh_alg = min(dw * ratio_alg, maxh)
        dw_alg = dh_alg / ratio_alg

        ratio_st = img_st.getSize()[1] / img_st.getSize()[0]
        dh_st = min(dw * ratio_st, maxh)
        dw_st = dh_st / ratio_st

        x_alg = PAGE_W - MG - dw_alg  # right
        x_st  = MG                      # left
        c.drawImage(img_alg, x_alg, y - dh_alg, dw_alg, dh_alg)
        c.drawImage(img_st,  x_st,  y - dh_st,  dw_st,  dh_st)

        self.c.showPage()


# ═══════════════════════════════════════════════════════════════
# PART 7: TADAWUL TICKERS LIST
# ═══════════════════════════════════════════════════════════════
#
# Complete list of all Tadawul (Saudi stock exchange) tickers.
# Used for Wolfe wave scanning.
# To add/remove tickers: simply edit this list.
# ═══════════════════════════════════════════════════════════════

TADAWUL_TICKERS = [
    '^TASI.SR',
    '1010.SR', '1020.SR', '1030.SR', '1050.SR', '1060.SR', '1080.SR',
    '1111.SR', '1120.SR', '1140.SR', '1150.SR', '1180.SR', '1182.SR',
    '1183.SR', '1201.SR', '1202.SR', '1210.SR', '1211.SR', '1212.SR',
    '1213.SR', '1214.SR', '1301.SR', '1302.SR', '1303.SR', '1304.SR',
    '1320.SR', '1321.SR', '1322.SR', '1323.SR', '1810.SR', '1820.SR',
    '1830.SR', '1831.SR', '1832.SR', '1833.SR', '1834.SR', '1835.SR',
    '2001.SR', '2010.SR', '2020.SR', '2030.SR', '2040.SR', '2050.SR',
    '2060.SR', '2070.SR', '2080.SR', '2081.SR', '2082.SR', '2083.SR',
    '2084.SR', '2090.SR', '2100.SR', '2110.SR', '2120.SR', '2130.SR',
    '2140.SR', '2150.SR', '2160.SR', '2170.SR', '2180.SR', '2190.SR',
    '2200.SR', '2210.SR', '2220.SR', '2222.SR', '2223.SR', '2230.SR',
    '2240.SR', '2250.SR', '2270.SR', '2280.SR', '2281.SR', '2282.SR',
    '2283.SR', '2284.SR', '2285.SR', '2286.SR', '2287.SR', '2288.SR',
    '2290.SR', '2300.SR', '2310.SR', '2320.SR', '2330.SR', '2340.SR',
    '2350.SR', '2360.SR', '2370.SR', '2380.SR', '2381.SR', '2382.SR',
    '3002.SR', '3003.SR', '3004.SR', '3005.SR', '3007.SR', '3008.SR',
    '3010.SR', '3020.SR', '3030.SR', '3040.SR', '3050.SR', '3060.SR',
    '3080.SR', '3090.SR', '3091.SR', '3092.SR',
    '4001.SR', '4002.SR', '4003.SR', '4004.SR', '4005.SR', '4006.SR',
    '4007.SR', '4008.SR', '4009.SR', '4011.SR', '4012.SR', '4013.SR',
    '4014.SR', '4015.SR', '4016.SR', '4017.SR', '4018.SR', '4019.SR',
    '4020.SR', '4021.SR', '4030.SR', '4031.SR', '4040.SR', '4050.SR',
    '4051.SR', '4061.SR', '4070.SR', '4071.SR', '4072.SR', '4080.SR',
    '4081.SR', '4082.SR', '4083.SR', '4084.SR', '4090.SR', '4100.SR',
    '4110.SR', '4130.SR', '4140.SR', '4141.SR', '4142.SR', '4143.SR',
    '4144.SR', '4145.SR', '4146.SR', '4147.SR', '4148.SR', '4150.SR',
    '4160.SR', '4161.SR', '4162.SR', '4163.SR', '4164.SR', '4165.SR',
    '4170.SR', '4180.SR', '4190.SR', '4191.SR', '4192.SR', '4193.SR',
    '4194.SR', '4200.SR', '4210.SR', '4220.SR', '4230.SR', '4240.SR',
    '4250.SR', '4260.SR', '4261.SR', '4262.SR', '4263.SR', '4264.SR',
    '4265.SR', '4270.SR', '4280.SR', '4290.SR', '4291.SR', '4292.SR',
    '4300.SR', '4310.SR', '4320.SR', '4321.SR', '4322.SR', '4323.SR',
    '4324.SR', '4325.SR', '4326.SR', '4327.SR',
    '4330.SR', '4331.SR', '4332.SR', '4333.SR', '4334.SR', '4335.SR',
    '4336.SR', '4337.SR', '4338.SR', '4339.SR', '4340.SR', '4342.SR',
    '4344.SR', '4345.SR', '4346.SR', '4347.SR', '4348.SR', '4349.SR',
    '4350.SR',
    '5110.SR',
    '6001.SR', '6002.SR', '6004.SR', '6010.SR', '6012.SR', '6013.SR',
    '6014.SR', '6015.SR', '6016.SR', '6017.SR', '6018.SR', '6019.SR',
    '6020.SR', '6040.SR', '6050.SR', '6060.SR', '6070.SR', '6090.SR',
    '7010.SR', '7020.SR', '7030.SR', '7040.SR', '7200.SR', '7201.SR',
    '7202.SR', '7203.SR', '7204.SR', '7211.SR',
    '8010.SR', '8012.SR', '8020.SR', '8030.SR', '8040.SR', '8050.SR',
    '8060.SR', '8070.SR', '8100.SR', '8120.SR', '8150.SR', '8160.SR',
    '8170.SR', '8180.SR', '8190.SR', '8200.SR', '8210.SR', '8230.SR',
    '8240.SR', '8250.SR', '8260.SR', '8270.SR', '8280.SR', '8300.SR',
    '8310.SR', '8311.SR', '8313.SR',
]
