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

# ── Optional STOCKS scraper dependencies ──
try:
    import sys, time, subprocess
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

# ═══════════════════════════════════════════════════════════════
# SECTION 1: FONT SETUP
# ───────────────────────────────────────────────────────────────
# Cairo  → used for Wolfe wave chart Arabic text
# Amiri  → used for PDF reports (regular + bold)
#
# To change fonts:
#   1. Replace the .ttf files in the same directory
#   2. Update the file names below
#   3. Update AR_FONT / AR_FONT_BOLD names if registering different fonts
# ═══════════════════════════════════════════════════════════════
HERE = os.path.dirname(os.path.abspath(__file__))

CAIRO_PATH      = os.path.join(HERE, 'Cairo-Regular.ttf')
AMIRI_REG_PATH  = os.path.join(HERE, 'Amiri-Regular.ttf')
AMIRI_BOLD_PATH = os.path.join(HERE, 'Amiri-Bold.ttf')

AR_FONT      = 'Amiri'
AR_FONT_BOLD = 'Amiri-Bold'
AR_RE        = re.compile(r'[\u0600-\u06FF]')

MPL_FONT_PROP      = None
MPL_FONT_PROP_BOLD = None


def _download_font(url, path):
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
            logger.info(f'Downloaded: {os.path.basename(path)}')
        except Exception as e:
            logger.warning(f'Font download failed ({os.path.basename(path)}): {e}')


def _init_fonts():
    global MPL_FONT_PROP, MPL_FONT_PROP_BOLD, ARABIC_FONT

    _download_font(
        'https://github.com/google/fonts/raw/main/ofl/cairo/static/Cairo-Regular.ttf',
        CAIRO_PATH)
    _download_font(
        'https://github.com/google/fonts/raw/refs/heads/main/ofl/amiri/Amiri-Regular.ttf',
        AMIRI_REG_PATH)
    _download_font(
        'https://github.com/google/fonts/raw/refs/heads/main/ofl/amiri/Amiri-Bold.ttf',
        AMIRI_BOLD_PATH)

    if os.path.exists(AMIRI_REG_PATH):
        try: pdfmetrics.registerFont(TTFont(AR_FONT, AMIRI_REG_PATH))
        except Exception: pass
    if os.path.exists(AMIRI_BOLD_PATH):
        try: pdfmetrics.registerFont(TTFont(AR_FONT_BOLD, AMIRI_BOLD_PATH))
        except Exception: pass

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

# ═══════════════════════════════════════════════════════════════
# SECTION 2: ARABIC TEXT HELPERS
# ───────────────────────────────────────────────────────────────
# ar()  → reshape+bidi for matplotlib chart labels (Wolfe)
# rtl() → reshape+bidi for PDF / matplotlib Arabic text
# tx()  → safe wrapper: returns '-' if None
#
# To add new text utilities, add them here.
# ═══════════════════════════════════════════════════════════════
def ar(text: str) -> str:
    try: return get_display(arabic_reshaper.reshape(str(text)))
    except Exception: return str(text)


def rtl(txt):
    if txt is None: return ''
    s = str(txt)
    if AR_RE.search(s): return get_display(arabic_reshaper.reshape(s))
    return s


def tx(txt):
    if txt is None: return '-'
    return rtl(str(txt))


def short_text(s, n=40):
    s = str(s or '-')
    return s if len(s) <= n else s[:n - 1] + '…'


def safe(info, key, default=None):
    v = info.get(key)
    return default if v is None else v


# ═══════════════════════════════════════════════════════════════
# SECTION 3: COMPANY NAMES, SECTOR MAP, STATIC DATA, TICKERS
# ───────────────────────────────────────────────────────────────
# COMPANY_NAMES  : ticker → Arabic display name
# SECTOR_MAP     : ticker → (sector, industry)
# STOCKS_STATIC_DATA : code → fundamental data dict
# TADAWUL_TICKERS: list of all tickers to scan
#
# To add a new company:
#   1. Add to COMPANY_NAMES
#   2. Add to SECTOR_MAP
#   3. Add to STOCKS_STATIC_DATA (if fundamentals known)
#   4. Add to TADAWUL_TICKERS
# ═══════════════════════════════════════════════════════════════

COMPANY_NAMES = {
    '^TASI.SR':'تاسي','1010.SR':'الرياض',
    # ... (keep ALL your existing entries exactly as-is) ...
    '8313.SR':'رسن',
}

TICKER_ALIASES = {
    "TASI": "^TASI.SR", "^TASI": "^TASI.SR", "^TASI.SR": "^TASI.SR",
    "تاسي": "^TASI.SR", "تاسى": "^TASI.SR",
}


def _normalize_arabic(text: str) -> str:
    text = text.strip()
    text = text.replace("ى", "ي").replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    return text

def _normalize_key(text: str) -> str:
    return _normalize_arabic(text.upper())

def get_name(ticker: str) -> str:
    return COMPANY_NAMES.get(ticker, ticker)

_ALIAS_MAP = {_normalize_key(a): c for a, c in TICKER_ALIASES.items()}


def find_ticker(query: str) -> str | None:
    query = query.strip()
    query_key = _normalize_key(query)
    query_normalized = _normalize_arabic(query)

    if query_key in _ALIAS_MAP:
        return _ALIAS_MAP[query_key]

    for ticker, name in COMPANY_NAMES.items():
        ticker_upper = ticker.upper()
        if query_key == ticker_upper: return ticker
        code = ticker_upper[:-3] if ticker_upper.endswith(".SR") else ticker_upper
        if query_key == code: return ticker
        code_clean = code.lstrip("^")
        if query_key == code_clean: return ticker
        if query_normalized == _normalize_arabic(name): return ticker

    if len(query_normalized) >= 2:
        for ticker, name in COMPANY_NAMES.items():
            if query_normalized in _normalize_arabic(name):
                return ticker
    return None


SECTOR_MAP = {
    '1010.SR':('المالية','البنوك'),'1020.SR':('المالية','البنوك'),
    # ... (keep ALL your existing entries exactly as-is) ...
    '1111.SR':('المالية','أسواق المال'),
}

def get_sector_industry(ticker):
    return SECTOR_MAP.get(ticker, (None, None))


STOCKS_STATIC_DATA = {
    "1010": {"Numberofshare": "3,000.00", "Eps": "3.47", "Bookvalue": "21.37",
             "Parallel_value": "10.00", "PE_ratio": "8.19", "PB_ratio": "1.33",
             "ROA": "", "ROE": "16.91"},
    "1020": {"Numberofshare": "1,281.25", "Eps": "1.17", "Bookvalue": "11.73",
             "Parallel_value": "10.00", "PE_ratio": "9.62", "PB_ratio": "0.96",
             "ROA": "", "ROE": "10.54"},
    # ... (keep ALL your existing entries exactly as-is) ...
}

_PE_NON_NUMERIC = {"سالب", "أكبر من 100", "-", ""}


TADAWUL_TICKERS = [
    '^TASI.SR','1010.SR',
    # ... (keep ALL your existing entries exactly as-is) ...
    '8313.SR',
]

# ═══════════════════════════════════════════════════════════════
# SECTION 3B: STOCKS STATIC DATA PARSING & ENRICHMENT
# ───────────────────────────────────────────────────────────────
# Priority: STOCKS_STATIC_DATA → Yahoo Finance fallback
#
# _safe_float()          : parse string to float (handles parenthesized negatives)
# _safe_display()        : clean display string
# _get_current_price()   : extract price from info dict
# _format_roa_roe()      : parse ROA/ROE strings → (numeric, display)
# _enrich_with_STOCKS()  : main enrichment function
#
# To add a new field from STOCKS_STATIC_DATA:
#   1. Add the field key to the dict entries above
#   2. Add a numbered section in _enrich_with_STOCKS()
#   3. Store result in info dict with both raw value and _display variant
# ═══════════════════════════════════════════════════════════════

def _safe_float(value: str) -> float | None:
    """Parse string to float. Handles '(0.23 )' → -0.23. Returns None if unparsable."""
    if value is None: return None
    value = str(value).strip()
    if not value or value in _PE_NON_NUMERIC or value == "-": return None
    if value.startswith("(") and value.endswith(")"):
        inner = value[1:-1].strip()
        try: return -abs(float(inner))
        except (ValueError, TypeError): return None
    try: return float(value)
    except (ValueError, TypeError): return None


def _safe_display(value: str) -> str | None:
    if value is None: return None
    value = str(value).strip()
    if not value or value == "-": return None
    return value


def _get_current_price(info: dict) -> float | None:
    for key in ("currentPrice", "regularMarketPrice", "previousClose"):
        val = info.get(key)
        if val is not None:
            try: return float(val)
            except (ValueError, TypeError): continue
    return None


def _ensure_eps_formatted(info: dict) -> None:
    """Ensure trailingEpsFormatted exists. Uses yfinance trailingEps if available."""
    if info.get("trailingEpsFormatted"): return
    yahoo_eps = info.get("trailingEps")
    if yahoo_eps is not None:
        try:
            yahoo_eps = float(yahoo_eps)
            info["trailingEpsFormatted"] = f"({abs(yahoo_eps)})" if yahoo_eps < 0 else str(yahoo_eps)
        except (ValueError, TypeError):
            info["trailingEpsFormatted"] = "لاحقا"
    else:
        info["trailingEpsFormatted"] = "لاحقا"


def _format_roa_roe(value_str: str) -> tuple[float | None, str]:
    """Parse ROA/ROE from STOCKS format → (numeric_value, display_string)."""
    if value_str is None: return None, ""
    value_str = str(value_str).strip()
    if not value_str or value_str == "-": return None, ""
    numeric = _safe_float(value_str)
    if numeric is not None:
        display = f"({abs(numeric)}%)" if numeric < 0 else f"{numeric}%"
        return numeric, display
    return None, ""


def _enrich_with_STOCKS(ticker: str, info: dict) -> None:
    """
    Enrich info dict with STOCKS_STATIC_DATA.
    Priority: STOCKS data FIRST → Yahoo Finance fallback.
    Modifies info dict in-place.
    """
    code = ticker.replace(".SR", "").strip()
    STOCKS = STOCKS_STATIC_DATA.get(code)

    if not STOCKS:
        logger.info(f"STOCKS static: no data for {code}, yfinance only")
        _ensure_eps_formatted(info)
        _ensure_roa_roe_from_yahoo(info)
        _ensure_bookvalue_display(info)
        return

    logger.info(f"STOCKS static: found data for {code}")

    # ── 1. ربح السهم (EPS) ──
    eps = _safe_float(STOCKS.get("Eps"))
    if eps is not None:
        info["trailingEps"] = eps
        info["trailingEpsFormatted"] = f"({abs(eps)})" if eps < 0 else str(eps)
    else:
        _ensure_eps_formatted(info)

    # ── 2. القيمة الدفترية (Book Value) ──
    bv = _safe_float(STOCKS.get("Bookvalue"))
    if bv is not None:
        info["bookValue"] = bv
        info["bookValue_display"] = f"({abs(bv)})" if bv < 0 else str(bv)
    else:
        _ensure_bookvalue_display(info)

    # ── 3. مكرر الربح (P/E Ratio) ──
    pe_raw = STOCKS.get("PE_ratio", "").strip()
    pe = _safe_float(pe_raw)
    if pe is not None and pe > 0:
        info["trailingPE"] = pe
        info["trailingPE_display"] = str(pe)
    elif pe_raw == "سالب":
        info["trailingPE"] = None; info["trailingPE_display"] = "سالب"
    elif pe_raw == "أكبر من 100":
        info["trailingPE"] = None; info["trailingPE_display"] = "أكبر من 100"
    else:
        yf_pe = info.get("trailingPE")
        info["trailingPE_display"] = str(round(float(yf_pe), 2)) if yf_pe is not None else "-"

    # ── 4. مضاعف القيمة الدفترية (P/B Ratio) ──
    pb = _safe_float(STOCKS.get("PB_ratio"))
    if pb is not None:
        info["priceToBook"] = pb

    # ── 5. عدد الأسهم (Shares Outstanding) — in millions ──
    shares_str = STOCKS.get("Numberofshare", "")
    shares_clean = shares_str.replace(",", "").strip() if shares_str else ""
    shares = _safe_float(shares_clean)
    if shares is not None:
        shares_actual = shares * 1_000_000
        info["sharesOutstanding"] = shares_actual
        info["floatShares"] = shares_actual
        price = _get_current_price(info)
        if price and price > 0:
            info["marketCap"] = price * shares_actual

    # ── 6. القيمة الاسمية (Par Value) ──
    par = _safe_float(STOCKS.get("Parallel_value"))
    if par is not None:
        info["parValue"] = par

    # ── 7. العائد على الأصول (ROA) ──
    roa_numeric, roa_display = _format_roa_roe(STOCKS.get("ROA"))
    if roa_numeric is not None:
        info["returnOnAssets"] = roa_numeric / 100.0
        info["returnOnAssets_display"] = roa_display
    else:
        # fallback to yfinance
        yf_roa = info.get("returnOnAssets")
        if yf_roa is not None:
            try:
                roa_pct = float(yf_roa) * 100
                info["returnOnAssets_display"] = (
                    f"({abs(round(roa_pct, 2))}%)" if roa_pct < 0
                    else f"{round(roa_pct, 2)}%"
                )
            except (ValueError, TypeError):
                info["returnOnAssets_display"] = "-"
        else:
            info["returnOnAssets_display"] = "-"

    # ── 8. العائد على حقوق المساهمين (ROE) ──
    roe_numeric, roe_display = _format_roa_roe(STOCKS.get("ROE"))
    if roe_numeric is not None:
        info["returnOnEquity"] = roe_numeric / 100.0
        info["returnOnEquity_display"] = roe_display
    else:
        yf_roe = info.get("returnOnEquity")
        if yf_roe is not None:
            try:
                roe_pct = float(yf_roe) * 100
                info["returnOnEquity_display"] = (
                    f"({abs(round(roe_pct, 2))}%)" if roe_pct < 0
                    else f"{round(roe_pct, 2)}%"
                )
            except (ValueError, TypeError):
                info["returnOnEquity_display"] = "-"
        else:
            info["returnOnEquity_display"] = "-"

    # ── 9. Recalculate P/E if still missing ──
    if not info.get("trailingPE") and not info.get("trailingPE_display"):
        eps_val = info.get("trailingEps")
        if eps_val and eps_val != 0:
            price = _get_current_price(info)
            if price and price > 0:
                try:
                    calculated_pe = round(price / float(eps_val), 2)
                    info["trailingPE"] = calculated_pe
                    info["trailingPE_display"] = str(calculated_pe)
                except Exception: pass

    # ── 10. Recalculate P/B if missing ──
    if not info.get("priceToBook"):
        bv_val = info.get("bookValue")
        if bv_val and float(bv_val) != 0:
            price = _get_current_price(info)
            if price and price > 0:
                try: info["priceToBook"] = round(price / float(bv_val), 4)
                except Exception: pass

    # ── 11. Defaults ──
    if not info.get("currency"):  info["currency"]  = "SAR"
    if not info.get("exchange"):  info["exchange"]  = "Tadawul"


def _ensure_roa_roe_from_yahoo(info: dict) -> None:
    """Ensure ROA/ROE _display fields exist using yfinance when STOCKS unavailable."""
    for field, display_key in [("returnOnAssets", "returnOnAssets_display"),
                                ("returnOnEquity", "returnOnEquity_display")]:
        if not info.get(display_key):
            yf_val = info.get(field)
            if yf_val is not None:
                try:
                    pct = float(yf_val) * 100
                    info[display_key] = (f"({abs(round(pct, 2))}%)" if pct < 0
                                         else f"{round(pct, 2)}%")
                except (ValueError, TypeError):
                    info[display_key] = "-"
            else:
                info[display_key] = "-"


def _ensure_bookvalue_display(info: dict) -> None:
    """Ensure bookValue_display exists using yfinance when STOCKS unavailable."""
    if not info.get("bookValue_display"):
        bv = info.get("bookValue")
        if bv is not None:
            try:
                bv_f = float(bv)
                info["bookValue_display"] = (f"({abs(bv_f)})" if bv_f < 0
                                              else str(round(bv_f, 2)))
            except (ValueError, TypeError):
                info["bookValue_display"] = "-"
        else:
            info["bookValue_display"] = "-"

# ═══════════════════════════════════════════════════════════════
# SECTION 4: PDF THEME CONSTANTS
# ───────────────────────────────────────────────────────────────
# To change colors: update the HEX values below.
# To change page size: update PAGE_W, PAGE_H (e.g. to letter).
# ═══════════════════════════════════════════════════════════════
NAVY_HEX, TEAL_HEX, GREEN_HEX = '#1B2A4A', '#00897B', '#4CAF50'
RED_HEX, ORANGE_HEX, LGRAY_HEX = '#E53935', '#FF9800', '#F0F3F7'
DGRAY_HEX, TXTDARK_HEX, WHITE_HEX = '#5A6272', '#1A1A2E', '#FFFFFF'
BLUE_HEX, VIOLET_HEX, BLACK_HEX = '#2196F3', '#C620F8', '#000000'

NAVY, TEAL, GREEN = HexColor(NAVY_HEX), HexColor(TEAL_HEX), HexColor(GREEN_HEX)
RED, ORANGE, LGRAY = HexColor(RED_HEX), HexColor(ORANGE_HEX), HexColor(LGRAY_HEX)
DGRAY, TXTDARK, WHITE = HexColor(DGRAY_HEX), HexColor(TXTDARK_HEX), HexColor(WHITE_HEX)
BLUE, VIOLET, BLACK = HexColor(BLUE_HEX), HexColor(VIOLET_HEX), HexColor(BLACK_HEX)

PAGE_W, PAGE_H = A4
MG = 18 * mm
CW = PAGE_W - 2 * MG

# ═══════════════════════════════════════════════════════════════
# SECTION 5: NUMBER / PERCENT FORMATTERS
# ═══════════════════════════════════════════════════════════════
def fmt_n(v, d=2):
    if v is None: return '-', None
    try:
        v = float(v); av = abs(v)
        if av >= 1e12: return f'{v/1e12:.{d}f} T', None
        if av >= 1e9:  return f'{v/1e9:.{d}f} B', None
        if av >= 1e6:  return f'{v/1e6:.{d}f} M', None
        return f'{v:,.{d}f}', RED if v < 0 else None
    except Exception: return '-', None


def fmt_p(v):
    if v is None: return '-', None
    try:
        fv = float(v)
        if abs(fv) <= 1: return f'{fv*100:.2f}%', RED if fv < 0 else None
        return f'{fv:.2f}%', RED if fv < 0 else None
    except Exception: return '-', None


def chart_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════
# SECTION 6: STOCK DATA FETCHING & INDICATORS
# ───────────────────────────────────────────────────────────────
# fetch_data()         : download price data + enrich info with STOCKS
# _compute_supertrend(): standalone supertrend calculator
# compute_indicators() : add all technical indicator columns to df
#
# To add a new indicator:
#   1. Add calculation in compute_indicators()
#   2. Reference it in compute_score_criteria() or gen_signals()
#   3. Add chart function if needed
# ═══════════════════════════════════════════════════════════════
def fetch_data(ticker):
    """Fetch price data + supplement info fields for Saudi stocks."""
    try:
        stk = yf.Ticker(ticker)
        df = stk.history(period='1y')
        if df.empty: return None, None, {}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df2 = stk.history(period='2y')
        if isinstance(df2.columns, pd.MultiIndex):
            df2.columns = df2.columns.get_level_values(0)
        try: info = stk.info or {}
        except Exception: info = {}

        price_now = float(df['Close'].iloc[-1])

        # ── fast_info supplementation ──
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

        # ── 52-week range from 2y data ──
        try:
            tail252 = df2['High'].tail(252) if len(df2) >= 252 else df['High']
            tail252l = df2['Low'].tail(252) if len(df2) >= 252 else df['Low']
            info['fiftyTwoWeekHigh'] = float(tail252.max())
            info['fiftyTwoWeekLow']  = float(tail252l.min())
        except Exception: pass

        if ticker.endswith('.SR'):
            if not info.get('currency'): info['currency'] = 'SAR'
            if not info.get('exchange'): info['exchange'] = 'Tadawul'

        sec, ind = get_sector_industry(ticker)
        if sec and not info.get('sector'):   info['sector']   = sec
        if ind and not info.get('industry'): info['industry'] = ind

        if not info.get('averageVolume'):
            try: info['averageVolume'] = int(df['Volume'].mean())
            except Exception: pass

        # ── Beta calculation ──
        if not info.get('beta'):
            try:
                mkt = yf.Ticker('^TASI.SR').history(period='1y')['Close']
                stk_ret = df['Close'].pct_change().dropna()
                mkt_ret = mkt.pct_change().dropna()
                aligned = pd.concat([stk_ret, mkt_ret], axis=1, join='inner').dropna()
                if len(aligned) > 30:
                    cov = aligned.cov().iloc[0,1]
                    var = aligned.iloc[:,1].var()
                    if var > 0: info['beta'] = round(cov / var, 3)
            except Exception: pass

        # ── Dividends ──
        try:
            divs = stk.dividends
            if divs is not None and len(divs) > 0:
                cutoff = pd.Timestamp.now(tz=divs.index.tz) - pd.DateOffset(years=1)
                annual_div = float(divs[divs.index >= cutoff].sum())
                if annual_div > 0:
                    if not info.get('dividendRate'):  info['dividendRate']  = annual_div
                    if not info.get('dividendYield'): info['dividendYield'] = annual_div / price_now
        except Exception: pass

        # ── Financial statements ──
        fin_ok = bs_ok = False
        rev = ni = equity = total_assets = curr_assets = curr_liab = inventory = None
        try:
            fin = stk.financials; bs = stk.balance_sheet
            fin_ok = not fin.empty; bs_ok = not bs.empty

            def _fv(frame, *cands):
                for c in cands:
                    ks = [k for k in frame.index if c.lower() in str(k).lower()]
                    if ks:
                        try: return float(frame.loc[ks[0]].iloc[0])
                        except Exception: continue
                return None

            if fin_ok:
                rev = _fv(fin, 'Total Revenue'); ni = _fv(fin, 'Net Income Common Stockholders', 'Net Income')
                op_inc = _fv(fin, 'Operating Income'); gross = _fv(fin, 'Gross Profit')
                ebitda = _fv(fin, 'EBITDA', 'Normalized EBITDA')
                if rev and not info.get('totalRevenue'):      info['totalRevenue']      = rev
                if ni  and not info.get('netIncomeToCommon'): info['netIncomeToCommon'] = ni
                if ebitda and not info.get('ebitda'):         info['ebitda']            = ebitda
                if rev and rev != 0:
                    if gross  and not info.get('grossMargins'):     info['grossMargins']     = gross / rev
                    if op_inc and not info.get('operatingMargins'): info['operatingMargins'] = op_inc / rev
                    if ni     and not info.get('profitMargins'):    info['profitMargins']    = ni / rev

            if bs_ok:
                equity      = _fv(bs, 'Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest')
                total_assets = _fv(bs, 'Total Assets')
                curr_assets  = _fv(bs, 'Current Assets')
                curr_liab    = _fv(bs, 'Current Liabilities')
                inventory    = _fv(bs, 'Inventory')
                cash_val = _fv(bs, 'Cash Cash Equivalents And Short Term Investments', 'Cash And Cash Equivalents')
                debt_val = _fv(bs, 'Total Debt', 'Long Term Debt And Capital Lease Obligation', 'Long Term Debt')
                if cash_val and not info.get('totalCash'): info['totalCash'] = cash_val
                if debt_val and not info.get('totalDebt'): info['totalDebt'] = debt_val
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
                        info['quickRatio'] = round((curr_assets - (inventory or 0)) / curr_liab, 2)
                if debt_val and equity and equity != 0 and not info.get('debtToEquity'):
                    info['debtToEquity'] = round((debt_val / abs(equity)) * 100, 2)

            shares_out = info.get('sharesOutstanding')
            if fin_ok and ni is not None and shares_out and float(shares_out) > 0:
                eps = ni / float(shares_out)
                if not info.get('trailingEps'): info['trailingEps'] = eps
                if not info.get('trailingPE') and eps != 0:
                    info['trailingPE'] = price_now / eps

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

            div_rate = info.get('dividendRate'); eps_v = info.get('trailingEps')
            if div_rate and eps_v and eps_v > 0 and not info.get('payoutRatio'):
                info['payoutRatio'] = div_rate / eps_v
            if not info.get('floatShares') and info.get('sharesOutstanding'):
                info['floatShares'] = info['sharesOutstanding']
        except Exception as e:
            logger.warning(f"fetch_data supplement error: {e}")

        # ── STOCKS enrichment (overrides yfinance fundamentals) ──
        try: _enrich_with_STOCKS(ticker, info)
        except Exception as _ae: logger.warning(f"STOCKS enrichment skipped: {_ae}")

        # ── YF fallbacks (only if STOCKS didn't populate) ──
        if not info.get('volume') and len(df) > 0 and 'Volume' in df.columns:
            try: info['volume'] = float(df['Volume'].iloc[-1])
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

        return df, df2, info
    except Exception as e:
        logger.error(f"fetch_data error: {e}")
        return None, None, {}


# ── SUPERTREND HELPER ──
def _compute_supertrend(df, period=10, multiplier=3.0):
    h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float)
    c = df['Close'].values.astype(float)
    n = len(c)
    if n == 0:
        empty = pd.Series(dtype=float, index=df.index)
        return empty, empty.astype(int)
    tr = np.zeros(n); tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().values
    hl2 = (h + l) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    upper_band = np.zeros(n); lower_band = np.zeros(n)
    supertrend = np.zeros(n); direction = np.zeros(n, dtype=int)
    upper_band[0] = upper_basic[0]; lower_band[0] = lower_basic[0]
    if c[0] <= upper_band[0]:
        supertrend[0] = upper_band[0]; direction[0] = -1
    else:
        supertrend[0] = lower_band[0]; direction[0] = 1
    for i in range(1, n):
        upper_band[i] = upper_basic[i] if (upper_basic[i] < upper_band[i-1] or c[i-1] > upper_band[i-1]) else upper_band[i-1]
        lower_band[i] = lower_basic[i] if (lower_basic[i] > lower_band[i-1] or c[i-1] < lower_band[i-1]) else lower_band[i-1]
        if supertrend[i-1] == upper_band[i-1]:
            if c[i] > upper_band[i]: supertrend[i] = lower_band[i]; direction[i] = 1
            else: supertrend[i] = upper_band[i]; direction[i] = -1
        else:
            if c[i] < lower_band[i]: supertrend[i] = upper_band[i]; direction[i] = -1
            else: supertrend[i] = lower_band[i]; direction[i] = 1
    return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)


# ── INDICATOR COMPUTATION ──
def compute_indicators(df):
    # (This function is unchanged from your original — keep it exactly as-is)
    d = df.copy()
    c, h, l, v = d['Close'], d['High'], d['Low'], d['Volume']
    d['SMA7'] = c.rolling(7, min_periods=1).mean()
    d['SMA20'] = c.rolling(20, min_periods=1).mean()
    d['SMA50'] = c.rolling(50, min_periods=1).mean()
    d['SMA100'] = c.rolling(100, min_periods=1).mean()
    d['SMA200'] = c.rolling(200, min_periods=1).mean()
    d['EMA7'] = c.ewm(span=7, adjust=False, min_periods=1).mean()
    d['EMA20'] = c.ewm(span=20, adjust=False, min_periods=1).mean()
    d['EMA50'] = c.ewm(span=50, adjust=False, min_periods=1).mean()
    d['EMA100'] = c.ewm(span=100, adjust=False, min_periods=1).mean()
    d['EMA200'] = c.ewm(span=200, adjust=False, min_periods=1).mean()
    d['EMA12'] = c.ewm(span=12, adjust=False, min_periods=1).mean()
    d['EMA26'] = c.ewm(span=26, adjust=False, min_periods=1).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d['RSI'] = 100 - 100 / (1 + rs)
    d['MACD'] = d['EMA12'] - d['EMA26']
    d['MACD_Sig'] = d['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
    d['MACD_H'] = d['MACD'] - d['MACD_Sig']
    bm = c.rolling(20, min_periods=1).mean(); bs = c.rolling(20, min_periods=1).std()
    d['BB_U'], d['BB_M'], d['BB_L'] = bm + 2*bs, bm, bm - 2*bs
    bb_range = d['BB_U'] - d['BB_L']
    d['BB_P'] = np.where(bb_range != 0, (c - d['BB_L']) / bb_range, 0.5)
    l14 = l.rolling(14, min_periods=1).min(); h14 = h.rolling(14, min_periods=1).max()
    stoch_range = h14 - l14
    d['SK'] = np.where(stoch_range != 0, 100*(c-l14)/stoch_range, 50)
    d['SD'] = pd.Series(d['SK'], index=d.index).rolling(3, min_periods=1).mean()
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    d['ATR'] = tr.rolling(14, min_periods=1).mean()
    d['OBV'] = (np.sign(c.diff()) * v).fillna(0).cumsum()
    tp = (h + l + c) / 3
    cumsum_vol = v.cumsum().replace(0, np.nan)
    d['VWAP'] = (tp * v).cumsum() / cumsum_vol
    plus_dm = h.diff().clip(lower=0); minus_dm = (-l.diff()).clip(lower=0)
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
    d['SAR'] = c.rolling(5, min_periods=1).min()
    high9 = h.rolling(9, min_periods=1).max(); low9 = l.rolling(9, min_periods=1).min()
    d['Tenkan'] = (high9 + low9) / 2
    high26 = h.rolling(26, min_periods=1).max(); low26 = l.rolling(26, min_periods=1).min()
    d['Kijun'] = (high26 + low26) / 2
    d['Senkou_A'] = ((d['Tenkan'] + d['Kijun']) / 2).shift(26)
    high52 = h.rolling(52, min_periods=1).max(); low52 = l.rolling(52, min_periods=1).min()
    d['Senkou_B'] = ((high52 + low52) / 2).shift(26)
    d['Chikou'] = c.shift(-26)
    d['Alligator_Jaw'] = c.shift(8).rolling(13, min_periods=1).mean()
    d['Alligator_Teeth'] = c.shift(5).rolling(8, min_periods=1).mean()
    d['Alligator_Lips'] = c.shift(3).rolling(5, min_periods=1).mean()
    d['Volume_Up'] = np.where(c > c.shift(1), v, 0)
    d['Volume_Down'] = np.where(c < c.shift(1), v, 0)
    d['Bull_Volume'] = pd.Series(d['Volume_Up'], index=d.index).rolling(20, min_periods=1).sum()
    d['Bear_Volume'] = pd.Series(d['Volume_Down'], index=d.index).rolling(20, min_periods=1).sum()
    tp_cci = (h + l + c) / 3
    tp_mean = tp_cci.rolling(20, min_periods=1).mean()
    tp_mad = tp_cci.rolling(20, min_periods=1).apply(lambda x: np.mean(np.abs(x-x.mean())), raw=True)
    d['CCI'] = np.where(tp_mad != 0, (tp_cci - tp_mean)/(0.015*tp_mad), 0)
    h14w = h.rolling(14, min_periods=1).max(); l14w = l.rolling(14, min_periods=1).min()
    wr_range = h14w - l14w
    d['WILLR'] = np.where(wr_range != 0, -100*(h14w-c)/wr_range, -50)
    d['ROC12'] = c.pct_change(12) * 100
    return d


##################################
###################################

#PART 4 

def compute_performance(df, df2):
    c, c2 = df['Close'], df2['Close']
    last = float(c.iloc[-1])
    pers = {}
    for days, label in [(2,'1 Day'),(5,'1 Week'),(22,'1 Month'),(66,'3 Months'),(132,'6 Months')]:
        if len(c) >= days:
            pers[label] = (last / float(c.iloc[-days]) - 1) * 100
    if len(c2) >= 252:
        pers['1 Year'] = (last / float(c2.iloc[-252]) - 1) * 100
    year_start = pd.Timestamp(f'{datetime.now().year}-01-01')
    if c.index.tz is not None:
        year_start = year_start.tz_localize(c.index.tz)
    ytd = c[c.index >= year_start]
    if len(ytd) >= 2:
        pers['YTD'] = (last / float(ytd.iloc[0]) - 1) * 100
    dr = c.pct_change().dropna()
    if len(dr) == 0:
        risk = {k: '-' for k in ['Daily Volatility','Annual Volatility','Sharpe Ratio','Sortino Ratio','Max Drawdown','Avg Daily Return','Best Day','Worst Day']}
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
    cum = (1 + dr).cumprod()
    pk = cum.cummax()
    dd = (cum - pk) / pk
    risk.update({'Max Drawdown': f'{dd.min()*100:.2f}%', 'Avg Daily Return': f'{dr.mean()*100:.4f}%', 'Best Day': f'{dr.max()*100:.2f}%', 'Worst Day': f'{dr.min()*100:.2f}%'})
    return pers, risk, dd

def compute_score_criteria(d):
    last = d.iloc[-1]
    c = float(last['Close'])
    criteria = {}
    score = 0
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
        #('10. ROC 12 إيجابي - زخم صاعد', pd.notna(last['ROC12']) and float(last['ROC12']) > 0),
        ('10. مؤشر معدل التغير > 0', pd.notna(last['ROC12']) and float(last['ROC12']) > 0),
        ('11. مؤشر ADX > 25', pd.notna(last['ADX']) and float(last['ADX']) > 25),
        ('12. مؤشر +DI > مؤشر -DI', pd.notna(last['PDI']) and pd.notna(last['MDI']) and float(last['PDI']) > float(last['MDI'])),
    ]
    if len(d) >= 5:
        obv_curr = float(last['OBV']) if pd.notna(last['OBV']) else 0
        obv_prev = float(d.iloc[-5]['OBV']) if pd.notna(d.iloc[-5]['OBV']) else 0
        checks.append(('13. مؤشر OBV صاعد', obv_curr > obv_prev))
    else:
        checks.append(('13. مؤشر OBV صاعد', False))
    checks += [
        ('14. سيولة شرائية مسيطرة', pd.notna(last['Bull_Volume']) and pd.notna(last['Bear_Volume']) and float(last['Bull_Volume']) > float(last['Bear_Volume'])),
    ]
    sa = float(last['Senkou_A']) if pd.notna(last['Senkou_A']) else 0
    sb = float(last['Senkou_B']) if pd.notna(last['Senkou_B']) else 0
    kumo_top = max(sa, sb)
    checks += [
        ('15. السعر > سحابة الإيشيموكو', c > kumo_top and kumo_top > 0),
        ('16. خط التنكن > خط الكيجن', pd.notna(last['Tenkan']) and pd.notna(last['Kijun']) and float(last['Tenkan']) > float(last['Kijun'])),
        ('17. مؤشر SAR إيجابي', pd.notna(last['SAR']) and c > float(last['SAR'])),
    ]
    jaw = float(last['Alligator_Jaw']) if pd.notna(last['Alligator_Jaw']) else 0
    teeth = float(last['Alligator_Teeth']) if pd.notna(last['Alligator_Teeth']) else 0
    lips = float(last['Alligator_Lips']) if pd.notna(last['Alligator_Lips']) else 0
    checks += [
        ('18. السعر > مؤشر التمساح', c > max(jaw, teeth, lips) and max(jaw, teeth, lips) > 0),
        ('19. السعر > خط منتصف البولينجر', pd.notna(last['BB_M']) and c > float(last['BB_M'])),
    ]
    if len(d) >= 5:
        week_high = d['High'].iloc[-6:-1].max()
        checks.append(('20. السعر > قمة الأسبوع السابق', c > float(week_high)))
    else:
        checks.append(('20. السعر > قمة الأسبوع السابق', False))
    for lbl, cond in checks:
        if cond:
            criteria[lbl] = ('✓', 1); score += 1
        else:
            criteria[lbl] = ('✗', 0)
    return criteria, score
    
def gen_signals(d):
    sig = {}
    last = d.iloc[-1]
    c = float(last['Close'])
    for col, lbl in [('SMA20','السعر مقابل SMA 20'),('SMA50','السعر مقابل SMA 50'),('SMA200','السعر مقابل SMA 200')]:
        v = float(last[col]) if pd.notna(last[col]) else None
        if v is not None:
            sig[lbl] = ('صاعد ▲', 1) if c > v else ('هابط ▼', -1)
    for col, lbl in [('EMA20','السعر مقابل EMA 20'),('EMA50','السعر مقابل EMA 50'),('EMA200','السعر مقابل EMA 200')]:
        v = float(last[col]) if pd.notna(last[col]) else None
        if v is not None:
            sig[lbl] = ('صاعد ▲', 1) if c > v else ('هابط ▼', -1)
    sma50 = float(last['SMA50']) if pd.notna(last['SMA50']) else None
    sma200 = float(last['SMA200']) if pd.notna(last['SMA200']) else None
    if sma50 and sma200:
        sig['تقاطع SMA 50 / 200'] = ('تقاطع ذهبي ▲', 1) if sma50 > sma200 else ('تقاطع سلبي ▼', -1)
    ema50 = float(last['EMA50']) if pd.notna(last['EMA50']) else None
    ema200 = float(last['EMA200']) if pd.notna(last['EMA200']) else None
    if ema50 and ema200:
        sig['تقاطع EMA 50 / 200'] = ('تقاطع ذهبي ▲', 1) if ema50 > ema200 else ('تقاطع سلبي ▼', -1)
    rsi = float(last['RSI']) if pd.notna(last['RSI']) else None
    if rsi is not None:
        if rsi > 70:   sig['RSI (14)'] = (f'تشبع شراء ({rsi:.1f})', -1)
        elif rsi < 30: sig['RSI (14)'] = (f'تشبع بيع ({rsi:.1f})', 1)
        else:          sig['RSI (14)'] = (f'محايد ({rsi:.1f})', 0)
    mh = float(last['MACD_H']) if pd.notna(last['MACD_H']) else None
    if mh is not None:
        sig['MACD'] = ('صاعد ▲', 1) if mh > 0 else ('هابط ▼', -1)
    roc12 = float(last['ROC12']) if pd.notna(last['ROC12']) else None
    if roc12 is not None:
        if roc12 > 5:     sig['ROC 12'] = (f'زخم صاعد قوي ({roc12:.2f}%)', 1)
        elif roc12 > 0:   sig['ROC 12'] = (f'زخم صاعد ({roc12:.2f}%)', 1)
        elif roc12 > -5:  sig['ROC 12'] = (f'زخم هابط ({roc12:.2f}%)', -1)
        else:             sig['ROC 12'] = (f'زخم هابط قوي ({roc12:.2f}%)', -1)
    adx = float(last['ADX']) if pd.notna(last['ADX']) else None
    pdi = float(last['PDI']) if pd.notna(last['PDI']) else None
    mdi_v = float(last['MDI']) if pd.notna(last['MDI']) else None
    if all(x is not None for x in [adx, pdi, mdi_v]):
        trend = 'قوي' if adx > 25 else 'ضعيف'
        if pdi > mdi_v: sig['اتجاه ADX'] = (f'اتجاه صاعد {trend} ({adx:.0f})', 1 if adx > 25 else 0)
        else:           sig['اتجاه ADX'] = (f'اتجاه هابط {trend} ({adx:.0f})', -1 if adx > 25 else 0)
    vavg = d['Volume'].rolling(20, min_periods=1).mean().iloc[-1]
    vnow = float(last['Volume'])
    if pd.notna(vavg) and vavg > 0:
        vr = vnow / vavg
        vtxt = f'حجم مرتفع ({vr:.1f}x)' if vr > 1.5 else (f'حجم منخفض ({vr:.1f}x)' if vr < 0.5 else f'حجم طبيعي ({vr:.1f}x)')
        sig['الحجم'] = (vtxt, 0)
    signal_score = sum(val[1] for val in sig.values())
    return sig, signal_score


def recommendation(score):
    if score >= 16: return 'إيجابي +', GREEN_HEX
    elif score >= 12: return 'إيجابي', '#66BB6A'
    elif score <= 4: return 'سلبي +', RED_HEX
    elif score <= 8: return 'سلبي', '#EF5350'
    return 'حياد', ORANGE_HEX


def decision_text(v):
    return 'إيجابي' if v > 0 else 'سلبي' if v < 0 else 'حياد'

# ################الشموع محدثة############################

def detect_candle_patterns(df):
    patterns = []
    if len(df) < 4:
        return patterns

    o = df['Open'].values.astype(float)
    h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float)
    c = df['Close'].values.astype(float)
    dates = df.index

    def body(i):
        return abs(c[i] - o[i])

    def candle(i):
        return h[i] - l[i]

    def upper_shadow(i):
        return h[i] - max(c[i], o[i])

    def lower_shadow(i):
        return min(c[i], o[i]) - l[i]

    def is_bull(i):
        return c[i] > o[i]

    def is_bear(i):
        return c[i] < o[i]

    def mid_body(i):
        return (o[i] + c[i]) / 2.0

    def body_top(i):
        return max(o[i], c[i])

    def body_bot(i):
        return min(o[i], c[i])

    def is_small_body(i, avg):
        return body(i) < avg * 0.3

    def is_large_body(i, avg):
        return body(i) > avg * 0.8

    def in_downtrend(i, lookback=5):
        start = max(0, i - lookback)
        return c[i] < c[start] and sum(1 for j in range(start, i) if c[j] < o[j]) >= lookback * 0.6

    def in_uptrend(i, lookback=5):
        start = max(0, i - lookback)
        return c[i] > c[start] and sum(1 for j in range(start, i) if c[j] > o[j]) >= lookback * 0.6

    for i in range(2, len(df)):
        avg_body = np.mean([body(j) for j in range(max(0, i - 10), i)]) or 1e-9
        date_lbl = dates[i].strftime('%Y-%m-%d')
        cr = candle(i)

        # SINGLE CANDLE -------------------------------------------------
        if body(i) < 0.05 * cr and cr > 0:
            if lower_shadow(i) > 2 * body(i) and upper_shadow(i) < cr * 0.1:
                patterns.append((date_lbl, 'دوجي اليعسوب', 'Dragonfly Doji', True))
                continue
            if upper_shadow(i) > 2 * body(i) and lower_shadow(i) < cr * 0.1:
                patterns.append((date_lbl, 'دوجي شاهد القبر', 'Gravestone Doji', False))
                continue
            if upper_shadow(i) > cr * 0.3 and lower_shadow(i) > cr * 0.3:
                patterns.append((date_lbl, 'دوجي طويل الأرجل', 'Long-Legged Doji', None))
                continue
            patterns.append((date_lbl, 'دوجي', 'Doji', None))
            continue

        if (lower_shadow(i) > 2 * body(i) and
                upper_shadow(i) < 0.3 * body(i) and body(i) > 0):
            if in_downtrend(i):
                patterns.append((date_lbl, 'المطرقة', 'Hammer', True))
            else:
                patterns.append((date_lbl, 'المشنقة', 'Hanging Man', False))
            continue

        if (upper_shadow(i) > 2 * body(i) and
                lower_shadow(i) < 0.3 * body(i) and body(i) > 0):
            if in_downtrend(i):
                patterns.append((date_lbl, 'المطرقة المقلوبة', 'Inverted Hammer', True))
            else:
                patterns.append((date_lbl, 'النجمة الساقطة', 'Shooting Star', False))
            continue

        if (upper_shadow(i) < body(i) * 0.05 and
                lower_shadow(i) < body(i) * 0.05 and
                is_large_body(i, avg_body)):
            if is_bull(i):
                patterns.append((date_lbl, 'ماروبوزو صاعد', 'Bullish Marubozu', True))
            else:
                patterns.append((date_lbl, 'ماروبوزو هابط', 'Bearish Marubozu', False))
            continue

        if (is_small_body(i, avg_body) and
                upper_shadow(i) > body(i) and
                lower_shadow(i) > body(i) and
                cr > avg_body * 0.5):
            patterns.append((date_lbl, 'القمة الدوارة', 'Spinning Top', None))
            continue

        # TWO CANDLES ---------------------------------------------------
        if (is_bear(i - 1) and is_bull(i) and
                o[i] <= c[i - 1] and c[i] >= o[i - 1] and
                body(i) > body(i - 1)):
            patterns.append((date_lbl, 'الابتلاع الصعودي', 'Bullish Engulfing', True))
            continue

        if (is_bull(i - 1) and is_bear(i) and
                o[i] >= c[i - 1] and c[i] <= o[i - 1] and
                body(i) > body(i - 1)):
            patterns.append((date_lbl, 'الابتلاع الهبوطي', 'Bearish Engulfing', False))
            continue

        if (is_bear(i - 1) and is_bull(i) and
                is_large_body(i - 1, avg_body) and
                body_top(i) < body_top(i - 1) and
                body_bot(i) > body_bot(i - 1)):
            patterns.append((date_lbl, 'الحرامي الصعودي', 'Bullish Harami', True))
            continue

        if (is_bull(i - 1) and is_bear(i) and
                is_large_body(i - 1, avg_body) and
                body_top(i) < body_top(i - 1) and
                body_bot(i) > body_bot(i - 1)):
            patterns.append((date_lbl, 'الحرامي الهبوطي', 'Bearish Harami', False))
            continue

        if (is_bear(i - 1) and is_bull(i) and
                abs(l[i] - l[i - 1]) < avg_body * 0.1 and
                in_downtrend(i)):
            patterns.append((date_lbl, 'قاع الملقط', 'Tweezer Bottom', True))
            continue

        if (is_bull(i - 1) and is_bear(i) and
                abs(h[i] - h[i - 1]) < avg_body * 0.1 and
                in_uptrend(i)):
            patterns.append((date_lbl, 'قمة الملقط', 'Tweezer Top', False))
            continue

        if (is_bear(i - 1) and is_bull(i) and
                o[i] < l[i - 1] and
                c[i] > mid_body(i - 1) and c[i] < o[i - 1]):
            patterns.append((date_lbl, 'خط الاختراق', 'Piercing Line', True))
            continue

        if (is_bull(i - 1) and is_bear(i) and
                o[i] > h[i - 1] and
                c[i] < mid_body(i - 1) and c[i] > o[i - 1]):
            patterns.append((date_lbl, 'الغطاء السحابي', 'Dark Cloud Cover', False))
            continue

        if (is_bear(i - 1) and is_bull(i) and
                is_large_body(i - 1, avg_body) and
                o[i] < c[i - 1] and
                abs(c[i] - c[i - 1]) < avg_body * 0.1):
            patterns.append((date_lbl, 'على العنق', 'On-Neck', False))
            continue

        # THREE+ CANDLES -----------------------------------------------
        if i >= 2:
            if (is_bear(i - 2) and
                    is_small_body(i - 1, avg_body) and
                    is_bull(i) and
                    c[i] > mid_body(i - 2) and
                    body_top(i - 1) < body_bot(i - 2)):
                patterns.append((date_lbl, 'نجمة الصباح', 'Morning Star', True))
                continue

            if (is_bull(i - 2) and
                    is_small_body(i - 1, avg_body) and
                    is_bear(i) and
                    c[i] < mid_body(i - 2) and
                    body_bot(i - 1) > body_top(i - 2)):
                patterns.append((date_lbl, 'نجمة المساء', 'Evening Star', False))
                continue

            if (is_bear(i - 2) and
                    body(i - 1) < 0.05 * candle(i - 1) and candle(i - 1) > 0 and
                    is_bull(i) and
                    c[i] > mid_body(i - 2)):
                patterns.append((date_lbl, 'نجمة الصباح دوجي', 'Morning Doji Star', True))
                continue

            if (is_bull(i - 2) and
                    body(i - 1) < 0.05 * candle(i - 1) and candle(i - 1) > 0 and
                    is_bear(i) and
                    c[i] < mid_body(i - 2)):
                patterns.append((date_lbl, 'نجمة المساء دوجي', 'Evening Doji Star', False))
                continue

            if (all(c[j] > o[j] for j in [i - 2, i - 1, i]) and
                    c[i - 1] > c[i - 2] and c[i] > c[i - 1] and
                    all(is_large_body(j, avg_body) for j in [i - 2, i - 1, i]) and
                    o[i - 1] > o[i - 2] and o[i] > o[i - 1]):
                patterns.append((date_lbl, 'ثلاثة جنود بيض', 'Three White Soldiers', True))
                continue

            if (all(c[j] < o[j] for j in [i - 2, i - 1, i]) and
                    c[i - 1] < c[i - 2] and c[i] < c[i - 1] and
                    all(is_large_body(j, avg_body) for j in [i - 2, i - 1, i]) and
                    o[i - 1] < o[i - 2] and o[i] < o[i - 1]):
                patterns.append((date_lbl, 'ثلاثة غربان سوداء', 'Three Black Crows', False))
                continue

            if (is_bear(i - 2) and is_bull(i - 1) and is_bull(i) and
                    body_top(i - 1) < body_top(i - 2) and
                    body_bot(i - 1) > body_bot(i - 2) and
                    c[i] > body_top(i - 2)):
                patterns.append((date_lbl, 'ثلاثة من الداخل صاعد', 'Three Inside Up', True))
                continue

            if (is_bull(i - 2) and is_bear(i - 1) and is_bear(i) and
                    body_top(i - 1) < body_top(i - 2) and
                    body_bot(i - 1) > body_bot(i - 2) and
                    c[i] < body_bot(i - 2)):
                patterns.append((date_lbl, 'ثلاثة من الداخل هابط', 'Three Inside Down', False))
                continue

            if (is_bear(i - 2) and is_bull(i - 1) and is_bull(i) and
                    body(i - 1) > body(i - 2) and
                    o[i - 1] <= c[i - 2] and c[i - 1] >= o[i - 2] and
                    c[i] > c[i - 1]):
                patterns.append((date_lbl, 'ثلاثة من الخارج صاعد', 'Three Outside Up', True))
                continue

            if (is_bull(i - 2) and is_bear(i - 1) and is_bear(i) and
                    body(i - 1) > body(i - 2) and
                    o[i - 1] >= c[i - 2] and c[i - 1] <= o[i - 2] and
                    c[i] < c[i - 1]):
                patterns.append((date_lbl, 'ثلاثة من الخارج هابط', 'Three Outside Down', False))
                continue

            if (is_bear(i - 2) and
                    body(i - 1) < 0.05 * candle(i - 1) and candle(i - 1) > 0 and
                    is_bull(i) and
                    h[i - 1] < l[i - 2] and h[i - 1] < l[i]):
                patterns.append((date_lbl, 'الطفل المتروك صاعد', 'Abandoned Baby Bull', True))
                continue

            if (is_bull(i - 2) and
                    body(i - 1) < 0.05 * candle(i - 1) and candle(i - 1) > 0 and
                    is_bear(i) and
                    l[i - 1] > h[i - 2] and l[i - 1] > h[i]):
                patterns.append((date_lbl, 'الطفل المتروك هابط', 'Abandoned Baby Bear', False))
                continue

            if i >= 4:
                if (is_bull(i - 4) and is_large_body(i - 4, avg_body) and
                        all(c[j] < o[j] and body(j) < body(i - 4) for j in [i - 3, i - 2, i - 1]) and
                        all(l[j] > l[i - 4] for j in [i - 3, i - 2, i - 1]) and
                        is_bull(i) and c[i] > c[i - 4] and is_large_body(i, avg_body)):
                    patterns.append((date_lbl, 'ثلاث طرق صاعدة', 'Rising Three Methods', True))
                    continue

                if (is_bear(i - 4) and is_large_body(i - 4, avg_body) and
                        all(c[j] > o[j] and body(j) < body(i - 4) for j in [i - 3, i - 2, i - 1]) and
                        all(h[j] < h[i - 4] for j in [i - 3, i - 2, i - 1]) and
                        is_bear(i) and c[i] < c[i - 4] and is_large_body(i, avg_body)):
                    patterns.append((date_lbl, 'ثلاث طرق هابطة', 'Falling Three Methods', False))
                    continue

    seen = set()
    unique = []
    for p in reversed(patterns):
        if p[2] not in seen:
            seen.add(p[2])
            unique.append(p)
        if len(unique) == 8:
            break
    return list(reversed(unique))
    
# ---------------------------------------------------------------------
# CHART WITH GUARANTEED NON‑CROSSING SMOOTH CONNECTORS
# ---------------------------------------------------------------------
def make_candle_pattern_chart(df, patterns):
    """
    Candlestick chart with pattern annotations.
    Straight connectors routed above the candles (no crossing).
    """
    from matplotlib.path import Path
    import matplotlib.patches as mpatches

    df = df.tail(60).copy()
    df_plot = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    mc = mpf.make_marketcolors(
        up='#26a69a', down='#ef5350', edge='inherit',
        wick='inherit', volume={'up': '#80cbc4', 'down': '#ef9a9a'}
    )
    st = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=':', gridcolor='#dddddd',
        rc={'axes.facecolor': '#FAFAFA'}
    )

    fig, axlist = mpf.plot(
        df_plot, type='candle', style=st, volume=False,
        figsize=(16, 7), returnfig=True, warn_too_much_data=10_000
    )
    ax = axlist[0]
    date_to_pos = {pd.Timestamp(d).date(): i for i, d in enumerate(df.index)}

    # ── Resolve patterns ─────────────────────────────────────────────
    resolved = []
    for date_lbl, ar_name, en_name, bullish in patterns:
        pos = date_to_pos.get(pd.to_datetime(date_lbl).date())
        if pos is None:
            continue
        row = df.iloc[pos]
        conn_y = float(row['High']) if (bullish is True or bullish is None) \
                 else float(row['Low'])
        resolved.append(dict(
            pos=float(pos), high=float(row['High']), low=float(row['Low']),
            conn_y=conn_y, name=ar_name, bullish=bullish,
        ))

    if not resolved:
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    # ── Chart geometry ─────────────────────────────────────────────
    ylo, yhi = ax.get_ylim()
    xlo, xhi = ax.get_xlim()
    xspan = xhi - xlo
    yspan = yhi - ylo
    xmid  = (xlo + xhi) / 2.0

    # Escape Y — sits above every candle
    escape_y = yhi + yspan * 0.03

    # ── Split into LEFT / RIGHT ─────────────────────────────────────
    left_anns  = [a for a in resolved if a['pos'] <  xmid]
    right_anns = [a for a in resolved if a['pos'] >= xmid]

    if not left_anns and len(right_anns) > 1:
        right_anns.sort(key=lambda a: a['pos'])
        left_anns.append(right_anns.pop(0))
    elif not right_anns and len(left_anns) > 1:
        left_anns.sort(key=lambda a: a['pos'], reverse=True)
        right_anns.append(left_anns.pop(0))

    # Sort by conn_y descending so labels flow top-to-bottom
    left_anns.sort(key=lambda  a: a['conn_y'], reverse=True)
    right_anns.sort(key=lambda a: a['conn_y'], reverse=True)

    # ── Label X positions ──────────────────────────────────────────
    L_LABEL_X = xlo - xspan * 0.22
    R_LABEL_X = xhi + xspan * 0.22

    # ── Vertical label slots ───────────────────────────────────────
    v_pad = yspan * 0.05
    y_top = yhi + v_pad
    y_bot = ylo - v_pad

    def _vslots(n):
        if n == 0:
            return []
        if n == 1:
            return [(y_top + y_bot) / 2.0]
        return np.linspace(y_top, y_bot, n).tolist()

    l_ys = _vslots(len(left_anns))
    r_ys = _vslots(len(right_anns))

    # ── Draw helper ─────────────────────────────────────────────────
    def _draw(ann, label_x, label_y, side):
        px, cy = ann['pos'], ann['conn_y']
        name, bull = ann['name'], ann['bullish']

        color = ('#1B5E20' if bull is True
                 else '#B71C1C' if bull is False
                 else '#E65100')

        # Build L-shaped path: candle → up to escape_y → to label
        if side == 'left':
            verts = [
                (px, cy),
                (px, escape_y),
                (label_x, label_y),
            ]
        else:
            verts = [
                (px, cy),
                (px, escape_y),
                (label_x, label_y),
            ]
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO]

        patch = mpatches.PathPatch(
            Path(verts, codes),
            facecolor='none', edgecolor=color,
            linewidth=1.4, clip_on=False, zorder=5, alpha=0.85,
        )
        ax.add_patch(patch)

        # Small dot on the candle
        ax.plot(px, cy, 'o', color=color, markersize=5,
                clip_on=False, zorder=6,
                markeredgecolor='white', markeredgewidth=0.5)

        # Label
        ha = 'right' if side == 'left' else 'left'
        ax.text(label_x, label_y, name, fontsize=9, color=color,
                ha=ha, va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=color, alpha=0.95, linewidth=1.0),
                clip_on=False, zorder=10)

    # ── Draw all ───────────────────────────────────────────────────
    for ann, ly in zip(left_anns, l_ys):
        _draw(ann, L_LABEL_X, ly, 'left')
    for ann, ry in zip(right_anns, r_ys):
        _draw(ann, R_LABEL_X, ry, 'right')

    # ── Widen axes ────────────────────────────────────────────────
    ax.set_xlim(xlo - xspan * 0.30, xhi + xspan * 0.30)

    # ── Save ───────────────────────────────────────────────────────
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()
    
###################################نهاية الشموع
def detect_divergences(d):
    divergences = []
    df = d.tail(60).copy()
    if len(df) < 20: return divergences
    c = df['Close'].values; rsi = df['RSI'].values; mcd = df['MACD'].values
    def find_local_min_idx(arr, n=5):
        return [i for i in range(n, len(arr)-n) if arr[i] == min(arr[i-n:i+n+1])]
    def find_local_max_idx(arr, n=5):
        return [i for i in range(n, len(arr)-n) if arr[i] == max(arr[i-n:i+n+1])]
    if not (np.all(np.isnan(rsi)) or np.sum(~np.isnan(rsi)) < 15):
        price_lows = find_local_min_idx(c); rsi_lows = find_local_min_idx(rsi)
        price_highs = find_local_max_idx(c); rsi_highs = find_local_max_idx(rsi)
        if len(price_lows)>=2 and len(rsi_lows)>=2:
            p1,p2 = price_lows[-2],price_lows[-1]; r1,r2 = rsi_lows[-2],rsi_lows[-1]
            if c[p2]<c[p1] and rsi[r2]>rsi[r1]:
                divergences.append(('RSI','تباعد إيجابي (صعودي)','Bullish Divergence'))
        if len(price_highs)>=2 and len(rsi_highs)>=2:
            p1,p2 = price_highs[-2],price_highs[-1]; r1,r2 = rsi_highs[-2],rsi_highs[-1]
            if c[p2]>c[p1] and rsi[r2]<rsi[r1]:
                divergences.append(('RSI','تباعد سلبي (هبوطي)','Bearish Divergence'))
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
    
def find_sr(df, d_ind=None, n_levels=8):
    h_arr = df['High'].values.astype(float); l_arr = df['Low'].values.astype(float)
    c_arr = df['Close'].values.astype(float); v_arr = df['Volume'].values.astype(float)
    cur = float(c_arr[-1])
    raw_sup = []; raw_res = []
    for order in (3,5,7,10):
        w = 1.0 + order*0.15
        ph = argrelextrema(h_arr, np.greater_equal, order=order)[0]
        pl = argrelextrema(l_arr, np.less_equal, order=order)[0]
        for i in ph: raw_res.append((round(h_arr[i],4), w))
        for i in pl: raw_sup.append((round(l_arr[i],4), w))
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
    vol_avg = np.mean(v_arr) if len(v_arr) > 0 else 1.0
    for i in range(len(v_arr)):
        if v_arr[i] > 1.5*vol_avg:
            price = round(c_arr[i],4); bar_mid = (h_arr[i]+l_arr[i])/2
            if price >= bar_mid: raw_res.append((price,1.0))
            else:                raw_sup.append((price,1.0))
    if d_ind is not None:
        for ema_col, ema_w in [('EMA20',2.0),('EMA50',2.5),('EMA100',2.5),('EMA200',3.0)]:
            if ema_col in d_ind.columns:
                val = d_ind[ema_col].iloc[-1]
                if pd.notna(val):
                    ev = round(float(val),4)
                    if ev < cur: raw_sup.append((ev,ema_w))
                    elif ev > cur: raw_res.append((ev,ema_w))
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
    
def gen_technical_review(d, sig, score, sup, res, info=None, patterns=None, divergences=None):
    last = d.iloc[-1]
    c_price = float(last['Close'])
    sections = []

    ema7   = float(last['EMA7'])   if pd.notna(last['EMA7'])   else None
    ema20  = float(last['EMA20'])  if pd.notna(last['EMA20'])  else None
    ema50  = float(last['EMA50'])  if pd.notna(last['EMA50'])  else None
    ema200 = float(last['EMA200']) if pd.notna(last['EMA200']) else None
    sma50  = float(last['SMA50'])  if pd.notna(last['SMA50'])  else None
    sma200 = float(last['SMA200']) if pd.notna(last['SMA200']) else None
    adx    = float(last['ADX'])    if pd.notna(last['ADX'])    else None
    pdi    = float(last['PDI'])    if pd.notna(last['PDI'])    else None
    mdi    = float(last['MDI'])    if pd.notna(last['MDI'])    else None

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

    rsi      = float(last['RSI'])      if pd.notna(last['RSI'])      else None
    macd     = float(last['MACD'])     if pd.notna(last['MACD'])     else None
    macd_sig = float(last['MACD_Sig']) if pd.notna(last['MACD_Sig']) else None
    macd_h   = float(last['MACD_H'])   if pd.notna(last['MACD_H'])   else None
    roc12    = float(last['ROC12'])    if pd.notna(last['ROC12'])    else None

    mom_parts = []
    if rsi is not None:
        if rsi >= 70:
            mom_parts.append(f'مؤشر القوة النسبية RSI يقرأ {rsi:.1f} في منطقة التشبع الشرائي، مما يستوجب الحذر من تصحيح وشيك.')
        elif rsi <= 30:
            mom_parts.append(f'مؤشر RSI عند {rsi:.1f} في منطقة التشبع البيعي، مما يستدعي مراقبة إشارات الانعكاس المحتملة.')
        elif rsi >= 55:
            mom_parts.append(f'مؤشر RSI عند {rsi:.1f} في المنطقة الإيجابية فوق خط 50، مما يعكس زخماً شرائياً متواصلاً.')
        elif rsi <= 45:
            mom_parts.append(f'مؤشر RSI عند {rsi:.1f} في المنطقة السلبية دون خط 50، مما يعكس ضعفاً في الزخم الشرائي.')
        else:
            mom_parts.append(f'مؤشر RSI محايد عند {rsi:.1f} قريباً من الخط 50.')
    if macd is not None and macd_sig is not None and macd_h is not None:
        if macd > macd_sig and macd_h > 0:
            mom_parts.append('مؤشر الماكد MACD يتداول فوق خط الإشارة مع هستوجرام إيجابي، مما يدعم الزخم الصعودي.')
        elif macd < macd_sig and macd_h < 0:
            mom_parts.append('مؤشر الماكد يتداول دون خط الإشارة مع هستوجرام سلبي، مما يعكس ضعف الزخم الحالي.')
        else:
            mom_parts.append('مؤشر الماكد في مرحلة تقاطع، مما قد يُنذر بتغيير في الاتجاه.')
    if roc12 is not None:
        if roc12 > 5:
            mom_parts.append(f'مؤشر ROC 12 عند {roc12:.2f}% يُشير إلى زخم صاعد قوي على المدى المتوسط.')
        elif roc12 > 0:
            mom_parts.append(f'مؤشر ROC 12 عند {roc12:.2f}% إيجابي، مما يدل على استمرار الزخم الشرائي.')
        elif roc12 > -5:
            mom_parts.append(f'مؤشر ROC 12 عند {roc12:.2f}% سلبي، مما يعكس ضعفاً في الزخم الحالي.')
        else:
            mom_parts.append(f'مؤشر ROC 12 عند {roc12:.2f}% يُشير إلى زخم هبوطي قوي على المدى المتوسط.')
    sections.append(('مؤشرات الزخم', ' '.join(mom_parts) if mom_parts else 'لا تتوفر بيانات كافية.'))

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
        if atr_pct > 3:
            vol_parts.append(f'مؤشر ATR يُسجّل {atr:.2f} ({atr_pct:.1f}% من السعر)، مما يُشير إلى تذبذب يومي مرتفع.')
        elif atr_pct < 1:
            vol_parts.append(f'مؤشر ATR عند {atr:.2f} ({atr_pct:.1f}% من السعر)، مما يعكس هدوءاً نسبياً في حركة السعر.')
        else:
            vol_parts.append(f'مؤشر ATR عند {atr:.2f} ({atr_pct:.1f}% من السعر) يُشير إلى تذبذب يومي معتدل.')
    sections.append(('التذبذب ونطاق بولنجر', ' '.join(vol_parts) if vol_parts else 'لا تتوفر بيانات كافية.'))

    obv      = float(last['OBV'])         if pd.notna(last['OBV'])         else None
    bull_vol = float(last['Bull_Volume']) if pd.notna(last['Bull_Volume']) else None
    bear_vol = float(last['Bear_Volume']) if pd.notna(last['Bear_Volume']) else None
    vol_now  = float(last['Volume'])
    vwap     = float(last['VWAP'])        if pd.notna(last['VWAP'])        else None
    vol_avg  = d['Volume'].rolling(20, min_periods=1).mean().iloc[-1]

    volume_parts = []
    if pd.notna(vol_avg) and vol_avg > 0:
        vr = vol_now / vol_avg
        if vr > 1.5:
            volume_parts.append(f'حجم التداول الحالي مرتفع بنسبة {vr:.1f}x فوق متوسط الـ20 يوماً، مما يعكس اهتماماً كبيراً من المشاركين في السوق.')
        elif vr < 0.5:
            volume_parts.append(f'حجم التداول منخفض ({vr:.1f}x من المتوسط)، مما يُشير إلى تراجع في النشاط التداولي وانخفاض الاهتمام.')
        else:
            volume_parts.append(f'حجم التداول طبيعي عند {vr:.1f}x من المتوسط الـ20 يوماً.')
    if bull_vol is not None and bear_vol is not None and (bull_vol + bear_vol) > 0:
        bull_pct = bull_vol / (bull_vol + bear_vol) * 100
        if bull_pct > 55:
            volume_parts.append(f'السيولة الشرائية تهيمن بنسبة {bull_pct:.0f}% من إجمالي الحجم الأخير، مؤشر إيجابي على الطلب الفعّال.')
        elif bull_pct < 45:
            volume_parts.append(f'السيولة على الجانب الهابط تهيمن بنسبة {100-bull_pct:.0f}%، مما يعكس ضغطاً على الورقة المالية.')
        else:
            volume_parts.append(f'توازن نسبي بين السيولة الشرائية ({bull_pct:.0f}%) والبيعية ({100-bull_pct:.0f}%).')
    if vwap is not None:
        if c_price > vwap:
            volume_parts.append(f'مؤشر VWAP يُسجَّل عند {vwap:.2f} والسعر يتداول فوقه، مما يُشير إلى هيمنة المشترين خلال جلسات التداول.')
        else:
            volume_parts.append(f'مؤشر VWAP يُسجَّل عند {vwap:.2f} والسعر يتداول دونه، مما يُشير إلى هيمنة البائعين خلال جلسات التداول.')
    if len(d) >= 5 and obv is not None:
        obv_prev = float(d.iloc[-5]['OBV']) if pd.notna(d.iloc[-5]['OBV']) else None
        if obv_prev is not None:
            if obv > obv_prev:
                volume_parts.append('OBV: المؤشر في اتجاه تصاعدي خلال الأسبوع الأخير، مما يدعم صحة الحركة الصعودية.')
            else:
                volume_parts.append('OBV: المؤشر في اتجاه تراجعي خلال الأسبوع الأخير، مما يُلمح إلى ضعف خفي رغم الحركة السعرية.')
    sections.append(('تحليل الحجم', ' '.join(volume_parts) if volume_parts else 'لا تتوفر بيانات كافية.'))

    tenkan = float(last['Tenkan'])   if pd.notna(last['Tenkan'])   else None
    kijun  = float(last['Kijun'])    if pd.notna(last['Kijun'])    else None
    senk_a = float(last['Senkou_A']) if pd.notna(last['Senkou_A']) else None
    senk_b = float(last['Senkou_B']) if pd.notna(last['Senkou_B']) else None
    sar    = float(last['SAR'])      if pd.notna(last['SAR'])      else None

    adv_parts = []
    if tenkan is not None and kijun is not None:
        if tenkan > kijun:
            adv_parts.append(f'خط التنكن ({tenkan:.2f}) أعلى من خط الكيجن ({kijun:.2f})، إشارة إيجابية في نظام الإيشيموكو.')
        else:
            adv_parts.append(f'خط التنكن ({tenkan:.2f}) أدنى من خط الكيجن ({kijun:.2f})، إشارة سلبية في نظام الإيشيموكو.')
    if senk_a is not None and senk_b is not None:
        kumo_top = max(senk_a, senk_b)
        kumo_bot = min(senk_a, senk_b)
        if c_price > kumo_top:
            adv_parts.append(f'السعر يتداول فوق السحابة السميكة (كوموه) عند {kumo_top:.2f}، مما يُعزز الاتجاه الصعودي.')
        elif c_price < kumo_bot:
            adv_parts.append(f'السعر دون السحابة (كوموه) عند {kumo_bot:.2f}، مما يُعزز الاتجاه الهبوطي.')
        else:
            adv_parts.append('السعر داخل السحابة (كوموه)، مما يدل على مرحلة تردد وعدم حسم في الاتجاه.')
    if sar is not None:
        if c_price > sar:
            adv_parts.append(f'مؤشر بارابوليك SAR عند {sar:.2f} يقع دون السعر الحالي، مما يدعم استمرار الاتجاه الصعودي.')
        else:
            adv_parts.append(f'مؤشر بارابوليك SAR عند {sar:.2f} يعلو السعر الحالي، مما يشير إلى ضعف الزخم الصعودي.')
    sections.append(('مؤشرات متقدمة (إيشيموكو / SAR)', ' '.join(adv_parts) if adv_parts else 'لا تتوفر بيانات كافية.'))

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
    v_arr   = d['Volume'].values.astype(float)
    vol_avg2 = np.mean(v_arr) if len(v_arr) > 0 else 1
    hv_closes = [float(d.iloc[i]['Close']) for i in np.where(v_arr > 1.5 * vol_avg2)[0]]
    hv_res = sorted([p for p in hv_closes if p > c_price])
    hv_sup = sorted([p for p in hv_closes if p <= c_price], reverse=True)
    if hv_sup:
        sr_parts.append(f'دعم حجمي بارز عند {hv_sup[0]:.2f} — ناتج عن تداولات كثيفة سابقة.')
    if hv_res:
        sr_parts.append(f'مقاومة حجمية بارزة عند {hv_res[0]:.2f} — ناتجة عن تداولات كثيفة سابقة.')
    if atr_val is not None:
        t1_up = c_price + 1.0 * atr_val; t2_up = c_price + 2.0 * atr_val
        t1_dn = c_price - 1.0 * atr_val; t2_dn = c_price - 2.0 * atr_val
        sr_parts.append(f'ATR ({atr_val:.2f}): الأهداف الصعودية {t1_up:.2f} و {t2_up:.2f}. مستويات التوقف {t1_dn:.2f} و {t2_dn:.2f}.')
    nearest_s = sup[0] if sup and sup[0] < c_price else None
    nearest_r = res[0] if res and res[0] > c_price else None
    if nearest_s and nearest_r and (c_price - nearest_s) > 0:
        rr = (nearest_r - c_price) / (c_price - nearest_s)
        sr_parts.append(f'نسبة العائد إلى المخاطرة (R:R): {rr:.1f}x بين أقرب دعم ومقاومة.')
    if not sr_parts:
        sr_parts.append('لم يتم رصد مستويات دعم أو مقاومة واضحة في الفترة المحللة.')
    sections.append(('مستويات الدعم والمقاومة والأهداف السعرية', ' '.join(sr_parts)))

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

    if divergences:
        div_parts = [f'{ind}: {ar_type} — يُشير إلى احتمال تغيير في الزخم الحالي.' for ind, ar_type, en_type in divergences]
        sections.append(('التباعد بين السعر والمؤشرات', ' '.join(div_parts)))
    else:
        sections.append(('التباعد بين السعر والمؤشرات',
                          'لا يُلاحَظ تباعد واضح بين السعر ومؤشري RSI وMACD في الفترة الأخيرة، مما يُشير إلى انسجام الزخم مع حركة السعر.'))

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

    cci_val   = float(last['CCI'])   if pd.notna(last['CCI'])   else None
    willr     = float(last['WILLR']) if pd.notna(last['WILLR']) else None
    osc_parts = []
    if cci_val is not None:
        if cci_val > 100:
            osc_parts.append(f'CCI: القراءة {cci_val:.0f} فوق مستوى +100 تُشير إلى زخم صعودي قوي مع احتمال تشبع.')
        elif cci_val < -100:
            osc_parts.append(f'CCI: القراءة {cci_val:.0f} دون مستوى -100 تُشير إلى زخم هبوطي مع احتمال تشبع بيعي.')
        else:
            osc_parts.append(f'CCI: القراءة {cci_val:.0f} داخل النطاق المحايد بين -100 و+100.')
    if willr is not None:
        if willr > -20:
            osc_parts.append(f'Williams %R: القراءة {willr:.0f} قريبة من منطقة التشبع الشرائي (فوق -20).')
        elif willr < -80:
            osc_parts.append(f'Williams %R: القراءة {willr:.0f} في منطقة التشبع البيعي (تحت -80).')
        else:
            osc_parts.append(f'Williams %R: القراءة {willr:.0f} في المنطقة المحايدة.')
    if osc_parts:
        sections.append(('مؤشرات CCI وWilliams %R', ' '.join(osc_parts)))

    return sections


def _draw_sr_lines(ax, sup, res, xmax, d_ind=None, pivots=None):
    sup_color = '#1565C0'
    res_color = '#B71C1C'
    ema_colors = {
        'EMA20':  ('#2196F3', 'EMA 20'),
        'EMA50':  ('#E53935', 'EMA 50'),
        'EMA100': ('#C620F8', 'EMA 100'),
        'EMA200': ('#000000', 'EMA 200'),
    }

    # ── Draw pivot markers ──
    if pivots:
        for bx, by in pivots.get('highs', []):
            ax.plot(bx, by, 'v', color='#B71C1C', markersize=5,
                    alpha=0.75, zorder=6, markeredgewidth=0)
        for bx, by in pivots.get('lows', []):
            ax.plot(bx, by, '^', color='#1565C0', markersize=5,
                    alpha=0.75, zorder=6, markeredgewidth=0)

    # ── Collect ALL labels ──
    items = []

    if d_ind is not None:
        for col, (clr, lbl) in ema_colors.items():
            if col in d_ind.columns:
                val = d_ind[col].iloc[-1]
                if pd.notna(val):
                    ev = float(val)
                    ax.axhline(ev, color=clr, lw=0.9, ls='-.',
                               alpha=0.55, zorder=8)
                    items.append((ev, lbl, clr, 'white', clr, 6.5))

    for i, s in enumerate(sup):
        ax.axhline(s, color=sup_color, lw=0.7, ls='--',
                   alpha=0.70, zorder=8)
        items.append((s, rtl(f'دعم {i+1}   {s:.2f}'),
                       sup_color, '#E8F4FD', sup_color, 6.5))

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

    # ── Evenly distribute label y-positions ──
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

    # ── Convert label y from data to axes fraction ──
    ylo, yhi = ax.get_ylim()
    y_span = (yhi - ylo) if (yhi - ylo) != 0 else 1.0

    def to_y_frac(y_val):
        return (y_val - ylo) / y_span

    # ── Label x in axes fraction: outside the right border ──
    # 1.0 = right edge of axes. >1.0 = in the existing margin space.
    # No chart resize, no xlim change — labels just overflow into
    # the empty area that already exists on the right side.
    near_frac = 1.10 #1.03 /موقع
    far_frac  = 1.10 #1.03 /موقع

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

    # ── Draw annotations ──
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

########################################################################################################

def _get_pivots(d, order=5):
    d=d.tail(180).copy().reset_index(drop=True)
    h=d['High'].values.astype(float); l=d['Low'].values.astype(float)
    ph=argrelextrema(h,np.greater_equal,order=order)[0]; pl=argrelextrema(l,np.less_equal,order=order)[0]
    return {'highs':[(int(i),round(h[i],4)) for i in ph],'lows':[(int(i),round(l[i],4)) for i in pl]}
    
def make_main_chart(d, sup=None, res=None):
    sup=sup or []; res=res or []; d=d.tail(180).copy()
    p=d[['Open','High','Low','Close','Volume']].copy()
    mc=mpf.make_marketcolors(up='#26a69a',down='#ef5350',edge='inherit',wick='inherit',volume={'up':'#80cbc4','down':'#ef9a9a'})
    st=mpf.make_mpf_style(marketcolors=mc,gridstyle=':',gridcolor='#dddddd',rc={'axes.facecolor':'#FAFAFA'})
    plot_kwargs=dict(type='candle',style=st,volume=True,figsize=(14,7),returnfig=True,warn_too_much_data=9999)
    fig,ax=mpf.plot(p,**plot_kwargs); main_ax=ax[0]
    xmax=len(d); pivots=_get_pivots(d,order=5)
    _draw_sr_lines(main_ax,sup,res,xmax,d_ind=d,pivots=pivots)
    fig.subplots_adjust(right=0.80); return chart_bytes(fig)
    
def make_tech_chart(d):
    d=d.tail(180).copy(); fig,(a1,a2,a3)=plt.subplots(3,1,figsize=(14,8.5),sharex=True); x=d.index
    r=d['RSI']; a1.plot(x,r,color='#7C4DFF',lw=1.5)
    a1.axhline(70,color=RED_HEX,ls='--',lw=.8,alpha=.7); a1.axhline(50,color='gray',ls='--',lw=.8,alpha=.7); a1.axhline(30,color='#43A047',ls='--',lw=.8,alpha=.7)
    a1.fill_between(x,r,70,where=(r>70),alpha=.25,color=RED_HEX); a1.fill_between(x,r,30,where=(r<30),alpha=.25,color='#43A047')
    a1.set_ylabel('RSI',fontproperties=MPL_FONT_PROP_BOLD); a1.set_ylim(0,100); a1.grid(True,alpha=.3)
    mh=d['MACD_H']; cols=['#26a69a' if v>=0 else '#ef5350' for v in mh.fillna(0)]
    a2.bar(x,mh,color=cols,width=.8,alpha=.6); a2.plot(x,d['MACD'],color=BLUE_HEX,lw=1.3,label='MACD')
    a2.plot(x,d['MACD_Sig'],color=ORANGE_HEX,lw=1.3,label=rtl('الإشارة')); a2.axhline(0,color='gray',lw=.5)
    a2.set_ylabel('MACD',fontproperties=MPL_FONT_PROP_BOLD); a2.legend(prop=MPL_FONT_PROP,fontsize=8); a2.grid(True,alpha=.3)
    a3.plot(x,d['ROC12'],color='#00897B',lw=1.3,label=rtl('ROC 12')); a3.axhline(0,color='gray',ls='--',lw=0.8,alpha=0.7)
    a3.fill_between(x,d['ROC12'],0,where=(d['ROC12']>0),alpha=0.2,color='#43A047'); a3.fill_between(x,d['ROC12'],0,where=(d['ROC12']<0),alpha=0.2,color=RED_HEX)
    a3.set_ylabel(rtl('ROC 12'),fontproperties=MPL_FONT_PROP_BOLD); a3.legend(prop=MPL_FONT_PROP,fontsize=8); a3.grid(True,alpha=.3); a3.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    plt.tight_layout(); return chart_bytes(fig)
    
def make_bb_chart(d):
    d=d.tail(180).copy(); p=d[['Open','High','Low','Close','Volume']].copy()
    aps=[]
    for col,clr,lbl in [('BB_U',RED_HEX,rtl('الحد العلوي')),('BB_M',ORANGE_HEX,rtl('المتوسط')),('BB_L','#43A047',rtl('الحد السفلي'))]:
        aps.append(mpf.make_addplot(d[col],color=clr,width=1.2,label=lbl))
    mc=mpf.make_marketcolors(up='#26a69a',down='#ef5350',edge='inherit',wick='inherit',volume={'up':'#80cbc4','down':'#ef9a9a'})
    st=mpf.make_mpf_style(marketcolors=mc,gridstyle=':',gridcolor='#dddddd',rc={'axes.facecolor':'#FAFAFA'})
    fig,ax=mpf.plot(p,type='candle',style=st,volume=True,figsize=(14,7),returnfig=True,warn_too_much_data=9999,addplot=aps)
    ax[0].legend(loc='upper left',fontsize=8,prop=MPL_FONT_PROP); return chart_bytes(fig)
    

def make_dd_chart(dd):
    fig,ax=plt.subplots(figsize=(14,3.4))
    ax.fill_between(dd.index,dd.values*100,0,color=RED_HEX,alpha=.35); ax.plot(dd.index,dd.values*100,color='#B71C1C',lw=1)
    ax.set_ylabel(rtl('التراجع %'),fontproperties=MPL_FONT_PROP_BOLD); ax.grid(True,alpha=.3); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    plt.tight_layout(); return chart_bytes(fig)


def make_volume_chart(d):
    d=d.tail(180).copy(); fig,ax=plt.subplots(figsize=(14,3.6)); x=d.index; vol=d['Volume']; avg=vol.rolling(20,min_periods=1).mean()
    cols=['#26a69a' if d['Close'].iloc[i]>=d['Open'].iloc[i] else '#ef5350' for i in range(len(d))]
    ax.bar(x,vol,color=cols,alpha=.5,width=.8); ax.plot(x,avg,color='#1565C0',lw=1.5,label=rtl('متوسط 20 يوم'))
    ax.set_ylabel(rtl('الحجم'),fontproperties=MPL_FONT_PROP_BOLD); ax.legend(prop=MPL_FONT_PROP,fontsize=8); ax.grid(True,alpha=.3); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    plt.tight_layout(); return chart_bytes(fig)


def make_gauge_chart(score):
    fig,(ax_gauge,ax_button)=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[3,1]})
    ax_gauge.set_xlim(-1.6,1.6); ax_gauge.set_ylim(-0.45,1.45); ax_gauge.set_aspect('equal'); ax_gauge.axis('off')
    R_OUTER=1.00; R_INNER=0.55; R_MID=(R_OUTER+R_INNER)/2; R_TICK=R_OUTER+0.13; N_PTS=80
    segments=[(0,4,'#C62828',rtl('سلبي +')),(4,8,'#EF5350',rtl('سلبي')),(8,12,'#FFA726',rtl('حياد')),(12,16,'#66BB6A',rtl('إيجابي')),(16,20,'#2E7D32',rtl('إيجابي +'))]
    for low,high,color,label in segments:
        t_start=np.deg2rad(180-(low/20)*180); t_end=np.deg2rad(180-(high/20)*180); theta=np.linspace(t_start,t_end,N_PTS)
        xo,yo=R_OUTER*np.cos(theta),R_OUTER*np.sin(theta); xi,yi=R_INNER*np.cos(theta),R_INNER*np.sin(theta)
        vx=np.concatenate([xo,xi[::-1]]); vy=np.concatenate([yo,yi[::-1]]); ax_gauge.fill(vx,vy,color=color,alpha=0.90,zorder=2)
        for t_edge in [t_start,t_end]: ax_gauge.plot([R_INNER*np.cos(t_edge),R_OUTER*np.cos(t_edge)],[R_INNER*np.sin(t_edge),R_OUTER*np.sin(t_edge)],color='white',lw=1.5,zorder=3)
        t_mid=(t_start+t_end)/2; lx=R_MID*np.cos(t_mid); ly=R_MID*np.sin(t_mid)
        rotation=np.rad2deg(t_mid)
        if rotation>90: rotation-=180
        ax_gauge.text(lx,ly,label,ha='center',va='center',fontsize=20,fontproperties=MPL_FONT_PROP,color='white',fontweight='bold',rotation=rotation,zorder=4)
    theta_all=np.linspace(0,np.pi,200)
    xo_all=(R_OUTER+0.03)*np.cos(theta_all); yo_all=(R_OUTER+0.03)*np.sin(theta_all)
    xi_all=R_OUTER*np.cos(theta_all); yi_all=R_OUTER*np.sin(theta_all)
    vx_rim=np.concatenate([xo_all,xi_all[::-1]]); vy_rim=np.concatenate([yo_all,yi_all[::-1]]); ax_gauge.fill(vx_rim,vy_rim,color='#37474F',alpha=0.85,zorder=1)
    for n in range(1,21):
        t=np.deg2rad(180-(n/20)*180); ax_gauge.plot([R_OUTER*np.cos(t),(R_OUTER+0.04)*np.cos(t)],[R_OUTER*np.sin(t),(R_OUTER+0.04)*np.sin(t)],color='#37474F',lw=1.0,zorder=5)
        tx=R_TICK*np.cos(t); ty=R_TICK*np.sin(t); ax_gauge.text(tx,ty,str(n),ha='center',va='center',fontsize=6.2,color='#37474F',fontweight='bold',zorder=5)
    inner_bg=plt.Circle((0,0),R_INNER,color='white',zorder=2); ax_gauge.add_patch(inner_bg)
    norm_score=np.clip(score,0,20); needle_angle=np.deg2rad(180-(norm_score/20)*180)
    nx=0.88*np.cos(needle_angle); ny=0.88*np.sin(needle_angle)
    ax_gauge.annotate('',xy=(nx,ny),xytext=(0,0),arrowprops=dict(arrowstyle='->',color='#1A1A2E',lw=2.8))
    ax_gauge.plot(0,0,'o',color='#1A1A2E',markersize=9,zorder=6)
    ax_gauge.text(0,-0.18,f'{score}/20',ha='center',va='center',fontsize=14,fontproperties=MPL_FONT_PROP_BOLD,color='#1A1A2E')
    ax_button.axis('off'); ax_button.set_xlim(0,1); ax_button.set_ylim(0,1)
    rec,rec_color=recommendation(score)
    circle=mpatches.Circle((0.5,0.5),0.35,facecolor=rec_color,edgecolor='#333333',linewidth=2,transform=ax_button.transAxes)
    ax_button.add_patch(circle); ax_button.text(0.5,0.5,rtl(rec),ha='center',va='center',fontsize=25,fontproperties=MPL_FONT_PROP_BOLD,color='white',transform=ax_button.transAxes)
    ax_button.text(0.5,0.12,f'{score}/20',ha='center',va='center',fontsize=10,fontproperties=MPL_FONT_PROP,transform=ax_button.transAxes)
    plt.tight_layout(); return chart_bytes(fig)


def make_ichimoku_chart(d):
    d=d.tail(180).copy().reset_index(); xs=np.arange(len(d))
    fig,ax=plt.subplots(figsize=(14,7)); ax.set_facecolor('#FAFAFA'); ax.grid(True,linestyle=':',color='#dddddd',alpha=0.7,zorder=0)
    W=0.4
    for i,row in d.iterrows():
        o,h,l,c=float(row['Open']),float(row['High']),float(row['Low']),float(row['Close'])
        bullish=c>=o; body_top=max(o,c); body_bottom=min(o,c)
        if bullish: body_color='none'; edge_color='#26a69a'; wick_color='#26a69a'
        else: body_color='#ef5350'; edge_color='#ef5350'; wick_color='#ef5350'
        ax.plot([i,i],[l,body_bottom],color=wick_color,lw=0.8,zorder=2); ax.plot([i,i],[body_top,h],color=wick_color,lw=0.8,zorder=2)
        rect=plt.Rectangle((i-W,body_bottom),2*W,body_top-body_bottom,facecolor=body_color,edgecolor=edge_color,linewidth=0.8,zorder=3); ax.add_patch(rect)
    ax.plot(xs,d['Tenkan'].values,color='#E91E63',lw=0.7,label=rtl('التنكن (9)'),zorder=4)
    ax.plot(xs,d['Kijun'].values,color='#1565C0',lw=0.7,label=rtl('الكيجن (26)'),zorder=4)
    sa=d['Senkou_A'].values; sb=d['Senkou_B'].values; valid=~(np.isnan(sa)|np.isnan(sb))
    if valid.sum()>1:
        xv=xs[valid]; sav,sbv=sa[valid],sb[valid]
        ax.fill_between(xv,sav,sbv,where=(sav>=sbv),alpha=0.2,color='#4CAF50',label=rtl('سحابة صاعدة'),zorder=1)
        ax.fill_between(xv,sav,sbv,where=(sav<sbv),alpha=0.2,color='#E53935',label=rtl('سحابة هابطة'),zorder=1)
        ax.plot(xv,sav,color='#4CAF50',lw=0.5,alpha=0.8,zorder=3); ax.plot(xv,sbv,color='#E53935',lw=0.5,alpha=0.8,zorder=3)
    if 'Chikou' in d.columns:
        chikou=d['Chikou'].values; valid_c=~np.isnan(chikou)
        if valid_c.sum()>1: ax.plot(xs[valid_c],chikou[valid_c],color='#FF6F00',lw=0.7,alpha=0.85,label=rtl('الشيكو (26)'),zorder=4)
    step=max(1,len(d)//6); tick_pos=xs[::step]
    tick_lbls=[str(d['Date'].iloc[i])[:7] if 'Date' in d.columns else str(d.index[i])[:7] for i in tick_pos]
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lbls,rotation=30,fontsize=8); ax.set_xlim(-1,len(d))
    ax.set_ylabel(rtl('السعر'),fontproperties=MPL_FONT_PROP_BOLD); ax.legend(prop=MPL_FONT_PROP,fontsize=8,loc='upper left')
    plt.tight_layout(); return chart_bytes(fig)


def make_cci_willr_chart(d):
    d=d.tail(180).copy(); fig,(a1,a2)=plt.subplots(2,1,figsize=(14,6),sharex=True); x=d.index
    cci=d['CCI']; a1.plot(x,cci,color='#7C4DFF',lw=1.3)
    a1.axhline(100,color=RED_HEX,ls='--',lw=0.8,alpha=0.7); a1.axhline(0,color='gray',ls='--',lw=0.8,alpha=0.5); a1.axhline(-100,color='#43A047',ls='--',lw=0.8,alpha=0.7)
    a1.fill_between(x,cci,100,where=(cci>100),alpha=0.2,color=RED_HEX); a1.fill_between(x,cci,-100,where=(cci<-100),alpha=0.2,color='#43A047')
    a1.set_ylabel('CCI (20)',fontproperties=MPL_FONT_PROP_BOLD); a1.grid(True,alpha=0.3)
    wr=d['WILLR']; a2.plot(x,wr,color='#0097A7',lw=1.3)
    a2.axhline(-20,color=RED_HEX,ls='--',lw=0.8,alpha=0.7); a2.axhline(-50,color='gray',ls='--',lw=0.8,alpha=0.5); a2.axhline(-80,color='#43A047',ls='--',lw=0.8,alpha=0.7)
    a2.fill_between(x,wr,-20,where=(wr>-20),alpha=0.2,color=RED_HEX); a2.fill_between(x,wr,-80,where=(wr<-80),alpha=0.2,color='#43A047')
    a2.set_ylabel('Williams %R',fontproperties=MPL_FONT_PROP_BOLD); a2.set_ylim(-105,5); a2.grid(True,alpha=0.3); a2.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    plt.tight_layout(); return chart_bytes(fig)

###############################
#######################################

# ═══════════════════════════════════════════════════════════════
# CORRECTED CHART LEGENDS
# ───────────────────────────────────────────────────────────────
# Legend titles now use the format:
#   "(SMA) المتوسطات المتحركة البسيطة"
#   "(EMA) المتوسطات المتحركة الأسية"
#   "(Alligator) مؤشر التمساح"
#   "(Supertrend) مؤشر السوبر تريند"
#
# To change a legend title, edit the `title` variable in each function.
# To change which periods are shown, edit the for-loop lists.
# ═══════════════════════════════════════════════════════════════

def make_price_chart(d, sup=None, res=None):
    """SMA chart with corrected Arabic legend title."""
    sup = sup or []; res = res or []; d = d.tail(180).copy()
    p = d[['Open','High','Low','Close','Volume']].copy()
    aps, labels = [], []
    for col, clr, lbl in [
        ('SMA20', BLUE_HEX,   'SMA 20'),
        ('SMA50', RED_HEX,    'SMA 50'),
        ('SMA100', VIOLET_HEX, 'SMA 100'),
        ('SMA200', BLACK_HEX,  'SMA 200'),
    ]:
        if col in d and d[col].notna().sum() > 10:
            aps.append(mpf.make_addplot(d[col], color=clr, width=1))
            labels.append(lbl)
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit',
                               wick='inherit', volume={'up':'#80cbc4','down':'#ef9a9a'})
    st = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#dddddd',
                             rc={'axes.facecolor':'#FAFAFA'})
    kw = dict(type='candle', style=st, volume=True, figsize=(14,7),
              returnfig=True, warn_too_much_data=9999)
    if aps: kw['addplot'] = aps
    fig, ax = mpf.plot(p, **kw)
    main_ax = ax[0]
    # ── Corrected legend title ──
    title = rtl('(SMA) المتوسطات المتحركة البسيطة')
    if labels:
        main_ax.legend(labels, loc='upper left', fontsize=8,
                       prop=MPL_FONT_PROP, title=title,
                       title_fontproperties=MPL_FONT_PROP)
    fig.subplots_adjust(right=0.95, left=0.05)
    return chart_bytes(fig)


def make_ema_chart(d, sup=None, res=None):
    """EMA chart with corrected Arabic legend title."""
    sup = sup or []; res = res or []; d = d.tail(180).copy()
    p = d[['Open','High','Low','Close','Volume']].copy()
    aps, labels = [], []
    for col, clr, lbl in [
        ('EMA20', BLUE_HEX,   'EMA 20'),
        ('EMA50', RED_HEX,    'EMA 50'),
        ('EMA100', VIOLET_HEX, 'EMA 100'),
        ('EMA200', BLACK_HEX,  'EMA 200'),
    ]:
        if col in d and d[col].notna().sum() > 10:
            aps.append(mpf.make_addplot(d[col], color=clr, width=1))
            labels.append(lbl)
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit',
                               wick='inherit', volume={'up':'#80cbc4','down':'#ef9a9a'})
    st = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#dddddd',
                             rc={'axes.facecolor':'#FAFAFA'})
    kw = dict(type='candle', style=st, volume=True, figsize=(14,7),
              returnfig=True, warn_too_much_data=9999)
    if aps: kw['addplot'] = aps
    fig, ax = mpf.plot(p, **kw)
    main_ax = ax[0]
    # ── Corrected legend title ──
    title = rtl('(EMA) المتوسطات المتحركة الأسية')
    if labels:
        main_ax.legend(labels, loc='upper left', fontsize=8,
                       prop=MPL_FONT_PROP, title=title,
                       title_fontproperties=MPL_FONT_PROP)
    fig.subplots_adjust(right=0.95, left=0.05)
    return chart_bytes(fig)


def make_alligator_chart(d):
    """Alligator chart with corrected Arabic legend title."""
    d = d.tail(180).copy()
    p = d[['Open','High','Low','Close','Volume']].copy()
    aps, labels = [], []
    for col, clr, lbl in [
        ('Alligator_Jaw',   '#1565C0', rtl('الفك (13,8)')),
        ('Alligator_Teeth', '#E91E63', rtl('الأسنان (8,5)')),
        ('Alligator_Lips',  '#4CAF50', rtl('الشفاه (5,3)')),
    ]:
        if col in d and d[col].notna().sum() > 10:
            aps.append(mpf.make_addplot(d[col], color=clr, width=1))
            labels.append(lbl)
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit',
                               wick='inherit', volume={'up':'#80cbc4','down':'#ef9a9a'})
    st = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#dddddd',
                             rc={'axes.facecolor':'#FAFAFA'})
    kw = dict(type='candle', style=st, volume=False, figsize=(14,7),
              returnfig=True, warn_too_much_data=9999)
    if aps: kw['addplot'] = aps
    fig, ax = mpf.plot(p, **kw)
    # ── Corrected legend title ──
    title = rtl('(Alligator) مؤشر التمساح')
    if labels:
        ax[0].legend(labels, loc='upper left', fontsize=8,
                     prop=MPL_FONT_PROP, title=title,
                     title_fontproperties=MPL_FONT_PROP)
    fig.subplots_adjust(right=0.95, left=0.05)
    return chart_bytes(fig)


def make_supertrend_chart(d):
    """Supertrend chart with corrected Arabic legend title."""
    d = d.tail(180).copy()
    p = d[['Open','High','Low','Close','Volume']].copy()
    st_val, st_dir = _compute_supertrend(d)
    st_up   = st_val.where(st_dir == 1,  other=np.nan)
    st_down = st_val.where(st_dir == -1, other=np.nan)
    aps, labels = [], []
    if st_up.notna().any():
        aps.append(mpf.make_addplot(st_up, color='#4CAF50', width=1))
        labels.append(rtl('صاعد'))
    if st_down.notna().any():
        aps.append(mpf.make_addplot(st_down, color='#E53935', width=1))
        labels.append(rtl('هابط'))
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit',
                               wick='inherit', volume={'up':'#80cbc4','down':'#ef9a9a'})
    sty = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#dddddd',
                              rc={'axes.facecolor':'#FAFAFA'})
    kw = dict(type='candle', style=sty, volume=False, figsize=(14,7),
              returnfig=True, warn_too_much_data=9999)
    if aps: kw['addplot'] = aps
    fig, ax = mpf.plot(p, **kw)
    # ── Corrected legend title ──
    title = rtl('(Supertrend) مؤشر السوبر تريند')
    if labels:
        ax[0].legend(labels, loc='upper left', fontsize=8,
                     prop=MPL_FONT_PROP, title=title,
                     title_fontproperties=MPL_FONT_PROP)
    fig.subplots_adjust(right=0.95, left=0.05)
    return chart_bytes(fig)



# ═══════════════════════════════════════════════════════════════
# SECTION 8: PDF REPORT CLASS
# ───────────────────────────────────────────────────────────────
# COVER PAGE BOXES (what changed):
#   Row 1: السعر الحالي | التغير اليومي | القيمة السوقية
#   Row 2: مكرر الربحية | ربحية السهم   | عائد التوزيعات
#   Row 3: مضاعف القيمة الدفترية | العائد على حقوق المساهمين | العائد على الأصول  ← NEW
#   Row 4: حجم التداول | القيمة الدفترية  ← NEW (was: قيمة التداول, عدد الصفقات)
#
# REMOVED: بيتا, قيمة التداول, عدد الصفقات
# ADDED:   العائد على الأصول, القيمة الدفترية
# ═══════════════════════════════════════════════════════════════

PERIOD_AR = {
    '1 Day':'يوم','1 Week':'أسبوع','1 Month':'شهر',
    '3 Months':'3 أشهر','6 Months':'6 أشهر','1 Year':'سنة','YTD':'منذ بداية السنة'
}
RISK_AR = {
    'Daily Volatility':'التذبذب اليومي','Annual Volatility':'التذبذب السنوي',
    'Sharpe Ratio':'نسبة شارب','Sortino Ratio':'نسبة سورتينو',
    'Max Drawdown':'أقصى تراجع','Avg Daily Return':'متوسط العائد اليومي',
    'Best Day':'أفضل يوم','Worst Day':'أسوأ يوم'
}


class Report:
    def __init__(self, path, ticker, info, display_ticker=None):
        self.c = pdfcanvas.Canvas(path, pagesize=A4)
        self.c.setTitle(f'{ticker} Arabic Stock Report')
        self.tk = ticker
        self.display_tk = display_ticker or ticker
        self.info = info
        self.pn = 0

    def _font(self, bold=False, size=10):
        self.c.setFont(AR_FONT_BOLD if bold else AR_FONT, size)

    def _bar(self, title):
        self.pn += 1; c = self.c
        c.setFillColor(NAVY); c.rect(0, PAGE_H-34*mm, PAGE_W, 34*mm, fill=1, stroke=0)
        c.setFillColor(TEAL); c.rect(0, PAGE_H-36*mm, PAGE_W, 2*mm, fill=1, stroke=0)
        c.setFillColor(WHITE); self._font(True, 15)
        c.drawRightString(PAGE_W-MG, PAGE_H-18*mm, rtl(title))
        self._font(False, 9)
        c.drawString(MG, PAGE_H-14*mm, self.display_tk)
        c.drawString(MG, PAGE_H-22*mm, datetime.now().strftime('%Y-%m-%d'))

    def _foot(self):
        c = self.c
        c.setFillColor(LGRAY); c.rect(0, 0, PAGE_W, 10*mm, fill=1, stroke=0)
        c.setFillColor(DGRAY); self._font(False, 7)
        c.drawString(MG, 4*mm, self.display_tk)
        c.drawRightString(PAGE_W-MG, 4*mm, rtl(f'الصفحة {self.pn}'))

    def _stitle(self, y, t):
        c = self.c; c.setFillColor(NAVY); self._font(True, 11)
        c.drawRightString(PAGE_W-MG, y, rtl(t))
        c.setStrokeColor(TEAL); c.setLineWidth(1.4)
        c.line(MG, y-4, PAGE_W-MG, y-4)
        return y - 18

    def _box(self, x, y, bw, bh, lbl, val, clr=None):
        c = self.c
        c.setFillColor(LGRAY); c.roundRect(x, y, bw, bh, 5, fill=1, stroke=0)
        c.setFillColor(DGRAY); self._font(False, 8)
        c.drawCentredString(x+bw/2, y+bh-14, tx(lbl))
        if isinstance(val, (int, float)):
            val_str, val_clr = fmt_n(val)
        else:
            val_str = str(val) if val is not None else '-'
            val_clr = None
        if val_clr:      fill_color = val_clr
        elif clr:         fill_color = HexColor(clr) if isinstance(clr, str) else clr
        else:             fill_color = NAVY
        c.setFillColor(fill_color); self._font(True, 12.5)
        c.drawCentredString(x+bw/2, y+8, tx(val_str))

    def _table(self, y, rows, cw_list, sig_mode=False, score_mode=False):
        # (unchanged from original — keep as-is)
        c = self.c; rh = 16; total_w = sum(cw_list)
        for i, row in enumerate(rows):
            ry = y - i*rh
            if i==0: bg,fg,is_bold = NAVY,WHITE,True
            elif i%2==1: bg,fg,is_bold = LGRAY,TXTDARK,False
            else: bg,fg,is_bold = WHITE,TXTDARK,False
            c.setFillColor(bg); c.rect(MG, ry-4, total_w, rh, fill=1, stroke=0)
            sx=MG; c.setStrokeColor(HexColor('#D8DEE9')); c.setLineWidth(0.5)
            for col_w in cw_list[:-1]: sx+=col_w; c.line(sx, ry-4, sx, ry-4+rh)
            cx=MG
            for j, cell in enumerate(row):
                fill=fg; is_cell_bold=is_bold
                if i>0 and j==0 and str(cell).startswith('-') and not sig_mode and not score_mode: fill=RED
                if i>0 and sig_mode and j==0:
                    raw=str(cell)
                    if 'إيجابي' in raw: fill=GREEN
                    elif 'سلبي' in raw: fill=RED
                    else: fill=ORANGE
                    is_cell_bold=True
                if i>0 and score_mode and j==0:
                    if cell=='✓': fill=GREEN
                    elif cell=='✗': fill=RED
                    is_cell_bold=True
                c.setFillColor(fill); self._font(is_cell_bold, 8)
                c.drawRightString(cx+cw_list[j]-4, ry+2, tx(cell))
                cx += cw_list[j]
        return y - len(rows)*rh - 6

    def _img(self, y, buf, max_h):
        c = self.c; buf.seek(0); img = ImageReader(buf)
        iw, ih = img.getSize(); ratio = ih/iw
        dw = CW; dh = dw*ratio
        if dh > max_h: dh = max_h; dw = dh/ratio
        x = MG+(CW-dw)/2
        c.drawImage(img, x, y-dh-2, dw, dh)
        return y - dh - 8

    def _wrap_arabic_text(self, text, max_width, font_size):
        words = str(text).split(); lines = []; current_line = []
        for word in words:
            test_shaped = rtl(' '.join(current_line + [word]))
            line_width = self.c.stringWidth(test_shaped, AR_FONT, font_size)
            if line_width <= max_width: current_line.append(word)
            else:
                if current_line: lines.append(rtl(' '.join(current_line)))
                current_line = [word]
        if current_line: lines.append(rtl(' '.join(current_line)))
        return lines

    # ══════════════════════════════════════════════════════════
    # COVER PAGE — CHANGED BOXES
    # ══════════════════════════════════════════════════════════
    def cover(self, price, chg, info, rec_txt, rec_color, score):
        c = self.c; self.pn += 1

        # ── Header banner ──
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
        c.drawRightString(PAGE_W-MG, PAGE_H-52*mm,
                          tx(f'{self.display_tk} | {safe(info,"exchange","-")}'))
        self._font(False, 9)
        c.drawString(MG, PAGE_H-18*mm, datetime.now().strftime('%Y-%m-%d'))

        # ── Recommendation badge ──
        c.setFillColor(HexColor(rec_color))
        c.roundRect(MG, PAGE_H-58*mm, 60*mm, 14*mm, 8, fill=1, stroke=0)
        c.setFillColor(WHITE); self._font(True, 11)
        c.drawCentredString(MG+30*mm, PAGE_H-50*mm, rtl(rec_txt))
        self._font(False, 8)
        c.drawCentredString(MG+30*mm, PAGE_H-54*mm, rtl(f'النتيجة {score}/20'))

        # ── Info boxes grid ──
        col_bw = CW/3 - 4*mm
        col_bh = 18*mm
        col_gap = 4*mm
        x1 = MG
        x2 = MG + col_bw + col_gap
        x3 = MG + 2*(col_bw + col_gap)
        y1 = PAGE_H - 110*mm
        y2 = y1 - col_bh - col_gap
        y3 = y2 - col_bh - col_gap
        y4 = y3 - col_bh - col_gap

        # ── Row 1: السعر | التغير | القيمة السوقية ──
        self._box(x1, y1, col_bw, col_bh, 'السعر الحالي', price)
        self._box(x2, y1, col_bw, col_bh, 'التغير اليومي',
                  f'{chg:+.2f}%', clr=GREEN_HEX if chg >= 0 else RED_HEX)
        self._box(x3, y1, col_bw, col_bh, 'القيمة السوقية',
                  fmt_n(safe(info, 'marketCap'))[0])

        # ── Row 2: مكرر الربحية | ربحية السهم | عائد التوزيعات ──
        pe = safe(info, 'trailingPE')
        pe_display = info.get('trailingPE_display')
        if pe_display:
            pe_str = pe_display
        elif pe:
            pe_str = f'{float(pe):.2f}'
        else:
            pe_str = '-'
        self._box(x1, y2, col_bw, col_bh, 'مكرر الربحية', pe_str)

        eps = safe(info, 'trailingEps')
        eps_formatted = info.get('trailingEpsFormatted')
        if eps_formatted:
            eps_str = eps_formatted
        elif eps is not None:
            eps_val = float(eps)
            eps_str = f'{eps_val:.2f}'
        else:
            eps_str = '-'
        eps_val_f = float(eps) if eps is not None else None
        eps_color = (GREEN_HEX if (eps_val_f is not None and eps_val_f >= 0)
                     else RED_HEX if eps_val_f is not None
                     else None)
        self._box(x2, y2, col_bw, col_bh, 'ربحية السهم', eps_str, clr=eps_color)

        dy = safe(info, 'dividendYield')
        self._box(x3, y2, col_bw, col_bh, 'عائد التوزيعات',
                  fmt_p(dy)[0] if dy else '-')

        # ── Row 3: مضاعف القيمة الدفترية | العائد على حقوق المساهمين | العائد على الأصول ──
        pb = safe(info, 'priceToBook')
        self._box(x1, y3, col_bw, col_bh, 'مضاعف القيمة الدفترية',
                  f'{float(pb):.2f}' if pb else '-')

        roe_display = info.get('returnOnEquity_display', '-')
        roe = safe(info, 'returnOnEquity')
        self._box(x2, y3, col_bw, col_bh, 'العائد على حقوق المساهمين',
                  roe_display if roe_display != '-' else (fmt_p(roe)[0] if roe else '-'))

        # ★ NEW: العائد على الأصول (was: بيتا)
        roa_display = info.get('returnOnAssets_display', '-')
        self._box(x3, y3, col_bw, col_bh, 'العائد على الأصول', roa_display)

        # ── Row 4: حجم التداول | القيمة الدفترية (CHANGED) ──
        # ★ REMOVED: قيمة التداول, عدد الصفقات
        # ★ ADDED: القيمة الدفترية
        self._box(x1, y4, col_bw, col_bh, 'حجم التداول',
                  fmt_n(safe(info, 'volume'), d=0)[0])

        # ★ NEW: القيمة الدفترية
        bv_display = info.get('bookValue_display', '-')
        bv = safe(info, 'bookValue')
        if bv_display and bv_display != '-':
            bv_str = bv_display
        elif bv is not None:
            bv_str = f'{float(bv):.2f}'
        else:
            bv_str = '-'
        self._box(x2, y4, col_bw, col_bh, 'القيمة الدفترية', bv_str)

        # Third box in row 4 is now empty — use it for something or leave blank
        # Option: show متوسط الحجم or leave as spacer
        self._box(x3, y4, col_bw, col_bh, 'متوسط الحجم',
                  fmt_n(safe(info, 'averageVolume'), d=0)[0])

        # ── 52-week range bar ──
        w52h = safe(info, 'fiftyTwoWeekHigh')
        w52l = safe(info, 'fiftyTwoWeekLow')
        if w52h and w52l and float(w52h) != float(w52l):
            bar_y = y4 - 12*mm
            c.setFillColor(NAVY); self._font(True, 8)
            c.drawRightString(PAGE_W-MG, bar_y+2, rtl('نطاق 52 أسبوع'))
            bar_x = MG; bar_w = CW; bar_h = 5*mm; bar_y2 = bar_y - bar_h - 2
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
            c.setFillColor(DGRAY); self._font(False, 7)
            c.drawString(bar_x, bar_y2-9, f'{float(w52l):.2f}')
            c.drawRightString(bar_x+bar_w, bar_y2-9, f'{float(w52h):.2f}')
            c.setFillColor(NAVY); self._font(True, 7)
            c.drawCentredString(bar_x+bar_w/2, bar_y2-9, f'{pos*100:.0f}%')
            y_after = bar_y2 - 12*mm
        else:
            y_after = y4 - 6*mm

        # ── Company info section ──
        y = y_after
        y = self._stitle(y, 'معلومات الشركة')
        sector   = short_text(safe(info, 'sector', '-') or '-', 26)
        industry = short_text(safe(info, 'industry', '-') or '-', 26)
        items = [
            ('القطاع', sector),
            ('الصناعة', industry),
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
            xr = MG + (col+1)*half - 5; yy = y - row*18
            c.setFillColor(NAVY); self._font(True, 8.5)
            c.drawRightString(xr, yy, rtl(f'{lbl}:'))
            c.setFillColor(TXTDARK); self._font(False, 8.5)
            c.drawRightString(xr-85, yy, tx(val))

        c.setFillColor(DGRAY); self._font(False, 7)
        c.drawCentredString(PAGE_W/2, 14*mm,
                            rtl('هذا التقرير لأغراض معلوماتية فقط وليس توصية استثمارية.'))
        self._foot(); c.showPage()

    # ══════════════════════════════════════════════════════════
    # CHARTS PAGE — corrected section titles
    # ══════════════════════════════════════════════════════════
    def main_charts_page(self, main_img, sma_img, ema_img, alligator_img, supertrend_img):
        self._bar('الرسم البياني'); self._foot()
        c = self.c; y = PAGE_H - 44*mm

        # ── Support/Resistance chart (full width) ──
        y = self._stitle(y, 'الدعوم والمقاومات')
        y = self._img(y, main_img, 78*mm)
        y -= 10*mm

        # ── SMA (right) | EMA (left) ──
        mid_x = MG + CW / 2
        right_title_x = PAGE_W - MG
        left_title_x  = mid_x - 4*mm

        c.setFillColor(NAVY); self._font(True, 9)
        c.drawRightString(right_title_x, y,
                          rtl('(SMA) المتوسطات المتحركة البسيطة'))
        c.drawRightString(left_title_x, y,
                          rtl('(EMA) المتوسطات المتحركة الأسية'))
        c.setStrokeColor(TEAL); c.setLineWidth(1.0)
        c.line(mid_x + 2*mm, y-3, PAGE_W - MG, y-3)
        c.line(MG, y-3, mid_x - 2*mm, y-3)
        y -= 8*mm

        sma_img.seek(0); img_sma = ImageReader(sma_img)
        ema_img.seek(0); img_ema = ImageReader(ema_img)
        dw = (CW / 2) - 3*mm; maxh = 58*mm

        ratio_sma = img_sma.getSize()[1] / img_sma.getSize()[0]
        dh_sma = min(dw * ratio_sma, maxh); dw_sma = dh_sma / ratio_sma

        ratio_ema = img_ema.getSize()[1] / img_ema.getSize()[0]
        dh_ema = min(dw * ratio_ema, maxh); dw_ema = dh_ema / ratio_ema

        x_sma = PAGE_W - MG - dw_sma; x_ema = MG
        c.drawImage(img_sma, x_sma, y - dh_sma, dw_sma, dh_sma)
        c.drawImage(img_ema, x_ema, y - dh_ema, dw_ema, dh_ema)
        y -= max(dh_sma, dh_ema) + 5*mm

        # ── Alligator (right) | Supertrend (left) ──
        c.setFillColor(NAVY); self._font(True, 9)
        c.drawRightString(right_title_x, y,
                          rtl('(Alligator) مؤشر التمساح'))
        c.drawRightString(left_title_x, y,
                          rtl('(Supertrend) مؤشر السوبر تريند'))
        c.setStrokeColor(TEAL); c.setLineWidth(1.0)
        c.line(mid_x + 2*mm, y-3, PAGE_W - MG, y-3)
        c.line(MG, y-3, mid_x - 2*mm, y-3)
        y -= 8*mm

        alligator_img.seek(0);  img_alg = ImageReader(alligator_img)
        supertrend_img.seek(0); img_st  = ImageReader(supertrend_img)

        ratio_alg = img_alg.getSize()[1] / img_alg.getSize()[0]
        dh_alg = min(dw * ratio_alg, maxh); dw_alg = dh_alg / ratio_alg

        ratio_st = img_st.getSize()[1] / img_st.getSize()[0]
        dh_st = min(dw * ratio_st, maxh); dw_st = dh_st / ratio_st

        x_alg = PAGE_W - MG - dw_alg; x_st = MG
        c.drawImage(img_alg, x_alg, y - dh_alg, dw_alg, dh_alg)
        c.drawImage(img_st, x_st, y - dh_st, dw_st, dh_st)
        self.c.showPage()

    # All remaining Report methods (tech_page, perf_page, fund_page,
    # signal_page, ichimoku_page, cci_willr_page, review_page, save)
    # are UNCHANGED from your original — keep them exactly as-is.
    def tech_page(self, tech_img, bb_img):
        # ... (unchanged) ...
        self._bar('المؤشرات الفنية'); self._foot()
        y=PAGE_H-44*mm; y=self._stitle(y,'RSI و MACD و ROC 12'); y=self._img(y, tech_img, 113*mm)
        y=self._stitle(y,'نطاقات بولنجر'); self._img(y, bb_img, 113*mm); self.c.showPage()

    def perf_page(self, pers, risk, dd_img, vol_img, score_criteria, total_score):
        # ... (unchanged — keep your entire original method) ...
        self._bar('الأداء والمخاطر'); self._foot()
        y=PAGE_H-44*mm; y=self._stitle(y,'العوائد حسب الفترة')
        rows=[['القيمة','الفترة']]
        for k,v in pers.items(): rows.append([f'{v:+.2f}%',PERIOD_AR.get(k,k)])
        y=self._table(y,rows,[CW*0.38,CW*0.62]); y=self._stitle(y,'مقاييس المخاطر')
        rows2=[['القيمة','المقياس']]
        for k,v in risk.items(): rows2.append([str(v),RISK_AR.get(k,k)])
        y=self._table(y,rows2,[CW*0.38,CW*0.62]); y=self._stitle(y,'منحنى التراجع'); self._img(y,dd_img,58*mm); self.c.showPage()
        self._bar('تحليل الحجم'); self._foot()
        y2=PAGE_H-44*mm; y2=self._stitle(y2,'الحجم اليومي مقابل متوسط 20 يوم'); y2=self._img(y2,vol_img,100*mm)
        y_table=y2-6*mm
        if y_table>80*mm:
            y_table=self._stitle(y_table,f'جدول نقاط النتيجة ({total_score}/20)')
            rows_score=[['النقاط','الحالة','البند']]
            for lbl,(symbol,pt) in score_criteria.items():
                status=rtl('نعم ✓') if pt==1 else rtl('لا ✗')
                rows_score.append([str(pt),status,rtl(short_text(lbl,35))])
            rows_score.append([str(total_score),rtl('من 20'),rtl('الإجمالي')])
            self._table(y_table,rows_score,[CW*0.15,CW*0.30,CW*0.55],score_mode=True)
        else:
            self.c.showPage(); self._bar('جدول نقاط النتيجة'); self._foot()
            y_table=PAGE_H-44*mm; y_table=self._stitle(y_table,f'جدول نقاط النتيجة ({total_score}/20)')
            rows_score=[['النقاط','الحالة','البند']]
            for lbl,(symbol,pt) in score_criteria.items():
                status=rtl('نعم ✓') if pt==1 else rtl('لا ✗')
                rows_score.append([str(pt),status,rtl(short_text(lbl,35))])
            rows_score.append([str(total_score),rtl('من 20'),rtl('الإجمالي')])
            self._table(y_table,rows_score,[CW*0.15,CW*0.30,CW*0.55],score_mode=True)
        self.c.showPage()

    def fund_page(self, info):
        # ... (unchanged — keep your entire original method) ...
        self._bar('التحليل الأساسي'); self._foot()
        y=PAGE_H-44*mm; y=self._stitle(y,'التقييم')
        val_items=[(safe(info,'trailingPE','-'),'Trailing P/E'),(safe(info,'forwardPE','-'),'Forward P/E'),(safe(info,'priceToBook','-'),'Price / Book'),(safe(info,'priceToSalesTrailing12Months','-'),'Price / Sales'),(safe(info,'enterpriseToEbitda','-'),'EV / EBITDA'),(safe(info,'enterpriseToRevenue','-'),'EV / Revenue'),(safe(info,'pegRatio','-'),'PEG Ratio')]
        val_rows=[['القيمة','البند']]
        for v,lbl in val_items: val_rows.append([f'{v:.2f}' if isinstance(v,(int,float)) else str(v),lbl])
        y=self._table(y,val_rows,[CW*0.40,CW*0.60]); y=self._stitle(y,'الربحية')
        prof_rows=[['القيمة','البند'],[fmt_n(safe(info,'totalRevenue'))[0],'الإيرادات'],[fmt_n(safe(info,'netIncomeToCommon'))[0],'صافي الدخل'],[fmt_n(safe(info,'ebitda'))[0],'EBITDA'],[fmt_p(safe(info,'grossMargins'))[0],'هامش إجمالي'],[fmt_p(safe(info,'operatingMargins'))[0],'هامش تشغيلي'],[fmt_p(safe(info,'profitMargins'))[0],'هامش صافي'],[fmt_p(safe(info,'returnOnEquity'))[0],'ROE'],[fmt_p(safe(info,'returnOnAssets'))[0],'ROA']]
        y=self._table(y,prof_rows,[CW*0.40,CW*0.60]); y=self._stitle(y,'المركز المالي')
        fin_items=[(fmt_n(safe(info,'totalCash'))[0],'إجمالي النقد'),(fmt_n(safe(info,'totalDebt'))[0],'إجمالي الدين'),(safe(info,'debtToEquity','-'),'Debt / Equity'),(safe(info,'currentRatio','-'),'Current Ratio'),(safe(info,'quickRatio','-'),'Quick Ratio'),(fmt_n(safe(info,'bookValue'))[0],'القيمة الدفترية للسهم')]
        fin_rows=[['القيمة','البند']]
        for v,lbl in fin_items: fin_rows.append([f'{v:.2f}' if isinstance(v,(int,float)) else str(v),lbl])
        y=self._table(y,fin_rows,[CW*0.40,CW*0.60])
        if y<60*mm: self.c.showPage(); self._bar('التحليل الأساسي (تابع)'); self._foot(); y=PAGE_H-44*mm
        y=self._stitle(y,'التوزيعات والأسهم')
        div_rows=[['القيمة','البند'],[fmt_n(safe(info,'dividendRate'))[0],'معدل التوزيع'],[fmt_p(safe(info,'dividendYield'))[0],'عائد التوزيع'],[fmt_p(safe(info,'payoutRatio'))[0],'نسبة التوزيع'],[fmt_n(safe(info,'sharesOutstanding'),d=0)[0],'الأسهم القائمة'],[fmt_n(safe(info,'floatShares'),d=0)[0],'الأسهم الحرة']]
        self._table(y,div_rows,[CW*0.40,CW*0.60]); self.c.showPage()

    def signal_page(self, gauge_img, sig, score, sup, res, d):
        # ... (unchanged — keep your entire original method) ...
        self._bar('الإشارات والتحليل'); self._foot()
        c=self.c; y=PAGE_H-44*mm; y=self._stitle(y,'مؤشر التحليل'); y=self._img(y,gauge_img,50*mm)
        c.setFillColor(TXTDARK); self._font(False,9); c.drawRightString(PAGE_W-MG, y-2, rtl(f'النتيجة الكلية: {score}/20')); y-=14
        y=self._stitle(y,'ملخص الإشارات الفنية')
        rows=[['التحليل','الإشارة','المؤشر']]
        for ind_name,(txt_signal,direction) in sig.items(): rows.append([decision_text(direction),txt_signal,ind_name])
        y=self._table(y,rows,[CW*0.18,CW*0.38,CW*0.44],sig_mode=True); y=self._stitle(y,'الدعم والمقاومة')
        sup_txt=' | '.join(f'{s:.2f}' for s in sup) if sup else 'N/A'; res_txt=' | '.join(f'{r:.2f}' for r in res) if res else 'N/A'
        c.setFillColor(GREEN); self._font(True,9); c.drawRightString(PAGE_W-MG, y, rtl(f'الدعوم: {sup_txt}')); y-=16
        c.setFillColor(RED); self._font(True,9); c.drawRightString(PAGE_W-MG, y, rtl(f'المقاومات: {res_txt}')); y-=22
        if y<100*mm: self.c.showPage(); self._bar('القيم الحالية للمؤشرات'); self._foot(); y=PAGE_H-44*mm
        y=self._stitle(y,'القيم الحالية للمؤشرات'); last=d.iloc[-1]
        rows2=[['القيمة','المؤشر'],[f"{float(last['Close']):.2f}",'سعر الإغلاق'],[f"{float(last['SMA20']):.2f}" if pd.notna(last['SMA20']) else '-','SMA 20'],[f"{float(last['SMA50']):.2f}" if pd.notna(last['SMA50']) else '-','SMA 50'],[f"{float(last['RSI']):.2f}" if pd.notna(last['RSI']) else '-','RSI'],[f"{float(last['MACD']):.4f}" if pd.notna(last['MACD']) else '-','MACD'],[f"{float(last['ATR']):.2f}" if pd.notna(last['ATR']) else '-','ATR'],[f"{float(last['ADX']):.1f}" if pd.notna(last['ADX']) else '-','ADX'],[f"{float(last['VWAP']):.2f}" if pd.notna(last['VWAP']) else '-','VWAP'],[f"{float(last['CCI']):.1f}" if pd.notna(last['CCI']) else '-','CCI (20)'],[f"{float(last['WILLR']):.1f}" if pd.notna(last['WILLR']) else '-','Williams %R']]
        self._table(y,rows2,[CW*0.35,CW*0.65])
        c.setFillColor(DGRAY); self._font(False,6.7); c.drawCentredString(PAGE_W/2, 16*mm, rtl('هذا التقرير آلي لأغراض معلوماتية فقط وليس نصيحة استثمارية.')); self.c.showPage()

    def ichimoku_page(self, ichimoku_img, pattern_img):
        # ... (unchanged) ...
        self._bar('الإيشيموكو ونماذج الشموع'); self._foot()
        y=PAGE_H-44*mm; y=self._stitle(y,'مخطط الإيشيموكو (Ichimoku Cloud)'); y=self._img(y,ichimoku_img,100*mm)
        y=self._stitle(y,'الشموع اليابانية (آخر 60 يوماً) مع النماذج المرصودة'); self._img(y,pattern_img,100*mm); self.c.showPage()

    def cci_willr_page(self, cci_img, patterns, divergences, d):
        # ... (unchanged — keep your entire original method) ...
        self._bar('مؤشرات إضافية: CCI وWilliams %R'); self._foot()
        c=self.c; y=PAGE_H-44*mm; y=self._stitle(y,'CCI (20) وWilliams %R (14)'); y=self._img(y,cci_img,90*mm); y-=6*mm
        last=d.iloc[-1]
        cci_v=float(last['CCI']) if pd.notna(last['CCI']) else None; willr_v=float(last['WILLR']) if pd.notna(last['WILLR']) else None
        osc_rows=[['القيمة','المؤشر'],[f'{cci_v:.1f}' if cci_v is not None else '-','CCI (20)'],[f'{willr_v:.1f}' if willr_v is not None else '-','Williams %R (14)']]
        y=self._table(y,osc_rows,[CW*0.35,CW*0.65]); y=self._stitle(y,'التباعد بين السعر والمؤشرات (Divergences)')
        if divergences:
            div_rows=[['النوع','المؤشر']]
            for ind,ar_type,en_type in divergences: div_rows.append([ar_type,ind])
            y=self._table(y,div_rows,[CW*0.60,CW*0.40])
        else:
            c.setFillColor(DGRAY); self._font(False,8.5); c.drawRightString(PAGE_W-MG, y, rtl('لا يوجد تباعد مرصود.')); y-=16
        y=self._stitle(y,'نماذج الشموع اليابانية المرصودة')
        if patterns:
            pat_rows=[['التوجه','النموذج','التاريخ']]
            for date_lbl,ar_name,en_name,bullish in patterns:
                direction='صعودي' if bullish is True else ('هبوطي' if bullish is False else 'محايد'); pat_rows.append([direction,ar_name,date_lbl])
            y=self._table(y,pat_rows,[CW*0.20,CW*0.45,CW*0.35],sig_mode=True)
        else:
            c.setFillColor(DGRAY); self._font(False,8.5); c.drawRightString(PAGE_W-MG, y, rtl('لم يُرصد نموذج شموع بارز.'))
        self.c.showPage()

    def review_page(self, review_sections, score):
        # ... (unchanged — keep your entire original method) ...
        self._bar('المراجعة الفنية الشاملة'); self._foot()
        c=self.c; y=PAGE_H-44*mm; line_h=13; para_gap=8; section_gap=6; font_size_body=9; max_text_width=CW-4*mm
        strip_h=10*mm; strip_y=y-strip_h
        if score>=14:   strip_fill=HexColor('#1B5E20')
        elif score>=10: strip_fill=HexColor('#1B2A4A')
        elif score>=7:  strip_fill=HexColor('#E65100')
        else:           strip_fill=HexColor('#B71C1C')
        c.setFillColor(strip_fill); c.roundRect(MG, strip_y, CW, strip_h, 5, fill=1, stroke=0)
        c.setFillColor(WHITE); self._font(True,12); c.drawCentredString(PAGE_W/2, strip_y + (strip_h - font_size_body) / 2, rtl(f'نتيجة التحليل الفني: {score} من أصل 20 نقطة')); y=strip_y-10*mm
        for section_title,paragraph in review_sections:
            if y<40*mm: c.showPage(); self._bar('المراجعة الفنية الشاملة (تابع)'); self._foot(); y=PAGE_H-44*mm
            y=self._stitle(y,section_title); y+=4
            lines=self._wrap_arabic_text(paragraph,max_text_width,font_size_body)
            c.setFillColor(TXTDARK); self._font(False,font_size_body)
            for line in lines:
                if y<25*mm: c.showPage(); self._bar('المراجعة الفنية الشاملة (تابع)'); self._foot(); y=PAGE_H-44*mm
                c.drawRightString(PAGE_W-MG, y, line); y-=line_h
            y-=para_gap+section_gap
        c.setFillColor(DGRAY); self._font(False,6.5)
        c.drawCentredString(PAGE_W/2, 16*mm, rtl('هذا التقرير آلي لأغراض معلوماتية فقط وليس نصيحة استثمارية.'))
        c.drawCentredString(PAGE_W/2, 12*mm, rtl('الأداء السابق لا يضمن النتائج المستقبلية. قم دائماً ببحثك الخاص.')); c.showPage()

    def save(self):
        self.c.save()



# ═══════════════════════════════════════════════════════════════
# SECTION 9: REPORT BUILDER
# ───────────────────────────────────────────────────────────────
# _build_report_sync(): called in thread pool
#   - Now uses find_ticker() to resolve Arabic names / partial codes
#   - Removed duplicate make_candle_pattern_chart call
# ═══════════════════════════════════════════════════════════════
_executor = ThreadPoolExecutor(max_workers=4)


def _build_report_sync(ticker_input: str):
    """Run in thread pool. Returns (pdf_bytes, summary_text, display_ticker)."""
    ti = ticker_input.strip()
    if not ti:
        raise ValueError('رمز فارغ')

    # ★ Try find_ticker first (handles Arabic names, partial codes, aliases)
    resolved = find_ticker(ti)
    if resolved:
        ticker = resolved
        # Display as code without .SR for cleaner output
        display_ticker = ticker.replace('.SR', '').lstrip('^')
    elif ti.replace('.', '').isdigit() and '.' not in ti:
        ticker = ti + '.SR'
        display_ticker = ti
    else:
        ticker = ti.upper()
        display_ticker = ticker

    df, df2, info = fetch_data(ticker)
    if df is None or len(df) < 30:
        raise ValueError(
            f'❌ البيانات غير كافية للرمز: {display_ticker}\n'
            'تأكد من صحة رقم الشركة.')

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
    cpat_img       = make_candle_pattern_chart(df, patterns)
    cci_img        = make_cci_willr_chart(d)

    price = float(df['Close'].iloc[-1])
    prev  = float(df['Close'].iloc[-2]) if len(df) > 1 else price
    chg   = (price / prev - 1) * 100

    pdf_buf = BytesIO()
    rpt = Report(pdf_buf, ticker, info, display_ticker)
    rpt.cover(price, chg, info, rec_txt, rec_color, score)
    rpt.main_charts_page(main_img, p_img, ema_img, alligator_img, supertrend_img)
    rpt.tech_page(t_img, bb_img)
    rpt.ichimoku_page(ichi_img, cpat_img)
    rpt.perf_page(pers, risk, dd_img, v_img, score_criteria, score)
    rpt.fund_page(info)
    rpt.signal_page(g_img, sig, score, sup, res, d)
    rpt.cci_willr_page(cci_img, patterns, divergences, d)
    rpt.review_page(review_sections, score)
    rpt.save()
    pdf_buf.seek(0)

    summary = (
        f"📊 *{display_ticker}*\n"
        f"السعر الحالي   : `{price:.2f}`\n"
        f"التغير اليومي  : `{chg:+.2f}%`\n"
        f"النتيجة        : `{score}/20`\n"
        f"التحليل        : `{rec_txt}`\n"
        f"الصفحات        : `{rpt.pn}`"
    )
    return pdf_buf, summary, display_ticker






