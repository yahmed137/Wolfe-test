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

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]

# ─────────────────────────────────────────────────────────────
# 1. FONT SETUP  (Cairo for Wolfe charts, Amiri for PDF reports)
# ─────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))

CAIRO_PATH = os.path.join(HERE, 'Cairo-Regular.ttf')
AMIRI_REG_PATH  = os.path.join(HERE, 'Amiri-Regular.ttf')
AMIRI_BOLD_PATH = os.path.join(HERE, 'Amiri-Bold.ttf')

AR_FONT      = 'Amiri'
AR_FONT_BOLD = 'Amiri-Bold'
AR_RE = re.compile(r'[\u0600-\u06FF]')
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

    # Cairo for wolfe charts
    _download_font(
        'https://github.com/google/fonts/raw/main/ofl/cairo/static/Cairo-Regular.ttf',
        CAIRO_PATH,
    )
    # Amiri for PDF reports
    _download_font(
        'https://github.com/google/fonts/raw/refs/heads/main/ofl/amiri/Amiri-Regular.ttf',
        AMIRI_REG_PATH,
    )
    _download_font(
        'https://github.com/google/fonts/raw/refs/heads/main/ofl/amiri/Amiri-Bold.ttf',
        AMIRI_BOLD_PATH,
    )

    # Register Amiri with ReportLab
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

    # Matplotlib fonts
    for fp in [AMIRI_REG_PATH, AMIRI_BOLD_PATH, CAIRO_PATH]:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)

    if os.path.exists(AMIRI_REG_PATH):
        MPL_FONT_PROP = fm.FontProperties(fname=AMIRI_REG_PATH)
        plt.rcParams['font.family'] = MPL_FONT_PROP.get_name()
    if os.path.exists(AMIRI_BOLD_PATH):
        MPL_FONT_PROP_BOLD = fm.FontProperties(fname=AMIRI_BOLD_PATH)
    plt.rcParams['axes.unicode_minus'] = False

    # Cairo for wolfe charts title
    if os.path.exists(CAIRO_PATH):
        prop = fm.FontProperties(fname=CAIRO_PATH)
        ARABIC_FONT = prop.get_name()
    else:
        ARABIC_FONT = 'DejaVu Sans'
    return ARABIC_FONT


ARABIC_FONT = _init_fonts()
logger.info(f'Fonts initialised. Arabic chart font: {ARABIC_FONT}')

# ─────────────────────────────────────────────────────────────
# 2. ARABIC TEXT HELPERS
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
    if txt is None:
        return '-'
    return rtl(str(txt))


def short_text(s, n=40):
    s = str(s or '-')
    return s if len(s) <= n else s[:n - 1] + '…'


def safe(info, key, default=None):
    v = info.get(key)
    return default if v is None else v

# ─────────────────────────────────────────────────────────────
# 3. COMPANY NAMES
# ─────────────────────────────────────────────────────────────
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


def get_name(ticker: str) -> str:
    return COMPANY_NAMES.get(ticker, ticker)

# ─────────────────────────────────────────────────────────────
# 4. PDF THEME CONSTANTS
# ─────────────────────────────────────────────────────────────
NAVY_HEX, TEAL_HEX, GREEN_HEX = '#1B2A4A', '#00897B', '#4CAF50'
RED_HEX, ORANGE_HEX, LGRAY_HEX = '#E53935', '#FF9800', '#F0F3F7'
DGRAY_HEX, TXTDARK_HEX, WHITE_HEX = '#5A6272', '#1A1A2E', '#FFFFFF'
BLUE_HEX = '#2196F3'

NAVY, TEAL, GREEN = HexColor(NAVY_HEX), HexColor(TEAL_HEX), HexColor(GREEN_HEX)
RED, ORANGE, LGRAY = HexColor(RED_HEX), HexColor(ORANGE_HEX), HexColor(LGRAY_HEX)
DGRAY, TXTDARK, WHITE = HexColor(DGRAY_HEX), HexColor(TXTDARK_HEX), HexColor(WHITE_HEX)
BLUE = HexColor(BLUE_HEX)

PAGE_W, PAGE_H = A4
MG = 18 * mm
CW = PAGE_W - 2 * MG

# ─────────────────────────────────────────────────────────────
# 5. NUMBER / PERCENT FORMATTERS
# ─────────────────────────────────────────────────────────────
def fmt_n(v, d=2):
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
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────────────────────
# 6. STOCK DATA & INDICATORS
# ─────────────────────────────────────────────────────────────
def fetch_data(ticker):
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
        return df, df2, info
    except Exception as e:
        logger.error(f"fetch_data error: {e}")
        return None, None, {}


def compute_indicators(df):
    d = df.copy()
    c, h, l, v = d['Close'], d['High'], d['Low'], d['Volume']
    d['SMA20'] = c.rolling(20, min_periods=1).mean()
    d['SMA50'] = c.rolling(50, min_periods=1).mean()
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
    bm = c.rolling(20, min_periods=1).mean()
    bs = c.rolling(20, min_periods=1).std()
    d['BB_U'], d['BB_M'], d['BB_L'] = bm + 2*bs, bm, bm - 2*bs
    bb_range = d['BB_U'] - d['BB_L']
    d['BB_P'] = np.where(bb_range != 0, (c - d['BB_L']) / bb_range, 0.5)
    l14 = l.rolling(14, min_periods=1).min()
    h14 = h.rolling(14, min_periods=1).max()
    stoch_range = h14 - l14
    d['SK'] = np.where(stoch_range != 0, 100*(c-l14)/stoch_range, 50)
    d['SD'] = pd.Series(d['SK'], index=d.index).rolling(3, min_periods=1).mean()
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    d['ATR'] = tr.rolling(14, min_periods=1).mean()
    d['OBV'] = (np.sign(c.diff()) * v).fillna(0).cumsum()
    tp = (h + l + c) / 3
    cumsum_vol = v.cumsum().replace(0, np.nan)
    d['VWAP'] = (tp * v).cumsum() / cumsum_vol
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
        ('10. ROC 12 إيجابي (زخم صاعد)', pd.notna(last['ROC12']) and float(last['ROC12']) > 0),
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


def detect_candle_patterns(df):
    patterns = []
    if len(df) < 3: return patterns
    o = df['Open'].values.astype(float); h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float);  c = df['Close'].values.astype(float)
    dates = df.index
    def body(i): return abs(c[i]-o[i])
    def candle(i): return h[i]-l[i]
    def upper_shadow(i): return h[i]-max(c[i],o[i])
    def lower_shadow(i): return min(c[i],o[i])-l[i]
    for i in range(2, len(df)):
        avg_body = np.mean([body(j) for j in range(max(0,i-5),i)]) or 1e-9
        date_lbl = dates[i].strftime('%Y-%m-%d')
        if body(i) < 0.05*candle(i) and candle(i) > 0:
            patterns.append((date_lbl,'دوجي','Doji',None)); continue
        if lower_shadow(i) > 2*body(i) and upper_shadow(i) < 0.3*body(i) and body(i) > 0:
            if c[i-1] < o[i-1]: patterns.append((date_lbl,'المطرقة','Hammer',True))
            else:                patterns.append((date_lbl,'المشنقة','Hanging Man',False))
            continue
        if upper_shadow(i) > 2*body(i) and lower_shadow(i) < 0.3*body(i) and body(i) > 0:
            if c[i-1] > o[i-1]: patterns.append((date_lbl,'النجمة الساقطة','Shooting Star',False))
            else:                patterns.append((date_lbl,'المطرقة المقلوبة','Inverted Hammer',True))
            continue
        if c[i-1]<o[i-1] and c[i]>o[i] and o[i]<c[i-1] and c[i]>o[i-1]:
            patterns.append((date_lbl,'الابتلاع الصعودي','Bullish Engulfing',True)); continue
        if c[i-1]>o[i-1] and c[i]<o[i] and o[i]>c[i-1] and c[i]<o[i-1]:
            patterns.append((date_lbl,'الابتلاع الهبوطي','Bearish Engulfing',False)); continue
        if i >= 2:
            if c[i-2]<o[i-2] and body(i-1)<0.3*body(i-2) and c[i]>o[i] and c[i]>(o[i-2]+c[i-2])/2:
                patterns.append((date_lbl,'نجمة الصباح','Morning Star',True)); continue
            if c[i-2]>o[i-2] and body(i-1)<0.3*body(i-2) and c[i]<o[i] and c[i]<(o[i-2]+c[i-2])/2:
                patterns.append((date_lbl,'نجمة المساء','Evening Star',False)); continue
            if all(c[j]>o[j] for j in [i-2,i-1,i]) and c[i-1]>c[i-2] and c[i]>c[i-1] and all(body(j)>avg_body*0.8 for j in [i-2,i-1,i]):
                patterns.append((date_lbl,'ثلاثة جنود بيض','Three White Soldiers',True)); continue
            if all(c[j]<o[j] for j in [i-2,i-1,i]) and c[i-1]<c[i-2] and c[i]<c[i-1] and all(body(j)>avg_body*0.8 for j in [i-2,i-1,i]):
                patterns.append((date_lbl,'ثلاثة غربان سوداء','Three Black Crows',False)); continue
    seen = set(); unique = []
    for p in reversed(patterns):
        if p[2] not in seen:
            seen.add(p[2]); unique.append(p)
        if len(unique) == 5: break
    return list(reversed(unique))


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
    last = d.iloc[-1]; c_price = float(last['Close']); sections = []
    ema7=float(last['EMA7']) if pd.notna(last['EMA7']) else None
    ema20=float(last['EMA20']) if pd.notna(last['EMA20']) else None
    ema50=float(last['EMA50']) if pd.notna(last['EMA50']) else None
    ema200=float(last['EMA200']) if pd.notna(last['EMA200']) else None
    sma50=float(last['SMA50']) if pd.notna(last['SMA50']) else None
    sma200=float(last['SMA200']) if pd.notna(last['SMA200']) else None
    adx=float(last['ADX']) if pd.notna(last['ADX']) else None
    pdi=float(last['PDI']) if pd.notna(last['PDI']) else None
    mdi=float(last['MDI']) if pd.notna(last['MDI']) else None
    trend_parts = []
    if all(v is not None for v in [ema7,ema20,ema50,ema200]):
        if ema7>ema20>ema50>ema200: trend_parts.append('المتوسطات المتحركة الأسية (7/20/50/200) مرتبة ترتيباً صاعداً مثالياً، مما يدل على اتجاه تصاعدي قوي ومنتظم.')
        elif c_price>ema50 and c_price>ema200: trend_parts.append('السعر يتداول فوق المتوسطَين الأسيَين 50 و200، مما يشير إلى استمرار الاتجاه الصعودي على المدى المتوسط والطويل.')
        elif c_price<ema50 and c_price<ema200: trend_parts.append('السعر يتداول دون المتوسطَين الأسيَين 50 و200، مما يعكس ضغطاً بيعياً سائداً على المدى المتوسط والطويل.')
        else: trend_parts.append('يتباين موقع السعر بالنسبة للمتوسطات المتحركة، مما يشير إلى اتجاه متذبذب يستدعي المراقبة.')
    if sma50 is not None and sma200 is not None:
        if sma50>sma200: trend_parts.append('يُسجَّل تقاطع ذهبي بين المتوسطَين البسيطَين 50 و200، وهو إشارة فنية إيجابية تاريخياً.')
        else:            trend_parts.append('يُلاحَظ تقاطع سلبي (موت) بين المتوسطَين البسيطَين 50 و200، وهو إشارة تحذيرية للمستثمرين.')
    if adx is not None and pdi is not None and mdi is not None:
        strength='قوي' if adx>25 else ('معتدل' if adx>15 else 'ضعيف')
        direction='صاعد' if pdi>mdi else 'هابط'
        trend_parts.append(f'يُشير مؤشر ADX إلى اتجاه {direction} {strength} بقيمة {adx:.1f}.')
    sections.append(('الاتجاه العام', ' '.join(trend_parts) if trend_parts else 'لا تتوفر بيانات كافية.'))
    rsi=float(last['RSI']) if pd.notna(last['RSI']) else None
    macd=float(last['MACD']) if pd.notna(last['MACD']) else None
    macd_sig=float(last['MACD_Sig']) if pd.notna(last['MACD_Sig']) else None
    macd_h=float(last['MACD_H']) if pd.notna(last['MACD_H']) else None
    roc12=float(last['ROC12']) if pd.notna(last['ROC12']) else None
    mom_parts=[]
    if rsi is not None:
        if rsi>=70: mom_parts.append(f'مؤشر RSI يقرأ {rsi:.1f} في منطقة التشبع الشرائي، مما يستوجب الحذر.')
        elif rsi<=30: mom_parts.append(f'مؤشر RSI عند {rsi:.1f} في منطقة التشبع البيعي.')
        elif rsi>=55: mom_parts.append(f'مؤشر RSI عند {rsi:.1f} في المنطقة الإيجابية.')
        else: mom_parts.append(f'مؤشر RSI محايد عند {rsi:.1f}.')
    if macd is not None and macd_sig is not None and macd_h is not None:
        if macd>macd_sig and macd_h>0: mom_parts.append('مؤشر الماكد يتداول فوق خط الإشارة مع هستوجرام إيجابي.')
        elif macd<macd_sig and macd_h<0: mom_parts.append('مؤشر الماكد يتداول دون خط الإشارة مع هستوجرام سلبي.')
        else: mom_parts.append('مؤشر الماكد في مرحلة تقاطع.')
    sections.append(('مؤشرات الزخم', ' '.join(mom_parts) if mom_parts else 'لا تتوفر بيانات كافية.'))
    bb_u=float(last['BB_U']) if pd.notna(last['BB_U']) else None
    bb_m=float(last['BB_M']) if pd.notna(last['BB_M']) else None
    bb_l=float(last['BB_L']) if pd.notna(last['BB_L']) else None
    bb_p=float(last['BB_P']) if pd.notna(last['BB_P']) else None
    atr=float(last['ATR']) if pd.notna(last['ATR']) else None
    vol_parts=[]
    if all(v is not None for v in [bb_u,bb_m,bb_l,bb_p]):
        if c_price>bb_u: vol_parts.append(f'السعر اخترق الحد العلوي لنطاق بولنجر ({bb_u:.2f}).')
        elif c_price<bb_l: vol_parts.append(f'السعر دون الحد السفلي لنطاق بولنجر ({bb_l:.2f}).')
        else: vol_parts.append(f'السعر داخل نطاق بولنجر عند الموقع {bb_p*100:.0f}%.')
    if atr is not None:
        atr_pct=(atr/c_price)*100
        vol_parts.append(f'مؤشر ATR يُسجّل {atr:.2f} ({atr_pct:.1f}% من السعر).')
    sections.append(('التذبذب ونطاق بولنجر', ' '.join(vol_parts) if vol_parts else 'لا تتوفر بيانات كافية.'))
    volume_parts=[]
    obv=float(last['OBV']) if pd.notna(last['OBV']) else None
    bull_vol=float(last['Bull_Volume']) if pd.notna(last['Bull_Volume']) else None
    bear_vol=float(last['Bear_Volume']) if pd.notna(last['Bear_Volume']) else None
    vol_now=float(last['Volume'])
    vwap=float(last['VWAP']) if pd.notna(last['VWAP']) else None
    vol_avg=d['Volume'].rolling(20, min_periods=1).mean().iloc[-1]
    if pd.notna(vol_avg) and vol_avg>0:
        vr=vol_now/vol_avg
        if vr>1.5: volume_parts.append(f'حجم التداول مرتفع بنسبة {vr:.1f}x.')
        elif vr<0.5: volume_parts.append(f'حجم التداول منخفض ({vr:.1f}x).')
        else: volume_parts.append(f'حجم التداول طبيعي ({vr:.1f}x).')
    if vwap is not None:
        volume_parts.append(f'VWAP عند {vwap:.2f} والسعر {"فوقه" if c_price>vwap else "دونه"}.')
    sections.append(('تحليل الحجم', ' '.join(volume_parts) if volume_parts else 'لا تتوفر بيانات كافية.'))
    tenkan=float(last['Tenkan']) if pd.notna(last['Tenkan']) else None
    kijun=float(last['Kijun']) if pd.notna(last['Kijun']) else None
    senk_a=float(last['Senkou_A']) if pd.notna(last['Senkou_A']) else None
    senk_b=float(last['Senkou_B']) if pd.notna(last['Senkou_B']) else None
    sar=float(last['SAR']) if pd.notna(last['SAR']) else None
    adv_parts=[]
    if tenkan is not None and kijun is not None:
        adv_parts.append(f'خط التنكن ({tenkan:.2f}) {"أعلى" if tenkan>kijun else "أدنى"} من خط الكيجن ({kijun:.2f}).')
    if senk_a is not None and senk_b is not None:
        kumo_top=max(senk_a,senk_b); kumo_bot=min(senk_a,senk_b)
        if c_price>kumo_top: adv_parts.append(f'السعر فوق السحابة.')
        elif c_price<kumo_bot: adv_parts.append(f'السعر دون السحابة.')
        else: adv_parts.append('السعر داخل السحابة.')
    if sar is not None:
        adv_parts.append(f'SAR عند {sar:.2f} {"دون" if c_price>sar else "فوق"} السعر.')
    sections.append(('مؤشرات متقدمة (إيشيموكو / SAR)', ' '.join(adv_parts) if adv_parts else 'لا تتوفر بيانات كافية.'))
    atr_val=float(last['ATR']) if pd.notna(last['ATR']) else None
    sr_parts=[]
    if sup:
        for i,s in enumerate(sup[:5]):
            gap=abs(c_price-s)/c_price*100; side='دون' if s<c_price else 'فوق'
            rank=['الأول','الثاني','الثالث','الرابع','الخامس'][i]
            sr_parts.append(f'الدعم {rank}: {s:.2f} ({gap:.1f}% {side} السعر).')
    if res:
        for i,r in enumerate(res[:5]):
            gap=abs(r-c_price)/c_price*100; side='فوق' if r>c_price else 'دون'
            rank=['الأولى','الثانية','الثالثة','الرابعة','الخامسة'][i]
            sr_parts.append(f'المقاومة {rank}: {r:.2f} ({gap:.1f}% {side} السعر).')
    if atr_val is not None:
        t1_up=c_price+1.0*atr_val; t2_up=c_price+2.0*atr_val
        t1_dn=c_price-1.0*atr_val; t2_dn=c_price-2.0*atr_val
        sr_parts.append(f'ATR ({atr_val:.2f}): أهداف صعود {t1_up:.2f} / {t2_up:.2f}. توقف {t1_dn:.2f} / {t2_dn:.2f}.')
    sections.append(('مستويات الدعم والمقاومة والأهداف السعرية', ' '.join(sr_parts) if sr_parts else 'لا مستويات واضحة.'))
    rec_txt,_=recommendation(score)
    if score>=16: outlook=f'الصورة الفنية إيجابية بدرجة عالية ({score}/20).'
    elif score>=12: outlook=f'القراءة الفنية تميل نحو الإيجابية ({score}/20).'
    elif score<=4: outlook=f'الصورة الفنية سلبية ({score}/20).'
    elif score<=8: outlook=f'القراءة الفنية تميل نحو السلبية ({score}/20).'
    else: outlook=f'قراءة محايدة ({score}/20).'
    sections.append(('الخلاصة الفنية', outlook))
    if divergences:
        div_parts=[f'{ind}: {ar_type}' for ind,ar_type,en_type in divergences]
        sections.append(('التباعد بين السعر والمؤشرات', ' '.join(div_parts)))
    else:
        sections.append(('التباعد بين السعر والمؤشرات','لا يُلاحَظ تباعد واضح.'))
    cci_val=float(last['CCI']) if pd.notna(last['CCI']) else None
    willr=float(last['WILLR']) if pd.notna(last['WILLR']) else None
    osc_parts=[]
    if cci_val is not None:
        if cci_val>100: osc_parts.append(f'CCI: {cci_val:.0f} زخم صعودي قوي.')
        elif cci_val<-100: osc_parts.append(f'CCI: {cci_val:.0f} زخم هبوطي.')
        else: osc_parts.append(f'CCI: {cci_val:.0f} في النطاق المحايد.')
    if willr is not None:
        if willr>-20: osc_parts.append(f'Williams %R: {willr:.0f} قريب من التشبع الشرائي.')
        elif willr<-80: osc_parts.append(f'Williams %R: {willr:.0f} في التشبع البيعي.')
        else: osc_parts.append(f'Williams %R: {willr:.0f} في المنطقة المحايدة.')
    if osc_parts:
        sections.append(('مؤشرات CCI وWilliams %R', ' '.join(osc_parts)))
    return sections

# ─────────────────────────────────────────────────────────────
# 7. CHART FUNCTIONS
# ─────────────────────────────────────────────────────────────
def _draw_sr_lines(ax, sup, res, xmax, d_ind=None, pivots=None):
    sup_color='#1565C0'; res_color='#B71C1C'
    ema_colors={'EMA20':('#00897B','EMA 20'),'EMA50':('#F57F17','EMA 50'),'EMA100':('#6A1B9A','EMA 100'),'EMA200':('#C62828','EMA 200')}
    all_levels=list(sup)+list(res)
    if not all_levels: return
    price_range=max(all_levels)-min(all_levels) if len(all_levels)>1 else max(all_levels)*0.1
    min_gap=max(price_range*0.018,1e-6)
    placed=[]
    def place_label(ideal_y):
        for delta in [0,min_gap,-min_gap,2*min_gap,-2*min_gap,3*min_gap,-3*min_gap,4*min_gap,-4*min_gap,5*min_gap,-5*min_gap]:
            c=ideal_y+delta
            if all(abs(c-py)>=min_gap for py in placed):
                placed.append(c); return c
        y=(max(placed)+min_gap) if placed else ideal_y; placed.append(y); return y
    label_x=xmax-0.5; text_x=xmax+0.8
    if pivots:
        for bx,by in pivots.get('highs',[]):
            ax.plot(bx,by,'v',color='#B71C1C',markersize=5,alpha=0.75,zorder=6,markeredgewidth=0)
        for bx,by in pivots.get('lows',[]):
            ax.plot(bx,by,'^',color='#1565C0',markersize=5,alpha=0.75,zorder=6,markeredgewidth=0)
    if d_ind is not None:
        for col,(clr,lbl) in ema_colors.items():
            if col in d_ind.columns:
                val=d_ind[col].iloc[-1]
                if pd.notna(val):
                    ev=float(val); ax.axhline(ev,color=clr,lw=0.9,ls='-.',alpha=0.55,zorder=4)
                    ly=place_label(ev)
                    ax.annotate(lbl,xy=(label_x,ev),xytext=(text_x,ly),fontsize=6.2,color=clr,ha='left',va='center',fontproperties=MPL_FONT_PROP,arrowprops=dict(arrowstyle='-',color=clr,lw=0.6,connectionstyle='arc3,rad=0.0'),bbox=dict(boxstyle='round,pad=0.18',facecolor='white',edgecolor=clr,alpha=0.88,linewidth=0.6),clip_on=False,zorder=7)
    for i,s in enumerate(sup):
        ax.axhline(s,color=sup_color,lw=0.7,ls='--',alpha=0.70,zorder=5); ly=place_label(s)
        ax.annotate(rtl(f'دعم {i+1}   {s:.2f}'),xy=(label_x,s),xytext=(text_x,ly),fontsize=6.8,color=sup_color,ha='left',va='center',fontproperties=MPL_FONT_PROP,arrowprops=dict(arrowstyle='-',color=sup_color,lw=0.7,connectionstyle='arc3,rad=0.0'),bbox=dict(boxstyle='round,pad=0.22',facecolor='#E8F4FD',edgecolor=sup_color,alpha=0.92,linewidth=0.7),clip_on=False,zorder=8)
    for i,r in enumerate(res):
        ax.axhline(r,color=res_color,lw=0.7,ls='--',alpha=0.70,zorder=5); ly=place_label(r)
        ax.annotate(rtl(f'مقاومة {i+1}   {r:.2f}'),xy=(label_x,r),xytext=(text_x,ly),fontsize=6.8,color=res_color,ha='left',va='center',fontproperties=MPL_FONT_PROP,arrowprops=dict(arrowstyle='-',color=res_color,lw=0.7,connectionstyle='arc3,rad=0.0'),bbox=dict(boxstyle='round,pad=0.22',facecolor='#FDEDED',edgecolor=res_color,alpha=0.92,linewidth=0.7),clip_on=False,zorder=8)


def _get_pivots(d, order=5):
    d=d.tail(180).copy().reset_index(drop=True)
    h=d['High'].values.astype(float); l=d['Low'].values.astype(float)
    ph=argrelextrema(h,np.greater_equal,order=order)[0]; pl=argrelextrema(l,np.less_equal,order=order)[0]
    return {'highs':[(int(i),round(h[i],4)) for i in ph],'lows':[(int(i),round(l[i],4)) for i in pl]}


def make_price_chart(d, sup=None, res=None):
    sup=sup or []; res=res or []; d=d.tail(180).copy()
    p=d[['Open','High','Low','Close','Volume']].copy()
    aps,labels=[],[]
    for col,clr,lbl in [('SMA20',BLUE_HEX,'SMA 20'),('SMA50',ORANGE_HEX,'SMA 50'),('SMA200','#E91E63','SMA 200')]:
        if col in d and d[col].notna().sum()>10:
            aps.append(mpf.make_addplot(d[col],color=clr,width=1.2)); labels.append(lbl)
    mc=mpf.make_marketcolors(up='#26a69a',down='#ef5350',edge='inherit',wick='inherit',volume={'up':'#80cbc4','down':'#ef9a9a'})
    st=mpf.make_mpf_style(marketcolors=mc,gridstyle=':',gridcolor='#dddddd',rc={'axes.facecolor':'#FAFAFA'})
    plot_kwargs=dict(type='candle',style=st,volume=True,figsize=(14,7),returnfig=True,warn_too_much_data=9999)
    if aps: plot_kwargs['addplot']=aps
    fig,ax=mpf.plot(p,**plot_kwargs); main_ax=ax[0]
    if labels: main_ax.legend(labels,loc='upper left',fontsize=8,prop=MPL_FONT_PROP)
    xmax=len(d); pivots=_get_pivots(d,order=5)
    _draw_sr_lines(main_ax,sup,res,xmax,d_ind=d,pivots=pivots)
    fig.subplots_adjust(right=0.80); return chart_bytes(fig)


def make_ema_chart(d, sup=None, res=None):
    sup=sup or []; res=res or []; d=d.tail(180).copy()
    p=d[['Open','High','Low','Close','Volume']].copy()
    aps,labels=[],[]
    for col,clr,lbl in [('EMA20',BLUE_HEX,'EMA 20'),('EMA50',ORANGE_HEX,'EMA 50'),('EMA100','#6A1B9A','EMA 100'),('EMA200','#E91E63','EMA 200')]:
        if col in d and d[col].notna().sum()>10:
            aps.append(mpf.make_addplot(d[col],color=clr,width=1.2)); labels.append(lbl)
    mc=mpf.make_marketcolors(up='#26a69a',down='#ef5350',edge='inherit',wick='inherit',volume={'up':'#80cbc4','down':'#ef9a9a'})
    st=mpf.make_mpf_style(marketcolors=mc,gridstyle=':',gridcolor='#dddddd',rc={'axes.facecolor':'#FAFAFA'})
    plot_kwargs=dict(type='candle',style=st,volume=True,figsize=(14,7),returnfig=True,warn_too_much_data=9999)
    if aps: plot_kwargs['addplot']=aps
    fig,ax=mpf.plot(p,**plot_kwargs); main_ax=ax[0]
    if labels: main_ax.legend(labels,loc='upper left',fontsize=8,prop=MPL_FONT_PROP)
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
        ax_gauge.text(lx,ly,label,ha='center',va='center',fontsize=8.5,fontproperties=MPL_FONT_PROP,color='white',fontweight='bold',rotation=rotation,zorder=4)
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
    ax_button.add_patch(circle); ax_button.text(0.5,0.5,rtl(rec),ha='center',va='center',fontsize=13,fontproperties=MPL_FONT_PROP_BOLD,color='white',transform=ax_button.transAxes)
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


def make_candle_pattern_chart(d, patterns):
    d=d.tail(60).copy(); p=d[['Open','High','Low','Close','Volume']].copy()
    mc=mpf.make_marketcolors(up='#26a69a',down='#ef5350',edge='inherit',wick='inherit',volume={'up':'#80cbc4','down':'#ef9a9a'})
    st=mpf.make_mpf_style(marketcolors=mc,gridstyle=':',gridcolor='#dddddd',rc={'axes.facecolor':'#FAFAFA'})
    fig,axlist=mpf.plot(p,type='candle',style=st,volume=False,figsize=(16,7),returnfig=True,warn_too_much_data=9999); ax=axlist[0]
    date_to_pos={str(dt.date()):i for i,dt in enumerate(d.index)}
    resolved=[]
    for date_lbl,ar_name,en_name,bullish in patterns:
        pos=date_to_pos.get(date_lbl)
        if pos is None: continue
        row_d=d.iloc[pos]; resolved.append({'pos':pos,'high':float(row_d['High']),'low':float(row_d['Low']),'name':ar_name,'bullish':bullish})
    if not resolved: plt.tight_layout(); return chart_bytes(fig)
    ylo,yhi=ax.get_ylim(); xlo,xhi=ax.get_xlim(); xspan=xhi-xlo; yspan=yhi-ylo
    col_right_x=xhi+xspan*0.06; col_left_x=xlo-xspan*0.06
    right_anns=[ann for i,ann in enumerate(resolved) if i%2==0]; left_anns=[ann for i,ann in enumerate(resolved) if i%2==1]
    def make_label_ys(count):
        if count==0: return []
        step=yspan/(count+1); return sorted([ylo+step*(k+1) for k in range(count)],reverse=True)
    right_ys=make_label_ys(len(right_anns)); left_ys=make_label_ys(len(left_anns))
    def draw_ann(ann,label_x,label_y,align,rad):
        pos=ann['pos']; bullish=ann['bullish']
        color='#1B5E20' if bullish is True else ('#B71C1C' if bullish is False else '#E65100')
        label=rtl(ann['name']); anchor_y=ann['high'] if bullish is not False else ann['low']
        ax.annotate(label,xy=(pos,anchor_y),xytext=(label_x,label_y),fontsize=8.5,color=color,ha=align,va='center',fontproperties=MPL_FONT_PROP,arrowprops=dict(arrowstyle='-|>',color=color,lw=1.1,mutation_scale=9,connectionstyle=f'arc3,rad={rad}'),bbox=dict(boxstyle='round,pad=0.32',facecolor='white',edgecolor=color,alpha=0.95,linewidth=1.0),clip_on=False,zorder=10)
    arcs_r=[0.25,0.15,0.35,0.10,0.20]; arcs_l=[-0.25,-0.15,-0.35,-0.10,-0.20]
    for k,ann in enumerate(right_anns): draw_ann(ann,col_right_x,right_ys[k],'left',arcs_r[k%len(arcs_r)])
    for k,ann in enumerate(left_anns): draw_ann(ann,col_left_x,left_ys[k],'right',arcs_l[k%len(arcs_l)])
    ax.set_xlim(col_left_x-xspan*0.32,col_right_x+xspan*0.32); fig.subplots_adjust(left=0.12,right=0.88)
    return chart_bytes(fig)


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

# ─────────────────────────────────────────────────────────────
# 8. PDF REPORT CLASS
# ─────────────────────────────────────────────────────────────
PERIOD_AR = {'1 Day':'يوم','1 Week':'أسبوع','1 Month':'شهر','3 Months':'3 أشهر','6 Months':'6 أشهر','1 Year':'سنة','YTD':'منذ بداية السنة'}
RISK_AR = {'Daily Volatility':'التذبذب اليومي','Annual Volatility':'التذبذب السنوي','Sharpe Ratio':'نسبة شارب','Sortino Ratio':'نسبة سورتينو','Max Drawdown':'أقصى تراجع','Avg Daily Return':'متوسط العائد اليومي','Best Day':'أفضل يوم','Worst Day':'أسوأ يوم'}


class Report:
    def __init__(self, path, ticker, info, display_ticker=None):
        self.c = pdfcanvas.Canvas(path, pagesize=A4)
        self.c.setTitle(f'{ticker} Arabic Stock Report')
        self.tk = ticker; self.display_tk = display_ticker or ticker; self.info = info; self.pn = 0

    def _font(self, bold=False, size=10):
        self.c.setFont(AR_FONT_BOLD if bold else AR_FONT, size)

    def _bar(self, title):
        self.pn += 1; c = self.c
        c.setFillColor(NAVY); c.rect(0, PAGE_H-34*mm, PAGE_W, 34*mm, fill=1, stroke=0)
        c.setFillColor(TEAL); c.rect(0, PAGE_H-36*mm, PAGE_W, 2*mm, fill=1, stroke=0)
        c.setFillColor(WHITE); self._font(True, 15); c.drawRightString(PAGE_W-MG, PAGE_H-18*mm, rtl(title))
        self._font(False, 9); c.drawString(MG, PAGE_H-14*mm, self.display_tk); c.drawString(MG, PAGE_H-22*mm, datetime.now().strftime('%Y-%m-%d'))

    def _foot(self):
        c = self.c; c.setFillColor(LGRAY); c.rect(0, 0, PAGE_W, 10*mm, fill=1, stroke=0)
        c.setFillColor(DGRAY); self._font(False, 7); c.drawString(MG, 4*mm, self.display_tk); c.drawRightString(PAGE_W-MG, 4*mm, rtl(f'الصفحة {self.pn}'))

    def _stitle(self, y, t):
        c = self.c; c.setFillColor(NAVY); self._font(True, 11); c.drawRightString(PAGE_W-MG, y, rtl(t))
        c.setStrokeColor(TEAL); c.setLineWidth(1.4); c.line(MG, y-4, PAGE_W-MG, y-4); return y-18

    def _box(self, x, y, bw, bh, lbl, val, clr=None):
        c = self.c; c.setFillColor(LGRAY); c.roundRect(x, y, bw, bh, 5, fill=1, stroke=0)
        c.setFillColor(DGRAY); self._font(False, 8); c.drawCentredString(x+bw/2, y+bh-14, tx(lbl))
        if isinstance(val, (int, float)): val_str, val_clr = fmt_n(val)
        else: val_str = str(val) if val is not None else '-'; val_clr = None
        if val_clr: fill_color = val_clr
        elif clr:
            fill_color = HexColor(clr) if isinstance(clr, str) else clr
        else: fill_color = NAVY
        c.setFillColor(fill_color); self._font(True, 12.5); c.drawCentredString(x+bw/2, y+8, tx(val_str))

    def _table(self, y, rows, cw_list, sig_mode=False, score_mode=False):
        c = self.c; rh = 16; total_w = sum(cw_list)
        for i, row in enumerate(rows):
            ry = y - i*rh
            if i==0: bg,fg,is_bold = NAVY,WHITE,True
            elif i%2==1: bg,fg,is_bold = LGRAY,TXTDARK,False
            else: bg,fg,is_bold = WHITE,TXTDARK,False
            c.setFillColor(bg); c.rect(MG, ry-4, total_w, rh, fill=1, stroke=0)
            sx=MG; c.setStrokeColor(HexColor('#D8DEE9')); c.setLineWidth(0.5)
            for col_w in cw_list[:-1]:
                sx+=col_w; c.line(sx, ry-4, sx, ry-4+rh)
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
                c.setFillColor(fill); self._font(is_cell_bold, 8); c.drawRightString(cx+cw_list[j]-4, ry+2, tx(cell)); cx+=cw_list[j]
        return y - len(rows)*rh - 6

    def _img(self, y, buf, max_h):
        c = self.c; buf.seek(0); img = ImageReader(buf); iw,ih = img.getSize()
        ratio=ih/iw; dw=CW; dh=dw*ratio
        if dh>max_h: dh=max_h; dw=dh/ratio
        x=MG+(CW-dw)/2; c.drawImage(img, x, y-dh-2, dw, dh); return y-dh-8

    def _wrap_arabic_text(self, text, max_width, font_size):
        words=str(text).split(); lines=[]; current_line=[]
        for word in words:
            test_shaped=rtl(' '.join(current_line+[word]))
            line_width=self.c.stringWidth(test_shaped, AR_FONT, font_size)
            if line_width<=max_width: current_line.append(word)
            else:
                if current_line: lines.append(rtl(' '.join(current_line)))
                current_line=[word]
        if current_line: lines.append(rtl(' '.join(current_line)))
        return lines

    def cover(self, price, chg, info, rec_txt, rec_color, score):
        c=self.c; self.pn+=1
        c.setFillColor(NAVY); c.rect(0, PAGE_H-78*mm, PAGE_W, 78*mm, fill=1, stroke=0)
        c.setFillColor(TEAL); c.rect(0, PAGE_H-80*mm, PAGE_W, 2*mm, fill=1, stroke=0)
        company_name=safe(info,'longName',safe(info,'shortName',self.tk)) or self.tk
        company_name=short_text(company_name, 42)
        c.setFillColor(WHITE); self._font(True,23); c.drawRightString(PAGE_W-MG, PAGE_H-22*mm, rtl('تقرير تحليل السهم'))
        self._font(True,18); c.drawRightString(PAGE_W-MG, PAGE_H-39*mm, tx(company_name))
        self._font(False,11); c.drawRightString(PAGE_W-MG, PAGE_H-52*mm, tx(f'{self.display_tk} | {safe(info,"exchange","-")}'))
        self._font(False,9); c.drawString(MG, PAGE_H-18*mm, datetime.now().strftime('%Y-%m-%d'))
        c.setFillColor(HexColor(rec_color)); c.roundRect(MG, PAGE_H-58*mm, 60*mm, 14*mm, 8, fill=1, stroke=0)
        c.setFillColor(WHITE); self._font(True,11); c.drawCentredString(MG+30*mm, PAGE_H-53*mm, rtl(rec_txt))
        self._font(False,8); c.drawCentredString(MG+30*mm, PAGE_H-57*mm, rtl(f'النتيجة {score}/20'))
        col_bw=CW/3-4*mm; col_bh=20*mm; col_gap=6*mm
        x1=MG; x2=MG+col_bw+col_gap; x3=MG+2*(col_bw+col_gap)
        y1=PAGE_H-112*mm; y2=y1-col_bh-col_gap; y3=y2-col_bh-col_gap
        self._box(x1,y1,col_bw,col_bh,'السعر الحالي',price)
        self._box(x2,y1,col_bw,col_bh,'التغير اليومي',f'{chg:+.2f}%',clr=GREEN_HEX if chg>=0 else RED_HEX)
        self._box(x3,y1,col_bw,col_bh,'القيمة السوقية',fmt_n(safe(info,'marketCap'))[0])
        pe=safe(info,'trailingPE'); eps=safe(info,'trailingEps'); dy=safe(info,'dividendYield')
        self._box(x1,y2,col_bw,col_bh,'مكرر الربحية',f'{float(pe):.2f}' if pe else '-')
        eps_val=float(eps) if eps else None; eps_str=f'{eps_val:.2f}' if eps_val is not None else '-'
        eps_color=GREEN_HEX if (eps_val is not None and eps_val>=0) else RED_HEX if eps_val is not None else None
        self._box(x2,y2,col_bw,col_bh,'ربحية السهم',eps_str,clr=eps_color)
        self._box(x3,y2,col_bw,col_bh,'عائد التوزيعات',fmt_p(dy)[0] if dy else '-')
        pb=safe(info,'priceToBook'); roe=safe(info,'returnOnEquity'); beta=safe(info,'beta')
        self._box(x1,y3,col_bw,col_bh,'مضاعف القيمة الدفترية',f'{float(pb):.2f}' if pb else '-')
        self._box(x2,y3,col_bw,col_bh,'العائد على حقوق المساهمين',fmt_p(roe)[0] if roe else '-')
        self._box(x3,y3,col_bw,col_bh,'بيتا',f'{float(beta):.2f}' if beta else '-')
        w52h=safe(info,'fiftyTwoWeekHigh'); w52l=safe(info,'fiftyTwoWeekLow')
        if w52h and w52l and float(w52h)!=float(w52l):
            bar_y=y3-14*mm; c.setFillColor(NAVY); self._font(True,8); c.drawRightString(PAGE_W-MG, bar_y+2, rtl('نطاق 52 أسبوع'))
            bar_x=MG; bar_w=CW; bar_h=5*mm; bar_y2=bar_y-bar_h-2
            c.setFillColor(HexColor('#ECEFF1')); c.roundRect(bar_x, bar_y2, bar_w, bar_h, 3, fill=1, stroke=0)
            pos=(price-float(w52l))/(float(w52h)-float(w52l)); pos=max(0.0,min(1.0,pos)); fill_w=bar_w*pos
            fill_c=HexColor(RED_HEX) if pos<0.35 else (HexColor(GREEN_HEX) if pos>0.65 else HexColor(ORANGE_HEX))
            c.setFillColor(fill_c); c.roundRect(bar_x, bar_y2, max(fill_w,4), bar_h, 3, fill=1, stroke=0)
            c.setFillColor(DGRAY); self._font(False,7); c.drawString(bar_x, bar_y2-9, f'{float(w52l):.2f}'); c.drawRightString(bar_x+bar_w, bar_y2-9, f'{float(w52h):.2f}')
            c.setFillColor(NAVY); self._font(True,7); c.drawCentredString(bar_x+bar_w/2, bar_y2-9, f'{pos*100:.0f}%'); y3=bar_y2-14*mm
        else: y3=y3-8*mm
        y=y3-6*mm; y=self._stitle(y,'معلومات الشركة')
        sector=short_text(safe(info,'sector','-') or '-',26); industry=short_text(safe(info,'industry','-') or '-',26)
        items=[('القطاع',sector),('الصناعة',industry),('العملة',safe(info,'currency','SAR') or '-'),('متوسط الحجم',fmt_n(safe(info,'averageVolume'),d=0)[0]),('أعلى 52 أسبوع',fmt_n(safe(info,'fiftyTwoWeekHigh'))[0]),('أدنى 52 أسبوع',fmt_n(safe(info,'fiftyTwoWeekLow'))[0]),('الأسهم القائمة',fmt_n(safe(info,'sharesOutstanding'),d=0)[0]),('الأسهم الحرة',fmt_n(safe(info,'floatShares'),d=0)[0])]
        half=CW/2
        for idx,(lbl,val) in enumerate(items):
            col=idx%2; row=idx//2; xr=MG+(col+1)*half-5; yy=y-row*18
            c.setFillColor(NAVY); self._font(True,8.5); c.drawRightString(xr, yy, rtl(f'{lbl}:'))
            c.setFillColor(TXTDARK); self._font(False,8.5); c.drawRightString(xr-85, yy, tx(val))
        c.setFillColor(DGRAY); self._font(False,7); c.drawCentredString(PAGE_W/2, 14*mm, rtl('هذا التقرير لأغراض معلوماتية فقط وليس توصية استثمارية.'))
        self._foot(); c.showPage()

    def price_page(self, img):
        self._bar('حركة السعر والمتوسطات المتحركة'); self._foot()
        y=PAGE_H-44*mm; y=self._stitle(y,'الرسم الشمعي مع SMA 20 / 50 / 200 وحجم التداول'); self._img(y, img, 178*mm)

    def ema_page(self, img):
        self._bar('حركة السعر والمتوسطات المتحركة (EMA)'); self._foot()
        y=PAGE_H-160*mm; y=self._stitle(y,'الرسم الشمعي مع EMA 20 / 50 / 200 وحجم التداول'); self._img(y, img, 178*mm); self.c.showPage()

    def tech_page(self, tech_img, bb_img):
        self._bar('المؤشرات الفنية'); self._foot()
        y=PAGE_H-44*mm; y=self._stitle(y,'RSI و MACD و ROC 12'); y=self._img(y, tech_img, 113*mm)
        y=self._stitle(y,'نطاقات بولنجر'); self._img(y, bb_img, 113*mm); self.c.showPage()

    def perf_page(self, pers, risk, dd_img, vol_img, score_criteria, total_score):
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
                status=rtl('نعم (+1)') if pt==1 else rtl('لا (0)'); rows_score.append([str(pt),status,rtl(short_text(lbl,35))])
            rows_score.append([str(total_score),rtl('من 20'),rtl('الإجمالي')]); self._table(y_table,rows_score,[CW*0.15,CW*0.30,CW*0.55],score_mode=True)
        else:
            self.c.showPage(); self._bar('جدول نقاط النتيجة'); self._foot()
            y_table=PAGE_H-44*mm; y_table=self._stitle(y_table,f'جدول نقاط النتيجة ({total_score}/20)')
            rows_score=[['النقاط','الحالة','البند']]
            for lbl,(symbol,pt) in score_criteria.items():
                status=rtl('نعم (+1)') if pt==1 else rtl('لا (0)'); rows_score.append([str(pt),status,rtl(short_text(lbl,35))])
            rows_score.append([str(total_score),rtl('من 20'),rtl('الإجمالي')]); self._table(y_table,rows_score,[CW*0.15,CW*0.30,CW*0.55],score_mode=True)
        self.c.showPage()

    def fund_page(self, info):
        self._bar('التحليل الأساسي'); self._foot()
        y=PAGE_H-44*mm; y=self._stitle(y,'التقييم')
        val_items=[(safe(info,'trailingPE','-'),'Trailing P/E'),(safe(info,'forwardPE','-'),'Forward P/E'),(safe(info,'priceToBook','-'),'Price / Book'),(safe(info,'priceToSalesTrailing12Months','-'),'Price / Sales'),(safe(info,'enterpriseToEbitda','-'),'EV / EBITDA'),(safe(info,'enterpriseToRevenue','-'),'EV / Revenue'),(safe(info,'pegRatio','-'),'PEG Ratio')]
        val_rows=[['القيمة','البند']]
        for v,lbl in val_items:
            val_rows.append([f'{v:.2f}' if isinstance(v,(int,float)) else str(v),lbl])
        y=self._table(y,val_rows,[CW*0.40,CW*0.60]); y=self._stitle(y,'الربحية')
        prof_rows=[['القيمة','البند'],[fmt_n(safe(info,'totalRevenue'))[0],'الإيرادات'],[fmt_n(safe(info,'netIncomeToCommon'))[0],'صافي الدخل'],[fmt_n(safe(info,'ebitda'))[0],'EBITDA'],[fmt_p(safe(info,'grossMargins'))[0],'هامش إجمالي'],[fmt_p(safe(info,'operatingMargins'))[0],'هامش تشغيلي'],[fmt_p(safe(info,'profitMargins'))[0],'هامش صافي'],[fmt_p(safe(info,'returnOnEquity'))[0],'ROE'],[fmt_p(safe(info,'returnOnAssets'))[0],'ROA']]
        y=self._table(y,prof_rows,[CW*0.40,CW*0.60]); y=self._stitle(y,'المركز المالي')
        fin_items=[(fmt_n(safe(info,'totalCash'))[0],'إجمالي النقد'),(fmt_n(safe(info,'totalDebt'))[0],'إجمالي الدين'),(safe(info,'debtToEquity','-'),'Debt / Equity'),(safe(info,'currentRatio','-'),'Current Ratio'),(safe(info,'quickRatio','-'),'Quick Ratio'),(fmt_n(safe(info,'bookValue'))[0],'القيمة الدفترية للسهم')]
        fin_rows=[['القيمة','البند']]
        for v,lbl in fin_items:
            fin_rows.append([f'{v:.2f}' if isinstance(v,(int,float)) else str(v),lbl])
        y=self._table(y,fin_rows,[CW*0.40,CW*0.60])
        if y<60*mm: self.c.showPage(); self._bar('التحليل الأساسي (تابع)'); self._foot(); y=PAGE_H-44*mm
        y=self._stitle(y,'التوزيعات والأسهم')
        div_rows=[['القيمة','البند'],[fmt_n(safe(info,'dividendRate'))[0],'معدل التوزيع'],[fmt_p(safe(info,'dividendYield'))[0],'عائد التوزيع'],[fmt_p(safe(info,'payoutRatio'))[0],'نسبة التوزيع'],[fmt_n(safe(info,'sharesOutstanding'),d=0)[0],'الأسهم القائمة'],[fmt_n(safe(info,'floatShares'),d=0)[0],'الأسهم الحرة']]
        self._table(y,div_rows,[CW*0.40,CW*0.60]); self.c.showPage()

    def signal_page(self, gauge_img, sig, score, sup, res, d):
        self._bar('الإشارات والتحليل'); self._foot()
        c=self.c; y=PAGE_H-44*mm; y=self._stitle(y,'مؤشر التوصية'); y=self._img(y,gauge_img,50*mm)
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
        self._bar('الإيشيموكو ونماذج الشموع'); self._foot()
        y=PAGE_H-44*mm; y=self._stitle(y,'مخطط الإيشيموكو (Ichimoku Cloud)'); y=self._img(y,ichimoku_img,100*mm)
        y=self._stitle(y,'الشموع اليابانية (آخر 60 يوماً) مع النماذج المرصودة'); self._img(y,pattern_img,95*mm); self.c.showPage()

    def cci_willr_page(self, cci_img, patterns, divergences, d):
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
        self._bar('المراجعة الفنية الشاملة'); self._foot()
        c=self.c; y=PAGE_H-44*mm; line_h=13; para_gap=8; section_gap=6; font_size_body=9; max_text_width=CW-4*mm
        strip_h=10*mm; strip_y=y-strip_h
        if score>=14:   strip_fill=HexColor('#1B5E20')
        elif score>=10: strip_fill=HexColor('#1B2A4A')
        elif score>=7:  strip_fill=HexColor('#E65100')
        else:           strip_fill=HexColor('#B71C1C')
        c.setFillColor(strip_fill); c.roundRect(MG, strip_y, CW, strip_h, 5, fill=1, stroke=0)
        c.setFillColor(WHITE); self._font(True,9); c.drawCentredString(PAGE_W/2, strip_y+3, rtl(f'نتيجة التحليل الفني: {score} من أصل 20 نقطة')); y=strip_y-10*mm
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

# ─────────────────────────────────────────────────────────────
# 9. ANALYZER BOT — generate PDF in thread executor
# ─────────────────────────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=4)


def _build_report_sync(ticker_input: str):
    """Run in thread pool. Returns (pdf_bytes, summary_text) or raises."""
    ti = ticker_input.strip()
    if not ti:
        raise ValueError('رمز فارغ')

    if ti.replace('.', '').isdigit() and '.' not in ti:
        ticker = ti + '.SR'
        display_ticker = ti
    else:
        ticker = ti.upper()
        display_ticker = ticker

    df, df2, info = fetch_data(ticker)
    if df is None or len(df) < 30:
        raise ValueError(f'❌ البيانات غير كافية للرمز: {display_ticker}\nتأكد من صحة رقم الشركة.')

    d = compute_indicators(df)
    pers, risk, dd = compute_performance(df, df2)
    sig, _ = gen_signals(d)
    score_criteria, score = compute_score_criteria(d)
    sup, res = find_sr(df, d_ind=d)
    rec_txt, rec_color = recommendation(score)
    patterns = detect_candle_patterns(df)
    divergences = detect_divergences(d)
    review_sections = gen_technical_review(d, sig, score, sup, res, info, patterns=patterns, divergences=divergences)

    p_img    = make_price_chart(d, sup=sup, res=res)
    ema_img  = make_ema_chart(d, sup=sup, res=res)
    t_img    = make_tech_chart(d)
    bb_img   = make_bb_chart(d)
    dd_img   = make_dd_chart(dd)
    v_img    = make_volume_chart(d)
    g_img    = make_gauge_chart(score)
    ichi_img = make_ichimoku_chart(d)
    cpat_img = make_candle_pattern_chart(d, patterns)
    cci_img  = make_cci_willr_chart(d)

    price = float(df['Close'].iloc[-1])
    prev  = float(df['Close'].iloc[-2]) if len(df) > 1 else price
    chg   = (price / prev - 1) * 100

    pdf_buf = BytesIO()
    rpt = Report(pdf_buf, ticker, info, display_ticker)
    rpt.cover(price, chg, info, rec_txt, rec_color, score)
    rpt.price_page(p_img)
    rpt.ema_page(ema_img)
    rpt.tech_page(t_img, bb_img)
    rpt.perf_page(pers, risk, dd_img, v_img, score_criteria, score)
    rpt.fund_page(info)
    rpt.signal_page(g_img, sig, score, sup, res, d)
    rpt.ichimoku_page(ichi_img, cpat_img)
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

# ─────────────────────────────────────────────────────────────
# 10. WOLFE WAVE FUNCTIONS
# ─────────────────────────────────────────────────────────────
def find_pivots(df, order=5):
    high=df['High'].values; low=df['Low'].values
    sh=argrelextrema(high,np.greater_equal,order=order)[0]; sl=argrelextrema(low,np.less_equal,order=order)[0]
    pivots=[]
    for i in sh: pivots.append({'bar':int(i),'price':high[i],'type':'H','date':df.index[i]})
    for i in sl: pivots.append({'bar':int(i),'price':low[i],'type':'L','date':df.index[i]})
    pivots.sort(key=lambda x: x['bar']); return pivots


def get_alternating_pivots(pivots):
    if not pivots: return []
    alt=[pivots[0]]
    for p in pivots[1:]:
        if p['type']==alt[-1]['type']:
            if p['type']=='H' and p['price']>alt[-1]['price']: alt[-1]=p
            elif p['type']=='L' and p['price']<alt[-1]['price']: alt[-1]=p
        else: alt.append(p)
    return alt


def line_at(x,x1,y1,x2,y2):
    if x2==x1: return y1
    return y1+(y2-y1)*(x-x1)/(x2-x1)


def resample_ohlc(df, rule):
    return df.resample(rule).agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()


def validate_bullish(p0,p1,p2,p3,p4,p5,tol=0.03):
    v=[p['price'] for p in [p1,p2,p3,p4,p5]]; b=[p['bar'] for p in [p1,p2,p3,p4,p5]]
    if p0['type']!='H': return None
    if not (p1['type']=='L' and p2['type']=='H' and p3['type']=='L' and p4['type']=='H' and p5['type']=='L'): return None
    if p0['price']<=p2['price']: return None
    if v[2]>=v[0] or v[3]>=v[1] or v[3]<=v[0] or v[4]>=v[2]: return None
    s13=(v[2]-v[0])/(b[2]-b[0]) if b[2]!=b[0] else 0; s24=(v[3]-v[1])/(b[3]-b[1]) if b[3]!=b[1] else 0
    if s13>=0 or s24>=0 or s13>=s24: return None
    proj=line_at(b[4],b[0],v[0],b[2],v[2])
    if proj!=0 and (proj-v[4])/abs(proj)<-tol: return None
    return {'direction':'Bullish','points':[p0,p1,p2,p3,p4,p5],'entry_price':v[4],'p5_date':p5['date']}


def validate_bearish(p0,p1,p2,p3,p4,p5,tol=0.03):
    v=[p['price'] for p in [p1,p2,p3,p4,p5]]; b=[p['bar'] for p in [p1,p2,p3,p4,p5]]
    if p0['type']!='L': return None
    if not (p1['type']=='H' and p2['type']=='L' and p3['type']=='H' and p4['type']=='L' and p5['type']=='H'): return None
    if p0['price']>=p2['price']: return None
    if v[2]<=v[0] or v[3]<=v[1] or v[3]>=v[0] or v[4]<=v[2]: return None
    s13=(v[2]-v[0])/(b[2]-b[0]) if b[2]!=b[0] else 0; s24=(v[3]-v[1])/(b[3]-b[1]) if b[3]!=b[1] else 0
    if s13<=0 or s24<=0 or s13>=s24: return None
    proj=line_at(b[4],b[0],v[0],b[2],v[2])
    if proj!=0 and (v[4]-proj)/abs(proj)<-tol: return None
    return {'direction':'Bearish','points':[p0,p1,p2,p3,p4,p5],'entry_price':v[4],'p5_date':p5['date']}


def find_active_wolfe(df, max_bars_since_p5=8):
    n=len(df); best_bull=None; best_bear=None
    for order in [4,5,6,7]:
        piv=get_alternating_pivots(find_pivots(df,order=order))
        if len(piv)<6: continue
        for offset in range(min(4,len(piv)-5)):
            idx=len(piv)-6-offset
            if idx<0: break
            combo=piv[idx:idx+6]
            if n-1-combo[5]['bar']>max_bars_since_p5: continue
            r=validate_bullish(*combo)
            if r and (best_bull is None or combo[5]['bar']>best_bull['points'][5]['bar']): best_bull=r
            r=validate_bearish(*combo)
            if r and (best_bear is None or combo[5]['bar']>best_bear['points'][5]['bar']): best_bear=r
    return [x for x in [best_bull,best_bear] if x]


def plot_wolfe_chart(ticker, df, result, tf_label):
    pts=result['points']; direction=result['direction']; entry=result['entry_price']
    target=result['target_price']; is_bull=direction=='Bullish'
    company=get_name(ticker); ticker_code=ticker.split('.')[0]
    b=[p['bar'] for p in pts]; v=[p['price'] for p in pts]
    last_bar=len(df)-1; last_close=float(df['Close'].iloc[-1])
    pct=((target-entry)/entry)*100
    pad_l=max(0,b[0]-10); pad_r=min(last_bar,b[5]+30)
    df_z=df.iloc[pad_l:pad_r+1].copy(); off=pad_l; zb=[x-off for x in b]; n_z=len(df_z)
    C_W='#0D47A1' if is_bull else '#B71C1C'; C_T='#2E7D32' if is_bull else '#C62828'
    C_24='#E65100'; C_E='#6A1B9A'; C_A='#00695C' if is_bull else '#880E4F'; C_P0='#FF6F00'
    mc=mpf.make_marketcolors(up='#26A69A',down='#EF5350',edge='inherit',wick='inherit')
    sty=mpf.make_mpf_style(marketcolors=mc,gridcolor='#EEEEEE',gridstyle='-',facecolor='#FAFBFC',y_on_right=False,rc={'font.size':10,'grid.alpha':0.2})
    fig,axes=mpf.plot(df_z,type='candle',style=sty,figsize=(16,8),returnfig=True,volume=False)
    ax=axes[0]; fig.subplots_adjust(left=0.04,right=0.96,top=0.92,bottom=0.06)
    ax.plot(zb,v,color=C_W,lw=2.5,zorder=6,alpha=0.8); ax.scatter(zb,v,s=120,c='white',edgecolors=C_W,linewidths=2.5,zorder=7)
    ax.scatter([zb[0]],[v[0]],s=160,c='white',edgecolors=C_P0,linewidths=3,zorder=8)
    ext=zb[5]+8
    ax.plot([zb[1],ext],[v[1],line_at(ext+off,b[1],v[1],b[3],v[3])],color=C_W,lw=1.0,ls='--',alpha=0.3)
    ax.plot([zb[2],ext],[v[2],line_at(ext+off,b[2],v[2],b[4],v[4])],color=C_24,lw=1.0,ls='--',alpha=0.3)
    fx=np.arange(zb[1],zb[5]+1); f1=[line_at(x+off,b[1],v[1],b[3],v[3]) for x in fx]; f2=[line_at(x+off,b[2],v[2],b[4],v[4]) for x in fx]
    ax.fill_between(fx,f1,f2,alpha=0.04,color=C_W)
    tgt_end_zb=n_z+5
    ax.plot([zb[1],tgt_end_zb],[v[1],line_at(tgt_end_zb+off,b[1],v[1],b[4],v[4])],color=C_T,lw=3.0,ls='-.',alpha=0.85,zorder=5)
    z_last=min(last_bar-off,n_z-1)
    ax.plot(z_last,target,marker='D',ms=14,color=C_T,markeredgecolor='white',markeredgewidth=2,zorder=9)
    ax.axhline(y=target,color=C_T,lw=0.6,ls=':',alpha=0.25); ax.axhline(y=entry,color=C_E,lw=0.6,ls=':',alpha=0.25)
    arrow_land_zb=min(zb[5]+max(4,(z_last-zb[5])//2),n_z+3); arrow_land_price=line_at(arrow_land_zb+off,b[1],v[1],b[4],v[4])
    ax.annotate('',xy=(arrow_land_zb,arrow_land_price),xytext=(zb[5],entry),arrowprops=dict(arrowstyle='-|>',color=C_A,lw=3.0,mutation_scale=22,connectionstyle='arc3,rad=0.15' if is_bull else 'arc3,rad=-0.15'),zorder=8)
    price_range=max(v)-min(v); label_offset=price_range*0.08
    pct_y=arrow_land_price+label_offset if is_bull else arrow_land_price-label_offset
    ax.text(arrow_land_zb,pct_y,f'{pct:+.1f}%',fontsize=13,fontweight='bold',color=C_A,ha='center',va='bottom' if is_bull else 'top',fontfamily=ARABIC_FONT,bbox=dict(boxstyle='round,pad=0.3',fc='white',ec=C_A,alpha=0.9,lw=0.8),zorder=10)
    for i in range(6):
        is_low=pts[i]['type']=='L'; dt_str=pts[i]['date'].strftime('%b %d'); label_color=C_P0 if i==0 else C_W
        ax.annotate(f'P{i}  {v[i]:.2f}\n{dt_str}',xy=(zb[i],v[i]),xytext=(0,-28 if is_low else 28),textcoords='offset points',ha='center',va='top' if is_low else 'bottom',fontsize=8.5,fontweight='bold',color=label_color,bbox=dict(boxstyle='round,pad=0.3',fc='white',ec=label_color,alpha=0.9,lw=0.6),arrowprops=dict(arrowstyle='-',color=label_color,lw=0.6))
    emoji='📈' if is_bull else '📉'; direction_ar=ar('ولفي صاعد') if is_bull else ar('ولفي هابط')
    ax.set_title(f'{emoji}  {ticker_code}  |  {ar(company)}  |  {direction_ar}  |  {ar(tf_label)}',fontsize=15,fontweight='bold',pad=16,color='#212121',fontfamily=ARABIC_FONT)
    ax.set_ylabel('')
    bc='#E8F5E9' if is_bull else '#FFEBEE'; bt='#2E7D32' if is_bull else '#C62828'
    info_lines=[f"  {ar(company)}",f"  {'─'*24}",f"  {last_close:.2f}        :  {ar('الإغلاق')}",f"  {entry:.2f}        :  {ar('الموجة 5')}",f"  {target:.2f}       :  {ar('4 ← 1')}",f"  {pct:+.1f}%        :  {ar('النسبة')}",f"  {ar(tf_label)}     :  {ar('الفاصل')}"]
    ax.text(0.01,0.03,'\n'.join(info_lines),transform=ax.transAxes,fontsize=10,fontweight='bold',color=bt,fontfamily=ARABIC_FONT,verticalalignment='bottom',horizontalalignment='left',bbox=dict(boxstyle='round,pad=0.6',facecolor=bc,edgecolor=bt,alpha=0.92,lw=1.2),zorder=10)
    plt.tight_layout(); buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=150,bbox_inches='tight'); plt.close(fig); buf.seek(0); return buf


def process_ticker(ticker, period, interval, resample_rule=None):
    try:
        df=yf.Ticker(ticker).history(period=period,interval=interval)
        if df is None or len(df)<30: return ticker,[],None
        if resample_rule: df=resample_ohlc(df,resample_rule)
        if len(df)<30: return ticker,[],None
        found=find_active_wolfe(df,max_bars_since_p5=8); last_bar=len(df)-1
        for r in found:
            b1=r['points'][1]['bar']; v1=r['points'][1]['price']; b4=r['points'][4]['bar']; v4=r['points'][4]['price']
            r['target_price']=round(line_at(last_bar,b1,v1,b4,v4),2); r['last_close']=round(float(df['Close'].iloc[-1]),2)
        return ticker,found,df
    except Exception: return ticker,[],None


def scan_tickers(tickers, period, interval, resample_rule=None, max_workers=15):
    all_res={}; ohlc={}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs={pool.submit(process_ticker,t,period,interval,resample_rule):t for t in tickers}
        for f in as_completed(futs):
            tk,found,df=f.result()
            if found: all_res[tk]=found; ohlc[tk]=df
    return all_res,ohlc

# ─────────────────────────────────────────────────────────────
# 11. TICKERS & TIMEFRAME MAP
# ─────────────────────────────────────────────────────────────
TADAWUL_TICKERS = [
    '^TASI.SR','1010.SR','1020.SR','1030.SR','1050.SR','1060.SR','1080.SR','1111.SR','1120.SR',
    '1140.SR','1150.SR','1180.SR','1182.SR','1183.SR','1201.SR','1202.SR','1210.SR',
    '1211.SR','1212.SR','1213.SR','1214.SR','1301.SR','1302.SR','1303.SR','1304.SR',
    '1320.SR','1321.SR','1322.SR','1323.SR','1810.SR','1820.SR','1830.SR','1831.SR',
    '1832.SR','1833.SR','1834.SR','1835.SR','2001.SR','2010.SR','2020.SR','2030.SR',
    '2040.SR','2050.SR','2060.SR','2070.SR','2080.SR','2081.SR','2082.SR','2083.SR',
    '2084.SR','2090.SR','2100.SR','2110.SR','2120.SR','2130.SR','2140.SR','2150.SR',
    '2160.SR','2170.SR','2180.SR','2190.SR','2200.SR','2210.SR','2220.SR','2222.SR',
    '2223.SR','2230.SR','2240.SR','2250.SR','2270.SR','2280.SR','2281.SR','2282.SR',
    '2283.SR','2284.SR','2285.SR','2286.SR','2287.SR','2288.SR','2290.SR','2300.SR',
    '2310.SR','2320.SR','2330.SR','2340.SR','2350.SR','2360.SR','2370.SR','2380.SR',
    '2381.SR','2382.SR','3002.SR','3003.SR','3004.SR','3005.SR','3007.SR','3008.SR',
    '3010.SR','3020.SR','3030.SR','3040.SR','3050.SR','3060.SR','3080.SR','3090.SR',
    '3091.SR','3092.SR','4001.SR','4002.SR','4003.SR','4004.SR','4005.SR','4006.SR',
    '4007.SR','4008.SR','4009.SR','4011.SR','4012.SR','4013.SR','4014.SR','4015.SR',
    '4016.SR','4017.SR','4018.SR','4019.SR','4020.SR','4021.SR','4030.SR','4031.SR',
    '4040.SR','4050.SR','4051.SR','4061.SR','4070.SR','4071.SR','4072.SR','4080.SR',
    '4081.SR','4082.SR','4083.SR','4084.SR','4090.SR','4100.SR','4110.SR','4130.SR',
    '4140.SR','4141.SR','4142.SR','4143.SR','4144.SR','4145.SR','4146.SR','4147.SR',
    '4148.SR','4150.SR','4160.SR','4161.SR','4162.SR','4163.SR','4164.SR','4165.SR',
    '4170.SR','4180.SR','4190.SR','4191.SR','4192.SR','4193.SR','4194.SR','4200.SR',
    '4210.SR','4220.SR','4230.SR','4240.SR','4250.SR','4260.SR','4261.SR','4262.SR',
    '4263.SR','4264.SR','4265.SR','4270.SR','4280.SR','4290.SR','4291.SR','4292.SR',
    '4300.SR','4310.SR','4320.SR','4321.SR','4322.SR','4323.SR','4324.SR','4325.SR',
    '4326.SR','4327.SR','4330.SR','4331.SR','4332.SR','4333.SR','4334.SR','4335.SR',
    '4336.SR','4337.SR','4338.SR','4339.SR','4340.SR','4342.SR','4344.SR','4345.SR',
    '4346.SR','4347.SR','4348.SR','4349.SR','4350.SR','5110.SR','6001.SR','6002.SR',
    '6004.SR','6010.SR','6012.SR','6013.SR','6014.SR','6015.SR','6016.SR','6017.SR',
    '6018.SR','6019.SR','6020.SR','6040.SR','6050.SR','6060.SR','6070.SR','6090.SR',
    '7010.SR','7020.SR','7030.SR','7040.SR','7200.SR','7201.SR','7202.SR','7203.SR',
    '7204.SR','7211.SR','8010.SR','8012.SR','8020.SR','8030.SR','8040.SR','8050.SR',
    '8060.SR','8070.SR','8100.SR','8120.SR','8150.SR','8160.SR','8170.SR','8180.SR',
    '8190.SR','8200.SR','8210.SR','8230.SR','8240.SR','8250.SR','8260.SR','8270.SR',
    '8280.SR','8300.SR','8310.SR','8311.SR','8313.SR',
]

TF_MAP = {
    '30m': ('30 دقيقة', '30m',  '60d', None),
    '1h':  ('1 ساعة',   '60m',  '60d', None),
    '2h':  ('2 ساعة',   '60m',  '60d', '2h'),
    '4h':  ('4 ساعات',  '60m',  '60d', '4h'),
    '1d':  ('يومي',     '1d',   '1y',  None),
    '1w':  ('أسبوعي',   '1wk',  '5y',  None),
}

# ─────────────────────────────────────────────────────────────
# 12. KEYBOARD BUILDERS
# ─────────────────────────────────────────────────────────────
def build_main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 بوت موجات الولفي", callback_data="bot_wolfe")],
        [InlineKeyboardButton("📊 بوت المحلل الرقمي", callback_data="bot_analyzer")],
    ])


def build_tf_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("30 دقيقة", callback_data="scan_30m"), InlineKeyboardButton("1 ساعة", callback_data="scan_1h")],
        [InlineKeyboardButton("2 ساعة",  callback_data="scan_2h"),  InlineKeyboardButton("4 ساعات",callback_data="scan_4h")],
        [InlineKeyboardButton("يومي",    callback_data="scan_1d"),  InlineKeyboardButton("أسبوعي", callback_data="scan_1w")],
        [InlineKeyboardButton("🔙 رجوع للقائمة الرئيسية", callback_data="back_to_main")],
    ])


def build_filter_keyboard(tf_key):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 صاعد فقط", callback_data=f"filter_{tf_key}_bullish"), InlineKeyboardButton("📉 هابط فقط", callback_data=f"filter_{tf_key}_bearish")],
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

# ─────────────────────────────────────────────────────────────
# 13. MESSAGES
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# 14. LANDING HTML (unchanged from original)
# ─────────────────────────────────────────────────────────────
LANDING_HTML = """<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head><meta charset="UTF-8"><title>بوت السوق السعودي</title>
<style>body{font-family:Arial,sans-serif;background:#080c1a;color:#fff;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;}
.card{background:rgba(255,255,255,0.07);padding:40px;border-radius:16px;text-align:center;max-width:480px;}
h1{color:#7c6df5;font-size:2rem;}p{color:#aaa;}</style></head>
<body><div class="card"><h1>🤖 بوت السوق السعودي</h1>
<p>بوت موجات الولفي ويف + المحلل الرقمي</p>
<p style="color:#7c6df5;">ابدأ المحادثة على تيليغرام</p></div></body></html>"""

# ─────────────────────────────────────────────────────────────
# 15. HANDLERS
# ─────────────────────────────────────────────────────────────
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop('waiting_ticker', None)
    await update.message.reply_text(
        MAIN_MENU_MSG, parse_mode="Markdown",
        reply_markup=build_main_keyboard(),
    )


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    # ── Main menu ────────────────────────────────────────
    if data == "back_to_main":
        context.user_data.pop('waiting_ticker', None)
        await query.edit_message_text(
            MAIN_MENU_MSG, parse_mode="Markdown",
            reply_markup=build_main_keyboard(),
        )
        return

    if data == "bot_wolfe":
        context.user_data.pop('waiting_ticker', None)
        await query.edit_message_text(
            WOLFE_WELCOME_MSG, parse_mode="Markdown",
            reply_markup=build_tf_keyboard(),
        )
        return

    if data == "bot_analyzer":
        context.user_data['waiting_ticker'] = True
        await query.edit_message_text(
            ANALYZER_MSG, parse_mode="Markdown",
            reply_markup=build_back_main_keyboard(),
        )
        return

    # ── Wolfe: back to wolfe menu ────────────────────────
    if data == "back_to_wolfe":
        await query.edit_message_text(
            WOLFE_WELCOME_MSG, parse_mode="Markdown",
            reply_markup=build_tf_keyboard(),
        )
        return

    # ── Wolfe: timeframe selected ────────────────────────
    if data.startswith("scan_"):
        tf_key = data[5:]
        if tf_key not in TF_MAP:
            await query.edit_message_text("فاصل زمني غير معروف.", reply_markup=build_back_main_keyboard())
            return
        await query.edit_message_text(
            f"⏱ الفاصل: *{TF_MAP[tf_key][0]}*\n\nاختر الفلتر:",
            parse_mode="Markdown",
            reply_markup=build_filter_keyboard(tf_key),
        )
        return

    # ── Wolfe: filter selected → run scan ───────────────
    if data.startswith("filter_"):
        parts = data.split("_", 2)
        if len(parts) != 3:
            await query.edit_message_text("بيانات غير صالحة.", reply_markup=build_back_main_keyboard())
            return
        _, tf_key, direction = parts
        if tf_key not in TF_MAP:
            await query.edit_message_text("فاصل زمني غير معروف.", reply_markup=build_back_main_keyboard())
            return

        tf_label, interval, period, resample_rule = TF_MAP[tf_key]
        chat_id = query.message.chat_id

        await query.edit_message_text(
            f"⏳ جاري فحص *{len(TADAWUL_TICKERS)}* سهم...\n"
            f"الفاصل: *{tf_label}*\n\nيرجى الانتظار ⏳",
            parse_mode="Markdown",
        )

        loop = asyncio.get_event_loop()
        results, ohlc_data = await loop.run_in_executor(
            _executor, scan_tickers, TADAWUL_TICKERS, period, interval, resample_rule
        )

        bullish_list = []; bearish_list = []
        is_intraday = interval not in ('1d', '1wk')

        for tk, patterns in results.items():
            for r in patterns:
                pct = ((r['target_price'] - r['entry_price']) / r['entry_price']) * 100
                item = {
                    'ticker': tk, 'name': get_name(tk),
                    'last_close': r['last_close'], 'entry': round(r['entry_price'], 2),
                    'target': r['target_price'], 'pct': round(pct, 1),
                    'p5_date': (r['points'][5]['date'].strftime('%Y-%m-%d %H:%M') if is_intraday else r['points'][5]['date'].strftime('%Y-%m-%d')),
                    '_r': r, '_df': ohlc_data[tk],
                }
                if r['direction'] == 'Bullish': bullish_list.append(item)
                else: bearish_list.append(item)

        bullish_list.sort(key=lambda x: x['pct'], reverse=True)
        bearish_list.sort(key=lambda x: x['pct'])

        show_bull = direction in ('bullish', 'both')
        show_bear = direction in ('bearish', 'both')

        summary = f"✅ *اكتمل الفحص — {tf_label}*\n\n"
        if show_bull: summary += f"📈 ولفي صاعد: *{len(bullish_list)}*\n"
        if show_bear: summary += f"📉 ولفي هابط: *{len(bearish_list)}*\n"
        if not bullish_list and not bearish_list: summary += "\nلا توجد نتائج لهذا الفلتر."

        await context.bot.send_message(chat_id=chat_id, text=summary, parse_mode="Markdown")

        if show_bull and bullish_list:
            await context.bot.send_message(chat_id=chat_id, text="📈 *— نتائج الولفي الصاعد —*", parse_mode="Markdown")
            for item in bullish_list:
                try:
                    buf = plot_wolfe_chart(item['ticker'], item['_df'], item['_r'], tf_label)
                    await context.bot.send_photo(chat_id=chat_id, photo=buf)
                except Exception as e:
                    logger.error(f"Chart error {item['ticker']}: {e}")
                msg = (f"رمز السهم: *{item['ticker'].split('.')[0]}*\nالاسم       : `{item['name']}`\n"
                       f"الفاصل       : `{tf_label}`\nآخر إغلاق : `{item['last_close']}`\n"
                       f"قاع (5)    : `{item['entry']}`\nخط (1←4)  : `{item['target']}`\n"
                       f"النسبة      : `{item['pct']:+.1f}%`\nتاريخ (5)  : `{item['p5_date']}`")
                await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")

        if show_bear and bearish_list:
            await context.bot.send_message(chat_id=chat_id, text="📉 *— نتائج الولفي الهابط —*", parse_mode="Markdown")
            for item in bearish_list:
                try:
                    buf = plot_wolfe_chart(item['ticker'], item['_df'], item['_r'], tf_label)
                    await context.bot.send_photo(chat_id=chat_id, photo=buf)
                except Exception as e:
                    logger.error(f"Chart error {item['ticker']}: {e}")
                msg = (f"رمز السهم: *{item['ticker'].split('.')[0]}*\nالاسم       : `{item['name']}`\n"
                       f"الفاصل       : `{tf_label}`\nآخر إغلاق : `{item['last_close']}`\n"
                       f"قمة (5)    : `{item['entry']}`\nخط (1←4)  : `{item['target']}`\n"
                       f"النسبة      : `{item['pct']:+.1f}%`\nتاريخ (5)  : `{item['p5_date']}`")
                await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")

        await context.bot.send_message(
            chat_id=chat_id, text="🔄 *انتهى الفحص*", parse_mode="Markdown",
            reply_markup=build_after_wolfe_keyboard(),
        )
        return


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages — used for analyzer bot ticker input."""
    if not context.user_data.get('waiting_ticker'):
        return  # ignore text if not in analyzer mode

    ticker_input = update.message.text.strip()
    chat_id = update.effective_chat.id

    # Show processing message
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

# ─────────────────────────────────────────────────────────────
# 16. MAIN
# ─────────────────────────────────────────────────────────────
def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
    PORT = int(os.environ.get("PORT", 10000))

    if RENDER_EXTERNAL_URL:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"

        async def home(_request):
            return aio_web.Response(text=LANDING_HTML, content_type='text/html')

        async def webhook_route(request):
            data   = await request.json()
            update = Update.de_json(data, app.bot)
            await app.update_queue.put(update)
            return aio_web.Response(text='OK')

        async def run_all():
            await app.bot.set_webhook(webhook_url)
            logger.info(f"Webhook set → {webhook_url}")
            web_app = aio_web.Application()
            web_app.router.add_get('/',         home)
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
        logger.info("Starting polling (local dev)")
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
