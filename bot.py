import os
import io
import asyncio
import logging
import warnings
import urllib.request
warnings.filterwarnings('ignore')

import arabic_reshaper
from bidi.algorithm import get_display
from aiohttp import web as aio_web

import yfinance as yf
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import mplfinance as mpf

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]

# ────────────────────────────────────────────────────────────
# 1. ARABIC FONT + TEXT SUPPORT
# ────────────────────────────────────────────────────────────
def _init_arabic_font():
    here      = os.path.dirname(os.path.abspath(__file__))
    font_file = os.path.join(here, 'Cairo-Regular.ttf')

    if not os.path.exists(font_file):
        url = ('https://github.com/google/fonts/raw/main/'
               'ofl/cairo/static/Cairo-Regular.ttf')
        try:
            urllib.request.urlretrieve(url, font_file)
            logger.info('Cairo font downloaded successfully.')
        except Exception as exc:
            logger.warning(f'Cairo font download failed: {exc}')

    if os.path.exists(font_file):
        fm.fontManager.addfont(font_file)
        prop = fm.FontProperties(fname=font_file)
        logger.info(f'Using font: {prop.get_name()}')
        return prop.get_name()

    for f in fm.fontManager.ttflist:
        if any(k in f.name.lower() for k in ('cairo', 'tajawal', 'almarai')):
            return f.name

    return 'DejaVu Sans'


ARABIC_FONT = _init_arabic_font()
logger.info(f'Arabic font in use: {ARABIC_FONT}')


def ar(text: str) -> str:
    try:
        return get_display(arabic_reshaper.reshape(str(text)))
    except Exception:
        return str(text)

# ────────────────────────────────────────────────────────────
# 2. COMPANY NAMES
# ────────────────────────────────────────────────────────────
COMPANY_NAMES = {
    '^TASI.SR':'تاسي',
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


def get_name(ticker: str) -> str:
    return COMPANY_NAMES.get(ticker, ticker)

# ────────────────────────────────────────────────────────────
# 3. PIVOT DETECTION
# ────────────────────────────────────────────────────────────
def find_pivots(df, order=5):
    high = df['High'].values
    low  = df['Low'].values
    sh = argrelextrema(high, np.greater_equal, order=order)[0]
    sl = argrelextrema(low,  np.less_equal,    order=order)[0]
    pivots = []
    for i in sh:
        pivots.append({'bar': int(i), 'price': high[i],
                       'type': 'H', 'date': df.index[i]})
    for i in sl:
        pivots.append({'bar': int(i), 'price': low[i],
                       'type': 'L', 'date': df.index[i]})
    pivots.sort(key=lambda x: x['bar'])
    return pivots


def get_alternating_pivots(pivots):
    if not pivots:
        return []
    alt = [pivots[0]]
    for p in pivots[1:]:
        if p['type'] == alt[-1]['type']:
            if p['type'] == 'H' and p['price'] > alt[-1]['price']:
                alt[-1] = p
            elif p['type'] == 'L' and p['price'] < alt[-1]['price']:
                alt[-1] = p
        else:
            alt.append(p)
    return alt

# ────────────────────────────────────────────────────────────
# 4. UTILITIES
# ────────────────────────────────────────────────────────────
def line_at(x, x1, y1, x2, y2):
    if x2 == x1:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def resample_ohlc(df, rule):
    return df.resample(rule).agg({
        'Open':   'first',
        'High':   'max',
        'Low':    'min',
        'Close':  'last',
        'Volume': 'sum',
    }).dropna()

# ────────────────────────────────────────────────────────────
# 5. WOLFE WAVE VALIDATORS  (now with P0)
# ────────────────────────────────────────────────────────────
def validate_bullish(p0, p1, p2, p3, p4, p5, tol=0.03):
    """
    Bullish Wolfe Wave:
      P0=H, P1=L, P2=H, P3=L, P4=H, P5=L
      P0 > P2  (preceding high is above the wave-2 high)
    """
    v = [p['price'] for p in [p1, p2, p3, p4, p5]]
    b = [p['bar']   for p in [p1, p2, p3, p4, p5]]

    # ── type checks ──────────────────────────────────────────
    if p0['type'] != 'H':
        return None
    if not (p1['type'] == 'L' and p2['type'] == 'H' and
            p3['type'] == 'L' and p4['type'] == 'H' and p5['type'] == 'L'):
        return None

    # ── P0 rule: P0 > P2 ────────────────────────────────────
    if p0['price'] <= p2['price']:
        return None

    # ── existing wave rules ──────────────────────────────────
    if v[2] >= v[0]:          return None   # P3 >= P1
    if v[3] >= v[1]:          return None   # P4 >= P2
    if v[3] <= v[0]:          return None   # P4 <= P1
    if v[4] >= v[2]:          return None   # P5 >= P3

    s13 = (v[2]-v[0])/(b[2]-b[0]) if b[2] != b[0] else 0
    s24 = (v[3]-v[1])/(b[3]-b[1]) if b[3] != b[1] else 0
    if s13 >= 0 or s24 >= 0:  return None
    if s13 >= s24:            return None

    proj = line_at(b[4], b[0], v[0], b[2], v[2])
    if proj != 0 and (proj - v[4]) / abs(proj) < -tol:
        return None

    return {'direction': 'Bullish',
            'points': [p0, p1, p2, p3, p4, p5],
            'entry_price': v[4],
            'p5_date': p5['date']}


def validate_bearish(p0, p1, p2, p3, p4, p5, tol=0.03):
    """
    Bearish Wolfe Wave:
      P0=L, P1=H, P2=L, P3=H, P4=L, P5=H
      P0 < P2  (preceding low is below the wave-2 low)
    """
    v = [p['price'] for p in [p1, p2, p3, p4, p5]]
    b = [p['bar']   for p in [p1, p2, p3, p4, p5]]

    # ── type checks ──────────────────────────────────────────
    if p0['type'] != 'L':
        return None
    if not (p1['type'] == 'H' and p2['type'] == 'L' and
            p3['type'] == 'H' and p4['type'] == 'L' and p5['type'] == 'H'):
        return None

    # ── P0 rule: P0 < P2 ────────────────────────────────────
    if p0['price'] >= p2['price']:
        return None

    # ── existing wave rules ──────────────────────────────────
    if v[2] <= v[0]:          return None   # P3 <= P1
    if v[3] <= v[1]:          return None   # P4 <= P2
    if v[3] >= v[0]:          return None   # P4 >= P1
    if v[4] <= v[2]:          return None   # P5 <= P3

    s13 = (v[2]-v[0])/(b[2]-b[0]) if b[2] != b[0] else 0
    s24 = (v[3]-v[1])/(b[3]-b[1]) if b[3] != b[1] else 0
    if s13 <= 0 or s24 <= 0:  return None
    if s13 >= s24:            return None

    proj = line_at(b[4], b[0], v[0], b[2], v[2])
    if proj != 0 and (v[4] - proj) / abs(proj) < -tol:
        return None

    return {'direction': 'Bearish',
            'points': [p0, p1, p2, p3, p4, p5],
            'entry_price': v[4],
            'p5_date': p5['date']}

# ────────────────────────────────────────────────────────────
# 6. ACTIVE PATTERN FINDER  (now expects 6 pivots: P0–P5)
# ────────────────────────────────────────────────────────────
def find_active_wolfe(df, max_bars_since_p5=8):
    n         = len(df)
    best_bull = None
    best_bear = None

    for order in [4, 5, 6, 7]:
        piv = get_alternating_pivots(find_pivots(df, order=order))
        if len(piv) < 6:                          # need P0–P5
            continue
        for offset in range(min(4, len(piv) - 5)):
            idx = len(piv) - 6 - offset
            if idx < 0:
                break
            combo = piv[idx: idx + 6]             # [P0,P1,P2,P3,P4,P5]
            if n - 1 - combo[5]['bar'] > max_bars_since_p5:
                continue
            r = validate_bullish(*combo)
            if r and (best_bull is None or
                      combo[5]['bar'] > best_bull['points'][5]['bar']):
                best_bull = r
            r = validate_bearish(*combo)
            if r and (best_bear is None or
                      combo[5]['bar'] > best_bear['points'][5]['bar']):
                best_bear = r

    return [x for x in [best_bull, best_bear] if x]

# ────────────────────────────────────────────────────────────
# 7. CHART → PNG bytes  (P0–P5 version)
# ────────────────────────────────────────────────────────────
def plot_wolfe_chart(ticker, df, result, tf_label):
    pts          = result['points']          # 6 points: P0–P5
    direction    = result['direction']
    entry        = result['entry_price']
    target       = result['target_price']
    is_bull      = direction == 'Bullish'
    company      = get_name(ticker)
    ticker_code  = ticker.split('.')[0]

    b          = [p['bar']   for p in pts]   # indices 0–5
    v          = [p['price'] for p in pts]
    last_bar   = len(df) - 1
    last_close = float(df['Close'].iloc[-1])
    pct        = ((target - entry) / entry) * 100

    pad_l = max(0, b[0] - 10)
    pad_r = min(last_bar, b[5] + 30)
    df_z  = df.iloc[pad_l: pad_r + 1].copy()
    off   = pad_l
    zb    = [x - off for x in b]
    n_z   = len(df_z)

    C_W  = '#0D47A1' if is_bull else '#B71C1C'
    C_T  = '#2E7D32' if is_bull else '#C62828'
    C_24 = '#E65100'
    C_E  = '#6A1B9A'
    C_A  = '#00695C' if is_bull else '#880E4F'
    C_P0 = '#FF6F00'                         # distinct colour for P0

    mc  = mpf.make_marketcolors(
        up='#26A69A', down='#EF5350', edge='inherit', wick='inherit'
    )
    sty = mpf.make_mpf_style(
        marketcolors=mc, gridcolor='#EEEEEE', gridstyle='-',
        facecolor='#FAFBFC', y_on_right=False,
        rc={'font.size': 10, 'grid.alpha': 0.2},
    )

    fig, axes = mpf.plot(
        df_z, type='candle', style=sty,
        figsize=(16, 8), returnfig=True, volume=False,
    )
    ax = axes[0]
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.06)

    # ── zigzag P0→P5 ────────────────────────────────────────
    ax.plot(zb, v, color=C_W, lw=2.5, zorder=6, alpha=0.8)
    ax.scatter(zb, v, s=120, c='white', edgecolors=C_W,
               linewidths=2.5, zorder=7)
    # highlight P0 with distinct colour
    ax.scatter([zb[0]], [v[0]], s=160, c='white', edgecolors=C_P0,
               linewidths=3, zorder=8)

    # ── channel lines (1-3 and 2-4) ─────────────────────────
    ext = zb[5] + 8
    ax.plot([zb[1], ext],
            [v[1], line_at(ext+off, b[1], v[1], b[3], v[3])],
            color=C_W, lw=1.0, ls='--', alpha=0.3)
    ax.plot([zb[2], ext],
            [v[2], line_at(ext+off, b[2], v[2], b[4], v[4])],
            color=C_24, lw=1.0, ls='--', alpha=0.3)

    # ── wedge fill ───────────────────────────────────────────
    fx = np.arange(zb[1], zb[5] + 1)
    f1 = [line_at(x+off, b[1], v[1], b[3], v[3]) for x in fx]
    f2 = [line_at(x+off, b[2], v[2], b[4], v[4]) for x in fx]
    ax.fill_between(fx, f1, f2, alpha=0.04, color=C_W)

    # ── target line (P1→P4 extended) ─────────────────────────
    tgt_end_zb = n_z + 5
    ax.plot([zb[1], tgt_end_zb],
            [v[1], line_at(tgt_end_zb+off, b[1], v[1], b[4], v[4])],
            color=C_T, lw=3.0, ls='-.', alpha=0.85, zorder=5)

    # ── target diamond & guides ──────────────────────────────
    z_last = min(last_bar - off, n_z - 1)
    ax.plot(z_last, target, marker='D', ms=14, color=C_T,
            markeredgecolor='white', markeredgewidth=2, zorder=9)
    ax.axhline(y=target, color=C_T, lw=0.6, ls=':', alpha=0.25)
    ax.axhline(y=entry,  color=C_E, lw=0.6, ls=':', alpha=0.25)

    # ── arrow from P5 to target zone ─────────────────────────
    arrow_land_zb    = min(zb[5] + max(4, (z_last-zb[5])//2), n_z+3)
    arrow_land_price = line_at(arrow_land_zb+off, b[1], v[1], b[4], v[4])

    ax.annotate(
        '',
        xy=(arrow_land_zb, arrow_land_price),
        xytext=(zb[5], entry),
        arrowprops=dict(
            arrowstyle='-|>', color=C_A, lw=3.0, mutation_scale=22,
            connectionstyle='arc3,rad=0.15' if is_bull else 'arc3,rad=-0.15',
        ),
        zorder=8,
    )

    price_range  = max(v) - min(v)
    label_offset = price_range * 0.08
    pct_y = (arrow_land_price + label_offset if is_bull
             else arrow_land_price - label_offset)
    ax.text(
        arrow_land_zb, pct_y, f'{pct:+.1f}%',
        fontsize=13, fontweight='bold', color=C_A,
        ha='center', va='bottom' if is_bull else 'top',
        fontfamily=ARABIC_FONT,
        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                  ec=C_A, alpha=0.9, lw=0.8),
        zorder=10,
    )

    # ── point labels P0–P5 ───────────────────────────────────
    for i in range(6):
        is_low = pts[i]['type'] == 'L'
        dt_str = pts[i]['date'].strftime('%b %d')
        label_color = C_P0 if i == 0 else C_W
        ax.annotate(
            f'P{i}  {v[i]:.2f}\n{dt_str}',
            xy=(zb[i], v[i]),
            xytext=(0, -28 if is_low else 28),
            textcoords='offset points',
            ha='center', va='top' if is_low else 'bottom',
            fontsize=8.5, fontweight='bold', color=label_color,
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec=label_color, alpha=0.9, lw=0.6),
            arrowprops=dict(arrowstyle='-', color=label_color, lw=0.6),
        )

    # ── Title ────────────────────────────────────────────────
    emoji        = '📈' if is_bull else '📉'
    direction_ar = ar('ولفي صاعد') if is_bull else ar('ولفي هابط')
    ax.set_title(
        f'{emoji}  {ticker_code}  |  {ar(company)}  |  '
        f'{direction_ar}  |  {ar(tf_label)}',
        fontsize=15, fontweight='bold', pad=16,
        color='#212121', fontfamily=ARABIC_FONT,
    )
    ax.set_ylabel('')

    # ── Info box ─────────────────────────────────────────────
    bc = '#E8F5E9' if is_bull else '#FFEBEE'
    bt = '#2E7D32' if is_bull else '#C62828'

    info_lines = [
        f"  {ar(company)}",
        f"  {'─' * 24}",
        f"  {last_close:.2f}        :  {ar('الإغلاق')}",
        f"  {entry:.2f}        :  {ar('الموجة 5')}",
        f"  {target:.2f}       :  {ar('4 ← 1')}",
        f"  {pct:+.1f}%        :  {ar('النسبة')}",
        f"  {ar(tf_label)}     :  {ar('الفاصل')}",
    ]
    info = '\n'.join(info_lines)

    ax.text(
        0.01, 0.03, info,
        transform=ax.transAxes,
        fontsize=10, fontweight='bold', color=bt,
        fontfamily=ARABIC_FONT,
        verticalalignment='bottom', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.6', facecolor=bc,
                  edgecolor=bt, alpha=0.92, lw=1.2),
        zorder=10,
    )

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

# ────────────────────────────────────────────────────────────
# 8. TICKER PROCESSING  (P1→P4 target line, indices 1 & 4)
# ────────────────────────────────────────────────────────────
def process_ticker(ticker, period, interval, resample_rule=None):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df is None or len(df) < 30:
            return ticker, [], None
        if resample_rule:
            df = resample_ohlc(df, resample_rule)
            if len(df) < 30:
                return ticker, [], None
        found    = find_active_wolfe(df, max_bars_since_p5=8)
        last_bar = len(df) - 1
        for r in found:
            # target is projection of line P1→P4
            b1 = r['points'][1]['bar'];  v1 = r['points'][1]['price']
            b4 = r['points'][4]['bar'];  v4 = r['points'][4]['price']
            r['target_price'] = round(line_at(last_bar, b1, v1, b4, v4), 2)
            r['last_close']   = round(float(df['Close'].iloc[-1]), 2)
        return ticker, found, df
    except Exception:
        return ticker, [], None


def scan_tickers(tickers, period, interval, resample_rule=None, max_workers=15):
    all_res = {}
    ohlc    = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(process_ticker, t, period, interval, resample_rule): t
            for t in tickers
        }
        for f in as_completed(futs):
            tk, found, df = f.result()
            if found:
                all_res[tk] = found
                ohlc[tk]    = df
    return all_res, ohlc

# ────────────────────────────────────────────────────────────
# 9. TICKERS
# ────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────
# 10. TIMEFRAME MAP
# ────────────────────────────────────────────────────────────
TF_MAP = {
    '30m': ('30 دقيقة', '30m',  '60d', None),
    '1h':  ('1 ساعة',   '60m',  '60d', None),
    '2h':  ('2 ساعة',   '60m',  '60d', '2h'),
    '4h':  ('4 ساعات',  '60m',  '60d', '4h'),
    '1d':  ('يومي',     '1d',   '1y',  None),
    '1w':  ('أسبوعي',   '1wk',  '5y',  None),
}

# ────────────────────────────────────────────────────────────
# 11. KEYBOARDS
# ────────────────────────────────────────────────────────────
def build_tf_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("30 دقيقة", callback_data="scan_30m"),
            InlineKeyboardButton("1 ساعة",   callback_data="scan_1h"),
        ],
        [
            InlineKeyboardButton("2 ساعة",  callback_data="scan_2h"),
            InlineKeyboardButton("4 ساعات", callback_data="scan_4h"),
        ],
        [
            InlineKeyboardButton("يومي",   callback_data="scan_1d"),
            InlineKeyboardButton("أسبوعي", callback_data="scan_1w"),
        ],
    ])


def build_filter_keyboard(tf_key):
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📈 صاعد فقط",
                                 callback_data=f"filter_{tf_key}_bullish"),
            InlineKeyboardButton("📉 هابط فقط",
                                 callback_data=f"filter_{tf_key}_bearish"),
        ],
        [
            InlineKeyboardButton("📊 الكل",
                                 callback_data=f"filter_{tf_key}_both"),
        ],
        [
            InlineKeyboardButton("🔙 رجوع", callback_data="back_to_start"),
        ],
    ])


def build_start_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🏠 البداية", callback_data="back_to_start")]
    ])

# ────────────────────────────────────────────────────────────
# 12. WELCOME MESSAGE
# ────────────────────────────────────────────────────────────
WELCOME_MSG = (
    "🎯 *فاحص موجات الولفي ويف — السوق السعودي*\n\n"
    "اختر الفاصل الزمني للفحص:\n\n"
    "⚠️ تنبيه: هذا بحث عن موجات الولفي ويف فقط، "
    "لا يجب الاعتماد عليه وقد يكون خطأ. "
    "يجب متابعة الحركة السعرية."
)

# ────────────────────────────────────────────────────────────
# 13. LANDING PAGE HTML
# ────────────────────────────────────────────────────────────
LANDING_HTML = """<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>فاحص الولفي ويف — السوق السعودي</title>
  <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800;900&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

    :root {
      --accent:    #7c6df5;
      --accent2:   #a78bfa;
      --green:     #10d97e;
      --blue:      #60a5fa;
      --card-bg:   rgba(15, 18, 35, 0.82);
      --border:    rgba(255,255,255,0.08);
      --text-mute: rgba(255,255,255,0.45);
    }

    body {
      font-family: 'Tajawal', 'Segoe UI', Tahoma, Arial, sans-serif;
      background: #080c1a;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow-x: hidden;
      position: relative;
      padding: 20px 0 140px;
    }

    /* ── Animated background ── */
    .bg-canvas {
      position: fixed;
      inset: 0;
      z-index: 0;
      background:
        radial-gradient(ellipse 80% 60% at 20% 10%,  rgba(124,109,245,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 70% 50% at 80% 90%,  rgba(16,217,126,0.12) 0%, transparent 55%),
        radial-gradient(ellipse 60% 70% at 75% 15%,  rgba(96,165,250,0.10) 0%, transparent 50%),
        linear-gradient(160deg, #080c1a 0%, #0d1224 50%, #080c1a 100%);
    }

    .bg-grid {
      position: fixed;
      inset: 0;
      z-index: 0;
      background-image:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
      background-size: 48px 48px;
      mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 30%, transparent 100%);
    }

    .orb {
      position: fixed;
      border-radius: 50%;
      filter: blur(70px);
      opacity: 0.25;
      animation: drift 18s ease-in-out infinite alternate;
      z-index: 0;
      pointer-events: none;
    }
    .orb-1 { width: 380px; height: 380px; background: #7c6df5; top: -80px;  left: -80px;  animation-duration: 20s; }
    .orb-2 { width: 300px; height: 300px; background: #10d97e; bottom: -60px; right: -60px; animation-duration: 25s; animation-delay: -8s; }
    .orb-3 { width: 220px; height: 220px; background: #60a5fa; top: 40%;   right: 10%;   animation-duration: 22s; animation-delay: -4s; }

    @keyframes drift {
      from { transform: translate(0, 0)   scale(1); }
      to   { transform: translate(40px, 30px) scale(1.12); }
    }

    /* ── Main card ── */
    .card {
      position: relative;
      z-index: 1;
      background: var(--card-bg);
      backdrop-filter: blur(28px) saturate(1.4);
      -webkit-backdrop-filter: blur(28px) saturate(1.4);
      border: 1px solid var(--border);
      border-radius: 32px;
      padding: 52px 48px 44px;
      text-align: center;
      max-width: 560px;
      width: 93%;
      box-shadow:
        0 0 0 1px rgba(255,255,255,0.04) inset,
        0 32px 80px rgba(0,0,0,0.65),
        0 0 60px rgba(124,109,245,0.08);
      color: white;
      animation: cardIn 0.7s cubic-bezier(0.22,1,0.36,1) both;
    }

    @keyframes cardIn {
      from { opacity: 0; transform: translateY(32px) scale(0.97); }
      to   { opacity: 1; transform: translateY(0)    scale(1); }
    }

    .card::before {
      content: '';
      position: absolute;
      top: 0; left: 10%; right: 10%;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
      border-radius: 50%;
    }

    /* ── Logo ring ── */
    .logo-wrap {
      position: relative;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 24px;
    }
    .logo-ring {
      width: 90px; height: 90px;
      border-radius: 50%;
      background: linear-gradient(135deg, rgba(124,109,245,0.25), rgba(16,217,126,0.15));
      border: 1.5px solid rgba(124,109,245,0.35);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 42px;
      box-shadow: 0 0 32px rgba(124,109,245,0.25), 0 0 0 8px rgba(124,109,245,0.06);
      animation: pulse-ring 3s ease-in-out infinite;
    }
    @keyframes pulse-ring {
      0%,100% { box-shadow: 0 0 32px rgba(124,109,245,0.25), 0 0 0 8px  rgba(124,109,245,0.06); }
      50%      { box-shadow: 0 0 48px rgba(124,109,245,0.40), 0 0 0 14px rgba(124,109,245,0.10); }
    }

    /* ── Typography ── */
    h1 {
      font-size: 27px;
      font-weight: 900;
      margin-bottom: 10px;
      background: linear-gradient(120deg, #fff 30%, var(--accent2) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      line-height: 1.3;
    }
    .subtitle {
      font-size: 14.5px;
      color: rgba(255,255,255,0.52);
      line-height: 1.85;
      margin-bottom: 28px;
    }
    .subtitle strong {
      color: rgba(255,255,255,0.80);
      font-weight: 700;
    }

    /* ── Status badge ── */
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(16,217,126,0.10);
      border: 1px solid rgba(16,217,126,0.30);
      color: var(--green);
      padding: 8px 20px;
      border-radius: 50px;
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 32px;
      letter-spacing: 0.3px;
    }
    .dot {
      width: 8px; height: 8px;
      background: var(--green);
      border-radius: 50%;
      flex-shrink: 0;
      box-shadow: 0 0 6px var(--green);
      animation: blink 1.6s ease-in-out infinite;
    }
    @keyframes blink {
      0%,100% { opacity: 1;    transform: scale(1); }
      50%      { opacity: 0.2; transform: scale(0.75); }
    }

    /* ── Divider ── */
    .divider {
      border: none;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.09), transparent);
      margin: 28px 0;
    }

    /* ── Stats ── */
    .stats {
      display: flex;
      justify-content: center;
      gap: 0;
      margin-bottom: 34px;
    }
    .stat {
      flex: 1;
      text-align: center;
      padding: 16px 8px;
      border-radius: 16px;
      transition: background 0.2s;
    }
    .stat:hover { background: rgba(255,255,255,0.04); }
    .stat + .stat { border-right: 1px solid rgba(255,255,255,0.08); }

    .stat-num {
      font-size: 30px;
      font-weight: 900;
      background: linear-gradient(120deg, var(--blue), var(--accent2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      line-height: 1.1;
    }
    .stat-lbl {
      font-size: 12px;
      color: var(--text-mute);
      margin-top: 5px;
      font-weight: 500;
    }
    .stat-icon { font-size: 18px; margin-bottom: 4px; }

    /* ── Feature pills ── */
    .features {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 8px;
      margin-bottom: 32px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.09);
      color: rgba(255,255,255,0.70);
      padding: 6px 14px;
      border-radius: 50px;
      font-size: 12.5px;
      font-weight: 500;
      transition: background 0.2s, border-color 0.2s;
    }
    .pill:hover {
      background: rgba(124,109,245,0.12);
      border-color: rgba(124,109,245,0.30);
      color: var(--accent2);
    }

    /* ── CTA Button ── */
    .btn-wrap { position: relative; display: inline-block; }

    .btn-glow {
      position: absolute;
      inset: -3px;
      border-radius: 54px;
      background: linear-gradient(135deg, var(--accent), var(--green));
      filter: blur(14px);
      opacity: 0.45;
      animation: glow-pulse 2.4s ease-in-out infinite;
      z-index: -1;
    }
    @keyframes glow-pulse {
      0%,100% { opacity: 0.35; transform: scale(0.98); }
      50%      { opacity: 0.60; transform: scale(1.02); }
    }

    .btn {
      position: relative;
      display: inline-flex;
      align-items: center;
      gap: 10px;
      background: linear-gradient(135deg, #7c6df5 0%, #5b4fe0 50%, #4338ca 100%);
      color: white;
      text-decoration: none;
      padding: 16px 42px;
      border-radius: 50px;
      font-size: 16px;
      font-weight: 800;
      font-family: 'Tajawal', sans-serif;
      box-shadow:
        0 8px 28px rgba(124,109,245,0.40),
        0 0 0 1px rgba(255,255,255,0.10) inset;
      transition: transform 0.22s cubic-bezier(0.22,1,0.36,1), box-shadow 0.22s;
      letter-spacing: 0.3px;
    }
    .btn:hover {
      transform: translateY(-4px) scale(1.03);
      box-shadow:
        0 16px 40px rgba(124,109,245,0.55),
        0 0 0 1px rgba(255,255,255,0.15) inset;
    }
    .btn:active { transform: translateY(-1px) scale(1.01); }

    .btn-icon { font-size: 20px; animation: wave-icon 2s ease-in-out infinite; }
    @keyframes wave-icon {
      0%,100% { transform: rotate(0deg); }
      25%      { transform: rotate(-8deg); }
      75%      { transform: rotate(8deg); }
    }

    /* ── Footer ── */
    .footer {
      margin-top: 28px;
      font-size: 11.5px;
      color: rgba(255,255,255,0.22);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .footer-dot {
      width: 3px; height: 3px;
      background: rgba(255,255,255,0.22);
      border-radius: 50%;
    }

    /* ── Bottom 3-bar stocks ticker ── */
    .ticker-wrap {
      position: fixed;
      bottom: 0; left: 0; right: 0;
      z-index: 2;
      overflow: hidden;
      background: rgba(8,12,26,0.96);
      border-top: 1px solid rgba(255,255,255,0.07);
      padding: 8px 0 10px;
      display: flex;
      flex-direction: column;
      gap: 7px;
    }

    /* fade edges */
    .ticker-wrap::before,
    .ticker-wrap::after {
      content: '';
      position: absolute;
      top: 0; bottom: 0;
      width: 100px;
      z-index: 3;
      pointer-events: none;
    }
    .ticker-wrap::before {
      right: 0;
      background: linear-gradient(to left, rgba(8,12,26,0.98), transparent);
    }
    .ticker-wrap::after {
      left: 0;
      background: linear-gradient(to right, rgba(8,12,26,0.98), transparent);
    }

    .marquee-row {
      display: flex;
      gap: 8px;
      width: max-content;
      will-change: transform;
    }

    .stock-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: 50px;
      padding: 4px 12px;
      font-size: 12.5px;
      font-weight: 700;
      color: rgba(255,255,255,0.70);
      white-space: nowrap;
      flex-shrink: 0;
      cursor: default;
      user-select: none;
      transition: background 0.2s, color 0.2s;
    }
    .stock-chip:hover {
      background: rgba(124,109,245,0.14);
      border-color: rgba(124,109,245,0.28);
      color: var(--accent2);
    }
    .chip-code {
      font-size: 10px;
      color: rgba(255,255,255,0.28);
      font-weight: 500;
    }

    /* ── Responsive ── */
    @media (max-width: 480px) {
      .card { padding: 40px 20px 36px; border-radius: 24px; }
      h1 { font-size: 22px; }
      .stat-num { font-size: 24px; }
      .btn { padding: 14px 28px; font-size: 15px; }
      body { padding-bottom: 150px; }
    }
  </style>
</head>
<body>

  <div class="bg-canvas"></div>
  <div class="bg-grid"></div>
  <div class="orb orb-1"></div>
  <div class="orb orb-2"></div>
  <div class="orb orb-3"></div>

  <!-- Main card -->
  <div class="card" role="main">

    <div class="logo-wrap">
      <div class="logo-ring">📈</div>
    </div>

    <h1>فاحص موجات الولفي ويف</h1>
    <p class="subtitle">
      بوت تيليغرام ذكي لرصد موجات الولفي ويف<br>
      على <strong>جميع أسهم تداول السعودي</strong> بدقة عالية
    </p>

    <div class="badge">
      <div class="dot"></div>
      البوت يعمل الآن
    </div>

    <hr class="divider">

    <div class="stats">
      <div class="stat">
        <div class="stat-icon">🏦</div>
        <div class="stat-num">240+</div>
        <div class="stat-lbl">سهم مفحوص</div>
      </div>
      <div class="stat">
        <div class="stat-icon">🕐</div>
        <div class="stat-num">6</div>
        <div class="stat-lbl">أطر زمنية</div>
      </div>
      <div class="stat">
        <div class="stat-icon">〽️</div>
        <div class="stat-num">2</div>
        <div class="stat-lbl">نوع موجة</div>
      </div>
    </div>

    <div class="features">
      <span class="pill">⚡ فحص فوري</span>
      <span class="pill">📊 تحليل متعدد الأطر</span>
      <span class="pill">🎯 دقة عالية</span>
      <span class="pill">🔔 تنبيهات لحظية</span>
      <span class="pill">🇸🇦 سوق تداول</span>
    </div>

    <div class="btn-wrap">
      <div class="btn-glow"></div>
      <a class="btn" href="https://t.me/BOT_USERNAME" target="_blank" rel="noopener">
        <span class="btn-icon">🤖</span>
        فتح البوت في تيليغرام
      </a>
    </div>

    <div class="footer">
      <span>Wolfe Wave Scanner</span>
      <div class="footer-dot"></div>
      <span>Saudi Market — Tadawul</span>
      <div class="footer-dot"></div>
      <span>2026</span>
    </div>

  </div>

  <!-- Bottom 3-bar stocks -->
  <div class="ticker-wrap" aria-hidden="true">
    <div class="marquee-row" id="bottom1"></div>
    <div class="marquee-row" id="bottom2"></div>
    <div class="marquee-row" id="bottom3"></div>
  </div>

  <script>
    const STOCKS = [
      ['^TASI', 'تاسي'],
      ['1010',  'الرياض'],
      ['1020',  'الجزيرة'],
      ['1030',  'الإستثمار'],
      ['1050',  'بي اس اف'],
      ['1060',  'الأول'],
      ['1080',  'العربي'],
      ['1111',  'مجموعة تداول'],
      ['1120',  'الراجحي'],
      ['1140',  'البلاد'],
      ['1150',  'الإنماء'],
      ['1180',  'الأهلي'],
      ['1182',  'أملاك'],
      ['1183',  'سهل'],
      ['1201',  'تكوين'],
      ['1202',  'مبكو'],
      ['1210',  'بي سي آي'],
      ['1211',  'معادن'],
      ['1212',  'أسترا الصناعية'],
      ['1213',  'نسيج'],
      ['1214',  'شاكر'],
      ['1301',  'أسلاك'],
      ['1302',  'بوان'],
      ['1303',  'الصناعات الكهربائية'],
      ['1304',  'اليمامة للحديد'],
      ['1320',  'أنابيب السعودية'],
      ['1321',  'أنابيب الشرق'],
      ['1322',  'أماك'],
      ['1323',  'يو سي آي سي'],
      ['1810',  'سيرا'],
      ['1820',  'بان'],
      ['1830',  'لجام للرياضة'],
      ['1831',  'مهارة'],
      ['1832',  'صدر'],
      ['1833',  'الموارد'],
      ['1834',  'سماسكو'],
      ['1835',  'تمكين'],
      ['2001',  'كيمانول'],
      ['2010',  'سابك'],
      ['2020',  'سابك للمغذيات الزراعية'],
      ['2030',  'المصافي'],
      ['2040',  'الخزف السعودي'],
      ['2050',  'مجموعة صافولا'],
      ['2060',  'التصنيع'],
      ['2070',  'الدوائية'],
      ['2080',  'الغاز'],
      ['2081',  'الخريف'],
      ['2082',  'أكوا'],
      ['2083',  'مرافق'],
      ['2084',  'مياهنا'],
      ['2090',  'جبسكو'],
      ['2100',  'وفرة'],
      ['2110',  'الكابلات السعودية'],
      ['2120',  'متطورة'],
      ['2130',  'صدق'],
      ['2140',  'أيان'],
      ['2150',  'زجاج'],
      ['2160',  'أميانتيت'],
      ['2170',  'اللجين'],
      ['2180',  'فيبكو'],
      ['2190',  'سيسكو القابضة'],
      ['2200',  'أنابيب'],
      ['2210',  'نماء للكيماويات'],
      ['2220',  'معدنية'],
      ['2222',  'أرامكو السعودية'],
      ['2223',  'لوبريف'],
      ['2230',  'الكيميائية'],
      ['2240',  'صناعات'],
      ['2250',  'المجموعة السعودية'],
      ['2270',  'سدافكو'],
      ['2280',  'المراعي'],
      ['2281',  'تنمية'],
      ['2282',  'نقي'],
      ['2283',  'المطاحن الأولى'],
      ['2284',  'المطاحن الحديثة'],
      ['2285',  'المطاحن العربية'],
      ['2286',  'المطاحن الرابعة'],
      ['2287',  'إنتاج'],
      ['2288',  'نفوذ'],
      ['2290',  'ينساب'],
      ['2300',  'صناعة الورق'],
      ['2310',  'سبكيم العالمية'],
      ['2320',  'البابطين'],
      ['2330',  'المتقدمة'],
      ['2340',  'ارتيكس'],
      ['2350',  'كيان السعودية'],
      ['2360',  'الفخارية'],
      ['2370',  'مسك'],
      ['2380',  'بترو رابغ'],
      ['2381',  'الحفر العربية'],
      ['2382',  'أديس'],
      ['3002',  'أسمنت نجران'],
      ['3003',  'أسمنت المدينة'],
      ['3004',  'أسمنت الشمالية'],
      ['3005',  'أسمنت ام القرى'],
      ['3007',  'الواحة'],
      ['3008',  'الكثيري'],
      ['3010',  'أسمنت العربية'],
      ['3020',  'أسمنت اليمامة'],
      ['3030',  'أسمنت السعودية'],
      ['3040',  'أسمنت القصيم'],
      ['3050',  'أسمنت الجنوب'],
      ['3060',  'أسمنت ينبع'],
      ['3080',  'أسمنت الشرقية'],
      ['3090',  'أسمنت تبوك'],
      ['3091',  'أسمنت الجوف'],
      ['3092',  'أسمنت الرياض'],
      ['4001',  'أسواق ع العثيم'],
      ['4002',  'المواساة'],
      ['4003',  'إكسترا'],
      ['4004',  'دله الصحية'],
      ['4005',  'رعاية'],
      ['4006',  'أسواق المزرعة'],
      ['4007',  'الحمادي'],
      ['4008',  'ساكو'],
      ['4009',  'السعودي الألماني'],
      ['4011',  'لازوردي'],
      ['4012',  'الأصيل'],
      ['4013',  'سليمان الحبيب'],
      ['4014',  'دار المعدات'],
      ['4015',  'جمجوم فارما'],
      ['4016',  'أفالون فارما'],
      ['4017',  'فقيه الطبية'],
      ['4018',  'الموسى'],
      ['4019',  'اس ام سي'],
      ['4020',  'العقارية'],
      ['4021',  'المركز الكندي الطبي'],
      ['4030',  'البحري'],
      ['4031',  'الخدمات الأرضية'],
      ['4040',  'سابتكو'],
      ['4050',  'ساسكو'],
      ['4051',  'باعظيم'],
      ['4061',  'أنعام القابضة'],
      ['4070',  'تهامة'],
      ['4071',  'العربية'],
      ['4072',  'إم بي سي'],
      ['4080',  'سناد القابضة'],
      ['4081',  'النايفات'],
      ['4082',  'مرنة'],
      ['4083',  'تسهيل'],
      ['4084',  'دراية'],
      ['4090',  'طيبة'],
      ['4100',  'مكة'],
      ['4110',  'باتك'],
      ['4130',  'درب السعودية'],
      ['4140',  'صادرات'],
      ['4141',  'العمران'],
      ['4142',  'كابلات الرياض'],
      ['4143',  'تالكو'],
      ['4144',  'رؤوم'],
      ['4145',  'أو جي سي'],
      ['4146',  'جاز'],
      ['4147',  'سي جي إس'],
      ['4148',  'الوسائل الصناعية'],
      ['4150',  'التعمير'],
      ['4160',  'ثمار'],
      ['4161',  'بن داود'],
      ['4162',  'المنجم'],
      ['4163',  'الدواء'],
      ['4164',  'النهدي'],
      ['4165',  'الماجد للعود'],
      ['4170',  'شمس'],
      ['4180',  'مجموعة فتيحي'],
      ['4190',  'جرير'],
      ['4191',  'أبو معطي'],
      ['4192',  'السيف غاليري'],
      ['4193',  'نايس ون'],
      ['4194',  'محطة البناء'],
      ['4200',  'الدريس'],
      ['4210',  'الأبحاث والإعلام'],
      ['4220',  'إعمار'],
      ['4230',  'البحر الأحمر'],
      ['4240',  'سينومي ريتيل'],
      ['4250',  'جبل عمر'],
      ['4260',  'بدجت السعودية'],
      ['4261',  'ذيب'],
      ['4262',  'لومي'],
      ['4263',  'سال'],
      ['4264',  'طيران ناس'],
      ['4265',  'شري'],
      ['4270',  'طباعة وتغليف'],
      ['4280',  'المملكة'],
      ['4290',  'الخليج للتدريب'],
      ['4291',  'الوطنية للتعليم'],
      ['4292',  'عطاء'],
      ['4300',  'دار الأركان'],
      ['4310',  'مدينة المعرفة'],
      ['4320',  'الأندلس'],
      ['4321',  'سينومي سنترز'],
      ['4322',  'رتال'],
      ['4323',  'سمو'],
      ['4324',  'بنان'],
      ['4325',  'مسار'],
      ['4326',  'الماجدية'],
      ['4327',  'الرمز'],
      ['4330',  'الرياض ريت'],
      ['4331',  'الجزيرة ريت'],
      ['4332',  'جدوى ريت الحرمين'],
      ['4333',  'تعليم ريت'],
      ['4334',  'المعذر ريت'],
      ['4335',  'مشاركة ريت'],
      ['4336',  'ملكية ريت'],
      ['4337',  'العزيزية ريت'],
      ['4338',  'الأهلي ريت 1'],
      ['4339',  'دراية ريت'],
      ['4340',  'الراجحي ريت'],
      ['4342',  'جدوى ريت السعودية'],
      ['4344',  'سدكو كابيتال ريت'],
      ['4345',  'الإنماء ريت للتجزئة'],
      ['4346',  'ميفك ريت'],
      ['4347',  'بنيان ريت'],
      ['4348',  'الخبير ريت'],
      ['4349',  'الإنماء ريت الفندقي'],
      ['4350',  'الإستثمار ريت'],
      ['5110',  'كهرباء السعودية'],
      ['6001',  'حلواني إخوان'],
      ['6002',  'هرفي للأغذية'],
      ['6004',  'كاتريون'],
      ['6010',  'نادك'],
      ['6012',  'ريدان'],
      ['6013',  'التطويرية الغذائية'],
      ['6014',  'الآمار'],
      ['6015',  'أمريكانا'],
      ['6016',  'برغرايززر'],
      ['6017',  'جاهز'],
      ['6018',  'الأندية للرياضة'],
      ['6019',  'المسار الشامل'],
      ['6020',  'جاكو'],
      ['6040',  'تبوك الزراعية'],
      ['6050',  'الأسماك'],
      ['6060',  'الشرقية للتنمية'],
      ['6070',  'الجوف'],
      ['6090',  'جازادكو'],
      ['7010',  'اس تي سي'],
      ['7020',  'إتحاد إتصالات'],
      ['7030',  'زين السعودية'],
      ['7040',  'قو للإتصالات'],
      ['7200',  'ام آي اس'],
      ['7201',  'بحر العرب'],
      ['7202',  'سلوشنز'],
      ['7203',  'علم'],
      ['7204',  'توبي'],
      ['7211',  'عزم'],
      ['8010',  'التعاونية'],
      ['8012',  'جزيرة تكافل'],
      ['8020',  'ملاذ للتأمين'],
      ['8030',  'ميدغلف للتأمين'],
      ['8040',  'متكاملة'],
      ['8050',  'سلامة'],
      ['8060',  'ولاء'],
      ['8070',  'الدرع العربي'],
      ['8100',  'سايكو'],
      ['8120',  'إتحاد الخليج الأهلية'],
      ['8150',  'أسيج'],
      ['8160',  'التأمين العربية'],
      ['8170',  'الاتحاد'],
      ['8180',  'الصقر للتأمين'],
      ['8190',  'المتحدة للتأمين'],
      ['8200',  'الإعادة السعودية'],
      ['8210',  'بوبا العربية'],
      ['8230',  'تكافل الراجحي'],
      ['8240',  'تْشب'],
      ['8250',  'جي آي جي'],
      ['8260',  'الخليجية العامة'],
      ['8270',  'ليفا'],
      ['8280',  'ليفا'],
      ['8300',  'الوطنية'],
      ['8310',  'أمانة للتأمين'],
      ['8311',  'عناية'],
      ['8313',  'رسن'],
    ];

    function makeChip(code, name) {
      return `<div class="stock-chip"><span class="chip-code">${code}</span>${name}</div>`;
    }

    // Split into 3 rows
    const third = Math.ceil(STOCKS.length / 3);
    const rows = [
      STOCKS.slice(0, third),
      STOCKS.slice(third, third * 2),
      STOCKS.slice(third * 2),
    ];

    // Speeds in seconds per full one-set scroll (higher = slower)
    const speeds = [180, 200, 190];
    // Row 2 scrolls opposite direction
    const directions = [-1, 1, -1];

    ['bottom1', 'bottom2', 'bottom3'].forEach((id, i) => {
      const el = document.getElementById(id);
      if (!el) return;

      // Build chips — repeat 3× for guaranteed seamless loop on any screen
      const once = rows[i].map(([c, n]) => makeChip(c, n)).join('');
      el.innerHTML = once + once + once;

      // Wait for layout so scrollWidth is accurate
      requestAnimationFrame(() => {
        const oneSetWidth = el.scrollWidth / 3;
        const pxPerMs     = oneSetWidth / (speeds[i] * 1000);
        const dir         = directions[i];

        let pos  = dir === -1 ? 0 : oneSetWidth;
        let last = null;

        function tick(ts) {
          if (last !== null) {
            const delta = ts - last;
            pos += pxPerMs * delta * dir * -1;

            // Reset seamlessly when one full set has scrolled
            if (dir === -1 && pos <= -oneSetWidth) pos += oneSetWidth;
            if (dir ===  1 && pos >= 0)            pos -= oneSetWidth;
          }
          last = ts;
          el.style.transform = `translateX(${pos}px)`;
          requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
      });
    });
  </script>

</body>
</html>"""
# ────────────────────────────────────────────────────────────
# 14. HANDLERS
# ────────────────────────────────────────────────────────────
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        WELCOME_MSG, parse_mode="Markdown",
        reply_markup=build_tf_keyboard(),
    )


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "اختر الفاصل الزمني للفحص:",
        reply_markup=build_tf_keyboard(),
    )


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data  = query.data

    if data == "back_to_start":
        await query.edit_message_text(
            WELCOME_MSG, parse_mode="Markdown",
            reply_markup=build_tf_keyboard(),
        )
        return

    if data.startswith("scan_"):
        tf_key = data[5:]
        if tf_key not in TF_MAP:
            await query.edit_message_text("فاصل زمني غير معروف.")
            return
        await query.edit_message_text(
            f"⏱ الفاصل: *{TF_MAP[tf_key][0]}*\n\nاختر الفلتر:",
            parse_mode="Markdown",
            reply_markup=build_filter_keyboard(tf_key),
        )
        return

    if data.startswith("filter_"):
        parts = data.split("_", 2)
        if len(parts) != 3:
            await query.edit_message_text("بيانات غير صالحة.")
            return
        _, tf_key, direction = parts
        if tf_key not in TF_MAP:
            await query.edit_message_text("فاصل زمني غير معروف.")
            return

        tf_label, interval, period, resample_rule = TF_MAP[tf_key]
        chat_id = query.message.chat_id

        await query.edit_message_text(
            f"⏳ جاري فحص *{len(TADAWUL_TICKERS)}* سهم...\n"
            f"الفاصل: *{tf_label}*\n\nيرجى الانتظار ⏳",
            parse_mode="Markdown",
        )

        results, ohlc_data = scan_tickers(
            TADAWUL_TICKERS, period, interval, resample_rule
        )

        bullish_list = []
        bearish_list = []
        is_intraday  = interval not in ('1d', '1wk')

        for tk, patterns in results.items():
            for r in patterns:
                pct  = ((r['target_price'] - r['entry_price'])
                        / r['entry_price']) * 100
                item = {
                    'ticker':     tk,
                    'name':       get_name(tk),
                    'last_close': r['last_close'],
                    'entry':      round(r['entry_price'], 2),
                    'target':     r['target_price'],
                    'pct':        round(pct, 1),
                    'p5_date':    (
                        r['points'][5]['date'].strftime('%Y-%m-%d %H:%M')
                        if is_intraday
                        else r['points'][5]['date'].strftime('%Y-%m-%d')
                    ),
                    '_r':  r,
                    '_df': ohlc_data[tk],
                }
                if r['direction'] == 'Bullish':
                    bullish_list.append(item)
                else:
                    bearish_list.append(item)

        bullish_list.sort(key=lambda x: x['pct'], reverse=True)
        bearish_list.sort(key=lambda x: x['pct'])

        show_bull = direction in ('bullish', 'both')
        show_bear = direction in ('bearish', 'both')

        summary = f"✅ *اكتمل الفحص — {tf_label}*\n\n"
        if show_bull:
            summary += f"📈 ولفي صاعد: *{len(bullish_list)}*\n"
        if show_bear:
            summary += f"📉 ولفي هابط: *{len(bearish_list)}*\n"
        if not bullish_list and not bearish_list:
            summary += "\nلا توجد نتائج لهذا الفلتر."

        await context.bot.send_message(
            chat_id=chat_id, text=summary, parse_mode="Markdown"
        )

        # ── BULLISH RESULTS: chart first, then message ──────
        if show_bull and bullish_list:
            await context.bot.send_message(
                chat_id=chat_id,
                text="📈 *— نتائج الولفي الصاعد —*",
                parse_mode="Markdown",
            )
            for item in bullish_list:
                try:
                    buf = plot_wolfe_chart(
                        item['ticker'], item['_df'], item['_r'], tf_label
                    )
                    await context.bot.send_photo(chat_id=chat_id, photo=buf)
                except Exception as e:
                    logger.error(f"Chart error {item['ticker']}: {e}")
                msg = (
                    f"رمز السهم: *{item['ticker'].split('.')[0]}*\n"
                    f"الاسم       : `{item['name']}`\n"
                    f"الفاصل       : `{tf_label}`\n"
                    f"آخر إغلاق : `{item['last_close']}`\n"
                    f"قاع (5)    : `{item['entry']}`\n"
                    f"خط (1←4)  : `{item['target']}`\n"
                    f"النسبة      : `{item['pct']:+.1f}%`\n"
                    f"تاريخ (5)  : `{item['p5_date']}`"
                )
                await context.bot.send_message(
                    chat_id=chat_id, text=msg, parse_mode="Markdown"
                )

        # ── BEARISH RESULTS: chart first, then message ──────
        if show_bear and bearish_list:
            await context.bot.send_message(
                chat_id=chat_id,
                text="📉 *— نتائج الولفي الهابط —*",
                parse_mode="Markdown",
            )
            for item in bearish_list:
                try:
                    buf = plot_wolfe_chart(
                        item['ticker'], item['_df'], item['_r'], tf_label
                    )
                    await context.bot.send_photo(chat_id=chat_id, photo=buf)
                except Exception as e:
                    logger.error(f"Chart error {item['ticker']}: {e}")
                msg = (
                    f"رمز السهم: *{item['ticker'].split('.')[0]}*\n"
                    f"الاسم       : `{item['name']}`\n"
                    f"الفاصل       : `{tf_label}`\n"
                    f"آخر إغلاق : `{item['last_close']}`\n"
                    f"قمة (5)    : `{item['entry']}`\n"
                    f"خط (1←4)  : `{item['target']}`\n"
                    f"النسبة      : `{item['pct']:+.1f}%`\n"
                    f"تاريخ (5)  : `{item['p5_date']}`"
                )
                await context.bot.send_message(
                    chat_id=chat_id, text=msg, parse_mode="Markdown"
                )

        await context.bot.send_message(
            chat_id=chat_id,
            text="🔄 *انتهى الفحص — اضغط للبدء من جديد*",
            parse_mode="Markdown",
            reply_markup=build_start_keyboard(),
        )
        return

# ────────────────────────────────────────────────────────────
# 15. MAIN
# ────────────────────────────────────────────────────────────
def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("scan",  scan_command))
    app.add_handler(CallbackQueryHandler(button_handler))

    RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
    PORT = int(os.environ.get("PORT", 10000))

    if RENDER_EXTERNAL_URL:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"

        # ── aiohttp route: landing page ───────────────────────
        async def home(_request):
            return aio_web.Response(
                text=LANDING_HTML, content_type='text/html'
            )

        # ── aiohttp route: telegram webhook ──────────────────
        async def webhook_route(request):
            data   = await request.json()
            update = Update.de_json(data, app.bot)
            await app.update_queue.put(update)
            return aio_web.Response(text='OK')

        async def run_all():
            # Register webhook with Telegram
            await app.bot.set_webhook(webhook_url)
            logger.info(f"Webhook set → {webhook_url}")

            # Build aiohttp server with both routes
            web_app = aio_web.Application()
            web_app.router.add_get('/',         home)
            web_app.router.add_post('/webhook', webhook_route)

            runner = aio_web.AppRunner(web_app)
            await runner.setup()
            await aio_web.TCPSite(runner, '0.0.0.0', PORT).start()
            logger.info(f"Server listening on port {PORT}")

            # Start PTB application
            async with app:
                await app.start()
                await asyncio.Event().wait()   # run forever
                await app.stop()

        asyncio.run(run_all())

    else:
        logger.info("Starting polling (local dev)")
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
