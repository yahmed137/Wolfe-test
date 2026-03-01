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
    'TASI.SR':'تاسي',
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
# 5. WOLFE WAVE VALIDATORS
# ────────────────────────────────────────────────────────────
def validate_bullish(p1, p2, p3, p4, p5, tol=0.03):
    v = [p['price'] for p in [p1, p2, p3, p4, p5]]
    b = [p['bar']   for p in [p1, p2, p3, p4, p5]]

    if not (p1['type'] == 'L' and p2['type'] == 'H' and
            p3['type'] == 'L' and p4['type'] == 'H' and p5['type'] == 'L'):
        return None
    if v[2] >= v[0]:          return None
    if v[3] >= v[1]:          return None
    if v[3] <= v[0]:          return None
    if v[4] >= v[2]:          return None

    s13 = (v[2]-v[0])/(b[2]-b[0]) if b[2] != b[0] else 0
    s24 = (v[3]-v[1])/(b[3]-b[1]) if b[3] != b[1] else 0
    if s13 >= 0 or s24 >= 0:  return None
    if s13 >= s24:            return None

    proj = line_at(b[4], b[0], v[0], b[2], v[2])
    if proj != 0 and (proj - v[4]) / abs(proj) < -tol:
        return None

    return {'direction': 'Bullish', 'points': [p1,p2,p3,p4,p5],
            'entry_price': v[4], 'p5_date': p5['date']}


def validate_bearish(p1, p2, p3, p4, p5, tol=0.03):
    v = [p['price'] for p in [p1, p2, p3, p4, p5]]
    b = [p['bar']   for p in [p1, p2, p3, p4, p5]]

    if not (p1['type'] == 'H' and p2['type'] == 'L' and
            p3['type'] == 'H' and p4['type'] == 'L' and p5['type'] == 'H'):
        return None
    if v[2] <= v[0]:          return None
    if v[3] <= v[1]:          return None
    if v[3] >= v[0]:          return None
    if v[4] <= v[2]:          return None

    s13 = (v[2]-v[0])/(b[2]-b[0]) if b[2] != b[0] else 0
    s24 = (v[3]-v[1])/(b[3]-b[1]) if b[3] != b[1] else 0
    if s13 <= 0 or s24 <= 0:  return None
    if s13 >= s24:            return None

    proj = line_at(b[4], b[0], v[0], b[2], v[2])
    if proj != 0 and (v[4] - proj) / abs(proj) < -tol:
        return None

    return {'direction': 'Bearish', 'points': [p1,p2,p3,p4,p5],
            'entry_price': v[4], 'p5_date': p5['date']}

# ────────────────────────────────────────────────────────────
# 6. ACTIVE PATTERN FINDER
# ────────────────────────────────────────────────────────────
def find_active_wolfe(df, max_bars_since_p5=8):
    n         = len(df)
    best_bull = None
    best_bear = None

    for order in [4, 5, 6, 7]:
        piv = get_alternating_pivots(find_pivots(df, order=order))
        if len(piv) < 5:
            continue
        for offset in range(min(4, len(piv) - 4)):
            idx = len(piv) - 5 - offset
            if idx < 0:
                break
            combo = piv[idx: idx + 5]
            if n - 1 - combo[4]['bar'] > max_bars_since_p5:
                continue
            r = validate_bullish(*combo)
            if r and (best_bull is None or
                      combo[4]['bar'] > best_bull['points'][4]['bar']):
                best_bull = r
            r = validate_bearish(*combo)
            if r and (best_bear is None or
                      combo[4]['bar'] > best_bear['points'][4]['bar']):
                best_bear = r

    return [x for x in [best_bull, best_bear] if x]

# ────────────────────────────────────────────────────────────
# 7. CHART → PNG bytes
# ────────────────────────────────────────────────────────────
def plot_wolfe_chart(ticker, df, result, tf_label):
    pts          = result['points']
    direction    = result['direction']
    entry        = result['entry_price']
    target       = result['target_price']
    is_bull      = direction == 'Bullish'
    company      = get_name(ticker)
    ticker_code  = ticker.split('.')[0]

    b          = [p['bar']   for p in pts]
    v          = [p['price'] for p in pts]
    last_bar   = len(df) - 1
    last_close = float(df['Close'].iloc[-1])
    pct        = ((target - entry) / entry) * 100

    pad_l = max(0, b[0] - 10)
    pad_r = min(last_bar, b[4] + 30)
    df_z  = df.iloc[pad_l: pad_r + 1].copy()
    off   = pad_l
    zb    = [x - off for x in b]
    n_z   = len(df_z)

    C_W  = '#0D47A1' if is_bull else '#B71C1C'
    C_T  = '#2E7D32' if is_bull else '#C62828'
    C_24 = '#E65100'
    C_E  = '#6A1B9A'
    C_A  = '#00695C' if is_bull else '#880E4F'

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

    ax.plot(zb, v, color=C_W, lw=2.5, zorder=6, alpha=0.8)
    ax.scatter(zb, v, s=120, c='white', edgecolors=C_W,
               linewidths=2.5, zorder=7)

    ext = zb[4] + 8
    ax.plot([zb[0], ext],
            [v[0], line_at(ext+off, b[0], v[0], b[2], v[2])],
            color=C_W, lw=1.0, ls='--', alpha=0.3)
    ax.plot([zb[1], ext],
            [v[1], line_at(ext+off, b[1], v[1], b[3], v[3])],
            color=C_24, lw=1.0, ls='--', alpha=0.3)

    fx = np.arange(zb[0], zb[4] + 1)
    f1 = [line_at(x+off, b[0], v[0], b[2], v[2]) for x in fx]
    f2 = [line_at(x+off, b[1], v[1], b[3], v[3]) for x in fx]
    ax.fill_between(fx, f1, f2, alpha=0.04, color=C_W)

    tgt_end_zb = n_z + 5
    ax.plot([zb[0], tgt_end_zb],
            [v[0], line_at(tgt_end_zb+off, b[0], v[0], b[3], v[3])],
            color=C_T, lw=3.0, ls='-.', alpha=0.85, zorder=5)

    z_last = min(last_bar - off, n_z - 1)
    ax.plot(z_last, target, marker='D', ms=14, color=C_T,
            markeredgecolor='white', markeredgewidth=2, zorder=9)
    ax.axhline(y=target, color=C_T, lw=0.6, ls=':', alpha=0.25)
    ax.axhline(y=entry,  color=C_E, lw=0.6, ls=':', alpha=0.25)

    arrow_land_zb    = min(zb[4] + max(4, (z_last-zb[4])//2), n_z+3)
    arrow_land_price = line_at(arrow_land_zb+off, b[0], v[0], b[3], v[3])

    ax.annotate(
        '',
        xy=(arrow_land_zb, arrow_land_price),
        xytext=(zb[4], entry),
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

    for i in range(5):
        is_low = pts[i]['type'] == 'L'
        dt_str = pts[i]['date'].strftime('%b %d')
        ax.annotate(
            f'P{i+1}  {v[i]:.2f}\n{dt_str}',
            xy=(zb[i], v[i]),
            xytext=(0, -28 if is_low else 28),
            textcoords='offset points',
            ha='center', va='top' if is_low else 'bottom',
            fontsize=8.5, fontweight='bold', color=C_W,
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec=C_W, alpha=0.9, lw=0.6),
            arrowprops=dict(arrowstyle='-', color=C_W, lw=0.6),
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
        f"  {last_close:.2f}  :  {ar('الإغلاق')}",
        f"  {entry:.2f}  :  {ar('الموجة 5')}",
        f"  {target:.2f}  :  {ar('1→4')}",
        f"  {pct:+.1f}%  :  {ar('النسبة')}",
        f"  {ar(tf_label)}  :  {ar('الفاصل')}",
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
# 8. TICKER PROCESSING
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
            b1 = r['points'][0]['bar'];  v1 = r['points'][0]['price']
            b4 = r['points'][3]['bar'];  v4 = r['points'][3]['price']
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
    'TASI.SR','1010.SR','1020.SR','1030.SR','1050.SR','1060.SR','1080.SR','1111.SR','1120.SR',
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
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
      background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .card {
      background: rgba(255,255,255,0.07);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(255,255,255,0.13);
      border-radius: 28px;
      padding: 56px 44px;
      text-align: center;
      max-width: 500px;
      width: 92%;
      box-shadow: 0 24px 70px rgba(0,0,0,0.45);
      color: white;
    }
    .logo { font-size: 76px; margin-bottom: 18px; }
    h1 {
      font-size: 26px;
      font-weight: 800;
      margin-bottom: 10px;
      letter-spacing: 0.5px;
    }
    .subtitle {
      font-size: 15px;
      color: rgba(255,255,255,0.6);
      line-height: 1.7;
      margin-bottom: 32px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(46,213,115,0.12);
      border: 1px solid rgba(46,213,115,0.35);
      color: #2ed573;
      padding: 9px 22px;
      border-radius: 50px;
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 36px;
    }
    .dot {
      width: 9px; height: 9px;
      background: #2ed573;
      border-radius: 50%;
      animation: blink 1.4s ease-in-out infinite;
    }
    @keyframes blink {
      0%,100% { opacity:1; transform:scale(1); }
      50%      { opacity:0.25; transform:scale(0.8); }
    }
    .divider {
      border: none;
      border-top: 1px solid rgba(255,255,255,0.1);
      margin: 28px 0;
    }
    .stats {
      display: flex;
      justify-content: center;
      gap: 28px;
      margin-bottom: 32px;
    }
    .stat { text-align: center; }
    .stat-num {
      font-size: 28px;
      font-weight: 800;
      color: #74b9ff;
    }
    .stat-lbl {
      font-size: 12px;
      color: rgba(255,255,255,0.45);
      margin-top: 2px;
    }
    .btn {
      display: inline-block;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      text-decoration: none;
      padding: 15px 36px;
      border-radius: 50px;
      font-size: 16px;
      font-weight: 700;
      box-shadow: 0 6px 24px rgba(102,126,234,0.45);
      transition: transform .2s, box-shadow .2s;
    }
    .btn:hover {
      transform: translateY(-3px);
      box-shadow: 0 10px 32px rgba(102,126,234,0.65);
    }
    .footer {
      margin-top: 32px;
      font-size: 11px;
      color: rgba(255,255,255,0.28);
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">📈</div>
    <h1>فاحص موجات الولفي ويف</h1>
    <p class="subtitle">
      بوت تيليغرام ذكي لفحص موجات الولفي ويف<br>
      على جميع أسهم السوق السعودي — تداول
    </p>

    <div class="badge">
      <div class="dot"></div>
      البوت يعمل الآن
    </div>

    <hr class="divider">

    <div class="stats">
      <div class="stat">
        <div class="stat-num">240+</div>
        <div class="stat-lbl">سهم مفحوص</div>
      </div>
      <div class="stat">
        <div class="stat-num">6</div>
        <div class="stat-lbl">أطر زمنية</div>
      </div>
      <div class="stat">
        <div class="stat-num">2</div>
        <div class="stat-lbl">نوع موجة</div>
      </div>
    </div>

    <a class="btn" href="https://t.me/BOT_USERNAME" target="_blank">
      🤖 &nbsp; فتح البوت في تيليغرام
    </a>

    <div class="footer">
      Wolfe Wave Scanner &nbsp;•&nbsp; Saudi Market (Tadawul) &nbsp;•&nbsp; 2026
    </div>
  </div>
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
                        r['points'][4]['date'].strftime('%Y-%m-%d %H:%M')
                        if is_intraday
                        else r['points'][4]['date'].strftime('%Y-%m-%d')
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
