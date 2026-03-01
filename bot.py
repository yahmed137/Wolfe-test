import os
import io
import logging
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
# 1. PIVOT DETECTION
# ────────────────────────────────────────────────────────────
def find_pivots(df, order=5):
    high = df['High'].values
    low  = df['Low'].values
    sh = argrelextrema(high, np.greater_equal, order=order)[0]
    sl = argrelextrema(low,  np.less_equal,    order=order)[0]
    pivots = []
    for i in sh:
        pivots.append({'bar': int(i), 'price': high[i], 'type': 'H', 'date': df.index[i]})
    for i in sl:
        pivots.append({'bar': int(i), 'price': low[i],  'type': 'L', 'date': df.index[i]})
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
# 2. UTILITIES
# ────────────────────────────────────────────────────────────
def line_at(x, x1, y1, x2, y2):
    if x2 == x1:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def resample_ohlc(df, rule):
    df_r = df.resample(rule).agg({
        'Open':   'first',
        'High':   'max',
        'Low':    'min',
        'Close':  'last',
        'Volume': 'sum',
    }).dropna()
    return df_r

# ────────────────────────────────────────────────────────────
# 3. WOLFE WAVE VALIDATORS
# ────────────────────────────────────────────────────────────
def validate_bullish(p1, p2, p3, p4, p5, tol=0.03):
    v = [p['price'] for p in [p1, p2, p3, p4, p5]]
    b = [p['bar']   for p in [p1, p2, p3, p4, p5]]

    if not (p1['type'] == 'L' and p2['type'] == 'H' and
            p3['type'] == 'L' and p4['type'] == 'H' and p5['type'] == 'L'):
        return None
    if v[2] >= v[0]:         return None
    if v[3] >= v[1]:         return None
    if v[3] <= v[0]:         return None
    if v[4] >= v[2]:         return None

    s13 = (v[2] - v[0]) / (b[2] - b[0]) if b[2] != b[0] else 0
    s24 = (v[3] - v[1]) / (b[3] - b[1]) if b[3] != b[1] else 0
    if s13 >= 0 or s24 >= 0: return None
    if s13 >= s24:           return None

    proj = line_at(b[4], b[0], v[0], b[2], v[2])
    if proj != 0:
        if (proj - v[4]) / abs(proj) < -tol:
            return None

    return {
        'direction':   'Bullish',
        'points':      [p1, p2, p3, p4, p5],
        'entry_price': v[4],
        'p5_date':     p5['date'],
    }


def validate_bearish(p1, p2, p3, p4, p5, tol=0.03):
    v = [p['price'] for p in [p1, p2, p3, p4, p5]]
    b = [p['bar']   for p in [p1, p2, p3, p4, p5]]

    if not (p1['type'] == 'H' and p2['type'] == 'L' and
            p3['type'] == 'H' and p4['type'] == 'L' and p5['type'] == 'H'):
        return None
    if v[2] <= v[0]:         return None
    if v[3] <= v[1]:         return None
    if v[3] >= v[0]:         return None
    if v[4] <= v[2]:         return None

    s13 = (v[2] - v[0]) / (b[2] - b[0]) if b[2] != b[0] else 0
    s24 = (v[3] - v[1]) / (b[3] - b[1]) if b[3] != b[1] else 0
    if s13 <= 0 or s24 <= 0: return None
    if s13 >= s24:           return None

    proj = line_at(b[4], b[0], v[0], b[2], v[2])
    if proj != 0:
        if (v[4] - proj) / abs(proj) < -tol:
            return None

    return {
        'direction':   'Bearish',
        'points':      [p1, p2, p3, p4, p5],
        'entry_price': v[4],
        'p5_date':     p5['date'],
    }

# ────────────────────────────────────────────────────────────
# 4. ACTIVE PATTERN FINDER
# ────────────────────────────────────────────────────────────
def find_active_wolfe(df, max_bars_since_p5=8):
    pivot_orders = [4, 5, 6, 7]
    n = len(df)
    best_bull = None
    best_bear = None

    for order in pivot_orders:
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

    out = []
    if best_bull:
        out.append(best_bull)
    if best_bear:
        out.append(best_bear)
    return out

# ────────────────────────────────────────────────────────────
# 5. CHART → PNG bytes
# ────────────────────────────────────────────────────────────
def plot_wolfe_chart(ticker, df, result, tf_label):
    pts       = result['points']
    direction = result['direction']
    entry     = result['entry_price']
    target    = result['target_price']
    is_bull   = direction == 'Bullish'

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

    # Wolfe wave lines
    ax.plot(zb, v, color=C_W, lw=2.5, zorder=6, alpha=0.8)
    ax.scatter(zb, v, s=120, c='white', edgecolors=C_W, linewidths=2.5, zorder=7)

    ext = zb[4] + 8
    ax.plot(
        [zb[0], ext],
        [v[0], line_at(ext + off, b[0], v[0], b[2], v[2])],
        color=C_W, lw=1.0, ls='--', alpha=0.3,
    )
    ax.plot(
        [zb[1], ext],
        [v[1], line_at(ext + off, b[1], v[1], b[3], v[3])],
        color=C_24, lw=1.0, ls='--', alpha=0.3,
    )

    fx = np.arange(zb[0], zb[4] + 1)
    f1 = [line_at(x + off, b[0], v[0], b[2], v[2]) for x in fx]
    f2 = [line_at(x + off, b[1], v[1], b[3], v[3]) for x in fx]
    ax.fill_between(fx, f1, f2, alpha=0.04, color=C_W)

    # Target line
    tgt_end_zb  = n_z + 5
    tgt_end_bar = tgt_end_zb + off
    ax.plot(
        [zb[0], tgt_end_zb],
        [v[0], line_at(tgt_end_bar, b[0], v[0], b[3], v[3])],
        color=C_T, lw=3.0, ls='-.', alpha=0.85, zorder=5,
    )

    z_last = min(last_bar - off, n_z - 1)
    ax.plot(z_last, target, marker='D', ms=14, color=C_T,
            markeredgecolor='white', markeredgewidth=2, zorder=9)
    ax.axhline(y=target, color=C_T, lw=0.6, ls=':', alpha=0.25)
    ax.axhline(y=entry,  color=C_E, lw=0.6, ls=':', alpha=0.25)

    # Arrow
    arrow_land_zb    = zb[4] + max(4, (z_last - zb[4]) // 2)
    arrow_land_zb    = min(arrow_land_zb, n_z + 3)
    arrow_land_price = line_at(arrow_land_zb + off, b[0], v[0], b[3], v[3])

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

    # Percent label
    price_range  = max(v) - min(v)
    label_offset = price_range * 0.08
    pct_y  = arrow_land_price + label_offset if is_bull else arrow_land_price - label_offset
    ax.text(
        arrow_land_zb, pct_y, f'{pct:+.1f}%',
        fontsize=13, fontweight='bold', color=C_A,
        ha='center', va='bottom' if is_bull else 'top',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_A, alpha=0.9, lw=0.8),
        zorder=10,
    )

    # P1–P5 labels
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
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_W, alpha=0.9, lw=0.6),
            arrowprops=dict(arrowstyle='-', color=C_W, lw=0.6),
        )

    emoji = '📈' if is_bull else '📉'
    ax.set_title(
        f'{emoji}   {ticker}   —   {direction} Wolfe Wave   |   Timeframe: {tf_label}',
        fontsize=16, fontweight='bold', pad=16, color='#212121',
    )
    ax.set_ylabel('')

    # Info box
    bc   = '#E8F5E9' if is_bull else '#FFEBEE'
    bt   = '#2E7D32' if is_bull else '#C62828'
    info = (
        f"  {direction.upper()} WOLFE WAVE\n"
        f"  ─────────────────────\n"
        f"  Last Close :  {last_close:.2f}\n"
        f"  Entry (P5)  :  {entry:.2f}\n"
        f"  Target 1→4 :  {target:.2f}\n"
        f"  Potential    :  {pct:+.1f}%\n"
        f"  Timeframe  :  {tf_label}"
    )
    ax.text(
        0.01, 0.03, info,
        transform=ax.transAxes,
        fontsize=10, fontfamily='monospace', fontweight='bold', color=bt,
        verticalalignment='bottom', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.6', facecolor=bc, edgecolor=bt, alpha=0.92, lw=1.2),
        zorder=10,
    )

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

# ────────────────────────────────────────────────────────────
# 6. TICKER PROCESSING
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
# 7. TICKERS
# ────────────────────────────────────────────────────────────
TADAWUL_TICKERS = [
    '1010.SR','1020.SR','1030.SR','1050.SR','1060.SR','1080.SR','1111.SR','1120.SR',
    '1140.SR','1150.SR','1180.SR','1182.SR','1183.SR','1201.SR','1202.SR','1210.SR',
    '1211.SR','1212.SR','1213.SR','1214.SR','1301.SR','1302.SR','1303.SR','1304.SR',
    '1320.SR','1321.SR','1322.SR','1810.SR','1820.SR','1830.SR','1831.SR','1832.SR',
    '1833.SR','2001.SR','2010.SR','2020.SR','2030.SR','2040.SR','2050.SR','2060.SR',
    '2070.SR','2080.SR','2081.SR','2082.SR','2083.SR','2090.SR','2100.SR','2110.SR',
    '2120.SR','2130.SR','2140.SR','2150.SR','2160.SR','2170.SR','2180.SR','2190.SR',
    '2200.SR','2210.SR','2220.SR','2222.SR','2223.SR','2230.SR','2240.SR','2250.SR',
    '2270.SR','2280.SR','2281.SR','2282.SR','2283.SR','2290.SR','2300.SR','2310.SR',
    '2320.SR','2330.SR','2340.SR','2350.SR','2360.SR','2370.SR','2380.SR','2381.SR',
    '2382.SR','3002.SR','3003.SR','3004.SR','3005.SR','3007.SR','3008.SR','3010.SR',
    '3020.SR','3030.SR','3040.SR','3050.SR','3060.SR','3080.SR','3090.SR','3091.SR',
    '3092.SR','4001.SR','4002.SR','4003.SR','4004.SR','4005.SR','4006.SR','4007.SR',
    '4008.SR','4009.SR','4011.SR','4012.SR','4013.SR','4014.SR','4015.SR','4020.SR',
    '4030.SR','4031.SR','4040.SR','4050.SR','4051.SR','4061.SR','4070.SR','4071.SR',
    '4080.SR','4081.SR','4082.SR','4090.SR','4100.SR','4110.SR','4130.SR','4140.SR',
    '4141.SR','4142.SR','4150.SR','4160.SR','4161.SR','4162.SR','4163.SR','4164.SR',
    '4170.SR','4180.SR','4190.SR','4191.SR','4192.SR','4200.SR','4210.SR','4220.SR',
    '4230.SR','4240.SR','4250.SR','4260.SR','4261.SR','4262.SR','4263.SR','4270.SR',
    '4280.SR','4290.SR','4291.SR','4292.SR','4300.SR','4310.SR','4320.SR','4321.SR',
    '4322.SR','4323.SR','4330.SR','4331.SR','4332.SR','4333.SR','4334.SR','4335.SR',
    '4336.SR','4337.SR','4338.SR','4339.SR','4340.SR','4342.SR','4344.SR','4345.SR',
    '4346.SR','4347.SR','4348.SR','4349.SR','5110.SR','6001.SR','6002.SR','6004.SR',
    '6010.SR','6012.SR','6013.SR','6014.SR','6015.SR','6020.SR','6040.SR','6050.SR',
    '6060.SR','6070.SR','6090.SR','7010.SR','7020.SR','7030.SR','7040.SR','7200.SR',
    '7201.SR','7202.SR','7203.SR','7204.SR','8010.SR','8012.SR','8020.SR','8030.SR',
    '8040.SR','8050.SR','8060.SR','8070.SR','8100.SR','8120.SR','8150.SR','8160.SR',
    '8170.SR','8180.SR','8190.SR','8200.SR','8210.SR','8230.SR','8240.SR','8250.SR',
    '8260.SR','8270.SR','8280.SR','8300.SR','8310.SR','8311.SR',
]

# ────────────────────────────────────────────────────────────
# 8. TIMEFRAME MAP
# ────────────────────────────────────────────────────────────
# key → (label, yf_interval, yf_period, resample_rule)
TF_MAP = {
    '30m': ('30 دقيقة', '30m',  '60d', None),
    '1h':  ('1 ساعة',   '60m',  '60d', None),
    '2h':  ('2 ساعة',   '60m',  '60d', '2h'),
    '4h':  ('4 ساعات',  '60m',  '60d', '4h'),
    '1d':  ('يومي',     '1d',   '1y',  None),
    '1w':  ('أسبوعي',   '1wk',  '5y',  None),
}

# ────────────────────────────────────────────────────────────
# 9. KEYBOARDS
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
    # callback format:  filter_{tf_key}_{direction}
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📈 صاعد فقط", callback_data=f"filter_{tf_key}_bullish"),
            InlineKeyboardButton("📉 هابط فقط", callback_data=f"filter_{tf_key}_bearish"),
        ],
        [
            InlineKeyboardButton("📊 الكل", callback_data=f"filter_{tf_key}_both"),
        ],
        [
            InlineKeyboardButton("🔙 رجوع", callback_data="back_to_start"),
        ],
    ])

# ────────────────────────────────────────────────────────────
# 10. WELCOME MESSAGE
# ────────────────────────────────────────────────────────────
WELCOME_MSG = (
    "🎯 *فاحص موجات الولفي ويف — السوق السعودي*\n\n"
    "اختر الفاصل الزمني للفحص:\n\n"
    "⚠️ تنبيه: هذا بحث عن موجات الولفي ويف فقط، "
    "لا يجب الاعتماد عليه وقد يكون خطأ. "
    "يجب متابعة الحركة السعرية."
)

# ────────────────────────────────────────────────────────────
# 11. HANDLERS
# ────────────────────────────────────────────────────────────
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        WELCOME_MSG,
        parse_mode="Markdown",
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
    data = query.data

    # ── Back to start ────────────────────────────────────────
    if data == "back_to_start":
        await query.edit_message_text(
            WELCOME_MSG,
            parse_mode="Markdown",
            reply_markup=build_tf_keyboard(),
        )
        return

    # ── Timeframe selected → show filter buttons ─────────────
    if data.startswith("scan_"):
        tf_key = data[5:]                       # e.g. "30m"
        if tf_key not in TF_MAP:
            await query.edit_message_text("فاصل زمني غير معروف.")
            return
        tf_label = TF_MAP[tf_key][0]
        await query.edit_message_text(
            f"⏱ الفاصل: *{tf_label}*\n\nاختر الفلتر:",
            parse_mode="Markdown",
            reply_markup=build_filter_keyboard(tf_key),
        )
        return

    # ── Filter selected → run scan ───────────────────────────
    # callback format:  filter_{tf_key}_{direction}
    if data.startswith("filter_"):
        parts = data.split("_", 2)              # ['filter', tf_key, direction]
        if len(parts) != 3:
            await query.edit_message_text("بيانات غير صالحة.")
            return
        _, tf_key, direction = parts            # direction: bullish/bearish/both

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

        # ── Run the scan ─────────────────────────────────────
        results, ohlc_data = scan_tickers(
            TADAWUL_TICKERS, period, interval, resample_rule
        )

        # ── Categorise results ────────────────────────────────
        bullish_list = []
        bearish_list = []
        is_intraday  = interval not in ('1d', '1wk')

        for tk, patterns in results.items():
            for r in patterns:
                pct  = ((r['target_price'] - r['entry_price']) / r['entry_price']) * 100
                item = {
                    'ticker':     tk,
                    'last_close': r['last_close'],
                    'entry':      round(r['entry_price'], 2),
                    'target':     r['target_price'],
                    'pct':        round(pct, 1),
                    'p5_date':    (r['points'][4]['date'].strftime('%Y-%m-%d %H:%M')
                                   if is_intraday
                                   else r['points'][4]['date'].strftime('%Y-%m-%d')),
                    '_r':         r,
                    '_df':        ohlc_data[tk],
                }
                if r['direction'] == 'Bullish':
                    bullish_list.append(item)
                else:
                    bearish_list.append(item)

        bullish_list.sort(key=lambda x: x['pct'], reverse=True)
        bearish_list.sort(key=lambda x: x['pct'])

        show_bull = direction in ('bullish', 'both')
        show_bear = direction in ('bearish', 'both')

        # ── Summary ───────────────────────────────────────────
        summary = f"✅ *اكتمل الفحص — {tf_label}*\n\n"
        if show_bull:
            summary += f"📈 ولفي صاعد: *{len(bullish_list)}*\n"
        if show_bear:
            summary += f"📉 ولفي هابط: *{len(bearish_list)}*\n"
        if not bullish_list and not bearish_list:
            summary += "\nلا توجد نتائج لهذا الفلتر."

        await context.bot.send_message(
            chat_id=chat_id,
            text=summary,
            parse_mode="Markdown",
            reply_markup=build_tf_keyboard(),
        )

        # ── Bullish results ───────────────────────────────────
        if show_bull and bullish_list:
            await context.bot.send_message(
                chat_id=chat_id,
                text="📈 *— نتائج الولفي الصاعد —*",
                parse_mode="Markdown",
            )
            for item in bullish_list:
                msg = (
                    f"📈 *{item['ticker']}*\n"
                    f"آخر إغلاق : `{item['last_close']}`\n"
                    f"دخول (P5) : `{item['entry']}`\n"
                    f"هدف (1→4) : `{item['target']}`\n"
                    f"النسبة     : `{item['pct']:+.1f}%`\n"
                    f"تاريخ P5   : `{item['p5_date']}`"
                )
                await context.bot.send_message(
                    chat_id=chat_id, text=msg, parse_mode="Markdown"
                )
                try:
                    buf = plot_wolfe_chart(
                        item['ticker'], item['_df'], item['_r'], tf_label
                    )
                    await context.bot.send_photo(chat_id=chat_id, photo=buf)
                except Exception as e:
                    logger.error(f"Chart error {item['ticker']}: {e}")

        # ── Bearish results ───────────────────────────────────
        if show_bear and bearish_list:
            await context.bot.send_message(
                chat_id=chat_id,
                text="📉 *— نتائج الولفي الهابط —*",
                parse_mode="Markdown",
            )
            for item in bearish_list:
                msg = (
                    f"📉 *{item['ticker']}*\n"
                    f"آخر إغلاق : `{item['last_close']}`\n"
                    f"دخول (P5) : `{item['entry']}`\n"
                    f"هدف (1→4) : `{item['target']}`\n"
                    f"النسبة     : `{item['pct']:+.1f}%`\n"
                    f"تاريخ P5   : `{item['p5_date']}`"
                )
                await context.bot.send_message(
                    chat_id=chat_id, text=msg, parse_mode="Markdown"
                )
                try:
                    buf = plot_wolfe_chart(
                        item['ticker'], item['_df'], item['_r'], tf_label
                    )
                    await context.bot.send_photo(chat_id=chat_id, photo=buf)
                except Exception as e:
                    logger.error(f"Chart error {item['ticker']}: {e}")
###############################################################                    
        # ── زر العودة للبداية بعد انتهاء النتائج ─────────────
        await context.bot.send_message(
            chat_id=chat_id,
            text="🔄 *انتهى الفحص — اضغط للبدء من جديد*",
            parse_mode="Markdown",
            reply_markup=build_start_keyboard(),
        )
        return
############################################################
# ────────────────────────────────────────────────────────────
# 12. MAIN — Webhook (Render) / Polling (local)
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
        logger.info(f"Starting webhook → {webhook_url}  (port {PORT})")
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path="/webhook",
            webhook_url=webhook_url,
        )
    else:
        logger.info("Starting polling (local dev)")
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
