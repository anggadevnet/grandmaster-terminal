import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import json
import os
import copy
import hashlib
import time   # <--- TAMBAHKAN BARIS INI

warnings.filterwarnings('ignore')

# ======================== CONSTANTS ========================
SAVE_DIR = r"C:\trading"
os.makedirs(SAVE_DIR, exist_ok=True)
WIB_TZ = timezone(timedelta(hours=7))

# ======================== PAGE CONFIG ========================
st.set_page_config(
    page_title="🏆 Holy Grail Spot Scanner v10.9 + Bandarmology",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 1400px; }
    .section-header { background: linear-gradient(90deg, #1a1f36, #2d3250); padding: 8px 16px;
                      border-radius: 8px; border-left: 4px solid #4a90d9; margin: 12px 0 8px 0; }
    div[data-testid="metric-container"] { background: #1e2130; border-radius: 8px;
                                           padding: 8px; border: 1px solid #2d3250; }
    div[data-testid="metric-container"] label { font-size: 0.78em; color: #aaa; }
    div[data-testid="metric-container"] div[data-testid="metric-value"] { font-size: 1.05em; }
    h3 { border-bottom: 1px solid #2d3250; padding-bottom: 4px; margin-top: 1.2rem; }
    .good { color: #00ff88; }
    .warning { color: #ffaa00; }
    .bad { color: #ff4444; }
</style>
""", unsafe_allow_html=True)

st.title("🏆 Holy Grail Spot Scanner v10.9 — Spot Market + Bandarmology Edition")
st.caption(
    "Spot Market Only · Fixed Trade Plan · Conservative Entry Max 10% · "
    "Bearish Guard (min bull_signals ≥ 5) · Calibrated Regime Weights (0.5–1.2) · "
    "MC mu=0 (Short-Term) · 15 Core Indicators · Hurst · Kalman-Bayes · H+0→H+7 · "
    "Altcoin-Tuned Parameters · Smart Money Score (Bandarmology) · FVG Detector"
)

# ======================== TIME HELPERS ========================
def wib_now():
    return datetime.now(timezone.utc).astimezone(WIB_TZ)

def wib_trading_day_start_utc(ref_dt_utc=None):
    if ref_dt_utc is None:
        ref_dt_utc = datetime.now(timezone.utc)
    if ref_dt_utc.tzinfo is None:
        ref_dt_utc = ref_dt_utc.replace(tzinfo=timezone.utc)
    day_start_utc = ref_dt_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    if ref_dt_utc < day_start_utc:
        day_start_utc -= timedelta(days=1)
    return day_start_utc

def wib_trading_day_date(ref_dt_utc=None):
    start = wib_trading_day_start_utc(ref_dt_utc)
    return (start + timedelta(hours=7)).date()

def current_trading_date_str():
    return str(wib_trading_day_date())

# ======================== DETERMINISTIC SEED ========================
def make_mc_seed(symbol: str, trading_date: str, day: int) -> int:
    raw = f"{symbol}|{trading_date}|{day}"
    return int(hashlib.md5(raw.encode()).hexdigest(), 16) % (2**31)

# ======================== SNAPSHOT SAVE / LOAD ========================
def snapshot_filename(symbol, exchange, trading_date):
    safe_sym = symbol.replace('/', '_').replace(':', '_')
    return os.path.join(SAVE_DIR, f"snapshot_{safe_sym}_{exchange}_{trading_date}.json")

def pred_filename(symbol, exchange, trading_date):
    safe_sym = symbol.replace('/', '_').replace(':', '_')
    return os.path.join(SAVE_DIR, f"pred_{safe_sym}_{exchange}_{trading_date}.json")

def _make_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()
                if not isinstance(v, (pd.DataFrame, np.ndarray))}
    if isinstance(obj, list):
        return [_make_serializable(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

def save_full_snapshot(symbol, exchange, trading_date, result):
    try:
        path = snapshot_filename(symbol, exchange, trading_date)
        keys_to_save = [
            'symbol', 'pred_locked_at', 'pred_locked_trading_date',
            'predictions_7d', 'pred_summary',
            'conservative_entry', 'aggressive_entry', 'stop_loss',
            'sl_pct', 'tp1', 'tp2', 'tp3',
            'tp1_pct_entry', 'tp2_pct_entry', 'tp3_pct_entry',
            'tp1_pct_current', 'tp2_pct_current', 'tp3_pct_current',
            'rr', 'risk', 'pos_size_pct', 'atr',
            'rsi', 'adx', 'supertrend_bull', 'cmf',
            'tsi', 'kst', 'aroon_osc', 'psar_bull',
            'hma', 'dema', 'elder_bull',
            'macd_bull_1d', 'obv_rising', 'above_ma200', 'above_vwap',
            'hurst', 'structure_desc', 'wyckoff_phase', 'wyckoff_msg',
            'poc', 'vah', 'val', 'divs_1d', 'candle_patterns',
            'momentum', 'bull_signals', 'vol_regime', 'vol_mult',
            'mtf_score', 'mtf_desc', 'current_price',
            'open_wib_today', 'high_wib_today', 'low_wib_today', 'close_wib_today',
            'spot_tradeable', 'smc_score', 'nearest_fvg',
        ]
        snapshot = {k: _make_serializable(result.get(k)) for k in keys_to_save}
        for k in ('supports', 'resistances'):
            raw = result.get(k, [])
            snapshot[k] = [_make_serializable(s) for s in raw]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
        return True
    except Exception:
        return False

def load_full_snapshot(symbol, exchange, trading_date):
    try:
        path = snapshot_filename(symbol, exchange, trading_date)
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def save_predictions(symbol, exchange, trading_date, pred_data):
    try:
        path = pred_filename(symbol, exchange, trading_date)
        serializable = [_make_serializable(p) for p in pred_data]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
        return True
    except Exception:
        return False

def load_predictions(symbol, exchange, trading_date):
    try:
        path = pred_filename(symbol, exchange, trading_date)
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def list_saved_predictions():
    try:
        files = [f for f in os.listdir(SAVE_DIR)
                 if (f.startswith('pred_') or f.startswith('snapshot_')) and f.endswith('.json')]
        return sorted(files, reverse=True)
    except Exception:
        return []

# ======================== FORMAT HELPERS ========================
def fmt_price(price):
    if price is None or (isinstance(price, float) and np.isnan(price)):
        return "N/A"
    price = float(price)
    if price == 0: return "0"
    if price >= 10000:      return f"{price:,.2f}"
    elif price >= 100:      return f"{price:,.3f}"
    elif price >= 1:        return f"{price:.4f}"
    elif price >= 0.01:     return f"{price:.5f}"
    elif price >= 0.0001:   return f"{price:.6f}"
    elif price >= 0.000001: return f"{price:.8f}"
    else:                   return f"{price:.10f}"

def fmt_pct(val, plus=True):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    sign = "+" if val >= 0 and plus else ""
    return f"{sign}{val:.2f}%"

def _day_name_id(weekday):
    return ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"][weekday]

# ======================== DATA FETCH ========================
@st.cache_data(ttl=180)
def fetch_ohlcv_cached(symbol, exchange_name, timeframe='1d', limit=300):
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            'timeout': 20000
        })
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv or len(ohlcv) < 5: return None
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.dropna().reset_index(drop=True)
        for col in ['open','high','low','close','volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception:
        return None

# ======================== UTILS ========================
def safe_get(v, default=np.nan):
    try:
        if isinstance(v, pd.Series): v = v.iloc[-1]
        if v is None or (isinstance(v, float) and np.isnan(v)): return default
        return v
    except Exception:
        return default

def rolling_slope(series, n=5):
    try:
        vals = series.tail(n).values
        if len(vals) < n or np.isnan(vals).any(): return 0.0
        return np.polyfit(range(n), vals, 1)[0]
    except Exception:
        return 0.0

# ======================== ADVANCED MATH TOOLS ========================

def hurst_exponent(series, min_lag=2, max_lag=20):
    try:
        ts = np.array(series, dtype=float)
        ts = ts[~np.isnan(ts)]
        if len(ts) < max_lag * 2: return 0.5
        lags = range(min_lag, min(max_lag, len(ts) // 2))
        rs_vals = []
        for lag in lags:
            sub_series = [ts[i:i+lag] for i in range(0, len(ts)-lag, lag)]
            rs_lag = []
            for sub in sub_series:
                if len(sub) < 2: continue
                mean_sub = np.mean(sub)
                dev = np.cumsum(sub - mean_sub)
                r = np.max(dev) - np.min(dev)
                s = np.std(sub, ddof=1)
                if s > 0: rs_lag.append(r / s)
            if rs_lag: rs_vals.append(np.mean(rs_lag))
        if len(rs_vals) < 3: return 0.5
        log_lags = np.log(list(lags)[:len(rs_vals)])
        log_rs   = np.log(rs_vals)
        H = np.polyfit(log_lags, log_rs, 1)[0]
        return float(np.clip(H, 0.01, 0.99))
    except Exception:
        return 0.5

def kalman_smooth(prices, pn=1e-4, mn=1e-2):
    try:
        n = len(prices); prices = np.array(prices, dtype=float)
        x = np.array([prices[0], 0.0]); P = np.eye(2)
        F = np.array([[1,1],[0,1]]); H = np.array([[1,0]])
        Q = np.eye(2)*pn; R = np.array([[mn]])
        sm = np.zeros(n); vl = np.zeros(n)
        for i in range(n):
            x = F@x; P = F@P@F.T+Q
            S = H@P@H.T+R; K = P@H.T@np.linalg.inv(S)
            y = prices[i]-(H@x)[0]
            x = x+(K@[[y]]).flatten(); P = (np.eye(2)-K@H)@P
            sm[i]=x[0]; vl[i]=x[1]
        return sm, vl
    except Exception:
        return prices, np.zeros(len(prices))

def kalman_with_uncertainty(prices, pn=1e-4, mn=1e-2):
    try:
        n = len(prices); prices = np.array(prices, dtype=float)
        x = np.array([prices[0], 0.0]); P = np.eye(2)
        F = np.array([[1,1],[0,1]]); H = np.array([[1,0]])
        Q = np.eye(2)*pn; R = np.array([[mn]])
        sm = np.zeros(n); vl = np.zeros(n); unc = np.zeros(n)
        for i in range(n):
            x = F@x; P = F@P@F.T+Q
            S = H@P@H.T+R; K = P@H.T@np.linalg.inv(S)
            y = prices[i]-(H@x)[0]
            x = x+(K@[[y]]).flatten(); P = (np.eye(2)-K@H)@P
            sm[i]=x[0]; vl[i]=x[1]; unc[i]=float(P[0,0])**0.5
        return sm, vl, unc
    except Exception:
        return prices, np.zeros(len(prices)), np.ones(len(prices))*0.01

def garman_klass_volatility(df, window=20):
    try:
        log_hl = np.log(df['high'] / df['low'].replace(0, np.nan)) ** 2
        log_co = np.log(df['close'] / df['open'].replace(0, np.nan)) ** 2
        gk = (0.5 * log_hl - (2*np.log(2)-1) * log_co).rolling(window).mean()
        return (gk * 252) ** 0.5
    except Exception:
        return pd.Series(np.full(len(df), 0.5), index=df.index)

def yang_zhang_volatility(df, window=20):
    try:
        k = 0.34 / (1.34 + (window+1)/(window-1))
        log_oc = np.log(df['open'] / df['close'].shift(1).replace(0, np.nan))
        log_co = np.log(df['close'] / df['open'].replace(0, np.nan))
        log_ho = np.log(df['high']  / df['open'].replace(0, np.nan))
        log_lo = np.log(df['low']   / df['open'].replace(0, np.nan))
        var_oc = log_oc.rolling(window).var()
        var_co = log_co.rolling(window).var()
        rs     = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window).mean()
        yz_var = var_oc + k*var_co + (1-k)*rs
        return (yz_var * 252) ** 0.5
    except Exception:
        return pd.Series(np.full(len(df), 0.5), index=df.index)

def monte_carlo_price_range(current_price, sigma, days, n_sims=2000, seed=42):
    """
    Use mu=0 for short-term spot market prediction.
    Historical mu is unreliable and introduces bias for 1-7 day forecasts.
    Only volatility (sigma) matters at this horizon.
    """
    try:
        rng = np.random.default_rng(seed)
        dt = 1.0
        mu = 0.0
        daily_sigma = sigma / np.sqrt(252)
        mc_band_mult = 0.42
        paths = np.zeros((n_sims, days+1))
        paths[:, 0] = current_price
        for t in range(1, days+1):
            z = rng.standard_normal(n_sims)
            paths[:, t] = paths[:, t-1] * np.exp(
                (mu - 0.5*daily_sigma**2)*dt + daily_sigma*np.sqrt(dt)*z
            )
        final = paths[:, -1]
        p50 = float(np.percentile(final, 50))
        p10_raw = float(np.percentile(final, 10))
        p90_raw = float(np.percentile(final, 90))
        p10 = p50 + (p10_raw - p50) * mc_band_mult / 0.5
        p90 = p50 + (p90_raw - p50) * mc_band_mult / 0.5
        return {
            'p10': p10,
            'p25': float(np.percentile(final, 25)),
            'p50': p50,
            'p75': float(np.percentile(final, 75)),
            'p90': p90,
        }
    except Exception:
        return {'p10': current_price*0.95, 'p25': current_price*0.98,
                'p50': current_price, 'p75': current_price*1.02, 'p90': current_price*1.05}

def bayesian_signal_update(prior_signal, likelihood_data, confidence=0.6):
    try:
        alpha = confidence
        posterior = alpha * likelihood_data + (1 - alpha) * prior_signal
        return float(np.clip(posterior, -1, 1))
    except Exception:
        return prior_signal

def regime_switching_signal(signals_dict, regime):
    """
    Only multiply weights for categories relevant to the regime.
    Cap all multipliers at 0.5–1.2.
    """
    regime_weights = {
        'trending': {
            'trend_following': 1.20, 'momentum': 1.15,
            'oscillators': 0.70, 'volume': 1.10, 'structure': 1.15,
        },
        'trending_volatile': {
            'trend_following': 1.15, 'momentum': 1.10,
            'oscillators': 0.80, 'volume': 1.20, 'structure': 1.10,
        },
        'ranging': {
            'trend_following': 0.60, 'momentum': 0.75,
            'oscillators': 1.20, 'volume': 1.05, 'structure': 0.85,
        },
        'squeeze': {
            'trend_following': 0.80, 'momentum': 1.20,
            'oscillators': 0.90, 'volume': 1.20, 'structure': 1.15,
        },
        'volatile': {
            'trend_following': 0.85, 'momentum': 0.80,
            'oscillators': 1.00, 'volume': 1.20, 'structure': 1.05,
        },
        'quiet': {
            'trend_following': 1.10, 'momentum': 0.85,
            'oscillators': 1.15, 'volume': 0.90, 'structure': 1.05,
        },
        'normal': {
            'trend_following': 1.0, 'momentum': 1.0,
            'oscillators': 1.0, 'volume': 1.0, 'structure': 1.0,
        },
    }
    category_map = {
        'supertrend': 'trend_following', 'ichimoku': 'trend_following',
        'psar': 'trend_following',       'hma_dema': 'trend_following',
        'adx': 'trend_following',        'weekly': 'trend_following',
        'donchian': 'trend_following',
        'macd': 'momentum',              'tsi': 'momentum',
        'kst': 'momentum',               'kalman_velocity': 'momentum',
        'momentum': 'momentum',
        'rsi': 'oscillators',            'bollinger': 'oscillators',
        'obv': 'volume',                 'cmf': 'volume',
        'vol_delta': 'volume',
        'market_structure': 'structure', 'hurst': 'structure',
        'candle': 'structure',
    }
    rw = regime_weights.get(regime, regime_weights['normal'])
    weighted = {}
    for k, (score, base_w, label) in signals_dict.items():
        cat = category_map.get(k, 'momentum')
        regime_mult = rw.get(cat, 1.0)
        regime_mult = float(np.clip(regime_mult, 0.5, 1.2))
        weighted[k] = (score, base_w * regime_mult, label)
    return weighted

def candle_proxy_microstructure(df_15m_today, df_15m_full):
    score = 0.0
    details = {'method': 'candle_proxy'}
    try:
        if df_15m_today is None or len(df_15m_today) < 4:
            return 0.0, details
        df = df_15m_today.copy()
        cp = float(df['close'].iloc[-1])
        typical = (df['high'] + df['low'] + df['close']) / 3
        cum_tv = (typical * df['volume']).cumsum()
        cum_v  = df['volume'].cumsum()
        vwap_intra = cum_tv / cum_v.replace(0, np.nan)
        vwap_val   = float(vwap_intra.iloc[-1]) if not pd.isna(vwap_intra.iloc[-1]) else cp
        vwap_dev   = (cp - vwap_val) / max(vwap_val, 1e-10)
        score += np.clip(vwap_dev * 30, -0.40, 0.40)
        details['vwap_dev'] = vwap_dev
        bull_vol = df['volume'][df['close'] >= df['open']].sum()
        bear_vol = df['volume'][df['close'] < df['open']].sum()
        total_vol = bull_vol + bear_vol
        ofi_proxy = (bull_vol - bear_vol) / max(total_vol, 1e-10)
        score += np.clip(ofi_proxy * 0.35, -0.35, 0.35)
        details['ofi_proxy'] = float(ofi_proxy)
        if len(df) >= 6:
            closes = df['close'].values
            v1 = (closes[-1] - closes[-3]) / max(closes[-3], 1e-10)
            v2 = (closes[-3] - closes[-5]) / max(closes[-5], 1e-10)
            accel = v1 - v2
            score += np.clip(accel * 50, -0.25, 0.25)
            details['close_accel'] = float(accel)
    except Exception:
        pass
    return float(np.clip(score, -1, 1)), details

def compute_fourier_cycle(series, n_harmonics=3):
    try:
        arr = np.array(series, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 20: return 10, 0.0
        trend = np.polyfit(range(len(arr)), arr, 1)
        detrended = arr - np.polyval(trend, range(len(arr)))
        fft_vals = np.fft.rfft(detrended)
        freqs    = np.fft.rfftfreq(len(detrended))
        power    = np.abs(fft_vals)**2
        power[0] = 0
        if len(power) < 2: return 10, 0.0
        dom_freq_idx = np.argmax(power[1:]) + 1
        dom_period   = int(1 / freqs[dom_freq_idx]) if freqs[dom_freq_idx] > 0 else 10
        dom_period   = max(3, min(dom_period, len(arr)//2))
        phase_pos = (len(arr) % dom_period) / dom_period
        phase_signal = np.sin(2 * np.pi * phase_pos)
        return dom_period, float(phase_signal)
    except Exception:
        return 10, 0.0

# ======================== CORE INDICATORS (15 focused for spot) ========================
def calculate_indicators(df):
    if df is None or len(df) < 50: return None
    try:
        df = df.copy()
        c = df['close']

        df['MA20']  = c.rolling(20).mean()
        df['MA50']  = c.rolling(50).mean()
        df['MA200'] = c.rolling(200).mean()
        df['EMA9']  = c.ewm(span=9,  adjust=False).mean()
        df['EMA21'] = c.ewm(span=21, adjust=False).mean()

        half_len = 10
        wh = c.rolling(half_len).apply(lambda x: np.average(x, weights=np.arange(1,len(x)+1)), raw=True)
        wf = c.rolling(half_len*2).apply(lambda x: np.average(x, weights=np.arange(1,len(x)+1)), raw=True)
        hma_raw = 2*wh - wf
        df['HMA'] = hma_raw.rolling(int(np.sqrt(half_len*2))).apply(
            lambda x: np.average(x, weights=np.arange(1,len(x)+1)), raw=True)

        ema1 = c.ewm(span=21, adjust=False).mean()
        ema2 = ema1.ewm(span=21, adjust=False).mean()
        df['DEMA'] = 2*ema1 - ema2

        delta = c.diff()
        gain = delta.where(delta>0, 0.0).ewm(com=13, adjust=False).mean()
        loss = (-delta.where(delta<0, 0.0)).ewm(com=13, adjust=False).mean()
        df['RSI'] = 100 - (100/(1 + gain/loss.replace(0, np.nan)))
        df['RSI_MA'] = df['RSI'].rolling(14).mean()

        exp12 = c.ewm(span=12, adjust=False).mean()
        exp26 = c.ewm(span=26, adjust=False).mean()
        df['MACD']      = exp12 - exp26
        df['Signal']    = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']

        pc2 = c.diff()
        ds_pc  = pc2.ewm(span=25,adjust=False).mean().ewm(span=13,adjust=False).mean()
        ds_apc = pc2.abs().ewm(span=25,adjust=False).mean().ewm(span=13,adjust=False).mean()
        df['TSI']        = 100 * ds_pc / ds_apc.replace(0, np.nan)
        df['TSI_Signal'] = df['TSI'].ewm(span=7, adjust=False).mean()

        r1 = c.pct_change(10)*100; r2 = c.pct_change(13)*100
        r3 = c.pct_change(14)*100; r4 = c.pct_change(15)*100
        df['KST'] = (r1.rolling(10).mean()*1 + r2.rolling(13).mean()*2 +
                     r3.rolling(14).mean()*3 + r4.rolling(15).mean()*4)
        df['KST_Signal'] = df['KST'].rolling(9).mean()

        df = _psar(df)

        df['Volume_MA20']  = df['volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['volume'] / df['Volume_MA20'].replace(0, np.nan)
        df['Vol_Delta'] = np.where(
            df['close'] > df['open'],
            df['volume']*(df['close']-df['open'])/(df['high']-df['low']+1e-10),
            -df['volume']*(df['open']-df['close'])/(df['high']-df['low']+1e-10)
        )
        df['Vol_Delta_MA'] = df['Vol_Delta'].rolling(10).mean()

        df['VWAP'] = (c*df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()

        hl = df['high']-df['low']
        hc = (df['high']-c.shift()).abs()
        lc = (df['low']-c.shift()).abs()
        tr = pd.concat([hl,hc,lc],axis=1).max(axis=1)
        df['TR']   = tr
        df['ATR']  = tr.rolling(14).mean()
        df['ATR7'] = tr.rolling(7).mean()
        df['ATR3'] = tr.rolling(3).mean()
        df['ATR_Ratio'] = df['ATR'] / df['ATR'].rolling(30).mean().replace(0,np.nan)

        df['BB_Middle'] = c.rolling(20).mean()
        bbs = c.rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2*bbs
        df['BB_Lower'] = df['BB_Middle'] - 2*bbs
        df['BB_Width'] = (df['BB_Upper']-df['BB_Lower'])/df['BB_Middle'].replace(0,np.nan)
        df['BB_Pct']   = (c-df['BB_Lower'])/(df['BB_Upper']-df['BB_Lower']).replace(0,np.nan)

        df['KC_Middle'] = c.ewm(span=20,adjust=False).mean()
        df['KC_Upper']  = df['KC_Middle'] + 1.5*df['ATR']
        df['KC_Lower']  = df['KC_Middle'] - 1.5*df['ATR']
        df['TT_Squeeze'] = (df['BB_Lower']>df['KC_Lower']) & (df['BB_Upper']<df['KC_Upper'])

        df['DC_High'] = df['high'].rolling(20).max()
        df['DC_Low']  = df['low'].rolling(20).min()

        log_ret     = np.log(c/c.shift(1))
        df['LogRet'] = log_ret
        df['HV20']  = log_ret.rolling(20).std() * np.sqrt(365)
        df['HV5']   = log_ret.rolling(5).std()  * np.sqrt(365)
        df['GK_Vol'] = garman_klass_volatility(df, window=20)
        df['YZ_Vol'] = yang_zhang_volatility(df, window=20)

        df['ROC_3']  = c.pct_change(3)*100
        df['ROC_5']  = c.pct_change(5)*100
        df['ROC']    = c.pct_change(10)*100
        df['ROC_1']  = c.pct_change(1)*100

        df['body']         = (c - df['open']).abs()
        df['candle_range'] = df['high'] - df['low']
        df['body_ratio']   = df['body'] / df['candle_range'].replace(0,np.nan)
        df['upper_wick']   = df['high'] - df[['close','open']].max(axis=1)
        df['lower_wick']   = df[['close','open']].min(axis=1) - df['low']
        df['is_bull']      = (c > df['open']).astype(int)

        atr_sum = tr.rolling(14).sum()
        hl14    = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        df['Choppiness'] = 100 * np.log10(atr_sum / hl14.replace(0,np.nan)) / np.log10(14)

        df['Pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['PP_R1'] = 2*df['Pivot'] - df['low'].shift(1)
        df['PP_S1'] = 2*df['Pivot'] - df['high'].shift(1)
        df['PP_R2'] = df['Pivot'] + (df['high'].shift(1) - df['low'].shift(1))
        df['PP_S2'] = df['Pivot'] - (df['high'].shift(1) - df['low'].shift(1))

        df['RVOL'] = df['volume'] / df['volume'].rolling(20).mean().replace(0, np.nan)

        df = _adx(df)
        df = _supertrend(df)
        df = _cmf(df)
        return df
    except Exception:
        return None

def _psar(df, step=0.02, max_step=0.2):
    try:
        df = df.copy(); n = len(df)
        psar = df['close'].values.copy()
        bull = np.ones(n, dtype=bool)
        af   = np.full(n, step)
        ep   = df['low'].values.copy()
        for i in range(2, n):
            psar[i] = psar[i-1] + af[i-1]*(ep[i-1]-psar[i-1])
            if bull[i-1]:
                if df['low'].values[i] < psar[i]:
                    bull[i]=False; psar[i]=ep[i-1]; ep[i]=df['low'].values[i]; af[i]=step
                else:
                    bull[i]=True
                    psar[i]=min(psar[i], df['low'].values[i-1],
                                df['low'].values[i-2] if i>=2 else df['low'].values[i-1])
                    if df['high'].values[i] > ep[i-1]:
                        ep[i]=df['high'].values[i]; af[i]=min(af[i-1]+step,max_step)
                    else: ep[i]=ep[i-1]; af[i]=af[i-1]
            else:
                if df['high'].values[i] > psar[i]:
                    bull[i]=True; psar[i]=ep[i-1]; ep[i]=df['high'].values[i]; af[i]=step
                else:
                    bull[i]=False
                    psar[i]=max(psar[i], df['high'].values[i-1],
                                df['high'].values[i-2] if i>=2 else df['high'].values[i-1])
                    if df['low'].values[i] < ep[i-1]:
                        ep[i]=df['low'].values[i]; af[i]=min(af[i-1]+step,max_step)
                    else: ep[i]=ep[i-1]; af[i]=af[i-1]
        df['PSAR']      = psar
        df['PSAR_Bull'] = bull.astype(int)
        return df
    except Exception:
        df['PSAR']=np.nan; df['PSAR_Bull']=0; return df

def _adx(df, period=14):
    try:
        df=df.copy(); h=df['high']; l=df['low']; c=df['close']
        pdm = h.diff().clip(lower=0)
        mdm = (-l.diff()).clip(lower=0)
        pdm[pdm<mdm]=0; mdm[mdm<=pdm]=0
        tr  = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        atr14= tr.ewm(alpha=1/period,adjust=False).mean()
        pdi = 100*pdm.ewm(alpha=1/period,adjust=False).mean()/atr14.replace(0,np.nan)
        mdi = 100*mdm.ewm(alpha=1/period,adjust=False).mean()/atr14.replace(0,np.nan)
        dx  = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
        df['ADX']      = dx.ewm(alpha=1/period,adjust=False).mean()
        df['Plus_DI']  = pdi
        df['Minus_DI'] = mdi
        return df
    except Exception:
        return df

def _supertrend(df, period=10, mult=3.0):
    try:
        df=df.copy()
        if 'ATR' not in df.columns: return df
        hl2 = (df['high']+df['low'])/2
        bu  = (hl2 + mult*df['ATR']).values.copy()
        bl  = (hl2 - mult*df['ATR']).values.copy()
        n   = len(df)
        fu,fl = bu.copy(), bl.copy()
        st  = np.full(n, np.nan)
        dr  = np.zeros(n)
        cls = df['close'].values
        for i in range(1,n):
            fu[i] = bu[i] if (bu[i]<fu[i-1] or cls[i-1]>fu[i-1]) else fu[i-1]
            fl[i] = bl[i] if (bl[i]>fl[i-1] or cls[i-1]<fl[i-1]) else fl[i-1]
            if np.isnan(st[i-1]):
                st[i]=fu[i]; dr[i]=-1
            elif st[i-1]==fu[i-1]:
                if cls[i]>fu[i]: st[i]=fl[i]; dr[i]=1
                else:            st[i]=fu[i]; dr[i]=-1
            else:
                if cls[i]<fl[i]: st[i]=fu[i]; dr[i]=-1
                else:            st[i]=fl[i]; dr[i]=1
        df['Supertrend']     = st
        df['Supertrend_Dir'] = dr
        return df
    except Exception:
        return df

def _cmf(df, period=20):
    try:
        df=df.copy()
        hl=(df['high']-df['low']).replace(0,np.nan)
        clv=((df['close']-df['low'])-(df['high']-df['close']))/hl
        mfv=clv*df['volume']
        df['CMF']=mfv.rolling(period).sum()/df['volume'].rolling(period).sum()
        return df
    except Exception:
        return df

def calculate_ichimoku(df, t=9, k=26, sb=52):
    if df is None or len(df)<max(t,k,sb): return df
    try:
        df=df.copy()
        df['tenkan']          = (df['high'].rolling(t).max()+df['low'].rolling(t).min())/2
        df['kijun']           = (df['high'].rolling(k).max()+df['low'].rolling(k).min())/2
        df['senkou_a']        = ((df['tenkan']+df['kijun'])/2).shift(k)
        sbh = df['high'].rolling(sb).max(); sbl = df['low'].rolling(sb).min()
        df['senkou_b']        = ((sbh+sbl)/2).shift(k)
        df['chikou']          = df['close'].shift(-k)
        df['future_senkou_a'] = (df['tenkan']+df['kijun'])/2
        df['future_senkou_b'] = (sbh+sbl)/2
        return df
    except Exception:
        return df

def calculate_obv(df):
    if df is None or len(df)<2: return df
    try:
        df=df.copy()
        diff = df['close'].diff()
        vs   = np.where(diff>0, df['volume'], np.where(diff<0, -df['volume'], 0))
        df['OBV']      = vs.cumsum()
        df['OBV_MA']   = pd.Series(df['OBV'].values).rolling(20).mean().values
        df['OBV_trend']= df['OBV'] > df['OBV_MA']
        ov = df['OBV'].tail(20).values
        if not np.isnan(ov).any() and np.abs(ov).mean()>0:
            df['OBV_slope'] = np.polyfit(range(20),ov,1)[0]/(np.abs(ov).mean()+1e-10)
        else:
            df['OBV_slope'] = 0.0
        return df
    except Exception:
        return df

# ======================== WIB TRADING DAY BUILDER ========================
def build_wib_trading_days_from_15m(df_15m, num_days=30):
    if df_15m is None or len(df_15m) < 5: return None
    try:
        df = df_15m.copy()
        df['ts_utc'] = df['timestamp']
        df['day_start_utc'] = df['ts_utc'].dt.floor('D')
        df['trading_date_wib'] = (df['day_start_utc'] + pd.Timedelta(hours=7)).dt.date
        grouped = df.groupby('trading_date_wib').agg(
            day_start_utc=('day_start_utc', 'first'),
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last'),
            volume=('volume', 'sum'),
            candle_count=('open', 'count'),
            last_ts=('ts_utc', 'last'),
        ).reset_index()
        grouped = grouped.sort_values('trading_date_wib').reset_index(drop=True)
        if num_days and len(grouped) > num_days:
            grouped = grouped.tail(num_days).reset_index(drop=True)
        return grouped
    except Exception:
        return None

def build_wib_daily_df(df_15m, df_d_raw):
    if df_15m is not None and len(df_15m) >= 30:
        wib_daily = build_wib_trading_days_from_15m(df_15m, num_days=60)
        if wib_daily is not None and len(wib_daily) >= 30:
            wib_daily_df = pd.DataFrame({
                'timestamp': pd.to_datetime(wib_daily['day_start_utc'], utc=True),
                'open':   wib_daily['open'].astype(float),
                'high':   wib_daily['high'].astype(float),
                'low':    wib_daily['low'].astype(float),
                'close':  wib_daily['close'].astype(float),
                'volume': wib_daily['volume'].astype(float),
            }).reset_index(drop=True)
            return wib_daily_df, 'wib_15m_aggregated'
    if df_d_raw is not None and len(df_d_raw) >= 30:
        return df_d_raw.copy(), 'exchange_1d_fallback'
    return None, 'unavailable'

# ======================== SESSION BIAS ========================
def compute_session_bias(df_1h):
    if df_1h is None or len(df_1h) < 48:
        return 0.0, "N/A"
    try:
        df = df_1h.copy()
        df['hour_utc'] = df['timestamp'].dt.hour
        recent = df.tail(5 * 24)
        asia   = recent[recent['hour_utc'].between(0, 6)]
        london = recent[recent['hour_utc'].between(7, 15)]
        ny     = recent[recent['hour_utc'].between(13, 21)]
        bias = 0.0
        if len(asia) >= 3 and len(london) >= 3:
            asia_mid   = (asia['high'].mean() + asia['low'].mean()) / 2
            london_cls = london['close'].mean()
            if london_cls > asia_mid: bias += 0.30
            else: bias -= 0.30
        if len(london) >= 3 and len(ny) >= 3:
            london_mid = (london['high'].mean() + london['low'].mean()) / 2
            ny_cls     = ny['close'].mean()
            if ny_cls > london_mid: bias += 0.20
            else: bias -= 0.20
        if len(recent) >= 6:
            last6h = recent.tail(6)['close'].values
            if not np.isnan(last6h).any() and len(last6h) >= 2:
                trend6h = (last6h[-1] - last6h[0]) / max(abs(last6h[0]), 1e-10)
                bias += np.clip(trend6h * 15, -0.30, 0.30)
        bias = float(np.clip(bias, -1, 1))
        desc = "Bullish Session" if bias > 0.2 else ("Bearish Session" if bias < -0.2 else "Neutral Session")
        return bias, desc
    except Exception:
        return 0.0, "N/A"

# ======================== LIQUIDITY (SMC) ========================
def compute_liquidity_levels(df, lookback=100):
    if df is None or len(df) < 30:
        return {'bull_liq': [], 'bear_liq': [], 'order_blocks': [], 'score': 0.0}
    try:
        rec = df.tail(lookback).copy().reset_index(drop=True)
        cp  = float(rec['close'].iloc[-1])
        highs = rec['high'].values
        lows  = rec['low'].values
        bull_liq = []; bear_liq = []
        tol = 0.003
        for i in range(5, len(rec)-1):
            for j in range(i-5, i):
                if j < 0: continue
                if abs(highs[i] - highs[j]) / max(highs[j], 1e-10) < tol:
                    bear_liq.append(highs[i])
                if abs(lows[i] - lows[j]) / max(lows[j], 1e-10) < tol:
                    bull_liq.append(lows[i])
        order_blocks = []
        vma = rec['volume'].mean()
        for i in range(3, len(rec)-1):
            vol_ok  = rec['volume'].iloc[i] > vma * 1.5
            body    = abs(rec['close'].iloc[i] - rec['open'].iloc[i])
            rng     = rec['high'].iloc[i] - rec['low'].iloc[i]
            body_ok = body > rng * 0.6
            if vol_ok and body_ok:
                move = (rec['close'].iloc[i+1] - rec['close'].iloc[i]) / max(rec['close'].iloc[i], 1e-10)
                if abs(move) > 0.005:
                    order_blocks.append({
                        'price': float(rec['close'].iloc[i]),
                        'type': 'bull' if rec['close'].iloc[i] > rec['open'].iloc[i] else 'bear',
                        'idx': i
                    })
        score = 0.0
        near_buy  = [l2 for l2 in bull_liq if abs(l2-cp)/max(cp,1e-10)<0.02 and l2<cp]
        near_sell = [l2 for l2 in bear_liq if abs(l2-cp)/max(cp,1e-10)<0.02 and l2>cp]
        if near_buy:  score += 0.40
        if near_sell: score -= 0.40
        return {
            'bull_liq': bull_liq[:5], 'bear_liq': bear_liq[:5],
            'order_blocks': order_blocks[-5:], 'score': float(np.clip(score, -1, 1)),
        }
    except Exception:
        return {'bull_liq': [], 'bear_liq': [], 'order_blocks': [], 'score': 0.0}

# ======================== VOLATILITY REGIME ========================
def classify_volatility_regime(df):
    if df is None or len(df) < 30:
        return "unknown", 1.0
    try:
        last = df.iloc[-1]
        atr_ratio  = safe_get(last.get('ATR_Ratio', np.nan), 1.0)
        chop       = safe_get(last.get('Choppiness', np.nan), 50.0)
        hv20       = safe_get(last.get('HV20', np.nan), 0.5)
        hv5        = safe_get(last.get('HV5', np.nan), 0.5)
        tt_squeeze = bool(safe_get(last.get('TT_Squeeze', 0), 0))
        adx        = safe_get(last.get('ADX', np.nan), 20.0)
        is_trending = (chop < 45 and adx > 25)
        is_ranging  = (chop > 58 or tt_squeeze)
        is_volatile = (atr_ratio > 1.9 or (hv5/max(hv20,1e-10)) > 1.6)
        is_quiet    = (atr_ratio < 0.7 and not tt_squeeze)
        if tt_squeeze:
            regime = "squeeze"; mult = 1.15
        elif is_trending and is_volatile:
            regime = "trending_volatile"; mult = 1.25
        elif is_trending:
            regime = "trending"; mult = 1.10
        elif is_volatile:
            regime = "volatile"; mult = 0.85
        elif is_ranging:
            regime = "ranging"; mult = 0.75
        elif is_quiet:
            regime = "quiet"; mult = 0.90
        else:
            regime = "normal"; mult = 1.00
        return regime, mult
    except Exception:
        return "normal", 1.0

# ======================== MTF CONFLUENCE ========================
def compute_mtf_confluence(df_15m, df_1h, df_4h, df_1w):
    score = 0.0
    details = []
    def tf_score(df, label, weight):
        nonlocal score
        if df is None or len(df) < 20: return
        try:
            last = df.iloc[-1]
            s = 0.0
            ema21 = safe_get(last.get('EMA21', np.nan), 0)
            c2 = float(last['close'])
            if c2 > ema21 > 0: s += 0.4
            elif c2 < ema21:   s -= 0.4
            rsi = safe_get(last.get('RSI', np.nan), 50)
            if rsi < 40:   s += 0.3
            elif rsi > 65: s -= 0.3
            mh = safe_get(last.get('MACD_Hist', np.nan), 0)
            s += 0.3 if mh > 0 else -0.3
            s = float(np.clip(s, -1, 1))
            score += s * weight
            details.append(f"{label}: {s:+.2f}")
        except Exception:
            pass
    tf_score(df_15m, "15m", 0.15)
    tf_score(df_1h,  "1h",  0.25)
    tf_score(df_4h,  "4h",  0.35)
    tf_score(df_1w,  "1w",  0.25)
    score = float(np.clip(score, -1, 1))
    return score, " | ".join(details)

# ======================== FACTOR ENSEMBLE (15 core indicators) ========================
def compute_factor_ensemble(ind_df, df_weekly=None, df_1h=None, liq_data=None):
    if ind_df is None or len(ind_df) < 60: return {}
    last = ind_df.iloc[-1]
    cp   = float(last['close'])
    factors = {}

    sa   = safe_get(last.get('senkou_a', np.nan), cp)
    sb2  = safe_get(last.get('senkou_b', np.nan), cp)
    tk   = safe_get(last.get('tenkan',   np.nan), cp)
    kj   = safe_get(last.get('kijun',    np.nan), cp)
    ct   = max(sa, sb2)
    sc   = 0.0
    sc  += 0.40 if cp>ct else (-0.40 if cp<min(sa,sb2) else 0)
    sc  += 0.25 if tk>kj  else -0.20
    fa   = safe_get(last.get('future_senkou_a', np.nan), cp)
    fb   = safe_get(last.get('future_senkou_b', np.nan), cp)
    sc  += 0.20 if fa>fb  else -0.15
    factors['ichimoku'] = (np.clip(sc,-1,1), 2.5, "Ichimoku")

    sd = safe_get(last.get('Supertrend_Dir', np.nan), 0)
    ss = 0.90 if sd==1 else -0.90
    if len(ind_df)>=2 and safe_get(ind_df['Supertrend_Dir'].iloc[-2],0) != sd:
        ss = np.clip(ss*1.3,-1,1)
    factors['supertrend'] = (np.clip(ss,-1,1), 2.5, "Supertrend")

    adx = safe_get(last.get('ADX',np.nan),20)
    pdi = safe_get(last.get('Plus_DI',np.nan),20)
    mdi = safe_get(last.get('Minus_DI',np.nan),20)
    if adx>30:   as_ = min(1.0,adx/50)*(1 if pdi>mdi else -1)
    elif adx>20: as_ = 0.4*(1 if pdi>mdi else -1)
    else:        as_ = 0.1*(1 if pdi>mdi else -1)
    factors['adx'] = (np.clip(as_,-1,1), 2.0, f"ADX={adx:.1f}")

    ms = 0.0
    if 'MACD_Hist' in ind_df.columns:
        mh2  = ind_df['MACD_Hist'].tail(10).fillna(0)
        mh_l = float(mh2.iloc[-1])
        ms  += 0.35 if mh_l>0 else -0.35
        ms  += np.clip(rolling_slope(mh2,6)*300,-0.40,0.40)
        if len(mh2)>=3:
            ms += 0.25 if mh2.iloc[-3]<0<mh_l else (-0.25 if mh2.iloc[-3]>0>mh_l else 0)
    factors['macd'] = (np.clip(ms,-1,1), 2.0, "MACD")

    rsi    = safe_get(last.get('RSI',np.nan),50)
    rs_sl  = rolling_slope(ind_df['RSI'].fillna(50), 8)
    rsi_ma = safe_get(last.get('RSI_MA', np.nan), 50)
    if   rsi<25: rb=0.90
    elif rsi<35: rb=0.55
    elif rsi<45: rb=0.20
    elif rsi<55: rb=0.05
    elif rsi<65: rb=-0.05
    elif rsi<75: rb=-0.45
    else:        rb=-0.85
    rm = np.clip(rs_sl/5,-0.30,0.30)
    rm += 0.10 if rsi > rsi_ma else -0.10
    factors['rsi'] = (np.clip(rb+rm,-1,1), 2.0, f"RSI={rsi:.1f}")

    tsi  = safe_get(last.get('TSI',np.nan),0)
    tsig = safe_get(last.get('TSI_Signal',np.nan),0)
    factors['tsi'] = (np.clip((0.50 if tsi>tsig else -0.50)+np.clip(tsi/50,-0.50,0.50),-1,1), 1.8, f"TSI={tsi:.1f}")

    kst  = safe_get(last.get('KST',np.nan),0)
    ksig = safe_get(last.get('KST_Signal',np.nan),0)
    factors['kst'] = (np.clip((0.60 if kst>ksig else -0.60)+np.clip(kst/100,-0.40,0.40),-1,1), 1.5, f"KST={kst:.1f}")

    bbp = safe_get(last.get('BB_Pct',np.nan),0.5)
    bbw = safe_get(last.get('BB_Width',np.nan),0.05)
    abw = ind_df['BB_Width'].tail(30).mean() if 'BB_Width' in ind_df.columns else bbw
    bcr = bbw/max(abw,1e-10)
    if   bbp<0.10: bbs=0.75
    elif bbp<0.25: bbs=0.40
    elif bbp>0.90: bbs=-0.75
    elif bbp>0.75: bbs=-0.40
    else:          bbs=(0.5-bbp)*0.5
    if bcr<0.5: bbs += 0.15 if bbp>0.50 else -0.10
    factors['bollinger'] = (np.clip(bbs,-1,1), 1.5, f"BB%={bbp:.2f}")

    hma  = safe_get(last.get('HMA',np.nan),cp)
    dema = safe_get(last.get('DEMA',np.nan),cp)
    hs   = 0.0
    if not np.isnan(hma):
        hs += 0.45 if cp>hma else -0.45
        if len(ind_df)>=3 and 'HMA' in ind_df.columns:
            hsl = float(ind_df['HMA'].iloc[-1])-float(ind_df['HMA'].iloc[-3])
            hs += np.clip(hsl/(cp*0.01+1e-10),-0.35,0.35)
    da = 0.25 if (not np.isnan(dema) and cp>dema) else -0.25
    factors['hma_dema'] = (np.clip(hs+da*0.5,-1,1), 1.8, "HMA+DEMA")

    pb = int(safe_get(last.get('PSAR_Bull',np.nan),0))
    ps = 0.80 if pb==1 else -0.80
    if len(ind_df)>=2 and int(safe_get(ind_df['PSAR_Bull'].iloc[-2],0))!=pb:
        ps = np.clip(ps*1.2,-1,1)
    factors['psar'] = (ps, 1.8, "PSAR")

    cmf = safe_get(last.get('CMF',np.nan),0)
    cs  = np.clip(cmf*5,-0.75,0.75)
    factors['cmf'] = (np.clip(cs,-1,1), 1.5, f"CMF={cmf:.3f}")

    os_ = 0.0
    if 'OBV' in ind_df.columns and len(ind_df)>=20:
        ov2  = ind_df['OBV'].tail(20).values
        pv2  = ind_df['close'].tail(20).values
        if not (np.isnan(ov2).any() or np.isnan(pv2).any()):
            osl = np.polyfit(range(20),ov2,1)[0]
            psl = np.polyfit(range(20),pv2,1)[0]
            no  = osl/(np.abs(ov2).mean()+1e-10)
            np2 = psl/(np.abs(pv2).mean()+1e-10)
            if no>0 and np2>=0:   os_= min(1.0,no*20)
            elif no>0:            os_= 0.70
            elif no<0 and np2>0:  os_=-0.70
            else:                 os_= max(-1.0,no*20)
    factors['obv'] = (np.clip(os_,-1,1), 2.0, "OBV")

    vd = 0.0
    if 'Vol_Delta_MA' in ind_df.columns:
        v_ = safe_get(last.get('Vol_Delta_MA',np.nan),0)
        av = safe_get(ind_df['volume'].tail(20).mean(),1)
        vd = np.clip(v_/(av+1e-10),-1,1)*0.8
    factors['vol_delta'] = (np.clip(vd,-1,1), 1.5, "Vol Delta")

    try:
        ca  = ind_df['close'].tail(60).values.astype(float)
        _,vl= kalman_smooth(ca)
        kvs = np.clip(vl[-1]/(cp*0.005+1e-10),-1,1)
    except Exception:
        kvs = 0.0
    factors['kalman_velocity'] = (kvs, 2.0, "Kalman Velocity")

    try:
        h_series = ind_df['close'].tail(100).values
        H = hurst_exponent(h_series)
        roc5 = safe_get(last.get('ROC_5',np.nan),0)
        if H > 0.6:   h_s = 0.65 if roc5 > 0 else -0.65
        elif H < 0.4: h_s = -0.30 if roc5 > 0 else 0.30
        else:          h_s = 0.0
    except Exception:
        H = 0.5; h_s = 0.0
    factors['hurst'] = (np.clip(h_s,-1,1), 1.8, f"Hurst={H:.3f}")

    mss = 0.0
    if len(ind_df)>=50:
        hh2 = ind_df['high'].tail(50).values; ll2 = ind_df['low'].tail(50).values
        sh2,sl2 = [],[]
        for i in range(3,len(hh2)-3):
            if hh2[i]==max(hh2[i-3:i+4]): sh2.append(hh2[i])
            if ll2[i]==min(ll2[i-3:i+4]): sl2.append(ll2[i])
        if len(sh2)>=2: mss += 0.45 if sh2[-1]>sh2[-2] else -0.45
        if len(sl2)>=2: mss += 0.45 if sl2[-1]>sl2[-2] else -0.45
    factors['market_structure'] = (np.clip(mss,-1,1), 2.2, "Market Structure")

    ws = 0.0
    if df_weekly is not None and len(df_weekly)>=20:
        if 'HMA' not in df_weekly.columns:
            df_weekly = calculate_indicators(df_weekly)
        if df_weekly is not None:
            wl2  = df_weekly.iloc[-1]
            wm20 = df_weekly['close'].rolling(20).mean().iloc[-1]
            wc   = float(wl2['close'])
            ws  += 0.50 if wc>wm20 else -0.50
            if 'RSI' in df_weekly.columns:
                wr = safe_get(df_weekly['RSI'].iloc[-1],50)
                ws += 0.30 if wr<40 else (-0.30 if wr>65 else 0)
    factors['weekly'] = (np.clip(ws,-1,1), 2.0, "Weekly Bias")

    if df_1h is not None:
        sess_score, sess_desc = compute_session_bias(df_1h)
        factors['session'] = (np.clip(sess_score,-1,1), 1.2, f"Session({sess_desc})")

    css = 0.0
    if 'is_bull' in ind_df.columns and 'body_ratio' in ind_df.columns:
        b5  = ind_df['is_bull'].tail(5).mean()
        br5 = ind_df['body_ratio'].tail(5).mean()
        css = (b5-0.5)*0.70 + (br5-0.5)*0.30
    factors['candle'] = (np.clip(css,-1,1), 1.0, "Candle Momentum")

    dc_h = safe_get(last.get('DC_High', np.nan), cp)
    dc_l = safe_get(last.get('DC_Low', np.nan), cp)
    dc_s = 0.0
    if cp >= dc_h * 0.998: dc_s = 0.75
    elif cp <= dc_l * 1.002: dc_s = -0.75
    factors['donchian'] = (np.clip(dc_s,-1,1), 1.5, "Donchian")

    return factors

def compute_adaptive_weights(factors, adx_val, hv20, vol_regime="normal"):
    regime_factors = regime_switching_signal(factors, vol_regime)
    weights = {}
    for k, (score, w, label) in regime_factors.items():
        weights[k] = w
    if hv20 > 1.2:
        for k in ['obv', 'cmf', 'vol_delta']:
            if k in weights:
                weights[k] = min(weights[k] * 1.30, factors[k][1] * 1.5)
    return weights

# ======================== H+0 ULTRA-PRECISION ENGINE ========================
def predict_h0_wib_ultra(daily_wib_df, df_15m=None, df_1h=None, df_4h=None,
                          df_weekly=None, current_price=None):
    if daily_wib_df is None or len(daily_wib_df) < 30: return None
    try:
        last_d = daily_wib_df.iloc[-1]
        cp = float(current_price if current_price else last_d['close'])
        actual_h0 = {
            'open':  float(last_d['open']),
            'high':  float(last_d['high']),
            'low':   float(last_d['low']),
            'close': float(last_d['close']),
        }
        day_start_utc = wib_trading_day_start_utc()
        day_end_utc   = day_start_utc + timedelta(days=1)
        now_utc       = datetime.now(timezone.utc)

        df_15m_today = None
        if df_15m is not None and len(df_15m) > 0:
            mask = (df_15m['timestamp'] >= day_start_utc) & (df_15m['timestamp'] < day_end_utc)
            df_15m_today = df_15m[mask].copy().reset_index(drop=True)

        total_day_min = 24 * 60
        elapsed_min   = int((now_utc - day_start_utc).total_seconds() / 60)
        elapsed_min   = max(0, min(elapsed_min, total_day_min))
        remaining_min = max(0, total_day_min - elapsed_min)
        day_completion= elapsed_min / total_day_min

        if df_15m_today is not None and len(df_15m_today) > 0:
            today_open    = float(df_15m_today['open'].iloc[0])
            today_high_sf = float(df_15m_today['high'].max())
            today_low_sf  = float(df_15m_today['low'].min())
            today_last    = float(df_15m_today['close'].iloc[-1])
            data_source   = "15m intraday"
        elif df_1h is not None and len(df_1h) > 0:
            mask_1h = (df_1h['timestamp'] >= day_start_utc) & (df_1h['timestamp'] < day_end_utc)
            df_1h_today = df_1h[mask_1h]
            if len(df_1h_today) > 0:
                today_open    = float(df_1h_today['open'].iloc[0])
                today_high_sf = float(df_1h_today['high'].max())
                today_low_sf  = float(df_1h_today['low'].min())
                today_last    = float(df_1h_today['close'].iloc[-1])
                data_source   = "1H intraday"
            else:
                today_open=actual_h0['open']; today_high_sf=actual_h0['high']
                today_low_sf=actual_h0['low']; today_last=actual_h0['close']
                data_source="WIB daily fallback"
        else:
            today_open=actual_h0['open']; today_high_sf=actual_h0['high']
            today_low_sf=actual_h0['low']; today_last=actual_h0['close']
            data_source="WIB daily fallback"

        if remaining_min <= 30:
            return {
                'open': today_open, 'high': today_high_sf, 'low': today_low_sf,
                'close': today_last, 'confidence': 95.0,
                'method': f'{data_source} · Day Complete',
                'elapsed_min': elapsed_min, 'remaining_min': remaining_min,
                'day_completion': day_completion, 'actual_h0': actual_h0,
                'data_source': data_source, 'composite_signal': 0.0,
                'micro_momentum': 0.0, 'slope_15m_pct': 0.0, 'atr_15m': 0.0,
                'micro_score': 0.0, 'kal_vel_pct': 0.0, 'kal_uncertainty': 0.01,
                'mtf_score': 0.0, 'vol_regime': 'normal',
            }

        wib_daily_indicators = calculate_indicators(daily_wib_df)
        liq_data = None
        if wib_daily_indicators is not None:
            wib_daily_indicators = calculate_ichimoku(wib_daily_indicators)
            wib_daily_indicators = calculate_obv(wib_daily_indicators)
            liq_data = compute_liquidity_levels(wib_daily_indicators)
            raw_factors = compute_factor_ensemble(wib_daily_indicators, df_weekly, df_1h, liq_data)
        else:
            raw_factors = {}

        vol_regime, _ = classify_volatility_regime(wib_daily_indicators) if wib_daily_indicators is not None else ("normal", 1.0)

        if raw_factors:
            last_ind = wib_daily_indicators.iloc[-1]
            adx_val  = safe_get(last_ind.get('ADX', np.nan), 20)
            hv20     = safe_get(last_ind.get('HV20', np.nan), 0.5)
            adj_weights = compute_adaptive_weights(raw_factors, adx_val, hv20, vol_regime)
            fn = list(raw_factors.keys())
            fs = np.array([raw_factors[k][0] for k in fn])
            fw = np.array([adj_weights.get(k, raw_factors[k][1]) for k in fn])
            ws = float(np.sum(fs * fw) / max(fw.sum(), 1e-10))
        else:
            ws = 0.0; adx_val = 20; hv20 = 0.5

        mtf_score, _ = compute_mtf_confluence(df_15m, df_1h, df_4h, df_weekly)
        micro_score, micro_details = candle_proxy_microstructure(df_15m_today, df_15m)
        micro_momentum = micro_details.get('ofi_proxy', 0.0)

        slope_15m_pct = 0.0
        if df_15m_today is not None and len(df_15m_today) >= 6:
            recent_closes = df_15m_today['close'].tail(12).values
            if len(recent_closes) >= 4 and not np.isnan(recent_closes).any():
                x = np.arange(len(recent_closes))
                slope = np.polyfit(x, recent_closes, 1)[0]
                slope_15m_pct = slope / max(today_last, 1e-10)

        kal_vel_pct = 0.0
        kal_uncertainty = 0.01
        try:
            if df_15m is not None and len(df_15m) >= 60:
                ca = df_15m['close'].tail(200).values.astype(float)
                _, vl, unc = kalman_with_uncertainty(ca)
                kal_vel_pct = float(vl[-1]) / (today_last + 1e-10)
                kal_uncertainty = float(unc[-1]) / (today_last + 1e-10)
        except Exception:
            pass

        atr_15m = None
        if df_15m is not None and len(df_15m) >= 30:
            tr_15m = (df_15m['high'] - df_15m['low']).tail(50)
            atr_15m = float(tr_15m.mean())
        if atr_15m is None or atr_15m <= 0:
            last_ind2 = wib_daily_indicators.iloc[-1] if wib_daily_indicators is not None else None
            atr_daily = safe_get(last_ind2.get('ATR', np.nan) if last_ind2 is not None else np.nan, cp*0.025)
            atr_15m = atr_daily / np.sqrt(96)

        remaining_15m_bars = max(1, remaining_min // 15)
        expected_residual_range = atr_15m * np.sqrt(remaining_15m_bars)

        tech_signal  = ws
        ms_signal    = micro_score
        slope_signal = float(np.tanh(slope_15m_pct * 500))
        kal_signal   = float(np.tanh(kal_vel_pct * 100))
        mtf_signal   = mtf_score

        bayes_signal = bayesian_signal_update(tech_signal, ms_signal, confidence=0.50)
        bayes_signal = bayesian_signal_update(bayes_signal, slope_signal, confidence=0.40)

        dc = day_completion
        composite_signal = (
            (1 - dc) * (0.35 * tech_signal + 0.20 * ms_signal + 0.20 * slope_signal +
                         0.15 * kal_signal + 0.10 * mtf_signal) +
            dc * (0.15 * tech_signal + 0.40 * ms_signal + 0.30 * slope_signal +
                   0.10 * kal_signal + 0.05 * mtf_signal)
        )
        composite_signal = bayesian_signal_update(composite_signal, bayes_signal, confidence=0.5)
        composite_signal = float(np.clip(composite_signal, -1, 1))

        residual_drift = composite_signal * (expected_residual_range / max(today_last, 1e-10)) * 0.50
        projected_close = today_last * (1 + residual_drift)

        base_ext = expected_residual_range * 0.50
        if composite_signal > 0:
            ext_up = base_ext * (1 + composite_signal * 0.5)
            ext_dn = base_ext * max(0.3, 1 - composite_signal * 0.5)
        else:
            ext_dn = base_ext * (1 + abs(composite_signal) * 0.5)
            ext_up = base_ext * max(0.3, 1 - abs(composite_signal) * 0.5)

        potential_high = today_last + ext_up
        potential_low  = today_last - ext_dn
        projected_high = max(today_high_sf, potential_high, projected_close, today_open)
        projected_low  = min(today_low_sf,  potential_low,  projected_close, today_open)

        if raw_factors:
            fn2 = list(raw_factors.keys())
            fs2 = np.array([raw_factors[k][0] for k in fn2])
            agree_ratio = float(np.sum(np.sign(fs2) == np.sign(composite_signal))) / len(fs2)
        else:
            agree_ratio = 0.5

        base_conf = 50 + day_completion*20 + agree_ratio*10
        if adx_val > 25: base_conf += 3
        if abs(micro_score) > 0.2: base_conf += 3
        if abs(slope_15m_pct) > 0.0003: base_conf += 2
        if df_15m_today is not None and len(df_15m_today) >= 8: base_conf += 2
        if kal_uncertainty > 0.02: base_conf -= 4
        confidence = float(np.clip(base_conf, 50, 85))

        return {
            'open': today_open, 'high': projected_high,
            'low': projected_low, 'close': projected_close,
            'confidence': confidence,
            'method': f'{data_source} (Elapsed: {elapsed_min}min, Remaining: {remaining_min}min)',
            'elapsed_min': elapsed_min, 'remaining_min': remaining_min,
            'day_completion': day_completion, 'composite_signal': composite_signal,
            'micro_momentum': micro_momentum, 'micro_score': micro_score,
            'slope_15m_pct': slope_15m_pct, 'atr_15m': atr_15m,
            'actual_h0': actual_h0, 'data_source': data_source,
            'vol_regime': vol_regime, 'mtf_score': mtf_score,
            'kal_vel_pct': kal_vel_pct, 'kal_uncertainty': kal_uncertainty,
        }
    except Exception:
        return None

# ======================== REGIME-ATR HELPERS ========================
def _compute_sharp_regime_atr(atr, cp, hv20, gk_vol, day, vol_regime, adx_val, H_exp):
    base = atr / max(cp, 1e-10)
    hurst_mult = 1.0
    if H_exp > 0.65:   hurst_mult = 1.10
    elif H_exp < 0.40: hurst_mult = 0.82

    vol_ratio = gk_vol / max(hv20, 1e-10)
    vol_scale = np.clip(vol_ratio, 0.6, 1.8)

    adx_mult = 1.0 + max(0, (adx_val - 20) / 120)

    day_decay_table = {0: 1.00, 1: 0.95, 2: 0.88, 3: 0.80}
    day_decay = day_decay_table.get(day, max(0.55, 0.80 - (day-3)*0.05))

    regime_mult = {
        'trending': 1.15, 'trending_volatile': 1.28,
        'volatile': 1.05, 'squeeze': 1.30,
        'ranging': 0.72, 'quiet': 0.78, 'normal': 1.00
    }.get(vol_regime, 1.00)

    return base * hurst_mult * vol_scale * adx_mult * day_decay * regime_mult

def _estimate_daily_range_sharp(atr, cp, hv20, gk_vol, day, vol_regime, adx_val, H_exp):
    scaled = _compute_sharp_regime_atr(atr, cp, hv20, gk_vol, day, vol_regime, adx_val, H_exp)
    spread_mult = 0.75
    return scaled * spread_mult

# ======================== CALIBRATED CONFIDENCE TABLE ========================
CONFIDENCE_CAPS = {
    0: 85, 1: 78, 2: 72, 3: 66, 4: 58, 5: 52, 6: 47, 7: 44,
}
CONFIDENCE_FLOORS = {
    0: 50, 1: 38, 2: 35, 3: 32, 4: 28, 5: 26, 6: 24, 7: 22,
}

# ======================== ULTRA-SHARP PREDICTION ENGINE ========================
def predict_hlc_7d(daily_wib_df, df_weekly=None, h0_ultra=None,
                   df_1h=None, df_4h=None, df_15m=None,
                   trading_date=None, symbol=None):
    if daily_wib_df is None or len(daily_wib_df) < 30: return []
    if trading_date is None: trading_date = current_trading_date_str()
    if symbol is None: symbol = "UNKNOWN"
    try:
        ind_df = calculate_indicators(daily_wib_df)
        if ind_df is None: return []
        ind_df = calculate_ichimoku(ind_df)
        ind_df = calculate_obv(ind_df)

        last   = ind_df.iloc[-1]
        cp     = float(last['close'])
        atr    = safe_get(last.get('ATR',  np.nan), cp*0.025)
        atr7   = safe_get(last.get('ATR7', np.nan), atr)
        atr3   = safe_get(last.get('ATR3', np.nan), atr7)
        hv20   = safe_get(last.get('HV20', np.nan), 0.5)
        adx_val= safe_get(last.get('ADX',  np.nan), 20)
        gk_vol = safe_get(last.get('GK_Vol',np.nan), hv20)
        yz_vol = safe_get(last.get('YZ_Vol',np.nan), hv20)

        actual_h0 = {
            'open': float(last['open']), 'high': float(last['high']),
            'low':  float(last['low']),  'close': float(last['close']),
        }

        liq_data    = compute_liquidity_levels(ind_df)
        raw_factors = compute_factor_ensemble(ind_df, df_weekly, df_1h, liq_data)
        if not raw_factors: return []

        vol_regime, vol_mult = classify_volatility_regime(ind_df)

        try:
            H_exp = hurst_exponent(ind_df['close'].tail(100).values)
        except Exception:
            H_exp = 0.5

        adj_weights = compute_adaptive_weights(raw_factors, adx_val, hv20, vol_regime)
        fn = list(raw_factors.keys())
        fs = np.array([raw_factors[k][0] for k in fn])
        fw = np.array([adj_weights.get(k, raw_factors[k][1]) for k in fn])
        tw = max(fw.sum(), 1e-10)
        ws = float(np.sum(fs * fw) / tw)
        agree_ratio = float(np.sum(np.sign(fs) == np.sign(ws))) / max(len(fs), 1)

        mtf_score, mtf_desc = compute_mtf_confluence(df_15m, df_1h, df_4h, df_weekly)

        close_arr = ind_df['close'].tail(60).values.astype(float)
        kal_smooth_vals, velocities, kal_unc = kalman_with_uncertainty(close_arr)
        kal_vel_pct = float(velocities[-1]) / (cp + 1e-10)
        kal_uncertainty = float(kal_unc[-1]) / (cp + 1e-10)

        dom_period, phase_sig = compute_fourier_cycle(ind_df['close'].tail(100).values)

        ma50  = float(ind_df['MA50'].iloc[-1]) if 'MA50' in ind_df.columns and not pd.isna(ind_df['MA50'].iloc[-1]) else cp
        dev50 = (cp - ma50) / (ma50 + 1e-10)

        if adx_val > 28:  regime = "trending"
        elif adx_val < 15: regime = "ranging"
        else:               regime = "transitional"

        rows = []
        today_wib = wib_now()

        date_str_0 = today_wib.strftime('%d %b %Y')
        day_name_0 = _day_name_id(today_wib.weekday())

        if h0_ultra is not None:
            sim_open_h0  = h0_ultra['open'];  sim_hi_h0   = h0_ultra['high']
            sim_lo_h0    = h0_ultra['low'];   sim_close_h0 = h0_ultra['close']
            conf_h0      = h0_ultra['confidence']
            method_h0    = h0_ultra['method']
        else:
            sim_open_h0  = actual_h0['open'];  sim_hi_h0  = actual_h0['high']
            sim_lo_h0    = actual_h0['low'];   sim_close_h0= actual_h0['close']
            conf_h0      = 60.0; method_h0 = 'Fallback'

        daily_move_h0 = (sim_close_h0 - actual_h0['open']) / max(actual_h0['open'], 1e-10)

        if   daily_move_h0 >  0.025: direction_h0 = "📈 Naik Kuat"
        elif daily_move_h0 >  0.010: direction_h0 = "📈 Naik"
        elif daily_move_h0 >  0.002: direction_h0 = "📈 Naik (Lemah)"
        elif daily_move_h0 < -0.025: direction_h0 = "📉 Turun Kuat"
        elif daily_move_h0 < -0.010: direction_h0 = "📉 Turun"
        elif daily_move_h0 < -0.002: direction_h0 = "📉 Turun (Lemah)"
        else:                         direction_h0 = "⚖️ Sideways"

        pva_O = (sim_open_h0  - actual_h0['open'])  / max(actual_h0['open'],  1e-10) * 100
        pva_H = (sim_hi_h0    - actual_h0['high'])  / max(actual_h0['high'],  1e-10) * 100
        pva_L = (sim_lo_h0    - actual_h0['low'])   / max(actual_h0['low'],   1e-10) * 100
        pva_C = (sim_close_h0 - actual_h0['close']) / max(actual_h0['close'], 1e-10) * 100

        rows.append({
            'day': 0, 'date': date_str_0, 'day_name_id': day_name_0,
            'direction': direction_h0, 'confidence': round(conf_h0, 1),
            'open':  round(sim_open_h0, 10), 'high': round(sim_hi_h0, 10),
            'low':   round(sim_lo_h0, 10),   'close': round(sim_close_h0, 10),
            'change_pct': round(daily_move_h0 * 100, 2),
            'score': round(float(ws), 3), 'reason': method_h0,
            'agreement_pct': round(agree_ratio * 100, 1), 'regime': regime,
            'is_today': True,
            'actual_open': actual_h0['open'], 'actual_high': actual_h0['high'],
            'actual_low':  actual_h0['low'],  'actual_close': actual_h0['close'],
            'pred_vs_actual_O': round(pva_O, 3), 'pred_vs_actual_H': round(pva_H, 3),
            'pred_vs_actual_L': round(pva_L, 3), 'pred_vs_actual_C': round(pva_C, 3),
            'vol_regime': vol_regime, 'mtf_score': round(mtf_score, 3),
            'hurst': round(H_exp, 3), 'mc_p50': round(sim_close_h0, 10),
        })

        sim_close = sim_close_h0
        mc_vol = float(yz_vol) if not np.isnan(yz_vol) else hv20
        MAX_DAILY_MOVE_PCT = 0.18

        for day in range(1, 8):
            pred_dt  = today_wib + timedelta(days=day)
            day_name = _day_name_id(pred_dt.weekday())
            date_str = pred_dt.strftime('%d %b %Y')

            mc_seed = make_mc_seed(symbol, trading_date, day)

            if H_exp > 0.60:
                decay = 0.96 ** day
            elif H_exp < 0.40:
                decay = 0.87 ** day
            else:
                decay = 0.92 ** day

            mr_strength = abs(dev50) * max(0, (day - 1) / 3)
            mr_pull = -np.sign(ws) * min(mr_strength, 0.12) if abs(dev50) > 0.05 else 0

            fourier_contrib = phase_sig * max(0, 1 - day / max(dom_period, 1)) * 0.10

            kal_decay = max(0, 1 - day / 4)
            kal_contrib = float(np.tanh(kal_vel_pct * 60)) * kal_decay

            mtf_contrib = mtf_score * max(0, 1 - day / 4) * 0.10

            if day <= 3:
                prior_signal = ws * decay + mr_pull
                likelihood   = float(np.clip(kal_contrib + fourier_contrib, -1, 1))
                bayesian_fused = bayesian_signal_update(prior_signal, likelihood, confidence=0.50)
                eff_total = float(np.clip(
                    bayesian_fused * 0.65 + mtf_contrib * 0.20 + fourier_contrib * 0.15,
                    -1, 1
                ))
            else:
                eff = ws * decay + mr_pull
                eff_total = float(np.clip(
                    eff + kal_contrib * 0.3 + mtf_contrib + fourier_contrib * 0.5,
                    -1, 1
                ))

            mc_ranges = monte_carlo_price_range(
                sim_close, mc_vol, days=1, n_sims=1500, seed=mc_seed
            )
            mc_low  = mc_ranges['p10']
            mc_high = mc_ranges['p90']
            mc_med  = mc_ranges['p50']

            regime_atr_pct = _compute_sharp_regime_atr(
                atr, cp, hv20, gk_vol, day, vol_regime, adx_val, H_exp
            )

            daily_move = eff_total * regime_atr_pct
            daily_move = float(np.clip(daily_move, -MAX_DAILY_MOVE_PCT, MAX_DAILY_MOVE_PCT))
            sim_close  = sim_close * (1 + daily_move)
            chg_pct    = (sim_close - cp) / cp * 100

            gap_pct    = float(eff_total) * (atr3/max(cp,1e-10) if not np.isnan(atr3) else atr7/max(cp,1e-10)) * 0.15
            sim_open   = (sim_close / (1 + daily_move)) * (1 + gap_pct)

            spread = _estimate_daily_range_sharp(
                atr, cp, hv20, gk_vol, day, vol_regime, adx_val, H_exp
            )
            bias_h = 0.55 if eff_total > 0 else 0.45
            bias_l = 1.0 - bias_h
            mid    = (sim_open + sim_close) / 2

            ta_high = mid + spread * cp * bias_h * (1 + abs(eff_total) * 0.4)
            ta_low  = mid - spread * cp * bias_l * (1 + abs(eff_total) * 0.4)

            if day <= 3:
                mc_w = 0.40 if day == 1 else (0.30 if day == 2 else 0.20)
                sim_hi = ta_high * (1 - mc_w) + mc_high * mc_w
                sim_lo = ta_low  * (1 - mc_w) + mc_low  * mc_w
            else:
                sim_hi = ta_high
                sim_lo = ta_low

            sim_hi = max(sim_hi, max(sim_open, sim_close))
            sim_lo = min(sim_lo, min(sim_open, sim_close))

            base_conf = 42 + agree_ratio * 22 + abs(eff_total) * 10
            if regime == "trending" and adx_val > 25: base_conf += 4
            if agree_ratio > 0.75: base_conf += 3
            elif agree_ratio < 0.40: base_conf -= 5
            if H_exp > 0.65: base_conf += 3
            elif H_exp < 0.40: base_conf -= 3

            if day <= 3:
                kal_bonus = min(3, max(0, (1 - kal_uncertainty * 20)) * 3)
                base_conf += kal_bonus

            cap   = CONFIDENCE_CAPS.get(day, 40)
            floor = CONFIDENCE_FLOORS.get(day, 22)
            conf  = max(floor, min(cap, base_conf))

            # Tentukan arah berdasarkan perubahan harga dari spot
            if chg_pct > 3.0:
                direction = "📈 Naik Kuat"
            elif chg_pct > 1.5:
                direction = "📈 Naik"
            elif chg_pct > 0.5:
                direction = "📈 Naik (Lemah)"
            elif chg_pct < -3.0:
                direction = "📉 Turun Kuat"
            elif chg_pct < -1.5:
                direction = "📉 Turun"
            elif chg_pct < -0.5:
                direction = "📉 Turun (Lemah)"
            else:
                direction = "⚖️ Sideways"

            # Anti-bug: pastikan Sideways benar-benar sideways
            if direction == "⚖️ Sideways" and abs(chg_pct) > 1.0:
                direction = "📈 Naik (Lemah)" if chg_pct > 0 else "📉 Turun (Lemah)"

            contribs = sorted(
                [(abs(fs[i]*fw[i]), fn[i], fs[i]) for i in range(len(fn))], reverse=True)
            dom    = [f"{n}{'↑' if v>0 else '↓'}" for _,n,v in contribs[:3] if abs(v)>0.15]
            if day <= 3:
                dom.insert(0, f"Kal{'↑' if kal_vel_pct>0 else '↓'}")
            reason = ' · '.join(dom) if dom else "Mixed"

            rows.append({
                'day': day, 'date': date_str, 'day_name_id': day_name,
                'direction': direction, 'confidence': round(conf, 1),
                'open':  round(sim_open, 10), 'high': round(sim_hi, 10),
                'low':   round(sim_lo,   10), 'close': round(sim_close, 10),
                'change_pct': round(chg_pct, 2),
                'score': round(eff_total, 3), 'reason': reason,
                'agreement_pct': round(agree_ratio * 100, 1), 'regime': regime,
                'is_today': False,
                'actual_open': None, 'actual_high': None,
                'actual_low':  None, 'actual_close': None,
                'pred_vs_actual_O': None, 'pred_vs_actual_H': None,
                'pred_vs_actual_L': None, 'pred_vs_actual_C': None,
                'vol_regime': vol_regime, 'mtf_score': round(mtf_score, 3),
                'hurst': round(H_exp, 3), 'mc_p50': round(mc_med, 10),
                'mc_p10': round(mc_low, 10), 'mc_p90': round(mc_high, 10),
                'mc_seed': mc_seed,
            })

        return rows
    except Exception:
        return []

def get_prediction_summary(predictions):
    if not predictions: return {}
    pf = [p for p in predictions if p['day'] > 0]
    if not pf: return {}
    bull_days  = sum(1 for p in pf if 'Naik' in p['direction'])
    bear_days  = sum(1 for p in pf if 'Turun' in p['direction'])
    avg_conf   = np.mean([p['confidence'] for p in pf])
    max_chg    = max(p['change_pct'] for p in pf)
    min_chg    = min(p['change_pct'] for p in pf)
    final_chg  = pf[-1]['change_pct']
    sharp_3    = [p for p in pf if p['day'] <= 3]
    far_4_7    = [p for p in pf if p['day'] > 3]
    s3_bull    = sum(1 for p in sharp_3 if 'Naik' in p['direction'])
    f47_bull   = sum(1 for p in far_4_7  if 'Naik' in p['direction']) if far_4_7 else 0
    best  = max(pf, key=lambda p: p['change_pct'])
    worst = min(pf, key=lambda p: p['change_pct'])
    if bull_days >= 5:   overall = "🟢 Sangat Bullish"
    elif bull_days >= 4: overall = "🟡 Bullish Dominan"
    elif bear_days >= 5: overall = "🔴 Sangat Bearish"
    elif bear_days >= 4: overall = "🟠 Bearish Dominan"
    else:                 overall = "⚪ Mixed/Sideways"
    return {
        'overall': overall, 'bull_days': bull_days, 'bear_days': bear_days,
        'side_days': len(pf)-bull_days-bear_days,
        'avg_confidence': round(avg_conf, 1),
        'max_upside': round(max_chg, 2), 'max_downside': round(min_chg, 2),
        'final_7d_change': round(final_chg, 2),
        'sharp_3_bull': s3_bull, 'far_4_7_bull': f47_bull,
        'sharp_bias': "Bullish" if s3_bull>=2 else ("Bearish" if s3_bull<=1 else "Mixed"),
        'far_bias':   "Bullish" if f47_bull>=3 else ("Bearish" if f47_bull<=1 else "Mixed"),
        'best_day': best, 'worst_day': worst,
    }

# ======================== SUPPORT / RESISTANCE ========================
def calculate_precise_sr(df, lookback=100):
    if df is None or len(df) < 30: return [], []
    rec = df.tail(lookback).copy().reset_index(drop=True)
    cp  = rec['close'].iloc[-1]
    lvl = []
    for lb in [2,3,5]:
        for i in range(lb, len(rec)-lb):
            h = rec['high'].iloc[i]; l = rec['low'].iloc[i]
            if h == rec['high'].iloc[i-lb:i+lb+1].max():
                lvl.append(('fractal_high', h, 2))
            if l == rec['low'].iloc[i-lb:i+lb+1].min():
                lvl.append(('fractal_low', l, 2))
    for ma,w in [('MA20',1),('MA50',2),('MA200',3)]:
        if ma in rec.columns:
            v = rec[ma].iloc[-1]
            if not pd.isna(v) and v>0: lvl.append((ma, v, w))
    for col,w in [('senkou_a',2),('senkou_b',2),('kijun',2),('tenkan',1),
                   ('BB_Upper',1),('BB_Lower',1),('PSAR',1),
                   ('Supertrend',2),('VWAP',1),('HMA',1),('DEMA',1),
                   ('DC_High',2),('DC_Low',2),('PP_R1',1),('PP_S1',1),
                   ('PP_R2',1),('PP_S2',1)]:
        if col in rec.columns:
            v = rec[col].iloc[-1]
            if not pd.isna(v) and v>0: lvl.append((col, v, w))
    pm = 10**max(0, len(str(int(cp)))-1)
    for mult in [0.25,0.5,1.0,2.0]:
        step = pm*mult
        if step>0:
            rl = round(cp/step)*step
            for off in [-1,0,1]:
                l2 = rl+off*step
                if l2>0: lvl.append(('round', l2, 1))
    def cluster(levels, tol=0.012):
        if not levels: return []
        prices  = np.array([l2[1] for l2 in levels])
        weights = np.array([l2[2] for l2 in levels])
        names   = [l2[0] for l2 in levels]
        visited = np.zeros(len(prices), dtype=bool)
        clusters= []
        for i in np.argsort(prices):
            if visited[i]: continue
            visited[i]=True
            cp2=[prices[i]]; cw=[weights[i]]; cn=[names[i]]
            for j in np.argsort(prices):
                if visited[j]: continue
                if abs(prices[j]-prices[i])/max(prices[i],1e-10)<tol:
                    visited[j]=True
                    cp2.append(prices[j]); cw.append(weights[j]); cn.append(names[j])
            wa = np.array(cw)
            clusters.append({'price': round(np.average(cp2,weights=wa),8),
                             'strength': len(cp2), 'weight': int(sum(cw)),
                             'methods': list(set(cn)), 'count': int(sum(cw))})
        return sorted(clusters, key=lambda x: x['price'])
    cls  = cluster(lvl)
    sups = sorted([c for c in cls if c['price']<cp*0.998],
                  key=lambda x: (-x['weight'],abs(x['price']-cp)))[:12]
    ress = sorted([c for c in cls if c['price']>cp*1.002],
                  key=lambda x: (-x['weight'],abs(x['price']-cp)))[:12]
    return sups, ress

def sr_strength_label(count):
    if   count>=6: return "💎 Ekstrem"
    elif count>=4: return "🔶 Sangat Kuat"
    elif count>=3: return "🔷 Kuat"
    elif count>=2: return "⬜ Sedang"
    else:          return "⬜ Lemah"

# ======================== CANDLESTICK PATTERNS ========================
def detect_candlestick_patterns(df):
    if df is None or len(df)<3: return []
    pts = []
    try:
        l  = df.iloc[-1]; p = df.iloc[-2]; p2 = df.iloc[-3]
        bd = abs(l['close']-l['open']); rn = l['high']-l['low']
        if rn==0: return pts
        uw = l['high']-max(l['close'],l['open']); lw = min(l['close'],l['open'])-l['low']
        if l['close']>l['open'] and p['close']<p['open'] and l['close']>p['open'] and l['open']<p['close']:
            pts.append("Bullish Engulfing")
        if l['close']<l['open'] and p['close']>p['open'] and l['close']<p['open'] and l['open']>p['close']:
            pts.append("Bearish Engulfing")
        if lw>bd*2 and uw<bd*0.5 and l['close']>l['open']: pts.append("Hammer")
        if uw>bd*2 and lw<bd*0.5 and l['close']<l['open']: pts.append("Shooting Star")
        if (p2['close']<p2['open'] and abs(p['close']-p['open'])<(p['high']-p['low'])*0.3
                and l['close']>l['open'] and l['close']>(p2['open']+p2['close'])/2):
            pts.append("Morning Star")
        if (p2['close']>p2['open'] and abs(p['close']-p['open'])<(p['high']-p['low'])*0.3
                and l['close']<l['open'] and l['close']<(p2['open']+p2['close'])/2):
            pts.append("Evening Star")
        if bd<=rn*0.1: pts.append("Doji")
        if lw > rn*0.6 and bd < rn*0.25: pts.append("Bull Pin Bar")
        if uw > rn*0.6 and bd < rn*0.25: pts.append("Bear Pin Bar")
    except Exception:
        pass
    return pts

# ======================== STRUCTURE & DIVERGENCE ========================
def detect_market_structure(df, lookback=50):
    if df is None or len(df)<lookback: return {}, "N/A"
    rec = df.tail(lookback)
    hh2 = rec['high'].values; ll2 = rec['low'].values
    sh2,sl2 = [],[]
    for i in range(2,len(hh2)-2):
        if hh2[i]==max(hh2[i-2:i+3]): sh2.append(hh2[i])
        if ll2[i]==min(ll2[i-2:i+3]): sl2.append(ll2[i])
    st={}
    st['hh'] = len(sh2)>=2 and sh2[-1]>sh2[-2]
    st['lh'] = len(sh2)>=2 and sh2[-1]<sh2[-2]
    st['hl'] = len(sl2)>=2 and sl2[-1]>sl2[-2]
    st['ll'] = len(sl2)>=2 and sl2[-1]<sl2[-2]
    if st['hh'] and st['hl']:   desc="🟢 Uptrend (HH+HL)"
    elif st['lh'] and st['ll']: desc="🔴 Downtrend (LH+LL)"
    elif st['hh'] and st['ll']: desc="⚠️ Volatil/Choppy"
    elif st['hl'] and st['lh']: desc="⚠️ Potensi Reversal"
    else:                        desc="⚖️ Netral"
    return st, desc

def detect_divergences(df, lookback=30):
    r={k:False for k in ['rsi_bullish','rsi_bearish','macd_bullish','hidden_bullish','hidden_bearish']}
    if df is None or len(df)<lookback: return r
    try:
        half=lookback//2
        lo=df['low'].tail(lookback).values; hi=df['high'].tail(lookback).values
        rs=df['RSI'].tail(lookback).fillna(50).values
        mh2=df['MACD_Hist'].tail(lookback).fillna(0).values
        li1=np.argmin(lo[:half]); li2=half+np.argmin(lo[half:])
        hi1=np.argmax(hi[:half]); hi2=half+np.argmax(hi[half:])
        r['rsi_bullish']   = lo[li2]<lo[li1] and rs[li2]>rs[li1]
        r['rsi_bearish']   = hi[hi2]>hi[hi1] and rs[hi2]<rs[hi1]
        r['macd_bullish']  = lo[li2]<lo[li1] and mh2[li2]>mh2[li1]
        r['hidden_bullish']= lo[li2]>lo[li1] and rs[li2]<rs[li1]
        r['hidden_bearish']= hi[hi2]<hi[hi1] and rs[hi2]>rs[hi1]
    except Exception:
        pass
    return r

# ======================== WYCKOFF ========================
def detect_wyckoff_phase(df, lookback=80):
    if df is None or len(df)<lookback: return "Unknown","Data tidak cukup"
    rec=df.tail(lookback).copy(); cp=rec['close'].iloc[-1]
    pr=(rec['high'].max()-rec['low'].min())/max(rec['low'].min(),1e-10)
    v1=rec['volume'].iloc[:lookback//2].mean(); v2=rec['volume'].iloc[lookback//2:].mean()
    vd=v2<v1*0.85
    sl=rec['low'].tail(30).min(); ll2=rec['low'].tail(10).min()
    sc2=(ll2==rec['low'].min() and cp>sl*1.01)
    li=rec['low'].idxmin()
    svs=False
    if isinstance(li,(int,np.integer)):
        sv=rec['volume'].iloc[max(0,li-2):li+3].max()
        svs=sv>rec['volume'].mean()*1.5
    mk = cp > rec['MA50'].iloc[-1] if 'MA50' in rec.columns and not pd.isna(rec['MA50'].iloc[-1]) else cp > rec['close'].mean()
    if sc2 and svs: return "Spring (Beli Diam-diam)","Wyckoff Spring terdeteksi"
    elif pr<0.15 and vd: return "Accumulation",f"Sideways {pr:.1%} vol menurun"
    elif mk and pr>0.1: return "Markup","Fase markup aktif"
    elif cp<sl*0.98: return "Distribution / Downtrend","Harga di bawah support"
    else: return "Transition","Fase transisi"

# ======================== SPOT MARKET GUARD ========================
MIN_BULL_SIGNALS_FOR_TRADE = 5

def is_spot_tradeable(bull_signals, result=None):
    if bull_signals < MIN_BULL_SIGNALS_FOR_TRADE:
        return False, (
            f"⚠️ Spot TIDAK LAYAK BELI: bull_signals={bull_signals}/12 "
            f"(min {MIN_BULL_SIGNALS_FOR_TRADE} diperlukan). "
            f"Pasar bearish/netral — tunggu konfirmasi."
        )
    if result is not None:
        st_bull = result.get('supertrend_bull', True)
        adx = result.get('adx', 20)
        if not st_bull and adx > 30:
            return False, (
                f"⚠️ Spot TIDAK LAYAK BELI: Supertrend Bear + ADX={adx:.0f} kuat. "
                f"Tren turun konfirmasi — hindari posisi long."
            )
    return True, "✅ Setup spot valid"

# ======================== TRADE PLAN ========================
def calc_trade_plan(df, supports, resistances, poc, val, atr, cp):
    if cp > 10000:
        max_entry_pct_below = 0.07
    elif cp > 100:
        max_entry_pct_below = 0.08
    else:
        max_entry_pct_below = 0.10

    valid_sups = sorted(
        [s for s in supports
         if s['price'] < cp * 0.999 and s['price'] >= cp * (1 - max_entry_pct_below)],
        key=lambda x: x['price'], reverse=True
    )
    valid_ress = sorted(
        [r for r in resistances if r['price'] > cp * 1.001],
        key=lambda x: x['price']
    )

    if valid_sups:
        best_sup = max(valid_sups, key=lambda x: x['weight'])
        conservative_entry = best_sup['price']
    else:
        conservative_entry = cp - atr * 0.5

    conservative_entry = max(conservative_entry, cp * (1 - max_entry_pct_below))
    conservative_entry = min(conservative_entry, cp * 0.999)

    aggressive_entry = cp - atr * 0.08
    aggressive_entry = max(aggressive_entry, conservative_entry * 1.001)
    aggressive_entry = min(aggressive_entry, cp * 0.999)

    if aggressive_entry <= conservative_entry:
        aggressive_entry = min(conservative_entry * 1.003, cp * 0.999)

    sups_below_entry = sorted(
        [s for s in supports if s['price'] < conservative_entry * 0.998],
        key=lambda x: x['price'], reverse=True
    )
    if sups_below_entry:
        sl_base = sups_below_entry[0]['price'] * 0.995
    else:
        sl_base = conservative_entry - atr * 1.5

    sl_candidate = min(sl_base, conservative_entry * 0.985)
    sl_min_allowed = conservative_entry * 0.85
    stop_loss = max(sl_candidate, sl_min_allowed)
    stop_loss = min(stop_loss, conservative_entry * 0.994)

    risk = conservative_entry - stop_loss
    if risk <= 0:
        risk = conservative_entry * 0.02
        stop_loss = conservative_entry - risk

    if valid_ress:
        tp1_cands = [r for r in valid_ress if r['price'] > conservative_entry * 1.003]
        tp1 = tp1_cands[0]['price'] if tp1_cands else conservative_entry + risk * 1.5
        tp2 = tp1_cands[1]['price'] if len(tp1_cands) >= 2 else tp1 + risk * 1.0
        tp3 = tp1_cands[2]['price'] if len(tp1_cands) >= 3 else tp2 + risk * 1.5
    else:
        tp1 = conservative_entry + risk * 1.5
        tp2 = conservative_entry + risk * 2.5
        tp3 = conservative_entry + risk * 4.0

    tp1 = max(tp1, conservative_entry + risk * 1.0)
    tp2 = max(tp2, tp1 + risk * 0.5)
    tp3 = max(tp3, tp2 + risk * 0.5)

    def pct_from_entry(tp_price):
        return round((tp_price - conservative_entry) / max(conservative_entry, 1e-10) * 100, 2)

    def pct_from_current(tp_price):
        return round((tp_price - cp) / max(cp, 1e-10) * 100, 2)

    sl_pct  = round((stop_loss - conservative_entry) / max(conservative_entry, 1e-10) * 100, 2)
    rr_ratio = round((tp1 - conservative_entry) / max(risk, 1e-10), 2)

    return {
        'conservative_entry': round(conservative_entry, 10),
        'aggressive_entry':   round(aggressive_entry, 10),
        'stop_loss':          round(stop_loss, 10),
        'sl_pct':             sl_pct,
        'tp1':  round(tp1, 10),
        'tp2':  round(tp2, 10),
        'tp3':  round(tp3, 10),
        'tp1_pct_entry':   pct_from_entry(tp1),
        'tp2_pct_entry':   pct_from_entry(tp2),
        'tp3_pct_entry':   pct_from_entry(tp3),
        'tp1_pct_current': pct_from_current(tp1),
        'tp2_pct_current': pct_from_current(tp2),
        'tp3_pct_current': pct_from_current(tp3),
        'rr':   rr_ratio,
        'risk': round(risk, 10),
    }

def pos_size(entry, stop_loss, risk_pct=0.02, account_size=10000):
    risk_per_unit = entry - stop_loss
    if risk_per_unit <= 0:
        return 0.0
    dollar_risk   = account_size * risk_pct
    units         = dollar_risk / risk_per_unit
    position_value= units * entry
    pos_pct       = (position_value / account_size) * 100
    return round(min(pos_pct, 100.0), 1)

def validate_trade_plan(trade, cp):
    warnings_list = []
    is_valid = True

    ce  = trade['conservative_entry']
    ae  = trade['aggressive_entry']
    sl  = trade['stop_loss']
    tp1 = trade['tp1']
    tp2 = trade['tp2']
    tp3 = trade['tp3']

    if sl >= ce:
        warnings_list.append(f"❌ SL ({fmt_price(sl)}) >= Entry ({fmt_price(ce)}) — TERBALIK!")
        is_valid = False

    if ae >= cp:
        warnings_list.append(f"⚠️ Entry Agresif ({fmt_price(ae)}) >= Harga ({fmt_price(cp)})")

    if ae < ce:
        warnings_list.append(f"⚠️ Entry Agresif ({fmt_price(ae)}) < Entry Konservatif ({fmt_price(ce)})")

    if tp1 <= ce:
        warnings_list.append(f"❌ TP1 ({fmt_price(tp1)}) <= Entry ({fmt_price(ce)}) — INVALID!")
        is_valid = False

    if not (tp1 < tp2 < tp3):
        warnings_list.append(f"❌ TP tidak ascending: {fmt_price(tp1)} / {fmt_price(tp2)} / {fmt_price(tp3)}")
        is_valid = False

    rr = trade.get('rr', 0)
    if rr <= 0:
        warnings_list.append(f"❌ R/R = {rr:.2f} — tidak valid (harus > 0)")
        is_valid = False
    elif rr < 1.0:
        warnings_list.append(f"⚠️ R/R = {rr:.2f} — kurang dari 1:1 (risiko tinggi)")
    elif rr > 20:
        warnings_list.append(f"⚠️ R/R = {rr:.2f} — sangat tinggi, periksa kembali")

    if cp > 10000:
        max_entry_pct = 7.0
    elif cp > 100:
        max_entry_pct = 8.0
    else:
        max_entry_pct = 10.0

    entry_pct_below = (cp - ce) / max(cp, 1e-10) * 100
    if entry_pct_below > max_entry_pct:
        warnings_list.append(
            f"⚠️ Entry konservatif terlalu jauh: {entry_pct_below:.1f}% di bawah spot "
            f"(max {max_entry_pct:.0f}% untuk aset ini)"
        )

    return is_valid, warnings_list

# ======================== VOLUME PROFILE ========================
def calculate_volume_profile(df, bins=30):
    if df is None or len(df)<20: return None,None,None
    try:
        rec=df.tail(60)
        pmn=rec['low'].min(); pmx=rec['high'].max()
        if pmx<=pmn: return None,None,None
        pb=np.linspace(pmn,pmx,bins+1); vap=np.zeros(bins)
        los=rec['low'].values; his=rec['high'].values; vols=rec['volume'].values
        rngs=np.maximum(his-los,1e-10)
        for j in range(bins):
            bl=pb[j]; bh=pb[j+1]
            ol=np.maximum(los,bl); oh=np.minimum(his,bh)
            vld=(oh>ol); rat=np.where(vld,(oh-ol)/rngs,0)
            vap[j]=np.sum(vols*rat)
        pi=np.argmax(vap); poc=(pb[pi]+pb[pi+1])/2
        tv=vap.sum(); si=np.argsort(vap)[::-1]
        cum=0; vab=[]
        for idx in si:
            cum+=vap[idx]; vab.append(idx)
            if cum>=tv*0.70: break
        vah=(pb[max(vab)]+pb[max(vab)+1])/2
        val=(pb[min(vab)]+pb[min(vab)+1])/2
        return round(poc,10),round(vah,10),round(val,10)
    except Exception:
        return None,None,None


# ======================== BANDARMOLOGY: SMART MONEY SCORE ========================
def calculate_smart_money_score(df, liq_data, poc, val, vah, current_price):
    """
    Menghitung skor konfluensi Smart Money (0-10)
    Untuk spot market: skor tinggi = zona akumulasi bandar bagus
    """
    if df is None or len(df) < 20:
        return {'score': 0, 'level': 'TIDAK CUKUP DATA', 'reasons': [], 'is_accumulation_zone': False}
    
    cp = current_price
    score = 0
    reasons = []
    
    # 1. Order block terdekat (dari liq_data)
    order_blocks = liq_data.get('order_blocks', [])
    nearest_ob = None
    for ob in order_blocks:
        if ob['price'] < cp and (nearest_ob is None or ob['price'] > nearest_ob['price']):
            nearest_ob = ob
    if nearest_ob:
        dist_pct = (cp - nearest_ob['price']) / cp * 100
        if dist_pct < 2:
            score += 3
            reasons.append(f"🔵 Order block {fmt_price(nearest_ob['price'])} terdekat ({dist_pct:.1f}% di bawah)")
        elif dist_pct < 5:
            score += 1
            reasons.append(f"🔵 Order block dalam jangkauan ({fmt_price(nearest_ob['price'])})")
    
    # 2. POC (Point of Control) - area volume tertinggi bandar
    if poc and poc > 0:
        dist_to_poc = abs(cp - poc) / cp * 100
        if dist_to_poc < 1:
            score += 3
            reasons.append(f"🟢 Harga di POC ({fmt_price(poc)}) — area konsensus bandar")
        elif dist_to_poc < 3:
            score += 1
            reasons.append(f"🟢 Dekat POC ({fmt_price(poc)})")
    
    # 3. Value Area - discount zone (area beli bandar)
    if val and cp < val:
        discount_pct = (val - cp) / val * 100
        score += 2
        reasons.append(f"💰 Discount zone ({discount_pct:.1f}% di bawah VAL) — area akumulasi")
    elif vah and cp > vah:
        score -= 2
        reasons.append(f"⚠️ Premium zone — di atas VAH ({fmt_price(vah)}), hati-hati")
    
    # 4. Liquidity sweep terdeteksi (stop hunt bandar)
    if liq_data.get('score', 0) > 0.3:
        score += 2
        reasons.append("🎯 Liquidity sweep detected — stop hunt bandar")
    
    # 5. CVD trend (akumulasi/distribusi)
    if 'CVD' in df.columns and len(df) > 20:
        cvd_trend = df['CVD'].diff().tail(5).mean()
        if cvd_trend > 0:
            score += 1
            reasons.append("📈 CVD positif — akumulasi berlangsung")
        elif cvd_trend < 0:
            score -= 1
            reasons.append("📉 CVD negatif — distribusi berlangsung")
    
    # 6. OBV divergence (bandar akumulasi tapi harga turun)
    if 'OBV' in df.columns and 'close' in df.columns and len(df) > 20:
        obv_change = (df['OBV'].iloc[-1] - df['OBV'].iloc[-20]) / abs(df['OBV'].iloc[-20] + 1e-10)
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
        if obv_change > 0.05 and price_change < -0.02:
            score += 2
            reasons.append("🐂 Bullish OBV divergence — bandar akumulasi di harga turun")
        elif obv_change < -0.05 and price_change > 0.02:
            score -= 2
            reasons.append("🐻 Bearish OBV divergence — bandar distribusi di harga naik")
    
    # Normalisasi skor ke 0-10
    score = max(0, min(10, score))
    
    # Tentukan level
    if score >= 8:
        level = "🔥 SANGAT BAGUS"
        is_accumulation = True
    elif score >= 6:
        level = "✅ BAGUS"
        is_accumulation = True
    elif score >= 4:
        level = "⚠️ CUKUP"
        is_accumulation = False
    elif score >= 2:
        level = "❌ LEMAH"
        is_accumulation = False
    else:
        level = "💀 SANGAT LEMAH"
        is_accumulation = False
    
    return {
        'score': score,
        'level': level,
        'reasons': reasons,
        'is_accumulation_zone': is_accumulation,
        'is_distribution_zone': score <= 2
    }


# ======================== BANDARMOLOGY: FAIR VALUE GAP (FVG) ========================
def detect_fair_value_gap(df):
    """
    Deteksi Fair Value Gap (imbalance 3 candle) — area likuiditas bandar
    Bullish FVG: low[i] > high[i-2] (gap ke atas)
    Bearish FVG: high[i] < low[i-2] (gap ke bawah)
    """
    if df is None or len(df) < 5:
        return [], []
    
    bullish_fvg = []
    bearish_fvg = []
    
    for i in range(2, len(df)):
        # Bullish FVG: candle i low > high i-2
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            gap_size = (df['low'].iloc[i] - df['high'].iloc[i-2]) / df['high'].iloc[i-2] * 100
            is_strong = df['volume'].iloc[i] > df['volume'].rolling(20).mean().iloc[i] if len(df) > 20 else False
            bullish_fvg.append({
                'price': (df['low'].iloc[i] + df['high'].iloc[i-2]) / 2,
                'zone_bottom': df['high'].iloc[i-2],
                'zone_top': df['low'].iloc[i],
                'gap_pct': gap_size,
                'strength': 'strong' if is_strong else 'normal',
                'candle_idx': i
            })
        
        # Bearish FVG: candle i high < low i-2
        if df['high'].iloc[i] < df['low'].iloc[i-2]:
            gap_size = (df['low'].iloc[i-2] - df['high'].iloc[i]) / df['high'].iloc[i] * 100
            is_strong = df['volume'].iloc[i] > df['volume'].rolling(20).mean().iloc[i] if len(df) > 20 else False
            bearish_fvg.append({
                'price': (df['high'].iloc[i] + df['low'].iloc[i-2]) / 2,
                'zone_bottom': df['high'].iloc[i],
                'zone_top': df['low'].iloc[i-2],
                'gap_pct': gap_size,
                'strength': 'strong' if is_strong else 'normal',
                'candle_idx': i
            })
    
    # Return 3 terakhir untuk masing-masing
    return bullish_fvg[-3:], bearish_fvg[-3:]


def get_nearest_fvg(bullish_fvg, bearish_fvg, current_price):
    """
    Mendapatkan FVG terdekat dengan harga saat ini
    """
    nearest = None
    nearest_dist = float('inf')
    fvg_type = None
    
    for fvg in bullish_fvg:
        dist = abs(fvg['price'] - current_price) / current_price * 100
        if dist < nearest_dist:
            nearest_dist = dist
            nearest = fvg
            fvg_type = 'bullish'
    
    for fvg in bearish_fvg:
        dist = abs(fvg['price'] - current_price) / current_price * 100
        if dist < nearest_dist:
            nearest_dist = dist
            nearest = fvg
            fvg_type = 'bearish'
    
    if nearest:
        nearest['type'] = fvg_type
        nearest['distance_pct'] = nearest_dist
    
    return nearest


def enhance_trade_plan_with_smc(original_trade, smc_score, fvg_nearest, current_price, atr):
    """
    Meningkatkan trade plan dengan informasi Smart Money.
    """
    enhanced = original_trade.copy()
    
    # Tambahkan SMC info
    enhanced['smc_score'] = smc_score['score']
    enhanced['smc_level'] = smc_score['level']
    enhanced['smc_reasons'] = smc_score['reasons']
    enhanced['is_accumulation_zone'] = smc_score['is_accumulation_zone']
    
    # Tambahkan FVG info
    if fvg_nearest:
        enhanced['nearest_fvg'] = fvg_nearest
        enhanced['fvg_distance_pct'] = fvg_nearest.get('distance_pct', 999)
        enhanced['fvg_type'] = fvg_nearest.get('type', 'unknown')
    
    # Hitung suggested entry (jika SMC bagus, bisa lebih agresif)
    if smc_score['score'] >= 6 and original_trade.get('conservative_entry'):
        discount = original_trade['conservative_entry'] * 0.005
        enhanced['conservative_entry_smc'] = round(original_trade['conservative_entry'] - discount, 10)
        enhanced['smc_boost_applied'] = True
        enhanced['smc_boost_pct'] = -0.5
    else:
        enhanced['conservative_entry_smc'] = original_trade.get('conservative_entry')
        enhanced['smc_boost_applied'] = False
    
    return enhanced


# ======================== FULL ANALYSIS ========================
def analyze_coin_full(symbol, exchange_name):
    try:
        trading_date = current_trading_date_str()

        df_15m  = fetch_ohlcv_cached(symbol, exchange_name, '15m', limit=700)
        df_1h   = fetch_ohlcv_cached(symbol, exchange_name, '1h',  limit=400)
        df_4h   = fetch_ohlcv_cached(symbol, exchange_name, '4h',  limit=300)
        df_1w   = fetch_ohlcv_cached(symbol, exchange_name, '1w',  limit=100)
        df_d_raw= fetch_ohlcv_cached(symbol, exchange_name, '1d',  limit=300)

        wib_daily_df, daily_source = build_wib_daily_df(df_15m, df_d_raw)
        if wib_daily_df is None: return None

        daily_ind = calculate_indicators(wib_daily_df)
        if daily_ind is None: return None
        daily_ind = calculate_ichimoku(daily_ind)
        daily_ind = calculate_obv(daily_ind)

        if df_4h is not None:
            df_4h = calculate_indicators(df_4h)
            if df_4h is not None:
                df_4h = calculate_ichimoku(df_4h)
                df_4h = calculate_obv(df_4h)
        if df_1h is not None:
            df_1h = calculate_indicators(df_1h)
            if df_1h is not None:
                df_1h = calculate_ichimoku(df_1h)
        if df_1w is not None:
            df_1w = calculate_indicators(df_1w)
            if df_1w is not None:
                df_1w = calculate_ichimoku(df_1w)
        if df_15m is not None:
            df_15m = calculate_indicators(df_15m)

        last = daily_ind.iloc[-1]
        cp   = float(last['close'])
        current_price = cp
        if df_15m is not None and len(df_15m)>0:
            current_price = float(df_15m['close'].iloc[-1])

        rsi  = safe_get(last.get('RSI',np.nan),50)
        adx_v= safe_get(last.get('ADX',np.nan),0)
        atr  = safe_get(last.get('ATR',np.nan),cp*0.02)
        st_dir= safe_get(last.get('Supertrend_Dir',np.nan),0); st_bull=st_dir==1
        cmf_v= safe_get(last.get('CMF',np.nan),0)
        tsi_v= safe_get(last.get('TSI',np.nan),0)
        kst_v= safe_get(last.get('KST',np.nan),0)
        ps_b = int(safe_get(last.get('PSAR_Bull',np.nan),0))==1
        hma_v= safe_get(last.get('HMA',np.nan),cp)
        dema_v=safe_get(last.get('DEMA',np.nan),cp)
        obv_up=bool(daily_ind['OBV_trend'].iloc[-1]) if 'OBV_trend' in daily_ind.columns else False
        ab200 =bool(not pd.isna(last.get('MA200',np.nan)) and last['close']>last['MA200'])
        abvwap=bool(not pd.isna(last.get('VWAP',np.nan)) and last['close']>last['VWAP'])
        mb1d  =bool(not pd.isna(last.get('MACD',np.nan)) and not pd.isna(last.get('Signal',np.nan)) and last['MACD']>last['Signal'])

        bull_signals = sum([
            ab200, st_bull, ps_b, tsi_v>0, kst_v>0, mb1d, obv_up, abvwap,
            cmf_v>0, cp>hma_v, cp>dema_v, rsi>50
        ])

        tradeable, trade_reason = is_spot_tradeable(bull_signals)

        h0_ultra = predict_h0_wib_ultra(
            wib_daily_df, df_15m=df_15m, df_1h=df_1h, df_4h=df_4h,
            df_weekly=df_1w, current_price=current_price)

        vol_regime, vol_mult = classify_volatility_regime(daily_ind)
        liq_data = compute_liquidity_levels(daily_ind)
        mtf_score, mtf_desc = compute_mtf_confluence(df_15m, df_1h, df_4h, df_1w)

        try:
            H_exp = hurst_exponent(daily_ind['close'].tail(100).values)
        except Exception:
            H_exp = 0.5

        struct,struct_desc = detect_market_structure(daily_ind)
        wy_phase,wy_msg    = detect_wyckoff_phase(daily_ind)
        poc,vah,val        = calculate_volume_profile(daily_ind)
        divs1d             = detect_divergences(daily_ind)
        cpats              = detect_candlestick_patterns(daily_ind)
        sups,ress          = calculate_precise_sr(daily_ind, lookback=80)

        trade = calc_trade_plan(daily_ind, sups, ress, poc, val, atr, current_price)
        psp   = pos_size(trade['conservative_entry'], trade['stop_loss']) if tradeable else 0.0

        preds_7d  = predict_hlc_7d(
            wib_daily_df, df_weekly=df_1w, h0_ultra=h0_ultra,
            df_1h=df_1h, df_4h=df_4h, df_15m=df_15m,
            trading_date=trading_date, symbol=symbol)
        pred_sum  = get_prediction_summary(preds_7d)

        # ======================== BANDARMOLOGY: Hitung Smart Money Score & FVG ========================
        smc_score = calculate_smart_money_score(daily_ind, liq_data, poc, val, vah, current_price)
        bullish_fvg, bearish_fvg = detect_fair_value_gap(daily_ind)
        nearest_fvg = get_nearest_fvg(bullish_fvg, bearish_fvg, current_price)
        
        # Enhance trade plan dengan SMC
        enhanced_trade = enhance_trade_plan_with_smc(trade, smc_score, nearest_fvg, current_price, atr)

        if "Spring" in wy_phase:    mom="🌱 WYCKOFF SPRING"
        elif st_bull and adx_v>25:  mom="📊 ST+ADX KUAT"
        elif st_bull:               mom="📈 TREND NAIK"
        elif "Accumulation" in wy_phase: mom="⏳ ACCUMULATION"
        else:                        mom="⚖️ NETRAL"

        result = {
            'symbol': symbol, 'current_price': current_price,
            'open_wib_today':  float(last['open']),
            'high_wib_today':  float(last['high']),
            'low_wib_today':   float(last['low']),
            'close_wib_today': float(last['close']),
            'daily_source': daily_source,
            **trade, 'pos_size_pct': psp, 'atr': atr,
            'rsi': round(rsi,1), 'adx': round(adx_v,1),
            'supertrend_bull': st_bull, 'cmf': round(cmf_v,3),
            'tsi': round(tsi_v,2), 'kst': round(kst_v,2),
            'psar_bull': ps_b,
            'hma': hma_v, 'dema': dema_v,
            'macd_bull_1d': mb1d, 'obv_rising': obv_up,
            'above_ma200': ab200, 'above_vwap': abvwap,
            'hurst': round(H_exp, 3),
            'structure': struct, 'structure_desc': struct_desc,
            'wyckoff_phase': wy_phase, 'wyckoff_msg': wy_msg,
            'poc': poc, 'vah': vah, 'val': val,
            'divs_1d': divs1d, 'candle_patterns': cpats,
            'supports': sups, 'resistances': ress,
            'momentum': mom, 'bull_signals': bull_signals,
            'spot_tradeable': tradeable, 'trade_reason': trade_reason,
            'predictions_7d': preds_7d, 'pred_summary': pred_sum,
            'h0_ultra': h0_ultra,
            'vol_regime': vol_regime, 'vol_mult': vol_mult,
            'liq_data': liq_data, 'mtf_score': mtf_score, 'mtf_desc': mtf_desc,
            'daily': daily_ind, 'tf_4h': df_4h, 'tf_1h': df_1h,
            'tf_15m': df_15m, 'tf_1w': df_1w,
            'pred_locked_at': wib_now().strftime('%d %b %Y %H:%M WIB'),
            'pred_locked_trading_date': trading_date,
            'actuals_refreshed_at': wib_now().strftime('%d %b %Y %H:%M WIB'),
            # BANDARMOLOGY ADDITIONS
            'smc_score': smc_score,
            'bullish_fvg': bullish_fvg,
            'bearish_fvg': bearish_fvg,
            'nearest_fvg': nearest_fvg,
            'enhanced_trade': enhanced_trade,
        }

        save_predictions(symbol, exchange_name, trading_date, preds_7d)
        save_full_snapshot(symbol, exchange_name, trading_date, result)
        return result
    except Exception:
        return None

# ======================== LIVE ACTUALS UPDATE ========================
def update_actuals_only(cached_result, symbol, exchange_name):
    try:
        df_15m_new = fetch_ohlcv_cached(symbol, exchange_name, '15m', limit=700)
        df_1h_new  = fetch_ohlcv_cached(symbol, exchange_name, '1h',  limit=400)
        df_4h_new  = fetch_ohlcv_cached(symbol, exchange_name, '4h',  limit=300)
        df_1w_new  = fetch_ohlcv_cached(symbol, exchange_name, '1w',  limit=100)
        df_d_raw   = fetch_ohlcv_cached(symbol, exchange_name, '1d',  limit=300)

        wib_daily_df, daily_source = build_wib_daily_df(df_15m_new, df_d_raw)
        if wib_daily_df is None: return cached_result

        daily_ind = calculate_indicators(wib_daily_df)
        if daily_ind is None: return cached_result
        daily_ind = calculate_ichimoku(daily_ind)
        daily_ind = calculate_obv(daily_ind)

        if df_4h_new is not None:
            df_4h_new = calculate_indicators(df_4h_new)
            if df_4h_new is not None: df_4h_new = calculate_ichimoku(df_4h_new)
        if df_1h_new is not None:
            df_1h_new = calculate_indicators(df_1h_new)
            if df_1h_new is not None: df_1h_new = calculate_ichimoku(df_1h_new)
        if df_1w_new is not None:
            df_1w_new = calculate_indicators(df_1w_new)
            if df_1w_new is not None: df_1w_new = calculate_ichimoku(df_1w_new)
        if df_15m_new is not None:
            df_15m_new = calculate_indicators(df_15m_new)

        last = daily_ind.iloc[-1]
        cp   = float(last['close'])
        current_price = cp
        if df_15m_new is not None and len(df_15m_new)>0:
            current_price = float(df_15m_new['close'].iloc[-1])

        live_actual = {
            'open':  float(wib_daily_df.iloc[-1]['open']),
            'high':  float(wib_daily_df.iloc[-1]['high']),
            'low':   float(wib_daily_df.iloc[-1]['low']),
            'close': float(wib_daily_df.iloc[-1]['close']),
        }

        updated = copy.deepcopy(cached_result)

        for p in updated.get('predictions_7d', []):
            if p['day'] == 0:
                p['actual_open']  = live_actual['open']
                p['actual_high']  = live_actual['high']
                p['actual_low']   = live_actual['low']
                p['actual_close'] = live_actual['close']
                frozen_o = p['open'];  frozen_h = p['high']
                frozen_l = p['low'];   frozen_c = p['close']
                p['pred_vs_actual_O'] = round((frozen_o-live_actual['open'])  /max(live_actual['open'],  1e-10)*100,3)
                p['pred_vs_actual_H'] = round((frozen_h-live_actual['high'])  /max(live_actual['high'],  1e-10)*100,3)
                p['pred_vs_actual_L'] = round((frozen_l-live_actual['low'])   /max(live_actual['low'],   1e-10)*100,3)
                p['pred_vs_actual_C'] = round((frozen_c-live_actual['close']) /max(live_actual['close'], 1e-10)*100,3)
                break

        updated['current_price']   = current_price
        updated['open_wib_today']  = live_actual['open']
        updated['high_wib_today']  = live_actual['high']
        updated['low_wib_today']   = live_actual['low']
        updated['close_wib_today'] = live_actual['close']
        updated['daily_source']    = daily_source
        updated['daily']    = daily_ind
        updated['tf_4h']    = df_4h_new
        updated['tf_1h']    = df_1h_new
        updated['tf_15m']   = df_15m_new
        updated['tf_1w']    = df_1w_new

        atr  = safe_get(last.get('ATR',np.nan),cp*0.02)
        rsi  = safe_get(last.get('RSI',np.nan),50)
        adx_v= safe_get(last.get('ADX',np.nan),0)
        st_dir=safe_get(last.get('Supertrend_Dir',np.nan),0); st_bull=st_dir==1
        cmf_v= safe_get(last.get('CMF',np.nan),0)
        tsi_v= safe_get(last.get('TSI',np.nan),0)
        kst_v= safe_get(last.get('KST',np.nan),0)
        ps_b = int(safe_get(last.get('PSAR_Bull',np.nan),0))==1
        hma_v= safe_get(last.get('HMA',np.nan),cp)
        dema_v=safe_get(last.get('DEMA',np.nan),cp)
        obv_up=bool(daily_ind['OBV_trend'].iloc[-1]) if 'OBV_trend' in daily_ind.columns else False
        ab200 =bool(not pd.isna(last.get('MA200',np.nan)) and last['close']>last['MA200'])
        abvwap=bool(not pd.isna(last.get('VWAP',np.nan)) and last['close']>last['VWAP'])
        mb1d  =bool(not pd.isna(last.get('MACD',np.nan)) and not pd.isna(last.get('Signal',np.nan)) and last['MACD']>last['Signal'])

        updated['rsi']=round(rsi,1); updated['adx']=round(adx_v,1); updated['atr']=atr
        updated['supertrend_bull']=st_bull; updated['cmf']=round(cmf_v,3)
        updated['tsi']=round(tsi_v,2); updated['kst']=round(kst_v,2)
        updated['psar_bull']=ps_b
        updated['hma']=hma_v; updated['dema']=dema_v
        updated['macd_bull_1d']=mb1d; updated['obv_rising']=obv_up
        updated['above_ma200']=ab200; updated['above_vwap']=abvwap

        try:
            H_exp = hurst_exponent(daily_ind['close'].tail(100).values)
            updated['hurst'] = round(H_exp, 3)
        except Exception:
            pass

        bull_signals=sum([ab200,st_bull,ps_b,tsi_v>0,kst_v>0,mb1d,obv_up,abvwap,
                          cmf_v>0,cp>hma_v,cp>dema_v,rsi>50])
        updated['bull_signals']=bull_signals

        tradeable, trade_reason = is_spot_tradeable(bull_signals, updated)
        updated['spot_tradeable'] = tradeable
        updated['trade_reason']   = trade_reason

        vol_regime,vol_mult = classify_volatility_regime(daily_ind)
        liq_data = compute_liquidity_levels(daily_ind)
        mtf_score,mtf_desc = compute_mtf_confluence(df_15m_new, df_1h_new, df_4h_new, df_1w_new)
        struct,struct_desc = detect_market_structure(daily_ind)
        wy_phase,wy_msg    = detect_wyckoff_phase(daily_ind)
        poc,vah,val2       = calculate_volume_profile(daily_ind)
        divs1d             = detect_divergences(daily_ind)
        cpats              = detect_candlestick_patterns(daily_ind)
        sups,ress          = calculate_precise_sr(daily_ind, lookback=80)
        trade              = calc_trade_plan(daily_ind, sups, ress, poc, val2, atr, current_price)
        psp                = pos_size(trade['conservative_entry'], trade['stop_loss']) if tradeable else 0.0
        
        # Update SMC untuk refresh
        smc_score = calculate_smart_money_score(daily_ind, liq_data, poc, val2, vah, current_price)
        bullish_fvg, bearish_fvg = detect_fair_value_gap(daily_ind)
        nearest_fvg = get_nearest_fvg(bullish_fvg, bearish_fvg, current_price)
        enhanced_trade = enhance_trade_plan_with_smc(trade, smc_score, nearest_fvg, current_price, atr)

        updated.update({'vol_regime':vol_regime,'vol_mult':vol_mult,
                        'liq_data':liq_data,'mtf_score':mtf_score,'mtf_desc':mtf_desc,
                        'structure':struct,'structure_desc':struct_desc,
                        'wyckoff_phase':wy_phase,'wyckoff_msg':wy_msg,
                        'poc':poc,'vah':vah,'val':val2,
                        'divs_1d':divs1d,'candle_patterns':cpats,
                        'supports':sups,'resistances':ress,'pos_size_pct':psp,
                        'smc_score':smc_score,
                        'bullish_fvg':bullish_fvg,
                        'bearish_fvg':bearish_fvg,
                        'nearest_fvg':nearest_fvg,
                        'enhanced_trade':enhanced_trade})
        updated.update(trade)

        if "Spring" in wy_phase:     mom="🌱 WYCKOFF SPRING"
        elif st_bull and adx_v>25:   mom="📊 ST+ADX KUAT"
        elif st_bull:                mom="📈 TREND NAIK"
        elif "Accumulation" in wy_phase: mom="⏳ ACCUMULATION"
        else:                         mom="⚖️ NETRAL"
        updated['momentum']=mom

        h0_live = predict_h0_wib_ultra(
            wib_daily_df, df_15m=df_15m_new, df_1h=df_1h_new, df_4h=df_4h_new,
            df_weekly=df_1w_new, current_price=current_price)
        updated['h0_ultra'] = h0_live
        updated['actuals_refreshed_at'] = wib_now().strftime('%d %b %Y %H:%M WIB')

        save_predictions(symbol, exchange_name, updated['pred_locked_trading_date'],
                         updated.get('predictions_7d', []))
        save_full_snapshot(symbol, exchange_name, updated['pred_locked_trading_date'], updated)
        return updated
    except Exception:
        return cached_result

# ======================== RESTORE FROM SNAPSHOT ========================
def restore_from_snapshot(snapshot, symbol, exchange_name):
    try:
        df_15m_new = fetch_ohlcv_cached(symbol, exchange_name, '15m', limit=700)
        df_1h_new  = fetch_ohlcv_cached(symbol, exchange_name, '1h',  limit=400)
        df_4h_new  = fetch_ohlcv_cached(symbol, exchange_name, '4h',  limit=300)
        df_1w_new  = fetch_ohlcv_cached(symbol, exchange_name, '1w',  limit=100)
        df_d_raw   = fetch_ohlcv_cached(symbol, exchange_name, '1d',  limit=300)

        wib_daily_df, daily_source = build_wib_daily_df(df_15m_new, df_d_raw)
        if wib_daily_df is None: return None

        daily_ind = calculate_indicators(wib_daily_df)
        if daily_ind is None: return None
        daily_ind = calculate_ichimoku(daily_ind)
        daily_ind = calculate_obv(daily_ind)

        if df_4h_new is not None:
            df_4h_new = calculate_indicators(df_4h_new)
            if df_4h_new is not None: df_4h_new = calculate_ichimoku(df_4h_new)
        if df_1h_new is not None:
            df_1h_new = calculate_indicators(df_1h_new)
            if df_1h_new is not None: df_1h_new = calculate_ichimoku(df_1h_new)
        if df_1w_new is not None:
            df_1w_new = calculate_indicators(df_1w_new)
            if df_1w_new is not None: df_1w_new = calculate_ichimoku(df_1w_new)
        if df_15m_new is not None:
            df_15m_new = calculate_indicators(df_15m_new)

        last = daily_ind.iloc[-1]
        cp   = float(last['close'])
        current_price = cp
        if df_15m_new is not None and len(df_15m_new) > 0:
            current_price = float(df_15m_new['close'].iloc[-1])

        live_actual = {
            'open':  float(wib_daily_df.iloc[-1]['open']),
            'high':  float(wib_daily_df.iloc[-1]['high']),
            'low':   float(wib_daily_df.iloc[-1]['low']),
            'close': float(wib_daily_df.iloc[-1]['close']),
        }

        result = copy.deepcopy(snapshot)
        result['daily']  = daily_ind
        result['tf_4h']  = df_4h_new
        result['tf_1h']  = df_1h_new
        result['tf_15m'] = df_15m_new
        result['tf_1w']  = df_1w_new
        result['daily_source'] = daily_source

        for p in result.get('predictions_7d', []):
            if p['day'] == 0:
                p['actual_open']  = live_actual['open']
                p['actual_high']  = live_actual['high']
                p['actual_low']   = live_actual['low']
                p['actual_close'] = live_actual['close']
                frozen_o = p['open'];  frozen_h = p['high']
                frozen_l = p['low'];   frozen_c = p['close']
                p['pred_vs_actual_O'] = round((frozen_o-live_actual['open'])  /max(live_actual['open'],  1e-10)*100,3)
                p['pred_vs_actual_H'] = round((frozen_h-live_actual['high'])  /max(live_actual['high'],  1e-10)*100,3)
                p['pred_vs_actual_L'] = round((frozen_l-live_actual['low'])   /max(live_actual['low'],   1e-10)*100,3)
                p['pred_vs_actual_C'] = round((frozen_c-live_actual['close']) /max(live_actual['close'], 1e-10)*100,3)
                break

        result['current_price']   = current_price
        result['open_wib_today']  = live_actual['open']
        result['high_wib_today']  = live_actual['high']
        result['low_wib_today']   = live_actual['low']
        result['close_wib_today'] = live_actual['close']

        atr  = safe_get(last.get('ATR',np.nan),cp*0.02)
        rsi  = safe_get(last.get('RSI',np.nan),50)
        adx_v= safe_get(last.get('ADX',np.nan),0)
        st_dir=safe_get(last.get('Supertrend_Dir',np.nan),0); st_bull=st_dir==1
        cmf_v= safe_get(last.get('CMF',np.nan),0)
        tsi_v= safe_get(last.get('TSI',np.nan),0)
        kst_v= safe_get(last.get('KST',np.nan),0)
        ps_b = int(safe_get(last.get('PSAR_Bull',np.nan),0))==1
        hma_v= safe_get(last.get('HMA',np.nan),cp)
        dema_v=safe_get(last.get('DEMA',np.nan),cp)
        obv_up=bool(daily_ind['OBV_trend'].iloc[-1]) if 'OBV_trend' in daily_ind.columns else False
        ab200 =bool(not pd.isna(last.get('MA200',np.nan)) and last['close']>last['MA200'])
        abvwap=bool(not pd.isna(last.get('VWAP',np.nan)) and last['close']>last['VWAP'])
        mb1d  =bool(not pd.isna(last.get('MACD',np.nan)) and not pd.isna(last.get('Signal',np.nan)) and last['MACD']>last['Signal'])

        result['rsi']=round(rsi,1); result['adx']=round(adx_v,1); result['atr']=atr
        result['supertrend_bull']=st_bull; result['cmf']=round(cmf_v,3)
        result['tsi']=round(tsi_v,2); result['kst']=round(kst_v,2)
        result['psar_bull']=ps_b
        result['hma']=hma_v; result['dema']=dema_v
        result['macd_bull_1d']=mb1d; result['obv_rising']=obv_up
        result['above_ma200']=ab200; result['above_vwap']=abvwap

        try:
            H_exp = hurst_exponent(daily_ind['close'].tail(100).values)
            result['hurst'] = round(H_exp, 3)
        except Exception:
            pass

        bull_signals=sum([ab200,st_bull,ps_b,tsi_v>0,kst_v>0,mb1d,obv_up,abvwap,
                          cmf_v>0,cp>hma_v,cp>dema_v,rsi>50])
        result['bull_signals']=bull_signals

        tradeable, trade_reason = is_spot_tradeable(bull_signals, result)
        result['spot_tradeable'] = tradeable
        result['trade_reason']   = trade_reason

        vol_regime,vol_mult = classify_volatility_regime(daily_ind)
        liq_data = compute_liquidity_levels(daily_ind)
        mtf_score,mtf_desc = compute_mtf_confluence(df_15m_new, df_1h_new, df_4h_new, df_1w_new)
        struct,struct_desc = detect_market_structure(daily_ind)
        wy_phase,wy_msg    = detect_wyckoff_phase(daily_ind)
        poc,vah,val2       = calculate_volume_profile(daily_ind)
        divs1d             = detect_divergences(daily_ind)
        cpats              = detect_candlestick_patterns(daily_ind)
        sups,ress          = calculate_precise_sr(daily_ind, lookback=80)
        trade              = calc_trade_plan(daily_ind, sups, ress, poc, val2, atr, current_price)
        psp                = pos_size(trade['conservative_entry'], trade['stop_loss']) if tradeable else 0.0
        
        smc_score = calculate_smart_money_score(daily_ind, liq_data, poc, val2, vah, current_price)
        bullish_fvg, bearish_fvg = detect_fair_value_gap(daily_ind)
        nearest_fvg = get_nearest_fvg(bullish_fvg, bearish_fvg, current_price)
        enhanced_trade = enhance_trade_plan_with_smc(trade, smc_score, nearest_fvg, current_price, atr)

        result.update({'vol_regime':vol_regime,'vol_mult':vol_mult,
                       'liq_data':liq_data,'mtf_score':mtf_score,'mtf_desc':mtf_desc,
                       'structure':struct,'structure_desc':struct_desc,
                       'wyckoff_phase':wy_phase,'wyckoff_msg':wy_msg,
                       'poc':poc,'vah':vah,'val':val2,
                       'divs_1d':divs1d,'candle_patterns':cpats,
                       'supports':sups,'resistances':ress,'pos_size_pct':psp,
                       'smc_score':smc_score,
                       'bullish_fvg':bullish_fvg,
                       'bearish_fvg':bearish_fvg,
                       'nearest_fvg':nearest_fvg,
                       'enhanced_trade':enhanced_trade})
        result.update(trade)

        if "Spring" in wy_phase:     mom="🌱 WYCKOFF SPRING"
        elif st_bull and adx_v>25:   mom="📊 ST+ADX KUAT"
        elif st_bull:                mom="📈 TREND NAIK"
        elif "Accumulation" in wy_phase: mom="⏳ ACCUMULATION"
        else:                         mom="⚖️ NETRAL"
        result['momentum']=mom

        h0_live = predict_h0_wib_ultra(
            wib_daily_df, df_15m=df_15m_new, df_1h=df_1h_new, df_4h=df_4h_new,
            df_weekly=df_1w_new, current_price=current_price)
        result['h0_ultra'] = h0_live
        result['actuals_refreshed_at'] = wib_now().strftime('%d %b %Y %H:%M WIB')
        result['pred_summary'] = get_prediction_summary(result.get('predictions_7d', []))

        save_predictions(symbol, exchange_name, result['pred_locked_trading_date'],
                         result.get('predictions_7d', []))

        return result
    except Exception:
        return None

# ======================== CHARTS ========================
def plot_main_chart(df, symbol, result):
    if df is None or len(df)<50: return go.Figure()
    df = df.tail(120)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.50,0.15,0.18,0.17],
                        subplot_titles=[f"{symbol} (WIB Daily)","Volume","RSI","MACD"])
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'], name='Price',
                                 increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1,col=1)
    for ma,clr,dash in [('MA20','#FFA726','dot'),('MA50','#42A5F5','dot'),
                         ('MA200','#EF5350','dash'),('HMA','#E040FB','solid')]:
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'],y=df[ma],name=ma,
                                     line=dict(color=clr,width=1,dash=dash)), row=1,col=1)
    if 'senkou_a' in df.columns and 'senkou_b' in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df['timestamp'],df['timestamp'][::-1]]),
            y=pd.concat([df['senkou_a'],df['senkou_b'][::-1]]),
            fill='toself', fillcolor='rgba(0,200,0,0.08)',
            line=dict(color='rgba(0,0,0,0)'), name='Kumo'), row=1,col=1)
    for s in result.get('supports',[])[:4]:
        fig.add_hline(y=s['price'], line_dash="solid", line_color="rgba(76,175,80,0.6)", line_width=1, row=1,col=1)
    for r2 in result.get('resistances',[])[:4]:
        fig.add_hline(y=r2['price'], line_dash="dash", line_color="rgba(239,83,80,0.6)", line_width=1, row=1,col=1)
    if result.get('spot_tradeable', False):
        for lv2,clr2,lbl in [(result['conservative_entry'],'#00E676','Entry Kons'),
                              (result['stop_loss'],'#FF1744','SL'),
                              (result['tp1'],'#FFD600','TP1'),
                              (result['tp2'],'#FFA000','TP2'),
                              (result['tp3'],'#FF6F00','TP3')]:
            fig.add_hline(y=lv2, line_dash="dot", line_color=clr2,
                          annotation_text=lbl, annotation_position="right", row=1,col=1)
    clrs=['#ef5350' if r2['open']>r2['close'] else '#26a69a' for _,r2 in df.iterrows()]
    fig.add_trace(go.Bar(x=df['timestamp'],y=df['volume'],marker_color=clrs,showlegend=False), row=2,col=1)
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'],y=df['RSI'],name='RSI',
                                 line=dict(color='#FF9800',width=1.5)), row=3,col=1)
        for lv2,clr2 in [(70,'red'),(30,'green'),(50,'gray')]:
            fig.add_hline(y=lv2, line_dash="dot", line_color=clr2, row=3,col=1)
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'],y=df['MACD'],name='MACD',
                                 line=dict(color='#2196F3',width=1.5)), row=4,col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'],y=df['Signal'],name='Signal',
                                 line=dict(color='#FF5722',width=1.5)), row=4,col=1)
        hc=['#26a69a' if v>=0 else '#ef5350' for v in df['MACD_Hist'].fillna(0)]
        fig.add_trace(go.Bar(x=df['timestamp'],y=df['MACD_Hist'],marker_color=hc,showlegend=False), row=4,col=1)
    h_exp = result.get('hurst', 0.5)
    h_desc = f"H={h_exp:.3f} ({'Trending' if h_exp>0.55 else 'MeanRev' if h_exp<0.45 else 'RandWalk'})"
    spot_status = "✅ LAYAK BELI" if result.get('spot_tradeable') else "🚫 TUNGGU KONFIRMASI"
    fig.update_layout(template="plotly_dark", height=900, hovermode='x unified',
                      xaxis_rangeslider_visible=False,
                      title_text=(f"{symbol} — Signals: {result['bull_signals']}/12 · "
                                  f"{result['momentum']} · {result.get('vol_regime','?')} · "
                                  f"{h_desc} · {spot_status}"),
                      title_font_size=11, showlegend=True,
                      legend=dict(orientation="h",y=1.01,x=1,font=dict(size=9)),
                      margin=dict(l=50,r=50,t=70,b=40))
    return fig

def plot_hlc_prediction_chart(preds, cp, symbol):
    if not preds: return go.Figure()
    p0 = next((p for p in preds if p['day']==0), None)
    pf = [p for p in preds if p['day']>0]
    all_preds = ([p0] if p0 else []) + pf
    x_indices = list(range(len(all_preds)))
    x_labels  = [p['date'] for p in all_preds]
    idx_0  = 0 if p0 else None
    idx_pf = list(range(1,len(all_preds))) if p0 else list(range(len(all_preds)))

    fig = go.Figure()

    mc_indices = [i for i in idx_pf if all_preds[i]['day'] <= 3 and
                  all_preds[i].get('mc_p10') is not None]
    if mc_indices:
        mc_hi = [all_preds[i].get('mc_p90', all_preds[i]['high']) for i in mc_indices]
        mc_lo = [all_preds[i].get('mc_p10', all_preds[i]['low'])  for i in mc_indices]
        fig.add_trace(go.Scatter(
            x=mc_indices + mc_indices[::-1],
            y=mc_hi + mc_lo[::-1],
            fill='toself', fillcolor='rgba(100,200,255,0.08)',
            line=dict(color='rgba(0,0,0,0)'), name='MC Band (H+1~H+3, mu=0, dampened)',
        ))

    if idx_pf:
        fig.add_trace(go.Candlestick(
            x=idx_pf,
            open=[all_preds[i]['open']   for i in idx_pf],
            high=[all_preds[i]['high']   for i in idx_pf],
            low= [all_preds[i]['low']    for i in idx_pf],
            close=[all_preds[i]['close'] for i in idx_pf],
            name='Proyeksi H+1→H+7 🔒',
            increasing=dict(line=dict(color='#26a69a',width=1.5), fillcolor='rgba(38,166,154,0.35)'),
            decreasing=dict(line=dict(color='#ef5350',width=1.5), fillcolor='rgba(239,83,80,0.35)'),
        ))

    if p0 is not None and idx_0 is not None:
        fig.add_trace(go.Candlestick(
            x=[idx_0], open=[p0['open']], high=[p0['high']], low=[p0['low']], close=[p0['close']],
            name=f"AI H+0 ({p0['confidence']:.0f}%) 🔒",
            increasing=dict(line=dict(color='#FFD700',width=2),  fillcolor='rgba(255,215,0,0.5)'),
            decreasing=dict(line=dict(color='#FFA726',width=2),  fillcolor='rgba(255,167,38,0.5)'),
        ))
        act_open=p0.get('actual_open'); act_high=p0.get('actual_high')
        act_low=p0.get('actual_low');   act_close=p0.get('actual_close')
        if act_open is not None and act_close is not None:
            fig.add_trace(go.Candlestick(
                x=[idx_0], open=[act_open], high=[act_high], low=[act_low], close=[act_close],
                name="Aktual H+0",
                increasing=dict(line=dict(color='#FFFFFF',width=1), fillcolor='rgba(255,255,255,0.15)'),
                decreasing=dict(line=dict(color='#FFFFFF',width=1), fillcolor='rgba(255,255,255,0.15)'),
            ))

    fig.add_hline(y=cp, line_dash="dot", line_color="rgba(255,255,255,0.35)",
                  annotation_text=f"Spot {fmt_price(cp)}", annotation_font_size=9)

    sharp_end_idx = 4 if p0 else 3
    if len(all_preds) > sharp_end_idx:
        fig.add_shape(type="line", x0=sharp_end_idx-0.5, x1=sharp_end_idx-0.5, y0=0, y1=1,
                      xref="x", yref="paper",
                      line=dict(color="rgba(100,200,255,0.50)",dash="dash",width=1.5))
        fig.add_annotation(x=sharp_end_idx-0.5, y=0.98, xref="x", yref="paper",
                           text="← Sharp | Statistical →", showarrow=False,
                           font=dict(size=8, color="#64C8FF"))

    for p2 in pf:
        xi = all_preds.index(p2)
        clr2 = '#00E676' if p2['change_pct'] >= 0 else '#ef5350'
        fig.add_annotation(x=xi, y=p2['high'], text=fmt_pct(p2['change_pct']),
                           showarrow=False, yshift=12, font=dict(size=8, color=clr2),
                           bgcolor='rgba(0,0,0,0.6)', borderpad=2)

    tick_step = max(1, len(x_labels)//6)
    tick_vals = list(range(0, len(x_labels), tick_step))
    tick_text = [x_labels[i] for i in tick_vals]
    fmt2 = '.8f' if cp<0.01 else ('.5f' if cp<1 else '.4f')
    fig.update_yaxes(tickformat=fmt2, gridcolor='rgba(80,80,80,0.3)')
    fig.update_xaxes(tickmode='array', tickvals=tick_vals, ticktext=tick_text,
                     tickangle=-40, tickfont=dict(size=8), gridcolor='rgba(80,80,80,0.3)')
    fig.update_layout(
        template="plotly_dark", height=600,
        title=dict(text=(f"{symbol} — Proyeksi HLC H+0→H+7 (🔒 FROZEN) · "
                         f"🎲 MC mu=0 (Spot, Dampened) · ⚡ Sharp H+1~H+3 · 📊 Statistical H+4~H+7"),
                   font=dict(size=11)),
        xaxis_title="Hari", yaxis_title="Harga (USDT)",
        xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(orientation="h",yanchor="bottom",y=1.02,x=0,font=dict(size=8)),
        hovermode='x unified', margin=dict(l=70,r=50,t=80,b=80),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(15,17,30,1)'
    )
    return fig

def plot_intraday_progress(df_15m, h0_ultra, symbol):
    if df_15m is None or h0_ultra is None: return go.Figure()
    try:
        day_start_utc = wib_trading_day_start_utc()
        day_end_utc   = day_start_utc + timedelta(days=1)
        mask  = (df_15m['timestamp']>=day_start_utc) & (df_15m['timestamp']<day_end_utc)
        today = df_15m[mask].copy().reset_index(drop=True)
        if len(today)==0: return go.Figure()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=today['timestamp'], open=today['open'], high=today['high'],
            low=today['low'], close=today['close'], name='15m Intraday',
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
        fig.add_hline(y=h0_ultra['high'],  line_dash="dash", line_color="#FFD700",
                      annotation_text=f"Pred High {fmt_price(h0_ultra['high'])}", annotation_position="right")
        fig.add_hline(y=h0_ultra['low'],   line_dash="dash", line_color="#FFD700",
                      annotation_text=f"Pred Low {fmt_price(h0_ultra['low'])}",  annotation_position="right")
        fig.add_hline(y=h0_ultra['close'], line_dash="dot",  line_color="#00E676",
                      annotation_text=f"Pred Close {fmt_price(h0_ultra['close'])}", annotation_position="right")
        fig.add_hline(y=h0_ultra['open'],  line_dash="dot",  line_color="#42A5F5",
                      annotation_text=f"Open WIB {fmt_price(h0_ultra['open'])}",  annotation_position="left")
        fig.update_layout(template="plotly_dark", height=380,
                          title=(f"{symbol} — Intraday Progress 15m (Spot) | "
                                 f"Score: {h0_ultra.get('micro_score',0):+.3f} | "
                                 f"Kal Vel: {h0_ultra.get('kal_vel_pct',0)*100:+.4f}%"),
                          xaxis_rangeslider_visible=False,
                          margin=dict(l=50,r=120,t=60,b=40))
        return fig
    except Exception:
        return go.Figure()

# ======================== CONCLUSION ========================
def generate_conclusion(result):
    if result is None: return "Data tidak tersedia."
    sym   = result['symbol']
    price = result['current_price']
    mom   = result['momentum']
    ps    = result.get('pred_summary',{})
    today_wib = wib_now()
    h0u   = result.get('h0_ultra')
    bc    = result['bull_signals']
    preds = result.get('predictions_7d',[])
    sups  = result.get('supports',[])
    ress  = result.get('resistances',[])
    h_exp = result.get('hurst', 0.5)
    daily_src = result.get('daily_source', 'N/A')
    tradeable = result.get('spot_tradeable', False)
    trade_reason = result.get('trade_reason', '')
    smc = result.get('smc_score', {})

    if price > 10000:
        max_ep_label = "7%"
    elif price > 100:
        max_ep_label = "8%"
    else:
        max_ep_label = "10%"

    lines=[]
    lines.append(f"## 📋 Analisis Spot: {sym}")
    lines.append(f"**Tanggal (WIB):** {today_wib.strftime('%d %B %Y, %H:%M')} WIB")
    lines.append(f"**Harga Spot:** `{fmt_price(price)} USDT`")
    lines.append(f"**🔒 Prediksi Dikunci:** `{result.get('pred_locked_at','N/A')}`")
    lines.append(f"**📊 Daily Source:** `{daily_src}`")
    lines.append(f"**Vol Regime:** `{result.get('vol_regime','?')}` | **MTF Score:** `{result.get('mtf_score',0):+.3f}`")
    h_label = 'Trending' if h_exp > 0.55 else ('Mean-Reverting' if h_exp < 0.45 else 'Random Walk')
    lines.append(f"**Hurst Exponent:** `{h_exp:.3f}` ({h_label})")
    lines.append("")

    lines.append("### 🎯 Spot Market Status")
    if tradeable:
        lines.append(f"✅ **LAYAK BELI (Spot Long)** — bull_signals={bc}/12")
    else:
        lines.append(f"🚫 **TIDAK LAYAK BELI** — {trade_reason}")
        lines.append(f"> Spot market = long only. Jika bearish, **tunggu konfirmasi** atau skip.")
    lines.append(f"**Momentum:** {mom} | **Signals:** {bc}/12")
    lines.append(f"**Wyckoff:** {result.get('wyckoff_phase','N/A')}")
    lines.append(f"**Structure:** {result.get('structure_desc','N/A')}")
    lines.append("")
    
    # Bandarmology Section
    lines.append("### 🧠 Smart Money / Bandarmology")
    lines.append(f"**Smart Money Score:** {smc.get('score', 0)}/10 — {smc.get('level', 'N/A')}")
    if smc.get('is_accumulation_zone'):
        lines.append("✅ **Zona Akumulasi Bandar Terdeteksi** — peluang bagus untuk entry")
    elif smc.get('score', 0) <= 2:
        lines.append("⚠️ **Zona Distribusi** — hati-hati, bandar mungkin sedang jual")
    else:
        lines.append("⚖️ **Netral** — tidak ada sinyal bandar kuat")
    if smc.get('reasons'):
        for reason in smc['reasons'][:3]:
            lines.append(f"  - {reason}")
    
    nearest_fvg = result.get('nearest_fvg')
    if nearest_fvg:
        fvg_type = "Bullish" if nearest_fvg.get('type') == 'bullish' else "Bearish"
        lines.append(f"**FVG Terdekat:** {fvg_type} FVG di `{fmt_price(nearest_fvg['price'])}` (jarak {nearest_fvg.get('distance_pct', 0):.2f}%)")
    lines.append("")

    if tradeable:
        enhanced_trade = result.get('enhanced_trade', {})
        entry_smc = enhanced_trade.get('conservative_entry_smc', result.get('conservative_entry'))
        lines.append(f"### 💹 Trade Plan (Spot Long · Entry max {max_ep_label} dari spot)")
        lines.append(f"- **Entry Konservatif (SMC Enhanced):** `{fmt_price(entry_smc)}` "
                     f"({(entry_smc-price)/price*100:+.2f}% dari spot)")
        if enhanced_trade.get('smc_boost_applied'):
            lines.append(f"  *✨ SMC Boost: diskon 0.5% dari entry standar karena zona akumulasi*")
        lines.append(f"- **Entry Agresif:** `{fmt_price(result.get('aggressive_entry'))}` "
                     f"({(result.get('aggressive_entry',price)-price)/price*100:+.2f}% dari spot)")
        lines.append(f"- **Stop Loss:** `{fmt_price(result.get('stop_loss'))}` ({fmt_pct(result.get('sl_pct',0))} dari entry)")
        lines.append(f"- **TP1:** `{fmt_price(result.get('tp1'))}` (+{result.get('tp1_pct_entry',0):.2f}%)")
        lines.append(f"- **TP2:** `{fmt_price(result.get('tp2'))}` (+{result.get('tp2_pct_entry',0):.2f}%)")
        lines.append(f"- **TP3:** `{fmt_price(result.get('tp3'))}` (+{result.get('tp3_pct_entry',0):.2f}%)")
        lines.append(f"- **R/R:** 1:{result.get('rr',0):.2f} | **Pos Size:** {result.get('pos_size_pct',0)}% akun")
    else:
        lines.append("### ⏸️ Trade Plan")
        lines.append(f"> Setup tidak valid untuk spot long saat ini. {trade_reason}")
        lines.append(f"> Level referensi (informatif, JANGAN digunakan untuk entry):")
        lines.append(f"> - Ref Entry: `{fmt_price(result.get('conservative_entry'))}`")
        lines.append(f"> - Ref SL: `{fmt_price(result.get('stop_loss'))}`")
    lines.append("")

    if ps:
        lines.append("### 🔮 Ringkasan Prediksi H+0→H+7")
        lines.append(f"**Bias:** {ps.get('overall','N/A')}")
        lines.append(f"**Sharp H+1~H+3:** {ps.get('sharp_3_bull',0)}/3 bull → **{ps.get('sharp_bias','N/A')}**")
        lines.append(f"**Statistical H+4~H+7:** {ps.get('far_4_7_bull',0)}/4 bull → **{ps.get('far_bias','N/A')}**")
        lines.append(f"**H+7:** {fmt_pct(ps.get('final_7d_change',0))} | "
                     f"**Avg Conf:** {ps.get('avg_confidence',0):.1f}%")
        lines.append("")

    lines.append("### 🔐 Support & Resistance")
    for i,s in enumerate(sups[:3],1):
        pct=(s['price']-price)/price*100
        lines.append(f"- **S{i}:** `{fmt_price(s['price'])}` ({pct:+.2f}%) — {sr_strength_label(s['count'])}")
    for i,r2 in enumerate(ress[:3],1):
        pct=(r2['price']-price)/price*100
        lines.append(f"- **R{i}:** `{fmt_price(r2['price'])}` ({pct:+.2f}%) — {sr_strength_label(r2['count'])}")

    return "\n".join(lines)

# ======================== RENDER ========================
def section_header(title):
    st.markdown(f'<div class="section-header"><b>{title}</b></div>', unsafe_allow_html=True)

def render_result(dr):
    if dr is None:
        st.error("❌ Analisis gagal atau data tidak tersedia.")
        return

    price     = dr['current_price']
    today_wib = wib_now()
    h0u       = dr.get('h0_ultra')
    h_exp     = dr.get('hurst', 0.5)
    h_label   = 'Trending' if h_exp > 0.55 else ('Mean-Rev' if h_exp < 0.45 else 'RandWalk')
    daily_src = dr.get('daily_source', 'N/A')
    tradeable = dr.get('spot_tradeable', False)
    trade_reason = dr.get('trade_reason', '')
    smc = dr.get('smc_score', {})

    if price > 10000:
        max_ep_label = "7%"
    elif price > 100:
        max_ep_label = "8%"
    else:
        max_ep_label = "10%"

    st.markdown(f"### 🎯 {dr['symbol']} — Spot Scanner v10.9 + Bandarmology")
    st.caption(f"⏰ {today_wib.strftime('%A, %d %B %Y · %H:%M')} WIB")

    if tradeable:
        st.success(f"✅ **SPOT LAYAK BELI** — bull_signals={dr['bull_signals']}/12 (min 5) · {trade_reason}")
    else:
        st.error(f"🚫 **SPOT TIDAK LAYAK BELI** — {trade_reason}\n\n"
                 f"*Spot = long only. Trade plan ditampilkan sebagai referensi, JANGAN dieksekusi.*")

    pred_locked_at    = dr.get('pred_locked_at','N/A')
    actuals_refreshed = dr.get('actuals_refreshed_at','Belum direfresh')
    trading_date      = dr.get('pred_locked_trading_date','?')

    st.info(
        f"🔒 **Prediksi FROZEN:** `{pred_locked_at}` · Aktual: `{actuals_refreshed}`  \n"
        f"📊 **Daily Source:** `{daily_src}` · "
        f"🎲 **MC mu=0** (spot short-term, dampened band)  \n"
        f"⚡ **Sharp H+1~H+3:** Hurst+Kalman-Bayes+MC | 📊 **H+4~H+7:** GBM+Fourier · "
        f"**15 Core Indicators** · Regime weights [0.5–1.2] · Entry max {max_ep_label}"
    )

    c1,c2,c3 = st.columns(3)
    with c1:
        st.metric("💲 Harga Spot", f"{fmt_price(price)} USDT")
        st.metric("📊 Bullish Signals", f"{dr['bull_signals']}/12",
                  "✅ Layak" if tradeable else f"🚫 Min {MIN_BULL_SIGNALS_FOR_TRADE}")
        st.metric("📐 Hurst Exp", f"{h_exp:.3f}", h_label)
    with c2:
        st.metric("📈 Momentum", dr['momentum'])
        if h0u:
            st.metric("🎯 H+0 Confidence", f"{h0u['confidence']:.1f}%",
                      f"Day {h0u['day_completion']*100:.0f}% done")
            st.metric("🔬 Micro Score", f"{h0u.get('micro_score',0):+.3f}")
    with c3:
        vol_r = dr.get('vol_regime','?')
        mtf_s = dr.get('mtf_score', 0)
        st.info(f"**Wyckoff:** {dr['wyckoff_phase']}\n\n"
                f"**Structure:** {dr['structure_desc']}\n\n"
                f"**Vol Regime:** {vol_r} | **MTF:** {mtf_s:+.3f}\n\n"
                f"**Hurst:** {h_exp:.3f} ({h_label})\n\n"
                f"**Src:** `{daily_src}`")

    if h0u:
        elapsed_h  = h0u['elapsed_min']//60;   elapsed_m  = h0u['elapsed_min']%60
        remaining_h= h0u['remaining_min']//60; remaining_m= h0u['remaining_min']%60
        st.success(
            f"🎯 **H+0 Ultra** | `{h0u['method']}`  \n"
            f"Elapsed: {elapsed_h}h {elapsed_m}min | Remaining: {remaining_h}h {remaining_m}min  \n"
            f"Signal: {h0u.get('composite_signal',0):+.3f} | "
            f"Micro: {h0u.get('micro_score',0):+.3f} | "
            f"KalVel: {h0u.get('kal_vel_pct',0)*100:+.4f}% | "
            f"KalUnc: {h0u.get('kal_uncertainty',0)*100:.4f}%"
        )

    st.divider()

    # ===== BANDARMOLOGY DISPLAY =====
    if smc:
        section_header("🧠 Smart Money / Bandarmology")
        
        score = smc.get('score', 0)
        level = smc.get('level', 'N/A')
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Smart Money Score", f"{score}/10", level)
        
        with col2:
            if smc.get('is_accumulation_zone'):
                st.success("✅ Zona Akumulasi Bandar Terdeteksi — peluang bagus untuk entry")
            elif smc.get('score', 0) <= 2:
                st.error("⚠️ Zona Distribusi — hati-hati, bandar mungkin sedang jual")
            else:
                st.info("⚖️ Netral — tidak ada sinyal bandar kuat")
        
        if smc.get('reasons'):
            with st.expander("📋 Detail Smart Money Signals", expanded=False):
                for reason in smc['reasons']:
                    st.markdown(f"- {reason}")
        
        # Tampilkan FVG terdekat
        nearest_fvg = dr.get('nearest_fvg')
        if nearest_fvg:
            fvg_type = "🐂 Bullish FVG" if nearest_fvg.get('type') == 'bullish' else "🐻 Bearish FVG"
            st.info(f"**{fvg_type}** terdeteksi pada `{fmt_price(nearest_fvg['price'])}` (jarak {nearest_fvg.get('distance_pct', 0):.2f}% dari spot)")
        
        st.divider()

    section_header("🎯 H+0 Prediksi (FROZEN 🔒) vs Aktual Live")
    p0 = next((p for p in dr.get('predictions_7d',[]) if p['day']==0), None)
    if p0:
        oc1,oc2,oc3,oc4 = st.columns(4)
        def show_ohlc_col(col, label, pred_val, actual_val, diff_val):
            with col:
                st.markdown(f"**{label}**")
                st.markdown(f"Pred 🔒: `{fmt_price(pred_val)}`")
                st.markdown(f"Aktual: `{fmt_price(actual_val)}`")
                color = "🟢" if abs(diff_val)<0.5 else ("🟡" if abs(diff_val)<1.5 else "🔴")
                st.markdown(f"Δ: {color} **{fmt_pct(diff_val)}**")
        show_ohlc_col(oc1,"🟦 OPEN",  p0['open'],  p0.get('actual_open'),  p0.get('pred_vs_actual_O') or 0)
        show_ohlc_col(oc2,"🟢 HIGH",  p0['high'],  p0.get('actual_high'),  p0.get('pred_vs_actual_H') or 0)
        show_ohlc_col(oc3,"🔴 LOW",   p0['low'],   p0.get('actual_low'),   p0.get('pred_vs_actual_L') or 0)
        show_ohlc_col(oc4,"🟡 CLOSE", p0['close'], p0.get('actual_close'), p0.get('pred_vs_actual_C') or 0)

    st.divider()

    if dr.get('tf_15m') is not None and h0u is not None:
        section_header("📊 Intraday Progress (15m, Spot)")
        st.plotly_chart(plot_intraday_progress(dr['tf_15m'], h0u, dr['symbol']), use_container_width=True)
        st.divider()

    section_header("📊 Core Indicators (15 Fokus Spot)")
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("RSI", f"{dr['rsi']:.1f}", "🟢OS" if dr['rsi']<35 else ("🔴OB" if dr['rsi']>70 else "Normal"))
    m2.metric("ADX", f"{dr.get('adx',0):.1f}", "✅Kuat" if dr.get('adx',0)>25 else "Lemah")
    m3.metric("TSI", f"{dr.get('tsi',0):.1f}")
    m4.metric("KST", f"{dr.get('kst',0):.1f}")
    m5.metric("CMF", f"{dr.get('cmf',0):.3f}")
    m6.metric("Hurst", f"{h_exp:.3f}", h_label)
    m7,m8,m9,m10,m11,m12 = st.columns(6)
    m7.metric("PSAR",       "✅Bull" if dr.get('psar_bull')       else "❌Bear")
    m8.metric("Supertrend", "✅Bull" if dr.get('supertrend_bull') else "❌Bear")
    m9.metric("MACD 1D",   "✅Bull" if dr.get('macd_bull_1d')    else "❌Bear")
    m10.metric("OBV",       "✅Rising" if dr.get('obv_rising')   else "❌Falling")
    m11.metric("MA200",     "✅Above" if dr.get('above_ma200')   else "❌Below")
    m12.metric("VWAP",      "✅Above" if dr.get('above_vwap')    else "❌Below")

    if dr.get('candle_patterns'):
        st.success(f"🕯️ **Candle Patterns:** {', '.join(dr['candle_patterns'])}")
    divs = dr.get('divs_1d',{})
    div_active = [k for k,v in divs.items() if v]
    if div_active:
        st.warning(f"⚡ **Divergences:** {', '.join(div_active)}")

    st.divider()

    section_header(f"💹 Trade Plan (Spot Long Only · Entry max {max_ep_label})")

    # Gunakan enhanced trade plan jika ada
    enhanced_trade = dr.get('enhanced_trade', {})
    entry_to_use = enhanced_trade.get('conservative_entry_smc', dr.get('conservative_entry'))
    smc_boost = enhanced_trade.get('smc_boost_applied', False)

    trade_valid, trade_warnings = validate_trade_plan(dr, price)

    if not tradeable:
        st.warning(
            f"🚫 **Trade plan ditampilkan sebagai REFERENSI saja.**\n\n"
            f"{trade_reason}\n\n"
            f"Spot market hanya mengizinkan posisi long. "
            f"Jangan buka posisi saat kondisi bearish/netral."
        )
    elif not trade_valid:
        st.error("⚠️ **Trade Plan Warning:**\n" + "\n".join(trade_warnings))
    elif trade_warnings:
        st.warning("📋 **Notes:**\n" + "\n".join(trade_warnings))
    else:
        st.success(f"✅ Trade Plan valid · SL < Entry Kons < Entry Aggr < TP1 < TP2 < TP3 · Entry max {max_ep_label} dari spot")

    ce  = entry_to_use
    ae  = dr['aggressive_entry']
    sl  = dr['stop_loss']
    tp1 = dr['tp1']
    tp2 = dr['tp2']
    tp3 = dr['tp3']

    tp_row1, tp_row2, tp_row3, tp_row4 = st.columns(4)
    tp_row1.metric("🔴 Stop Loss",         fmt_price(sl),  f"{dr.get('sl_pct',0):+.2f}% dari entry")
    if smc_boost:
        tp_row2.metric("🟢 Entry Konservatif", fmt_price(ce),  f"{(ce-price)/price*100:+.2f}% dari spot", help="SMC Boost: diskon 0.5% karena zona akumulasi bandar")
    else:
        tp_row2.metric("🟢 Entry Konservatif", fmt_price(ce),  f"{(ce-price)/price*100:+.2f}% dari spot")
    tp_row3.metric("🟡 Entry Agresif",     fmt_price(ae),  f"{(ae-price)/price*100:+.2f}% dari spot")
    tp_row4.metric("⚖️ R/R",              f"1:{dr.get('rr',0):.2f}")

    tp_row5, tp_row6, tp_row7, tp_row8 = st.columns(4)
    tp_row5.metric("🎯 TP1", fmt_price(tp1), f"+{dr.get('tp1_pct_entry',0):.2f}% entry")
    tp_row6.metric("🎯 TP2", fmt_price(tp2), f"+{dr.get('tp2_pct_entry',0):.2f}% entry")
    tp_row7.metric("🎯 TP3", fmt_price(tp3), f"+{dr.get('tp3_pct_entry',0):.2f}% entry")
    tp_row8.metric("📏 Pos Size",
                   f"{dr.get('pos_size_pct',0)}%" if tradeable else "0% (skip)",
                   "risk 2% akun")

    if smc_boost:
        st.caption("✨ **SMC Boost aktif:** Entry mendapatkan diskon 0.5% karena terdeteksi zona akumulasi bandar")

    with st.expander("🗺️ Peta Level Trade Plan", expanded=False):
        levels_data = [
            {"Level": "🔴 Stop Loss",         "Harga": fmt_price(sl),    "% dari Entry": f"{dr.get('sl_pct',0):+.2f}%",       "% dari Spot": fmt_pct((sl-price)/price*100)},
            {"Level": "🟢 Entry Konservatif", "Harga": fmt_price(ce),    "% dari Entry": "0.00%",                              "% dari Spot": fmt_pct((ce-price)/price*100)},
            {"Level": "🟡 Entry Agresif",     "Harga": fmt_price(ae),    "% dari Entry": fmt_pct((ae-ce)/ce*100),              "% dari Spot": fmt_pct((ae-price)/price*100)},
            {"Level": "💲 Spot Saat Ini",     "Harga": fmt_price(price), "% dari Entry": fmt_pct((price-ce)/ce*100),           "% dari Spot": "0.00%"},
            {"Level": "🎯 TP1",               "Harga": fmt_price(tp1),   "% dari Entry": fmt_pct(dr.get('tp1_pct_entry',0)),   "% dari Spot": fmt_pct(dr.get('tp1_pct_current',0))},
            {"Level": "🎯 TP2",               "Harga": fmt_price(tp2),   "% dari Entry": fmt_pct(dr.get('tp2_pct_entry',0)),   "% dari Spot": fmt_pct(dr.get('tp2_pct_current',0))},
            {"Level": "🎯 TP3",               "Harga": fmt_price(tp3),   "% dari Entry": fmt_pct(dr.get('tp3_pct_entry',0)),   "% dari Spot": fmt_pct(dr.get('tp3_pct_current',0))},
        ]
        st.dataframe(pd.DataFrame(levels_data), use_container_width=True, hide_index=True)
        st.caption(
            f"ATR: {fmt_price(dr.get('atr',0))} | Risk/unit: {fmt_price(dr.get('risk',0))} | "
            f"Max entry distance: {max_ep_label} dari spot | "
            f"SL max: {'15%' if price<=100 else '12%'} dari entry"
        )

    st.divider()

    preds = dr.get('predictions_7d',[])
    if preds:
        section_header("🔮 Proyeksi HLC H+0→H+7 (FROZEN 🔒 · MC mu=0, Dampened · Spot)")
        pred_sum = dr.get('pred_summary',{})
        if pred_sum:
            best2=pred_sum.get('best_day',{})
            sc1,sc2,sc3,sc4,sc5 = st.columns(5)
            sc1.metric("Bias 7H",     pred_sum.get('overall','N/A')[:14])
            sc2.metric("Max Upside",  f"+{pred_sum.get('max_upside',0):.2f}%")
            sc3.metric("Max Downside",f"{pred_sum.get('max_downside',0):.2f}%")
            sc4.metric("Avg Conf",    f"{pred_sum.get('avg_confidence',0):.1f}%")
            sc5.metric("🏆 Best",     f"H+{best2.get('day','?')}", fmt_pct(best2.get('change_pct',0)))

        st.plotly_chart(plot_hlc_prediction_chart(preds, price, dr['symbol']), use_container_width=True)

        with st.expander("📋 Tabel Lengkap HLC", expanded=False):
            rows=[]
            for p2 in preds:
                act_c=fmt_price(p2.get('actual_close')) if p2.get('actual_close') is not None else "—"
                diff_c2=fmt_pct(p2.get('pred_vs_actual_C',0)) if p2.get('pred_vs_actual_C') is not None else "—"
                day_label=f"H+{p2['day']}" + (" 🎯" if p2['day']==0 else (" ⚡" if p2['day']<=3 else ""))
                rows.append({
                    'Hari': day_label, 'Tanggal': p2['date'], 'Arah': p2['direction'],
                    'Open': fmt_price(p2['open']), 'High': fmt_price(p2['high']),
                    'Low':  fmt_price(p2['low']),  'Close': fmt_price(p2['close']),
                    'MC P10': fmt_price(p2.get('mc_p10')) if p2.get('mc_p10') else "—",
                    'MC P90': fmt_price(p2.get('mc_p90')) if p2.get('mc_p90') else "—",
                    'C Aktual': act_c, 'ΔC': diff_c2,
                    'Chg %': fmt_pct(p2['change_pct']), 'Conf': f"{p2['confidence']:.0f}%",
                    'Hurst': f"{p2.get('hurst',0.5):.3f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("⚡=Sharp · 📊=Statistical · 🔒=FROZEN · MC mu=0 (dampened) · Max move ±18%/hari")

    st.divider()

    section_header("🔐 Support & Resistance")
    sr1,sr2 = st.columns(2)
    with sr1:
        st.markdown("**📗 Support**")
        sups=dr.get('supports',[])
        if sups:
            st.dataframe(pd.DataFrame([
                {'Harga': fmt_price(s['price']),
                 '% dr Spot': fmt_pct((s['price']-price)/price*100),
                 'Kekuatan': sr_strength_label(s['count']),
                 'Metode': ', '.join(s['methods'][:3])}
                for s in sups[:6]]), use_container_width=True, hide_index=True)
    with sr2:
        st.markdown("**📕 Resistance**")
        ress=dr.get('resistances',[])
        if ress:
            st.dataframe(pd.DataFrame([
                {'Harga': fmt_price(r2['price']),
                 '% dr Spot': fmt_pct((r2['price']-price)/price*100),
                 'Kekuatan': sr_strength_label(r2['count']),
                 'Metode': ', '.join(r2['methods'][:3])}
                for r2 in ress[:6]]), use_container_width=True, hide_index=True)

    st.divider()

    if dr.get('poc'):
        section_header("📊 Volume Profile")
        vp1,vp2,vp3=st.columns(3)
        vp1.metric("POC", fmt_price(dr['poc']))
        vp2.metric("VAH", fmt_price(dr['vah']) if dr['vah'] else "N/A")
        vp3.metric("VAL", fmt_price(dr['val']) if dr['val'] else "N/A")
        st.divider()

    section_header("📋 Analisis Lengkap")
    st.markdown(generate_conclusion(dr))
    st.divider()

    section_header("📈 Chart Utama (WIB Daily)")
    if dr.get('daily') is not None:
        st.plotly_chart(plot_main_chart(dr['daily'], dr['symbol'], dr), use_container_width=True)

# ======================== SESSION STATE ========================
for k,v in [('manual_result',None),('locked_symbol',None),
            ('locked_exchange',None),('locked_trading_date',None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ======================== SIDEBAR ========================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    exchange_name = st.selectbox("Exchange (Spot)", ["binance","bybit","okx","kucoin"], index=0)
    st.markdown("---")
    st.subheader("🔍 Analisis")
    manual_sym = st.text_input("Symbol (contoh: ETH/USDT)")
    manual_btn = st.button("🔍 Deep Scan (Kunci Prediksi)", use_container_width=True)
    refresh_btn= st.button("🔄 Refresh Aktual", use_container_width=True,
                            disabled=(st.session_state.manual_result is None))
    reset_btn  = st.button("🗑️ Reset", use_container_width=True,
                            disabled=(st.session_state.manual_result is None))

    if st.session_state.locked_symbol:
        st.markdown("---")
        st.markdown(f"**🔒 Aktif:** `{st.session_state.locked_symbol}`")
        if st.session_state.manual_result:
            dr_side = st.session_state.manual_result
            tradeable_side = dr_side.get('spot_tradeable', False)
            bc_side = dr_side.get('bull_signals', 0)
            h_exp_side = dr_side.get('hurst', 0.5)
            price_side = dr_side.get('current_price', 0)
            smc_side = dr_side.get('smc_score', {})
            if price_side > 10000: ep_side = "7%"
            elif price_side > 100: ep_side = "8%"
            else: ep_side = "10%"
            if tradeable_side:
                st.markdown(f"✅ **LAYAK BELI** ({bc_side}/12)")
                st.markdown(f"SL: `{fmt_price(dr_side.get('stop_loss'))}`")
                st.markdown(f"Entry: `{fmt_price(dr_side.get('conservative_entry'))}`")
                st.markdown(f"TP1: `{fmt_price(dr_side.get('tp1'))}`")
                st.markdown(f"R/R: 1:{dr_side.get('rr',0):.2f}")
                st.markdown(f"Entry max: {ep_side}")
                st.markdown(f"SMC Score: {smc_side.get('score', 0)}/10")
            else:
                st.markdown(f"🚫 **SKIP** ({bc_side}/12 signals)")
            st.markdown(f"Hurst: `{h_exp_side:.3f}`")

    st.markdown("---")
    st.subheader("📁 Saved Files")
    saved_files = list_saved_predictions()
    if saved_files:
        st.markdown(f"**{len(saved_files)} file** di `C:\\trading`")
        for f in saved_files[:8]:
            icon = "📸" if f.startswith("snapshot_") else "📄"
            st.caption(f"{icon} {f}")
    else:
        st.caption("Belum ada file.")

    st.markdown("---")
    st.subheader("✅ v10.9 + Bandarmology")
    st.markdown(f"""
**Altcoin Parameters:**
- max_daily_move = ±18%
- Entry max 7–10% dari spot
- spread_mult = 0.75
- hurst_mult = 1.10
- vol_threshold = 1.9/1.6
- mc_band_mult = 0.42
- regime cap = 0.5–1.2
- volume_boost = 1.30x

**NEW - Bandarmology:**
- Smart Money Score (0-10)
- FVG (Fair Value Gap) Detector
- OBV & CVD Divergence
- Order Block Detection
- Accumulation Zone Alert
- SMC Boost Entry (0.5% diskon)
""")
    now_wib = wib_now()
    st.markdown(f"**⏰ WIB:** {now_wib.strftime('%H:%M')} | "
                f"**📅** {wib_trading_day_date().strftime('%d %b')}")

# ======================== MAIN LOGIC ========================
if reset_btn:
    st.session_state.manual_result       = None
    st.session_state.locked_symbol       = None
    st.session_state.locked_exchange     = None
    st.session_state.locked_trading_date = None
    st.rerun()

if manual_btn and manual_sym:
    sym = manual_sym.strip().upper()
    if not sym.endswith('/USDT'): sym += '/USDT'
    current_td = current_trading_date_str()

    already_locked = (
        st.session_state.locked_symbol       == sym and
        st.session_state.locked_exchange     == exchange_name and
        st.session_state.locked_trading_date == current_td and
        st.session_state.manual_result       is not None
    )

    if already_locked:
        st.toast(f"🔒 Prediksi {sym} sudah dikunci. Refresh aktual...", icon="🔄")
        with st.spinner(f"🔄 Refreshing {sym}..."):
            updated = update_actuals_only(st.session_state.manual_result, sym, exchange_name)
        st.session_state.manual_result = updated
        st.rerun()
    else:
        snapshot = load_full_snapshot(sym, exchange_name, current_td)
        if snapshot is not None:
            st.toast(f"📸 Snapshot {sym} ditemukan. Restore...", icon="📂")
            with st.spinner(f"📸 Restoring {sym}..."):
                res = restore_from_snapshot(snapshot, sym, exchange_name)
            if res is not None:
                res['pred_locked_trading_date'] = current_td
                st.session_state.manual_result      = res
                st.session_state.locked_symbol      = sym
                st.session_state.locked_exchange    = exchange_name
                st.session_state.locked_trading_date= current_td
            st.rerun()
        else:
            with st.spinner(f"🔍 Deep scanning {sym} (Spot + Bandarmology) — 15m/1h/4h/1d/1w..."):
                res = analyze_coin_full(sym, exchange_name)
            if res is not None:
                st.session_state.manual_result      = res
                st.session_state.locked_symbol      = sym
                st.session_state.locked_exchange    = exchange_name
                st.session_state.locked_trading_date= current_td
            st.rerun()

if refresh_btn and st.session_state.manual_result is not None:
    sym  = st.session_state.locked_symbol
    exch = st.session_state.locked_exchange
    with st.spinner(f"🔄 Refreshing {sym} (prediksi frozen)..."):
        updated = update_actuals_only(st.session_state.manual_result, sym, exch)
    st.session_state.manual_result = updated
    st.rerun()

if st.session_state.manual_result is not None:
    render_result(st.session_state.manual_result)
else:
    now_wib = wib_now()
    st.divider()
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f"""
### 🚀 Holy Grail Spot v10.9 + Bandarmology
**Spot Market Only · Long Only · Smart Money Detection**

📅 **WIB:** {now_wib.strftime('%d %B %Y, %H:%M')}

Masukkan symbol di sidebar → klik
**Deep Scan**.
""")
    with c2:
        st.markdown(f"""
### ✅ 8 Altcoin Params + Bandarmology
1. max_daily_move **±18%**
2. Entry max **7–10%** dari spot
3. spread_mult **0.75**
4. hurst_mult **1.10**
5. vol_threshold **1.9 / 1.6**
6. mc_band_mult **0.42**
7. regime_cap **0.5–1.2**
8. volume_boost **1.30x**
9. **Smart Money Score (0-10)**
10. **FVG Detector**
11. **SMC Boost Entry**
""")
    with c3:
        st.markdown("""
### 📊 Core Engine + Bandarmology
- 📐 Hurst (R/S Analysis)
- 🧮 Kalman-Bayesian Fusion
- 📡 Fourier Cycle Detection
- ⚙️ Regime-Conditional ATR
- 🎯 Wyckoff + Structure
- 🔒 True Frozen Snapshot
- 🎲 Seeded MC (mu=0, dampened)
- 🔮 15 Core Spot Indicators
- 🧠 Smart Money Score (Bandarmology)
- 🎯 FVG / Order Block Detection
""")
    st.info(
        "**Spot = Long Only.** Scanner v10.9 + Bandarmology:\n"
        "- ✅ Identifikasi setup beli valid (bull_signals ≥ 5/12)\n"
        "- 🚫 Otomatis skip jika kondisi bearish\n"
        "- 📏 Entry: max 7% (>10k) / 8% (mid) / 10% (altcoin)\n"
        "- 🎲 MC dampened bands · ±18% daily move cap\n"
        "- ⚙️ Regime weights capped [0.5–1.2]\n"
        "- 🧠 **NEW:** Smart Money Score 0-10 + FVG Detection + SMC Boost Entry**"
    )
# ======================== MULTI-MARKET SCANNER (FIXED - SCAN ALL, SHOW TOP 10) ========================

def get_all_usdt_pairs_full(exchange_name):
    """Ambil SEMUA pasangan USDT dari exchange"""
    try:
        st.write(f"🔄 Menghubungi {exchange_name}...")
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            'timeout': 30000
        })
        
        markets = exchange.load_markets()
        usdt_pairs = []
        for symbol in markets:
            if symbol.endswith('/USDT') and markets[symbol].get('spot', False):
                usdt_pairs.append(symbol)
        
        st.write(f"✅ Ditemukan {len(usdt_pairs)} pasangan USDT")
        return usdt_pairs
    except Exception as e:
        st.error(f"❌ Gagal: {e}")
        return []


def quick_check_full(symbol, exchange_name):
    """Cek cepat 1 pair"""
    try:
        df = fetch_ohlcv_cached(symbol, exchange_name, '1d', limit=100)
        if df is None or len(df) < 50:
            return None
        
        df = calculate_indicators(df)
        if df is None:
            return None
        
        df = calculate_ichimoku(df)
        df = calculate_obv(df)
        
        last = df.iloc[-1]
        cp = float(last['close'])
        
        # Bull signals
        rsi = safe_get(last.get('RSI', np.nan), 50)
        adx_val = safe_get(last.get('ADX', np.nan), 20)
        st_bull = safe_get(last.get('Supertrend_Dir', np.nan), 0) == 1
        ps_b = int(safe_get(last.get('PSAR_Bull', np.nan), 0)) == 1
        tsi_v = safe_get(last.get('TSI', np.nan), 0)
        kst_v = safe_get(last.get('KST', np.nan), 0)
        mb1d = safe_get(last.get('MACD', np.nan), 0) > safe_get(last.get('Signal', np.nan), 0)
        obv_up = bool(df['OBV_trend'].iloc[-1]) if 'OBV_trend' in df.columns else False
        ab200 = last.get('close', cp) > last.get('MA200', cp) if not pd.isna(last.get('MA200', cp)) else False
        abvwap = last.get('close', cp) > last.get('VWAP', cp) if not pd.isna(last.get('VWAP', cp)) else False
        cmf_v = safe_get(last.get('CMF', np.nan), 0)
        hma_v = safe_get(last.get('HMA', np.nan), cp)
        dema_v = safe_get(last.get('DEMA', np.nan), cp)
        
        bull_signals = sum([
            ab200, st_bull, ps_b, tsi_v > 0, kst_v > 0, mb1d, obv_up, abvwap,
            cmf_v > 0, cp > hma_v, cp > dema_v, rsi > 50
        ])
        
        # Hurst
        try:
            H_exp = hurst_exponent(df['close'].tail(100).values)
        except:
            H_exp = 0.5
        
        # Volume ratio
        volume_ratio = safe_get(last.get('RVOL', np.nan), 1.0)
        
        # SMC Score sederhana
        liq_data = compute_liquidity_levels(df)
        poc, vah, val = calculate_volume_profile(df)
        
        smc_score = 0
        if liq_data.get('score', 0) > 0.3:
            smc_score += 2
        if poc and abs(cp - poc) / cp < 0.02:
            smc_score += 2
        if val and cp < val:
            smc_score += 2
        if volume_ratio > 1.5:
            smc_score += 2
        smc_score = min(smc_score, 10)
        
        return {
            'symbol': symbol,
            'price': round(cp, 8),
            'bull_signals': bull_signals,
            'smc_score': smc_score,
            'hurst': round(H_exp, 3),
            'adx': round(adx_val, 1),
            'volume_ratio': round(volume_ratio, 2),
            'supertrend': 'BULL' if st_bull else 'BEAR',
            'rsi': round(rsi, 1),
        }
    except Exception as e:
        return None


def run_full_scanner():
    """Jalankan scanner untuk SEMUA pair, tampilkan 10 terbaik"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Multi-Market Scanner")
    
    scan_exchange = st.sidebar.selectbox("Exchange", ["binance", "bybit", "okx", "kucoin"], index=0)
    min_smc = st.sidebar.slider("Min SMC Score", 0, 10, 5, 1, key="min_smc_scan")
    min_bull = st.sidebar.slider("Min Bull Signals", 0, 12, 5, 1, key="min_bull_scan")
    scan_btn = st.sidebar.button("🚀 Mulai Scan SEMUA Pair", use_container_width=True)
    
    if scan_btn:
        # Ambil SEMUA pair
        with st.spinner("Mengambil daftar semua pasangan..."):
            all_pairs = get_all_usdt_pairs_full(scan_exchange)
        
        if not all_pairs:
            st.error("❌ Gagal mengambil daftar pasangan")
            return
        
        total_pairs = len(all_pairs)
        st.warning(f"⚠️ Akan scan SEMUA {total_pairs} pasangan. Ini akan memakan waktu 15-30 menit!")
        st.info("💡 Tips: Bisa ditinggal, nanti notifikasi muncul setelah selesai.")
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(all_pairs):
            status_text.text(f"🔍 [{i+1}/{total_pairs}] {symbol}")
            result = quick_check_full(symbol, scan_exchange)
            if result:
                results.append(result)
            progress_bar.progress((i + 1) / total_pairs)
            time.sleep(0.2)  # Hindari rate limit
        
        status_text.text("✅ Scan SEMUA pair selesai!")
        progress_bar.empty()
        
        # Urutkan berdasarkan SMC Score tertinggi
        results.sort(key=lambda x: (-x['smc_score'], -x['bull_signals'], -x['hurst']))
        
        # Filter berdasarkan kriteria user
        filtered = [r for r in results if r['smc_score'] >= min_smc and r['bull_signals'] >= min_bull]
        
        st.markdown("---")
        st.markdown(f"## 📊 Hasil Scan {scan_exchange.upper()}")
        st.caption(f"Total dipindai: **{len(results)}** pair | Memenuhi kriteria (SMC≥{min_smc}, Bull≥{min_bull}): **{len(filtered)}** pair")
        
        # Tampilkan TOP 10 TERBAIK (berdasarkan SMC score)
        st.markdown("### 🏆 TOP 10 ASET TERBAIK (SMC Tertinggi)")
        
        if results:
            top10 = results[:10]
            df_top = pd.DataFrame(top10)
            st.dataframe(df_top, use_container_width=True, hide_index=True)
            
            # Tampilkan juga yang memenuhi filter jika ada
            if filtered:
                st.markdown(f"### ✅ Aset yang Memenuhi Filter (SMC≥{min_smc}, Bull≥{min_bull})")
                df_filtered = pd.DataFrame(filtered)
                st.dataframe(df_filtered, use_container_width=True, hide_index=True)
            else:
                st.warning(f"❌ Tidak ada aset yang memenuhi SMC≥{min_smc} dan Bull≥{min_bull}. Coba turunkan filter.")
            
            # Pilih aset untuk analisis detail
            st.markdown("### 📊 Analisis Detail")
            selected = st.selectbox("Pilih aset untuk analisis lengkap:", [""] + [r['symbol'] for r in top10])
            if selected:
                st.session_state['manual_sym'] = selected.replace('/USDT', '')
                st.rerun()
        else:
            st.error("❌ Tidak ada hasil dari scan")


# Panggil scanner
run_full_scanner()
st.divider()
st.caption(
    "⚠️ Disclaimer: Hanya untuk edukasi. Bukan saran investasi. "
    "Selalu gunakan stop loss. v10.9 + Bandarmology: Altcoin-Tuned · 8 Params · MC mu=0 · "
    "regime cap 1.2 · entry max 10% · ±18% daily clamp · Smart Money Score · FVG Detector"
)
