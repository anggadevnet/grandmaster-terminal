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
import time
import threading
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import queue
import joblib
from scipy import stats
from scipy.signal import find_peaks
import math
import platform
import gc

# ===== LIGHTGBM =====
from lightgbm import LGBMClassifier

# ===== TENSORFLOW =====
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ===== SCIKIT-LEARN =====
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# ======================== CONSTANTS ========================
if platform.system() == 'Windows':
    SAVE_DIR = r"C:\trading"
else:
    SAVE_DIR = os.path.expanduser("~/trading")
AI_MODEL_DIR = os.path.join(SAVE_DIR, "ai_model")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(AI_MODEL_DIR, exist_ok=True)
WIB_TZ = timezone(timedelta(hours=7))

MAX_DAILY_CHANGE_LIMIT = {
    'BTC': 0.12,
    'ETH': 0.15,
    'altcoin': 0.30,
    'small_cap': 0.80,
}

MAX_POSITION_SIZE_PCT = 0.25
MAX_RR = 5.0
MAX_7D_CHANGE = 0.50

# ======================== PAGE CONFIG ========================
st.set_page_config(
    page_title="🏆 ACG SCANNER",
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
    .pump-high { background: linear-gradient(90deg, #1a1f36, #4a1f36); padding: 12px; border-radius: 8px; border-left: 4px solid #ff4444; }
    .pump-mid { background: linear-gradient(90deg, #1a1f36, #3a3f26); padding: 12px; border-radius: 8px; border-left: 4px solid #ffaa00; }
    .pump-low { background: linear-gradient(90deg, #1a1f36, #1a2f36); padding: 12px; border-radius: 8px; border-left: 4px solid #4a90d9; }
    .death-cat-high { background: linear-gradient(90deg, #2a1a1a, #4a1a1a); padding: 12px; border-radius: 8px; border-left: 4px solid #ff1744; }
    .death-cat-mid { background: linear-gradient(90deg, #2a1a2a, #4a1a3a); padding: 12px; border-radius: 8px; border-left: 4px solid #ff6d00; }
    .bandar-signal { background: linear-gradient(90deg, #1a2a1a, #1a3a2a); padding: 12px; border-radius: 8px; border-left: 4px solid #00e676; }
    .harmonic-bull { background: linear-gradient(90deg, #1a2a1a, #1a3a2a); padding: 8px 12px; border-radius: 8px; border-left: 4px solid #00e676; }
    .harmonic-bear { background: linear-gradient(90deg, #2a1a1a, #3a1a1a); padding: 8px 12px; border-radius: 8px; border-left: 4px solid #ff1744; }
    .tech-label { background: #1e2130; padding: 4px 10px; border-radius: 4px; font-size: 0.85em; display: inline-block; margin: 2px; }
</style>
""", unsafe_allow_html=True)

st.title("🏆 Holy Grail Spot Scanner v16.2 - ALL TECHNICALS")
st.caption("🔥 56+ TEKNIKAL · BANDARMOLOGI · DEATH CAT · SMC · AI · PREDIKSI 7H")

# ======================== SESSION STATE ========================
if 'manual_result' not in st.session_state:
    st.session_state.manual_result = None
if 'locked_symbol' not in st.session_state:
    st.session_state.locked_symbol = None
if 'locked_exchange' not in st.session_state:
    st.session_state.locked_exchange = None
if 'locked_trading_date' not in st.session_state:
    st.session_state.locked_trading_date = None
if 'ai_model_trained' not in st.session_state:
    st.session_state.ai_model_trained = False
if 'cached_results' not in st.session_state:
    st.session_state.cached_results = {}
if 'scanning_mode' not in st.session_state:
    st.session_state.scanning_mode = False
if 'death_cat_results' not in st.session_state:
    st.session_state.death_cat_results = None
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []

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

def _day_name_id(weekday):
    return ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"][weekday]

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

# ======================== PARAMETER DETECTION ========================
def get_asset_params(symbol, price=None):
    symbol_upper = symbol.upper() if symbol else ""
    is_btc = 'BTC' in symbol_upper
    is_eth = 'ETH' in symbol_upper
    
    is_small_cap = False
    if price is not None and price < 0.01:
        is_small_cap = True
    elif price is not None and price < 0.1:
        is_small_cap = True
    
    if is_btc:
        return {
            'max_daily_move': 0.12,
            'mc_band_mult': 0.70,
            'min_bull_signals': 3,
            'entry_max_pct': 0.05,
            'regime_min': 0.70,
            'regime_max': 1.10,
            'smc_threshold': 4,
            'name': 'BTC Mode',
            'asset_type': 'BTC',
            'is_small_cap': False
        }
    elif is_eth:
        return {
            'max_daily_move': 0.15,
            'mc_band_mult': 0.55,
            'min_bull_signals': 4,
            'entry_max_pct': 0.07,
            'regime_min': 0.60,
            'regime_max': 1.15,
            'smc_threshold': 5,
            'name': 'ETH Mode',
            'asset_type': 'ETH',
            'is_small_cap': False
        }
    elif is_small_cap:
        return {
            'max_daily_move': 0.80,
            'mc_band_mult': 0.30,
            'min_bull_signals': 6,
            'entry_max_pct': 0.15,
            'regime_min': 0.30,
            'regime_max': 1.40,
            'smc_threshold': 7,
            'name': 'Small Cap Mode',
            'asset_type': 'small_cap',
            'is_small_cap': True
        }
    else:
        return {
            'max_daily_move': 0.30,
            'mc_band_mult': 0.42,
            'min_bull_signals': 5,
            'entry_max_pct': 0.10,
            'regime_min': 0.50,
            'regime_max': 1.20,
            'smc_threshold': 6,
            'name': 'Altcoin Mode',
            'asset_type': 'altcoin',
            'is_small_cap': False
        }

# ======================== SNAPSHOT ========================
def snapshot_filename(symbol, exchange, trading_date):
    safe_sym = symbol.replace('/', '_').replace(':', '_')
    return os.path.join(SAVE_DIR, f"snapshot_{safe_sym}_{exchange}_{trading_date}.json")

def pred_filename(symbol, exchange, trading_date):
    safe_sym = symbol.replace('/', '_').replace(':', '_')
    return os.path.join(SAVE_DIR, f"pred_{safe_sym}_{exchange}_{trading_date}.json")

def _make_serializable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    if isinstance(obj, pd.Series):
        return obj.tolist()
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
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
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
            'tsi', 'kst', 'psar_bull',
            'hma', 'dema',
            'macd_bull_1d', 'obv_rising', 'above_ma200', 'above_vwap',
            'hurst', 'structure_desc', 'wyckoff_phase', 'wyckoff_msg',
            'poc', 'vah', 'val', 'divs_1d', 'candle_patterns',
            'momentum', 'bull_signals', 'vol_regime', 'vol_mult',
            'mtf_score', 'mtf_desc', 'current_price',
            'open_wib_today', 'high_wib_today', 'low_wib_today', 'close_wib_today',
            'spot_tradeable', 'smc_score', 'nearest_fvg',
            'pump_analysis', 'advanced_data', 'ai_prediction',
            'fibonacci_levels', 'chart_patterns', 'elliot_wave',
            'correlation_matrix', 'ensemble_weights', 'confidence_score',
            'bos_choch', 'fvg_status', 'ob_validation', 'liquidity_sweep',
            'institutional_candle', 'absorption_detected',
            'divergence_strength', 'failed_divergence',
            'volume_profile_data', 'wyckoff_events', 'cross_validation',
            'death_cat_bounce', 'bandar_signals', 'vwap_bands', 'adl_analysis',
            # NEW TECHNICALS
            'pivot_points', 'harmonic_patterns', 'atr_trailing_stop', 'heikin_ashi',
            'stoch_rsi', 'money_flow_index', 'aroon', 'zig_zag', 'cci',
            'ultimate_oscillator', 'vortex', 'mass_index', 'rvi', 'force_index'
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

# ======================== DATA FETCH ========================
@st.cache_data(ttl=120, hash_funcs={pd.DataFrame: lambda df: hash(df.to_json()) if df is not None else "None"})
def fetch_ohlcv_cached(symbol, exchange_name, timeframe='1d', limit=400):
    try:
        exchange_class = getattr(ccxt, exchange_name)
        
        # 🔥 KONFIGURASI DASAR
        config = {
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            'timeout': 60000,
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
        }
        
        # 🔥 KHUSUS BYBIT - PAKAI PUBLIC ENDPOINT
        if exchange_name == 'bybit':
            config['urls'] = {
                'api': {
                    'public': 'https://api.bybit.com',
                    'private': 'https://api.bybit.com'
                }
            }
            config['options']['defaultType'] = 'spot'
            # 🔥 BYBIT BUTUH INI
            config['options']['adjustForTimeDifference'] = True
            
        # 🔥 KHUSUS BINANCE
        elif exchange_name == 'binance':
            config['urls'] = {
                'api': {
                    'public': 'https://api.binance.com',
                    'private': 'https://api.binance.com'
                }
            }
            
        # 🔥 KHUSUS OKX
        elif exchange_name == 'okx':
            config['urls'] = {
                'api': {
                    'public': 'https://www.okx.com',
                    'private': 'https://www.okx.com'
                }
            }
        
        exchange = exchange_class(config)
        
        # 🔥 RETRY 3 KALI UNTUK BYBIT
        max_retries = 3 if exchange_name == 'bybit' else 1
        ohlcv = None
        
        for attempt in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if ohlcv and len(ohlcv) > 5:
                    break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)  # Tunggu 2 detik sebelum retry
        
        if not ohlcv or len(ohlcv) < 5:
            return None
            
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.dropna().reset_index(drop=True)
        for col in ['open','high','low','close','volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
        
    except Exception as e:
        # 🔥 KALAU BYBIT GAGAL, COBA PAKAI BINANCE/OKX
        if exchange_name == 'bybit':
            try:
                st.warning(f"⚠️ Bybit error, mencoba Binance...")
                return fetch_ohlcv_cached(symbol, 'binance', timeframe, limit)
            except:
                try:
                    st.warning(f"⚠️ Binance error, mencoba OKX...")
                    return fetch_ohlcv_cached(symbol, 'okx', timeframe, limit)
                except:
                    return None
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
        if np.std(vals) < 1e-10:
            return 0.0
        return np.polyfit(range(n), vals, 1)[0]
    except Exception:
        return 0.0

def find_swing_points(high, low, left_bars=3, right_bars=3, min_deviation=0.005):
    try:
        n = len(high)
        swing_highs = []
        swing_lows = []
        
        for i in range(left_bars, n - right_bars):
            if high[i] == max(high[i-left_bars:i+right_bars+1]):
                left_avg = np.mean(high[i-left_bars:i])
                right_avg = np.mean(high[i+1:i+right_bars+1])
                if left_avg > 0 and right_avg > 0:
                    dev_left = abs(high[i] - left_avg) / left_avg
                    dev_right = abs(high[i] - right_avg) / right_avg
                    if dev_left > min_deviation and dev_right > min_deviation:
                        swing_highs.append((i, high[i]))
            
            if low[i] == min(low[i-left_bars:i+right_bars+1]):
                left_avg = np.mean(low[i-left_bars:i])
                right_avg = np.mean(low[i+1:i+right_bars+1])
                if left_avg > 0 and right_avg > 0:
                    dev_left = abs(low[i] - left_avg) / left_avg
                    dev_right = abs(low[i] - right_avg) / right_avg
                    if dev_left > min_deviation and dev_right > min_deviation:
                        swing_lows.append((i, low[i]))
        
        return swing_highs, swing_lows
    except Exception:
        return [], []

# ======================== WEIGHTED ENSEMBLE ========================
def compute_ensemble_weights(factors, correlation_matrix=None):
    if not factors:
        return {}
    base_weights = {k: v[1] for k, v in factors.items()}
    if correlation_matrix is None or len(correlation_matrix) < 2:
        return base_weights
    redundancy = {}
    for k in factors.keys():
        if k in correlation_matrix:
            corr_sum = 0
            count = 0
            for other, corr in correlation_matrix[k].items():
                if other != k and other in factors:
                    corr_sum += abs(corr)
                    count += 1
            if count > 0:
                redundancy[k] = corr_sum / count
            else:
                redundancy[k] = 0
    adjusted_weights = {}
    for k in factors.keys():
        if k in redundancy:
            uniqueness = 1 - min(1, redundancy[k])
            adjusted = base_weights[k] * (0.5 + uniqueness)
            adjusted_weights[k] = max(0.3, min(2.0, adjusted))
        else:
            adjusted_weights[k] = base_weights[k]
    return adjusted_weights

def compute_correlation_matrix(factors):
    if not factors or len(factors) < 2:
        return {}
    names = list(factors.keys())
    scores = np.array([factors[k][0] for k in names])
    matrix = {}
    for i, name1 in enumerate(names):
        matrix[name1] = {}
        for j, name2 in enumerate(names):
            if i == j:
                matrix[name1][name2] = 1.0
            else:
                diff = abs(scores[i] - scores[j])
                corr = max(0, 1 - diff * 0.5)
                matrix[name1][name2] = min(0.99, corr)
    return matrix

# ======================== CONFIDENCE SCORE ========================
def compute_confidence_score(result):
    confidence = 50
    agreement = result.get('agreement_pct', 50)
    confidence += (agreement - 50) * 0.3
    smc = result.get('smc_score', {}).get('score', 0)
    confidence += smc * 1.5
    pump = result.get('pump_analysis', {}).get('score', 0)
    confidence += pump * 1.0
    ai = result.get('ai_prediction', {}).get('confidence', 50)
    confidence += (ai - 50) * 0.2
    hurst = result.get('hurst', 0.5)
    if hurst > 0.65:
        confidence += 5
    elif hurst < 0.40:
        confidence -= 5
    regime = result.get('vol_regime', 'normal')
    if regime in ['trending', 'trending_volatile']:
        confidence += 3
    elif regime in ['squeeze']:
        confidence += 5
    elif regime in ['ranging', 'quiet']:
        confidence -= 5
    elif regime in ['volatile']:
        confidence -= 3
    
    cv = result.get('cross_validation', {})
    bullish_pct = cv.get('bullish_pct', 50)
    bearish_pct = cv.get('bearish_pct', 50)
    
    if abs(bullish_pct - bearish_pct) < 15:
        confidence = min(confidence, 55)
    
    return max(10, min(95, confidence))

# ======================== AI PREDICTOR ========================
class AIPredictor:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.lstm_model = None
        self.rf_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.sequence_length = 30
        self.performance_score = 0
        self.model_dir = AI_MODEL_DIR
        self.last_training_time = None
        self._training_lock = threading.Lock()
        self._initialized = True
        self._model_cache = {}
    
    def get_model_paths(self, symbol):
        safe_sym = symbol.replace('/', '_').replace(':', '_')
        return {
            'rf': os.path.join(self.model_dir, f"rf_{safe_sym}.pkl"),
            'lstm': os.path.join(self.model_dir, f"lstm_{safe_sym}.keras"),
            'lgb': os.path.join(self.model_dir, f"lgb_{safe_sym}.pkl"),
            'scaler': os.path.join(self.model_dir, f"scaler_{safe_sym}.pkl"),
            'info': os.path.join(self.model_dir, f"info_{safe_sym}.json")
        }
    
    def save_models(self, symbol):
        try:
            paths = self.get_model_paths(symbol)
            if self.rf_model:
                joblib.dump(self.rf_model, paths['rf'])
            if self.lgb_model:
                joblib.dump(self.lgb_model, paths['lgb'])
            if self.lstm_model:
                self.lstm_model.save(paths['lstm'])
            joblib.dump(self.scaler, paths['scaler'])
            info = {
                'trained_at': datetime.now().isoformat(),
                'symbol': symbol,
                'performance_score': self.performance_score,
                'sequence_length': self.sequence_length
            }
            with open(paths['info'], 'w') as f:
                json.dump(info, f)
            return True
        except Exception:
            return False
    
    def load_models(self, symbol):
        try:
            if symbol in self._model_cache:
                return True
            
            paths = self.get_model_paths(symbol)
            if not all([os.path.exists(p) for p in [paths['rf'], paths['lstm'], paths['lgb'], paths['scaler']]]):
                return False
            
            self.rf_model = joblib.load(paths['rf'])
            self.lgb_model = joblib.load(paths['lgb'])
            self.lstm_model = tf.keras.models.load_model(paths['lstm'])
            self.scaler = joblib.load(paths['scaler'])
            
            if os.path.exists(paths['info']):
                with open(paths['info'], 'r') as f:
                    info = json.load(f)
                    self.performance_score = info.get('performance_score', 0)
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            self._model_cache[symbol] = True
            return True
        except Exception:
            return False
    
    def prepare_features(self, df):
        features = []
        features.append(df['close'].pct_change().values)
        features.append(df['close'].diff().values)
        
        if 'RSI' in df.columns:
            features.append(df['RSI'].values)
        if 'MACD' in df.columns:
            features.append(df['MACD'].values)
        if 'MACD_Hist' in df.columns:
            features.append(df['MACD_Hist'].values)
        if 'ADX' in df.columns:
            features.append(df['ADX'].values)
        if 'ATR' in df.columns:
            features.append(df['ATR'].values)
        if 'Volume_Ratio' in df.columns:
            features.append(df['Volume_Ratio'].values)
        if 'CMF' in df.columns:
            features.append(df['CMF'].values)
        if 'OBV_slope' in df.columns:
            features.append(df['OBV_slope'].values)
        if 'Supertrend_Dir' in df.columns:
            features.append(df['Supertrend_Dir'].values)
        
        for lag in [1, 3, 5, 10]:
            features.append(df['close'].shift(lag).pct_change().values)
        
        for window in [5, 10, 20]:
            features.append(df['close'].rolling(window).mean().pct_change().values)
            features.append(df['close'].rolling(window).std().values)
        
        X = np.column_stack([f for f in features if len(f) == len(df)])
        X = X[~np.isnan(X).any(axis=1)]
        return X
    
    def prepare_lstm_data(self, data, sequence_length=30):
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def train(self, df):
        if df is None or len(df) < 100:
            return False
        
        with self._training_lock:
            try:
                X = self.prepare_features(df)
                if len(X) < 50:
                    return False
                
                y = df['close'].pct_change(3).shift(-3).values
                y = y[~np.isnan(y)][:len(X)]
                X = X[:len(y)]
                
                if len(X) < 30:
                    return False
                
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                y_train_class = (y_train > 0).astype(int)
                y_test_class = (y_test > 0).astype(int)
                
                self.rf_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                self.rf_model.fit(X_train_scaled, y_train_class)
                rf_acc = accuracy_score(y_test_class, self.rf_model.predict(X_test_scaled))
                
                self.lgb_model = LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
                self.lgb_model.fit(X_train_scaled, y_train_class)
                lgb_acc = accuracy_score(y_test_class, self.lgb_model.predict(X_test_scaled))
                
                lstm_acc = 0
                if len(X_train_scaled) > self.sequence_length + 10:
                    X_lstm, y_lstm = self.prepare_lstm_data(X_train_scaled, self.sequence_length)
                    
                    if len(X_lstm) > 20:
                        self.lstm_model = Sequential([
                            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, X_lstm.shape[2])),
                            Dropout(0.2),
                            LSTM(25, return_sequences=False),
                            Dropout(0.2),
                            Dense(1)
                        ])
                        
                        self.lstm_model.compile(optimizer='adam', loss='mse')
                        self.lstm_model.fit(
                            X_lstm, y_lstm,
                            epochs=20,
                            batch_size=32,
                            verbose=0,
                            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
                        )
                        
                        if len(X_test_scaled) >= self.sequence_length:
                            X_test_lstm = X_test_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
                            lstm_pred = self.lstm_model.predict(X_test_lstm, verbose=0)[0][0]
                            lstm_acc = 0.6 + abs(lstm_pred) * 0.3
                
                self.is_trained = True
                self.last_training_time = datetime.now()
                self.performance_score = (rf_acc + lgb_acc + lstm_acc) / 3
                return True
                    
            except Exception:
                return False
    
    def predict(self, df):
        if not self.is_trained or df is None or len(df) < 50:
            return {'signal': 0, 'confidence': 50, 'details': 'Model not trained'}
        
        try:
            X = self.prepare_features(df)
            if len(X) == 0:
                return {'signal': 0, 'confidence': 50, 'details': 'No features'}
            
            X_scaled = self.scaler.transform(X)
            
            rf_pred = 0
            if self.rf_model is not None:
                rf_pred = self.rf_model.predict_proba([X_scaled[-1]])[0][1]
                rf_pred = (rf_pred - 0.5) * 2
            
            lgb_pred = 0
            if self.lgb_model is not None:
                lgb_pred = self.lgb_model.predict_proba([X_scaled[-1]])[0][1]
                lgb_pred = (lgb_pred - 0.5) * 2
            
            lstm_pred = 0
            if self.lstm_model is not None and len(X_scaled) >= self.sequence_length:
                lstm_input = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
                lstm_pred = float(self.lstm_model.predict(lstm_input, verbose=0)[0][0])
                lstm_pred = np.clip(lstm_pred, -1, 1)
            
            weights = {'rf': 0.30, 'lgb': 0.35, 'lstm': 0.35}
            valid_preds = []
            valid_weights = []
            
            if self.rf_model is not None:
                valid_preds.append(rf_pred)
                valid_weights.append(weights['rf'])
            if self.lgb_model is not None:
                valid_preds.append(lgb_pred)
                valid_weights.append(weights['lgb'])
            if self.lstm_model is not None:
                valid_preds.append(lstm_pred)
                valid_weights.append(weights['lstm'])
            
            total_weight = sum(valid_weights) if valid_weights else 1
            if valid_preds:
                ensemble_signal = sum(p * w for p, w in zip(valid_preds, valid_weights)) / total_weight
            else:
                ensemble_signal = 0
            
            model_signals = []
            if self.rf_model is not None:
                model_signals.append(1 if rf_pred > 0 else -1)
            if self.lgb_model is not None:
                model_signals.append(1 if lgb_pred > 0 else -1)
            if self.lstm_model is not None:
                model_signals.append(1 if lstm_pred > 0 else -1)
            
            agreement = abs(sum(model_signals)) / len(model_signals) if model_signals else 0
            base_conf = 50 + abs(ensemble_signal) * 30 + self.performance_score * 10
            confidence = min(95, base_conf * (0.7 + 0.3 * agreement))
            
            return {
                'signal': round(ensemble_signal, 3),
                'rf_signal': round(rf_pred, 3),
                'lgb_signal': round(lgb_pred, 3),
                'lstm_signal': round(lstm_pred, 3),
                'confidence': min(95, confidence),
                'details': f'RF: {rf_pred:+.3f} | LGB: {lgb_pred:+.3f} | LSTM: {lstm_pred:+.3f}',
                'direction': 'Bullish' if ensemble_signal > 0.1 else ('Bearish' if ensemble_signal < -0.1 else 'Neutral'),
                'performance_score': round(self.performance_score * 100, 1)
            }
        except Exception as e:
            return {'signal': 0, 'confidence': 50, 'details': f'Error: {e}'}

_ai_predictor = AIPredictor()

# ======================== NEW TECHNICAL INDICATORS ========================

def calculate_pivot_points(df):
    """
    📌 Pivot Points (Classic + Fibonacci)
    Fungsi: Menghitung level support/resistance harian berdasarkan high, low, close kemarin
    Kegunaan: Entry/exit level, lihat area reversal potensial
    
    CARA BACA:
    - Pivot: Level tengah, harga di atas = bullish, di bawah = bearish
    - R1, R2, R3: Resistance level (harga bisa reversal di sini)
    - S1, S2, S3: Support level (harga bisa bounce di sini)
    - Fibonacci Pivot: Lebih akurat untuk trend kuat
    """
    try:
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        # Classic Pivot
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        # Fibonacci Pivot
        fib_pivot = (high + low + close) / 3
        diff = high - low
        fib_r1 = fib_pivot + 0.382 * diff
        fib_r2 = fib_pivot + 0.618 * diff
        fib_r3 = fib_pivot + 1.000 * diff
        fib_s1 = fib_pivot - 0.382 * diff
        fib_s2 = fib_pivot - 0.618 * diff
        fib_s3 = fib_pivot - 1.000 * diff
        
        # Interpretasi
        cp = close
        if cp > pivot:
            pivot_sentiment = "🟢 Bullish (di atas pivot)"
        elif cp < pivot:
            pivot_sentiment = "🔴 Bearish (di bawah pivot)"
        else:
            pivot_sentiment = "⚪ Netral (di pivot)"
        
        return {
            'classic': {
                'pivot': round(pivot, 10),
                'r1': round(r1, 10), 'r2': round(r2, 10), 'r3': round(r3, 10),
                's1': round(s1, 10), 's2': round(s2, 10), 's3': round(s3, 10)
            },
            'fibonacci': {
                'pivot': round(fib_pivot, 10),
                'r1': round(fib_r1, 10), 'r2': round(fib_r2, 10), 'r3': round(fib_r3, 10),
                's1': round(fib_s1, 10), 's2': round(fib_s2, 10), 's3': round(fib_s3, 10)
            },
            'sentiment': pivot_sentiment,
            'nearest_resistance': min([r for r in [r1, r2, r3] if r > cp], default=None),
            'nearest_support': max([s for s in [s1, s2, s3] if s < cp], default=None)
        }
    except Exception:
        return {'classic': {}, 'fibonacci': {}, 'sentiment': 'N/A', 'nearest_resistance': None, 'nearest_support': None}


def detect_harmonic_patterns(df, lookback=100):
    """
    📐 Harmonic Patterns
    Fungsi: Mendeteksi pola harmonic untuk reversal (Gartley, Bat, Crab, Butterfly, Shark)
    Kegunaan: Entry di titik reversal yang akurat
    
    CARA BACA:
    - Gartley 🟢: Reversal up, entry di D point
    - Bat 🦇: Reversal kuat, entry di D point
    - Crab 🦀: Reversal sangat kuat, entry di D point
    - Butterfly 🦋: Reversal kuat, entry di D point
    - Semakin tinggi score = semakin akurat polanya
    """
    try:
        if len(df) < lookback:
            return {'patterns': [], 'type': 'none', 'score': 0, 'signal': 'No harmonic pattern'}
        
        rec = df.tail(lookback).copy()
        high = rec['high'].values
        low = rec['low'].values
        
        swing_highs, swing_lows = find_swing_points(high, low, left_bars=3, right_bars=3, min_deviation=0.005)
        
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return {'patterns': [], 'type': 'none', 'score': 0, 'signal': 'No harmonic pattern'}
        
        # Gabung dan sortir swing
        all_swings = sorted(swing_highs + swing_lows, key=lambda x: x[0])
        
        if len(all_swings) < 5:
            return {'patterns': [], 'type': 'none', 'score': 0, 'signal': 'No harmonic pattern'}
        
        # Ambil 5 swing terakhir untuk pola X-A-B-C-D
        last_5 = all_swings[-5:]
        x, a, b, c, d = [p[1] for p in last_5]
        
        patterns = []
        score = 0
        pattern_type = 'none'
        signal = 'No harmonic pattern'
        
        # Hitung Fibonacci ratios
        xa = abs(a - x)
        ab = abs(b - a)
        bc = abs(c - b)
        cd = abs(d - c)
        
        if xa == 0 or ab == 0 or bc == 0 or cd == 0:
            return {'patterns': [], 'type': 'none', 'score': 0, 'signal': 'No harmonic pattern'}
        
        # Gartley: XA = 0.618, AB = 0.382, BC = 1.272
        if 0.55 < ab/xa < 0.65 and 0.35 < bc/ab < 0.45 and 1.2 < cd/bc < 1.35:
            patterns.append('🟢 Gartley - Potensi Reversal Up!')
            score += 3
            pattern_type = 'bullish'
            signal = '🟢 Gartley Pattern Detected!'
        
        # Bat: XA = 0.382, AB = 0.382, BC = 1.618
        if 0.35 < ab/xa < 0.45 and 0.35 < bc/ab < 0.45 and 1.5 < cd/bc < 1.7:
            patterns.append('🦇 Bat - Strong Reversal!')
            score += 4
            pattern_type = 'bullish'
            signal = '🦇 Bat Pattern Detected!'
        
        # Crab: XA = 0.618, AB = 0.382, BC = 2.618
        if 0.55 < ab/xa < 0.65 and 0.35 < bc/ab < 0.45 and 2.5 < cd/bc < 2.7:
            patterns.append('🦀 Crab - Very Strong Reversal!')
            score += 5
            pattern_type = 'bullish'
            signal = '🦀 Crab Pattern Detected!'
        
        # Butterfly: XA = 0.786, AB = 0.786, BC = 1.618
        if 0.75 < ab/xa < 0.82 and 0.75 < bc/ab < 0.82 and 1.5 < cd/bc < 1.7:
            patterns.append('🦋 Butterfly - Strong Reversal!')
            score += 4
            pattern_type = 'bullish'
            signal = '🦋 Butterfly Pattern Detected!'
        
        # Shark: XA = 0.886, AB = 1.13, BC = 1.618
        if 0.85 < ab/xa < 0.92 and 1.1 < bc/ab < 1.2 and 1.5 < cd/bc < 1.7:
            patterns.append('🦈 Shark - Reversal!')
            score += 3
            pattern_type = 'bullish'
            signal = '🦈 Shark Pattern Detected!'
        
        return {
            'patterns': patterns,
            'type': pattern_type,
            'score': score,
            'signal': signal,
            'swing_points': {'x': x, 'a': a, 'b': b, 'c': c, 'd': d}
        }
    except Exception:
        return {'patterns': [], 'type': 'none', 'score': 0, 'signal': 'Error detecting harmonic patterns'}


def calculate_atr_trailing_stop(df, period=14, multiplier=2.0):
    """
    🎯 ATR Trailing Stop
    Fungsi: Stop loss yang bergerak mengikuti harga berdasarkan volatilitas
    Kegunaan: Proteksi profit, trailing stop yang dinamis
    
    CARA BACA:
    - Long Stop: Level stop untuk posisi beli (harga turun sampai sini = exit)
    - Short Stop: Level stop untuk posisi jual (harga naik sampai sini = exit)
    - Semakin tinggi multiplier = stop lebih jauh (lebih longgar)
    - Untuk low cap, pakai multiplier 2.5-3 (volatilitas tinggi)
    - Untuk high cap, pakai multiplier 1.5-2 (volatilitas rendah)
    """
    try:
        if 'ATR' not in df.columns:
            return {'long_stop': None, 'short_stop': None, 'current_stop': None, 'stop_distance_pct': 0}
        
        atr = df['ATR'].iloc[-1]
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        # Long stop (untuk posisi buy) - trailing di bawah harga
        long_stop = high - multiplier * atr
        
        # Short stop (untuk posisi sell) - trailing di atas harga
        short_stop = low + multiplier * atr
        
        # Current stop berdasarkan posisi
        if close > long_stop:
            stop = long_stop
            stop_type = 'Long (Buy)'
        else:
            stop = short_stop
            stop_type = 'Short (Sell)'
        
        stop_distance = (close - stop) / close * 100
        
        # Rekomendasi multiplier berdasarkan kapitalisasi
        price = close
        if price < 0.01:  # Small cap
            recommended_mult = 3.0
            recommendation = "Small Cap - Pakai multiplier 3.0"
        elif price < 0.1:  # Mid-low cap
            recommended_mult = 2.5
            recommendation = "Mid-Low Cap - Pakai multiplier 2.5"
        elif price < 1:  # Mid cap
            recommended_mult = 2.0
            recommendation = "Mid Cap - Pakai multiplier 2.0"
        else:  # High cap
            recommended_mult = 1.5
            recommendation = "High Cap - Pakai multiplier 1.5"
        
        return {
            'long_stop': round(long_stop, 10),
            'short_stop': round(short_stop, 10),
            'current_stop': round(stop, 10),
            'stop_type': stop_type,
            'stop_distance_pct': round(stop_distance, 2),
            'recommended_mult': recommended_mult,
            'recommendation': recommendation
        }
    except Exception:
        return {'long_stop': None, 'short_stop': None, 'current_stop': None, 'stop_distance_pct': 0}


def calculate_heikin_ashi(df):
    """
    📈 Heikin Ashi
    Fungsi: Candle yang di-smooth untuk filter noise, lihat trend sebenarnya
    Kegunaan: Lihat trend tanpa gangguan noise, deteksi reversal lebih awal
    
    CARA BACA:
    - HA Bull = 🟢 (candle hijau) = trend naik
    - HA Bear = 🔴 (candle merah) = trend turun
    - HA Streak = berapa candle searah berturut-turut
    - Semakin panjang streak = trend semakin kuat
    - HA reversal = ketika warna berubah (potensi reversal)
    """
    try:
        df = df.copy()
        df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['ha_open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
        df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
        df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)
        
        df['ha_bull'] = df['ha_close'] > df['ha_open']
        
        # Hitung streak
        df['ha_streak'] = df['ha_bull'].groupby((df['ha_bull'] != df['ha_bull'].shift()).cumsum()).cumsum()
        
        last = df.iloc[-1]
        last_ha_bull = bool(last['ha_bull'])
        streak = int(last['ha_streak'])
        
        # Deteksi reversal
        reversal = False
        if len(df) >= 3:
            prev_bull = bool(df['ha_bull'].iloc[-2])
            if last_ha_bull != prev_bull:
                reversal = True
        
        return {
            'ha_close': round(last['ha_close'], 10),
            'ha_open': round(last['ha_open'], 10),
            'ha_high': round(last['ha_high'], 10),
            'ha_low': round(last['ha_low'], 10),
            'ha_bull': last_ha_bull,
            'ha_trend': '🟢 Bullish' if last_ha_bull else '🔴 Bearish',
            'ha_streak': streak,
            'ha_reversal': reversal,
            'ha_signal': '🔄 Reversal Detected!' if reversal else ('📈 Trend Bullish' if last_ha_bull else '📉 Trend Bearish')
        }
    except Exception:
        return {'ha_trend': 'N/A', 'ha_streak': 0, 'ha_reversal': False, 'ha_signal': 'N/A'}


def calculate_stoch_rsi(df, period=14, smooth=3):
    """
    📊 Stochastic RSI
    Fungsi: RSI yang lebih sensitif untuk deteksi oversold/overbought
    Kegunaan: Entry timing di area oversold/overbought
    
    CARA BACA:
    - K < 20 = Oversold (potensi beli)
    - K > 80 = Overbought (potensi jual)
    - K > D = Bullish momentum
    - K < D = Bearish momentum
    - Cross Up = K naik di atas D (sinyal beli)
    - Cross Down = K turun di bawah D (sinyal jual)
    """
    try:
        if 'RSI' not in df.columns:
            return {'stoch_rsi': 50, 'k': 50, 'd': 50, 'signal': 'N/A', 'cross_up': False, 'cross_down': False}
        
        rsi = df['RSI']
        min_rsi = rsi.rolling(period).min()
        max_rsi = rsi.rolling(period).max()
        stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
        
        k = stoch_rsi.rolling(smooth).mean()
        d = k.rolling(smooth).mean()
        
        k_val = k.iloc[-1] if not pd.isna(k.iloc[-1]) else 50
        d_val = d.iloc[-1] if not pd.isna(d.iloc[-1]) else 50
        
        # Deteksi cross
        cross_up = False
        cross_down = False
        if len(k) >= 2:
            if k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2]:
                cross_up = True
            if k.iloc[-1] < d.iloc[-1] and k.iloc[-2] >= d.iloc[-2]:
                cross_down = True
        
        if k_val < 20:
            signal = "🟢 Oversold - Potensi Beli!"
        elif k_val > 80:
            signal = "🔴 Overbought - Potensi Jual!"
        elif k_val > d_val:
            signal = "📈 Bullish Momentum"
        else:
            signal = "📉 Bearish Momentum"
        
        return {
            'stoch_rsi': round(k_val, 1),
            'k': round(k_val, 1),
            'd': round(d_val, 1),
            'signal': signal,
            'cross_up': cross_up,
            'cross_down': cross_down,
            'cross_signal': '🟢 Cross Up - Sinyal Beli!' if cross_up else ('🔴 Cross Down - Sinyal Jual!' if cross_down else 'Tidak ada cross')
        }
    except Exception:
        return {'stoch_rsi': 50, 'k': 50, 'd': 50, 'signal': 'N/A', 'cross_up': False, 'cross_down': False}


def calculate_mfi(df, period=14):
    """
    💰 Money Flow Index (MFI)
    Fungsi: Volume + RSI, deteksi akumulasi/distribusi
    Kegunaan: Lihat apakah bandar akumulasi atau distribusi
    
    CARA BACA:
    - MFI < 20 = Oversold + volume tinggi = Akumulasi!
    - MFI > 80 = Overbought + volume tinggi = Distribusi!
    - Bullish Divergence = Harga turun tapi MFI naik = Akumulasi!
    - Bearish Divergence = Harga naik tapi MFI turun = Distribusi!
    """
    try:
        typical = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical * df['volume']
        
        positive_flow = money_flow.where(df['close'] > df['close'].shift(1), 0)
        negative_flow = money_flow.where(df['close'] < df['close'].shift(1), 0)
        
        pos_sum = positive_flow.rolling(period).sum()
        neg_sum = negative_flow.rolling(period).sum()
        
        mfi = 100 - (100 / (1 + pos_sum / neg_sum.replace(0, np.nan)))
        
        mfi_val = mfi.iloc[-1] if not pd.isna(mfi.iloc[-1]) else 50
        
        # Deteksi divergensi
        divergence = 'None'
        if len(mfi) >= 6 and len(df) >= 6:
            if mfi.iloc[-1] > mfi.iloc[-5] and df['close'].iloc[-1] < df['close'].iloc[-5]:
                divergence = '🟢 Bullish Divergence - Akumulasi!'
            elif mfi.iloc[-1] < mfi.iloc[-5] and df['close'].iloc[-1] > df['close'].iloc[-5]:
                divergence = '🔴 Bearish Divergence - Distribusi!'
        
        if mfi_val < 20:
            signal = "🟢 Oversold - Potensi Akumulasi!"
        elif mfi_val > 80:
            signal = "🔴 Overbought - Potensi Distribusi!"
        else:
            signal = "⚪ Netral"
        
        return {
            'mfi': round(mfi_val, 1),
            'signal': signal,
            'divergence': divergence,
            'is_accumulation': mfi_val < 30 and divergence != 'Bearish Divergence - Distribusi!',
            'is_distribution': mfi_val > 70 and divergence != 'Bullish Divergence - Akumulasi!'
        }
    except Exception:
        return {'mfi': 50, 'signal': 'N/A', 'divergence': 'None', 'is_accumulation': False, 'is_distribution': False}


def calculate_aroon(df, period=25):
    """
    📊 Aroon Indicator
    Fungsi: Deteksi trend strength dan reversal
    Kegunaan: Lihat apakah trend kuat atau lemah
    
    CARA BACA:
    - Aroon Up > 70 = Trend naik kuat
    - Aroon Down > 70 = Trend turun kuat
    - Aroon Up < 50 dan Aroon Down < 50 = Trend lemah (sideways)
    - Cross = Aroon Up dan Down bertemu = Potensi reversal
    """
    try:
        high_idx = df['high'].rolling(period).apply(lambda x: x.argmax() if len(x) == period else 0)
        low_idx = df['low'].rolling(period).apply(lambda x: x.argmin() if len(x) == period else 0)
        
        aroon_up = (period - high_idx) / period * 100
        aroon_down = (period - low_idx) / period * 100
        
        au = aroon_up.iloc[-1] if not pd.isna(aroon_up.iloc[-1]) else 50
        ad = aroon_down.iloc[-1] if not pd.isna(aroon_down.iloc[-1]) else 50
        
        # Deteksi cross
        cross = False
        if len(aroon_up) >= 2:
            if (aroon_up.iloc[-1] > aroon_down.iloc[-1] and aroon_up.iloc[-2] <= aroon_down.iloc[-2]):
                cross = True
                cross_signal = "🟢 Aroon Cross Up - Potensi Trend Naik!"
            elif (aroon_up.iloc[-1] < aroon_down.iloc[-1] and aroon_up.iloc[-2] >= aroon_down.iloc[-2]):
                cross = True
                cross_signal = "🔴 Aroon Cross Down - Potensi Trend Turun!"
            else:
                cross_signal = "Tidak ada cross"
        else:
            cross_signal = "Tidak ada cross"
        
        if au > 70 and ad < 30:
            signal = "🟢 Strong Uptrend!"
        elif ad > 70 and au < 30:
            signal = "🔴 Strong Downtrend!"
        elif au < 50 and ad < 50:
            signal = "⚪ Weak Trend (Sideways)"
        elif au > ad:
            signal = "📈 Uptrend (Moderate)"
        else:
            signal = "📉 Downtrend (Moderate)"
        
        return {
            'aroon_up': round(au, 1),
            'aroon_down': round(ad, 1),
            'signal': signal,
            'cross': cross,
            'cross_signal': cross_signal
        }
    except Exception:
        return {'aroon_up': 50, 'aroon_down': 50, 'signal': 'N/A', 'cross': False, 'cross_signal': 'N/A'}


def calculate_zig_zag(df, lookback=100, deviation=0.03):
    """
    📐 Zig Zag
    Fungsi: Filter noise, lihat swing points dengan jelas
    Kegunaan: Wave counting, support/resistance, harmonic patterns
    
    CARA BACA:
    - Zig Zag = garis yang menghubungkan swing high dan swing low
    - Semakin kecil deviation = lebih banyak swing (lebih detail)
    - Semakin besar deviation = lebih sedikit swing (lebih jelas)
    - Untuk low cap, pakai deviation 5-8% (volatilitas tinggi)
    - Untuk high cap, pakai deviation 2-3% (volatilitas rendah)
    """
    try:
        if len(df) < lookback:
            return {'swing_highs': [], 'swing_lows': [], 'zig_zag': []}
        
        rec = df.tail(lookback).copy()
        high = rec['high'].values
        low = rec['low'].values
        close = rec['close'].values
        cp = float(rec['close'].iloc[-1])
        
        # Sesuaikan deviation berdasarkan harga
        if cp < 0.01:
            actual_dev = deviation * 3  # Small cap lebih volatile
        elif cp < 0.1:
            actual_dev = deviation * 2  # Mid-low cap
        else:
            actual_dev = deviation  # High cap
        
        swing_highs, swing_lows = find_swing_points(high, low, left_bars=3, right_bars=3, min_deviation=actual_dev)
        
        # Ambil 10 swing terakhir
        last_highs = swing_highs[-10:] if len(swing_highs) >= 10 else swing_highs
        last_lows = swing_lows[-10:] if len(swing_lows) >= 10 else swing_lows
        
        # Buat zig zag line (gabungan swing high dan low)
        zig_zag = []
        all_swings = sorted(swing_highs + swing_lows, key=lambda x: x[0])
        
        for idx, price in all_swings[-20:]:
            zig_zag.append({'index': idx, 'price': round(price, 10)})
        
        return {
            'swing_highs': [{'index': i, 'price': round(p, 10)} for i, p in last_highs],
            'swing_lows': [{'index': i, 'price': round(p, 10)} for i, p in last_lows],
            'zig_zag': zig_zag,
            'deviation_used': round(actual_dev * 100, 2),
            'total_swings': len(swing_highs) + len(swing_lows)
        }
    except Exception:
        return {'swing_highs': [], 'swing_lows': [], 'zig_zag': [], 'deviation_used': 0, 'total_swings': 0}


def calculate_cci(df, period=20):
    """
    📊 CCI (Commodity Channel Index)
    Fungsi: Deteksi cycle dan reversal
    Kegunaan: Entry timing di range market
    
    CARA BACA:
    - CCI > 100 = Overbought (potensi jual)
    - CCI < -100 = Oversold (potensi beli)
    - CCI di antara -100 dan 100 = Range (sideways)
    - CCI breakout dari +100/-100 = Trend kuat
    """
    try:
        typical = (df['high'] + df['low'] + df['close']) / 3
        sma = typical.rolling(period).mean()
        mad = typical.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        
        cci = (typical - sma) / (0.015 * mad)
        
        cci_val = cci.iloc[-1] if not pd.isna(cci.iloc[-1]) else 0
        
        if cci_val > 100:
            signal = "🔴 Overbought - Potensi Jual!"
        elif cci_val < -100:
            signal = "🟢 Oversold - Potensi Beli!"
        elif cci_val > 0:
            signal = "📈 Bullish (Momentum Naik)"
        else:
            signal = "📉 Bearish (Momentum Turun)"
        
        return {
            'cci': round(cci_val, 1),
            'signal': signal,
            'is_overbought': cci_val > 100,
            'is_oversold': cci_val < -100
        }
    except Exception:
        return {'cci': 0, 'signal': 'N/A', 'is_overbought': False, 'is_oversold': False}


def calculate_ultimate_oscillator(df, period1=7, period2=14, period3=28):
    """
    📊 Ultimate Oscillator
    Fungsi: Multi-timeframe oscillator untuk konfirmasi reversal
    Kegunaan: Konfirmasi sinyal reversal dari indikator lain
    
    CARA BACA:
    - Ultimate < 30 = Oversold (potensi beli)
    - Ultimate > 70 = Overbought (potensi jual)
    - Divergence dengan harga = sinyal kuat
    """
    try:
        bp = df['close'] - df[['low', 'close.shift(1)']].min(axis=1)
        tr = df[['high', 'close.shift(1)']].max(axis=1) - df[['low', 'close.shift(1)']].min(axis=1)
        
        avg7 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
        avg14 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
        avg28 = bp.rolling(period3).sum() / tr.rolling(period3).sum()
        
        ultimate = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7
        
        uo = ultimate.iloc[-1] if not pd.isna(ultimate.iloc[-1]) else 50
        
        if uo < 30:
            signal = "🟢 Oversold - Potensi Beli!"
        elif uo > 70:
            signal = "🔴 Overbought - Potensi Jual!"
        else:
            signal = "⚪ Netral"
        
        return {
            'ultimate': round(uo, 1),
            'signal': signal,
            'is_oversold': uo < 30,
            'is_overbought': uo > 70
        }
    except Exception:
        return {'ultimate': 50, 'signal': 'N/A', 'is_oversold': False, 'is_overbought': False}


def calculate_vortex(df, period=14):
    """
    📊 Vortex Indicator
    Fungsi: Deteksi trend direction
    Kegunaan: Konfirmasi trend sedang naik atau turun
    
    CARA BACA:
    - VI+ > VI- = Trend naik (bullish)
    - VI- > VI+ = Trend turun (bearish)
    - Cross = Potensi reversal
    """
    try:
        tr = np.maximum(df['high'] - df['low'], 
                        np.maximum(abs(df['high'] - df['close'].shift(1)),
                                   abs(df['low'] - df['close'].shift(1))))
        
        vm_plus = abs(df['high'] - df['low'].shift(1))
        vm_minus = abs(df['low'] - df['high'].shift(1))
        
        vi_plus = vm_plus.rolling(period).sum() / tr.rolling(period).sum()
        vi_minus = vm_minus.rolling(period).sum() / tr.rolling(period).sum()
        
        vip = vi_plus.iloc[-1] if not pd.isna(vi_plus.iloc[-1]) else 1
        vim = vi_minus.iloc[-1] if not pd.isna(vi_minus.iloc[-1]) else 1
        
        # Deteksi cross
        cross = False
        cross_signal = "Tidak ada cross"
        if len(vi_plus) >= 2:
            if vi_plus.iloc[-1] > vi_minus.iloc[-1] and vi_plus.iloc[-2] <= vi_minus.iloc[-2]:
                cross = True
                cross_signal = "🟢 VI+ Cross Up - Potensi Trend Naik!"
            elif vi_plus.iloc[-1] < vi_minus.iloc[-1] and vi_plus.iloc[-2] >= vi_minus.iloc[-2]:
                cross = True
                cross_signal = "🔴 VI+ Cross Down - Potensi Trend Turun!"
        
        if vip > vim:
            signal = "📈 Trend Naik (Bullish)"
        elif vim > vip:
            signal = "📉 Trend Turun (Bearish)"
        else:
            signal = "⚪ Netral"
        
        return {
            'vi_plus': round(vip, 3),
            'vi_minus': round(vim, 3),
            'signal': signal,
            'cross': cross,
            'cross_signal': cross_signal
        }
    except Exception:
        return {'vi_plus': 1, 'vi_minus': 1, 'signal': 'N/A', 'cross': False, 'cross_signal': 'N/A'}


def calculate_mass_index(df, period=9, ema_period=25):
    """
    📊 Mass Index
    Fungsi: Deteksi reversal dari volatility spike
    Kegunaan: Warning reversal ketika volatility meningkat
    
    CARA BACA:
    - Mass Index > 27 = Potensi reversal!
    - Biasanya diikuti oleh reversal dalam 1-2 hari
    - Untuk low cap, threshold lebih tinggi (30)
    - Untuk high cap, threshold lebih rendah (25)
    """
    try:
        high_low = df['high'] - df['low']
        ema9 = high_low.ewm(span=period, adjust=False).mean()
        ema9_ema9 = ema9.ewm(span=period, adjust=False).mean()
        mass = (ema9 / ema9_ema9).rolling(ema_period).sum()
        
        mass_val = mass.iloc[-1] if not pd.isna(mass.iloc[-1]) else 0
        
        # Threshold berdasarkan harga
        cp = df['close'].iloc[-1]
        if cp < 0.01:
            threshold = 30  # Small cap
        elif cp < 0.1:
            threshold = 28  # Mid-low cap
        else:
            threshold = 26  # High cap
        
        if mass_val > threshold:
            signal = f"🔴 Mass Index {mass_val:.1f} > {threshold} - Potensi Reversal!"
            reversal_potential = "HIGH"
        elif mass_val > threshold - 2:
            signal = f"⚠️ Mass Index {mass_val:.1f} - Mendekati Reversal"
            reversal_potential = "MEDIUM"
        else:
            signal = f"✅ Mass Index {mass_val:.1f} - Normal"
            reversal_potential = "LOW"
        
        return {
            'mass_index': round(mass_val, 1),
            'threshold': threshold,
            'signal': signal,
            'reversal_potential': reversal_potential
        }
    except Exception:
        return {'mass_index': 0, 'threshold': 26, 'signal': 'N/A', 'reversal_potential': 'N/A'}


def calculate_rvi(df, period=10):
    """
    📊 RVI (Relative Vigor Index)
    Fungsi: Deteksi momentum dengan close/open
    Kegunaan: Konfirmasi momentum
    
    CARA BACA:
    - RVI > 0 = Bullish momentum
    - RVI < 0 = Bearish momentum
    - Cross RVI dan signal = Potensi reversal
    """
    try:
        close_open = df['close'] - df['open']
        high_low = df['high'] - df['low']
        
        num = close_open.rolling(period).sum()
        den = high_low.rolling(period).sum()
        
        rvi = num / den.replace(0, np.nan) * 100
        
        rvi_val = rvi.iloc[-1] if not pd.isna(rvi.iloc[-1]) else 0
        
        if rvi_val > 50:
            signal = "📈 Bullish Momentum"
        elif rvi_val < -50:
            signal = "📉 Bearish Momentum"
        else:
            signal = "⚪ Netral Momentum"
        
        return {
            'rvi': round(rvi_val, 1),
            'signal': signal
        }
    except Exception:
        return {'rvi': 0, 'signal': 'N/A'}


def calculate_force_index(df, period=13):
    """
    📊 Elder's Force Index
    Fungsi: Volume + price movement, deteksi kekuatan trend
    Kegunaan: Lihat kekuatan trend sebenarnya
    
    CARA BACA:
    - Force Index > 0 = Kekuatan beli (bullish)
    - Force Index < 0 = Kekuatan jual (bearish)
    - Semakin besar nilai = semakin kuat trend
    - Divergence = potensi reversal
    """
    try:
        force = (df['close'] - df['close'].shift(1)) * df['volume']
        force_ema = force.ewm(span=period, adjust=False).mean()
        
        fi = force_ema.iloc[-1] if not pd.isna(force_ema.iloc[-1]) else 0
        
        # Normalisasi
        vol_avg = df['volume'].mean()
        if vol_avg > 0:
            fi_norm = fi / vol_avg * 100
        else:
            fi_norm = 0
        
        if fi > 0:
            signal = f"📈 Force Index {fi_norm:.1f} - Kekuatan Beli"
        else:
            signal = f"📉 Force Index {fi_norm:.1f} - Kekuatan Jual"
        
        return {
            'force_index': round(fi, 2),
            'force_index_norm': round(fi_norm, 2),
            'signal': signal,
            'is_bullish': fi > 0,
            'is_bearish': fi < 0
        }
    except Exception:
        return {'force_index': 0, 'force_index_norm': 0, 'signal': 'N/A', 'is_bullish': False, 'is_bearish': False}


# ======================== BANDARMOLOGI (EXISTING) ========================

def add_vwap_bands(df):
    try:
        if df is None or len(df) < 20:
            return df
        
        df = df.copy()
        typical = (df['high'] + df['low'] + df['close']) / 3
        cum_tv = (typical * df['volume']).cumsum()
        cum_v = df['volume'].cumsum()
        df['VWAP'] = cum_tv / cum_v.replace(0, np.nan)
        df['VWAP_STD'] = df['close'].rolling(20).std()
        df['VWAP_Upper_1'] = df['VWAP'] + df['VWAP_STD']
        df['VWAP_Upper_2'] = df['VWAP'] + 2 * df['VWAP_STD']
        df['VWAP_Lower_1'] = df['VWAP'] - df['VWAP_STD']
        df['VWAP_Lower_2'] = df['VWAP'] - 2 * df['VWAP_STD']
        df['VWAP_Sentiment'] = np.where(df['close'] > df['VWAP'], 'Bullish', 'Bearish')
        df['VWAP_Distance'] = (df['close'] - df['VWAP']) / df['VWAP'] * 100
        return df
    except Exception:
        return df

def add_adl_analysis(df):
    try:
        if df is None or len(df) < 20:
            return df
        
        df = df.copy()
        high_low = df['high'] - df['low']
        mfm = np.where(high_low != 0, ((df['close'] - df['low']) - (df['high'] - df['close'])) / high_low, 0)
        mfv = mfm * df['volume']
        df['ADL'] = mfv.cumsum()
        df['ADL_Trend'] = df['ADL'] > df['ADL'].rolling(20).mean()
        df['ADL_Slope'] = df['ADL'].diff().rolling(5).mean()
        price_slope = df['close'].diff().rolling(5).mean()
        df['ADL_Divergence'] = np.where(
            (df['ADL_Slope'] > 0) & (price_slope < 0),
            'Bullish Divergence - Bandar Accumulating!',
            np.where(
                (df['ADL_Slope'] < 0) & (price_slope > 0),
                'Bearish Divergence - Bandar Distributing!',
                'Neutral'
            )
        )
        return df
    except Exception:
        return df

def detect_fake_breakout(df, resistance=None):
    if df is None or len(df) < 20:
        return {'fake_breakout': False, 'type': 'none', 'score': 0}
    
    try:
        last = df.iloc[-1]
        cp = float(last['close'])
        fake_breakout = False
        breakout_type = 'none'
        score = 0
        
        if resistance is None:
            resistance = df['high'].rolling(20).max().iloc[-1]
        
        if cp > resistance * 0.995:
            vol_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            body = abs(last['close'] - last['open'])
            range_candle = last['high'] - last['low']
            
            if vol_ratio > 1.5 and body / range_candle < 0.3:
                fake_breakout = True
                breakout_type = 'UP'
                score = 3
                if last['high'] - max(last['close'], last['open']) > range_candle * 0.5:
                    score = 5
                    breakout_type = 'UP_STRONG'
        
        return {
            'fake_breakout': fake_breakout,
            'type': breakout_type,
            'score': score,
            'message': f'🎯 Fake Breakout {breakout_type}!' if fake_breakout else 'No fake breakout'
        }
    except Exception:
        return {'fake_breakout': False, 'type': 'none', 'score': 0}

def detect_whale_activity_enhanced(df_15m, df_1h):
    if df_15m is None or len(df_15m) < 50:
        return {'whale_detected': False, 'score': 0, 'type': 'none'}
    
    try:
        whale_signals = []
        score = 0
        
        for i in range(10, len(df_15m) - 1):
            vol_ma = df_15m['volume'].rolling(20).mean().iloc[i]
            vol_ratio = df_15m['volume'].iloc[i] / vol_ma if vol_ma > 0 else 1
            
            if vol_ratio > 5:
                range_candle = df_15m['high'].iloc[i] - df_15m['low'].iloc[i]
                body = abs(df_15m['close'].iloc[i] - df_15m['open'].iloc[i])
                
                if body / range_candle > 0.7:
                    direction = 'BUY' if df_15m['close'].iloc[i] > df_15m['open'].iloc[i] else 'SELL'
                    whale_signals.append({'direction': direction, 'vol_ratio': vol_ratio, 'price': df_15m['close'].iloc[i]})
                    score += 2
        
        if df_1h is not None and len(df_1h) > 30:
            recent_low = df_1h['low'].tail(10).min()
            cp = df_1h['close'].iloc[-1]
            
            if cp < recent_low * 1.02:
                vol_ratio = df_1h['volume'].iloc[-1] / df_1h['volume'].rolling(20).mean().iloc[-1]
                if vol_ratio > 2:
                    whale_signals.append({'direction': 'DEFEND_SUPPORT', 'vol_ratio': vol_ratio, 'price': cp})
                    score += 3
        
        if df_1h is not None and len(df_1h) > 30:
            price_range = (df_1h['high'].tail(10).max() - df_1h['low'].tail(10).min()) / df_1h['close'].iloc[-1]
            vol_avg = df_1h['volume'].tail(10).mean() / df_1h['volume'].rolling(20).mean().iloc[-1]
            
            if price_range < 0.015 and vol_avg > 1.5:
                whale_signals.append({'direction': 'STEALTH_ACCUMULATION', 'vol_ratio': vol_avg, 'price_range': price_range * 100})
                score += 4
        
        if df_1h is not None and 'CVD' in df_1h.columns and len(df_1h) > 30:
            cvd_slope = df_1h['CVD'].diff().tail(5).mean()
            price_slope = df_1h['close'].diff().tail(5).mean()
            
            if cvd_slope > 0 and price_slope < 0:
                whale_signals.append({'direction': 'WHALE_BUYING_DIP', 'cvd_slope': cvd_slope, 'price_slope': price_slope})
                score += 3
        
        if score >= 8:
            status = "🐋 STRONG WHALE ACTIVITY!"
            action = "Follow the whale - Accumulation confirmed!"
        elif score >= 5:
            status = "🐳 WHALE DETECTED"
            action = "Watch carefully - Smart money moving!"
        elif score >= 3:
            status = "🔍 POSSIBLE WHALE"
            action = "Observe - Wait for confirmation"
        else:
            status = "⚡ NO WHALE ACTIVITY"
            action = "Wait for setup"
        
        return {
            'whale_detected': score >= 3,
            'score': min(10, score),
            'status': status,
            'action': action,
            'signals': whale_signals[-3:],
            'type': 'accumulation' if score >= 5 else 'observation'
        }
    except Exception:
        return {'whale_detected': False, 'score': 0, 'type': 'none'}

def detect_bandar_reversal_candles(df):
    if df is None or len(df) < 3:
        return {'pattern': 'none', 'score': 0, 'signal': 'N/A'}
    
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        cp = float(last['close'])
        op = float(last['open'])
        high = float(last['high'])
        low = float(last['low'])
        
        body = abs(cp - op)
        range_candle = high - low
        
        if range_candle == 0:
            return {'pattern': 'none', 'score': 0, 'signal': 'N/A'}
        
        upper_wick = high - max(cp, op)
        lower_wick = min(cp, op) - low
        body_pct = body / range_candle
        
        pattern = 'none'
        score = 0
        signal = 'N/A'
        
        if lower_wick > body * 2 and lower_wick > upper_wick * 2 and cp > op:
            pattern = 'BULLISH_PIN_BAR'
            score = 3
            signal = '📌 Bullish Pin Bar - Bandar Reversal!'
            if lower_wick > body * 3:
                score = 5
                signal = '🔥 STRONG Bullish Pin Bar - Bandar Accumulating!'
        
        if upper_wick > body * 2 and upper_wick > lower_wick * 2 and cp < op:
            pattern = 'BEARISH_PIN_BAR'
            score = -3
            signal = '📌 Bearish Pin Bar - Bandar Distribution!'
            if upper_wick > body * 3:
                score = -5
                signal = '🔥 STRONG Bearish Pin Bar - Bandar Selling!'
        
        if cp > op and prev['close'] < prev['open']:
            if cp > prev['open'] and op < prev['close']:
                pattern = 'BULLISH_ENGULFING'
                score = 4
                signal = '🟢 Bullish Engulfing - Bandar Buy!'
        
        if cp < op and prev['close'] > prev['open']:
            if cp < prev['open'] and op > prev['close']:
                pattern = 'BEARISH_ENGULFING'
                score = -4
                signal = '🔴 Bearish Engulfing - Bandar Sell!'
        
        if body_pct < 0.1:
            if upper_wick > range_candle * 0.3 and lower_wick > range_candle * 0.3:
                pattern = 'DOJI_LONG_WICK'
                score = 0
                signal = '⚡ Doji - Bandar Testing, Breakout Imminent!'
            elif upper_wick > range_candle * 0.6:
                pattern = 'GRAVESTONE_DOJI'
                score = -2
                signal = '🔴 Gravestone Doji - Potential Top!'
            elif lower_wick > range_candle * 0.6:
                pattern = 'DRAGONFLY_DOJI'
                score = 2
                signal = '🟢 Dragonfly Doji - Potential Bottom!'
        
        return {
            'pattern': pattern,
            'score': score,
            'signal': signal,
            'is_bullish': score > 0,
            'is_bearish': score < 0
        }
    except Exception:
        return {'pattern': 'none', 'score': 0, 'signal': 'N/A'}

# ======================== EXISTING INDICATORS ========================
# (Saya akan singkatkan karena sudah ada di versi sebelumnya)

def hurst_exponent_fixed(series, min_lag=2, max_lag=None):
    try:
        ts = np.array(series, dtype=float)
        ts = ts[~np.isnan(ts)]
        if len(ts) < 50: return 0.5
        if max_lag is None: max_lag = min(len(ts) // 4, 80)
        max_lag = max(min_lag + 1, min(max_lag, len(ts) // 3))
        lags = range(min_lag, max_lag)
        rs_vals = []
        for lag in lags:
            if lag >= len(ts): break
            sub_series = [ts[i:i+lag] for i in range(0, len(ts)-lag+1, lag//2)]
            rs_lag = []
            for sub in sub_series:
                if len(sub) < 2: continue
                mean_sub = np.mean(sub)
                dev = np.cumsum(sub - mean_sub)
                r = np.max(dev) - np.min(dev)
                s = np.std(sub, ddof=1)
                if s > 0 and r > 0: rs_lag.append(r / s)
            if rs_lag: rs_vals.append(np.mean(rs_lag))
        if len(rs_vals) < 3: return 0.5
        log_lags = np.log(list(lags)[:len(rs_vals)])
        log_rs = np.log(rs_vals)
        H = np.polyfit(log_lags, log_rs, 1)[0]
        return float(np.clip(H, 0.15, 0.85))
    except Exception:
        return 0.5

def kalman_smooth_adaptive(prices, lookback=50):
    try:
        n = len(prices)
        prices = np.array(prices, dtype=float)
        if n < 10: return prices, np.zeros(n), np.ones(n) * 0.01
        volatility = np.std(prices[-min(lookback, n):]) / max(np.mean(prices[-min(lookback, n):]), 1e-10)
        pn = max(1e-5, min(1e-3, volatility * 0.01))
        mn = max(1e-4, min(0.05, volatility * 0.05))
        x = np.array([prices[0], 0.0])
        P = np.eye(2)
        F = np.array([[1, 1], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.eye(2) * pn
        R = np.array([[mn]])
        sm = np.zeros(n)
        vl = np.zeros(n)
        unc = np.zeros(n)
        for i in range(n):
            x = F @ x
            P = F @ P @ F.T + Q
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            y = prices[i] - (H @ x)[0]
            x = x + (K @ [[y]]).flatten()
            P = (np.eye(2) - K @ H) @ P
            sm[i] = x[0]
            vl[i] = x[1]
            unc[i] = float(P[0, 0]) ** 0.5
        return sm, vl, unc
    except Exception:
        return prices, np.zeros(len(prices)), np.ones(len(prices)) * 0.01

def monte_carlo_price_range_adaptive(current_price, sigma, days, mu_estimate=0.0, n_sims=2000, seed=42, mc_band_mult=0.42):
    try:
        rng = np.random.default_rng(seed)
        dt = 1.0
        daily_sigma = sigma / np.sqrt(252)
        mu = np.clip(mu_estimate, -0.001, 0.001)
        paths = np.zeros((n_sims, days + 1))
        paths[:, 0] = current_price
        for t in range(1, days + 1):
            z = rng.standard_normal(n_sims)
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * daily_sigma**2) * dt + daily_sigma * np.sqrt(dt) * z)
        final = paths[:, -1]
        p50 = float(np.percentile(final, 50))
        p10_raw = float(np.percentile(final, 10))
        p90_raw = float(np.percentile(final, 90))
        p10 = p50 + (p10_raw - p50) * mc_band_mult / 0.5
        p90 = p50 + (p90_raw - p50) * mc_band_mult / 0.5
        return {'p10': p10, 'p25': float(np.percentile(final, 25)), 'p50': p50, 'p75': float(np.percentile(final, 75)), 'p90': p90}
    except Exception:
        return {'p10': current_price * 0.95, 'p25': current_price * 0.98, 'p50': current_price, 'p75': current_price * 1.02, 'p90': current_price * 1.05}

def bayesian_signal_update_adaptive(prior_signal, likelihood_data, market_condition='normal'):
    try:
        confidence_map = {'trending': 0.70, 'volatile': 0.55, 'ranging': 0.40, 'squeeze': 0.50, 'normal': 0.60}
        confidence = confidence_map.get(market_condition, 0.60)
        alpha = confidence
        posterior = alpha * likelihood_data + (1 - alpha) * prior_signal
        return float(np.clip(posterior, -1, 1))
    except Exception:
        return prior_signal

def garman_klass_volatility(df, window=20):
    try:
        log_hl = np.log(df['high'] / df['low'].replace(0, np.nan)) ** 2
        log_co = np.log(df['close'] / df['open'].replace(0, np.nan)) ** 2
        gk = (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(window).mean()
        return (gk * 252) ** 0.5
    except Exception:
        return pd.Series(np.full(len(df), 0.5), index=df.index)

def yang_zhang_volatility(df, window=20):
    try:
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        log_oc = np.log(df['open'] / df['close'].shift(1).replace(0, np.nan))
        log_co = np.log(df['close'] / df['open'].replace(0, np.nan))
        log_ho = np.log(df['high'] / df['open'].replace(0, np.nan))
        log_lo = np.log(df['low'] / df['open'].replace(0, np.nan))
        var_oc = log_oc.rolling(window).var()
        var_co = log_co.rolling(window).var()
        rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window).mean()
        yz_var = var_oc + k * var_co + (1 - k) * rs
        return (yz_var * 252) ** 0.5
    except Exception:
        return pd.Series(np.full(len(df), 0.5), index=df.index)

def compute_fourier_cycle(series, n_harmonics=3):
    try:
        arr = np.array(series, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 20: return 10, 0.0
        trend = np.polyfit(range(len(arr)), arr, 1)
        detrended = arr - np.polyval(trend, range(len(arr)))
        fft_vals = np.fft.rfft(detrended)
        freqs = np.fft.rfftfreq(len(detrended))
        power = np.abs(fft_vals) ** 2
        power[0] = 0
        if len(power) < 2: return 10, 0.0
        dom_freq_idx = np.argmax(power[1:]) + 1
        dom_period = int(1 / freqs[dom_freq_idx]) if freqs[dom_freq_idx] > 0 else 10
        dom_period = max(3, min(dom_period, len(arr) // 2))
        phase_pos = (len(arr) % dom_period) / dom_period
        phase_signal = np.sin(2 * np.pi * phase_pos)
        return dom_period, float(phase_signal)
    except Exception:
        return 10, 0.0

def classify_volatility_regime_improved(df):
    if df is None or len(df) < 30:
        return "unknown", 1.0
    try:
        last = df.iloc[-1]
        atr_ratio = safe_get(last.get('ATR_Ratio', np.nan), 0.8)
        chop = safe_get(last.get('Choppiness', np.nan), 50.0)
        hv20 = safe_get(last.get('HV20', np.nan), 0.5)
        hv5 = safe_get(last.get('HV5', np.nan), 0.5)
        tt_squeeze = bool(safe_get(last.get('TT_Squeeze', 0), 0))
        adx = safe_get(last.get('ADX', np.nan), 20.0)
        vol_trend = hv5 / max(hv20, 1e-10)
        is_trending = (chop < 45 and adx > 25)
        is_ranging = (chop > 58 or tt_squeeze)
        is_volatile = (atr_ratio > 1.9 or vol_trend > 1.6)
        is_quiet = (atr_ratio < 0.7 and not tt_squeeze)
        is_squeeze = tt_squeeze
        regime_score = {'trending': 0, 'ranging': 0, 'volatile': 0, 'squeeze': 0, 'quiet': 0}
        if is_trending: regime_score['trending'] += 2
        if adx > 30: regime_score['trending'] += 1
        if is_ranging: regime_score['ranging'] += 2
        if tt_squeeze: regime_score['squeeze'] += 3
        if is_volatile: regime_score['volatile'] += 2
        if vol_trend > 1.5: regime_score['volatile'] += 1
        if is_quiet: regime_score['quiet'] += 2
        if atr_ratio < 0.6: regime_score['quiet'] += 1
        regime = max(regime_score, key=regime_score.get)
        if regime_score[regime] == 0: regime = "normal"
        mult_map = {'trending': 1.10, 'trending_volatile': 1.25, 'volatile': 0.85, 'squeeze': 1.15, 'ranging': 0.75, 'quiet': 0.90, 'normal': 1.00}
        return regime, mult_map.get(regime, 1.0)
    except Exception:
        return "normal", 1.0

# ======================== CORE INDICATORS ========================
def calculate_indicators_upgraded(df):
    if df is None or len(df) < 50:
        return None
    try:
        df = df.copy()
        c = df['close']
        h = df['high']
        l = df['low']
        o = df['open']
        v = df['volume']

        df['MA20'] = c.rolling(20).mean()
        df['MA50'] = c.rolling(50).mean()
        df['MA200'] = c.rolling(200).mean()
        df['EMA9'] = c.ewm(span=9, adjust=False).mean()
        df['EMA21'] = c.ewm(span=21, adjust=False).mean()
        df['EMA50'] = c.ewm(span=50, adjust=False).mean()

        half_len = 10
        wh = c.rolling(half_len).apply(lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True)
        wf = c.rolling(half_len * 2).apply(lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True)
        hma_raw = 2 * wh - wf
        df['HMA'] = hma_raw.rolling(int(np.sqrt(half_len * 2))).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
        )
        ema1 = c.ewm(span=21, adjust=False).mean()
        ema2 = ema1.ewm(span=21, adjust=False).mean()
        df['DEMA'] = 2 * ema1 - ema2

        for period in [14, 21, 50]:
            delta = c.diff()
            gain = delta.where(delta > 0, 0.0).ewm(com=period - 1, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0.0)).ewm(com=period - 1, adjust=False).mean()
            rs = gain / loss.replace(0, np.nan)
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI_14']
        df['RSI_MA'] = df['RSI_14'].rolling(14).mean()
        
        df['RSI_Divergence'] = 0
        if len(df) > 30:
            price_low = df['low'].tail(20).min()
            rsi_low = df['RSI'].tail(20).min()
            if price_low == df['low'].iloc[-1] and rsi_low > df['RSI'].iloc[-5]:
                df['RSI_Divergence'].iloc[-1] = 1

        exp8 = c.ewm(span=8, adjust=False).mean()
        exp17 = c.ewm(span=17, adjust=False).mean()
        df['MACD'] = exp8 - exp17
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
        
        exp12 = c.ewm(span=12, adjust=False).mean()
        exp26 = c.ewm(span=26, adjust=False).mean()
        df['MACD_Std'] = exp12 - exp26
        df['Signal_Std'] = df['MACD_Std'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist_Std'] = df['MACD_Std'] - df['Signal_Std']
        
        df['MACD_Divergence'] = 0
        if len(df) > 30:
            if df['MACD_Hist'].iloc[-1] > 0 and df['MACD_Hist'].iloc[-5] < 0:
                df['MACD_Divergence'].iloc[-1] = 1

        pc2 = c.diff()
        ds_pc = pc2.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
        ds_apc = pc2.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
        df['TSI'] = 100 * ds_pc / ds_apc.replace(0, np.nan)
        df['TSI_Signal'] = df['TSI'].ewm(span=7, adjust=False).mean()
        r1 = c.pct_change(10) * 100
        r2 = c.pct_change(13) * 100
        r3 = c.pct_change(14) * 100
        r4 = c.pct_change(15) * 100
        df['KST'] = (r1.rolling(10).mean() * 1 + r2.rolling(13).mean() * 2 +
                     r3.rolling(14).mean() * 3 + r4.rolling(15).mean() * 4)
        df['KST_Signal'] = df['KST'].rolling(9).mean()

        df = _psar_upgraded(df)

        df['Volume_MA20'] = v.rolling(20).mean()
        df['Volume_MA50'] = v.rolling(50).mean()
        df['Volume_Ratio'] = v / df['Volume_MA20'].replace(0, np.nan)
        df['RVOL'] = v / v.rolling(20).mean().replace(0, np.nan)
        df['Volume_Trend'] = df['Volume_Ratio'].rolling(5).mean()

        df['Vol_Delta'] = np.where(
            c > o,
            v * (c - o) / (h - l + 1e-10),
            -v * (o - c) / (h - l + 1e-10)
        )
        df['Vol_Delta_MA'] = df['Vol_Delta'].rolling(10).mean()
        
        df['CVD'] = df['Vol_Delta'].expanding().mean() * len(df)
        df['CVD_Trend'] = df['CVD'].diff().rolling(5).mean()

        df['VWAP'] = (c * v).rolling(20).sum() / v.rolling(20).sum()

        hl = h - l
        hc = (h - c.shift()).abs()
        lc = (l - c.shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['TR'] = tr
        df['ATR'] = tr.rolling(14).mean()
        df['ATR7'] = tr.rolling(7).mean()
        df['ATR3'] = tr.rolling(3).mean()
        df['ATR_Ratio'] = df['ATR'] / df['ATR'].rolling(30).mean().replace(0, np.nan)

        df['BB_Middle'] = c.rolling(20).mean()
        bbs = c.rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2.5 * bbs
        df['BB_Lower'] = df['BB_Middle'] - 2.5 * bbs
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'].replace(0, np.nan)
        df['BB_Pct'] = (c - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']).replace(0, np.nan)
        df['BB_Squeeze'] = (df['BB_Width'] < df['BB_Width'].rolling(50).mean() * 0.5)

        df['KC_Middle'] = c.ewm(span=20, adjust=False).mean()
        df['KC_Upper'] = df['KC_Middle'] + 1.5 * df['ATR']
        df['KC_Lower'] = df['KC_Middle'] - 1.5 * df['ATR']
        df['TT_Squeeze'] = (df['BB_Lower'] > df['KC_Lower']) & (df['BB_Upper'] < df['KC_Upper'])

        df['DC_High'] = h.rolling(20).max()
        df['DC_Low'] = l.rolling(20).min()

        log_ret = np.log(c / c.shift(1))
        df['LogRet'] = log_ret
        df['HV20'] = log_ret.rolling(20).std() * np.sqrt(365)
        df['HV5'] = log_ret.rolling(5).std() * np.sqrt(365)
        df['GK_Vol'] = garman_klass_volatility(df, window=20)
        df['YZ_Vol'] = yang_zhang_volatility(df, window=20)
        df['Vol_Ratio'] = df['HV5'] / df['HV20'].replace(0, np.nan)

        df['ROC_3'] = c.pct_change(3) * 100
        df['ROC_5'] = c.pct_change(5) * 100
        df['ROC'] = c.pct_change(10) * 100
        df['ROC_1'] = c.pct_change(1) * 100

        df['body'] = (c - o).abs()
        df['candle_range'] = h - l
        df['body_ratio'] = df['body'] / df['candle_range'].replace(0, np.nan)
        df['upper_wick'] = h - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - l
        df['is_bull'] = (c > o).astype(int)
        df['bull_streak'] = df['is_bull'].groupby((df['is_bull'] != df['is_bull'].shift()).cumsum()).cumsum()
        df['bear_streak'] = (1 - df['is_bull']).groupby((df['is_bull'] != df['is_bull'].shift()).cumsum()).cumsum()

        atr_sum = tr.rolling(14).sum()
        hl14 = h.rolling(14).max() - l.rolling(14).min()
        df['Choppiness'] = 100 * np.log10(atr_sum / hl14.replace(0, np.nan)) / np.log10(14)

        df['Pivot'] = (h.shift(1) + l.shift(1) + c.shift(1)) / 3
        df['PP_R1'] = 2 * df['Pivot'] - l.shift(1)
        df['PP_S1'] = 2 * df['Pivot'] - h.shift(1)
        df['PP_R2'] = df['Pivot'] + (h.shift(1) - l.shift(1))
        df['PP_S2'] = df['Pivot'] - (h.shift(1) - l.shift(1))

        df = _adx_upgraded(df)
        df = _supertrend_upgraded(df)
        df = _cmf_upgraded(df)
        
        df['EWO'] = c.ewm(span=5, adjust=False).mean() - c.ewm(span=35, adjust=False).mean()
        df['EWO_Signal'] = df['EWO'].rolling(10).mean()

        # Bandarmologi
        df = add_vwap_bands(df)
        df = add_adl_analysis(df)

        return df
    except Exception:
        return None

def _psar_upgraded(df, step=0.02, max_step=0.2):
    try:
        df = df.copy()
        n = len(df)
        psar = df['close'].values.copy()
        bull = np.ones(n, dtype=bool)
        af = np.full(n, step)
        ep = df['low'].values.copy()
        for i in range(2, n):
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            if bull[i-1]:
                if df['low'].values[i] < psar[i]:
                    bull[i] = False
                    psar[i] = ep[i-1]
                    ep[i] = df['low'].values[i]
                    af[i] = step
                else:
                    bull[i] = True
                    psar[i] = min(psar[i], df['low'].values[i-1], df['low'].values[i-2] if i >= 2 else df['low'].values[i-1])
                    if df['high'].values[i] > ep[i-1]:
                        ep[i] = df['high'].values[i]
                        af[i] = min(af[i-1] + step, max_step)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:
                if df['high'].values[i] > psar[i]:
                    bull[i] = True
                    psar[i] = ep[i-1]
                    ep[i] = df['high'].values[i]
                    af[i] = step
                else:
                    bull[i] = False
                    psar[i] = max(psar[i], df['high'].values[i-1], df['high'].values[i-2] if i >= 2 else df['high'].values[i-1])
                    if df['low'].values[i] < ep[i-1]:
                        ep[i] = df['low'].values[i]
                        af[i] = min(af[i-1] + step, max_step)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        df['PSAR'] = psar
        df['PSAR_Bull'] = bull.astype(int)
        return df
    except Exception:
        df['PSAR'] = np.nan
        df['PSAR_Bull'] = 0
        return df

def _adx_upgraded(df, period=14):
    try:
        df = df.copy()
        h = df['high']
        l = df['low']
        c = df['close']
        pdm = h.diff().clip(lower=0)
        mdm = (-l.diff()).clip(lower=0)
        pdm[pdm < mdm] = 0
        mdm[mdm <= pdm] = 0
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.ewm(alpha=1/period, adjust=False).mean()
        pdi = 100 * pdm.ewm(alpha=1/period, adjust=False).mean() / atr14.replace(0, np.nan)
        mdi = 100 * mdm.ewm(alpha=1/period, adjust=False).mean() / atr14.replace(0, np.nan)
        dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
        df['ADX'] = dx.ewm(alpha=1/period, adjust=False).mean()
        df['Plus_DI'] = pdi
        df['Minus_DI'] = mdi
        return df
    except Exception:
        return df

def _supertrend_upgraded(df, period=10, mult=3.0):
    try:
        df = df.copy()
        if 'ATR' not in df.columns: return df
        hl2 = (df['high'] + df['low']) / 2
        bu = (hl2 + mult * df['ATR']).values.copy()
        bl = (hl2 - mult * df['ATR']).values.copy()
        n = len(df)
        fu, fl = bu.copy(), bl.copy()
        st = np.full(n, np.nan)
        dr = np.zeros(n)
        cls = df['close'].values
        for i in range(1, n):
            fu[i] = bu[i] if (bu[i] < fu[i-1] or cls[i-1] > fu[i-1]) else fu[i-1]
            fl[i] = bl[i] if (bl[i] > fl[i-1] or cls[i-1] < fl[i-1]) else fl[i-1]
            if np.isnan(st[i-1]):
                st[i] = fu[i]
                dr[i] = -1
            elif abs(st[i-1] - fu[i-1]) < 1e-10:
                if cls[i] > fu[i]:
                    st[i] = fl[i]
                    dr[i] = 1
                else:
                    st[i] = fu[i]
                    dr[i] = -1
            else:
                if cls[i] < fl[i]:
                    st[i] = fu[i]
                    dr[i] = -1
                else:
                    st[i] = fl[i]
                    dr[i] = 1
        df['Supertrend'] = st
        df['Supertrend_Dir'] = dr
        return df
    except Exception:
        return df

def _cmf_upgraded(df, period=20):
    try:
        df = df.copy()
        hl = (df['high'] - df['low']).replace(0, np.nan)
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / hl
        mfv = clv * df['volume']
        df['CMF'] = mfv.rolling(period).sum() / df['volume'].rolling(period).sum()
        return df
    except Exception:
        return df

# ======================== ICHIMOKU ========================
def calculate_ichimoku_crypto(df, t=7, k=22, sb=44):
    if df is None or len(df) < max(t, k, sb): return df
    try:
        df = df.copy()
        df['tenkan'] = (df['high'].rolling(t).max() + df['low'].rolling(t).min()) / 2
        df['kijun'] = (df['high'].rolling(k).max() + df['low'].rolling(k).min()) / 2
        df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(k)
        sbh = df['high'].rolling(sb).max()
        sbl = df['low'].rolling(sb).min()
        df['senkou_b'] = ((sbh + sbl) / 2).shift(k)
        df['chikou'] = df['close'].shift(-k)
        df['future_senkou_a'] = (df['tenkan'] + df['kijun']) / 2
        df['future_senkou_b'] = (sbh + sbl) / 2
        df['cloud_top'] = df[['senkou_a', 'senkou_b']].max(axis=1)
        df['cloud_bottom'] = df[['senkou_a', 'senkou_b']].min(axis=1)
        df['in_cloud'] = (df['close'] > df['cloud_bottom']) & (df['close'] < df['cloud_top'])
        df['above_cloud'] = df['close'] > df['cloud_top']
        df['below_cloud'] = df['close'] < df['cloud_bottom']
        df['ichi_signal'] = 0
        df.loc[df['tenkan'] > df['kijun'], 'ichi_signal'] = 1
        df.loc[df['tenkan'] < df['kijun'], 'ichi_signal'] = -1
        return df
    except Exception:
        return df

# ======================== OBV ========================
def calculate_obv_upgraded(df):
    if df is None or len(df) < 2: return df
    try:
        df = df.copy()
        diff = df['close'].diff()
        vs = np.where(diff > 0, df['volume'], np.where(diff < 0, -df['volume'], 0))
        df['OBV'] = vs.cumsum()
        df['OBV_MA'] = pd.Series(df['OBV'].values).rolling(20).mean().values
        df['OBV_trend'] = df['OBV'] > df['OBV_MA']
        ov = df['OBV'].tail(20).values
        if not np.isnan(ov).any() and np.abs(ov).mean() > 0:
            df['OBV_slope'] = np.polyfit(range(20), ov, 1)[0] / (np.abs(ov).mean() + 1e-10)
        else:
            df['OBV_slope'] = 0.0
        if len(df) > 30:
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            obv_change = (df['OBV'].iloc[-1] - df['OBV'].iloc[-20]) / abs(df['OBV'].iloc[-20] + 1e-10)
            df['obv_divergence'] = 'bullish' if (price_change < -0.02 and obv_change > 0.05) else 'bearish' if (price_change > 0.02 and obv_change < -0.05) else 'none'
        return df
    except Exception:
        return df

# ======================== SMC - BOS/CHoCH ========================
def detect_bos_choch(df, lookback=100):
    if df is None or len(df) < lookback:
        return {'bos': False, 'choch': False, 'description': 'N/A', 'score': 0}
    
    try:
        rec = df.tail(lookback).copy().reset_index(drop=True)
        high = rec['high'].values
        low = rec['low'].values
        close = rec['close'].values
        cp = float(rec['close'].iloc[-1])
        
        swing_highs, swing_lows = find_swing_points(high, low, left_bars=3, right_bars=3, min_deviation=0.005)
        
        bos = False
        choch = False
        desc = "N/A"
        score = 0
        bos_price = None
        choch_price = None
        
        if swing_highs:
            if len(swing_highs) >= 2:
                if cp > swing_highs[-1][1] and swing_highs[-1][1] > swing_highs[-2][1]:
                    bos = True
                    bos_price = cp
                    desc = "🟢 BOS (Break of Structure) - Bullish Continuation"
                    score = 2
        
        if swing_lows:
            if len(swing_lows) >= 2:
                if cp < swing_lows[-1][1] and swing_lows[-1][1] < swing_lows[-2][1]:
                    bos = True
                    bos_price = cp
                    desc = "🔴 BOS (Break of Structure) - Bearish Continuation"
                    score = -2
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            uptrend = (swing_highs[-1][1] > swing_highs[-2][1] and swing_lows[-1][1] > swing_lows[-2][1])
            if uptrend:
                if cp < swing_lows[-2][1]:
                    choch = True
                    choch_price = cp
                    desc = "🔴 CHoCH (Change of Character) - Potential Reversal Down"
                    score = -3
            else:
                downtrend = (swing_highs[-1][1] < swing_highs[-2][1] and swing_lows[-1][1] < swing_lows[-2][1])
                if downtrend:
                    if cp > swing_highs[-2][1]:
                        choch = True
                        choch_price = cp
                        desc = "🟢 CHoCH (Change of Character) - Potential Reversal Up"
                        score = 3
        
        return {
            'bos': bos,
            'choch': choch,
            'description': desc,
            'score': score,
            'bos_price': bos_price,
            'choch_price': choch_price
        }
    except Exception:
        return {'bos': False, 'choch': False, 'description': 'N/A', 'score': 0}

# ======================== FVG ========================
def detect_fair_value_gap_improved(df):
    if df is None or len(df) < 5:
        return [], [], []
    
    bullish_fvg = []
    bearish_fvg = []
    filled_fvg = []
    
    try:
        for i in range(2, len(df) - 1):
            if df['low'].iloc[i] > df['high'].iloc[i - 2]:
                zone_bottom = df['high'].iloc[i - 2]
                zone_top = df['low'].iloc[i]
                is_filled = False
                fill_pct = 0
                for j in range(i + 1, min(i + 20, len(df))):
                    if df['low'].iloc[j] <= zone_top and df['high'].iloc[j] >= zone_bottom:
                        overlap_bottom = max(zone_bottom, df['low'].iloc[j])
                        overlap_top = min(zone_top, df['high'].iloc[j])
                        if overlap_top > overlap_bottom:
                            fill_pct = (overlap_top - overlap_bottom) / (zone_top - zone_bottom)
                            if fill_pct > 0.5:
                                is_filled = True
                                break
                fvg_data = {'price': (zone_bottom + zone_top) / 2, 'zone_bottom': zone_bottom, 'zone_top': zone_top, 'is_filled': is_filled, 'age': len(df) - i}
                if is_filled: filled_fvg.append(fvg_data)
                else: bullish_fvg.append(fvg_data)
            
            if df['high'].iloc[i] < df['low'].iloc[i - 2]:
                zone_bottom = df['high'].iloc[i]
                zone_top = df['low'].iloc[i - 2]
                is_filled = False
                fill_pct = 0
                for j in range(i + 1, min(i + 20, len(df))):
                    if df['low'].iloc[j] <= zone_top and df['high'].iloc[j] >= zone_bottom:
                        overlap_bottom = max(zone_bottom, df['low'].iloc[j])
                        overlap_top = min(zone_top, df['high'].iloc[j])
                        if overlap_top > overlap_bottom:
                            fill_pct = (overlap_top - overlap_bottom) / (zone_top - zone_bottom)
                            if fill_pct > 0.5:
                                is_filled = True
                                break
                fvg_data = {'price': (zone_bottom + zone_top) / 2, 'zone_bottom': zone_bottom, 'zone_top': zone_top, 'is_filled': is_filled, 'age': len(df) - i}
                if is_filled: filled_fvg.append(fvg_data)
                else: bearish_fvg.append(fvg_data)
        
        return bullish_fvg[:5], bearish_fvg[:5], filled_fvg
    except Exception:
        return [], [], []

def get_nearest_fvg(bullish_fvg, bearish_fvg, current_price):
    nearest = None
    nearest_dist = float('inf')
    fvg_type = None
    for fvg in bullish_fvg:
        if isinstance(fvg, dict):
            price = fvg.get('price', 0)
            if price > 0:
                dist = abs(price - current_price) / current_price * 100
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = fvg
                    fvg_type = 'bullish'
    for fvg in bearish_fvg:
        if isinstance(fvg, dict):
            price = fvg.get('price', 0)
            if price > 0:
                dist = abs(price - current_price) / current_price * 100
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = fvg
                    fvg_type = 'bearish'
    if nearest:
        nearest['type'] = fvg_type
        nearest['distance_pct'] = nearest_dist
    return nearest

# ======================== ORDER BLOCK ========================
def detect_order_blocks_improved(df, lookback=100):
    if df is None or len(df) < lookback:
        return {'order_blocks': [], 'unmitigated': [], 'nearest': None}
    
    try:
        rec = df.tail(lookback).copy().reset_index(drop=True)
        cp = float(rec['close'].iloc[-1])
        order_blocks = []
        
        for i in range(3, len(rec) - 1):
            candle = rec.iloc[i]
            vol_ma = rec['volume'].rolling(20).mean().iloc[i] if i >= 20 else candle['volume']
            body = abs(candle['close'] - candle['open'])
            range_candle = candle['high'] - candle['low']
            
            if candle['volume'] > vol_ma * 1.5 and body > range_candle * 0.6:
                is_mitigated = False
                mitigation_pct = 0
                zone_bottom = candle['low']
                zone_top = candle['high']
                for j in range(i + 1, min(i + 30, len(rec))):
                    if rec['low'].iloc[j] <= zone_top and rec['high'].iloc[j] >= zone_bottom:
                        overlap_bottom = max(zone_bottom, rec['low'].iloc[j])
                        overlap_top = min(zone_top, rec['high'].iloc[j])
                        if overlap_top > overlap_bottom:
                            mitigation_pct = (overlap_top - overlap_bottom) / (zone_top - zone_bottom)
                            if mitigation_pct > 0.3:
                                is_mitigated = True
                                break
                ob_type = 'bull' if candle['close'] > candle['open'] else 'bear'
                ob = {'price': candle['close'], 'type': ob_type, 'strength': candle['volume'] / vol_ma, 'is_mitigated': is_mitigated, 'age': len(rec) - i}
                order_blocks.append(ob)
        
        unmitigated = [ob for ob in order_blocks if not ob['is_mitigated']]
        nearest = None
        if unmitigated:
            for ob in unmitigated:
                dist = abs(ob['price'] - cp) / cp
                if nearest is None or dist < abs(nearest['price'] - cp) / cp:
                    nearest = ob
        
        return {'order_blocks': order_blocks[-10:], 'unmitigated': unmitigated[-5:], 'nearest': nearest}
    except Exception:
        return {'order_blocks': [], 'unmitigated': [], 'nearest': None}

# ======================== LIQUIDITY SWEEP ========================
def detect_liquidity_sweep(df, lookback=50):
    if df is None or len(df) < lookback:
        return {'sweep': False, 'type': 'none', 'score': 0, 'description': 'N/A'}
    
    try:
        rec = df.tail(lookback).copy().reset_index(drop=True)
        high = rec['high'].values
        low = rec['low'].values
        close = rec['close'].values
        swing_highs, swing_lows = find_swing_points(high, low, left_bars=3, right_bars=3, min_deviation=0.005)
        
        sweep_bullish = False
        sweep_bearish = False
        sweep_price = None
        description = "N/A"
        score = 0
        
        if len(swing_lows) >= 2:
            recent_low = swing_lows[-1][1]
            for i in range(max(0, len(rec)-20), len(rec)):
                if low[i] < recent_low and close[i] > recent_low:
                    sweep_bullish = True
                    sweep_price = low[i]
                    description = "🟢 Liquidity Sweep - Stop Hunt (Bullish)"
                    score = 3
                    break
        
        if len(swing_highs) >= 2:
            recent_high = swing_highs[-1][1]
            for i in range(max(0, len(rec)-20), len(rec)):
                if high[i] > recent_high and close[i] < recent_high:
                    sweep_bearish = True
                    sweep_price = high[i]
                    description = "🔴 Liquidity Sweep - Stop Hunt (Bearish)"
                    score = -3
                    break
        
        return {'sweep': sweep_bullish or sweep_bearish, 'type': 'bullish' if sweep_bullish else ('bearish' if sweep_bearish else 'none'), 'price': sweep_price, 'score': score, 'description': description}
    except Exception:
        return {'sweep': False, 'type': 'none', 'score': 0, 'description': 'N/A'}

# ======================== INSTITUTIONAL CANDLE ========================
def detect_institutional_candle(df):
    if df is None or len(df) < 20:
        return {'detected': False, 'candles': [], 'score': 0}
    
    try:
        institutional_candles = []
        vol_ma = df['volume'].rolling(20).mean()
        
        for i in range(max(0, len(df)-50), len(df)):
            candle = df.iloc[i]
            body = abs(candle['close'] - candle['open'])
            range_candle = candle['high'] - candle['low']
            vol_ratio = candle['volume'] / vol_ma.iloc[i] if vol_ma.iloc[i] > 0 else 1
            
            if body > range_candle * 0.6 and vol_ratio > 2:
                is_bullish = candle['close'] > candle['open']
                institutional_candles.append({'type': 'bull' if is_bullish else 'bear', 'vol_ratio': vol_ratio, 'price': candle['close']})
        
        recent_inst = [c for c in institutional_candles if c['idx'] >= len(df) - 10] if 'idx' in institutional_candles[0] else []
        score = min(5, len(recent_inst) * 2)
        
        return {'detected': len(institutional_candles) > 0, 'candles': institutional_candles[-5:], 'recent_count': len(recent_inst), 'score': score}
    except Exception:
        return {'detected': False, 'candles': [], 'score': 0}

# ======================== ABSORPTION DETECTION ========================
def detect_absorption(df):
    if df is None or len(df) < 20:
        return {'detected': False, 'score': 0, 'candles': []}
    
    try:
        absorption_candles = []
        vol_ma = df['volume'].rolling(20).mean()
        
        for i in range(max(0, len(df)-50), len(df)):
            candle = df.iloc[i]
            spread = candle['high'] - candle['low']
            spreads = df['high'].iloc[max(0,i-20):i] - df['low'].iloc[max(0,i-20):i]
            avg_spread = spreads.mean() if len(spreads) > 0 else spread
            
            vol_ratio = candle['volume'] / vol_ma.iloc[i] if vol_ma.iloc[i] > 0 else 1
            
            if vol_ratio > 1.5 and spread < avg_spread * 0.6:
                is_bullish = candle['close'] > candle['open']
                absorption_candles.append({'type': 'bull' if is_bullish else 'bear', 'vol_ratio': vol_ratio, 'price': candle['close']})
        
        recent_abs = [c for c in absorption_candles if c['idx'] >= len(df) - 10] if 'idx' in absorption_candles[0] else []
        score = min(5, len(recent_abs) * 2)
        
        return {'detected': len(absorption_candles) > 0, 'candles': absorption_candles[-5:], 'recent_count': len(recent_abs), 'score': score}
    except Exception:
        return {'detected': False, 'score': 0, 'candles': []}

# ======================== DIVERGENCE ========================
def detect_divergences_improved(df, lookback=30):
    if df is None or len(df) < lookback:
        return {'rsi_bullish': False, 'rsi_bearish': False, 'macd_bullish': False, 'macd_bearish': False, 'hidden_bullish': False, 'hidden_bearish': False, 'strength': 0, 'failed': False, 'rsi_diff': 0}
    
    try:
        rec = df.tail(lookback).copy().reset_index(drop=True)
        lo = rec['low'].values
        hi = rec['high'].values
        close = rec['close'].values
        rs = rec['RSI'].tail(lookback).fillna(50).values
        mh = rec['MACD_Hist'].tail(lookback).fillna(0).values
        
        split = lookback // 2
        for i in range(10, lookback - 10):
            if hi[i] > max(hi[i-5:i]) and hi[i] > max(hi[i:i+5]):
                if abs(i - lookback/2) < lookback * 0.2:
                    split = i
                    break
        
        li1 = np.argmin(lo[:split])
        hi1 = np.argmax(hi[:split])
        li2 = split + np.argmin(lo[split:])
        hi2 = split + np.argmax(hi[split:])
        
        rsi_bullish = lo[li2] < lo[li1] and rs[li2] > rs[li1]
        rsi_bearish = hi[hi2] > hi[hi1] and rs[hi2] < rs[hi1]
        macd_bullish = lo[li2] < lo[li1] and mh[li2] > mh[li1]
        macd_bearish = hi[hi2] > hi[hi1] and mh[hi2] < mh[hi1]
        hidden_bullish = lo[li2] > lo[li1] and rs[li2] < rs[li1]
        hidden_bearish = hi[hi2] < hi[hi1] and rs[hi2] > rs[hi1]
        
        strength = 0
        rsi_diff = 0
        if rsi_bullish or macd_bullish or hidden_bullish:
            if rsi_bullish:
                rsi_diff = abs(rs[li2] - rs[li1])
                price_diff = abs(lo[li2] - lo[li1]) / max(lo[li1], 1e-10)
                strength = min(1.0, (rsi_diff / 20) * 0.7 + (price_diff * 50) * 0.3)
            else:
                strength = 0.5
        elif rsi_bearish or macd_bearish or hidden_bearish:
            if rsi_bearish:
                rsi_diff = abs(rs[hi2] - rs[hi1])
                price_diff = abs(hi[hi2] - hi[hi1]) / max(hi[hi1], 1e-10)
                strength = min(1.0, (rsi_diff / 20) * 0.7 + (price_diff * 50) * 0.3)
            else:
                strength = 0.5
        
        failed = False
        if (rsi_bullish or macd_bullish) and close[-1] < lo[li2]:
            failed = True
        elif (rsi_bearish or macd_bearish) and close[-1] > hi[hi2]:
            failed = True
        
        return {'rsi_bullish': rsi_bullish, 'rsi_bearish': rsi_bearish, 'macd_bullish': macd_bullish, 'macd_bearish': macd_bearish, 'hidden_bullish': hidden_bullish, 'hidden_bearish': hidden_bearish, 'strength': round(strength, 2), 'failed': failed, 'rsi_diff': round(rsi_diff, 1)}
    except Exception:
        return {'rsi_bullish': False, 'rsi_bearish': False, 'macd_bullish': False, 'macd_bearish': False, 'hidden_bullish': False, 'hidden_bearish': False, 'strength': 0, 'failed': False, 'rsi_diff': 0}

# ======================== VOLUME PROFILE ========================
def calculate_volume_profile_improved(df, bins=30):
    if df is None or len(df) < 20:
        return None, None, None
    
    try:
        rec = df.tail(60)
        pmn = rec['low'].min()
        pmx = rec['high'].max()
        if pmx <= pmn: return None, None, None
        
        pb = np.linspace(pmn, pmx, bins + 1)
        vap = np.zeros(bins)
        los = rec['low'].values
        his = rec['high'].values
        vols = rec['volume'].values
        rngs = np.maximum(his - los, 1e-10)
        
        for j in range(bins):
            bl = pb[j]
            bh = pb[j + 1]
            ol = np.maximum(los, bl)
            oh = np.minimum(his, bh)
            vld = (oh > ol)
            rat = np.where(vld, (oh - ol) / rngs, 0)
            vap[j] = np.sum(vols * rat)
        
        pi = np.argmax(vap)
        poc = (pb[pi] + pb[pi + 1]) / 2
        
        tv = vap.sum()
        si = np.argsort(vap)[::-1]
        cum = 0
        vab = []
        for idx in si:
            cum += vap[idx]
            vab.append(idx)
            if cum >= tv * 0.70:
                break
        
        vah = (pb[max(vab)] + pb[max(vab) + 1]) / 2
        val = (pb[min(vab)] + pb[min(vab) + 1]) / 2
        
        return round(poc, 10), round(vah, 10), round(val, 10)
    except Exception:
        return None, None, None

# ======================== WYCKOFF ========================
def detect_wyckoff_improved(df, lookback=80):
    if df is None or len(df) < lookback:
        return "Unknown", "Data tidak cukup", []
    
    try:
        rec = df.tail(lookback).copy()
        high = rec['high'].values
        low = rec['low'].values
        close = rec['close'].values
        volume = rec['volume'].values
        
        events = []
        phase = "Transition"
        msg = "N/A"
        
        swing_highs, swing_lows = find_swing_points(high, low, left_bars=3, right_bars=3, min_deviation=0.005)
        
        vol_avg = np.mean(volume[-30:])
        vol_trend = volume[-1] / vol_avg if vol_avg > 0 else 1
        
        if len(swing_lows) >= 2:
            prev_low = swing_lows[-2][1]
            for i in range(max(0, len(rec)-10), len(rec)):
                if low[i] < prev_low and close[i] > prev_low:
                    events.append("🌱 Spring (Buying Opportunity)")
                    phase = "Accumulation"
                    msg = "Spring detected - Smart Money buying the dip!"
                    break
        
        if len(swing_highs) >= 2:
            prev_high = swing_highs[-2][1]
            if high[-1] > prev_high and close[-1] < prev_high:
                events.append("📊 UTAD (Upthrust After Distribution)")
                phase = "Distribution"
                msg = "UTAD detected - Fake breakout, distribution in progress!"
        
        if len(swing_lows) >= 3:
            if close[-1] > max(close[-10:-1]) and vol_trend > 1.5:
                events.append("📈 SOS (Sign of Strength)")
                if phase == "Accumulation":
                    phase = "Markup"
                    msg = "SOS detected - Markup phase starting!"
        
        if price_range := (max(high[-30:]) - min(low[-30:])) / min(low[-30:]):
            if price_range < 0.15 and vol_trend < 1.2:
                if len(swing_lows) >= 2 and swing_lows[-1][1] > swing_lows[-2][1]:
                    phase = "Accumulation"
                    if not msg or msg == "N/A":
                        msg = "Accumulation phase - Smart Money accumulating"
        
        if len(swing_highs) >= 2 and swing_highs[-1][1] > swing_highs[-2][1]:
            if len(swing_lows) >= 2 and swing_lows[-1][1] > swing_lows[-2][1]:
                if vol_trend > 1.3:
                    phase = "Markup"
                    if not msg or msg == "N/A":
                        msg = "Markup phase - Uptrend with volume confirmation"
        
        return phase, msg, events
    except Exception:
        return "Unknown", "N/A", []

# ======================== SUPPORT / RESISTANCE ========================
def calculate_precise_sr(df, lookback=100):
    if df is None or len(df) < 30:
        return [], []
    rec = df.tail(lookback).copy().reset_index(drop=True)
    cp = rec['close'].iloc[-1]
    
    if cp < 0.01:
        tol = 0.05
    elif cp < 0.1:
        tol = 0.025
    else:
        tol = 0.012
    
    lvl = []
    for lb in [2, 3, 5]:
        for i in range(lb, len(rec) - lb):
            h = rec['high'].iloc[i]
            l = rec['low'].iloc[i]
            if h == rec['high'].iloc[i - lb:i + lb + 1].max():
                lvl.append(('fractal_high', h, 2))
            if l == rec['low'].iloc[i - lb:i + lb + 1].min():
                lvl.append(('fractal_low', l, 2))
    for ma, w in [('MA20', 1), ('MA50', 2), ('MA200', 3)]:
        if ma in rec.columns:
            v = rec[ma].iloc[-1]
            if not pd.isna(v) and v > 0:
                lvl.append((ma, v, w))
    for col, w in [('senkou_a', 2), ('senkou_b', 2), ('kijun', 2), ('tenkan', 1),
                   ('BB_Upper', 1), ('BB_Lower', 1), ('PSAR', 1),
                   ('Supertrend', 2), ('VWAP', 1), ('HMA', 1), ('DEMA', 1),
                   ('DC_High', 2), ('DC_Low', 2), ('PP_R1', 1), ('PP_S1', 1),
                   ('PP_R2', 1), ('PP_S2', 1)]:
        if col in rec.columns:
            v = rec[col].iloc[-1]
            if not pd.isna(v) and v > 0:
                lvl.append((col, v, w))
    fib = calculate_fibonacci(rec)
    for level, price in fib.get('levels', {}).items():
        if price > 0:
            lvl.append((f'Fib_{level}', price, 2))
    pm = 10 ** max(0, len(str(int(cp))) - 1)
    for mult in [0.25, 0.5, 1.0, 2.0]:
        step = pm * mult
        if step > 0:
            rl = round(cp / step) * step
            for off in [-1, 0, 1]:
                l2 = rl + off * step
                if l2 > 0:
                    lvl.append(('round', l2, 1))
    
    def cluster(levels, tolerance=tol):
        if not levels:
            return []
        prices = np.array([l2[1] for l2 in levels])
        weights = np.array([l2[2] for l2 in levels])
        names = [l2[0] for l2 in levels]
        visited = np.zeros(len(prices), dtype=bool)
        clusters = []
        for i in np.argsort(prices):
            if visited[i]:
                continue
            visited[i] = True
            cp2 = [prices[i]]
            cw = [weights[i]]
            cn = [names[i]]
            for j in np.argsort(prices):
                if visited[j]:
                    continue
                if abs(prices[j] - prices[i]) / max(prices[i], 1e-10) < tolerance:
                    visited[j] = True
                    cp2.append(prices[j])
                    cw.append(weights[j])
                    cn.append(names[j])
            wa = np.array(cw)
            clusters.append({
                'price': round(np.average(cp2, weights=wa), 8),
                'strength': len(cp2),
                'weight': int(sum(cw)),
                'methods': list(set(cn)),
                'count': int(sum(cw))
            })
        return sorted(clusters, key=lambda x: x['price'])
    
    cls = cluster(lvl)
    sups = sorted([c for c in cls if c['price'] < cp * 0.998],
                  key=lambda x: (-x['weight'], abs(x['price'] - cp)))[:12]
    ress = sorted([c for c in cls if c['price'] > cp * 1.002],
                  key=lambda x: (-x['weight'], abs(x['price'] - cp)))[:12]
    return sups, ress

def sr_strength_label(count):
    if count >= 6:
        return "💎 Ekstrem"
    elif count >= 4:
        return "🔶 Sangat Kuat"
    elif count >= 3:
        return "🔷 Kuat"
    elif count >= 2:
        return "⬜ Sedang"
    else:
        return "⬜ Lemah"

# ======================== FIBONACCI ========================
def calculate_fibonacci(df, lookback=50):
    if df is None or len(df) < lookback:
        return {}
    try:
        rec = df.tail(lookback)
        high = rec['high'].max()
        low = rec['low'].min()
        diff = high - low
        fib_levels = {
            '0': low,
            '0.236': low + diff * 0.236,
            '0.382': low + diff * 0.382,
            '0.5': low + diff * 0.5,
            '0.618': low + diff * 0.618,
            '0.786': low + diff * 0.786,
            '1': high,
            '1.272': high + diff * 0.272,
            '1.618': high + diff * 0.618,
            '2.618': high + diff * 1.618,
        }
        cp = df['close'].iloc[-1]
        nearest_level = min(fib_levels.items(), key=lambda x: abs(x[1] - cp))
        fib_sentiment = (cp - low) / diff if diff > 0 else 0.5
        return {
            'levels': fib_levels,
            'nearest': nearest_level[0],
            'nearest_price': nearest_level[1],
            'high': high,
            'low': low,
            'sentiment': 'Bullish' if fib_sentiment > 0.5 else 'Bearish',
            'sentiment_score': round((fib_sentiment - 0.5) * 2, 3)
        }
    except Exception:
        return {}

# ======================== CANDLESTICK PATTERNS ========================
def detect_candlestick_patterns_advanced(df):
    if df is None or len(df) < 5:
        return [], {}
    
    patterns = []
    pattern_scores = {}
    pattern_quality = {}
    
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev3 = df.iloc[-4] if len(df) > 3 else None
        
        cp = float(last['close'])
        body = abs(last['close'] - last['open'])
        range_candle = last['high'] - last['low']
        if range_candle == 0:
            return [], {}
        
        upper_wick = last['high'] - max(last['close'], last['open'])
        lower_wick = min(last['close'], last['open']) - last['low']
        body_pct = body / range_candle if range_candle > 0 else 0
        
        if lower_wick > body * 2 and upper_wick < body * 0.5 and last['close'] > last['open']:
            wick_ratio = lower_wick / body if body > 0 else 0
            quality = min(1.0, (wick_ratio - 2) / 3)
            patterns.append("Hammer")
            pattern_scores['Hammer'] = 1.0 + quality * 0.5
            pattern_quality['Hammer'] = quality
        
        if upper_wick > body * 2 and lower_wick < body * 0.5 and last['close'] < last['open']:
            wick_ratio = upper_wick / body if body > 0 else 0
            quality = min(1.0, (wick_ratio - 2) / 3)
            patterns.append("Shooting Star")
            pattern_scores['Shooting Star'] = -1.0 - quality * 0.5
            pattern_quality['Shooting Star'] = quality
        
        if (last['close'] > last['open'] and prev['close'] < prev['open'] and 
            last['close'] > prev['open'] and last['open'] < prev['close']):
            engulf_ratio = (last['close'] - last['open']) / (prev['open'] - prev['close']) if (prev['open'] - prev['close']) > 0 else 0
            quality = min(1.0, engulf_ratio)
            patterns.append("Bullish Engulfing")
            pattern_scores['Bullish Engulfing'] = 1.5 + quality * 0.5
            pattern_quality['Bullish Engulfing'] = quality
        
        if (last['close'] < last['open'] and prev['close'] > prev['open'] and 
            last['close'] < prev['open'] and last['open'] > prev['close']):
            engulf_ratio = (last['open'] - last['close']) / (prev['close'] - prev['open']) if (prev['close'] - prev['open']) > 0 else 0
            quality = min(1.0, engulf_ratio)
            patterns.append("Bearish Engulfing")
            pattern_scores['Bearish Engulfing'] = -1.5 - quality * 0.5
            pattern_quality['Bearish Engulfing'] = quality
        
        if prev3 is not None:
            if (prev3['close'] < prev3['open'] and 
                abs(prev['close'] - prev['open']) < (prev['high'] - prev['low']) * 0.3 and
                last['close'] > last['open'] and 
                last['close'] > (prev3['open'] + prev3['close']) / 2):
                patterns.append("Morning Star")
                pattern_scores['Morning Star'] = 2.0
                pattern_quality['Morning Star'] = 1.0
        
        if prev3 is not None:
            if (prev3['close'] > prev3['open'] and 
                abs(prev['close'] - prev['open']) < (prev['high'] - prev['low']) * 0.3 and
                last['close'] < last['open'] and 
                last['close'] < (prev3['open'] + prev3['close']) / 2):
                patterns.append("Evening Star")
                pattern_scores['Evening Star'] = -2.0
                pattern_quality['Evening Star'] = 1.0
        
        if (prev['close'] > prev['open'] and last['close'] > last['open'] and
            last['high'] < prev['close'] and last['low'] > prev['open']):
            patterns.append("Bullish Harami")
            pattern_scores['Bullish Harami'] = 0.8
            pattern_quality['Bullish Harami'] = 1.0
        
        if (prev['close'] < prev['open'] and last['close'] < last['open'] and
            last['high'] < prev['close'] and last['low'] > prev['open']):
            patterns.append("Bearish Harami")
            pattern_scores['Bearish Harami'] = -0.8
            pattern_quality['Bearish Harami'] = 1.0
        
        if 0.05 < body_pct < 0.3:
            patterns.append("Spinning Top")
            pattern_scores['Spinning Top'] = 0.0
            pattern_quality['Spinning Top'] = 1.0 - (body_pct / 0.3)
        
        net_score = sum(pattern_scores.values()) if pattern_scores else 0
        patterns.append(f"Net Score: {net_score:+.1f}")
        
    except Exception:
        pass
    
    return patterns, pattern_quality

# ======================== CHART PATTERNS ========================
def detect_chart_patterns_improved(df, lookback=100):
    if df is None or len(df) < lookback:
        return {'patterns': [], 'description': 'N/A', 'score': 0, 'reliability': 0, 'target': None}
    
    patterns = []
    desc = "N/A"
    score = 0
    reliability = 0
    target = None
    
    try:
        rec = df.tail(lookback).copy().reset_index(drop=True)
        high = rec['high'].values
        low = rec['low'].values
        close = rec['close'].values
        cp = float(rec['close'].iloc[-1])
        
        swing_highs, swing_lows = find_swing_points(high, low, left_bars=3, right_bars=3, min_deviation=0.005)
        
        def calc_reliability(formation_length):
            if formation_length >= 50:
                return 0.9
            elif formation_length >= 30:
                return 0.7
            elif formation_length >= 20:
                return 0.5
            else:
                return 0.3
        
        if len(swing_highs) >= 3:
            for i in range(len(swing_highs) - 2):
                h1, h2, h3 = swing_highs[i], swing_highs[i+1], swing_highs[i+2]
                if (h1[1] < h2[1] and h3[1] < h2[1] and 
                    abs(h1[1] - h3[1]) / h2[1] < 0.05):
                    neckline = (h1[1] + h3[1]) / 2
                    pattern_length = h3[0] - h1[0]
                    reliability = calc_reliability(pattern_length)
                    
                    if cp < neckline * 0.98:
                        desc = "🔴 Head and Shoulders - BREAKOUT DOWN! (Bearish)"
                        score = -4
                        target = neckline - (h2[1] - neckline)
                        patterns.append("Head and Shoulders (Breakout Confirmed)")
                    elif cp < neckline:
                        desc = "🔴 Head and Shoulders Forming - Near Breakout"
                        score = -2
                        patterns.append("Head and Shoulders (Forming)")
                    else:
                        desc = "🔴 Head and Shoulders Detected"
                        score = -1
                        patterns.append("Head and Shoulders")
                    break
        
        if len(swing_lows) >= 3:
            for i in range(len(swing_lows) - 2):
                l1, l2, l3 = swing_lows[i], swing_lows[i+1], swing_lows[i+2]
                if (l1[1] > l2[1] and l3[1] > l2[1] and 
                    abs(l1[1] - l3[1]) / l2[1] < 0.05):
                    neckline = (l1[1] + l3[1]) / 2
                    pattern_length = l3[0] - l1[0]
                    reliability = calc_reliability(pattern_length)
                    
                    if cp > neckline * 1.02:
                        desc = "🟢 Inverse H&S - BREAKOUT UP! (Bullish)"
                        score = 4
                        target = neckline + (neckline - l2[1])
                        patterns.append("Inverse H&S (Breakout Confirmed)")
                    elif cp > neckline:
                        desc = "🟢 Inverse H&S Forming - Near Breakout"
                        score = 2
                        patterns.append("Inverse H&S (Forming)")
                    else:
                        desc = "🟢 Inverse H&S Detected"
                        score = 1
                        patterns.append("Inverse H&S")
                    break
        
        if len(swing_highs) >= 2:
            h1, h2 = swing_highs[-2], swing_highs[-1]
            if abs(h1[1] - h2[1]) / h1[1] < 0.02 and h1[0] < h2[0]:
                pattern_length = h2[0] - h1[0]
                reliability = calc_reliability(pattern_length)
                support = min(min(high[h1[0]:h2[0]]), min(high[h2[0]:]))
                if cp < support * 0.98:
                    desc = "🔴 Double Top - BREAKOUT DOWN! (Bearish)"
                    score = -3
                    target = support - (h1[1] - support)
                    patterns.append("Double Top (Breakout Confirmed)")
                else:
                    desc = "🔴 Double Top Detected"
                    score = -1
                    patterns.append("Double Top")
        
        if len(swing_lows) >= 2:
            l1, l2 = swing_lows[-2], swing_lows[-1]
            if abs(l1[1] - l2[1]) / l1[1] < 0.02 and l1[0] < l2[0]:
                pattern_length = l2[0] - l1[0]
                reliability = calc_reliability(pattern_length)
                resistance = max(max(low[l1[0]:l2[0]]), max(low[l2[0]:]))
                if cp > resistance * 1.02:
                    desc = "🟢 Double Bottom - BREAKOUT UP! (Bullish)"
                    score = 3
                    target = resistance + (resistance - l1[1])
                    patterns.append("Double Bottom (Breakout Confirmed)")
                else:
                    desc = "🟢 Double Bottom Detected"
                    score = 1
                    patterns.append("Double Bottom")
        
        if len(high) > 30:
            highs_last30 = high[-30:]
            lows_last30 = low[-30:]
            high_range = max(highs_last30) - min(highs_last30)
            low_range = max(lows_last30) - min(lows_last30)
            
            if high_range < low_range * 0.3 and lows_last30[-1] > lows_last30[0]:
                resistance = max(highs_last30)
                if cp > resistance * 0.98:
                    desc = "🟢 Ascending Triangle - BREAKOUT UP! (Bullish)"
                    score = 3
                    target = resistance + (resistance - min(lows_last30))
                    patterns.append("Ascending Triangle (Breakout)")
                else:
                    desc = "🟢 Ascending Triangle Forming"
                    score = 1
                    patterns.append("Ascending Triangle")
                    reliability = 0.6
        
        if len(high) > 30:
            highs_last30 = high[-30:]
            lows_last30 = low[-30:]
            high_range = max(highs_last30) - min(highs_last30)
            low_range = max(lows_last30) - min(lows_last30)
            
            if low_range < high_range * 0.3 and highs_last30[-1] < highs_last30[0]:
                support = min(lows_last30)
                if cp < support * 1.02:
                    desc = "🔴 Descending Triangle - BREAKOUT DOWN! (Bearish)"
                    score = -3
                    target = support - (max(highs_last30) - support)
                    patterns.append("Descending Triangle (Breakout)")
                else:
                    desc = "🔴 Descending Triangle Forming"
                    score = -1
                    patterns.append("Descending Triangle")
                    reliability = 0.6
        
        if len(high) > 30:
            highs_last30 = high[-30:]
            lows_last30 = low[-30:]
            high_range = max(highs_last30) - min(highs_last30)
            low_range = max(lows_last30) - min(lows_last30)
            
            if high_range < low_range * 0.4 and low_range < high_range * 0.4:
                high_slope = (highs_last30[-1] - highs_last30[0]) / 30
                low_slope = (lows_last30[-1] - lows_last30[0]) / 30
                if high_slope < 0 and low_slope > 0:
                    desc = "⚡ Symmetric Triangle - Breakout Imminent"
                    score = 0
                    patterns.append("Symmetric Triangle")
                    reliability = 0.7
        
        if len(close) > 30:
            for i in range(10, len(close) - 10):
                move = (close[i] - close[i-10]) / close[i-10]
                if move > 0.15:
                    consolidation = close[i:i+10].std() / close[i:i+10].mean()
                    if consolidation < 0.02:
                        desc = "🟢 Bull Flag - Continuation Up"
                        score = 2
                        target = close[i] * (1 + move)
                        patterns.append("Bull Flag")
                        reliability = 0.7
                        break
        
        if len(close) > 30:
            for i in range(10, len(close) - 10):
                move = (close[i] - close[i-10]) / close[i-10]
                if move < -0.15:
                    consolidation = close[i:i+10].std() / close[i:i+10].mean()
                    if consolidation < 0.02:
                        desc = "🔴 Bear Flag - Continuation Down"
                        score = -2
                        target = close[i] * (1 + move)
                        patterns.append("Bear Flag")
                        reliability = 0.7
                        break
        
        if len(close) > 40:
            for i in range(20, len(close) - 10):
                cup_start = close[i-20]
                cup_bottom = min(close[i-20:i])
                cup_end = close[i]
                handle = close[i:i+10]
                
                if (cup_end > cup_start * 0.95 and 
                    cup_bottom < cup_start * 0.85 and
                    len(handle) >= 5 and
                    min(handle) > cup_end * 0.95):
                    desc = "🟢 Cup and Handle - Breakout Up"
                    score = 3
                    target = cup_start + (cup_start - cup_bottom)
                    patterns.append("Cup and Handle")
                    reliability = 0.8
                    break
        
        if len(close) > 40:
            mid = len(close) // 2
            left = close[:mid]
            right = close[mid:]
            if (left[-1] < left[0] and right[-1] > right[0] and
                min(left) < left[0] * 0.9 and min(right) < right[0] * 0.9):
                desc = "🟢 Rounding Bottom - Bullish"
                score = 2
                patterns.append("Rounding Bottom")
                reliability = 0.7
        
        if len(high) > 30 and len(low) > 30:
            high_slope = np.polyfit(range(30), high[-30:], 1)[0]
            low_slope = np.polyfit(range(30), low[-30:], 1)[0]
            
            if abs(high_slope - low_slope) / max(abs(high_slope), abs(low_slope), 0.001) < 0.2:
                if high_slope > 0 and low_slope > 0:
                    desc = "📈 Ascending Channel"
                    score = 1
                    patterns.append("Ascending Channel")
                    reliability = 0.6
                elif high_slope < 0 and low_slope < 0:
                    desc = "📉 Descending Channel"
                    score = -1
                    patterns.append("Descending Channel")
                    reliability = 0.6
        
        if not patterns:
            desc = "⚖️ No Chart Pattern Detected"
            score = 0
            reliability = 0
        
        if "Breakout" in desc or "BREAKOUT" in desc:
            reliability = min(0.9, reliability + 0.2)
        
        return {
            'patterns': patterns,
            'description': desc,
            'score': score,
            'reliability': reliability,
            'target': target
        }
    except Exception:
        return {'patterns': [], 'description': 'N/A', 'score': 0, 'reliability': 0, 'target': None}

# ======================== ELLIOTT WAVE ========================
def detect_elliot_wave_improved(df, lookback=100):
    if df is None or len(df) < lookback:
        return {'phase': 'Unknown', 'description': 'N/A', 'score': 0, 'wave_count': 0}
    
    try:
        rec = df.tail(lookback).copy()
        close = rec['close'].values
        high = rec['high'].values
        low = rec['low'].values
        
        swing_highs, swing_lows = find_swing_points(high, low, left_bars=3, right_bars=3, min_deviation=0.005)
        
        impulse = False
        corrective = False
        wave_count = 0
        score = 0
        desc = "N/A"
        
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            waves = []
            all_swings = sorted(swing_highs + swing_lows, key=lambda x: x[0])
            
            for i in range(len(all_swings) - 1):
                if all_swings[i][0] < all_swings[i+1][0]:
                    waves.append((all_swings[i][1], all_swings[i+1][1]))
            
            if len(waves) >= 5:
                wave_lengths = [abs(waves[i+1][1] - waves[i][1]) for i in range(len(waves)-1)]
                if len(wave_lengths) >= 3:
                    if wave_lengths[1] > wave_lengths[0] * 1.2 and wave_lengths[1] > wave_lengths[2] * 1.2:
                        impulse = True
                        wave_count = 5
                        score = 3
                        desc = "📈 Elliott Wave Impulse - Wave 3 Extension"
        
        if not impulse and len(swing_highs) >= 2 and len(swing_lows) >= 2:
            if (swing_lows[0][1] > swing_highs[0][1] > swing_lows[1][1] and
                swing_highs[1][1] > swing_lows[1][1]):
                corrective = True
                wave_count = 3
                score = -2
                desc = "📉 Elliott Wave Corrective - ABC Pattern"
        
        if impulse:
            if len(swing_highs) >= 5 and len(swing_lows) >= 4:
                if close[-1] > swing_highs[-1][1]:
                    desc = "📈 Wave 5 Extension - Potential Top"
                    score = 2
                elif close[-1] < swing_lows[-1][1]:
                    desc = "📈 Corrective Wave (A-B-C) Starting"
                    score = 1
        
        if not impulse and not corrective:
            range_high = max(high[-20:])
            range_low = min(low[-20:])
            range_pct = (range_high - range_low) / range_low
            if range_pct < 0.05:
                desc = "⚖️ Consolidation - Accumulation/Distribution"
                score = 0
            else:
                desc = "⚖️ Transition - Wait for Confirmation"
                score = 0
        
        return {
            'phase': 'Impulse' if impulse else ('Corrective' if corrective else 'Transition'),
            'description': desc,
            'wave_count': wave_count,
            'score': score,
            'direction': 'Bullish' if score > 0 else ('Bearish' if score < 0 else 'Neutral')
        }
    except Exception:
        return {'phase': 'Unknown', 'description': 'N/A', 'score': 0, 'wave_count': 0}

# ======================== FACTOR ENSEMBLE ========================
def compute_factor_ensemble_advanced(ind_df, df_weekly=None, df_1h=None, liq_data=None):
    if ind_df is None or len(ind_df) < 60:
        return {}
    last = ind_df.iloc[-1]
    cp = float(last['close'])
    factors = {}
    
    sa = safe_get(last.get('senkou_a', np.nan), cp)
    sb2 = safe_get(last.get('senkou_b', np.nan), cp)
    tk = safe_get(last.get('tenkan', np.nan), cp)
    kj = safe_get(last.get('kijun', np.nan), cp)
    ct = max(sa, sb2)
    sc = 0.0
    sc += 0.40 if cp > ct else (-0.40 if cp < min(sa, sb2) else 0)
    sc += 0.25 if tk > kj else -0.20
    fa = safe_get(last.get('future_senkou_a', np.nan), cp)
    fb = safe_get(last.get('future_senkou_b', np.nan), cp)
    sc += 0.20 if fa > fb else -0.15
    cloud_thick = abs(fa - fb) / cp
    sc += np.clip(cloud_thick * 5, -0.15, 0.15)
    factors['ichimoku'] = (np.clip(sc, -1, 1), 2.5, "Ichimoku")
    
    sd = safe_get(last.get('Supertrend_Dir', np.nan), 0)
    ss = 0.90 if sd == 1 else -0.90
    if len(ind_df) >= 2 and safe_get(ind_df['Supertrend_Dir'].iloc[-2], 0) != sd:
        ss = np.clip(ss * 1.3, -1, 1)
    factors['supertrend'] = (np.clip(ss, -1, 1), 2.5, "Supertrend")
    
    adx = safe_get(last.get('ADX', np.nan), 20)
    pdi = safe_get(last.get('Plus_DI', np.nan), 20)
    mdi = safe_get(last.get('Minus_DI', np.nan), 20)
    if adx > 30:
        as_ = min(1.0, adx / 50) * (1 if pdi > mdi else -1)
    elif adx > 20:
        as_ = 0.4 * (1 if pdi > mdi else -1)
    else:
        as_ = 0.1 * (1 if pdi > mdi else -1)
    if len(ind_df) >= 3:
        adx_trend = safe_get(ind_df['ADX'].iloc[-1] - ind_df['ADX'].iloc[-3], 0)
        as_ += np.clip(adx_trend / 30, -0.2, 0.2)
    factors['adx'] = (np.clip(as_, -1, 1), 2.0, f"ADX={adx:.1f}")
    
    ms = 0.0
    if 'MACD_Hist' in ind_df.columns:
        mh2 = ind_df['MACD_Hist'].tail(10).fillna(0)
        mh_l = float(mh2.iloc[-1])
        ms += 0.35 if mh_l > 0 else -0.35
        ms += np.clip(rolling_slope(mh2, 6) * 300, -0.40, 0.40)
        if len(mh2) >= 3:
            if mh2.iloc[-3] < 0 < mh_l:
                ms += 0.25
            elif mh2.iloc[-3] > 0 > mh_l:
                ms -= 0.25
        if 'MACD_Hist_Std' in ind_df.columns:
            mh_std = safe_get(last.get('MACD_Hist_Std', 0), 0)
            if mh_l > 0 and mh_std > 0:
                ms += 0.15
    factors['macd'] = (np.clip(ms, -1, 1), 2.0, "MACD")
    
    rsi_values = []
    rsi_scores = []
    for period in [14, 21, 50]:
        rsi = safe_get(last.get(f'RSI_{period}', np.nan), 50)
        rsi_values.append(rsi)
        if rsi < 25:
            rsi_scores.append(0.90)
        elif rsi < 35:
            rsi_scores.append(0.55)
        elif rsi < 45:
            rsi_scores.append(0.20)
        elif rsi < 55:
            rsi_scores.append(0.05)
        elif rsi < 65:
            rsi_scores.append(-0.05)
        elif rsi < 75:
            rsi_scores.append(-0.45)
        else:
            rsi_scores.append(-0.85)
    rsi = np.mean(rsi_values) if rsi_values else 50
    rs_sl = rolling_slope(ind_df['RSI'].fillna(50), 8)
    rsi_ma = safe_get(last.get('RSI_MA', np.nan), 50)
    rb = np.mean(rsi_scores) if rsi_scores else 0.05
    rm = np.clip(rs_sl / 5, -0.30, 0.30)
    rm += 0.10 if rsi > rsi_ma else -0.10
    factors['rsi'] = (np.clip(rb + rm, -1, 1), 2.0, f"RSI={rsi:.1f}")
    
    tsi = safe_get(last.get('TSI', np.nan), 0)
    tsig = safe_get(last.get('TSI_Signal', np.nan), 0)
    tsi_score = (0.50 if tsi > tsig else -0.50) + np.clip(tsi / 50, -0.50, 0.50)
    factors['tsi'] = (np.clip(tsi_score, -1, 1), 1.8, f"TSI={tsi:.1f}")
    
    kst = safe_get(last.get('KST', np.nan), 0)
    ksig = safe_get(last.get('KST_Signal', np.nan), 0)
    kst_score = (0.60 if kst > ksig else -0.60) + np.clip(kst / 100, -0.40, 0.40)
    factors['kst'] = (np.clip(kst_score, -1, 1), 1.5, f"KST={kst:.1f}")
    
    bbp = safe_get(last.get('BB_Pct', np.nan), 0.5)
    bbw = safe_get(last.get('BB_Width', np.nan), 0.05)
    abw = ind_df['BB_Width'].tail(30).mean() if 'BB_Width' in ind_df.columns else bbw
    bcr = bbw / max(abw, 1e-10)
    if bbp < 0.10:
        bbs = 0.75
    elif bbp < 0.25:
        bbs = 0.40
    elif bbp > 0.90:
        bbs = -0.75
    elif bbp > 0.75:
        bbs = -0.40
    else:
        bbs = (0.5 - bbp) * 0.5
    if bcr < 0.5:
        bbs += 0.15 if bbp > 0.50 else -0.10
    factors['bollinger'] = (np.clip(bbs, -1, 1), 1.5, f"BB%={bbp:.2f}")
    
    hma = safe_get(last.get('HMA', np.nan), cp)
    dema = safe_get(last.get('DEMA', np.nan), cp)
    hs = 0.0
    if not np.isnan(hma):
        hs += 0.45 if cp > hma else -0.45
        if len(ind_df) >= 3 and 'HMA' in ind_df.columns:
            hsl = float(ind_df['HMA'].iloc[-1]) - float(ind_df['HMA'].iloc[-3])
            hs += np.clip(hsl / (cp * 0.01 + 1e-10), -0.35, 0.35)
    da = 0.25 if (not np.isnan(dema) and cp > dema) else -0.25
    factors['hma_dema'] = (np.clip(hs + da * 0.5, -1, 1), 1.8, "HMA+DEMA")
    
    pb = int(safe_get(last.get('PSAR_Bull', np.nan), 0))
    ps = 0.80 if pb == 1 else -0.80
    if len(ind_df) >= 2 and int(safe_get(ind_df['PSAR_Bull'].iloc[-2], 0)) != pb:
        ps = np.clip(ps * 1.2, -1, 1)
    factors['psar'] = (ps, 1.8, "PSAR")
    
    cmf = safe_get(last.get('CMF', np.nan), 0)
    cs = np.clip(cmf * 5, -0.75, 0.75)
    if len(ind_df) >= 20:
        cmf_trend = safe_get(ind_df['CMF'].iloc[-1] - ind_df['CMF'].iloc[-5], 0)
        cs += np.clip(cmf_trend * 10, -0.25, 0.25)
    factors['cmf'] = (np.clip(cs, -1, 1), 1.5, f"CMF={cmf:.3f}")
    
    os_ = 0.0
    if 'OBV' in ind_df.columns and len(ind_df) >= 20:
        ov2 = ind_df['OBV'].tail(20).values
        pv2 = ind_df['close'].tail(20).values
        if not (np.isnan(ov2).any() or np.isnan(pv2).any()):
            osl = np.polyfit(range(20), ov2, 1)[0]
            psl = np.polyfit(range(20), pv2, 1)[0]
            no = osl / (np.abs(ov2).mean() + 1e-10)
            np2 = psl / (np.abs(pv2).mean() + 1e-10)
            if no > 0 and np2 >= 0:
                os_ = min(1.0, no * 20)
            elif no > 0:
                os_ = 0.70
            elif no < 0 and np2 > 0:
                os_ = -0.70
            else:
                os_ = max(-1.0, no * 20)
            if np2 < 0 and no > 0:
                os_ += 0.2
            elif np2 > 0 and no < 0:
                os_ -= 0.2
    factors['obv'] = (np.clip(os_, -1, 1), 2.0, "OBV")
    
    vd = 0.0
    if 'Vol_Delta_MA' in ind_df.columns:
        v_ = safe_get(last.get('Vol_Delta_MA', np.nan), 0)
        av = safe_get(ind_df['volume'].tail(20).mean(), 1)
        vd = np.clip(v_ / (av + 1e-10), -1, 1) * 0.8
        if len(ind_df) >= 5:
            delta_trend = safe_get(ind_df['Vol_Delta_MA'].iloc[-1] - ind_df['Vol_Delta_MA'].iloc[-5], 0)
            vd += np.clip(delta_trend / (av + 1e-10), -0.2, 0.2)
    factors['vol_delta'] = (np.clip(vd, -1, 1), 1.5, "Vol Delta")
    
    cvd_score = 0.0
    if 'CVD' in ind_df.columns and len(ind_df) >= 20:
        cvd_slope = rolling_slope(ind_df['CVD'].fillna(0), 10)
        cvd_score = np.clip(cvd_slope / 1000, -1, 1)
        if len(ind_df) >= 5:
            cvd_accel = safe_get(ind_df['CVD'].iloc[-1] - 2*ind_df['CVD'].iloc[-3] + ind_df['CVD'].iloc[-5], 0)
            cvd_score += np.clip(cvd_accel / 5000, -0.2, 0.2)
    factors['cvd'] = (np.clip(cvd_score, -1, 1), 1.5, "CVD")
    
    try:
        ca = ind_df['close'].tail(60).values.astype(float)
        _, vl, unc = kalman_smooth_adaptive(ca)
        kvs = np.clip(vl[-1] / (cp * 0.005 + 1e-10), -1, 1)
        uncertainty = unc[-1] / cp
        if uncertainty > 0.02:
            kvs *= 0.8
    except Exception:
        kvs = 0.0
    factors['kalman_velocity'] = (kvs, 2.0, "Kalman Velocity")
    
    try:
        h_series = ind_df['close'].tail(100).values
        H = hurst_exponent_fixed(h_series)
        roc5 = safe_get(last.get('ROC_5', np.nan), 0)
        if H > 0.6:
            h_s = 0.65 if roc5 > 0 else -0.65
        elif H < 0.4:
            h_s = -0.30 if roc5 > 0 else 0.30
        else:
            h_s = 0.0
        if H > 0.6 and roc5 > 0:
            h_s += 0.15
        elif H > 0.6 and roc5 < 0:
            h_s -= 0.15
    except Exception:
        H = 0.5
        h_s = 0.0
    factors['hurst'] = (np.clip(h_s, -1, 1), 1.8, f"Hurst={H:.3f}")
    
    mss = 0.0
    if len(ind_df) >= 50:
        hh2 = ind_df['high'].tail(50).values
        ll2 = ind_df['low'].tail(50).values
        sh2, sl2 = [], []
        for i in range(3, len(hh2) - 3):
            if hh2[i] == max(hh2[i-3:i+4]):
                sh2.append(hh2[i])
            if ll2[i] == min(ll2[i-3:i+4]):
                sl2.append(ll2[i])
        if len(sh2) >= 2:
            mss += 0.45 if sh2[-1] > sh2[-2] else -0.45
        if len(sl2) >= 2:
            mss += 0.45 if sl2[-1] > sl2[-2] else -0.45
        if len(sh2) >= 4 and len(sl2) >= 4:
            mss *= 1.2
    factors['market_structure'] = (np.clip(mss, -1, 1), 2.2, "Market Structure")
    
    ws = 0.0
    if df_weekly is not None and len(df_weekly) >= 20:
        if 'HMA' not in df_weekly.columns:
            df_weekly = calculate_indicators_upgraded(df_weekly)
        if df_weekly is not None:
            wl2 = df_weekly.iloc[-1]
            wm20 = df_weekly['close'].rolling(20).mean().iloc[-1]
            wc = float(wl2['close'])
            ws += 0.50 if wc > wm20 else -0.50
            if 'RSI' in df_weekly.columns:
                wr = safe_get(df_weekly['RSI'].iloc[-1], 50)
                ws += 0.30 if wr < 40 else (-0.30 if wr > 65 else 0)
            if len(df_weekly) >= 5:
                w_trend = safe_get(df_weekly['close'].iloc[-1] - df_weekly['close'].iloc[-5], 0)
                ws += np.clip(w_trend / wc, -0.2, 0.2)
    factors['weekly'] = (np.clip(ws, -1, 1), 2.0, "Weekly Bias")
    
    if df_1h is not None:
        sess_score, sess_desc = compute_session_bias(df_1h)
        factors['session'] = (np.clip(sess_score, -1, 1), 1.2, f"Session({sess_desc})")
    
    css = 0.0
    if 'is_bull' in ind_df.columns and 'body_ratio' in ind_df.columns:
        b5 = ind_df['is_bull'].tail(5).mean()
        br5 = ind_df['body_ratio'].tail(5).mean()
        css = (b5 - 0.5) * 0.70 + (br5 - 0.5) * 0.30
        if 'bull_streak' in ind_df.columns:
            streak = safe_get(ind_df['bull_streak'].iloc[-1], 0)
            css += np.clip(streak * 0.05, -0.2, 0.2)
    factors['candle'] = (np.clip(css, -1, 1), 1.0, "Candle Momentum")
    
    dc_h = safe_get(last.get('DC_High', np.nan), cp)
    dc_l = safe_get(last.get('DC_Low', np.nan), cp)
    dc_s = 0.0
    if cp >= dc_h * 0.998:
        dc_s = 0.75
    elif cp <= dc_l * 1.002:
        dc_s = -0.75
    factors['donchian'] = (np.clip(dc_s, -1, 1), 1.5, "Donchian")
    
    chart_patterns = detect_chart_patterns_improved(ind_df)
    cp_score = chart_patterns.get('score', 0) / 4
    factors['chart_pattern'] = (np.clip(cp_score, -1, 1), 1.5, "Chart Pattern")
    
    ew = detect_elliot_wave_improved(ind_df)
    ew_score = ew.get('score', 0) / 3
    factors['elliot_wave'] = (np.clip(ew_score, -1, 1), 1.5, "Elliott Wave")
    
    fib = calculate_fibonacci(ind_df)
    fib_score = fib.get('sentiment_score', 0)
    factors['fibonacci'] = (np.clip(fib_score, -1, 1), 1.2, "Fibonacci")
    
    if 'EWO' in ind_df.columns:
        ewo = safe_get(last.get('EWO', 0), 0)
        ewo_signal = safe_get(last.get('EWO_Signal', 0), 0)
        ewo_score = np.clip(ewo / max(abs(ewo_signal), 0.001), -1, 1)
        factors['ewo'] = (np.clip(ewo_score, -1, 1), 1.2, "EWO")
    
    bos_choch = detect_bos_choch(ind_df)
    bos_score = bos_choch.get('score', 0) / 3
    factors['bos_choch'] = (np.clip(bos_score, -1, 1), 2.0, "BOS/CHoCH")
    
    if 'VWAP' in ind_df.columns:
        vwap_dist = safe_get(ind_df['VWAP_Distance'].iloc[-1], 0)
        vwap_factor = np.clip(vwap_dist / 5, -1, 1)
        factors['vwap'] = (vwap_factor, 1.5, f"VWAP Dist: {vwap_dist:.2f}%")
    
    if 'ADL' in ind_df.columns and 'ADL_Slope' in ind_df.columns:
        adl_slope = safe_get(ind_df['ADL_Slope'].iloc[-1], 0)
        adl_factor = np.clip(adl_slope * 100, -1, 1)
        factors['adl'] = (adl_factor, 1.5, f"ADL Slope: {adl_slope:.2f}")
    
    return factors

def compute_adaptive_weights(factors, adx_val, hv20, vol_regime="normal", symbol=''):
    regime_factors = regime_switching_signal(factors, vol_regime, symbol)
    weights = {}
    for k, (score, w, label) in regime_factors.items():
        weights[k] = w
    if hv20 > 1.2:
        for k in ['obv', 'cmf', 'vol_delta', 'cvd']:
            if k in weights:
                weights[k] = min(weights[k] * 1.30, factors[k][1] * 1.5)
    if vol_regime in ['trending', 'trending_volatile']:
        for k in ['supertrend', 'adx', 'hurst', 'ichimoku', 'hma_dema', 'bos_choch']:
            if k in weights:
                weights[k] = min(weights[k] * 1.15, factors[k][1] * 1.3)
    if vol_regime in ['ranging', 'quiet']:
        for k in ['rsi', 'bollinger', 'tsi', 'kst']:
            if k in weights:
                weights[k] = min(weights[k] * 1.20, factors[k][1] * 1.4)
    return weights

def regime_switching_signal(signals_dict, regime, symbol=''):
    params = get_asset_params(symbol)
    regime_min = params['regime_min']
    regime_max = params['regime_max']
    regime_weights = {
        'trending': {'trend_following': 1.20, 'momentum': 1.15, 'oscillators': 0.70, 'volume': 1.10, 'structure': 1.15},
        'trending_volatile': {'trend_following': 1.15, 'momentum': 1.10, 'oscillators': 0.80, 'volume': 1.20, 'structure': 1.10},
        'ranging': {'trend_following': 0.60, 'momentum': 0.75, 'oscillators': 1.20, 'volume': 1.05, 'structure': 0.85},
        'squeeze': {'trend_following': 0.80, 'momentum': 1.20, 'oscillators': 0.90, 'volume': 1.20, 'structure': 1.15},
        'volatile': {'trend_following': 0.85, 'momentum': 0.80, 'oscillators': 1.00, 'volume': 1.20, 'structure': 1.05},
        'quiet': {'trend_following': 1.10, 'momentum': 0.85, 'oscillators': 1.15, 'volume': 0.90, 'structure': 1.05},
        'normal': {'trend_following': 1.0, 'momentum': 1.0, 'oscillators': 1.0, 'volume': 1.0, 'structure': 1.0},
    }
    if 'BTC' in symbol.upper() or 'ETH' in symbol.upper():
        btc_weights = {
            'trending': {'trend_following': 1.10, 'momentum': 1.05, 'oscillators': 0.85, 'volume': 1.05, 'structure': 1.10},
            'ranging': {'trend_following': 0.85, 'momentum': 0.90, 'oscillators': 1.10, 'volume': 1.10, 'structure': 0.95},
            'squeeze': {'trend_following': 0.90, 'momentum': 1.10, 'oscillators': 1.00, 'volume': 1.10, 'structure': 1.05},
        }
        if regime in btc_weights:
            regime_weights[regime] = btc_weights[regime]
    category_map = {
        'supertrend': 'trend_following', 'ichimoku': 'trend_following',
        'psar': 'trend_following', 'hma_dema': 'trend_following',
        'adx': 'trend_following', 'weekly': 'trend_following',
        'donchian': 'trend_following',
        'macd': 'momentum', 'tsi': 'momentum',
        'kst': 'momentum', 'kalman_velocity': 'momentum',
        'momentum': 'momentum',
        'rsi': 'oscillators', 'bollinger': 'oscillators',
        'obv': 'volume', 'cmf': 'volume',
        'vol_delta': 'volume', 'cvd': 'volume',
        'market_structure': 'structure', 'hurst': 'structure',
        'candle': 'structure', 'chart_pattern': 'structure',
        'elliot_wave': 'structure', 'fibonacci': 'structure',
        'bos_choch': 'structure',
        'vwap': 'volume',
        'adl': 'volume',
    }
    rw = regime_weights.get(regime, regime_weights['normal'])
    weighted = {}
    for k, (score, base_w, label) in signals_dict.items():
        cat = category_map.get(k, 'momentum')
        regime_mult = rw.get(cat, 1.0)
        regime_mult = float(np.clip(regime_mult, regime_min, regime_max))
        if abs(score) > 0.5 and regime in ['trending', 'trending_volatile']:
            regime_mult *= 1.1
        elif abs(score) < 0.2 and regime in ['ranging', 'quiet']:
            regime_mult *= 1.1
        weighted[k] = (score, base_w * regime_mult, label)
    return weighted

# ======================== MTF CONFLUENCE ========================
def compute_mtf_confluence(df_15m, df_1h, df_4h, df_1w):
    score = 0.0
    details = []
    weights = {'15m': 0.15, '1h': 0.25, '4h': 0.35, '1w': 0.25}
    timeframe_scores = {}
    
    def tf_score(df, label, weight):
        nonlocal score
        if df is None or len(df) < 20:
            timeframe_scores[label] = 0
            return
        try:
            last = df.iloc[-1]
            s = 0.0
            ema21 = safe_get(last.get('EMA21', np.nan), 0)
            c2 = float(last['close'])
            if c2 > ema21 > 0:
                s += 0.4
            elif c2 < ema21:
                s -= 0.4
            rsi = safe_get(last.get('RSI', np.nan), 50)
            if rsi < 40:
                s += 0.3
            elif rsi > 65:
                s -= 0.3
            mh = safe_get(last.get('MACD_Hist', np.nan), 0)
            s += 0.3 if mh > 0 else -0.3
            st_dir = safe_get(last.get('Supertrend_Dir', np.nan), 0)
            if st_dir == 1:
                s += 0.2
            elif st_dir == -1:
                s -= 0.2
            s = float(np.clip(s, -1, 1))
            timeframe_scores[label] = s
            score += s * weight
        except Exception:
            timeframe_scores[label] = 0
    
    tf_score(df_15m, "15m", weights['15m'])
    tf_score(df_1h, "1h", weights['1h'])
    tf_score(df_4h, "4h", weights['4h'])
    tf_score(df_1w, "1w", weights['1w'])
    
    scores = [timeframe_scores.get(tf, 0) for tf in ['15m', '1h', '4h', '1w']]
    agreement = sum(1 for s in scores if s > 0) / max(len(scores), 1)
    disagreement = sum(1 for s in scores if s < 0) / max(len(scores), 1)
    if agreement == 1:
        score += 0.2
    elif disagreement == 1:
        score -= 0.2
    score = float(np.clip(score, -1, 1))
    return score, " | ".join(details)

# ======================== SESSION BIAS ========================
def compute_session_bias(df_1h):
    if df_1h is None or len(df_1h) < 48:
        return 0.0, "N/A"
    try:
        df = df_1h.copy()
        df['hour_utc'] = df['timestamp'].dt.hour
        recent = df.tail(5 * 24)
        asia = recent[recent['hour_utc'].between(0, 6)]
        london = recent[recent['hour_utc'].between(7, 15)]
        ny = recent[recent['hour_utc'].between(13, 21)]
        bias = 0.0
        if len(asia) >= 3 and len(london) >= 3:
            asia_mid = (asia['high'].mean() + asia['low'].mean()) / 2
            london_cls = london['close'].mean()
            if london_cls > asia_mid:
                bias += 0.30
            else:
                bias -= 0.30
        if len(london) >= 3 and len(ny) >= 3:
            london_mid = (london['high'].mean() + london['low'].mean()) / 2
            ny_cls = ny['close'].mean()
            if ny_cls > london_mid:
                bias += 0.20
            else:
                bias -= 0.20
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

# ======================== LIQUIDITY ========================
def compute_liquidity_levels(df, lookback=100):
    if df is None or len(df) < 30:
        return {'bull_liq': [], 'bear_liq': [], 'order_blocks': [], 'score': 0.0}
    try:
        rec = df.tail(lookback).copy().reset_index(drop=True)
        cp = float(rec['close'].iloc[-1])
        highs = rec['high'].values
        lows = rec['low'].values
        bull_liq = []
        bear_liq = []
        tol = 0.003
        for i in range(5, len(rec) - 1):
            for j in range(i - 5, i):
                if j < 0:
                    continue
                if abs(highs[i] - highs[j]) / max(highs[j], 1e-10) < tol:
                    bear_liq.append(highs[i])
                if abs(lows[i] - lows[j]) / max(lows[j], 1e-10) < tol:
                    bull_liq.append(lows[i])
        order_blocks = []
        vma = rec['volume'].mean()
        for i in range(3, len(rec) - 1):
            vol_ok = rec['volume'].iloc[i] > vma * 1.5
            body = abs(rec['close'].iloc[i] - rec['open'].iloc[i])
            rng = rec['high'].iloc[i] - rec['low'].iloc[i]
            body_ok = body > rng * 0.6
            if vol_ok and body_ok:
                move = (rec['close'].iloc[i+1] - rec['close'].iloc[i]) / max(rec['close'].iloc[i], 1e-10)
                if abs(move) > 0.005:
                    order_blocks.append({'price': float(rec['close'].iloc[i]), 'type': 'bull' if rec['close'].iloc[i] > rec['open'].iloc[i] else 'bear', 'idx': i, 'strength': float(rec['volume'].iloc[i] / vma)})
        score = 0.0
        near_buy = [l2 for l2 in bull_liq if abs(l2 - cp) / max(cp, 1e-10) < 0.02 and l2 < cp]
        near_sell = [l2 for l2 in bear_liq if abs(l2 - cp) / max(cp, 1e-10) < 0.02 and l2 > cp]
        if near_buy:
            score += 0.40
        if near_sell:
            score -= 0.40
        nearest_ob = None
        for ob in order_blocks:
            if ob['price'] < cp and (nearest_ob is None or ob['price'] > nearest_ob['price']):
                nearest_ob = ob
        if nearest_ob:
            score += min(0.3, nearest_ob['strength'] * 0.1)
        return {'bull_liq': bull_liq[:5], 'bear_liq': bear_liq[:5], 'order_blocks': order_blocks[-5:], 'score': float(np.clip(score, -1, 1))}
    except Exception:
        return {'bull_liq': [], 'bear_liq': [], 'order_blocks': [], 'score': 0.0}

# ======================== CANDLE PROXY ========================
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
        cum_v = df['volume'].cumsum()
        vwap_intra = cum_tv / cum_v.replace(0, np.nan)
        vwap_val = float(vwap_intra.iloc[-1]) if not pd.isna(vwap_intra.iloc[-1]) else cp
        vwap_dev = (cp - vwap_val) / max(vwap_val, 1e-10)
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
        if len(df) >= 20:
            vol_high = df['volume'].tail(6).mean() / max(df['volume'].tail(20).mean(), 1e-10)
            if vol_high > 1.5:
                score += 0.15
                details['vol_spike'] = float(vol_high)
    except Exception:
        pass
    return float(np.clip(score, -1, 1)), details

# ======================== WIB TRADING DAY BUILDER ========================
def build_wib_trading_days_from_15m(df_15m, num_days=30):
    if df_15m is None or len(df_15m) < 5:
        return None
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
                'open': wib_daily['open'].astype(float),
                'high': wib_daily['high'].astype(float),
                'low': wib_daily['low'].astype(float),
                'close': wib_daily['close'].astype(float),
                'volume': wib_daily['volume'].astype(float),
            }).reset_index(drop=True)
            return wib_daily_df, 'wib_15m_aggregated'
    if df_d_raw is not None and len(df_d_raw) >= 30:
        return df_d_raw.copy(), 'exchange_1d_fallback'
    return None, 'unavailable'

# ======================== SMART MONEY SCORE ========================
def calculate_smart_money_score(df, liq_data, poc, val, vah, current_price):
    if df is None or len(df) < 20:
        return {'score': 0, 'level': 'TIDAK CUKUP DATA', 'reasons': [], 'is_accumulation_zone': False, 'cvd_trend': 0}
    cp = current_price
    score = 0
    reasons = []
    factor_scores = {}
    
    cvd_trend = 0
    if 'CVD' in df.columns and len(df) > 20:
        cvd_trend = df['CVD'].diff().tail(5).mean()
    
    order_blocks = liq_data.get('order_blocks', [])
    nearest_ob = None
    for ob in order_blocks:
        if ob['price'] < cp and (nearest_ob is None or ob['price'] > nearest_ob['price']):
            nearest_ob = ob
    
    if nearest_ob and cvd_trend > 0:
        score += 4
        reasons.append(f"🔵 Order block + CVD positif = ✅ AKUMULASI KUAT!")
        factor_scores['ob_cvd'] = 4
    elif nearest_ob and cvd_trend < 0:
        score -= 2
        reasons.append(f"🔴 Order block + CVD negatif = 🚨 DISTRIBUSI!")
        factor_scores['ob_cvd'] = -2
    elif nearest_ob:
        score += 1.5
        reasons.append(f"🔵 Order block terdekat")
        factor_scores['order_block'] = 1.5
    
    if poc and poc > 0 and cvd_trend > 0:
        dist_to_poc = abs(cp - poc) / cp * 100
        if dist_to_poc < 2:
            score += 3
            reasons.append(f"🟢 POC + CVD positif = ✅ AKUMULASI!")
            factor_scores['poc_cvd'] = 3
    elif poc and poc > 0:
        dist_to_poc = abs(cp - poc) / cp * 100
        if dist_to_poc < 1:
            score += 1
            reasons.append(f"🟢 Harga di POC")
            factor_scores['poc'] = 1
    
    if val and cp < val:
        discount_pct = (val - cp) / val * 100
        if cvd_trend > 0:
            score += 2
            reasons.append(f"💰 Discount zone + CVD positif = ✅ AKUMULASI!")
            factor_scores['discount_cvd'] = 2
        else:
            score -= 1
            reasons.append(f"⚠️ Discount zone tapi CVD negatif = 🚨 DISTRIBUSI PALSU!")
            factor_scores['discount_false'] = -1
    elif vah and cp > vah:
        if cvd_trend < 0:
            score += 1
            reasons.append(f"📈 Premium zone + CVD negatif = ✅ DISTRIBUSI VALID")
            factor_scores['premium_cvd'] = 1
        else:
            score -= 1
            reasons.append(f"⚠️ Premium zone tapi CVD positif = 🚨 BREAKOUT PALSU!")
            factor_scores['premium_false'] = -1
    
    if liq_data.get('score', 0) > 0.3:
        if cvd_trend > 0:
            score += 2
            reasons.append("🎯 Liquidity sweep + CVD positif = ✅ STOP HUNT BANDAR!")
            factor_scores['liq_sweep_bull'] = 2
        else:
            score -= 1
            reasons.append("🎯 Liquidity sweep tapi CVD negatif = ⚠️ STOP HUNT PALSU")
            factor_scores['liq_sweep_false'] = -1
    
    if 'OBV' in df.columns and 'close' in df.columns and len(df) > 20:
        obv_change = (df['OBV'].iloc[-1] - df['OBV'].iloc[-20]) / abs(df['OBV'].iloc[-20] + 1e-10)
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
        if obv_change > 0.05 and price_change < -0.02:
            if cvd_trend > 0:
                score += 2
                reasons.append("🐂 Bullish OBV divergence + CVD positif = ✅ BANDAR AKUMULASI!")
                factor_scores['obv_div'] = 2
            else:
                score += 0.5
                reasons.append("🐂 Bullish OBV divergence tapi CVD netral")
                factor_scores['obv_div_neutral'] = 0.5
        elif obv_change < -0.05 and price_change > 0.02:
            if cvd_trend < 0:
                score -= 2
                reasons.append("🐻 Bearish OBV divergence + CVD negatif = ✅ DISTRIBUSI!")
                factor_scores['obv_div_bear'] = -2
    
    if cvd_trend > 0:
        score += 1.5
        reasons.append("📈 CVD positif — konfirmasi akumulasi (bonus +1.5)")
        factor_scores['cvd'] = 1.5
    elif cvd_trend < 0:
        score -= 1.0
        reasons.append("📉 CVD negatif — konfirmasi distribusi (bonus -1.0)")
        factor_scores['cvd'] = -1.0
    
    if poc and vah and val:
        if cp < (val + poc) / 2 and cvd_trend > 0:
            score += 1
            reasons.append("📊 Harga di bawah POC + CVD positif = area diskon VALID")
            factor_scores['vp'] = 1
    
    score = max(0, min(10, score))
    
    if score >= 8:
        level = "🔥 SANGAT BAGUS (AKUMULASI VALID)"
        is_accumulation = True
    elif score >= 6:
        level = "✅ BAGUS (AKUMULASI)"
        is_accumulation = True
    elif score >= 4:
        level = "⚠️ CUKUP (KONFLIK)"
        is_accumulation = False
    elif score >= 2:
        level = "❌ LEMAH (DISTRIBUSI)"
        is_accumulation = False
    else:
        level = "💀 SANGAT LEMAH (DISTRIBUSI KUAT)"
        is_accumulation = False
    
    return {
        'score': score,
        'level': level,
        'reasons': reasons,
        'is_accumulation_zone': is_accumulation,
        'is_distribution_zone': score <= 2,
        'cvd_trend': round(cvd_trend, 2),
        'factor_scores': factor_scores
    }

# ======================== DEATH CAT BOUNCE DETECTION ========================
def detect_death_cat_bounce(df, symbol=''):
    if df is None or len(df) < 50:
        return {
            'is_death_cat': False,
            'score': 0,
            'signal': 'N/A',
            'reasons': [],
            'risk_level': 'LOW',
            'confidence': 0,
            'action': 'N/A'
        }
    
    try:
        last = df.iloc[-1]
        cp = float(last['close'])
        score = 0
        reasons = []
        confidence = 50
        
        ma20 = safe_get(last.get('MA20', cp), cp)
        ma50 = safe_get(last.get('MA50', cp), cp)
        ma200 = safe_get(last.get('MA200', cp), cp)
        
        is_below_ma20 = cp < ma20
        is_below_ma50 = cp < ma50
        is_below_ma200 = cp < ma200
        
        if is_below_ma20 and is_below_ma50 and is_below_ma200:
            score += 3
            reasons.append("📉 Harga di bawah semua MA - Downtrend konfirmasi")
            confidence += 10
        elif is_below_ma20 and is_below_ma50:
            score += 2
            reasons.append("📉 Harga di bawah MA20 & MA50 - Downtrend")
            confidence += 5
        
        if len(df) >= 7:
            change_7d = (df['close'].iloc[-1] - df['close'].iloc[-7]) / df['close'].iloc[-7] * 100
            if change_7d < -15:
                score += 3
                reasons.append(f"📉 Turun drastis {change_7d:.1f}% dalam 7 hari - Potensi bounce")
                confidence += 15
            elif change_7d < -10:
                score += 2
                reasons.append(f"📉 Turun {change_7d:.1f}% dalam 7 hari")
                confidence += 10
            elif change_7d < -5:
                score += 1
                reasons.append(f"📉 Turun {change_7d:.1f}% dalam 7 hari")
                confidence += 5
        
        vol_ma20 = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 1
        vol_ratio = df['volume'].iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1
        if vol_ratio > 2.5:
            score += 2
            reasons.append(f"🔥 Volume spike {vol_ratio:.1f}x - Bisa jadi pump palsu!")
            confidence += 10
        elif vol_ratio > 1.5:
            score += 1
            reasons.append(f"📊 Volume meningkat {vol_ratio:.1f}x")
            confidence += 5
        
        rsi = safe_get(last.get('RSI', 50), 50)
        rsi_5_ago = safe_get(df['RSI'].iloc[-6] if len(df) >= 6 else 50, 50)
        if rsi < 35 and rsi > rsi_5_ago:
            score += 2
            reasons.append(f"📈 RSI oversold bounce {rsi_5_ago:.0f} → {rsi:.0f}")
            confidence += 10
        elif rsi < 40:
            score += 1
            reasons.append(f"📊 RSI {rsi:.0f} - Mendekati oversold")
            confidence += 5
        
        macd_hist = safe_get(last.get('MACD_Hist', 0), 0)
        macd_hist_5_ago = safe_get(df['MACD_Hist'].iloc[-6] if len(df) >= 6 else 0, 0)
        if macd_hist > 0 and macd_hist_5_ago < 0 and rsi < 50:
            score += 2
            reasons.append("🔄 MACD crossover tapi RSI < 50 - Bisa jadi trap!")
            confidence += 10
        
        st_dir = safe_get(last.get('Supertrend_Dir', 0), 0)
        if st_dir == -1:
            score += 2
            reasons.append("📉 Supertrend still bearish - Bounce bisa palsu!")
            confidence += 10
        
        cvd_trend = 0
        if 'CVD' in df.columns and len(df) > 20:
            cvd_trend = df['CVD'].diff().tail(5).mean()
            if cvd_trend < 0 and vol_ratio > 2:
                score += 3
                reasons.append("⚠️ CVD negatif + Volume spike = Distribusi! DEATH CAT BOUNCE!")
                confidence += 20
            elif cvd_trend < 0:
                score += 1
                reasons.append("📉 CVD negatif - Smart money still distributing")
                confidence += 5
            elif cvd_trend > 0 and vol_ratio > 2:
                reasons.append("✅ CVD positif + Volume spike = Bisa jadi real reversal!")
                confidence += 10
                score -= 1
        
        if 'OBV' in df.columns and len(df) > 30:
            obv_change = (df['OBV'].iloc[-1] - df['OBV'].iloc[-10]) / abs(df['OBV'].iloc[-10] + 1e-10)
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
            if price_change > 0.05 and obv_change < -0.05:
                score += 3
                reasons.append("⚠️ Harga naik tapi OBV turun - Distribution! DEATH CAT!")
                confidence += 20
        
        sups, ress = calculate_precise_sr(df)
        if ress:
            nearest_res = ress[0]['price']
            dist_to_res = (nearest_res - cp) / cp * 100
            if dist_to_res < 3 and dist_to_res > 0:
                score += 2
                reasons.append(f"📊 Resistance dekat {fmt_price(nearest_res)} - Bisa reversal di sini!")
                confidence += 10
        
        wy_phase, wy_msg, wy_events = detect_wyckoff_improved(df)
        if "Distribution" in wy_phase or "UTAD" in str(wy_events):
            score += 3
            reasons.append(f"⚠️ Wyckoff Distribution detected - {wy_msg}")
            confidence += 15
        
        if len(df) >= 3:
            pump_24h = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
            if pump_24h > 5:
                reasons.append(f"🚀 Pump 24h +{pump_24h:.1f}% - Waspada DEATH CAT!")
                score += 2
                confidence += 10
        
        if rsi > 50:
            score -= 1
            reasons.append("📊 RSI > 50 - Bukan oversold bounce")
        
        score = max(0, min(10, score))
        
        if score >= 7:
            risk_level = "🔴 HIGH"
            signal = "💀 DEATH CAT BOUNCE DETECTED!"
            action = "❌ JANGAN BELI - HOLD / SELL"
            confidence = min(95, confidence + 10)
        elif score >= 5:
            risk_level = "🟠 MEDIUM"
            signal = "🐱 Potential Death Cat Bounce"
            action = "⚠️ HATI-HATI - Tunggu konfirmasi"
            confidence = min(85, confidence + 5)
        elif score >= 3:
            risk_level = "🟡 LOW-MEDIUM"
            signal = "🔍 OBSERVING - Mungkin bounce palsu"
            action = "⏳ OBSERVE - Jangan FOMO"
            confidence = min(70, confidence)
        else:
            risk_level = "🟢 LOW"
            signal = "✅ Aman - Bukan death cat"
            action = "✅ Bisa dipertimbangkan"
            confidence = min(60, confidence)
        
        return {
            'is_death_cat': score >= 5,
            'score': score,
            'signal': signal,
            'reasons': reasons[:5],
            'risk_level': risk_level,
            'confidence': min(95, confidence),
            'action': action,
            'rsi': round(rsi, 1),
            'vol_ratio': round(vol_ratio, 2),
            'cvd_trend': round(cvd_trend, 2) if 'CVD' in df.columns else 0,
            'ma_status': 'Bearish' if is_below_ma20 and is_below_ma50 else 'Mixed'
        }
    except Exception:
        return {
            'is_death_cat': False,
            'score': 0,
            'signal': 'Error',
            'reasons': ['Error in detection'],
            'risk_level': 'LOW',
            'confidence': 0,
            'action': 'N/A'
        }

# ======================== PUMP DETECTION ========================
def detect_pump_opportunity_upgraded(symbol, df_15m, df_1h, df_daily, ress, supports, advanced_data=None):
    if df_daily is None or len(df_daily) < 30:
        return {'score': 0, 'signal': '⏸️ NO SETUP', 'action': '❌ SKIP', 'reasons': ['Data tidak cukup'], 'confidence': 0}
    
    cp = df_daily['close'].iloc[-1]
    score = 0
    reasons = []
    factor_weights = {}
    
    if df_15m is not None and len(df_15m) > 12:
        recent = df_15m.tail(12)
        momentum = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / max(recent['close'].iloc[0], 1e-10)
        vol_ratio = recent['volume'].iloc[-1] / max(recent['volume'].mean(), 1e-10)
        if momentum > 0.02 and vol_ratio > 1.5:
            score += 3
            reasons.append(f"🚀 Intraday momentum +{momentum*100:.1f}%")
            factor_weights['intraday'] = 3
        elif momentum > 0.01 and vol_ratio > 1.2:
            score += 1
            reasons.append(f"📈 Intraday momentum +{momentum*100:.1f}%")
            factor_weights['intraday'] = 1
    
    if ress:
        nearest_res = ress[0]['price']
        near_resistance = (cp / max(nearest_res, 1e-10)) > 0.99
        vol_breakout = df_daily['volume'].iloc[-1] > df_daily['volume'].tail(20).mean() * 2
        is_bullish = df_daily['close'].iloc[-1] > df_daily['open'].iloc[-1]
        if near_resistance and vol_breakout and is_bullish:
            score += 3
            reasons.append(f"🚀 BREAKOUT IMMINENT! Resistance at {fmt_price(nearest_res)}")
            factor_weights['breakout'] = 3
        elif near_resistance:
            score += 1
            reasons.append(f"⚡ Near resistance {fmt_price(nearest_res)}")
            factor_weights['breakout'] = 1
    
    if len(df_daily) >= 25:
        vol_increase = df_daily['volume'].pct_change().tail(5)
        steady_increase = (vol_increase > 0).sum() >= 3
        price_range = (df_daily['high'].tail(5).max() - df_daily['low'].tail(5).min()) / max(cp, 1e-10)
        is_sideways = price_range < 0.03
        vol_high = df_daily['volume'].tail(5).mean() > df_daily['volume'].tail(20).mean() * 1.5
        if steady_increase and is_sideways and vol_high:
            score += 2
            reasons.append("📊 Pre-pump volume accumulation detected!")
            factor_weights['volume_acc'] = 2
        elif steady_increase and vol_high:
            score += 1
            reasons.append("📊 Volume increasing")
            factor_weights['volume_acc'] = 1
    
    if 'RSI' in df_daily.columns:
        rsi = df_daily['RSI'].iloc[-1]
        if rsi < 35:
            rsi_trend = df_daily['RSI'].diff().tail(3).mean()
            if rsi_trend > 0:
                score += 2
                reasons.append(f"📈 RSI reversal from {rsi:.0f}")
                factor_weights['rsi_reversal'] = 2
        elif rsi < 45:
            score += 1
            reasons.append(f"📈 RSI {rsi:.0f} (neutral-oversold)")
            factor_weights['rsi_reversal'] = 1
    
    divs = detect_divergences_improved(df_daily)
    if divs.get('rsi_bullish', False) or divs.get('macd_bullish', False):
        strength = divs.get('strength', 0.5)
        score += 2 + strength
        reasons.append(f"🔄 Bullish divergence (strength: {strength:.2f})")
        factor_weights['divergence'] = 2 + strength
    
    wy_phase, wy_msg, wy_events = detect_wyckoff_improved(df_daily)
    if "Spring" in wy_phase or "Spring" in str(wy_events):
        score += 3
        reasons.append(f"🌱 Wyckoff Spring: {wy_msg}")
        factor_weights['wyckoff'] = 3
    
    liq_data = compute_liquidity_levels(df_daily)
    poc, vah, val = calculate_volume_profile_improved(df_daily)
    smc = calculate_smart_money_score(df_daily, liq_data, poc, val, vah, cp)
    if smc.get('is_accumulation_zone', False):
        score += 2
        reasons.append("💰 Smart Money accumulation zone")
        factor_weights['smc'] = 2
    
    if supports:
        nearest_sup = supports[0]['price']
        near_support = (cp / max(nearest_sup, 1e-10)) < 1.01
        if near_support and df_daily['close'].iloc[-1] > df_daily['open'].iloc[-1]:
            score += 1
            reasons.append(f"🔄 Bounce from support {fmt_price(nearest_sup)}")
            factor_weights['support'] = 1
    
    bos_choch = detect_bos_choch(df_daily)
    if bos_choch.get('bos', False) and 'Bullish' in bos_choch.get('description', ''):
        score += 2
        reasons.append(f"📈 BOS Bullish: {bos_choch.get('description', '')}")
        factor_weights['bos'] = 2
    elif bos_choch.get('choch', False) and 'Bullish' in bos_choch.get('description', ''):
        score += 3
        reasons.append(f"🔄 CHoCH Bullish: {bos_choch.get('description', '')}")
        factor_weights['choch'] = 3
    
    sweep = detect_liquidity_sweep(df_daily)
    if sweep.get('sweep', False) and sweep.get('type') == 'bullish':
        score += 3
        reasons.append(f"🎯 {sweep.get('description', 'Liquidity Sweep')}")
        factor_weights['sweep'] = 3
    
    if advanced_data and advanced_data.get('ai_prediction'):
        ai_signal = advanced_data['ai_prediction'].get('signal', 0)
        if ai_signal > 0.2:
            score += 2
            reasons.append(f"🤖 AI Bullish: {ai_signal:+.2f}")
            factor_weights['ai'] = 2
        elif ai_signal < -0.2:
            score -= 1
            factor_weights['ai'] = -1
    
    chart_patterns = detect_chart_patterns_improved(df_daily)
    if 'Bullish' in chart_patterns.get('description', '') and chart_patterns.get('reliability', 0) > 0.5:
        score += 2
        reasons.append(f"📊 {chart_patterns['description']} (reliability: {chart_patterns['reliability']:.2f})")
        factor_weights['chart'] = 2
    elif 'Bearish' in chart_patterns.get('description', ''):
        score -= 1
        factor_weights['chart'] = -1
    
    if 'BB_Squeeze' in df_daily.columns:
        squeeze = bool(safe_get(df_daily['BB_Squeeze'].iloc[-1], False))
        if squeeze:
            score += 1
            reasons.append("📉 BB Squeeze - Potential breakout")
            factor_weights['squeeze'] = 1
    
    total_weight = sum(abs(v) for v in factor_weights.values()) or 1
    weighted_score = score * (1 + 0.2 * (len(factor_weights) / 12))
    score = min(10, int(weighted_score))
    active_factors = len([f for f in factor_weights.values() if abs(f) > 0])
    confidence_base = 40 + active_factors * 5
    
    if score >= 7 and active_factors >= 4:
        signal = "🚀 STRONG PUMP SETUP"
        action = "✅ READY TO BUY"
        confidence = min(95, confidence_base + 15)
    elif score >= 5 and active_factors >= 3:
        signal = "📈 POTENTIAL PUMP"
        action = "⚠️ WATCHLIST"
        confidence = min(85, confidence_base + 10)
    elif score >= 3 and active_factors >= 2:
        signal = "🔍 OBSERVING"
        action = "⏳ WAIT FOR CONFIRMATION"
        confidence = min(70, confidence_base + 5)
    else:
        signal = "⏸️ NO SETUP"
        action = "❌ SKIP"
        confidence = max(30, confidence_base - 10)
    
    return {
        'score': score,
        'signal': signal,
        'action': action,
        'reasons': reasons,
        'confidence': min(95, confidence),
        'active_factors': active_factors,
        'factor_weights': factor_weights
    }

# ======================== TRADE PLAN ========================
def calc_trade_plan(df, supports, resistances, poc, val, atr, cp, symbol=''):
    params = get_asset_params(symbol, cp)
    entry_max_pct = params['entry_max_pct']
    vol_adj = 1.0
    
    BUFFER_ENTRY = 0.005
    BUFFER_EXIT = 0.005
    BUFFER_SL = 0.005
    
    if 'Vol_Ratio' in df.columns:
        vol_ratio = safe_get(df['Vol_Ratio'].iloc[-1], 1.0)
        if vol_ratio > 1.5:
            vol_adj = 0.8
        elif vol_ratio < 0.7:
            vol_adj = 1.2
    
    if cp > 10000:
        max_entry_pct_below = entry_max_pct * vol_adj
    elif cp > 100:
        max_entry_pct_below = (entry_max_pct + 0.01) * vol_adj
    elif cp < 0.01:
        max_entry_pct_below = (entry_max_pct + 0.10) * vol_adj
    else:
        max_entry_pct_below = (entry_max_pct + 0.03) * vol_adj
    
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
        support_price = best_sup['price']
        
        conservative_entry = support_price * (1 + BUFFER_ENTRY)
        entry_range_bottom = support_price * (1 - BUFFER_ENTRY * 0.5)
        entry_range_top = support_price * (1 + BUFFER_ENTRY * 2)
        aggressive_entry = (entry_range_bottom + entry_range_top) / 2
        premium_entry = entry_range_bottom
        
    else:
        conservative_entry = cp - atr * 0.5 * vol_adj
        entry_range_bottom = conservative_entry * 0.99
        entry_range_top = conservative_entry * 1.01
        aggressive_entry = conservative_entry
        premium_entry = entry_range_bottom
        support_price = conservative_entry
    
    conservative_entry = max(conservative_entry, cp * (1 - max_entry_pct_below))
    conservative_entry = min(conservative_entry, cp * 0.999)
    
    aggressive_entry = max(aggressive_entry, conservative_entry * 1.001)
    aggressive_entry = min(aggressive_entry, cp * 0.999)
    
    premium_entry = max(premium_entry, cp * (1 - max_entry_pct_below * 1.2))
    premium_entry = min(premium_entry, conservative_entry * 0.998)
    
    if aggressive_entry <= conservative_entry:
        aggressive_entry = min(conservative_entry * 1.003, cp * 0.999)
    
    sups_below_entry = sorted(
        [s for s in supports if s['price'] < conservative_entry * 0.998],
        key=lambda x: x['price'], reverse=True
    )
    
    if sups_below_entry:
        sl_base = sups_below_entry[0]['price'] * (1 - BUFFER_SL)
    else:
        sl_base = conservative_entry - atr * 1.5 * vol_adj
    
    sl_candidate = min(sl_base, conservative_entry * 0.985)
    sl_min_allowed = conservative_entry * 0.85
    stop_loss = max(sl_candidate, sl_min_allowed)
    stop_loss = min(stop_loss, conservative_entry * 0.994)
    
    risk = conservative_entry - stop_loss
    if risk <= 0:
        risk = conservative_entry * 0.02 * vol_adj
        stop_loss = conservative_entry - risk
    
    if valid_ress:
        tp1_cands = [r for r in valid_ress if r['price'] > conservative_entry * 1.003]
        
        if tp1_cands:
            resistance_price = tp1_cands[0]['price']
            tp1 = resistance_price * (1 - BUFFER_EXIT)
            tp1_range_bottom = resistance_price * (1 - BUFFER_EXIT * 2)
            tp1_range_top = resistance_price * (1 - BUFFER_EXIT * 0.5)
        else:
            tp1 = conservative_entry + risk * 1.5
            tp1_range_bottom = tp1 * 0.99
            tp1_range_top = tp1 * 1.01
        
        if len(tp1_cands) >= 2:
            resistance_price_2 = tp1_cands[1]['price']
            tp2 = resistance_price_2 * (1 - BUFFER_EXIT)
        else:
            tp2 = tp1 + risk * 1.0
        
        if len(tp1_cands) >= 3:
            resistance_price_3 = tp1_cands[2]['price']
            tp3 = resistance_price_3 * (1 - BUFFER_EXIT)
        else:
            tp3 = tp2 + risk * 1.5
    else:
        tp1 = conservative_entry + risk * 1.5
        tp1_range_bottom = tp1 * 0.99
        tp1_range_top = tp1 * 1.01
        tp2 = conservative_entry + risk * 2.5
        tp3 = conservative_entry + risk * 4.0
    
    tp1 = max(tp1, conservative_entry + risk * 1.0)
    tp1_range_bottom = max(tp1_range_bottom, conservative_entry + risk * 0.8)
    tp1_range_top = max(tp1_range_top, tp1 * 0.98)
    
    tp2 = max(tp2, tp1 + risk * 0.5)
    tp3 = max(tp3, tp2 + risk * 0.5)
    
    def pct_from_entry(tp_price):
        return round((tp_price - conservative_entry) / max(conservative_entry, 1e-10) * 100, 2)
    
    def pct_from_current(tp_price):
        return round((tp_price - cp) / max(cp, 1e-10) * 100, 2)
    
    sl_pct = round((stop_loss - conservative_entry) / max(conservative_entry, 1e-10) * 100, 2)
    rr_ratio = round((tp1 - conservative_entry) / max(risk, 1e-10), 2)
    rr_ratio = min(rr_ratio, MAX_RR)
    
    return {
        'conservative_entry': round(conservative_entry, 10),
        'aggressive_entry': round(aggressive_entry, 10),
        'premium_entry': round(premium_entry, 10),
        'entry_range_bottom': round(entry_range_bottom, 10),
        'entry_range_top': round(entry_range_top, 10),
        'support_price': round(support_price, 10),
        'stop_loss': round(stop_loss, 10),
        'sl_pct': sl_pct,
        'tp1': round(tp1, 10),
        'tp2': round(tp2, 10),
        'tp3': round(tp3, 10),
        'tp1_range_bottom': round(tp1_range_bottom, 10),
        'tp1_range_top': round(tp1_range_top, 10),
        'tp1_pct_entry': pct_from_entry(tp1),
        'tp2_pct_entry': pct_from_entry(tp2),
        'tp3_pct_entry': pct_from_entry(tp3),
        'tp1_pct_current': pct_from_current(tp1),
        'tp2_pct_current': pct_from_current(tp2),
        'tp3_pct_current': pct_from_current(tp3),
        'rr': rr_ratio,
        'risk': round(risk, 10),
        'buffer_entry_pct': round(BUFFER_ENTRY * 100, 1),
        'buffer_exit_pct': round(BUFFER_EXIT * 100, 1),
    }

# ======================== POSITION SIZE ========================
def pos_size(entry, stop_loss, risk_pct=0.02, account_size=10000):
    risk_per_unit = entry - stop_loss
    if risk_per_unit <= 0:
        return 0.0
    dollar_risk = account_size * risk_pct
    units = dollar_risk / risk_per_unit
    position_value = units * entry
    pos_pct = (position_value / account_size) * 100
    max_pos_pct = MAX_POSITION_SIZE_PCT * 100
    return min(round(min(pos_pct, max_pos_pct), 1), 100)

# ======================== BULL SIGNALS ========================
def calculate_bull_signals(last, cp, st_bull, ps_b, tsi_v, kst_v, mb1d, obv_up, ab200, abvwap, cmf_v, hma_v, dema_v, rsi):
    momentum_bull = (tsi_v > 0 and kst_v > 0) or (tsi_v > 0 or kst_v > 0)
    
    bull_signals = sum([
        ab200,
        st_bull,
        ps_b,
        int(momentum_bull),
        mb1d,
        obv_up,
        abvwap,
        cmf_v > 0,
        cp > hma_v,
        cp > dema_v,
        rsi > 50
    ])
    return bull_signals

# ======================== SPOT TRADEABLE ========================
def is_spot_tradeable(bull_signals, result=None, symbol='', price=None):
    params = get_asset_params(symbol, price)
    min_signals = params['min_bull_signals']
    if result and result.get('vol_regime') == 'volatile':
        min_signals = max(3, min_signals - 1)
    if result and result.get('vol_regime') == 'trending':
        min_signals = max(3, min_signals - 1)
    if bull_signals < min_signals:
        return False, f"⚠️ Spot TIDAK LAYAK BELI: bull_signals={bull_signals}/{min_signals} (min {min_signals} diperlukan). Pasar bearish/netral — tunggu konfirmasi."
    if result is not None:
        st_bull = result.get('supertrend_bull', True)
        adx = result.get('adx', 20)
        if not st_bull and adx > 30:
            return False, f"⚠️ Spot TIDAK LAYAK BELI: Supertrend Bear + ADX={adx:.0f} kuat. Tren turun konfirmasi — hindari posisi long."
    return True, "✅ Setup spot valid"

def validate_trade_plan(trade, cp):
    warnings_list = []
    is_valid = True
    ce = trade['conservative_entry']
    ae = trade['aggressive_entry']
    sl = trade['stop_loss']
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
    elif cp < 0.01:
        max_entry_pct = 20.0
    else:
        max_entry_pct = 10.0
    entry_pct_below = (cp - ce) / max(cp, 1e-10) * 100
    if entry_pct_below > max_entry_pct:
        warnings_list.append(f"⚠️ Entry konservatif terlalu jauh: {entry_pct_below:.1f}% di bawah spot (max {max_entry_pct:.0f}% untuk aset ini)")
    return is_valid, warnings_list

def enhance_trade_plan_with_smc(original_trade, smc_score, fvg_nearest, current_price, atr):
    enhanced = original_trade.copy()
    enhanced['smc_score'] = smc_score['score']
    enhanced['smc_level'] = smc_score['level']
    enhanced['smc_reasons'] = smc_score['reasons']
    enhanced['is_accumulation_zone'] = smc_score['is_accumulation_zone']
    if fvg_nearest:
        enhanced['nearest_fvg'] = fvg_nearest
        enhanced['fvg_distance_pct'] = fvg_nearest.get('distance_pct', 999)
        enhanced['fvg_type'] = fvg_nearest.get('type', 'unknown')
    if smc_score['score'] >= 6 and original_trade.get('conservative_entry'):
        discount = original_trade['conservative_entry'] * 0.005
        enhanced['conservative_entry_smc'] = round(original_trade['conservative_entry'] - discount, 10)
        enhanced['smc_boost_applied'] = True
        enhanced['smc_boost_pct'] = -0.5
    else:
        enhanced['conservative_entry_smc'] = original_trade.get('conservative_entry')
        enhanced['smc_boost_applied'] = False
    return enhanced

# ======================== CROSS VALIDATION ========================
def cross_validate_signals(result):
    if result is None:
        return {
            'conflict_detected': False, 
            'score_adjustment': 0, 
            'message': 'N/A',
            'priority_signal': 'neutral',
            'action': 'HOLD',
            'conflicts': [],
            'bullish_weight': 0,
            'bearish_weight': 0,
            'bullish_pct': 0,
            'bearish_pct': 0,
            'dominant': 'neutral',
            'rr': 0,
            'rr_status': 'N/A'
        }
    
    cp = result.get('current_price', 0)
    smc = result.get('smc_score', {})
    chart = result.get('chart_patterns', {})
    candle_patterns = str(result.get('candle_patterns', []))
    divs = result.get('divs_1d', {})
    elliot = result.get('elliot_wave', {})
    bos_choch = result.get('bos_choch', {})
    trade = result.get('trade_plan', {})
    symbol = result.get('symbol', '')
    
    conflicts = []
    score_adjustment = 0
    priority_signal = 'neutral'
    action = '⏳ HOLD - Tunggu konfirmasi'
    
    bullish_weight = 0.0
    bearish_weight = 0.0
    total_weight = 0.0
    
    chart_desc = chart.get('description', '')
    is_hs_breakout = 'Head and Shoulders' in chart_desc and 'BREAKOUT' in chart_desc
    is_inv_hs_breakout = 'Inverse H&S' in chart_desc and 'BREAKOUT' in chart_desc
    
    if is_hs_breakout:
        weight = 7.0
        if 'BTC' in symbol.upper():
            weight = 8.0
        
        bearish_weight += weight
        total_weight += weight
        priority_signal = 'bearish'
        action = '❌ SKIP BUY — H&S Breakout Confirmed!'
        score_adjustment -= weight
        conflicts.append(f"🔴 H&S Breakout (Weight: {weight:.1f}) - Sinyal bearish TERKUAT!")
        
        if 'Bullish' in elliot.get('description', '') or 'Impulse' in elliot.get('description', ''):
            conflicts.append("⚠️ Elliott Wave bullish (Weight: 2.0) vs H&S bearish — H&S MENANG!")
            bullish_weight += 2.0
            total_weight += 2.0
            score_adjustment -= 1
        
        if smc.get('cvd_trend', 0) > 0:
            conflicts.append("⚠️ CVD positif (Weight: 1.5) tapi H&S bearish — kemungkinan bull trap!")
            bullish_weight += 1.5
            total_weight += 1.5
            score_adjustment -= 1
    
    elif is_inv_hs_breakout:
        weight = 7.0
        if 'BTC' in symbol.upper():
            weight = 8.0
        
        bullish_weight += weight
        total_weight += weight
        priority_signal = 'bullish'
        action = '✅ READY TO BUY — Inverse H&S Breakout!'
        score_adjustment += weight
        conflicts.append(f"🟢 Inverse H&S Breakout (Weight: {weight:.1f}) - Sinyal bullish TERKUAT!")
        
        if 'Bearish' in elliot.get('description', ''):
            bearish_weight += 1.5
            total_weight += 1.5
            conflicts.append("⚠️ Elliott bearish (Weight: 1.5) vs H&S bullish — H&S MENANG!")
    
    if 'Bear Flag' in chart_desc:
        bearish_weight += 4.0
        total_weight += 4.0
        if priority_signal == 'neutral':
            priority_signal = 'bearish'
            action = '⚠️ Bear Flag — continuation down'
        conflicts.append("📉 Bear Flag (Weight: 4.0) - Lanjutan turun")
        score_adjustment -= 2
    
    if 'Bull Flag' in chart_desc:
        bullish_weight += 4.0
        total_weight += 4.0
        if priority_signal == 'neutral':
            priority_signal = 'bullish'
            action = '📈 Bull Flag — continuation up'
        conflicts.append("📈 Bull Flag (Weight: 4.0) - Lanjutan naik")
        score_adjustment += 2
    
    if 'Descending Channel' in chart_desc:
        bearish_weight += 3.0
        total_weight += 3.0
        if priority_signal == 'neutral':
            priority_signal = 'bearish'
            action = '⚠️ WAIT — Descending Channel'
        conflicts.append("📉 Descending Channel (Weight: 3.0) - Trend turun")
        score_adjustment -= 2
    
    if 'Ascending Channel' in chart_desc:
        bullish_weight += 3.0
        total_weight += 3.0
        if priority_signal == 'neutral':
            priority_signal = 'bullish'
            action = '📈 Ascending Channel — trend naik'
        conflicts.append("📈 Ascending Channel (Weight: 3.0) - Trend naik")
        score_adjustment += 2
    
    if 'Double Bottom' in chart_desc and 'Breakout' in chart_desc:
        bullish_weight += 4.0
        total_weight += 4.0
        if priority_signal == 'neutral':
            priority_signal = 'bullish'
            action = '✅ Double Bottom Breakout — Buy!'
        conflicts.append("🟢 Double Bottom Breakout (Weight: 4.0)")
        score_adjustment += 3
    
    if 'Double Top' in chart_desc and 'Breakout' in chart_desc:
        bearish_weight += 4.0
        total_weight += 4.0
        if priority_signal == 'neutral':
            priority_signal = 'bearish'
            action = '❌ Double Top Breakout — Sell/Skip!'
        conflicts.append("🔴 Double Top Breakout (Weight: 4.0)")
        score_adjustment -= 3
    
    if 'Cup and Handle' in chart_desc:
        bullish_weight += 3.5
        total_weight += 3.5
        if priority_signal == 'neutral':
            priority_signal = 'bullish'
            action = '✅ Cup and Handle — Breakout Up'
        conflicts.append("🟢 Cup and Handle (Weight: 3.5)")
        score_adjustment += 3
    
    smc_score_val = smc.get('score', 0)
    if smc_score_val >= 7:
        bullish_weight += 3.5
        total_weight += 3.5
        conflicts.append(f"💰 SMC Score {smc_score_val}/10 (Weight: 3.5) - Strong accumulation")
        if priority_signal == 'neutral':
            priority_signal = 'bullish'
            action = '📈 SMC Strong — Accumulation'
        score_adjustment += 3
    elif smc_score_val >= 5:
        bullish_weight += 2.5
        total_weight += 2.5
        conflicts.append(f"💰 SMC Score {smc_score_val}/10 (Weight: 2.5) - Moderate accumulation")
        if priority_signal == 'neutral':
            priority_signal = 'bullish'
            action = '📈 SMC Moderate — Observing'
        score_adjustment += 2
    elif smc_score_val <= 2:
        bearish_weight += 2.5
        total_weight += 2.5
        conflicts.append(f"💰 SMC Score {smc_score_val}/10 (Weight: 2.5) - Distribution")
        if priority_signal == 'neutral':
            priority_signal = 'bearish'
            action = '⚠️ SMC Weak — Distribution'
        score_adjustment -= 2
    
    if smc.get('cvd_trend', 0) > 0:
        bullish_weight += 1.5
        total_weight += 1.5
        conflicts.append("📈 CVD Positif (Weight: 1.5) - Akumulasi berlangsung")
        score_adjustment += 1
    elif smc.get('cvd_trend', 0) < 0:
        bearish_weight += 1.5
        total_weight += 1.5
        conflicts.append("📉 CVD Negatif (Weight: 1.5) - Distribusi berlangsung")
        score_adjustment -= 1
    
    ai_signal = result.get('ai_prediction', {}).get('signal', 0)
    ai_confidence = result.get('ai_prediction', {}).get('confidence', 50) / 100
    
    if ai_signal > 0.2:
        weight = 2.5 * ai_confidence
        bullish_weight += weight
        total_weight += weight
        conflicts.append(f"🤖 AI Bullish (Weight: {weight:.1f}) - Signal: {ai_signal:+.3f}")
        if priority_signal == 'neutral':
            priority_signal = 'bullish'
            action = '📈 AI Bullish'
        score_adjustment += 2
    elif ai_signal < -0.2:
        weight = 2.5 * ai_confidence
        bearish_weight += weight
        total_weight += weight
        conflicts.append(f"🤖 AI Bearish (Weight: {weight:.1f}) - Signal: {ai_signal:+.3f}")
        if priority_signal == 'neutral':
            priority_signal = 'bearish'
            action = '📉 AI Bearish'
        score_adjustment -= 2
    
    if bos_choch.get('bos', False) and 'Bullish' in bos_choch.get('description', ''):
        bullish_weight += 2.5
        total_weight += 2.5
        conflicts.append("📈 BOS Bullish (Weight: 2.5) - Trend continuation")
        if priority_signal == 'neutral':
            priority_signal = 'bullish'
            action = '📈 BOS Bullish'
        score_adjustment += 2
    elif bos_choch.get('bos', False) and 'Bearish' in bos_choch.get('description', ''):
        bearish_weight += 2.5
        total_weight += 2.5
        conflicts.append("📉 BOS Bearish (Weight: 2.5) - Trend continuation")
        if priority_signal == 'neutral':
            priority_signal = 'bearish'
            action = '📉 BOS Bearish'
        score_adjustment -= 2
    elif bos_choch.get('choch', False) and 'Bullish' in bos_choch.get('description', ''):
        bullish_weight += 3.0
        total_weight += 3.0
        conflicts.append("🔄 CHoCH Bullish (Weight: 3.0) - Potential reversal!")
        if priority_signal == 'neutral':
            priority_signal = 'bullish'
            action = '🔄 CHoCH Bullish — Reversal up!'
        score_adjustment += 3
    elif bos_choch.get('choch', False) and 'Bearish' in bos_choch.get('description', ''):
        bearish_weight += 3.0
        total_weight += 3.0
        conflicts.append("🔄 CHoCH Bearish (Weight: 3.0) - Potential reversal!")
        if priority_signal == 'neutral':
            priority_signal = 'bearish'
            action = '🔄 CHoCH Bearish — Reversal down!'
        score_adjustment -= 3
    
    if 'Impulse' in elliot.get('description', '') and 'Wave 3' in elliot.get('description', ''):
        if priority_signal != 'bearish':
            bullish_weight += 2.0
            total_weight += 2.0
            conflicts.append("📈 Elliott Wave 3 Extension (Weight: 2.0) - Potensi bullish")
            if priority_signal == 'neutral':
                priority_signal = 'bullish'
                action = '📈 Elliott Wave 3 — Bullish'
            score_adjustment += 2
        else:
            conflicts.append("⚠️ Elliott Wave 3 bullish (Weight: 2.0) TAPI H&S bearish — H&S MENANG!")
    
    if divs.get('rsi_bullish', False) or divs.get('macd_bullish', False):
        strength = divs.get('strength', 0.5)
        weight = 1.5 + strength * 1.5
        bullish_weight += weight
        total_weight += weight
        conflicts.append(f"🔄 Bullish Divergence (Weight: {weight:.1f}) - Strength: {strength:.2f}")
        score_adjustment += 1
    
    if divs.get('rsi_bearish', False) or divs.get('macd_bearish', False):
        strength = divs.get('strength', 0.5)
        weight = 1.5 + strength * 1.5
        bearish_weight += weight
        total_weight += weight
        conflicts.append(f"🔄 Bearish Divergence (Weight: {weight:.1f}) - Strength: {strength:.2f}")
        score_adjustment -= 1
    
    bullish_patterns = ['Bullish Engulfing', 'Hammer', 'Morning Star', 'Bull Pin Bar', 'Piercing Line', 'Bullish Harami']
    bearish_patterns = ['Bearish Engulfing', 'Shooting Star', 'Evening Star', 'Bear Pin Bar', 'Dark Cloud Cover', 'Bearish Harami']
    
    has_bullish_candle = any(p in candle_patterns for p in bullish_patterns)
    has_bearish_candle = any(p in candle_patterns for p in bearish_patterns)
    
    if has_bullish_candle:
        bullish_weight += 1.0
        total_weight += 1.0
        conflicts.append("🕯️ Bullish Candlestick (Weight: 1.0)")
        score_adjustment += 1
    
    if has_bearish_candle:
        bearish_weight += 1.0
        total_weight += 1.0
        conflicts.append("🕯️ Bearish Candlestick (Weight: 1.0)")
        score_adjustment -= 1
    
    if total_weight > 0:
        bullish_pct = (bullish_weight / total_weight) * 100
        bearish_pct = (bearish_weight / total_weight) * 100
    else:
        bullish_pct = 0
        bearish_pct = 0
    
    if bullish_pct > bearish_pct and bullish_pct - bearish_pct > 10:
        dominant = 'bullish'
    elif bearish_pct > bullish_pct and bearish_pct - bullish_pct > 10:
        dominant = 'bearish'
    else:
        dominant = 'mixed'
    
    if divs.get('hidden_bullish', False) and divs.get('hidden_bearish', False):
        conflicts.append("⚠️ Hidden Bullish & Bearish Divergence (Market Indecision)")
        score_adjustment -= 1
    
    if divs.get('failed', False):
        conflicts.append("⚡ Failed Divergence - Signal tidak valid!")
        score_adjustment -= 2
    
    rr = trade.get('rr', 0) if isinstance(trade, dict) else 0
    
    if rr >= 2.0:
        rr_status = "🔥 EXCELLENT R/R"
    elif rr >= 1.5:
        rr_status = "✅ GOOD R/R"
    elif rr >= 1.0:
        rr_status = "⚠️ MINIMAL R/R"
    else:
        rr_status = "❌ POOR R/R"
    
    if priority_signal == 'neutral':
        if dominant == 'bullish':
            priority_signal = 'bullish'
            action = '📈 Bullish Dominant'
        elif dominant == 'bearish':
            priority_signal = 'bearish'
            action = '📉 Bearish Dominant'
        else:
            action = '⏳ MIXED — Tunggu konfirmasi'
    
    return {
        'conflict_detected': len(conflicts) > 0,
        'score_adjustment': score_adjustment,
        'conflicts': conflicts,
        'priority_signal': priority_signal,
        'action': action,
        'message': ' | '.join(conflicts) if conflicts else 'No conflicts detected',
        'bullish_weight': round(bullish_weight, 1),
        'bearish_weight': round(bearish_weight, 1),
        'bullish_pct': round(bullish_pct, 1),
        'bearish_pct': round(bearish_pct, 1),
        'dominant': dominant,
        'total_weight': round(total_weight, 1),
        'rr': round(rr, 2),
        'rr_status': rr_status
    }

# ======================== AI CONCLUSION ========================
def generate_ai_conclusion(result):
    if result is None or not isinstance(result, dict):
        return "Data tidak cukup untuk analisis."
    
    cv = result.get('cross_validation', {})
    priority_signal = cv.get('priority_signal', 'neutral')
    action = cv.get('action', 'HOLD')
    conflicts = cv.get('conflicts', [])
    
    bullish_pct = cv.get('bullish_pct', 0)
    bearish_pct = cv.get('bearish_pct', 0)
    dominant = cv.get('dominant', 'neutral')
    bullish_weight = cv.get('bullish_weight', 0)
    bearish_weight = cv.get('bearish_weight', 0)
    
    rr = cv.get('rr', 0)
    rr_status = cv.get('rr_status', 'N/A')
    
    price = result.get('current_price', 0)
    bull_signals = result.get('bull_signals', 0)
    smc_score = result.get('smc_score', {}).get('score', 0)
    pump_score = result.get('pump_analysis', {}).get('score', 0)
    confidence = result.get('confidence_score', 50)
    tradeable = result.get('spot_tradeable', False)
    pred_sum = result.get('pred_summary', {})
    chart = result.get('chart_patterns', {})
    death_cat = result.get('death_cat_bounce', {})
    bandar = result.get('bandar_signals', {})
    
    # NEW TECHNICALS
    pivot = result.get('pivot_points', {})
    harmonic = result.get('harmonic_patterns', {})
    atr_ts = result.get('atr_trailing_stop', {})
    ha = result.get('heikin_ashi', {})
    stoch = result.get('stoch_rsi', {})
    mfi = result.get('money_flow_index', {})
    aroon = result.get('aroon', {})
    cci = result.get('cci', {})
    ultimate = result.get('ultimate_oscillator', {})
    vortex = result.get('vortex', {})
    mass = result.get('mass_index', {})
    rvi = result.get('rvi', {})
    force = result.get('force_index', {})
    
    lines = []
    lines.append("=" * 50)
    lines.append("📋 **AI CONCLUSION (SPOT OPTIMIZED)**")
    lines.append("=" * 50)
    lines.append("")
    
    lines.append("### ⚖️ WEIGHTED VOTING RESULT:")
    lines.append(f"🟢 Bullish Weight: {bullish_weight:.1f} ({bullish_pct:.1f}%)")
    lines.append(f"🔴 Bearish Weight: {bearish_weight:.1f} ({bearish_pct:.1f}%)")
    
    if dominant == 'bullish':
        lines.append(f"📊 **Dominant: BULLISH** (Difference: {bullish_pct - bearish_pct:.1f}%)")
    elif dominant == 'bearish':
        lines.append(f"📊 **Dominant: BEARISH** (Difference: {bearish_pct - bullish_pct:.1f}%)")
    else:
        lines.append("📊 **Dominant: MIXED** (Difference < 10% - Market indecisive!)")
    
    lines.append("")
    lines.append(f"📊 **Risk/Reward:** {rr:.2f} — {rr_status}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # NEW TECHNICALS SUMMARY
    lines.append("### 📊 TEKNIKAL LENGKAP (56+ Indicators):")
    lines.append("")
    
    # Pivot Points
    if pivot.get('sentiment'):
        lines.append(f"📌 **Pivot Points:** {pivot.get('sentiment', 'N/A')}")
        classic = pivot.get('classic', {})
        if classic:
            lines.append(f"   - R1: {fmt_price(classic.get('r1'))} | R2: {fmt_price(classic.get('r2'))} | R3: {fmt_price(classic.get('r3'))}")
            lines.append(f"   - S1: {fmt_price(classic.get('s1'))} | S2: {fmt_price(classic.get('s2'))} | S3: {fmt_price(classic.get('s3'))}")
    
    # Harmonic Patterns
    if harmonic and harmonic.get('patterns'):
        for pattern in harmonic.get('patterns', []):
            lines.append(f"📐 **{pattern}**")
    
    # ATR Trailing Stop
    if atr_ts and atr_ts.get('current_stop'):
        lines.append(f"🎯 **ATR Trailing Stop:** {fmt_price(atr_ts.get('current_stop'))}")
        lines.append(f"   - Distance: {atr_ts.get('stop_distance_pct', 0)}%")
        lines.append(f"   - {atr_ts.get('recommendation', 'N/A')}")
    
    # Heikin Ashi
    if ha:
        lines.append(f"📈 **Heikin Ashi:** {ha.get('ha_trend', 'N/A')}")
        if ha.get('ha_reversal'):
            lines.append(f"   - 🔄 {ha.get('ha_signal', 'N/A')}")
        lines.append(f"   - Streak: {ha.get('ha_streak', 0)} candles")
    
    # Stochastic RSI
    if stoch:
        lines.append(f"📊 **Stochastic RSI:** K={stoch.get('k', 50):.1f} | D={stoch.get('d', 50):.1f}")
        lines.append(f"   - {stoch.get('signal', 'N/A')}")
        if stoch.get('cross_up'):
            lines.append(f"   - 🟢 {stoch.get('cross_signal', '')}")
        elif stoch.get('cross_down'):
            lines.append(f"   - 🔴 {stoch.get('cross_signal', '')}")
    
    # Money Flow Index
    if mfi:
        lines.append(f"💰 **MFI:** {mfi.get('mfi', 50):.1f} - {mfi.get('signal', 'N/A')}")
        if mfi.get('divergence') != 'None':
            lines.append(f"   - {mfi.get('divergence', '')}")
        if mfi.get('is_accumulation'):
            lines.append(f"   - ✅ Akumulasi terdeteksi!")
        elif mfi.get('is_distribution'):
            lines.append(f"   - ❌ Distribusi terdeteksi!")
    
    # Aroon
    if aroon:
        lines.append(f"📊 **Aroon:** Up={aroon.get('aroon_up', 50):.1f} | Down={aroon.get('aroon_down', 50):.1f}")
        lines.append(f"   - {aroon.get('signal', 'N/A')}")
        if aroon.get('cross'):
            lines.append(f"   - {aroon.get('cross_signal', 'N/A')}")
    
    # CCI
    if cci:
        lines.append(f"📊 **CCI:** {cci.get('cci', 0):.1f} - {cci.get('signal', 'N/A')}")
    
    # Ultimate Oscillator
    if ultimate:
        lines.append(f"📊 **Ultimate Oscillator:** {ultimate.get('ultimate', 50):.1f} - {ultimate.get('signal', 'N/A')}")
    
    # Vortex
    if vortex:
        lines.append(f"📊 **Vortex:** VI+={vortex.get('vi_plus', 1):.3f} | VI-={vortex.get('vi_minus', 1):.3f}")
        lines.append(f"   - {vortex.get('signal', 'N/A')}")
        if vortex.get('cross'):
            lines.append(f"   - {vortex.get('cross_signal', 'N/A')}")
    
    # Mass Index
    if mass:
        lines.append(f"📊 **Mass Index:** {mass.get('mass_index', 0):.1f} (Threshold: {mass.get('threshold', 26)})")
        lines.append(f"   - {mass.get('signal', 'N/A')}")
    
    # RVI
    if rvi:
        lines.append(f"📊 **RVI:** {rvi.get('rvi', 0):.1f} - {rvi.get('signal', 'N/A')}")
    
    # Force Index
    if force:
        lines.append(f"📊 **Force Index:** {force.get('force_index_norm', 0):.2f} - {force.get('signal', 'N/A')}")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Bandarmologi Signals
    if bandar:
        lines.append("### 🐋 BANDARMOLOGI SIGNALS:")
        if bandar.get('whale', {}).get('whale_detected', False):
            lines.append(f"🐋 **{bandar['whale'].get('status', 'N/A')}**")
            lines.append(f"📌 {bandar['whale'].get('action', 'N/A')}")
        if bandar.get('fake_breakout', {}).get('fake_breakout', False):
            lines.append(f"🎯 **{bandar['fake_breakout'].get('message', '')}**")
        reversal = bandar.get('reversal_candle', {})
        if reversal.get('pattern') != 'none':
            lines.append(f"📌 **{reversal.get('signal', 'N/A')}**")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Death Cat Bounce Warning
    if death_cat.get('is_death_cat', False):
        lines.append("")
        lines.append("⚠️⚠️⚠️ **DEATH CAT BOUNCE WARNING!** ⚠️⚠️⚠️")
        lines.append("")
        lines.append(f"💀 **Risk Level:** {death_cat.get('risk_level', 'N/A')}")
        lines.append(f"🐱 **Signal:** {death_cat.get('signal', 'N/A')}")
        lines.append(f"📌 **Action:** {death_cat.get('action', 'N/A')}")
        lines.append(f"🎯 **Score:** {death_cat.get('score', 0)}/10")
        lines.append("")
        lines.append("**⚠️ REASONS:**")
        for reason in death_cat.get('reasons', [])[:4]:
            lines.append(f"  - {reason}")
        lines.append("")
        lines.append("💡 **This could be a FAKE PUMP after a downtrend!**")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    if priority_signal == 'bearish':
        lines.append("🔴 **SINYAL DOMINAN: BEARISH**")
        lines.append(f"📌 **Action:** {action}")
        lines.append("")
        lines.append("⚠️ **ALASAN:**")
        for conflict in conflicts[:4]:
            lines.append(f"  - {conflict}")
        if not conflicts:
            lines.append("  - Sinyal bearish dominan berdasarkan weighted voting")
    elif priority_signal == 'bullish':
        lines.append("🟢 **SINYAL DOMINAN: BULLISH**")
        lines.append(f"📌 **Action:** {action}")
        lines.append("")
        lines.append("✅ **ALASAN:**")
        for conflict in conflicts[:4]:
            lines.append(f"  - {conflict}")
        if not conflicts:
            lines.append("  - Sinyal bullish dominan berdasarkan weighted voting")
    else:
        lines.append("⚪ **SINYAL: MIXED / NETRAL**")
        lines.append(f"📌 **Action:** {action}")
        lines.append("")
        if conflicts:
            lines.append("⚠️ **KONFLIK TERDETEKSI:**")
            for conflict in conflicts[:4]:
                lines.append(f"  - {conflict}")
        else:
            lines.append("ℹ️ Tidak ada sinyal dominan — tunggu konfirmasi.")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    lines.append("### 📊 RINGKASAN:")
    lines.append(f"- Bull Signals: {bull_signals}/12")
    lines.append(f"- SMC Score: {smc_score}/10")
    lines.append(f"- Pump Score: {pump_score}/10")
    lines.append(f"- Confidence: {confidence:.0f}%")
    lines.append(f"- R/R: {rr:.2f} ({rr_status})")
    lines.append(f"- Chart Pattern: {chart.get('description', 'N/A')}")
    lines.append(f"- Prediksi 7H: {pred_sum.get('overall', 'N/A')}")
    lines.append(f"- H+7 Change: {pred_sum.get('final_7d_change', 0):+.2f}%")
    
    if death_cat.get('is_death_cat', False):
        lines.append(f"- 🐱 Death Cat Score: {death_cat.get('score', 0)}/10")
        lines.append(f"- ⚠️ Risk: {death_cat.get('risk_level', 'N/A')}")
    
    if bandar.get('whale', {}).get('whale_detected', False):
        lines.append(f"- 🐋 Whale Score: {bandar['whale'].get('score', 0)}/10")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### 🎯 SPOT TRADING STRATEGY")
    
    conservative_entry = result.get('conservative_entry', price * 0.95)
    aggressive_entry = result.get('aggressive_entry', price * 0.97)
    premium_entry = result.get('premium_entry', price * 0.94)
    entry_range_bottom = result.get('entry_range_bottom', price * 0.94)
    entry_range_top = result.get('entry_range_top', price * 0.98)
    stop_loss = result.get('stop_loss', price * 0.92)
    tp1 = result.get('tp1', price * 1.05)
    tp2 = result.get('tp2', price * 1.10)
    tp3 = result.get('tp3', price * 1.15)
    tp1_range_bottom = result.get('tp1_range_bottom', tp1 * 0.98)
    tp1_range_top = result.get('tp1_range_top', tp1 * 1.02)
    buffer_entry_pct = result.get('buffer_entry_pct', 0.5)
    buffer_exit_pct = result.get('buffer_exit_pct', 0.5)
    
    pred_7d = pred_sum.get('final_7d_change', 0)
    
    if death_cat.get('is_death_cat', False) and death_cat.get('score', 0) >= 5:
        lines.append("")
        lines.append("💀 **DEATH CAT BOUNCE DETECTED!**")
        lines.append("")
        lines.append("📌 **STRATEGI:**")
        lines.append("")
        lines.append("1. ❌ **JANGAN BELI SEKARANG!**")
        lines.append("   - Ini adalah PUMP PALSU setelah downtrend")
        lines.append("   - Harga akan turun lagi setelah pump selesai")
        lines.append("")
        lines.append("2. 🚨 **IF YOU HAVE POSITION:**")
        lines.append("   - SELL / TAKE PROFIT SEKARANG!")
        lines.append("   - Jangan serakah, ambil untung sebelum turun")
        lines.append("")
        lines.append("3. ⏳ **TUNGGU HARGA TURUN LAGI**")
        lines.append("   - Setelah pump selesai, harga akan test support baru")
        lines.append("   - Baru beli setelah ada konfirmasi reversal yang valid")
        lines.append("")
        lines.append("4. 🎯 **DEATH CAT TARGETS:**")
        lines.append(f"   - Support Terdekat: {fmt_price(stop_loss * 1.02)}")
        lines.append(f"   - Jika breakdown, target: {fmt_price(stop_loss * 0.95)}")
        lines.append("")
        lines.append("⚠️ **JANGAN FOMO! Ini TRAP!**")
    
    elif priority_signal == 'bearish' or dominant == 'bearish':
        lines.append("")
        lines.append("🔴 **SITUASI: BEARISH DOMINAN**")
        lines.append("")
        lines.append("📌 **STRATEGI SPOT:**")
        lines.append("")
        lines.append("1. ❌ **JANGAN BUY SEKARANG**")
        lines.append("   - Sinyal bearish dominan, harga berpotensi turun")
        lines.append(f"   - Bearish Weight: {bearish_weight:.1f} vs Bullish: {bullish_weight:.1f}")
        lines.append("")
        
        if pred_7d > 0:
            lines.append("2. ⏳ **TUNGGU KOREKSI**")
            lines.append(f"   - Prediksi 7H: +{pred_7d:.2f}% (bullish jangka menengah)")
            lines.append(f"   - Tapi saat ini bearish, jadi TUNGGU HARGA TURUN DULU")
            lines.append(f"   - 🎯 **Target Entry:** {fmt_price(conservative_entry)}")
            lines.append(f"   - 📊 Entry Range: {fmt_price(entry_range_bottom)} - {fmt_price(entry_range_top)}")
            lines.append(f"   - 💡 Buffer {buffer_entry_pct:.1f}% di atas support")
            lines.append("")
            lines.append("3. ✅ **KONFIRMASI BELI:**")
            lines.append("   - Tunggu candle bullish (close > open)")
            lines.append("   - Atau tunggu breakout di atas resistance terdekat")
            lines.append(f"   - 🚨 Jangan FOMO beli di harga {fmt_price(price)}!")
        else:
            lines.append("2. ⏳ **TUNGGU SINYAL BULLISH**")
            lines.append("   - Prediksi 7H juga bearish, jadi TUNGGU")
            lines.append("   - Belum ada sinyal beli yang valid")
            lines.append("   - Masukkan watchlist, pantau setiap hari")
        
        lines.append("")
        lines.append("4. 🎯 **TARGET BEARISH (jika turun):**")
        if chart.get('target'):
            lines.append(f"   - Measured Move Target: {fmt_price(chart['target'])}")
            lines.append(f"   - 💡 Ini adalah area SUPPORT POTENSIAL untuk beli!")
        lines.append("")
        lines.append("5. 🔴 **STOP LOSS MENTAL:**")
        lines.append(f"   - Jika entry di {fmt_price(conservative_entry)}")
        lines.append(f"   - SL mental: {fmt_price(stop_loss)}")
        lines.append("   - Jangan panik sell, ini buat proteksi")
    
    elif priority_signal == 'bullish' and tradeable and not death_cat.get('is_death_cat', False):
        lines.append("")
        lines.append("🟢 **SITUASI: BULLISH DOMINAN**")
        lines.append("")
        lines.append("📌 **STRATEGI SPOT:**")
        lines.append("")
        lines.append(f"1. ✅ **SIAP BELI**")
        lines.append(f"   - Bullish Weight: {bullish_weight:.1f} vs Bearish: {bearish_weight:.1f}")
        lines.append(f"   - R/R: {rr:.2f} ({rr_status})")
        lines.append("")
        
        lines.append("2. 🎯 **ENTRY PLAN:**")
        lines.append(f"   - 🟢 Conservative Entry: {fmt_price(conservative_entry)}")
        lines.append(f"   - 🟡 Aggressive Entry: {fmt_price(aggressive_entry)}")
        lines.append(f"   - 🔵 Premium Entry: {fmt_price(premium_entry)}")
        lines.append(f"   - 📊 Entry Range: {fmt_price(entry_range_bottom)} - {fmt_price(entry_range_top)}")
        lines.append(f"   - 💡 Buffer {buffer_entry_pct:.1f}% di atas support")
        lines.append("")
        lines.append("3. 🎯 **TAKE PROFIT:**")
        lines.append(f"   - TP1: {fmt_price(tp1)} (+{result.get('tp1_pct_entry', 0):.2f}%)")
        lines.append(f"   - TP1 Range: {fmt_price(tp1_range_bottom)} - {fmt_price(tp1_range_top)}")
        lines.append(f"   - 💡 Buffer {buffer_exit_pct:.1f}% di bawah resistance")
        lines.append(f"   - TP2: {fmt_price(tp2)} (+{result.get('tp2_pct_entry', 0):.2f}%)")
        lines.append(f"   - TP3: {fmt_price(tp3)} (+{result.get('tp3_pct_entry', 0):.2f}%)")
        lines.append("")
        lines.append("4. 🔴 **STOP LOSS:**")
        lines.append(f"   - {fmt_price(stop_loss)} ({result.get('sl_pct', 0):.2f}%)")
        lines.append("   - 💡 Stop loss MENTAL untuk spot")
        lines.append("")
        lines.append("5. 📊 **POSITION SIZE:**")
        lines.append(f"   - {result.get('pos_size_pct', 0)}% dari akun (risk 2%)")
    
    else:
        lines.append("")
        lines.append("⚪ **SITUASI: MIXED / INDECISIVE**")
        lines.append("")
        lines.append("📌 **STRATEGI SPOT:**")
        lines.append("")
        lines.append("1. ⏳ **TUNGGU**")
        lines.append("   - Bullish vs Bearish weight hampir sama")
        lines.append(f"   - Bullish: {bullish_pct:.1f}% vs Bearish: {bearish_pct:.1f}%")
        lines.append("   - Market belum kasih arah jelas")
        lines.append("")
        lines.append("2. 🔍 **OBSERVE**")
        lines.append("   - Masukkan watchlist")
        lines.append("   - Pantau breakout di atas resistance atau breakdown di bawah support")
        lines.append("   - Baru entry setelah arah jelas")
        lines.append("")
        lines.append("3. ❌ **JANGAN FOMO**")
        lines.append("   - Jangan entry di kondisi mixed")
        lines.append("   - Resiko loss lebih tinggi")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    if death_cat.get('is_death_cat', False) and death_cat.get('score', 0) >= 5:
        lines.append("🎯 **FINAL:** 💀 **DEATH CAT BOUNCE - JANGAN BELI!**")
    elif priority_signal == 'bearish' or dominant == 'bearish':
        lines.append("🎯 **FINAL:** ❌ **SKIP BUY - TUNGGU KOREKSI**")
    elif priority_signal == 'bullish' and tradeable and not death_cat.get('is_death_cat', False):
        lines.append("🎯 **FINAL:** ✅ **READY TO BUY**")
    elif priority_signal == 'bullish' and not tradeable:
        lines.append("🎯 **FINAL:** ⏳ **WATCHLIST - BELUM TRADEABLE**")
    else:
        lines.append("🎯 **FINAL:** ⏳ **WAIT & OBSERVE**")
    
    lines.append("")
    lines.append("=" * 50)
    
    return "\n".join(lines)

# ======================== PREDICTIONS ========================
CONFIDENCE_CAPS = {0: 85, 1: 78, 2: 72, 3: 66, 4: 58, 5: 52, 6: 47, 7: 44}
CONFIDENCE_FLOORS = {0: 50, 1: 38, 2: 35, 3: 32, 4: 28, 5: 26, 6: 24, 7: 22}

def get_daily_limit_for_asset(symbol, price=None):
    symbol_upper = symbol.upper() if symbol else ""
    if price is not None and price < 0.01:
        return 0.80
    elif price is not None and price < 0.1:
        return 0.50
    if 'BTC' in symbol_upper:
        return 0.12
    elif 'ETH' in symbol_upper:
        return 0.15
    else:
        return 0.30

def _compute_sharp_regime_atr(atr, cp, hv20, gk_vol, day, vol_regime, adx_val, H_exp, symbol='', price=None):
    params = get_asset_params(symbol, price)
    regime_min = params['regime_min']
    regime_max = params['regime_max']
    base = atr / max(cp, 1e-10)
    hurst_mult = 1.10 if H_exp > 0.65 else (0.82 if H_exp < 0.40 else 1.0)
    vol_scale = np.clip(gk_vol / max(hv20, 1e-10), 0.6, 1.8)
    adx_mult = 1.0 + max(0, (adx_val - 20) / 120)
    day_decay = {0: 1.00, 1: 0.95, 2: 0.88, 3: 0.80}.get(day, max(0.55, 0.80 - (day-3)*0.05))
    regime_mult = {'trending': 1.15, 'trending_volatile': 1.28, 'volatile': 1.05,
                   'squeeze': 1.30, 'ranging': 0.72, 'quiet': 0.78, 'normal': 1.00}.get(vol_regime, 1.00)
    regime_mult = np.clip(regime_mult, regime_min, regime_max)
    return base * hurst_mult * vol_scale * adx_mult * day_decay * regime_mult

def _estimate_daily_range_sharp(atr, cp, hv20, gk_vol, day, vol_regime, adx_val, H_exp, symbol='', price=None):
    scaled = _compute_sharp_regime_atr(atr, cp, hv20, gk_vol, day, vol_regime, adx_val, H_exp, symbol, price)
    return scaled * 0.75

def predict_hlc_7d(daily_wib_df, df_weekly=None, h0_ultra=None,
                   df_1h=None, df_4h=None, df_15m=None,
                   trading_date=None, symbol=None):
    if daily_wib_df is None or len(daily_wib_df) < 30:
        return []
    if trading_date is None:
        trading_date = current_trading_date_str()
    if symbol is None:
        symbol = "UNKNOWN"
    
    cp = float(daily_wib_df['close'].iloc[-1])
    params = get_asset_params(symbol, cp)
    mc_band_mult = params['mc_band_mult']
    daily_limit = get_daily_limit_for_asset(symbol, cp)
    
    try:
        ind_df = calculate_indicators_upgraded(daily_wib_df)
        if ind_df is None:
            return []
        ind_df = calculate_ichimoku_crypto(ind_df)
        ind_df = calculate_obv_upgraded(ind_df)
        
        last = ind_df.iloc[-1]
        atr = safe_get(last.get('ATR', np.nan), cp * 0.025)
        atr7 = safe_get(last.get('ATR7', np.nan), atr)
        atr3 = safe_get(last.get('ATR3', np.nan), atr7)
        hv20 = safe_get(last.get('HV20', np.nan), 0.5)
        adx_val = safe_get(last.get('ADX', np.nan), 20)
        gk_vol = safe_get(last.get('GK_Vol', np.nan), hv20)
        yz_vol = safe_get(last.get('YZ_Vol', np.nan), hv20)
        
        actual_h0 = {'open': float(last['open']), 'high': float(last['high']),
                     'low': float(last['low']), 'close': float(last['close'])}
        
        liq_data = compute_liquidity_levels(ind_df)
        raw_factors = compute_factor_ensemble_advanced(ind_df, df_weekly, df_1h, liq_data)
        if not raw_factors:
            return []
        
        vol_regime, vol_mult = classify_volatility_regime_improved(ind_df)
        H_exp = hurst_exponent_fixed(ind_df['close'].tail(100).values)
        
        corr_matrix = compute_correlation_matrix(raw_factors)
        adj_weights = compute_ensemble_weights(raw_factors, corr_matrix)
        fn = list(raw_factors.keys())
        fs = np.array([raw_factors[k][0] for k in fn])
        fw = np.array([adj_weights.get(k, raw_factors[k][1]) for k in fn])
        tw = max(fw.sum(), 1e-10)
        ws = float(np.sum(fs * fw) / tw)
        agree_ratio = float(np.sum(np.sign(fs) == np.sign(ws))) / max(len(fs), 1)
        
        mtf_score, mtf_desc = compute_mtf_confluence(df_15m, df_1h, df_4h, df_weekly)
        close_arr = ind_df['close'].tail(60).values.astype(float)
        kal_smooth_vals, velocities, kal_unc = kalman_smooth_adaptive(close_arr)
        kal_vel_pct = float(velocities[-1]) / (cp + 1e-10)
        kal_uncertainty = float(kal_unc[-1]) / (cp + 1e-10)
        dom_period, phase_sig = compute_fourier_cycle(ind_df['close'].tail(100).values)
        ma50 = float(ind_df['MA50'].iloc[-1]) if 'MA50' in ind_df.columns and not pd.isna(ind_df['MA50'].iloc[-1]) else cp
        dev50 = (cp - ma50) / (ma50 + 1e-10)
        
        today_wib = wib_now()
        rows = []
        
        sim_open_h0 = actual_h0['open']
        sim_hi_h0 = actual_h0['high']
        sim_lo_h0 = actual_h0['low']
        sim_close_h0 = actual_h0['close']
        conf_h0 = 65.0
        daily_move_h0 = (sim_close_h0 - actual_h0['open']) / max(actual_h0['open'], 1e-10)
        direction_h0 = "⚖️ Sideways"
        if daily_move_h0 > 0.025: direction_h0 = "📈 Naik Kuat"
        elif daily_move_h0 > 0.010: direction_h0 = "📈 Naik"
        elif daily_move_h0 > 0.002: direction_h0 = "📈 Naik (Lemah)"
        elif daily_move_h0 < -0.025: direction_h0 = "📉 Turun Kuat"
        elif daily_move_h0 < -0.010: direction_h0 = "📉 Turun"
        elif daily_move_h0 < -0.002: direction_h0 = "📉 Turun (Lemah)"
        
        rows.append({
            'day': 0, 'date': today_wib.strftime('%d %b %Y'),
            'day_name_id': _day_name_id(today_wib.weekday()),
            'direction': direction_h0, 'confidence': conf_h0,
            'open': round(sim_open_h0, 10), 'high': round(sim_hi_h0, 10),
            'low': round(sim_lo_h0, 10), 'close': round(sim_close_h0, 10),
            'change_pct': 0.0, 'score': round(ws, 3),
            'reason': 'H+0 Actual', 'agreement_pct': round(agree_ratio * 100, 1),
            'regime': 'actual', 'is_today': True
        })
        
        sim_close = sim_close_h0
        mc_vol = float(yz_vol) if not np.isnan(yz_vol) else hv20
        
        chart_patterns = detect_chart_patterns_improved(ind_df)
        has_bear_flag = 'Bear Flag' in chart_patterns.get('description', '')
        
        for day in range(1, 8):
            pred_dt = today_wib + timedelta(days=day)
            day_name = _day_name_id(pred_dt.weekday())
            date_str = pred_dt.strftime('%d %b %Y')
            mc_seed = int(hashlib.md5(f"{symbol}_{trading_date}_{day}".encode()).hexdigest(), 16) % 1000000
            
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
                likelihood = float(np.clip(kal_contrib + fourier_contrib, -1, 1))
                bayesian_fused = bayesian_signal_update_adaptive(prior_signal, likelihood, vol_regime)
                eff_total = float(np.clip(bayesian_fused * 0.65 + mtf_contrib * 0.20 + fourier_contrib * 0.15, -1, 1))
            else:
                eff = ws * decay + mr_pull
                eff_total = float(np.clip(eff + kal_contrib * 0.3 + mtf_contrib + fourier_contrib * 0.5, -1, 1))
            
            if has_bear_flag and day <= 3:
                eff_total *= 0.7
            
            mc_ranges = monte_carlo_price_range_adaptive(
                sim_close, mc_vol, days=1, mu_estimate=eff_total * 0.3,
                n_sims=1500, seed=mc_seed, mc_band_mult=mc_band_mult
            )
            mc_low, mc_high, mc_med = mc_ranges['p10'], mc_ranges['p90'], mc_ranges['p50']
            
            regime_atr_pct = _compute_sharp_regime_atr(atr, cp, hv20, gk_vol, day, vol_regime, adx_val, H_exp, symbol, cp)
            daily_move = float(np.clip(eff_total * regime_atr_pct, -daily_limit, daily_limit))
            sim_close = sim_close * (1 + daily_move)
            chg_pct = (sim_close - cp) / cp * 100
            
            chg_pct = np.clip(chg_pct, -MAX_7D_CHANGE * 100, MAX_7D_CHANGE * 100)
            
            gap_pct = float(eff_total) * (atr3/max(cp,1e-10) if not np.isnan(atr3) else atr7/max(cp,1e-10)) * 0.15
            sim_open = (sim_close / (1 + daily_move)) * (1 + gap_pct)
            spread = _estimate_daily_range_sharp(atr, cp, hv20, gk_vol, day, vol_regime, adx_val, H_exp, symbol, cp)
            bias_h = 0.55 if eff_total > 0 else 0.45
            bias_l = 1.0 - bias_h
            mid = (sim_open + sim_close) / 2
            ta_high = mid + spread * cp * bias_h * (1 + abs(eff_total) * 0.4)
            ta_low = mid - spread * cp * bias_l * (1 + abs(eff_total) * 0.4)
            
            if day <= 3:
                mc_w = 0.40 if day == 1 else (0.30 if day == 2 else 0.20)
                sim_hi = ta_high * (1 - mc_w) + mc_high * mc_w
                sim_lo = ta_low * (1 - mc_w) + mc_low * mc_w
            else:
                sim_hi = ta_high
                sim_lo = ta_low
            sim_hi = max(sim_hi, max(sim_open, sim_close))
            sim_lo = min(sim_lo, min(sim_open, sim_close))
            
            daily_change_limit = daily_limit * 1.5
            max_hi = sim_close * (1 + daily_change_limit)
            min_lo = sim_close * (1 - daily_change_limit)
            sim_hi = min(sim_hi, max_hi)
            sim_lo = max(sim_lo, min_lo)
            
            base_conf = 42 + agree_ratio * 22 + abs(eff_total) * 10
            if vol_regime == "trending" and adx_val > 25:
                base_conf += 4
            if agree_ratio > 0.75:
                base_conf += 3
            elif agree_ratio < 0.40:
                base_conf -= 5
            if H_exp > 0.65:
                base_conf += 3
            elif H_exp < 0.40:
                base_conf -= 3
            if day <= 3:
                kal_bonus = min(3, max(0, (1 - kal_uncertainty * 20)) * 3)
                base_conf += kal_bonus
            cap = CONFIDENCE_CAPS.get(day, 40)
            floor = CONFIDENCE_FLOORS.get(day, 22)
            conf = max(floor, min(cap, base_conf))
            
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
            if direction == "⚖️ Sideways" and abs(chg_pct) > 1.0:
                direction = "📈 Naik (Lemah)" if chg_pct > 0 else "📉 Turun (Lemah)"
            
            rows.append({
                'day': day, 'date': date_str, 'day_name_id': day_name,
                'direction': direction, 'confidence': round(conf, 1),
                'open': round(sim_open, 10), 'high': round(sim_hi, 10),
                'low': round(sim_lo, 10), 'close': round(sim_close, 10),
                'change_pct': round(chg_pct, 2), 'score': round(eff_total, 3),
                'reason': f'Kal: {kal_contrib:+.2f} | Fourier: {fourier_contrib:+.2f}',
                'agreement_pct': round(agree_ratio * 100, 1), 'regime': vol_regime,
                'is_today': False, 'mc_p50': round(mc_med, 10),
                'mc_p10': round(mc_low, 10), 'mc_p90': round(mc_high, 10)
            })
        return rows
    except Exception:
        return []

def predict_hlc_7d_simple(daily_wib_df, symbol=None):
    if daily_wib_df is None or len(daily_wib_df) < 30:
        return []
    try:
        ind_df = calculate_indicators_upgraded(daily_wib_df)
        if ind_df is None:
            return []
        last = ind_df.iloc[-1]
        cp = float(last['close'])
        atr = safe_get(last.get('ATR', np.nan), cp * 0.02)
        ma20 = safe_get(last.get('MA20', cp), cp)
        trend = 1 if cp > ma20 else -1
        daily_limit = get_daily_limit_for_asset(symbol, cp)
        
        rows = []
        today_wib = wib_now()
        rows.append({
            'day': 0, 'date': today_wib.strftime('%d %b %Y'),
            'day_name_id': _day_name_id(today_wib.weekday()),
            'direction': '⚖️ Sideways', 'confidence': 65.0,
            'open': round(float(last['open']), 10), 'high': round(float(last['high']), 10),
            'low': round(float(last['low']), 10), 'close': round(float(last['close']), 10),
            'change_pct': 0.0, 'score': 0.0, 'reason': 'H+0 Aktual',
            'agreement_pct': 60, 'regime': 'actual', 'is_today': True
        })
        sim_price = cp
        for day in range(1, 8):
            trend_strength = trend * max(0, 1 - day * 0.1)
            random_move = np.random.normal(0, atr / cp * 0.3)
            daily_change = trend_strength * 0.005 + random_move * 0.5
            daily_change = np.clip(daily_change, -daily_limit, daily_limit)
            sim_price = sim_price * (1 + daily_change)
            chg_pct = (sim_price - cp) / cp * 100
            chg_pct = np.clip(chg_pct, -MAX_7D_CHANGE * 100, MAX_7D_CHANGE * 100)
            sim_open = sim_price / (1 + daily_change)
            sim_high = sim_price * (1 + abs(daily_change) * 0.6)
            sim_low = sim_price * (1 - abs(daily_change) * 0.6)
            pred_dt = today_wib + timedelta(days=day)
            if chg_pct > 2.5:
                direction = "📈 Naik Kuat"
            elif chg_pct > 1.0:
                direction = "📈 Naik"
            elif chg_pct > 0.3:
                direction = "📈 Naik (Lemah)"
            elif chg_pct < -2.5:
                direction = "📉 Turun Kuat"
            elif chg_pct < -1.0:
                direction = "📉 Turun"
            elif chg_pct < -0.3:
                direction = "📉 Turun (Lemah)"
            else:
                direction = "⚖️ Sideways"
            rows.append({
                'day': day, 'date': pred_dt.strftime('%d %b %Y'),
                'day_name_id': _day_name_id(pred_dt.weekday()),
                'direction': direction, 'confidence': max(45, 65 - day * 2),
                'open': round(sim_open, 10), 'high': round(sim_high, 10),
                'low': round(sim_low, 10), 'close': round(sim_price, 10),
                'change_pct': round(chg_pct, 2), 'score': round(daily_change * 100, 3),
                'reason': 'Fallback', 'agreement_pct': 55,
                'regime': 'normal', 'is_today': False
            })
        return rows
    except Exception:
        return []

def get_prediction_summary(predictions):
    if not predictions:
        return {}
    pf = [p for p in predictions if p['day'] > 0]
    if not pf:
        return {}
    bull_days = sum(1 for p in pf if 'Naik' in p['direction'])
    bear_days = sum(1 for p in pf if 'Turun' in p['direction'])
    avg_conf = np.mean([p['confidence'] for p in pf])
    max_chg = max(p['change_pct'] for p in pf)
    min_chg = min(p['change_pct'] for p in pf)
    final_chg = pf[-1]['change_pct']
    sharp_3 = [p for p in pf if p['day'] <= 3]
    far_4_7 = [p for p in pf if p['day'] > 3]
    s3_bull = sum(1 for p in sharp_3 if 'Naik' in p['direction'])
    f47_bull = sum(1 for p in far_4_7 if 'Naik' in p['direction']) if far_4_7 else 0
    best = max(pf, key=lambda p: p['change_pct'])
    worst = min(pf, key=lambda p: p['change_pct'])
    if bull_days >= 5:
        overall = "🟢 Sangat Bullish"
    elif bull_days >= 4:
        overall = "🟡 Bullish Dominan"
    elif bear_days >= 5:
        overall = "🔴 Sangat Bearish"
    elif bear_days >= 4:
        overall = "🟠 Bearish Dominan"
    else:
        overall = "⚪ Mixed/Sideways"
    return {
        'overall': overall, 'bull_days': bull_days, 'bear_days': bear_days,
        'side_days': len(pf)-bull_days-bear_days,
        'avg_confidence': round(avg_conf, 1),
        'max_upside': round(max_chg, 2), 'max_downside': round(min_chg, 2),
        'final_7d_change': round(final_chg, 2),
        'sharp_3_bull': s3_bull, 'far_4_7_bull': f47_bull,
        'sharp_bias': "Bullish" if s3_bull >= 2 else ("Bearish" if s3_bull <= 1 else "Mixed"),
        'far_bias': "Bullish" if f47_bull >= 3 else ("Bearish" if f47_bull <= 1 else "Mixed"),
        'best_day': best, 'worst_day': worst
    }

# ======================== ACCUMULATION DETECTION ========================
def detect_accumulation_zone(df, symbol=''):
    if df is None or len(df) < 30:
        return {'is_accumulating': False, 'score': 0, 'reasons': [], 'status': 'NO DATA', 'vol_ratio': 0, 'price_range': 0}
    
    try:
        last = df.iloc[-1]
        cp = float(last['close'])
        reasons = []
        score = 0
        
        cvd_trend = 0
        if 'CVD' in df.columns and len(df) > 20:
            cvd_trend = df['CVD'].diff().tail(5).mean()
        
        if cvd_trend < 0:
            reasons.append("⚠️ CVD negatif — bandar DISTRIBUSI, BUKAN akumulasi!")
            score -= 2
        
        vol_ma20 = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 1
        vol_ratio = df['volume'].iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1
        price_range = (df['high'].tail(10).max() - df['low'].tail(10).min()) / cp if cp > 0 else 1
        
        if cvd_trend >= 0:
            if vol_ratio > 1.5 and price_range < 0.03:
                score += 3
                reasons.append(f"📊 Stealth Acc: Vol {vol_ratio:.1f}x, Range {price_range*100:.2f}%")
            elif vol_ratio > 1.2 and price_range < 0.05:
                score += 2
                reasons.append(f"📊 Vol meningkat {vol_ratio:.1f}x, harga sideways")
        
        if len(df) > 20:
            recent_lows = df['low'].tail(20).min()
            if df['low'].iloc[-1] < recent_lows and df['close'].iloc[-1] > df['low'].iloc[-1]:
                if cvd_trend > 0:
                    score += 3
                    reasons.append("🎯 Liquidity Sweep + CVD positif = ✅ STOP HUNT BANDAR (BULLISH)!")
                else:
                    score += 1
                    reasons.append("🎯 Liquidity Sweep - Stop hunt bandar!")
        
        if 'OBV' in df.columns and len(df) > 30:
            obv_change = (df['OBV'].iloc[-1] - df['OBV'].iloc[-20]) / abs(df['OBV'].iloc[-20] + 1e-10)
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            if obv_change > 0.05 and price_change < -0.01:
                if cvd_trend > 0:
                    score += 3
                    reasons.append("🐂 Bullish OBV divergence + CVD positif = ✅ BANDAR AKUMULASI!")
                else:
                    score += 1
                    reasons.append("🐂 Bullish OBV divergence (CVD netral)")
        
        if cvd_trend > 0:
            score += 0.5
            reasons.append("📈 CVD positif — konfirmasi akumulasi")
        
        try:
            liq_data = compute_liquidity_levels(df)
            poc, vah, val = calculate_volume_profile_improved(df)
            smc = calculate_smart_money_score(df, liq_data, poc, val, vah, cp)
            if smc.get('is_accumulation_zone', False):
                if cvd_trend > 0:
                    score += 3
                    reasons.append(f"💰 SMC Acc zone + CVD positif = ✅ VALID!")
                else:
                    score += 1
                    reasons.append(f"💰 SMC Acc zone (Score: {smc.get('score', 0)}/10)")
        except:
            pass
        
        try:
            wy_phase, wy_msg, wy_events = detect_wyckoff_improved(df)
            if "Spring" in wy_phase or "Spring" in str(wy_events):
                if cvd_trend > 0:
                    score += 3
                    reasons.append(f"🌱 Wyckoff Spring + CVD positif = ✅ VALID!")
                else:
                    reasons.append(f"⚠️ Wyckoff Spring tapi CVD negatif = ❌ PALSU!")
        except:
            pass
        
        if 'BB_Squeeze' in df.columns:
            if bool(safe_get(df['BB_Squeeze'].iloc[-1], False)):
                score += 1
                reasons.append("📉 BB Squeeze - Potensi breakout")
        
        score = min(10, max(0, score))
        
        if cvd_trend < 0:
            status = "❌ DISTRIBUSI (CVD NEGATIF) — HATI-HATI!"
        elif score >= 7:
            status = "🔥 STRONG ACCUMULATION (VALID)"
        elif score >= 5:
            status = "✅ ACCUMULATING"
        elif score >= 3:
            status = "🔍 OBSERVING"
        else:
            status = "❌ NO ACCUMULATION"
        
        return {
            'is_accumulating': score >= 5 and cvd_trend >= 0,
            'score': score,
            'status': status,
            'reasons': reasons[:5],
            'vol_ratio': round(vol_ratio, 2),
            'price_range': round(price_range * 100, 2),
            'cvd_trend': round(cvd_trend, 2)
        }
    except Exception:
        return {'is_accumulating': False, 'score': 0, 'reasons': [], 'status': 'ERROR', 'vol_ratio': 0, 'price_range': 0}

# ======================== MTF ACCUMULATION ========================
def detect_accumulation_mtf(df_15m, df_1h, df_4h, symbol=''):
    result = {
        'is_accumulating': False,
        'score': 0,
        'tf_15m': {'score': 0, 'signal': 'N/A'},
        'tf_1h': {'score': 0, 'signal': 'N/A'},
        'tf_4h': {'score': 0, 'signal': 'N/A'},
        'reasons': [],
        'estimated_pump_time': 'N/A',
        'status': 'N/A'
    }
    
    if df_15m is not None and len(df_15m) > 50:
        tf_score = 0
        tf_reasons = []
        vol_ma20 = df_15m['volume'].rolling(20).mean().iloc[-1]
        vol_ratio = df_15m['volume'].iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1
        price_range = (df_15m['high'].tail(10).max() - df_15m['low'].tail(10).min()) / df_15m['close'].iloc[-1]
        if vol_ratio > 2.0 and price_range < 0.015:
            tf_score += 3
            tf_reasons.append(f"🔥 15M: Vol spike {vol_ratio:.1f}x, range {price_range*100:.2f}%")
        elif vol_ratio > 1.5 and price_range < 0.025:
            tf_score += 2
            tf_reasons.append(f"📊 15M: Vol meningkat {vol_ratio:.1f}x, konsolidasi")
        if 'CVD' in df_15m.columns and len(df_15m) > 30:
            cvd_slope = df_15m['CVD'].diff().tail(10).mean()
            if cvd_slope > 0:
                tf_score += 1
                tf_reasons.append("📈 15M: CVD naik - akumulasi")
        result['tf_15m'] = {'score': min(5, tf_score), 'signal': ' | '.join(tf_reasons[:2])}
        result['score'] += tf_score * 1.2
    
    if df_1h is not None and len(df_1h) > 50:
        tf_score = 0
        tf_reasons = []
        lows = df_1h['low'].tail(20).values
        if len(lows) >= 5:
            hl1 = min(lows[:5]); hl2 = min(lows[5:10]); hl3 = min(lows[10:15]); hl4 = min(lows[15:20])
            if hl4 > hl3 > hl2 > hl1:
                tf_score += 2
                tf_reasons.append("📈 1H: Higher Low - bandar cicil beli!")
        vol_ma20 = df_1h['volume'].rolling(20).mean().iloc[-1]
        vol_ratio = df_1h['volume'].iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1
        if vol_ratio > 1.8:
            tf_score += 1
            tf_reasons.append(f"📊 1H: Vol {vol_ratio:.1f}x")
        result['tf_1h'] = {'score': min(5, tf_score), 'signal': ' | '.join(tf_reasons[:2])}
        result['score'] += tf_score * 0.8
    
    if df_4h is not None and len(df_4h) > 30:
        tf_score = 0
        tf_reasons = []
        if 'Supertrend_Dir' in df_4h.columns:
            st_dir = safe_get(df_4h['Supertrend_Dir'].iloc[-1], 0)
            if st_dir == 1:
                tf_score += 1
                tf_reasons.append("📈 4H: Supertrend Bullish")
        if 'ADX' in df_4h.columns:
            adx = safe_get(df_4h['ADX'].iloc[-1], 0)
            if adx > 25:
                tf_score += 1
                tf_reasons.append(f"📊 4H: ADX {adx:.0f} - tren menguat")
        if 'MA20' in df_4h.columns and 'MA50' in df_4h.columns:
            if df_4h['MA20'].iloc[-1] > df_4h['MA50'].iloc[-1]:
                tf_score += 1
                tf_reasons.append("🟢 4H: MA20 > MA50")
        result['tf_4h'] = {'score': min(5, tf_score), 'signal': ' | '.join(tf_reasons[:2])}
        result['score'] += tf_score * 0.6
    
    total_score = result['score']
    if total_score >= 12:
        result['is_accumulating'] = True
        result['status'] = "🔥 STRONG ACCUMULATION (1-2 HARI)"
        result['estimated_pump_time'] = "1-2 hari"
    elif total_score >= 8:
        result['is_accumulating'] = True
        result['status'] = "✅ ACCUMULATING (2-3 HARI)"
        result['estimated_pump_time'] = "2-3 hari"
    elif total_score >= 5:
        result['is_accumulating'] = True
        result['status'] = "🔍 EARLY ACCUMULATION (3-5 HARI)"
        result['estimated_pump_time'] = "3-5 hari"
    else:
        result['is_accumulating'] = False
        result['status'] = "❌ NO ACCUMULATION"
        result['estimated_pump_time'] = "N/A"
    
    if result['tf_15m']['signal'] != 'N/A':
        result['reasons'].append(result['tf_15m']['signal'])
    if result['tf_1h']['signal'] != 'N/A':
        result['reasons'].append(result['tf_1h']['signal'])
    if result['tf_4h']['signal'] != 'N/A':
        result['reasons'].append(result['tf_4h']['signal'])
    
    return result

# ======================== WHALE ACCUMULATION ========================
def detect_whale_accumulation(df_15m, df_1h):
    if df_15m is None or len(df_15m) < 50:
        return {'is_whale_accumulating': False, 'score': 0, 'reasons': []}
    
    score = 0
    reasons = []
    
    if 'Vol_Delta' in df_15m.columns:
        delta_15m = df_15m['Vol_Delta'].tail(20).mean()
        price_change = (df_15m['close'].iloc[-1] - df_15m['close'].iloc[-20]) / df_15m['close'].iloc[-20] * 100
        if delta_15m > 0 and price_change < -1:
            score += 3
            reasons.append(f"🐋 Whale buy the dip! Delta +{delta_15m:.0f} di harga turun {price_change:.1f}%")
        elif delta_15m > 0 and price_change < 0:
            score += 2
            reasons.append(f"🐋 Whale accumulating di harga turun (Delta +{delta_15m:.0f})")
    
    if 'CVD' in df_1h.columns and len(df_1h) > 30:
        cvd_trend = df_1h['CVD'].diff().tail(10).mean()
        price_range = (df_1h['high'].tail(10).max() - df_1h['low'].tail(10).min()) / df_1h['close'].iloc[-1] * 100
        if cvd_trend > 0 and price_range < 2:
            score += 2
            reasons.append(f"📈 CVD naik {cvd_trend:.0f} di range {price_range:.1f}% - Stealth accumulation!")
    
    if len(df_15m) > 20:
        vol_ma = df_15m['volume'].rolling(20).mean().iloc[-1]
        vol_spike = df_15m['volume'].iloc[-1] / vol_ma if vol_ma > 0 else 1
        price_move = abs(df_15m['close'].pct_change().iloc[-1]) * 100
        if vol_spike > 3 and price_move < 1:
            score += 2
            reasons.append(f"🔥 Volume spike {vol_spike:.1f}x, harga {price_move:.1f}% - Whale masuk!")
    
    return {
        'is_whale_accumulating': score >= 5,
        'score': min(10, score),
        'reasons': reasons[:3]
    }

# ======================== BREAKOUT READINESS ========================
def detect_breakout_readiness(df_15m, df_1h):
    if df_15m is None or len(df_15m) < 50:
        return {'ready': False, 'score': 0, 'status': 'N/A', 'timeframe': 'N/A', 'reasons': []}
    
    score = 0
    reasons = []
    
    if 'BB_Squeeze' in df_15m.columns:
        squeeze_15m = bool(safe_get(df_15m['BB_Squeeze'].iloc[-1], False))
        squeeze_count = df_15m['BB_Squeeze'].tail(20).sum() if 'BB_Squeeze' in df_15m.columns else 0
        if squeeze_15m and squeeze_count > 10:
            score += 2
            reasons.append("📉 BB Squeeze 15M - Breakout imminent!")
    
    vol_ma20 = df_15m['volume'].rolling(20).mean().iloc[-1]
    vol_ma5 = df_15m['volume'].tail(5).mean()
    vol_ratio = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 1
    if vol_ratio > 1.5:
        score += 1
        reasons.append(f"📊 Volume meningkat {vol_ratio:.1f}x")
    
    if 'ATR' in df_15m.columns:
        atr_now = df_15m['ATR'].iloc[-1]
        atr_ma20 = df_15m['ATR'].rolling(20).mean().iloc[-1]
        atr_ratio = atr_now / atr_ma20 if atr_ma20 > 0 else 1
        if atr_ratio < 0.7:
            score += 1
            reasons.append(f"📉 ATR turun {atr_ratio*100:.0f}% - Range menyempit")
    
    if 'RSI' in df_1h.columns:
        rsi_now = df_1h['RSI'].iloc[-1]
        rsi_5ago = df_1h['RSI'].iloc[-6] if len(df_1h) >= 6 else rsi_now
        if rsi_now > rsi_5ago and rsi_now < 50:
            score += 1
            reasons.append(f"📈 RSI naik dari {rsi_5ago:.0f} ke {rsi_now:.0f}")
    
    if 'ADX' in df_1h.columns:
        adx_now = df_1h['ADX'].iloc[-1]
        adx_5ago = df_1h['ADX'].iloc[-6] if len(df_1h) >= 6 else adx_now
        if adx_now > adx_5ago and adx_now > 20:
            score += 1
            reasons.append(f"📈 ADX menguat {adx_5ago:.0f} -> {adx_now:.0f}")
    
    if score >= 5:
        status = "🔥 READY TO BREAKOUT (1-2 HARI)"
        ready = True
        timeframe = "1-2 hari"
    elif score >= 3:
        status = "📈 BUILDING UP (2-3 HARI)"
        ready = True
        timeframe = "2-3 hari"
    elif score >= 2:
        status = "🔍 OBSERVING (3-5 HARI)"
        ready = False
        timeframe = "3-5 hari"
    else:
        status = "⏳ CONSOLIDATING"
        ready = False
        timeframe = ">5 hari"
    
    return {'ready': ready, 'score': score, 'status': status, 'timeframe': timeframe, 'reasons': reasons}

# ======================== ANALYZE COIN FULL ========================
def analyze_coin_full_advanced(symbol, exchange_name):
    try:
        trading_date = current_trading_date_str()
        
        timeframes = ['15m', '1h', '4h', '1w', '1d']
        with ThreadPoolExecutor(max_workers=min(len(timeframes), 5)) as executor:
            future_to_tf = {
                executor.submit(fetch_ohlcv_cached, symbol, exchange_name, tf, 700 if tf == '15m' else 400): tf
                for tf in timeframes
            }
            results = {}
            for future in as_completed(future_to_tf):
                tf = future_to_tf[future]
                try:
                    results[tf] = future.result()
                except Exception:
                    results[tf] = None
        
        df_15m = results.get('15m')
        df_1h = results.get('1h')
        df_4h = results.get('4h')
        df_1w = results.get('1w')
        df_d_raw = results.get('1d')
        
        wib_daily_df, daily_source = build_wib_daily_df(df_15m, df_d_raw)
        if wib_daily_df is None:
            return None
        
        daily_ind = calculate_indicators_upgraded(wib_daily_df)
        if daily_ind is None:
            return None
        daily_ind = calculate_ichimoku_crypto(daily_ind)
        daily_ind = calculate_obv_upgraded(daily_ind)
        
        for tf, name in [(df_4h, '4h'), (df_1h, '1h'), (df_1w, '1w')]:
            if tf is not None:
                tf_calc = calculate_indicators_upgraded(tf)
                if tf_calc is not None:
                    if name == '4h':
                        df_4h = calculate_ichimoku_crypto(tf_calc)
                    elif name == '1h':
                        df_1h = tf_calc
                    elif name == '1w':
                        df_1w = calculate_ichimoku_crypto(tf_calc)
        
        if df_15m is not None:
            df_15m = calculate_indicators_upgraded(df_15m)
        
        last = daily_ind.iloc[-1]
        cp = float(last['close'])
        current_price = cp
        if df_15m is not None and len(df_15m) > 0:
            current_price = float(df_15m['close'].iloc[-1])
        
        rsi = safe_get(last.get('RSI', np.nan), 50)
        adx_v = safe_get(last.get('ADX', np.nan), 0)
        atr = safe_get(last.get('ATR', np.nan), cp * 0.02)
        st_dir = safe_get(last.get('Supertrend_Dir', np.nan), 0)
        st_bull = st_dir == 1
        cmf_v = safe_get(last.get('CMF', np.nan), 0)
        tsi_v = safe_get(last.get('TSI', np.nan), 0)
        kst_v = safe_get(last.get('KST', np.nan), 0)
        ps_b = int(safe_get(last.get('PSAR_Bull', np.nan), 0)) == 1
        hma_v = safe_get(last.get('HMA', np.nan), cp)
        dema_v = safe_get(last.get('DEMA', np.nan), cp)
        obv_up = bool(daily_ind['OBV_trend'].iloc[-1]) if 'OBV_trend' in daily_ind.columns else False
        ab200 = bool(not pd.isna(last.get('MA200', np.nan)) and last['close'] > last['MA200'])
        abvwap = bool(not pd.isna(last.get('VWAP', np.nan)) and last['close'] > last['VWAP'])
        mb1d = bool(not pd.isna(last.get('MACD_Hist', np.nan)) and last['MACD_Hist'] > 0)
        
        bull_signals = calculate_bull_signals(
            last, cp, st_bull, ps_b, tsi_v, kst_v, mb1d, obv_up, ab200, abvwap, cmf_v, hma_v, dema_v, rsi
        )
        
        vol_regime, vol_mult = classify_volatility_regime_improved(daily_ind)
        sups, ress = calculate_precise_sr(daily_ind, lookback=80)
        fib = calculate_fibonacci(daily_ind)
        chart_patterns = detect_chart_patterns_improved(daily_ind)
        elliot = detect_elliot_wave_improved(daily_ind)
        candle_patterns, candle_quality = detect_candlestick_patterns_advanced(daily_ind)
        divs = detect_divergences_improved(daily_ind)
        poc, vah, val = calculate_volume_profile_improved(daily_ind)
        liq_data = compute_liquidity_levels(daily_ind)
        mtf_score, mtf_desc = compute_mtf_confluence(df_15m, df_1h, df_4h, df_1w)
        H_exp = hurst_exponent_fixed(daily_ind['close'].tail(100).values)
        wyckoff_phase, wyckoff_msg, wyckoff_events = detect_wyckoff_improved(daily_ind)
        bos_choch = detect_bos_choch(daily_ind)
        bullish_fvg, bearish_fvg, filled_fvg = detect_fair_value_gap_improved(daily_ind)
        nearest_fvg = get_nearest_fvg(bullish_fvg, bearish_fvg, current_price)
        ob_data = detect_order_blocks_improved(daily_ind)
        sweep = detect_liquidity_sweep(daily_ind)
        inst_candle = detect_institutional_candle(daily_ind)
        absorption = detect_absorption(daily_ind)
        
        # NEW: Death Cat Bounce Detection
        death_cat = detect_death_cat_bounce(daily_ind, symbol)
        
        # NEW: Bandarmologi Enhanced Signals
        whale_activity = detect_whale_activity_enhanced(df_15m, df_1h)
        fake_breakout = detect_fake_breakout(daily_ind, ress[0]['price'] if ress else None)
        bandar_reversal = detect_bandar_reversal_candles(daily_ind)
        
        bandar_signals = {
            'whale': whale_activity,
            'fake_breakout': fake_breakout,
            'reversal_candle': bandar_reversal
        }
        
        # NEW: All Technical Indicators
        pivot_points = calculate_pivot_points(daily_ind)
        harmonic_patterns = detect_harmonic_patterns(daily_ind)
        atr_trailing_stop = calculate_atr_trailing_stop(daily_ind, period=14, multiplier=2.0)
        heikin_ashi = calculate_heikin_ashi(daily_ind)
        stoch_rsi = calculate_stoch_rsi(daily_ind, period=14, smooth=3)
        money_flow_index = calculate_mfi(daily_ind, period=14)
        aroon = calculate_aroon(daily_ind, period=25)
        zig_zag = calculate_zig_zag(daily_ind, lookback=100, deviation=0.03)
        cci = calculate_cci(daily_ind, period=20)
        ultimate_oscillator = calculate_ultimate_oscillator(daily_ind, period1=7, period2=14, period3=28)
        vortex = calculate_vortex(daily_ind, period=14)
        mass_index = calculate_mass_index(daily_ind, period=9, ema_period=25)
        rvi = calculate_rvi(daily_ind, period=10)
        force_index = calculate_force_index(daily_ind, period=13)
        
        if "Spring" in wyckoff_phase:
            mom = "🌱 WYCKOFF SPRING"
        elif st_bull and adx_v > 25:
            mom = "📊 ST+ADX KUAT"
        elif st_bull:
            mom = "📈 TREND NAIK"
        elif "Accumulation" in wyckoff_phase:
            mom = "⏳ ACCUMULATION"
        else:
            mom = "⚖️ NETRAL"
        
        raw_factors = compute_factor_ensemble_advanced(daily_ind, df_1w, df_1h, liq_data)
        
        if raw_factors:
            corr_matrix = compute_correlation_matrix(raw_factors)
            adj_weights = compute_ensemble_weights(raw_factors, corr_matrix)
            fn = list(raw_factors.keys())
            fs = np.array([raw_factors[k][0] for k in fn])
            fw = np.array([adj_weights.get(k, raw_factors[k][1]) for k in fn])
            tw = max(fw.sum(), 1e-10)
            ws = float(np.sum(fs * fw) / tw)
            agree_ratio = float(np.sum(np.sign(fs) == np.sign(ws))) / max(len(fs), 1)
        else:
            ws = 0.0
            agree_ratio = 0.5
        
        safe_symbol = symbol.replace('/', '_')
        if not st.session_state.get('ai_model_trained', False):
            if _ai_predictor.load_models(safe_symbol):
                st.write("✅ AI Model loaded from disk!")
                st.session_state['ai_model_trained'] = True
            else:
                if daily_ind is not None and len(daily_ind) > 100:
                    st.write("🧠 Training AI models (LSTM + Random Forest + LightGBM)...")
                    _ai_predictor.train(daily_ind)
                    _ai_predictor.save_models(safe_symbol)
                    st.session_state['ai_model_trained'] = True
                    st.write("✅ AI Model trained and saved to disk!")
        
        ai_pred = _ai_predictor.predict(daily_ind) if _ai_predictor.is_trained else {'signal': 0, 'confidence': 50}
        smc_score = calculate_smart_money_score(daily_ind, liq_data, poc, val, vah, current_price)
        advanced_data = {'ai_prediction': ai_pred}
        pump_analysis = detect_pump_opportunity_upgraded(symbol, df_15m, df_1h, daily_ind, ress, sups, advanced_data)
        tradeable, trade_reason = is_spot_tradeable(bull_signals, None, symbol, current_price)
        trade = calc_trade_plan(daily_ind, sups, ress, poc, val, atr, current_price, symbol)
        psp = pos_size(trade['conservative_entry'], trade['stop_loss']) if tradeable else 0.0
        enhanced_trade = enhance_trade_plan_with_smc(trade, smc_score, nearest_fvg, current_price, atr)
        
        cv_result = cross_validate_signals({
            'current_price': current_price,
            'smc_score': smc_score,
            'chart_patterns': chart_patterns,
            'candle_patterns': candle_patterns,
            'divs_1d': divs,
            'elliot_wave': elliot,
            'bos_choch': bos_choch,
            'trade_plan': trade,
            'symbol': symbol
        })
        
        h0_ultra = None
        if df_15m is not None and len(df_15m) >= 24:
            h0_ultra = {
                'open': float(df_15m['open'].iloc[-1]),
                'high': float(df_15m['high'].iloc[-1]),
                'low': float(df_15m['low'].iloc[-1]),
                'close': float(df_15m['close'].iloc[-1])
            }
        
        predictions_7d = []
        pred_sum = {}
        
        try:
            predictions_7d = predict_hlc_7d(
                wib_daily_df, df_weekly=df_1w, h0_ultra=h0_ultra,
                df_1h=df_1h, df_4h=df_4h, df_15m=df_15m,
                trading_date=trading_date, symbol=symbol
            )
            if predictions_7d:
                pred_sum = get_prediction_summary(predictions_7d)
            else:
                predictions_7d = predict_hlc_7d_simple(wib_daily_df, symbol)
                if predictions_7d:
                    pred_sum = get_prediction_summary(predictions_7d)
        except Exception:
            predictions_7d = predict_hlc_7d_simple(wib_daily_df, symbol)
            if predictions_7d:
                pred_sum = get_prediction_summary(predictions_7d)
        
        result = {
            'symbol': symbol,
            'current_price': current_price,
            'open_wib_today': float(last['open']),
            'high_wib_today': float(last['high']),
            'low_wib_today': float(last['low']),
            'close_wib_today': float(last['close']),
            'daily_source': daily_source,
            **trade,
            'pos_size_pct': psp,
            'atr': atr,
            'rsi': round(rsi, 1),
            'adx': round(adx_v, 1),
            'supertrend_bull': st_bull,
            'cmf': round(cmf_v, 3),
            'tsi': round(tsi_v, 2),
            'kst': round(kst_v, 2),
            'psar_bull': ps_b,
            'hma': hma_v,
            'dema': dema_v,
            'macd_bull_1d': mb1d,
            'obv_rising': obv_up,
            'above_ma200': ab200,
            'above_vwap': abvwap,
            'hurst': round(H_exp, 3),
            'structure_desc': chart_patterns.get('description', 'N/A'),
            'wyckoff_phase': wyckoff_phase,
            'wyckoff_msg': wyckoff_msg,
            'wyckoff_events': wyckoff_events,
            'poc': poc,
            'vah': vah,
            'val': val,
            'divs_1d': divs,
            'candle_patterns': candle_patterns,
            'candle_quality': candle_quality,
            'supports': sups,
            'resistances': ress,
            'momentum': mom,
            'bull_signals': bull_signals,
            'spot_tradeable': tradeable,
            'trade_reason': trade_reason,
            'vol_regime': vol_regime,
            'vol_mult': vol_mult,
            'liq_data': liq_data,
            'mtf_score': mtf_score,
            'mtf_desc': mtf_desc,
            'daily': daily_ind,
            'tf_4h': df_4h,
            'tf_1h': df_1h,
            'tf_15m': df_15m,
            'tf_1w': df_1w,
            'pred_locked_at': wib_now().strftime('%d %b %Y %H:%M WIB'),
            'pred_locked_trading_date': trading_date,
            'actuals_refreshed_at': wib_now().strftime('%d %b %Y %H:%M WIB'),
            'smc_score': smc_score,
            'bullish_fvg': bullish_fvg,
            'bearish_fvg': bearish_fvg,
            'filled_fvg': filled_fvg,
            'nearest_fvg': nearest_fvg,
            'enhanced_trade': enhanced_trade,
            'pump_analysis': pump_analysis,
            'advanced_data': advanced_data,
            'ai_prediction': ai_pred,
            'fibonacci_levels': fib,
            'chart_patterns': chart_patterns,
            'elliot_wave': elliot,
            'predictions_7d': predictions_7d,
            'pred_summary': pred_sum,
            'agreement_pct': round(agree_ratio * 100, 1),
            'confidence_score': compute_confidence_score({
                'agreement_pct': agree_ratio * 100,
                'smc_score': smc_score,
                'pump_analysis': pump_analysis,
                'ai_prediction': ai_pred,
                'hurst': H_exp,
                'vol_regime': vol_regime,
                'cross_validation': cv_result
            }),
            'bos_choch': bos_choch,
            'fvg_status': {'unfilled_bullish': bullish_fvg, 'unfilled_bearish': bearish_fvg, 'filled': filled_fvg},
            'ob_validation': ob_data,
            'liquidity_sweep': sweep,
            'institutional_candle': inst_candle,
            'absorption_detected': absorption,
            'divergence_strength': divs.get('strength', 0),
            'failed_divergence': divs.get('failed', False),
            'volume_profile_data': {'poc': poc, 'vah': vah, 'val': val},
            'cross_validation': cv_result,
            'death_cat_bounce': death_cat,
            'bandar_signals': bandar_signals,
            'vwap_bands': {
                'vwap': daily_ind['VWAP'].iloc[-1] if 'VWAP' in daily_ind.columns else None,
                'upper_1': daily_ind['VWAP_Upper_1'].iloc[-1] if 'VWAP_Upper_1' in daily_ind.columns else None,
                'lower_1': daily_ind['VWAP_Lower_1'].iloc[-1] if 'VWAP_Lower_1' in daily_ind.columns else None,
            },
            'adl_analysis': {
                'adl': daily_ind['ADL'].iloc[-1] if 'ADL' in daily_ind.columns else None,
                'divergence': daily_ind['ADL_Divergence'].iloc[-1] if 'ADL_Divergence' in daily_ind.columns else 'N/A',
            },
            # NEW TECHNICALS
            'pivot_points': pivot_points,
            'harmonic_patterns': harmonic_patterns,
            'atr_trailing_stop': atr_trailing_stop,
            'heikin_ashi': heikin_ashi,
            'stoch_rsi': stoch_rsi,
            'money_flow_index': money_flow_index,
            'aroon': aroon,
            'zig_zag': zig_zag,
            'cci': cci,
            'ultimate_oscillator': ultimate_oscillator,
            'vortex': vortex,
            'mass_index': mass_index,
            'rvi': rvi,
            'force_index': force_index
        }
        
        save_predictions(symbol, exchange_name, trading_date, predictions_7d)
        save_full_snapshot(symbol, exchange_name, trading_date, result)
        return result
    except Exception:
        return None

# ======================== RENDER RESULT (SINGKAT) ========================
def render_result_advanced(dr):
    if dr is None or not isinstance(dr, dict):
        st.warning("⚠️ Belum ada hasil scan. Silakan scan aset terlebih dahulu.")
        return
    
    price = dr['current_price']
    today_wib = wib_now()
    tradeable = dr.get('spot_tradeable', False)
    smc = dr.get('smc_score', {})
    pump = dr.get('pump_analysis', {})
    ai = dr.get('ai_prediction', {})
    fib = dr.get('fibonacci_levels', {})
    chart = dr.get('chart_patterns', {})
    elliot = dr.get('elliot_wave', {})
    preds = dr.get('predictions_7d', [])
    pred_sum = dr.get('pred_summary', {})
    params = get_asset_params(dr['symbol'], price)
    confidence = dr.get('confidence_score', 50)
    cv = dr.get('cross_validation', {})
    bos_choch = dr.get('bos_choch', {})
    sweep = dr.get('liquidity_sweep', {})
    inst_candle = dr.get('institutional_candle', {})
    absorption = dr.get('absorption_detected', {})
    div_strength = dr.get('divergence_strength', 0)
    death_cat = dr.get('death_cat_bounce', {})
    bandar = dr.get('bandar_signals', {})
    
    # NEW TECHNICALS
    pivot = dr.get('pivot_points', {})
    harmonic = dr.get('harmonic_patterns', {})
    atr_ts = dr.get('atr_trailing_stop', {})
    ha = dr.get('heikin_ashi', {})
    stoch = dr.get('stoch_rsi', {})
    mfi = dr.get('money_flow_index', {})
    aroon = dr.get('aroon', {})
    cci = dr.get('cci', {})
    ultimate = dr.get('ultimate_oscillator', {})
    vortex = dr.get('vortex', {})
    mass = dr.get('mass_index', {})
    rvi = dr.get('rvi', {})
    force = dr.get('force_index', {})
    
    st.markdown(f"### 🎯 {dr['symbol']} — v16.2 ALL TECHNICALS ({params['name']})")
    st.caption(f"⏰ {today_wib.strftime('%A, %d %B %Y · %H:%M')} WIB")
    
    # Bandarmologi Signals
    if bandar:
        whale = bandar.get('whale', {})
        if whale.get('whale_detected', False):
            st.markdown(f'<div class="bandar-signal">🐋 <b>{whale.get("status", "N/A")}</b><br>{whale.get("action", "N/A")}</div>', unsafe_allow_html=True)
        fake_breakout = bandar.get('fake_breakout', {})
        if fake_breakout.get('fake_breakout', False):
            st.markdown(f'<div class="bandar-signal">🎯 <b>{fake_breakout.get("message", "N/A")}</b></div>', unsafe_allow_html=True)
        reversal = bandar.get('reversal_candle', {})
        if reversal.get('pattern') != 'none':
            if reversal.get('is_bullish', False):
                st.success(f"📌 **{reversal.get('signal', 'N/A')}**")
            elif reversal.get('is_bearish', False):
                st.error(f"📌 **{reversal.get('signal', 'N/A')}**")
    
    # Harmonic Patterns
    if harmonic and harmonic.get('patterns'):
        for pattern in harmonic.get('patterns', []):
            if 'Bullish' in pattern or 'Gartley' in pattern or 'Bat' in pattern or 'Crab' in pattern or 'Butterfly' in pattern:
                st.markdown(f'<div class="harmonic-bull">📐 {pattern}</div>', unsafe_allow_html=True)
    
    # Death Cat Bounce Warning
    if death_cat.get('is_death_cat', False):
        if death_cat.get('score', 0) >= 7:
            st.markdown(f'<div class="death-cat-high">💀 <b>DEATH CAT BOUNCE DETECTED!</b> Score: {death_cat["score"]}/10 | Risk: {death_cat["risk_level"]}<br>{death_cat["action"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="death-cat-mid">🐱 <b>Potential Death Cat Bounce</b> Score: {death_cat["score"]}/10 | Risk: {death_cat["risk_level"]}<br>{death_cat["action"]}</div>', unsafe_allow_html=True)
        
        with st.expander("⚠️ Death Cat Bounce Details", expanded=True):
            st.markdown(f"**Signal:** {death_cat.get('signal', 'N/A')}")
            st.markdown(f"**Confidence:** {death_cat.get('confidence', 0):.0f}%")
            st.markdown(f"**RSI:** {death_cat.get('rsi', 0):.1f} | **Volume Ratio:** {death_cat.get('vol_ratio', 0):.2f}x | **CVD:** {death_cat.get('cvd_trend', 0):.2f}")
            st.markdown("**Reasons:**")
            for reason in death_cat.get('reasons', []):
                st.markdown(f"- {reason}")
    
    bullish_pct = cv.get('bullish_pct', 0)
    bearish_pct = cv.get('bearish_pct', 0)
    dominant = cv.get('dominant', 'neutral')
    rr = cv.get('rr', 0)
    rr_status = cv.get('rr_status', 'N/A')
    
    st.markdown("---")
    st.markdown("### ⚖️ Weighted Voting Result")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🟢 Bullish", f"{bullish_pct:.1f}%")
    with col2:
        st.metric("🔴 Bearish", f"{bearish_pct:.1f}%")
    with col3:
        if dominant == 'bullish':
            st.metric("📊 Dominant", "BULLISH", delta=f"+{bullish_pct - bearish_pct:.1f}%")
        elif dominant == 'bearish':
            st.metric("📊 Dominant", "BEARISH", delta=f"-{bearish_pct - bullish_pct:.1f}%")
        else:
            st.metric("📊 Dominant", "MIXED", delta="< 10% diff")
    with col4:
        st.metric("📊 R/R", f"{rr:.2f}", rr_status)
    
    st.progress(bullish_pct / 100, text=f"Bullish {bullish_pct:.1f}% | Bearish {bearish_pct:.1f}%")
    
    # Technical Indicators Summary
    st.markdown("---")
    st.markdown("### 📊 TEKNIKAL LENGKAP (56+ Indicators)")
    
    tech_cols = st.columns(4)
    
    with tech_cols[0]:
        st.markdown("**📌 Pivot Points**")
        if pivot.get('sentiment'):
            st.markdown(f"{pivot.get('sentiment', 'N/A')}")
            classic = pivot.get('classic', {})
            if classic:
                st.markdown(f"R1: {fmt_price(classic.get('r1'))}")
                st.markdown(f"S1: {fmt_price(classic.get('s1'))}")
        
        st.markdown("**📐 Harmonic**")
        if harmonic and harmonic.get('patterns'):
            for p in harmonic.get('patterns', [])[:2]:
                st.markdown(f"{p}")
        else:
            st.markdown("Tidak ada")
        
        st.markdown("**🎯 ATR Trailing Stop**")
        if atr_ts and atr_ts.get('current_stop'):
            st.markdown(f"Stop: {fmt_price(atr_ts.get('current_stop'))}")
            st.markdown(f"Distance: {atr_ts.get('stop_distance_pct', 0)}%")
    
    with tech_cols[1]:
        st.markdown("**📈 Heikin Ashi**")
        if ha:
            st.markdown(f"{ha.get('ha_trend', 'N/A')}")
            st.markdown(f"Streak: {ha.get('ha_streak', 0)}")
            if ha.get('ha_reversal'):
                st.markdown(f"🔄 {ha.get('ha_signal', '')}")
        
        st.markdown("**📊 Stochastic RSI**")
        if stoch:
            st.markdown(f"K={stoch.get('k', 50):.1f} | D={stoch.get('d', 50):.1f}")
            st.markdown(f"{stoch.get('signal', 'N/A')}")
        
        st.markdown("**💰 MFI**")
        if mfi:
            st.markdown(f"{mfi.get('mfi', 50):.1f} - {mfi.get('signal', 'N/A')}")
            if mfi.get('divergence') != 'None':
                st.markdown(f"{mfi.get('divergence', '')}")
    
    with tech_cols[2]:
        st.markdown("**📊 Aroon**")
        if aroon:
            st.markdown(f"Up={aroon.get('aroon_up', 50):.1f} | Down={aroon.get('aroon_down', 50):.1f}")
            st.markdown(f"{aroon.get('signal', 'N/A')}")
        
        st.markdown("**📊 CCI**")
        if cci:
            st.markdown(f"{cci.get('cci', 0):.1f} - {cci.get('signal', 'N/A')}")
        
        st.markdown("**📊 Ultimate Osc**")
        if ultimate:
            st.markdown(f"{ultimate.get('ultimate', 50):.1f} - {ultimate.get('signal', 'N/A')}")
    
    with tech_cols[3]:
        st.markdown("**📊 Vortex**")
        if vortex:
            st.markdown(f"VI+={vortex.get('vi_plus', 1):.3f} | VI-={vortex.get('vi_minus', 1):.3f}")
            st.markdown(f"{vortex.get('signal', 'N/A')}")
        
        st.markdown("**📊 Mass Index**")
        if mass:
            st.markdown(f"{mass.get('mass_index', 0):.1f} (Threshold: {mass.get('threshold', 26)})")
            st.markdown(f"{mass.get('signal', 'N/A')}")
        
        st.markdown("**📊 Force Index**")
        if force:
            st.markdown(f"{force.get('force_index_norm', 0):.2f} - {force.get('signal', 'N/A')}")
    
    # Rest of rendering (same as before)
    if bos_choch.get('bos', False):
        bos_desc = bos_choch.get('description', '')
        if 'Bullish' in bos_desc:
            st.success(f"📈 **BOS BULLISH** — {bos_desc}")
        else:
            st.error(f"📉 **BOS BEARISH** — {bos_desc}")
    
    if bos_choch.get('choch', False):
        choch_desc = bos_choch.get('description', '')
        if 'Bullish' in choch_desc:
            st.success(f"🔄 **CHoCH BULLISH** — {choch_desc} (POTENTIAL REVERSAL!)")
        else:
            st.error(f"🔄 **CHoCH BEARISH** — {choch_desc} (POTENTIAL REVERSAL!)")
    
    if sweep.get('sweep', False):
        if sweep.get('type') == 'bullish':
            st.success(f"🎯 **{sweep.get('description', 'Liquidity Sweep')}** — Stop Hunt Bullish!")
        else:
            st.error(f"🎯 **{sweep.get('description', 'Liquidity Sweep')}** — Stop Hunt Bearish!")
    
    if inst_candle.get('detected', False):
        st.info(f"🏦 **Institutional Activity Detected** — {inst_candle.get('recent_count', 0)} candles in last 10 periods")
    
    if absorption.get('detected', False):
        st.warning(f"📊 **Absorption Detected** — {absorption.get('recent_count', 0)} candles (high volume + small spread)")
    
    if cv.get('conflict_detected', False):
        st.warning(f"⚡ **Signal Conflict Detected** — Adjustment: {cv.get('score_adjustment', 0)}")
        for conflict in cv.get('conflicts', [])[:3]:
            st.markdown(f"- {conflict}")
    
    if death_cat.get('is_death_cat', False):
        st.error(f"💀 **DEATH CAT BOUNCE!** Score: {death_cat.get('score', 0)}/10 - {death_cat.get('action', 'N/A')}")
    
    if tradeable and not death_cat.get('is_death_cat', False):
        st.success(f"✅ **SPOT LAYAK BELI** — bull_signals={dr['bull_signals']}/12 · {dr.get('trade_reason', '')}")
    elif tradeable and death_cat.get('is_death_cat', False):
        st.error(f"⚠️ **TRADEABLE TAPI DEATH CAT!** - {death_cat.get('action', 'JANGAN BELI!')}")
    else:
        st.error(f"🚫 **SPOT TIDAK LAYAK BELI** — {dr.get('trade_reason', '')}")
    
    conf_color = "🟢" if confidence >= 70 else ("🟡" if confidence >= 50 else "🔴")
    st.info(f"{conf_color} **Confidence Score:** {confidence:.0f}% (Divergence Strength: {div_strength:.2f})")
    
    pump_score = pump.get('score', 0)
    if pump_score >= 7:
        st.markdown(f'<div class="pump-high">🚀 <b>PUMP: {pump.get("signal")}</b> (Score: {pump_score}/10, Active Factors: {pump.get("active_factors", 0)})</div>', unsafe_allow_html=True)
    elif pump_score >= 5:
        st.markdown(f'<div class="pump-mid">📈 <b>PUMP: {pump.get("signal")}</b> (Score: {pump_score}/10, Active Factors: {pump.get("active_factors", 0)})</div>', unsafe_allow_html=True)
    elif pump_score >= 3:
        st.markdown(f'<div class="pump-low">🔍 <b>PUMP: {pump.get("signal")}</b> (Score: {pump_score}/10)</div>', unsafe_allow_html=True)
    
    if pump.get('reasons'):
        with st.expander("📋 Detail Pump Signals", expanded=False):
            for reason in pump['reasons']:
                st.markdown(f"- {reason}")
    
    if ai and ai.get('signal') is not None:
        ai_signal = ai.get('signal', 0)
        ai_direction = "🟢 Bullish" if ai_signal > 0.1 else ("🔴 Bearish" if ai_signal < -0.1 else "⚪ Neutral")
        st.info(f"🤖 **AI Prediction:** {ai_direction} (Signal: {ai_signal:+.3f}) - Confidence: {ai.get('confidence', 0):.0f}%")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💲 Harga", fmt_price(price))
    col2.metric("📊 Bull Signals", f"{dr['bull_signals']}/12")
    col3.metric("🧠 SMC", f"{smc.get('score', 0)}/10", smc.get('level', ''))
    col4.metric("🚀 Pump", f"{pump_score}/10")
    col5.metric("📐 Hurst", f"{dr.get('hurst', 0.5):.3f}")
    
    # Predictions (shortened)
    if preds:
        st.markdown("---")
        st.markdown("### 🔮 Prediksi HLC H+0→H+7")
        
        if pred_sum:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Bias 7H", pred_sum.get('overall', 'N/A')[:20])
            col2.metric("Max Upside", f"+{pred_sum.get('max_upside', 0):.2f}%")
            col3.metric("Max Downside", f"{pred_sum.get('max_downside', 0):.2f}%")
            col4.metric("Avg Conf", f"{pred_sum.get('avg_confidence', 0):.1f}%")
            col5.metric("H+7 Change", f"{pred_sum.get('final_7d_change', 0):.2f}%")
    
    # Trade Plan (shortened)
    st.markdown("---")
    st.markdown("### 💹 Trade Plan (Spot Optimized)")
    
    enhanced_trade = dr.get('enhanced_trade', {})
    entry = enhanced_trade.get('conservative_entry_smc', dr.get('conservative_entry'))
    aggressive_entry = dr.get('aggressive_entry', entry)
    premium_entry = dr.get('premium_entry', entry * 0.99)
    entry_range_bottom = dr.get('entry_range_bottom', entry * 0.99)
    entry_range_top = dr.get('entry_range_top', entry * 1.01)
    
    sl = dr.get('stop_loss')
    tp1 = dr.get('tp1')
    tp2 = dr.get('tp2')
    tp3 = dr.get('tp3')
    
    if enhanced_trade.get('smc_boost_applied', False):
        st.caption("✨ SMC Boost aktif — diskon 0.5% dari entry standar")
    
    st.markdown("**📥 Entry Zone:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Premium", fmt_price(premium_entry))
    with col2:
        st.metric("🟢 Conservative", fmt_price(entry))
    with col3:
        st.metric("🟡 Aggressive", fmt_price(aggressive_entry))
    with col4:
        st.metric("📊 Support", fmt_price(dr.get('support_price', entry)))
    
    st.caption(f"📊 Entry Range: **{fmt_price(entry_range_bottom)}** — **{fmt_price(entry_range_top)}**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 Stop Loss", fmt_price(sl), f"{dr.get('sl_pct', 0):+.2f}%")
    with col2:
        st.metric("📊 R/R", f"1:{dr.get('rr', 0):.2f}")
    with col3:
        st.metric("📊 Pos Size", f"{dr.get('pos_size_pct', 0)}%")
    
    st.markdown("**🎯 Take Profit:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("TP1", fmt_price(tp1), f"+{dr.get('tp1_pct_entry', 0):.2f}%")
    with col2:
        st.metric("TP2", fmt_price(tp2), f"+{dr.get('tp2_pct_entry', 0):.2f}%")
    with col3:
        st.metric("TP3", fmt_price(tp3), f"+{dr.get('tp3_pct_entry', 0):.2f}%")
    
    # Portfolio Tracker
    render_portfolio()
    
    # AI Conclusion
    st.markdown("---")
    st.markdown("### 🧠 AI Conclusion & Recommendation")
    
    try:
        ai_conclusion = generate_ai_conclusion(dr)
        st.markdown(ai_conclusion)
    except Exception as e:
        st.error(f"❌ Error AI Conclusion: {e}")
        cv = dr.get('cross_validation', {})
        st.info(f"""
        **⚠️ AI Conclusion Error - Manual Display:**
        - Priority Signal: {cv.get('priority_signal', 'neutral')}
        - Action: {cv.get('action', 'HOLD')}
        - Conflicts: {' | '.join(cv.get('conflicts', [])) or 'No conflicts'}
        - Bullish: {cv.get('bullish_pct', 0):.1f}% | Bearish: {cv.get('bearish_pct', 0):.1f}%
        - R/R: {cv.get('rr', 0):.2f}
        - SMC Score: {dr.get('smc_score', {}).get('score', 0)}/10
        - Bull Signals: {dr.get('bull_signals', 0)}/12
        - Confidence: {dr.get('confidence_score', 50):.0f}%
        - Death Cat: {dr.get('death_cat_bounce', {}).get('signal', 'N/A')}
        - Whale: {dr.get('bandar_signals', {}).get('whale', {}).get('status', 'N/A')}
        """)

# ======================== PORTFOLIO TRACKER ========================
def render_portfolio():
    st.markdown("---")
    st.markdown("## 📊 Portfolio Tracker")
    
    if not st.session_state.portfolio:
        st.info("Belum ada posisi yang ditrack. Scan dan entry dulu!")
        return
    
    total_pnl = 0
    total_position = 0
    
    for symbol, positions in st.session_state.portfolio.items():
        st.markdown(f"### {symbol}")
        
        for i, pos in enumerate(positions):
            status_color = "🟢" if pos['status'] == 'OPEN' else "🔴" if pos['status'] == 'SL_HIT' else "🟡"
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Entry", fmt_price(pos['entry_price']))
            col2.metric("Current", fmt_price(pos['current_price']))
            col3.metric("PNL", f"{pos['pnl_pct']:.2f}%", delta=f"{pos['pnl_pct']:.2f}%")
            col4.metric("Size", f"{pos['position_size']:.2f}%")
            col5.metric("Status", f"{status_color} {pos['status']}")
            
            if pos['status'] == 'OPEN':
                total_pnl += pos['pnl_pct'] * pos['position_size']
                total_position += pos['position_size']
    
    if total_position > 0:
        st.metric("Total PNL (Weighted)", f"{total_pnl/total_position:.2f}%")

# ======================== MAIN ENTRY ========================
def main():
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        exchange_name = st.selectbox("Exchange", ["binance", "bybit", "okx", "kucoin"], index=0)
        st.markdown("---")
        st.subheader("🔍 Analisis")
        manual_sym = st.text_input("Symbol", placeholder="BTC, ETH, SOL, dll", key="input_symbol")
        st.caption("💡 Cukup ketik BTC → otomatis BTC/USDT")
        scan_clicked = st.button("🔍 Deep Scan v16.2 ALL TECHNICALS", use_container_width=True, type="primary")
        reset_btn = st.button("🗑️ Reset", use_container_width=True, disabled=(st.session_state.manual_result is None))
        st.markdown("---")
        st.subheader("✅ v16.2 ALL TECHNICALS")
        st.markdown("""
        **56+ TECHNICAL INDICATORS:**
        - Pivot Points (Classic + Fibonacci)
        - Harmonic Patterns (Gartley, Bat, Crab, Butterfly, Shark)
        - ATR Trailing Stop (Adaptive)
        - Heikin Ashi + Reversal Detection
        - Stochastic RSI + Cross Detection
        - Money Flow Index (MFI) + Divergence
        - Aroon + Cross Detection
        - Zig Zag + Swing Points
        - CCI + Overbought/Oversold
        - Ultimate Oscillator
        - Vortex + Cross Detection
        - Mass Index (Reversal Warning)
        - RVI (Relative Vigor Index)
        - Force Index (Elder's)
        - + 42 Existing Indicators
        """)
        now_wib = wib_now()
        st.markdown(f"**⏰ {now_wib.strftime('%H:%M')}**")
    
    if reset_btn:
        st.session_state.manual_result = None
        st.session_state.locked_symbol = None
        st.session_state.locked_exchange = None
        st.rerun()
    
    if scan_clicked and manual_sym:
        sym = manual_sym.strip().upper()
        if not sym.endswith('/USDT'):
            sym = sym + '/USDT'
        with st.spinner(f"🔍 Deep Scanning {sym} with 56+ TECHNICALS..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("📡 Menghubungi exchange (parallel fetch)...")
            progress_bar.progress(15)
            time.sleep(0.3)
            status_text.text("📊 Mengambil data 5 timeframes parallel...")
            progress_bar.progress(30)
            time.sleep(0.3)
            status_text.text("🧠 Menghitung 56+ Indicators + Bandarmologi...")
            progress_bar.progress(50)
            time.sleep(0.3)
            status_text.text("📐 Detecting Harmonic Patterns + Pivot Points...")
            progress_bar.progress(70)
            time.sleep(0.3)
            status_text.text("🤖 Training/Loading AI (LSTM + RF + LGB)...")
            progress_bar.progress(85)
            time.sleep(0.3)
            result = analyze_coin_full_advanced(sym, exchange_name)
            progress_bar.progress(100)
            status_text.text("✅ Selesai!")
            time.sleep(0.3)
            progress_bar.empty()
            status_text.empty()
        if result:
            st.session_state.manual_result = result
            st.session_state.locked_symbol = sym
            st.session_state.locked_exchange = exchange_name
            whale_status = result.get('bandar_signals', {}).get('whale', {}).get('status', 'N/A')
            harmonic = result.get('harmonic_patterns', {}).get('signal', 'Tidak ada')
            st.success(f"✅ Scan {sym} berhasil! Whale: {whale_status} | Harmonic: {harmonic} | Death Cat: {result.get('death_cat_bounce', {}).get('score', 0)}/10")
            st.rerun()
        else:
            st.error(f"❌ Gagal scan {sym}. Coba symbol lain atau ganti exchange.")
    
    if st.session_state.manual_result is not None:
        render_result_advanced(st.session_state.manual_result)
    else:
        st.info("""
        ### 🚀 Holy Grail Spot Scanner v16.2 — ALL TECHNICALS (56+ Indicators)
        
        **🆕 NEW TECHNICALS:**
        1. **Pivot Points** (Classic + Fibonacci) - Support/Resistance harian
        2. **Harmonic Patterns** (Gartley, Bat, Crab, Butterfly, Shark)
        3. **ATR Trailing Stop** - Stop loss dinamis
        4. **Heikin Ashi** - Filter noise, lihat trend
        5. **Stochastic RSI** - Entry timing
        6. **Money Flow Index (MFI)** - Akumulasi/Distribusi
        7. **Aroon** - Trend strength
        8. **Zig Zag** - Swing points
        9. **CCI** - Cycle reversal
        10. **Ultimate Oscillator** - Multi-TF oscillator
        11. **Vortex** - Trend direction
        12. **Mass Index** - Reversal warning
        13. **RVI** - Momentum
        14. **Force Index** - Trend strength
        
        **✅ EXISTING:** 42+ indicators (SMC, Bandarmologi, Death Cat, AI, dll)
        
        **Cara Pakai:**
        1. Di sidebar, ketik symbol (contoh: `BTC` atau `BTC/USDT`)
        2. Klik **"🔍 Deep Scan v16.2 ALL TECHNICALS"**
        3. Tunggu 5-10 detik
        4. Hasil muncul di halaman utama
        """)

if __name__ == "__main__":
    main()
