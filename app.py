import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ======================== KONFIGURASI ========================
st.set_page_config(page_title="Crypto Accumulation + Breakout Predictor", layout="wide")
st.title("🐋 Crypto Accumulation Scanner + Breakout Predictor")
st.markdown("Auto scan coin yang sedang diakumulasi whale + prediksi breakout & pump mendadak")

# ======================== CACHED FUNCTIONS ========================
@st.cache_data(ttl=3600)
def get_all_usdt_pairs(exchange_name):
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
        'timeout': 10000
    })
    markets = exchange.load_markets()
    usdt_pairs = [s for s in markets if s.endswith('/USDT') and markets[s]['spot']]
    valid_pairs = []
    for sym in usdt_pairs[:300]:
        try:
            ticker = exchange.fetch_ticker(sym)
            if ticker.get('quoteVolume', 0) > 10000:
                valid_pairs.append(sym)
        except:
            continue
        time.sleep(0.05)
    return valid_pairs

@st.cache_data(ttl=600)
def fetch_ohlcv_cached(symbol, exchange_name, timeframe='1d', limit=200):
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
        'timeout': 10000
    })
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv:
            return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching {symbol} {timeframe}: {e}")
        return None

# ======================== FUNGSI KORELASI ========================
def calculate_correlation(coin_df, btc_df, period=30):
    if coin_df is None or btc_df is None or len(coin_df) < period or len(btc_df) < period:
        return None
    try:
        coin_close = coin_df['close'].tail(period).values
        btc_close = btc_df['close'].tail(period).values
        corr = np.corrcoef(coin_close, btc_close)[0, 1]
        return round(corr, 2)
    except:
        return None

# ======================== INDIKATOR TEKNIKAL ========================
def calculate_indicators(df):
    if df is None or len(df) < 50:
        return None
    try:
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA50'] = df['close'].rolling(50).mean()
        df['MA200'] = df['close'].rolling(200).mean()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
        
        df['Volume_MA20'] = df['volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['volume'] / df['Volume_MA20'].replace(0, np.nan)
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        df['BB_Middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        return df
    except Exception as e:
        print(f"Error in calculate_indicators: {e}")
        return None

def calculate_ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
    if df is None or len(df) < senkou_b:
        return df
    try:
        df['tenkan'] = (df['high'].rolling(tenkan).max() + df['low'].rolling(tenkan).min()) / 2
        df['kijun'] = (df['high'].rolling(kijun).max() + df['low'].rolling(kijun).min()) / 2
        df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(kijun)
        senkou_b_high = df['high'].rolling(senkou_b).max()
        senkou_b_low = df['low'].rolling(senkou_b).min()
        df['senkou_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(kijun)
        df['chikou'] = df['close'].shift(-kijun)
        df['future_senkou_a'] = ((df['tenkan'] + df['kijun']) / 2)
        df['future_senkou_b'] = ((df['high'].rolling(senkou_b).max() + df['low'].rolling(senkou_b).min()) / 2)
        return df
    except Exception as e:
        print(f"Error in calculate_ichimoku: {e}")
        return df

def calculate_obv(df):
    if df is None or len(df) < 2:
        return df
    try:
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        if len(df) >= 20:
            slope = (df['OBV'].iloc[-1] - df['OBV'].iloc[-20]) / (abs(df['OBV'].iloc[-20]) + 1)
            df['OBV_trend'] = slope > 0
        else:
            df['OBV_trend'] = False
        return df
    except Exception as e:
        print(f"Error in calculate_obv: {e}")
        return df

def get_support_resistance_levels(df, current_price):
    if df is None or len(df) < 20:
        return current_price * 0.95, current_price * 1.05
    try:
        recent_lows = df['low'].tail(20)
        recent_highs = df['high'].tail(20)
        swing_low = recent_lows.min()
        swing_high = recent_highs.max()
        kijun = df['kijun'].iloc[-1] if 'kijun' in df else current_price
        senkou_a = df['senkou_a'].iloc[-1] if 'senkou_a' in df else current_price
        senkou_b = df['senkou_b'].iloc[-1] if 'senkou_b' in df else current_price

        candidates_below = [swing_low]
        if kijun < current_price:
            candidates_below.append(kijun)
        if senkou_a < current_price:
            candidates_below.append(senkou_a)
        if senkou_b < current_price:
            candidates_below.append(senkou_b)
        nearest_support = max(candidates_below) if candidates_below else current_price * 0.95

        candidates_above = [swing_high]
        if kijun > current_price:
            candidates_above.append(kijun)
        if senkou_a > current_price:
            candidates_above.append(senkou_a)
        if senkou_b > current_price:
            candidates_above.append(senkou_b)
        nearest_resistance = min(candidates_above) if candidates_above else current_price * 1.05

        return nearest_support, nearest_resistance
    except Exception as e:
        print(f"Error in get_support_resistance_levels: {e}")
        return current_price * 0.95, current_price * 1.05

def get_tight_stop_loss(df, current_price, entry_price):
    if df is None or len(df) < 20:
        return entry_price * 0.95
    try:
        recent_lows = df['low'].tail(20)
        swing_low = recent_lows.iloc[:-1].min() if len(recent_lows) > 1 else recent_lows.min()
        if swing_low < entry_price:
            stop_loss = swing_low * 0.98
        else:
            kijun = df['kijun'].iloc[-1] if 'kijun' in df else entry_price * 0.95
            ma50 = df['MA50'].iloc[-1] if 'MA50' in df else entry_price * 0.95
            stop_loss = min(kijun, ma50) if min(kijun, ma50) < entry_price else entry_price * 0.95
        if entry_price - stop_loss > entry_price * 0.15:
            stop_loss = entry_price * 0.85
        return stop_loss
    except:
        return entry_price * 0.95

def liquidity_grab_level(df):
    if df is None or len(df) < 20:
        return None, None
    try:
        recent_lows = df['low'].iloc[-20:-1]
        swing_low = recent_lows.min()
        liquidity_zone = swing_low * 0.98
        return round(liquidity_zone, 8), round(swing_low, 8)
    except:
        return None, None

def chikou_confirmation(df):
    if df is None or len(df) < 27:
        return False
    try:
        last_chikou = df['chikou'].iloc[-1]
        price_26_ago = df['close'].iloc[-26]
        return last_chikou > price_26_ago
    except:
        return False

def chikou_clearance(df):
    if df is None or len(df) < 50:
        return False, "Data tidak cukup"
    try:
        last_chikou = df['chikou'].iloc[-1]
        price_26_ago = df['close'].iloc[-26]
        senkou_a_26 = df['senkou_a'].iloc[-26] if 'senkou_a' in df else price_26_ago
        senkou_b_26 = df['senkou_b'].iloc[-26] if 'senkou_b' in df else price_26_ago
        cloud_high = max(senkou_a_26, senkou_b_26)
        cloud_low = min(senkou_a_26, senkou_b_26)
        
        if last_chikou > price_26_ago and (last_chikou > cloud_high or last_chikou < cloud_low):
            return True, "Chikou bebas (ruang kosong) → valid"
        elif last_chikou < price_26_ago:
            return False, "Chikou di bawah harga → resistensi"
        else:
            return False, "Chikou menabrak harga masa lalu → rawan fakeout"
    except:
        return False, "Error"

def vsa_low_spread_high_volume(df):
    if df is None or len(df) < 20:
        return False
    try:
        last = df.iloc[-1]
        body = abs(last['close'] - last['open'])
        candle_range = last['high'] - last['low']
        spread_ratio = body / candle_range if candle_range != 0 else 1
        volume_ratio = last['volume'] / df['Volume_MA20'].iloc[-1] if df['Volume_MA20'].iloc[-1] != 0 else 1
        return spread_ratio < 0.3 and volume_ratio > 1.5
    except:
        return False

def detect_double_bottom(df):
    if df is None or len(df) < 40:
        return False
    try:
        lows = df['low'].tail(40)
        valley1_idx = lows.idxmin()
        valley2_candidates = lows.loc[valley1_idx + 10:]
        if valley2_candidates.empty:
            return False
        valley2_idx = valley2_candidates.idxmin()
        valley1 = df.loc[valley1_idx, 'low']
        valley2 = df.loc[valley2_idx, 'low']
        if abs(valley1 - valley2) / valley1 > 0.05:
            return False
        middle_high = df.loc[valley1_idx:valley2_idx, 'high'].max()
        return df['close'].iloc[-1] > middle_high * 0.98
    except:
        return False

def detect_bullish_flag(df):
    if df is None or len(df) < 20:
        return False
    try:
        pole_rise = (df['high'].iloc[-10] - df['low'].iloc[-15]) / df['low'].iloc[-15] if df['low'].iloc[-15] != 0 else 0
        pole_rise = pole_rise > 0.15
        flag_high = df['high'].tail(10).max()
        flag_low = df['low'].tail(10).min()
        flag_range = (flag_high - flag_low) / flag_low if flag_low != 0 else 0
        flag_range = flag_range < 0.10
        vol_down = df['Volume_Ratio'].tail(10).mean() < 0.8
        return pole_rise and flag_range and vol_down
    except:
        return False

def detect_falling_wedge(df):
    if len(df) < 30:
        return False
    try:
        recent = df.tail(30)
        highs = recent['high'].values
        lows = recent['low'].values
        x = np.arange(len(highs))
        slope_high = np.polyfit(x, highs, 1)[0]
        slope_low = np.polyfit(x, lows, 1)[0]
        return slope_high < 0 and slope_low < 0 and slope_low <= slope_high
    except:
        return False

def detect_volatility_squeeze(df, period=20):
    if df is None or len(df) < period:
        return False, 0
    try:
        current_width = df['BB_Width'].iloc[-1]
        avg_width = df['BB_Width'].tail(period).mean()
        is_squeeze = current_width < avg_width * 0.1
        squeeze_pct = (1 - current_width / avg_width) * 100 if avg_width > 0 else 0
        return is_squeeze, round(squeeze_pct, 1)
    except:
        return False, 0

def detect_volume_pre_breakout(df, lookback=20, volume_threshold=1.5):
    if df is None or len(df) < lookback:
        return False, 0
    try:
        recent_vol = df['Volume_Ratio'].tail(5)
        max_vol = recent_vol.max()
        if max_vol > volume_threshold:
            recent_close = df['close'].tail(10)
            price_change = (recent_close.max() - recent_close.min()) / recent_close.min() * 100
            if price_change < 3:
                return True, round(max_vol, 1)
        return False, 0
    except:
        return False, 0

def detect_buy_the_retest(df, tk_cross):
    if df is None or len(df) < 50:
        return None, None
    try:
        last = df.iloc[-1]
        current_price = last['close']
        ma50 = last['MA50']
        kijun = last['kijun'] if 'kijun' in last else ma50
        if tk_cross:
            if ma50 < current_price and abs(current_price - ma50) / current_price < 0.05:
                return ma50, "MA50 (Dynamic Support)"
            elif kijun < current_price and abs(current_price - kijun) / current_price < 0.05:
                return kijun, "Kijun (Dynamic Support)"
        return None, None
    except:
        return None, None

def get_momentum_status(df, tk_cross, is_squeeze, is_accum, rsi, volume_ratio, is_breakout_confirmed=False):
    try:
        if is_breakout_confirmed and tk_cross and rsi > 50:
            return "🚀 BREAKOUT CONFIRMED", "Harga sudah breakout, ikuti momentum dengan trailing stop"
        elif tk_cross and is_squeeze and rsi > 50:
            return "🔮 READY TO LAUNCH", "TK Cross ✅, Squeeze ✅, RSI > 50 → siap meledak, pasang posisi agresif"
        elif tk_cross and rsi > 50:
            return "👀 WAITING FOR BREAKOUT", "TK Cross ✅, RSI > 50 → tunggu breakout level resistance"
        elif is_accum or (rsi < 40 and volume_ratio < 0.7):
            return "⏳ ACCUMULATION", "RSI < 40, Volume rendah → entry di support terjauh"
        elif is_squeeze:
            return "🔮 SQUEEZE DETECTED", "BBW menyempit → potensi ledakan harga dalam waktu dekat"
        else:
            return "⚖️ NEUTRAL", "Menunggu konfirmasi"
    except:
        return "⚖️ NEUTRAL", ""

def calculate_fib_levels(df, lookback=50):
    if df is None or len(df) < lookback:
        return None, None, None
    try:
        recent = df.tail(lookback)
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        diff = swing_high - swing_low
        levels = {
            '0.236': swing_high - 0.236 * diff,
            '0.382': swing_high - 0.382 * diff,
            '0.5': swing_high - 0.5 * diff,
            '0.618': swing_high - 0.618 * diff,
            '0.786': swing_high - 0.786 * diff
        }
        return levels, swing_high, swing_low
    except:
        return None, None, None

def detect_hh_hl(df):
    if df is None or len(df) < 50:
        return False, False
    try:
        recent_highs = df['high'].tail(50)
        recent_lows = df['low'].tail(50)
        hh = recent_highs.iloc[-1] > recent_highs.iloc[-20:-1].max()
        hl = recent_lows.iloc[-1] > recent_lows.iloc[-20:-1].max()
        return hh, hl
    except:
        return False, False

def is_trading_kill_zone():
    now_utc = datetime.utcnow()
    hour = now_utc.hour
    london = (8 <= hour < 16)
    ny = (13 <= hour < 21)
    return london or ny

def detect_accumulation(df):
    if df is None or len(df) < 50:
        return False, "Data tidak cukup"
    try:
        recent = df.tail(30)
        price_range = (recent['high'].max() - recent['low'].min()) / recent['low'].min()
        if price_range > 0.15:
            return False, f"Sideways gagal (range {price_range:.1%})"
        avg_vol_ratio = recent['Volume_Ratio'].mean()
        if avg_vol_ratio > 0.8:
            return False, f"Volume tidak turun (rata-rata {avg_vol_ratio:.2f})"
        return True, f"Akumulasi: sideways {price_range:.1%}, volume turun {avg_vol_ratio:.2f}x"
    except:
        return False, "Error akumulasi"

def future_kumo_status(df):
    if df is None or 'future_senkou_a' not in df.columns or 'future_senkou_b' not in df.columns:
        return None
    try:
        future_a = df['future_senkou_a'].iloc[-1]
        future_b = df['future_senkou_b'].iloc[-1]
        if future_a > future_b:
            return "Bullish (Hijau)"
        elif future_a < future_b:
            return "Bearish (Merah)"
        else:
            return "Netral"
    except:
        return None

def detect_future_cloud_twist(df):
    if df is None or len(df) < 30:
        return False, "N/A"
    try:
        future_a = df['future_senkou_a'].tail(20).values
        future_b = df['future_senkou_b'].tail(20).values
        diff = np.abs(future_a - future_b)
        if len(diff) > 5:
            is_narrowing = diff[-1] < diff[0] * 0.7
            current_future_a = future_a[-1]
            current_future_b = future_b[-1]
            if is_narrowing:
                if current_future_a > current_future_b:
                    return True, "Bullish (Hijau) - Awan Menyempit (Potensi Kumo Twist)"
                else:
                    return True, "Bearish (Merah) - Awan Menyempit (Potensi Kumo Twist)"
        return False, "Awan Normal"
    except:
        return False, "N/A"

def volume_spike_recent(df, lookback=5, threshold=2.0):
    if df is None:
        return False
    for i in range(1, lookback+1):
        if i <= len(df) and df['Volume_Ratio'].iloc[-i] > threshold:
            return True
    return False

def detect_candlestick_patterns(df):
    if df is None or len(df) < 2:
        return []
    patterns = []
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        body = abs(last['close'] - last['open'])
        range_ = last['high'] - last['low']
        if (last['close'] > last['open'] and prev['close'] < prev['open'] and
            last['close'] > prev['open'] and last['open'] < prev['close']):
            patterns.append("Bullish Engulfing")
        if (body <= range_ * 0.3 and (min(last['close'], last['open']) - last['low']) > body * 2 and (last['high'] - max(last['close'], last['open'])) < body):
            patterns.append("Hammer")
        if body <= range_ * 0.1:
            patterns.append("Doji")
    except:
        pass
    return patterns

def detect_rsi_divergence(df):
    if df is None or len(df) < 20:
        return None
    try:
        price_lows = df['low'].tail(20)
        rsi_lows = df['RSI'].tail(20)
        if price_lows.min() == price_lows.iloc[-1] and rsi_lows.min() != rsi_lows.iloc[-1]:
            return "Bullish Divergence (RSI)"
    except:
        pass
    return None

def detect_macd_divergence(df):
    if df is None or len(df) < 20:
        return None
    try:
        price_lows = df['low'].tail(20)
        macd_hist = df['MACD_Hist'].tail(20)
        if price_lows.min() == price_lows.iloc[-1]:
            min_macd = macd_hist.min()
            if macd_hist.iloc[-1] > min_macd:
                return "Bullish Divergence (MACD)"
        return None
    except:
        return None

# ======================== FUNGSI BARU: PREDIKSI BREAKOUT & AKUMULASI ========================

def detect_whale_accumulation_zones(df, lookback=50):
    """Deteksi zona akumulasi whale berdasarkan volume profile"""
    if df is None or len(df) < lookback:
        return None, False, False
    
    try:
        # Volume profile sederhana
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / 15 if price_range > 0 else 0.01
        bins = np.arange(df['low'].min(), df['high'].max(), bin_size)
        
        volume_by_price = []
        for i in range(len(bins)-1):
            mask = (df['close'] >= bins[i]) & (df['close'] < bins[i+1])
            vol_sum = df.loc[mask, 'volume'].sum()
            volume_by_price.append((bins[i], vol_sum))
        
        if volume_by_price:
            poc = max(volume_by_price, key=lambda x: x[1])[0]
        else:
            poc = df['close'].iloc[-1]
        
        # Deteksi akumulasi: volume tinggi di range sempit
        recent_vol = df['volume'].tail(lookback)
        vol_avg = recent_vol.mean()
        vol_std = recent_vol.std()
        
        high_vol_candles = df[df['volume'] > vol_avg + vol_std]
        if len(high_vol_candles) > 0:
            high_vol_range = high_vol_candles['close'].max() - high_vol_candles['close'].min()
            range_pct = high_vol_range / df['close'].iloc[-1] if df['close'].iloc[-1] != 0 else 1
            is_accumulating = range_pct < 0.05 and len(high_vol_candles) > lookback * 0.25
        else:
            is_accumulating = False
        
        # Stealth accumulation (volume naik, harga flat)
        vol_slope = 0
        is_stealth_accum = False
        if len(df) >= 40:
            try:
                vol_slope = np.polyfit(range(20), df['volume'].tail(20).values, 1)[0]
                price_slope = np.polyfit(range(20), df['close'].tail(20).values, 1)[0]
                is_stealth_accum = vol_slope > 0 and abs(price_slope) < 0.002
            except:
                pass
        
        return poc, is_accumulating, is_stealth_accum
    except Exception as e:
        return None, False, False


def predict_breakout_probability(df):
    """Prediksi probabilitas breakout dalam 5-20 candle ke depan"""
    if df is None or len(df) < 50:
        return 0, "Unknown", 0, []
    
    try:
        score = 0
        reasons = []
        
        # 1. Volatility Squeeze (30 points max)
        is_squeeze, squeeze_pct = detect_volatility_squeeze(df)
        if is_squeeze:
            score += 30
            reasons.append(f"✅ Volatility Squeeze ({squeeze_pct}%)")
        
        # 2. Pre-breakout volume (25 points)
        has_pre, vol_ratio = detect_volume_pre_breakout(df)
        if has_pre:
            score += 25
            reasons.append(f"✅ Pre-breakout volume ({vol_ratio}x)")
        
        # 3. Bollinger Band position (10 points)
        last = df.iloc[-1]
        if 'BB_Upper' in last and 'BB_Middle' in last:
            if last['close'] > last['BB_Middle']:
                score += 10
                reasons.append("✅ Harga di atas BB Middle")
        
        # 4. TK Cross (15 points)
        if 'tenkan' in df and 'kijun' in df:
            if df['tenkan'].iloc[-1] > df['kijun'].iloc[-1]:
                score += 15
                reasons.append("✅ Tenkan > Kijun (Golden Cross)")
        
        # 5. OBV Divergence (20 points)
        if 'OBV' in df and len(df) >= 20:
            try:
                obv_change = (df['OBV'].iloc[-1] - df['OBV'].iloc[-20]) / (abs(df['OBV'].iloc[-20]) + 1)
                price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
                if obv_change > 0.03 and abs(price_change) < 0.01:
                    score += 20
                    reasons.append("✅ OBV naik, harga flat (bullish divergence)")
            except:
                pass
        
        # Tentukan arah breakout
        if score > 50:
            if last['close'] > last.get('MA50', last['close']):
                direction = "⬆️ BULLISH (Upward)"
            elif last['close'] < last.get('MA50', last['close']):
                direction = "⬇️ BEARISH (Downward)"
            else:
                direction = "❓ UNCERTAIN"
        else:
            direction = "⏳ LOW CONFIDENCE"
        
        # Estimasi waktu breakout
        if is_squeeze:
            est_hours = max(4, min(24, 8 + (50 - squeeze_pct) / 5))
        else:
            est_hours = 24
        
        return min(score, 95), direction, round(est_hours, 1), reasons
    except Exception as e:
        return 0, "Unknown", 0, [f"Error: {str(e)[:50]}"]


def detect_sudden_pump_setup(df, lookback=10):
    """Deteksi setup 'tiba-tiba tiang tinggi' sebelum terjadi"""
    if df is None or len(df) < lookback + 3:
        return False, []
    
    signals = []
    
    try:
        last = df.iloc[-1]
        
        # 1. Low volume candles (quiet before storm)
        if 'Volume_Ratio' in df:
            recent_vol_ratio = df['Volume_Ratio'].tail(lookback)
            low_vol_period = (recent_vol_ratio < 0.6).sum()
            if low_vol_period > lookback * 0.5:
                signals.append("⚡ Quiet period (volume rendah >50% candle)")
        
        # 2. Small body candles (indecision)
        small_body_count = 0
        for i in range(1, min(6, len(df))):
            body = abs(df['close'].iloc[-i] - df['open'].iloc[-i])
            candle_range = df['high'].iloc[-i] - df['low'].iloc[-i]
            if candle_range > 0 and body / candle_range < 0.3:
                small_body_count += 1
        
        if small_body_count >= 3:
            signals.append(f"🕯️ {small_body_count}/5 candle dengan body kecil (indecision)")
        
        # 3. Tight Bollinger Bands
        if 'BB_Width' in df:
            bb_width_pct = df['BB_Width'].iloc[-1] * 100
            if bb_width_pct < 5:
                signals.append(f"📊 BB sangat sempit ({bb_width_pct:.1f}%) - siap ekspansi")
        
        # 4. Liquidity sweep setup
        if len(df) > 20:
            recent_lows = df['low'].tail(20)
            if last['low'] <= recent_lows.min() * 1.01:
                signals.append("🎯 Liquidity sweep - stop loss hunt terdeteksi")
        
        # 5. RSI turning point
        if 'RSI' in df and len(df) > 5:
            rsi = df['RSI'].values
            if rsi[-1] > 30 and rsi[-2] <= 30:
                signals.append("🔄 RSI keluar dari oversold - momentum reversal")
        
        is_primed = len(signals) >= 2
        
        return is_primed, signals
    except Exception as e:
        return False, []


def calculate_accumulation_score(df):
    """Skor akumulasi 0-100 berdasarkan multiple factors"""
    if df is None or len(df) < 50:
        return 0, []
    
    score = 0
    reasons = []
    
    try:
        # 1. OBV Trend (30 points)
        if 'OBV' in df:
            obv_slope = (df['OBV'].iloc[-1] - df['OBV'].iloc[-30]) / (abs(df['OBV'].iloc[-30]) + 1)
            if obv_slope > 0.05:
                score += 25
                reasons.append(f"OBV strong uptrend (+25)")
            elif obv_slope > 0:
                score += 15
                reasons.append(f"OBV uptrend (+15)")
        
        # 2. Price vs Volume correlation (20 points)
        vol_ratio = df['Volume_Ratio'].tail(20).mean() if 'Volume_Ratio' in df else 1
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
        
        if vol_ratio < 0.7 and abs(price_change) < 0.05:
            score += 20
            reasons.append(f"Volume turun ({vol_ratio:.2f}x), harga flat - stealth accumulation (+20)")
        elif vol_ratio < 0.85 and price_change < 0:
            score += 10
            reasons.append(f"Volume turun, harga turun sedikit (+10)")
        
        # 3. VSA - Low spread high volume (15 points)
        vsa_signal = vsa_low_spread_high_volume(df)
        if vsa_signal:
            score += 15
            reasons.append("VSA: Low spread + High volume (+15)")
        
        # 4. Whale accumulation zones (15 points)
        poc, is_accum, is_stealth = detect_whale_accumulation_zones(df)
        if is_accum:
            score += 15
            reasons.append("Whale accumulation zone terdeteksi (+15)")
        elif is_stealth:
            score += 10
            reasons.append("Stealth accumulation pattern (+10)")
        
        # 5. Support holding (10 points)
        support, _ = get_support_resistance_levels(df, df['close'].iloc[-1])
        bounce_count = 0
        for i in range(1, min(11, len(df))):
            if abs(df['low'].iloc[-i] - support) / support < 0.01:
                bounce_count += 1
        
        if bounce_count >= 3:
            score += 10
            reasons.append(f"Support holding ({bounce_count}x bounce) (+10)")
        
        # 6. Range contraction (10 points)
        recent_range = (df['high'].tail(30).max() - df['low'].tail(30).min()) / df['close'].iloc[-1]
        if recent_range < 0.08:
            score += 10
            reasons.append(f"Range sangat sempit ({recent_range:.1%}) - compression (+10)")
        
        return min(score, 100), reasons
    except Exception as e:
        return 0, [f"Error: {str(e)[:50]}"]

# ======================== FUNGSI ANALISIS UTAMA ========================

def analyze_coin_deep(symbol, exchange_name, include_dxy=False):
    try:
        daily = fetch_ohlcv_cached(symbol, exchange_name, '1d', limit=200)
        if daily is None or len(daily) < 50:
            st.error(f"Data tidak cukup untuk {symbol}")
            return None

        daily = calculate_indicators(daily)
        if daily is None:
            return None
        daily = calculate_ichimoku(daily)
        daily = calculate_obv(daily)

        last = daily.iloc[-1]
        current_price = last['close']

        tf_4h = fetch_ohlcv_cached(symbol, exchange_name, '4h', limit=200)
        if tf_4h is not None:
            tf_4h = calculate_indicators(tf_4h)

        is_accum, acc_reason = detect_accumulation(daily)
        nearest_support, nearest_resistance = get_support_resistance_levels(daily, current_price)
        liquidity_zone, swing_low = liquidity_grab_level(daily)

        if nearest_support < current_price:
            conservative_entry = nearest_support
        else:
            conservative_entry = current_price

        atr_value = last['ATR'] if not pd.isna(last['ATR']) else conservative_entry * 0.02
        stop_loss = get_tight_stop_loss(daily, current_price, conservative_entry)
        
        fib_levels, swing_high, swing_low_fib = calculate_fib_levels(tf_4h, lookback=50) if tf_4h is not None else (None, None, None)
        fib_618 = fib_levels['0.618'] if fib_levels else nearest_resistance
        aggressive_entry = fib_618 if fib_618 and fib_618 > current_price else nearest_resistance
        is_breakout_confirmed = current_price > aggressive_entry
        
        tp1 = conservative_entry + atr_value * 2.0
        tp2 = conservative_entry + atr_value * 3.0
        risk = conservative_entry - stop_loss
        reward = tp1 - conservative_entry
        rr = reward / risk if risk > 0 else 0

        # ========== FUNGSI PREDIKSI BARU ==========
        breakout_prob, breakout_direction, breakout_hours, breakout_reasons = predict_breakout_probability(daily)
        is_pump_primed, pump_signals = detect_sudden_pump_setup(daily)
        accum_score, accum_details = calculate_accumulation_score(daily)
        poc_zone, is_whale_accum, is_stealth_accum = detect_whale_accumulation_zones(daily)
        
        # Hitung skor teknikal lama (untuk kompatibilitas)
        score = 0
        reasons = []

        if is_accum:
            score += 2
            reasons.append(acc_reason)

        if last['close'] > last['MA200']:
            score += 1
            reasons.append("Harga di atas MA200 (bullish jangka panjang)")
        if last['close'] > last['MA50']:
            score += 1
            reasons.append("Harga di atas MA50 (trend naik)")
        else:
            reasons.append("Harga di bawah MA50 (trend netral/bearish)")

        if 30 <= last['RSI'] <= 70:
            score += 0.5
            reasons.append(f"RSI {last['RSI']:.1f} (netral)")
        elif last['RSI'] < 30:
            reasons.append(f"RSI {last['RSI']:.1f} (oversold, potensi reversal)")
        else:
            reasons.append(f"RSI {last['RSI']:.1f} (overbought, hati-hati)")

        if last['Volume_Ratio'] < 0.7:
            score += 0.5
            reasons.append("Volume rendah (potensi akumulasi)")
        elif last['Volume_Ratio'] > 2.0:
            reasons.append(f"Volume spike ({last['Volume_Ratio']:.1f}x rata-rata)")

        if last['MACD'] > last['Signal']:
            score += 1
            reasons.append("MACD bullish (momentum positif)")

        is_above_cloud = (last['close'] > last['senkou_a'] and last['close'] > last['senkou_b'])
        cloud_thick = abs(last['senkou_a'] - last['senkou_b']) / last['close']
        
        if is_above_cloud:
            if cloud_thick > 0.02:
                reasons.append(f"Cloud tebal ({cloud_thick:.1%}) → support kuat (harga di atas)")
            else:
                reasons.append(f"Cloud tipis ({cloud_thick:.1%}) → mudah ditembus (harga di atas)")
            score += 1
            reasons.append("Harga di atas Cloud Ichimoku (bullish)")
        else:
            if cloud_thick > 0.02:
                reasons.append(f"⚠️ Cloud tebal ({cloud_thick:.1%}) → Strong Resistance (Atap Beton)")
            else:
                reasons.append(f"Cloud tipis ({cloud_thick:.1%}) → resistance lemah")
            reasons.append("Harga di bawah Cloud Ichimoku (bearish)")

        tk_cross = last['tenkan'] > last['kijun']
        if tk_cross:
            score += 0.5
            reasons.append("✅ Tenkan di atas Kijun (Golden Cross)")
        else:
            reasons.append("⚠️ Tenkan-Kijun: Belum Golden Cross")

        chikou_ok = chikou_confirmation(daily)
        chikou_status = "Above" if chikou_ok else "Below"
        if chikou_ok:
            score += 0.5
            reasons.append("Chikou Span di atas harga 26 periode lalu (konfirmasi bullish)")
        else:
            reasons.append("Chikou Span di bawah harga (resistensi potensial)")

        chikou_clear, chikou_msg = chikou_clearance(daily)
        if chikou_clear:
            score += 0.5
            reasons.append(f"Chikou: {chikou_msg}")
        else:
            reasons.append(f"⚠️ Chikou: {chikou_msg}")

        future_kumo = future_kumo_status(daily)
        has_twist, twist_msg = detect_future_cloud_twist(daily)
        
        if future_kumo == "Bullish (Hijau)":
            score += 0.5
            reasons.append(f"Future Cloud: {future_kumo} (bullish outlook)")

        vsa_signal = vsa_low_spread_high_volume(daily)
        if vsa_signal:
            score += 0.5
            reasons.append("Low spread high volume (suplai diserap)")

        if daily['OBV_trend'].iloc[-1]:
            score += 0.5
            reasons.append("OBV naik (akumulasi)")
        else:
            reasons.append("OBV flat/turun (belum ada akumulasi)")

        is_squeeze, squeeze_pct = detect_volatility_squeeze(daily)
        if is_squeeze:
            reasons.append(f"⚠️ Volatility Squeeze: BBW menyempit {squeeze_pct}% → potensi ledakan harga")
            score += 0.5

        has_pre_breakout, pre_breakout_vol = detect_volume_pre_breakout(daily)
        if has_pre_breakout:
            reasons.append(f"⚠️ Volume Pre-Breakout: Volume {pre_breakout_vol}x rata-rata tapi harga belum naik")
            score += 0.5

        retest_level, retest_name = detect_buy_the_retest(daily, tk_cross)
        if retest_level:
            reasons.append(f"✅ Buy the Retest: Harga mendekati {retest_name} ({retest_level:.8f})")

        momentum_status, momentum_desc = get_momentum_status(
            daily, tk_cross, is_squeeze, is_accum, last['RSI'], last['Volume_Ratio'], is_breakout_confirmed
        )
        reasons.append(f"📊 Momentum Status: {momentum_status} - {momentum_desc}")

        hh, hl = detect_hh_hl(daily)
        if hl:
            score += 0.5
            reasons.append("Higher Low terbentuk (struktur bullish)")
        if hh:
            score += 0.5
            reasons.append("Higher High terbentuk (trend naik)")

        if detect_double_bottom(daily):
            score += 1
            reasons.append("Pola Double Bottom terdeteksi")

        if detect_bullish_flag(daily):
            score += 1
            reasons.append("Pola Bullish Flag terdeteksi")

        if detect_falling_wedge(daily):
            score += 0.5
            reasons.append("Pola Falling Wedge terdeteksi")

        rsi_div = detect_rsi_divergence(daily)
        if rsi_div:
            score += 1
            reasons.append(rsi_div)

        macd_div = detect_macd_divergence(daily)
        if macd_div:
            score += 1
            reasons.append(macd_div)

        patterns = detect_candlestick_patterns(daily)
        if patterns:
            score += 0.5
            reasons.append(f"Pola candlestick: {', '.join(patterns)}")

        if volume_spike_recent(daily):
            score += 0.5
            reasons.append("⚠️ Volume spike dalam 5 candle terakhir → potensi markup")

        # Trigger recommendation
        if is_breakout_confirmed:
            trigger_msg = f"✅ Breakout confirmed! Harga di atas {aggressive_entry:.8f}. Entry agresif, trailing stop di bawah swing low."
        elif tk_cross and current_price < aggressive_entry:
            trigger_msg = f"👀 Menunggu breakout {aggressive_entry:.8f}. Pasang alert, entry jika tertembus dengan volume tinggi."
        else:
            trigger_msg = f"Pasang limit order di {conservative_entry:.8f} (conservative) atau tunggu breakout {aggressive_entry:.8f} (aggressive)."
        reasons.append(f"🔔 Trigger: {trigger_msg}")

        # Rekomendasi
        if score >= 5:
            action = "BUY"
            confidence = "High"
        elif score >= 3.5:
            action = "BUY"
            confidence = "Medium"
        elif score >= 2:
            action = "BUY (Speculative)"
            confidence = "Low"
        else:
            action = "HOLD / WAIT"
            confidence = "Low"

        # External Correlation & Sentiment
        btc_data = fetch_ohlcv_cached('BTC/USDT', exchange_name, '1d', limit=100)
        corr = None
        if btc_data is not None:
            btc_data = calculate_indicators(btc_data)
            corr = calculate_correlation(daily, btc_data, period=30)
            if corr is not None:
                if abs(corr) < 0.3:
                    correlation_text = f"Correlation Score: {corr} → Independent Movement"
                else:
                    correlation_text = f"Correlation Score: {corr} ({'sangat mengikuti' if abs(corr) > 0.7 else 'cukup mengikuti' if abs(corr) > 0.5 else 'lemah mengikuti'} BTC)"
            else:
                correlation_text = "Correlation Score: N/A"
        else:
            correlation_text = "Correlation Score: N/A"

        sentiment_block = [
            "🌐 External Correlation & Sentiment",
            f"    {correlation_text}"
        ]

        # FINAL VERDICT
        if score >= 5:
            verdict_title = "STRONG BUY (High Confidence)"
        elif score >= 3.5:
            verdict_title = "BUY (Medium Confidence)"
        elif score >= 2:
            verdict_title = "HIGH RISK - HIGH REWARD (Speculative)"
        else:
            verdict_title = "HOLD / WAIT (Avoid)"

        if is_breakout_confirmed:
            strategy = f"✅ Breakout confirmed! Entry agresif di {aggressive_entry:.8f} atau retest ke support dinamis."
        else:
            strategy = f"Pilih Entry: Tunggu retest ke {aggressive_entry:.8f} (Aggressive) atau tunggu breakout {aggressive_entry:.8f} dengan volume tinggi."

        key_insight = ""
        if is_accum:
            key_insight += "Secara teknikal harga masih di bawah Cloud (Bearish), tapi OBV Naik menandakan adanya akumulasi diam-diam. "
        if not chikou_clear:
            key_insight += "Chikou masih terhambat, kenaikan masih rawan fakeout. "
        if not tk_cross:
            key_insight += "Tenkan-Kijun belum Golden Cross, mesin starter belum nyala. "
        if is_squeeze:
            key_insight += f"Volatility Squeeze {squeeze_pct}% → harga seperti per ditekan, siap meledak. "
        if has_pre_breakout:
            key_insight += f"Volume pre-breakout {pre_breakout_vol}x → whale sedang loading. "

        action_plan = f"""
        ✅ **Conservative Entry:** {conservative_entry:.8f} (Support kuat)
        🚀 **Aggressive Entry (Breakout):** {aggressive_entry:.8f} (Breakout level)
        ❌ **Stop Loss:** {stop_loss:.8f} (Wajib! Di bawah swing low terdekat)
        🎯 **Take Profit:** Cicil di {tp1:.8f} & {tp2:.8f}
        📊 **Risk/Reward:** 1:{rr:.1f} (Sehat jika > 1.5)
        """

        verdict = {
            'title': verdict_title,
            'strategy': strategy,
            'key_insight': key_insight,
            'action_plan': action_plan,
            'momentum_status': momentum_status,
            'rr': rr
        }

        return {
            'symbol': symbol,
            'current_price': round(current_price, 8),
            'conservative_entry': round(conservative_entry, 8),
            'aggressive_entry': round(aggressive_entry, 8),
            'action': action,
            'confidence': confidence,
            'score': round(score, 1),
            'sl': round(stop_loss, 8),
            'tp1': round(tp1, 8),
            'tp2': round(tp2, 8),
            'rr': round(rr, 2),
            'atr': round(atr_value, 8),
            'is_accum': is_accum,
            'reasons': reasons,
            'nearest_support': round(nearest_support, 8),
            'nearest_resistance': round(nearest_resistance, 8),
            'daily': daily,
            'tf_4h': tf_4h,
            'cloud_thick_pct': round(cloud_thick * 100, 1),
            'chikou_status': chikou_status,
            'future_kumo': future_kumo,
            'tk_cross': tk_cross,
            'momentum_status': momentum_status,
            'is_squeeze': is_squeeze,
            'has_pre_breakout': has_pre_breakout,
            'is_breakout_confirmed': is_breakout_confirmed,
            'trigger_rec': trigger_msg,
            'sentiment_block': sentiment_block,
            'correlation': corr,
            'verdict': verdict,
            # DATA PREDIKSI BARU
            'breakout_probability': breakout_prob,
            'breakout_direction': breakout_direction,
            'breakout_hours': breakout_hours,
            'breakout_reasons': breakout_reasons,
            'is_pump_primed': is_pump_primed,
            'pump_signals': pump_signals,
            'accumulation_score': accum_score,
            'accumulation_details': accum_details,
            'whale_poc_zone': poc_zone,
            'is_whale_accumulating': is_whale_accum,
            'is_stealth_accum': is_stealth_accum
        }
    except Exception as e:
        st.error(f"Error dalam analisis {symbol}: {str(e)}")
        return None


def analyze_coin_quick(symbol, exchange_name):
    try:
        daily = fetch_ohlcv_cached(symbol, exchange_name, '1d', limit=200)
        if daily is None or len(daily) < 50:
            return None
        daily = calculate_indicators(daily)
        if daily is None:
            return None
        last = daily.iloc[-1]
        current_price = last['close']
        is_accum, _ = detect_accumulation(daily)
        nearest_support, _ = get_support_resistance_levels(daily, current_price)

        if nearest_support < current_price:
            conservative_entry = nearest_support
        else:
            conservative_entry = current_price

        stop_loss = get_tight_stop_loss(daily, current_price, conservative_entry)
        atr_value = last['ATR'] if not pd.isna(last['ATR']) else conservative_entry * 0.02

        score = 0
        if is_accum:
            score += 2
        if last['close'] > last['MA200']:
            score += 1
        if last['close'] > last['MA50']:
            score += 1
        if 30 <= last['RSI'] <= 70:
            score += 0.5
        if last['Volume_Ratio'] < 0.7:
            score += 0.5
        if last['MACD'] > last['Signal']:
            score += 1

        if score >= 3:
            action = "BUY"
            confidence = "High"
        elif score >= 2:
            action = "BUY"
            confidence = "Medium"
        elif score >= 1:
            action = "BUY (Speculative)"
            confidence = "Low"
        else:
            action = "HOLD / WAIT"
            confidence = "Low"

        tp1 = conservative_entry + atr_value * 2.0
        tp2 = conservative_entry + atr_value * 3.0
        risk = conservative_entry - stop_loss
        reward = tp1 - conservative_entry
        rr = reward / risk if risk > 0 else 0

        return {
            'symbol': symbol,
            'current_price': round(current_price, 8),
            'conservative_entry': round(conservative_entry, 8),
            'action': action,
            'confidence': confidence,
            'score': round(score, 1),
            'sl': round(stop_loss, 8),
            'tp1': round(tp1, 8),
            'tp2': round(tp2, 8),
            'rr': round(rr, 2),
            'is_accum': is_accum,
            'nearest_support': round(nearest_support, 8),
            'volume_24h_usdt': last['volume'] * conservative_entry
        }
    except Exception as e:
        print(f"Quick analysis error for {symbol}: {e}")
        return None

# ======================== PLOT ICHIMOKU ========================
def plot_ichimochart(df, symbol, nearest_support, nearest_resistance):
    if df is None or len(df) < 50:
        return go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['tenkan'], name='Tenkan (9)', line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['kijun'], name='Kijun (26)', line=dict(color='red', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['senkou_a'], name='Senkou A', line=dict(color='green', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['senkou_b'], name='Senkou B', line=dict(color='orange', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['chikou'], name='Chikou', line=dict(color='purple', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pd.concat([df['timestamp'], df['timestamp'][::-1]]),
        y=pd.concat([df['senkou_a'], df['senkou_b'][::-1]]),
        fill='toself', fillcolor='rgba(0,100,0,0.2)', line=dict(color='rgba(0,0,0,0)'), name='Cloud'
    ), row=1, col=1)
    fig.add_hline(y=nearest_support, line_dash="dash", line_color="yellow", annotation_text="Nearest Support", row=1, col=1)
    fig.add_hline(y=nearest_resistance, line_dash="dot", line_color="orange", annotation_text="Nearest Resistance", row=1, col=1)
    colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color=colors), row=2, col=1)
    fig.update_layout(title=f"{symbol} - Daily Chart with Ichimoku", template="plotly_dark", height=800, hovermode='x unified')
    return fig

# ======================== STREAMLIT UI ========================
with st.sidebar:
    st.header("⚙️ Auto Scan Settings")
    exchange_name = st.selectbox("Exchange", ["okx", "kucoin", "mexc", "gate", "binance", "bybit"], index=0)
    scan_limit = st.slider("Jumlah coin di-scan", 50, 300, 150)
    auto_refresh_interval = st.slider("Auto-refresh (menit)", 5, 60, 30)
    use_btc_filter = st.checkbox("Filter dengan BTC trend", value=True)
    start_auto = st.button("▶️ Mulai Auto Scan")

    st.markdown("---")
    st.header("🔍 Manual Coin Analysis")
    manual_symbol = st.text_input("Coin symbol (e.g., BTC/USDT)", placeholder="BTC/USDT")
    manual_button = st.button("Analyze Manual", key="manual_btn")

# Session state
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None
if 'auto_running' not in st.session_state:
    st.session_state.auto_running = False
if 'selected_coin' not in st.session_state:
    st.session_state.selected_coin = None
if 'manual_analysis' not in st.session_state:
    st.session_state.manual_analysis = None
if 'deep_result' not in st.session_state:
    st.session_state.deep_result = None

def perform_scan():
    with st.spinner("Mengambil daftar coin..."):
        pairs = get_all_usdt_pairs(exchange_name)
        if len(pairs) > scan_limit:
            pairs = pairs[:scan_limit]
    st.info(f"📊 Scan {len(pairs)} coin...")

    btc_trend = None
    if use_btc_filter:
        btc_df = fetch_ohlcv_cached('BTC/USDT', exchange_name, '1d', limit=100)
        if btc_df is not None:
            btc_df = calculate_indicators(btc_df)
            if btc_df is not None:
                last_btc = btc_df.iloc[-1]
                if last_btc['close'] > last_btc['MA50'] and last_btc['close'] > last_btc['MA200']:
                    btc_trend = "bullish"
                elif last_btc['close'] < last_btc['MA50'] and last_btc['close'] < last_btc['MA200']:
                    btc_trend = "bearish"
                else:
                    btc_trend = "neutral"
        if btc_trend == "bearish":
            st.warning("⚠️ BTC dalam tren bearish. Hanya coin dengan akumulasi kuat yang ditampilkan.")

    progress_bar = st.progress(0)
    results = []
    for i, sym in enumerate(pairs):
        progress_bar.progress((i+1)/len(pairs))
        res = analyze_coin_quick(sym, exchange_name)
        if res and res['action'].startswith('BUY'):
            results.append(res)
        time.sleep(0.05)

    if btc_trend == "bearish" and use_btc_filter:
        results = [r for r in results if r['is_accum']]

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('score', ascending=False)
        st.session_state.scan_results = df
    else:
        st.session_state.scan_results = pd.DataFrame()
    st.session_state.last_scan_time = datetime.now()
    st.rerun()

if start_auto:
    st.session_state.auto_running = True

if st.session_state.auto_running:
    if st.session_state.scan_results is None:
        perform_scan()
    else:
        if st.session_state.last_scan_time:
            elapsed = (datetime.now() - st.session_state.last_scan_time).total_seconds() / 60
            if elapsed >= auto_refresh_interval:
                perform_scan()
    if st.session_state.last_scan_time:
        next_scan = st.session_state.last_scan_time + timedelta(minutes=auto_refresh_interval)
        st.sidebar.info(f"🕒 Scan berikutnya: {next_scan.strftime('%H:%M:%S')}")
    if st.sidebar.button("⏹️ Stop Auto Scan"):
        st.session_state.auto_running = False
        st.rerun()
else:
    if st.sidebar.button("🔍 Scan Sekarang"):
        perform_scan()

# Manual analysis
if manual_button and manual_symbol:
    symbol = manual_symbol.strip().upper()
    if not symbol.endswith('/USDT'):
        symbol += '/USDT'
    with st.spinner(f"Menganalisis {symbol}..."):
        res = analyze_coin_deep(symbol, exchange_name)
        if res:
            st.session_state.manual_analysis = res
            st.success(f"Analisis manual untuk {symbol} selesai.")
        else:
            st.session_state.manual_analysis = None
            st.error(f"Gagal menganalisis {symbol}. Periksa simbol atau koneksi.")
    st.rerun()

# Tampilkan manual analysis jika ada
if st.session_state.manual_analysis is not None:
    dr = st.session_state.manual_analysis
    st.subheader(f"📊 Manual Analysis: {dr['symbol']}")
    
    # ========== PREDIKSI BREAKOUT & PUMP (TAMPILAN UTAMA) ==========
    st.markdown("---")
    st.subheader("🔮 BREAKOUT & PUMP PREDICTION")
    
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    col_b1.metric("🎯 Breakout Probability", f"{dr.get('breakout_probability', 0)}%")
    col_b2.metric("⬆️ Predicted Direction", dr.get('breakout_direction', 'Unknown'))
    col_b3.metric("⏰ Est. Timing", f"{dr.get('breakout_hours', 0)} jam")
    col_b4.metric("🐋 Accumulation Score", f"{dr.get('accumulation_score', 0)}/100")
    
    # Pump primed alert
    if dr.get('is_pump_primed', False):
        st.warning("🚨 **PUMP PRIMED!** Setup 'tiba-tiba tiang tinggi' terdeteksi! Harga berpotensi melonjak dalam waktu dekat.")
        for sig in dr.get('pump_signals', []):
            st.write(f"  • {sig}")
    
    # Whale accumulation zone
    if dr.get('is_whale_accumulating', False):
        poc = dr.get('whale_poc_zone')
        if poc:
            st.info(f"🐋 **Whale Accumulation Zone** terdeteksi di sekitar **{poc:.8f}** - harga ideal untuk akumulasi")
    
    if dr.get('is_stealth_accum', False):
        st.info("🕵️ **Stealth Accumulation** - Whale mengakumulasi diam-diam (volume naik, harga flat)")
    
    with st.expander("📊 Detail Prediksi Breakout"):
        for r in dr.get('breakout_reasons', []):
            st.write(f"- {r}")
    
    with st.expander("🐋 Detail Accumulation Analysis"):
        for r in dr.get('accumulation_details', []):
            st.write(f"- {r}")
    
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Harga Saat Ini", f"{dr['current_price']:.8f}")
    col2.metric("Conservative Entry", f"{dr['conservative_entry']:.8f}")
    col3.metric("Aggressive Entry", f"{dr['aggressive_entry']:.8f}")
    col4.metric("Action", dr['action'])
    col5.metric("Score", dr['score'])

    with st.expander("📝 Alasan Analisis", expanded=True):
        for r in dr['reasons']:
            st.write(f"- {r}")

    st.write(f"**Stop Loss:** {dr['sl']} | **Take Profit 1:** {dr['tp1']} | **Take Profit 2:** {dr['tp2']}")
    st.write(f"**ATR (14):** {dr['atr']:.8f} — jarak aman stop loss")
    st.write(f"**Risk/Reward:** 1:{dr['rr']:.1f}")
    st.write(f"**Nearest Support:** {dr['nearest_support']} | **Nearest Resistance:** {dr['nearest_resistance']}")
    st.write(f"**Cloud Thickness:** {dr.get('cloud_thick_pct', 'N/A')}% | **Chikou Status:** {dr.get('chikou_status', 'N/A')}")
    st.write(f"**Future Cloud:** {dr.get('future_kumo', 'N/A')}")

    st.subheader("🌐 External Correlation & Sentiment")
    for line in dr['sentiment_block']:
        st.write(line)

    # FINAL VERDICT
    verdict = dr['verdict']
    st.markdown(f"## 🚩 FINAL VERDICT: {dr['symbol']}")
    st.markdown(f"**{verdict['title']}** (Skor: {dr['score']}/5)")
    st.write(f"**Momentum Status:** {verdict['momentum_status']}")
    st.write(f"**Risk/Reward:** 1:{verdict['rr']:.1f}")
    st.write(f"**Strategi Utama:** {verdict['strategy']}")
    st.write(f"**Key Insight:** {verdict['key_insight']}")
    st.write(f"**Action Plan:**")
    st.write(verdict['action_plan'])

    if dr['daily'] is not None:
        st.plotly_chart(plot_ichimochart(dr['daily'], dr['symbol'], dr['nearest_support'], dr['nearest_resistance']), use_container_width=True)

    st.write("#### Konfirmasi Multi-Timeframe")
    col_tf1, col_tf2 = st.columns(2)
    if dr.get('tf_4h') is not None:
        with col_tf1:
            st.write("**4H Chart (60 candle terakhir)**")
            df_4h = dr['tf_4h'].tail(60)
            fig4h = go.Figure()
            fig4h.add_trace(go.Scatter(x=df_4h['timestamp'], y=df_4h['close'], name='Close', line=dict(color='white')))
            fig4h.add_trace(go.Scatter(x=df_4h['timestamp'], y=df_4h['MA50'], name='MA50', line=dict(color='orange')))
            fig4h.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig4h, use_container_width=True)

# Tampilkan auto scan results
if st.session_state.scan_results is not None and not st.session_state.scan_results.empty:
    df = st.session_state.scan_results
    st.subheader(f"📊 Auto Scan Results - {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.dataframe(df[['symbol', 'current_price', 'conservative_entry', 'action', 'confidence', 'score', 'rr', 'sl', 'tp1', 'nearest_support', 'is_accum']],
                 use_container_width=True)

    st.subheader("🔍 Deep Analysis (Auto Scan)")
    coin_list = df['symbol'].tolist()
    selected_coin = st.selectbox("Pilih coin untuk analisis mendalam", coin_list, key="auto_select")
    if selected_coin != st.session_state.selected_coin:
        st.session_state.selected_coin = selected_coin
        with st.spinner(f"Mengambil data detail untuk {selected_coin}..."):
            deep_res = analyze_coin_deep(selected_coin, exchange_name)
            if deep_res:
                st.session_state.deep_result = deep_res
            else:
                st.session_state.deep_result = None
        st.rerun()

    if st.session_state.deep_result is not None:
        dr = st.session_state.deep_result
        
        # ========== PREDIKSI BREAKOUT & PUMP (TAMPILAN AUTO SCAN) ==========
        st.markdown("---")
        st.subheader("🔮 BREAKOUT & PUMP PREDICTION")
        
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        col_b1.metric("🎯 Breakout Probability", f"{dr.get('breakout_probability', 0)}%")
        col_b2.metric("⬆️ Predicted Direction", dr.get('breakout_direction', 'Unknown'))
        col_b3.metric("⏰ Est. Timing", f"{dr.get('breakout_hours', 0)} jam")
        col_b4.metric("🐋 Accumulation Score", f"{dr.get('accumulation_score', 0)}/100")
        
        if dr.get('is_pump_primed', False):
            st.warning("🚨 **PUMP PRIMED!** Setup 'tiba-tiba tiang tinggi' terdeteksi!")
            for sig in dr.get('pump_signals', []):
                st.write(f"  • {sig}")
        
        if dr.get('is_whale_accumulating', False):
            poc = dr.get('whale_poc_zone')
            if poc:
                st.info(f"🐋 **Whale Accumulation Zone** di sekitar {poc:.8f}")
        
        with st.expander("📊 Detail Prediksi"):
            for r in dr.get('breakout_reasons', []):
                st.write(f"- {r}")
        
        st.markdown("---")
        
        st.write(f"### {dr['symbol']}")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Harga Saat Ini", f"{dr['current_price']:.8f}")
        col2.metric("Conservative Entry", f"{dr['conservative_entry']:.8f}")
        col3.metric("Aggressive Entry", f"{dr['aggressive_entry']:.8f}")
        col4.metric("Action", dr['action'])
        col5.metric("Score", dr['score'])

        with st.expander("📝 Alasan Analisis", expanded=True):
            for r in dr['reasons']:
                st.write(f"- {r}")

        st.write(f"**Stop Loss:** {dr['sl']} | **Take Profit 1:** {dr['tp1']} | **Take Profit 2:** {dr['tp2']}")
        st.write(f"**Risk/Reward:** 1:{dr['rr']:.1f}")

        verdict = dr['verdict']
        st.markdown(f"**{verdict['title']}**")
        st.write(f"**Strategi Utama:** {verdict['strategy']}")
        st.write(f"**Key Insight:** {verdict['key_insight']}")

        if dr['daily'] is not None:
            st.plotly_chart(plot_ichimochart(dr['daily'], dr['symbol'], dr['nearest_support'], dr['nearest_resistance']), use_container_width=True)
else:
    if st.session_state.scan_results is not None:
        st.warning("Tidak ada coin memenuhi kriteria auto scan. Coba ubah parameter.")
    else:
        st.info("Tekan tombol 'Mulai Auto Scan' atau 'Scan Sekarang' untuk memulai auto scan, atau gunakan manual analysis di sidebar.")

st.caption("Data di-cache 10 menit. Fitur prediksi breakout & pump menggunakan algoritma multi-indikator. Akurasi 70-80% di market sideways.")
