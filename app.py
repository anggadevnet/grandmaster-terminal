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
st.set_page_config(page_title="Crypto Accumulation Scanner", layout="wide")
st.title("🐋 Crypto Accumulation Scanner + Breakout Predictor")
st.markdown("Auto scan coin yang sedang diakumulasi whale + prediksi breakout & pump")

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
            return "🔮 READY TO LAUNCH", "TK Cross ✅, Squeeze ✅, RSI > 50 → siap meledak"
        elif tk_cross and rsi > 50:
            return "👀 WAITING FOR BREAKOUT", "TK Cross ✅, RSI > 50 → tunggu breakout"
        elif is_accum or (rsi < 40 and volume_ratio < 0.7):
            return "⏳ ACCUMULATION", "RSI < 40, Volume rendah → entry di support"
        elif is_squeeze:
            return "🔮 SQUEEZE DETECTED", "BBW menyempit → potensi ledakan harga"
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

# ======================== FUNGSI PREDIKSI BARU ========================

def detect_whale_accumulation_zones(df, lookback=50):
    if df is None or len(df) < lookback:
        return None, False, False
    try:
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
        
        is_stealth_accum = False
        if len(df) >= 40:
            try:
                vol_slope = np.polyfit(range(20), df['volume'].tail(20).values, 1)[0]
                price_slope = np.polyfit(range(20), df['close'].tail(20).values, 1)[0]
                is_stealth_accum = vol_slope > 0 and abs(price_slope) < 0.002
            except:
                pass
        
        return poc, is_accumulating, is_stealth_accum
    except:
        return None, False, False

def predict_breakout_probability(df):
    if df is None or len(df) < 50:
        return 0, "Unknown", 0, []
    try:
        score = 0
        reasons = []
        
        is_squeeze, squeeze_pct = detect_volatility_squeeze(df)
        if is_squeeze:
            score += 30
            reasons.append(f"Volatility Squeeze ({squeeze_pct}%)")
        
        has_pre, vol_ratio = detect_volume_pre_breakout(df)
        if has_pre:
            score += 25
            reasons.append(f"Pre-breakout volume ({vol_ratio}x)")
        
        last = df.iloc[-1]
        if 'BB_Upper' in last and last['close'] > last['BB_Middle']:
            score += 10
            reasons.append("Harga di atas BB Middle")
        
        if 'tenkan' in df and df['tenkan'].iloc[-1] > df['kijun'].iloc[-1]:
            score += 15
            reasons.append("Tenkan > Kijun (Golden Cross)")
        
        if 'OBV' in df and len(df) >= 20:
            obv_change = (df['OBV'].iloc[-1] - df['OBV'].iloc[-20]) / (abs(df['OBV'].iloc[-20]) + 1)
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            if obv_change > 0.03 and abs(price_change) < 0.01:
                score += 20
                reasons.append("OBV naik, harga flat (bullish divergence)")
        
        if score > 50:
            direction = "⬆️ BULLISH" if last['close'] > last.get('MA50', last['close']) else "⬇️ BEARISH"
        else:
            direction = "LOW CONFIDENCE"
        
        est_hours = 24 if not is_squeeze else max(4, min(24, 8 + (50 - squeeze_pct) / 5))
        
        return min(score, 95), direction, round(est_hours, 1), reasons
    except:
        return 0, "Unknown", 24, []

def detect_sudden_pump_setup(df, lookback=10):
    if df is None or len(df) < lookback + 3:
        return False, []
    signals = []
    try:
        last = df.iloc[-1]
        
        if 'Volume_Ratio' in df:
            low_vol = (df['Volume_Ratio'].tail(lookback) < 0.6).sum()
            if low_vol > lookback * 0.5:
                signals.append("Quiet period (volume rendah)")
        
        small_body = 0
        for i in range(1, min(6, len(df))):
            body = abs(df['close'].iloc[-i] - df['open'].iloc[-i])
            candle_range = df['high'].iloc[-i] - df['low'].iloc[-i]
            if candle_range > 0 and body / candle_range < 0.3:
                small_body += 1
        if small_body >= 3:
            signals.append(f"{small_body}/5 candle dengan body kecil")
        
        if 'BB_Width' in df and df['BB_Width'].iloc[-1] * 100 < 5:
            signals.append("BB sangat sempit - siap ekspansi")
        
        if len(df) > 20 and last['low'] <= df['low'].tail(20).min() * 1.01:
            signals.append("Liquidity sweep terdeteksi")
        
        return len(signals) >= 2, signals
    except:
        return False, []

def calculate_accumulation_score(df):
    if df is None or len(df) < 50:
        return 0, []
    score = 0
    reasons = []
    try:
        if 'OBV' in df:
            obv_slope = (df['OBV'].iloc[-1] - df['OBV'].iloc[-30]) / (abs(df['OBV'].iloc[-30]) + 1)
            if obv_slope > 0.05:
                score += 25
                reasons.append("OBV strong uptrend (+25)")
            elif obv_slope > 0:
                score += 15
                reasons.append("OBV uptrend (+15)")
        
        vol_ratio = df['Volume_Ratio'].tail(20).mean() if 'Volume_Ratio' in df else 1
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
        if vol_ratio < 0.7 and abs(price_change) < 0.05:
            score += 20
            reasons.append(f"Volume turun ({vol_ratio:.2f}x), harga flat (+20)")
        
        if vsa_low_spread_high_volume(df):
            score += 15
            reasons.append("VSA: Low spread + High volume (+15)")
        
        _, is_accum, is_stealth = detect_whale_accumulation_zones(df)
        if is_accum:
            score += 15
            reasons.append("Whale accumulation zone (+15)")
        elif is_stealth:
            score += 10
            reasons.append("Stealth accumulation pattern (+10)")
        
        support, _ = get_support_resistance_levels(df, df['close'].iloc[-1])
        bounce = sum(1 for i in range(1, min(11, len(df))) if abs(df['low'].iloc[-i] - support) / support < 0.01)
        if bounce >= 3:
            score += 10
            reasons.append(f"Support holding ({bounce}x bounce) (+10)")
        
        recent_range = (df['high'].tail(30).max() - df['low'].tail(30).min()) / df['close'].iloc[-1]
        if recent_range < 0.08:
            score += 10
            reasons.append(f"Range sempit ({recent_range:.1%}) (+10)")
        
        return min(score, 100), reasons
    except:
        return 0, []

# ======================== FUNGSI ANALISIS UTAMA ========================

def analyze_coin_deep(symbol, exchange_name):
    try:
        daily = fetch_ohlcv_cached(symbol, exchange_name, '1d', limit=200)
        if daily is None or len(daily) < 50:
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

        conservative_entry = nearest_support if nearest_support < current_price else current_price
        atr_value = last['ATR'] if not pd.isna(last['ATR']) else conservative_entry * 0.02
        stop_loss = get_tight_stop_loss(daily, current_price, conservative_entry)
        
        fib_levels, _, _ = calculate_fib_levels(tf_4h, lookback=50) if tf_4h is not None else (None, None, None)
        fib_618 = fib_levels['0.618'] if fib_levels else nearest_resistance
        aggressive_entry = fib_618 if fib_618 and fib_618 > current_price else nearest_resistance
        is_breakout_confirmed = current_price > aggressive_entry
        
        tp1 = conservative_entry + atr_value * 2.0
        tp2 = conservative_entry + atr_value * 3.0
        rr = (tp1 - conservative_entry) / (conservative_entry - stop_loss) if conservative_entry - stop_loss > 0 else 0

        # PREDIKSI BARU
        breakout_prob, breakout_dir, breakout_hours, breakout_reasons = predict_breakout_probability(daily)
        is_pump_primed, pump_signals = detect_sudden_pump_setup(daily)
        accum_score, accum_details = calculate_accumulation_score(daily)
        poc_zone, is_whale_accum, is_stealth_accum = detect_whale_accumulation_zones(daily)
        
        # SKOR TEKNIKAL
        score = 0
        reasons = []

        if is_accum:
            score += 2
            reasons.append(acc_reason)
        if last['close'] > last['MA200']:
            score += 1
            reasons.append("Harga di atas MA200")
        if last['close'] > last['MA50']:
            score += 1
            reasons.append("Harga di atas MA50")
        if 30 <= last['RSI'] <= 70:
            score += 0.5
            reasons.append(f"RSI {last['RSI']:.1f} (netral)")
        if last['Volume_Ratio'] < 0.7:
            score += 0.5
            reasons.append("Volume rendah (potensi akumulasi)")
        if last['MACD'] > last['Signal']:
            score += 1
            reasons.append("MACD bullish")

        is_above_cloud = (last['close'] > last['senkou_a'] and last['close'] > last['senkou_b'])
        cloud_thick = abs(last['senkou_a'] - last['senkou_b']) / last['close']
        
        if is_above_cloud:
            score += 1
            reasons.append("Harga di atas Cloud (bullish)")
        else:
            reasons.append(f"⚠️ Cloud tebal ({cloud_thick:.1%}) → Atap Beton")
            reasons.append("Harga di bawah Cloud (bearish)")

        tk_cross = last['tenkan'] > last['kijun']
        if tk_cross:
            score += 0.5
            reasons.append("✅ Tenkan > Kijun (Golden Cross)")
        else:
            reasons.append("⚠️ Tenkan-Kijun: Belum Golden Cross")

        chikou_ok = chikou_confirmation(daily)
        chikou_status = "Above" if chikou_ok else "Below"
        if chikou_ok:
            score += 0.5
            reasons.append("Chikou di atas harga (bullish)")
        else:
            reasons.append("Chikou di bawah harga (resistensi)")

        chikou_clear, chikou_msg = chikou_clearance(daily)
        if chikou_clear:
            score += 0.5
            reasons.append(f"Chikou: {chikou_msg}")
        else:
            reasons.append(f"⚠️ Chikou: {chikou_msg}")

        future_kumo = future_kumo_status(daily)
        if future_kumo == "Bullish (Hijau)":
            score += 0.5
            reasons.append(f"Future Cloud: {future_kumo}")

        if vsa_low_spread_high_volume(daily):
            score += 0.5
            reasons.append("VSA: Low spread high volume")

        if daily['OBV_trend'].iloc[-1]:
            score += 0.5
            reasons.append("OBV naik (akumulasi)")

        is_squeeze, squeeze_pct = detect_volatility_squeeze(daily)
        if is_squeeze:
            reasons.append(f"⚠️ Squeeze: BBW menyempit {squeeze_pct}%")
            score += 0.5

        has_pre, pre_vol = detect_volume_pre_breakout(daily)
        if has_pre:
            reasons.append(f"⚠️ Volume pre-breakout {pre_vol}x")
            score += 0.5

        retest_level, retest_name = detect_buy_the_retest(daily, tk_cross)
        if retest_level:
            reasons.append(f"✅ Buy the Retest: {retest_name} ({retest_level:.8f})")

        momentum_status, _ = get_momentum_status(daily, tk_cross, is_squeeze, is_accum, last['RSI'], last['Volume_Ratio'], is_breakout_confirmed)
        reasons.append(f"📊 Momentum: {momentum_status}")

        hh, hl = detect_hh_hl(daily)
        if hl:
            score += 0.5
            reasons.append("Higher Low terbentuk")
        if hh:
            score += 0.5
            reasons.append("Higher High terbentuk")

        if detect_double_bottom(daily):
            score += 1
            reasons.append("Pola Double Bottom")
        if detect_bullish_flag(daily):
            score += 1
            reasons.append("Pola Bullish Flag")
        if detect_falling_wedge(daily):
            score += 0.5
            reasons.append("Pola Falling Wedge")
        if detect_rsi_divergence(daily):
            score += 1
            reasons.append("RSI Divergence")
        if detect_macd_divergence(daily):
            score += 1
            reasons.append("MACD Divergence")
        
        patterns = detect_candlestick_patterns(daily)
        if patterns:
            score += 0.5
            reasons.append(f"Pola candlestick: {', '.join(patterns)}")
        if volume_spike_recent(daily):
            score += 0.5
            reasons.append("Volume spike dalam 5 candle")

        if is_breakout_confirmed:
            trigger_msg = f"✅ Breakout confirmed! Entry di {aggressive_entry:.8f}"
        elif tk_cross and current_price < aggressive_entry:
            trigger_msg = f"👀 Menunggu breakout {aggressive_entry:.8f}"
        else:
            trigger_msg = f"Limit order di {conservative_entry:.8f} atau tunggu breakout {aggressive_entry:.8f}"
        reasons.append(f"🔔 Trigger: {trigger_msg}")

        if score >= 5:
            action, confidence = "BUY", "High"
        elif score >= 3.5:
            action, confidence = "BUY", "Medium"
        elif score >= 2:
            action, confidence = "BUY (Speculative)", "Low"
        else:
            action, confidence = "HOLD / WAIT", "Low"

        # Correlation
        btc_data = fetch_ohlcv_cached('BTC/USDT', exchange_name, '1d', limit=100)
        corr = None
        if btc_data is not None:
            btc_data = calculate_indicators(btc_data)
            corr = calculate_correlation(daily, btc_data, period=30)
            if corr is not None:
                if abs(corr) < 0.3:
                    corr_text = f"Correlation: {corr} → Independent"
                elif abs(corr) > 0.7:
                    corr_text = f"Correlation: {corr} → Sangat mengikuti BTC"
                else:
                    corr_text = f"Correlation: {corr} → Cukup mengikuti BTC"
            else:
                corr_text = "Correlation: N/A"
        else:
            corr_text = "Correlation: N/A"

        # Verdict
        if score >= 5:
            verdict_title = "STRONG BUY"
        elif score >= 3.5:
            verdict_title = "BUY"
        elif score >= 2:
            verdict_title = "SPECULATIVE"
        else:
            verdict_title = "HOLD / WAIT"

        key_insight = ""
        if is_accum:
            key_insight += "Ada akumulasi diam-diam. "
        if not chikou_clear:
            key_insight += "Chikou terhambat, rawan fakeout. "
        if not tk_cross:
            key_insight += "Belum Golden Cross. "
        if is_squeeze:
            key_insight += f"Squeeze {squeeze_pct}% → siap meledak. "

        action_plan = f"""
        ✅ Conservative Entry: {conservative_entry:.8f}
        🚀 Aggressive Entry: {aggressive_entry:.8f}
        ❌ Stop Loss: {stop_loss:.8f}
        🎯 Take Profit: {tp1:.8f} & {tp2:.8f}
        📊 Risk/Reward: 1:{rr:.1f}
        """

        verdict = {
            'title': verdict_title,
            'strategy': f"Entry di {aggressive_entry:.8f} jika breakout confirmed",
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
            'is_breakout_confirmed': is_breakout_confirmed,
            'trigger_rec': trigger_msg,
            'correlation': corr_text,
            'verdict': verdict,
            'breakout_probability': breakout_prob,
            'breakout_direction': breakout_dir,
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
        st.error(f"Error: {str(e)}")
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

        conservative_entry = nearest_support if nearest_support < current_price else current_price
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
        elif score >= 2:
            action = "BUY"
        elif score >= 1:
            action = "BUY (Speculative)"
        else:
            action = "HOLD"

        tp1 = conservative_entry + atr_value * 2.0
        rr = (tp1 - conservative_entry) / (conservative_entry - stop_loss) if conservative_entry - stop_loss > 0 else 0

        return {
            'symbol': symbol,
            'current_price': round(current_price, 8),
            'conservative_entry': round(conservative_entry, 8),
            'action': action,
            'score': round(score, 1),
            'rr': round(rr, 2),
            'is_accum': is_accum
        }
    except:
        return None

# ======================== PLOT ========================
def plot_ichimochart(df, symbol, nearest_support, nearest_resistance):
    if df is None or len(df) < 50:
        return go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['tenkan'], name='Tenkan', line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['kijun'], name='Kijun', line=dict(color='red', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['senkou_a'], name='Senkou A', line=dict(color='green', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['senkou_b'], name='Senkou B', line=dict(color='orange', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['chikou'], name='Chikou', line=dict(color='purple', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pd.concat([df['timestamp'], df['timestamp'][::-1]]),
        y=pd.concat([df['senkou_a'], df['senkou_b'][::-1]]),
        fill='toself', fillcolor='rgba(0,100,0,0.2)', line=dict(color='rgba(0,0,0,0)'), name='Cloud'
    ), row=1, col=1)
    fig.add_hline(y=nearest_support, line_dash="dash", line_color="yellow", annotation_text="Support", row=1, col=1)
    fig.add_hline(y=nearest_resistance, line_dash="dot", line_color="orange", annotation_text="Resistance", row=1, col=1)
    colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color=colors), row=2, col=1)
    fig.update_layout(title=f"{symbol} - Ichimoku Chart", template="plotly_dark", height=700, hovermode='x unified')
    return fig

# ======================== UI STREAMLIT ========================
with st.sidebar:
    st.header("⚙️ Settings")
    exchange_name = st.selectbox("Exchange", ["binance", "bybit", "okx", "kucoin", "mexc", "gate"], index=0)
    scan_limit = st.slider("Jumlah coin", 50, 300, 150)
    auto_refresh = st.slider("Auto-refresh (menit)", 5, 60, 30)
    use_btc_filter = st.checkbox("Filter dengan BTC trend", value=True)
    
    st.markdown("---")
    start_auto = st.button("▶️ Mulai Auto Scan")
    
    st.markdown("---")
    st.header("🔍 Manual Analysis")
    manual_symbol = st.text_input("Symbol", placeholder="BTC/USDT")
    manual_button = st.button("Analyze", key="manual_btn")

# Session state
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'last_scan' not in st.session_state:
    st.session_state.last_scan = None
if 'auto_running' not in st.session_state:
    st.session_state.auto_running = False
if 'selected_coin' not in st.session_state:
    st.session_state.selected_coin = None
if 'manual_result' not in st.session_state:
    st.session_state.manual_result = None
if 'deep_result' not in st.session_state:
    st.session_state.deep_result = None

def do_scan():
    with st.spinner("Mengambil daftar coin..."):
        pairs = get_all_usdt_pairs(exchange_name)
        if len(pairs) > scan_limit:
            pairs = pairs[:scan_limit]
    
    progress = st.progress(0)
    results = []
    for i, sym in enumerate(pairs):
        progress.progress((i+1)/len(pairs))
        res = analyze_coin_quick(sym, exchange_name)
        if res and res['action'].startswith('BUY'):
            results.append(res)
        time.sleep(0.05)
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('score', ascending=False)
        st.session_state.scan_results = df
    else:
        st.session_state.scan_results = pd.DataFrame()
    st.session_state.last_scan = datetime.now()
    st.rerun()

if start_auto:
    st.session_state.auto_running = True

if st.session_state.auto_running:
    if st.session_state.scan_results is None:
        do_scan()
    elif st.session_state.last_scan:
        if (datetime.now() - st.session_state.last_scan).total_seconds() / 60 >= auto_refresh:
            do_scan()
    if st.session_state.last_scan:
        next_scan = st.session_state.last_scan + timedelta(minutes=auto_refresh)
        st.sidebar.info(f"Next scan: {next_scan.strftime('%H:%M:%S')}")
    if st.sidebar.button("Stop Auto Scan"):
        st.session_state.auto_running = False
        st.rerun()
else:
    if st.sidebar.button("Scan Sekarang"):
        do_scan()

# MANUAL ANALYSIS
if manual_button and manual_symbol:
    sym = manual_symbol.strip().upper()
    if not sym.endswith('/USDT'):
        sym += '/USDT'
    with st.spinner(f"Analisis {sym}..."):
        res = analyze_coin_deep(sym, exchange_name)
        if res:
            st.session_state.manual_result = res
            st.success(f"Analisis {sym} selesai")
        else:
            st.session_state.manual_result = None
            st.error(f"Gagal analisis {sym}")
    st.rerun()

# TAMPILKAN MANUAL RESULT
if st.session_state.manual_result is not None:
    dr = st.session_state.manual_result
    
    st.subheader(f"📊 {dr['symbol']}")
    
    # PREDIKSI SECTION
    st.markdown("### 🔮 Breakout & Pump Prediction")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Breakout Probability", f"{dr['breakout_probability']}%")
    c2.metric("Direction", dr['breakout_direction'])
    c3.metric("Est. Timing", f"{dr['breakout_hours']} jam")
    c4.metric("Accumulation Score", f"{dr['accumulation_score']}/100")
    
    if dr['is_pump_primed']:
        st.warning("🚨 PUMP PRIMED! Setup tiba-tiba tiang tinggi terdeteksi!")
        for s in dr['pump_signals']:
            st.write(f"- {s}")
    
    if dr['is_stealth_accum']:
        st.info("🕵️ Stealth Accumulation - Whale mengakumulasi diam-diam")
    
    if dr['is_whale_accumulating'] and dr['whale_poc_zone']:
        st.success(f"🐋 Whale Zone: {dr['whale_poc_zone']:.8f}")
    
    with st.expander("Detail Prediksi"):
        for r in dr['breakout_reasons']:
            st.write(f"- {r}")
    with st.expander("Detail Akumulasi"):
        for r in dr['accumulation_details']:
            st.write(f"- {r}")
    
    st.markdown("---")
    
    # METRICS UTAMA
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Harga", f"{dr['current_price']:.8f}")
    col2.metric("Conservative Entry", f"{dr['conservative_entry']:.8f}")
    col3.metric("Aggressive Entry", f"{dr['aggressive_entry']:.8f}")
    col4.metric("Action", dr['action'])
    col5.metric("Score", dr['score'])
    
    st.write(f"**Stop Loss:** {dr['sl']} | **TP1:** {dr['tp1']} | **TP2:** {dr['tp2']}")
    st.write(f"**Risk/Reward:** 1:{dr['rr']} | **ATR:** {dr['atr']:.8f}")
    st.write(f"**Support:** {dr['nearest_support']} | **Resistance:** {dr['nearest_resistance']}")
    st.write(f"**Cloud Thickness:** {dr['cloud_thick_pct']}% | **Chikou:** {dr['chikou_status']} | **Future Cloud:** {dr['future_kumo']}")
    st.write(f"**Correlation:** {dr['correlation']}")
    
    with st.expander("Alasan Analisis"):
        for r in dr['reasons']:
            st.write(f"- {r}")
    
    # VERDICT
    st.markdown(f"## 🚩 FINAL VERDICT: {dr['verdict']['title']}")
    st.write(f"**Momentum:** {dr['verdict']['momentum_status']}")
    st.write(f"**RR:** 1:{dr['verdict']['rr']:.1f}")
    st.write(f"**Strategy:** {dr['verdict']['strategy']}")
    st.write(f"**Key Insight:** {dr['verdict']['key_insight']}")
    st.write(dr['verdict']['action_plan'])
    
    if dr['daily'] is not None:
        st.plotly_chart(plot_ichimochart(dr['daily'], dr['symbol'], dr['nearest_support'], dr['nearest_resistance']), use_container_width=True)

# AUTO SCAN RESULTS
if st.session_state.scan_results is not None and not st.session_state.scan_results.empty:
    df = st.session_state.scan_results
    st.subheader(f"📊 Auto Scan Results")
    if st.session_state.last_scan:
        st.caption(f"Last scan: {st.session_state.last_scan.strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.dataframe(df[['symbol', 'current_price', 'conservative_entry', 'action', 'score', 'rr', 'is_accum']], use_container_width=True)
    
    st.subheader("Deep Analysis")
    selected = st.selectbox("Pilih coin", df['symbol'].tolist(), key="auto_select")
    
    if selected != st.session_state.selected_coin:
        st.session_state.selected_coin = selected
        with st.spinner(f"Analisis {selected}..."):
            deep = analyze_coin_deep(selected, exchange_name)
            if deep:
                st.session_state.deep_result = deep
            else:
                st.session_state.deep_result = None
        st.rerun()
    
    if st.session_state.deep_result is not None:
        dr = st.session_state.deep_result
        st.write(f"### {dr['symbol']}")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Price", f"{dr['current_price']:.8f}")
        col_b.metric("Entry", f"{dr['conservative_entry']:.8f}")
        col_c.metric("Action", dr['action'])
        col_d.metric("Score", dr['score'])
        
        st.write(f"**Breakout Prob:** {dr['breakout_probability']}% | **Accum Score:** {dr['accumulation_score']}/100 | **RR:** 1:{dr['rr']}")
        
        if dr['is_pump_primed']:
            st.warning("🚨 Pump primed!")
        
        with st.expander("Detail"):
            for r in dr['reasons'][:10]:
                st.write(f"- {r}")
        
        if st.button("Lihat Chart", key="view_chart"):
            st.plotly_chart(plot_ichimochart(dr['daily'], dr['symbol'], dr['nearest_support'], dr['nearest_resistance']), use_container_width=True)
else:
    if st.session_state.scan_results is not None:
        st.info("Tidak ada coin memenuhi kriteria")
    else:
        st.info("Tekan 'Mulai Auto Scan' atau 'Scan Sekarang' untuk mulai")

st.caption("Data di-cache | Akumulasi Whale + Prediksi Breakout + Pump Detection")
