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
st.set_page_config(
    page_title="Crypto Scanner Spot Market - Whale Edition", 
    layout="wide",
    page_icon="🐋"
)

# ======================== CUSTOM CSS ========================
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .medium-font { font-size:16px !important; }
    .buy-box { background-color: #00ff0022; padding: 15px; border-radius: 10px; border-left: 5px solid #00ff00; }
    .wait-box { background-color: #ffaa0022; padding: 15px; border-radius: 10px; border-left: 5px solid #ffaa00; }
    .avoid-box { background-color: #ff000022; padding: 15px; border-radius: 10px; border-left: 5px solid #ff0000; }
    .whale-box { background-color: #aa00ff22; padding: 15px; border-radius: 10px; border-left: 5px solid #aa00ff; }
    .pump-badge { background-color: #ff4444; color: white; padding: 5px 10px; border-radius: 20px; font-weight: bold; display: inline-block; }
    .accum-badge { background-color: #44ff44; color: black; padding: 5px 10px; border-radius: 20px; font-weight: bold; display: inline-block; }
    .whale-badge { background-color: #aa44ff; color: white; padding: 5px 10px; border-radius: 20px; font-weight: bold; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# ======================== HEADER ========================
st.title("🐋 Crypto Scanner - Spot Market Edition (WHALE DETECTION)")
st.markdown("""
<div class="medium-font">
<b>Fokus:</b> Mendeteksi coin yang sedang <b>diakumulasi whale</b> dan <b>siap pump</b> di pasar SPOT.<br>
📌 <i>Dilengkapi deteksi manipulasi whale (wick panjang, fake breakout, OBV divergence)</i>
</div>
""", unsafe_allow_html=True)

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
    progress_bar = st.progress(0)
    for i, sym in enumerate(usdt_pairs[:300]):
        try:
            ticker = exchange.fetch_ticker(sym)
            if ticker.get('quoteVolume', 0) > 10000:
                valid_pairs.append(sym)
        except:
            continue
        progress_bar.progress((i+1)/min(len(usdt_pairs), 300))
        time.sleep(0.05)
    
    progress_bar.empty()
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
        print(f"Error fetching {symbol}: {e}")
        return None

# ======================== FUNGSI KORELASI ========================
def calculate_correlation(coin_df, btc_df, period=30):
    if coin_df is None or btc_df is None or len(coin_df) < period or len(btc_df) < period:
        return None, None
    try:
        coin_close = coin_df['close'].tail(period).values
        btc_close = btc_df['close'].tail(period).values
        corr = np.corrcoef(coin_close, btc_close)[0, 1]
        
        if abs(corr) < 0.3:
            interpret = "Bergerak Sendiri (Independen)"
        elif abs(corr) < 0.7:
            interpret = "Cukup Mengikuti BTC"
        else:
            interpret = "Sangat Mengikuti BTC"
        
        return round(corr, 2), interpret
    except:
        return None, None

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
        print(f"Error indicators: {e}")
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
        print(f"Error ichimoku: {e}")
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
            obv_awal = df['OBV'].iloc[-20]
            obv_akhir = df['OBV'].iloc[-1]
            if obv_awal != 0:
                obv_change = (obv_akhir - obv_awal) / abs(obv_awal) * 100
                df['OBV_trend'] = obv_change > 5
                df['OBV_change_pct'] = obv_change
            else:
                df['OBV_trend'] = False
                df['OBV_change_pct'] = 0
        else:
            df['OBV_trend'] = False
            df['OBV_change_pct'] = 0
            
        return df
    except Exception as e:
        print(f"Error OBV: {e}")
        return df

# ======================== WHALE DETECTION (FITUR BARU!) ========================
def detect_obv_divergence(df):
    """Deteksi OBV Divergence - PALING AKURAT buat lihat whale"""
    if df is None or len(df) < 30:
        return False, 0, "Data tidak cukup"
    
    try:
        # Ambil data 20 candle terakhir
        obv_values = df['OBV'].tail(20).values
        price_values = df['close'].tail(20).values
        
        # Hitung tren
        from scipy import stats
        obv_slope = stats.linregress(range(len(obv_values)), obv_values).slope if len(obv_values) > 1 else 0
        price_slope = stats.linregress(range(len(price_values)), price_values).slope if len(price_values) > 1 else 0
        
        # OBV naik tapi harga turun = BULLISH DIVERGENCE (whale beli)
        if obv_slope > 0 and price_slope < 0:
            return True, "bullish", f"OBV naik {obv_slope:.0f} tapi harga turun {abs(price_slope):.0f} → WHALE BELI DIAM-DIAM!"
        
        # OBV turun tapi harga naik = BEARISH DIVERGENCE (whale jual)
        elif obv_slope < 0 and price_slope > 0:
            return True, "bearish", f"OBV turun {abs(obv_slope):.0f} tapi harga naik {price_slope:.0f} → WHALE JUAL DIAM-DIAM!"
        
        return False, "neutral", "Tidak ada divergensi"
    except:
        return False, "neutral", "Error hitung divergensi"

def detect_wick_manipulation(df):
    """Deteksi manipulasi lewat wick panjang"""
    if df is None or len(df) < 5:
        return [], 0
    
    manipulations = []
    score = 0
    
    for i in range(1, 6):
        candle = df.iloc[-i]
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['close'], candle['open'])
        lower_wick = min(candle['close'], candle['open']) - candle['low']
        candle_range = candle['high'] - candle['low']
        
        if candle_range > 0 and body > 0:
            wick_ratio = (upper_wick + lower_wick) / candle_range
            
            # Wick panjang (>60% dari candle)
            if wick_ratio > 0.6:
                # Wick atas panjang = fake pump (whale jual)
                if upper_wick > lower_wick * 2:
                    manipulations.append(f"Candle {i}: Wick ATAS panjang ({upper_wick/body:.1f}x body) → whale jual di harga tinggi (fake pump)")
                    score += 1
                # Wick bawah panjang = fake dump (whale beli)
                elif lower_wick > upper_wick * 2:
                    manipulations.append(f"Candle {i}: Wick BAWAH panjang ({lower_wick/body:.1f}x body) → whale beli murah (fake dump)")
                    score += 2  # Bobot lebih besar karena ini peluang beli!
    
    return manipulations, score

def detect_volume_profile_whale(df):
    """Deteksi whale dari volume profile"""
    if df is None or len(df) < 30:
        return False, 0, ""
    
    try:
        recent = df.tail(50)
        price_bins = pd.cut(recent['close'], bins=10)
        vol_by_price = recent['volume'].groupby(price_bins).sum()
        
        if len(vol_by_price) > 0:
            max_vol = vol_by_price.max()
            total_vol = vol_by_price.sum()
            concentration = max_vol / total_vol if total_vol > 0 else 0
            
            if concentration > 0.3:
                max_bin = vol_by_price.idxmax()
                return True, round(concentration * 100, 1), f"Volume terkonsentrasi {concentration*100:.1f}% di range {max_bin} → whale akumulasi di level itu!"
        
        return False, 0, "Volume tersebar merata"
    except:
        return False, 0, "Error hitung volume profile"

def detect_large_transactions(df):
    """Deteksi transaksi besar dalam 5 candle terakhir"""
    if df is None or len(df) < 10:
        return [], 0
    
    large_tx = []
    avg_vol = df['volume'].tail(30).mean()
    
    for i in range(1, 6):
        candle_vol = df['volume'].iloc[-i]
        if avg_vol > 0 and candle_vol > avg_vol * 3:
            large_tx.append(f"Candle {i}: {candle_vol/avg_vol:.1f}x normal")
    
    return large_tx, len(large_tx)

def detect_fakeout(df, current_price, nearest_resistance):
    """Deteksi breakout palsu (jebakan whale)"""
    if df is None or len(df) < 10:
        return False, 0, []
    
    fake_signals = []
    fake_score = 0
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. Breakout dengan volume rendah = FAKE
    if current_price > nearest_resistance:
        vol_ratio = last['volume'] / df['volume'].tail(10).mean()
        if vol_ratio < 1.2:
            fake_signals.append(f"Breakout volume RENDAH ({vol_ratio:.1f}x) → FAKEOUT!")
            fake_score += 2
    
    # 2. Wick panjang di resistance = REJECTION
    upper_wick = last['high'] - max(last['close'], last['open'])
    body = abs(last['close'] - last['open'])
    if body > 0 and upper_wick > body * 2 and current_price > nearest_resistance:
        fake_signals.append(f"Wick panjang {upper_wick/body:.1f}x body di resistance → whale jual")
        fake_score += 2
    
    # 3. Candle sebelumnya tembus tapi nutup di bawah
    if prev['high'] > nearest_resistance and prev['close'] < nearest_resistance:
        fake_signals.append("Candle sebelumnya tembus tapi nutup di bawah → FALSE BREAKOUT")
        fake_score += 1
    
    return fake_score >= 2, fake_score, fake_signals

def detect_entry_timing(df, current_price, conservative_entry, aggressive_entry, nearest_resistance, has_breakout):
    """
    DETEKSI ENTRY TIMING - Apakah bakal turun dulu atau langsung naik?
    """
    if df is None or len(df) < 20:
        return "⚠️ Data tidak cukup", "Tunggu analisis lanjutan", 0
    
    last = df.iloc[-1]
    prev_5 = df.tail(6).head(5)
    
    # Hitung jarak ke entry
    distance_to_entry = (current_price - conservative_entry) / current_price * 100
    distance_to_resistance = (nearest_resistance - current_price) / current_price * 100
    
    reasons = []
    confidence = 0
    
    # KASUS 1: UDAH BREAKOUT
    if has_breakout or current_price > nearest_resistance:
        return "⚡ AGGRESSIVE ENTRY", f"Harga sudah breakout resistance! Bisa beli di {current_price:.6f} (market order)", 8
    
    # KASUS 2: HARGA UDAH DI ATAS ENTRY (potensi turun dulu)
    if current_price > conservative_entry:
        # Cek apakah ada sinyal bearish sementara
        bearish_signals = 0
        
        # Wick atas panjang?
        upper_wick = last['high'] - max(last['close'], last['open'])
        body = abs(last['close'] - last['open'])
        if body > 0 and upper_wick > body:
            bearish_signals += 1
            reasons.append("Ada wick atas panjang (rejection)")
        
        # RSI mulai overbought?
        if last['RSI'] > 65:
            bearish_signals += 1
            reasons.append(f"RSI {last['RSI']:.0f} (mendekati overbought)")
        
        # Volume turun?
        if last['Volume_Ratio'] < 0.8:
            bearish_signals += 1
            reasons.append("Volume mulai turun")
        
        if bearish_signals >= 2:
            return "⬇️ TUNGGU PULLBACK", f"Harga diperkirakan turun ke {conservative_entry:.6f} dalam {bearish_signals*1-2} hari. Pasang limit order.", 7
        else:
            return "⬇️ TUNGGU PULLBACK (LEMAH)", f"Harga di atas entry, tapi sinyal turun belum kuat. Bisa beli sedikit sekarang atau tunggu.", 4
    
    # KASUS 3: HARGA DI BAWAH ENTRY (langsung beli!)
    elif current_price < conservative_entry:
        return "✅ BELI SEKARANG", f"Harga {current_price:.6f} sudah di bawah entry {conservative_entry:.6f}. Ini kesempatan!", 9
    
    # KASUS 4: HARGA PERSIS DI ENTRY
    elif abs(current_price - conservative_entry) / current_price < 0.001:
        return "✅ PASANG LIMIT ORDER", f"Harga tepat di entry. Pasang limit order atau beli sekarang.", 8
    
    # KASUS 5: DEFAULT
    else:
        return "⏳ PANTAU", f"Entry di {conservative_entry:.6f}, harga sekarang {current_price:.6f}. Pasang alert.", 5

# ======================== FUNGSI DETEKSI LAINNYA ========================
def detect_accumulation(df):
    if df is None or len(df) < 50:
        return False, "Data tidak cukup"
    try:
        recent = df.tail(30)
        price_range = (recent['high'].max() - recent['low'].min()) / recent['low'].min()
        is_sideways = price_range < 0.15
        avg_vol_ratio = recent['Volume_Ratio'].mean()
        is_volume_low = avg_vol_ratio < 0.8
        obv_trend = df['OBV_trend'].iloc[-1] if 'OBV_trend' in df else False
        
        if is_sideways and is_volume_low and obv_trend:
            return True, f"✅ Akumulasi: sideways {price_range:.1%}, volume turun {avg_vol_ratio:.2f}x, OBV naik {df['OBV_change_pct'].iloc[-1]:.1f}%"
        elif is_sideways and is_volume_low:
            return True, f"⚠️ Akumulasi awal: sideways {price_range:.1%}, volume turun {avg_vol_ratio:.2f}x (OBV belum konfirmasi)"
        else:
            return False, f"Tidak akumulasi (range: {price_range:.1%}, volume: {avg_vol_ratio:.2f}x)"
    except Exception as e:
        return False, f"Error: {e}"

def detect_volume_spike(df, lookback=5, threshold=2.0):
    if df is None:
        return False, 0
    try:
        max_vol_ratio = df['Volume_Ratio'].tail(lookback).max()
        if max_vol_ratio > threshold:
            return True, round(max_vol_ratio, 1)
        return False, 0
    except:
        return False, 0

def detect_volatility_squeeze(df, period=20):
    if df is None or len(df) < period:
        return False, 0
    try:
        current_width = df['BB_Width'].iloc[-1]
        avg_width = df['BB_Width'].tail(period).mean()
        if avg_width > 0:
            squeeze_pct = (1 - current_width / avg_width) * 100
            is_squeeze = squeeze_pct > 50
            return is_squeeze, round(squeeze_pct, 1)
        return False, 0
    except:
        return False, 0

def detect_breakout(df, lookback=20):
    if df is None or len(df) < lookback:
        return False, 0
    try:
        recent = df.tail(lookback)
        resistance = recent['high'].max()
        current_price = df['close'].iloc[-1]
        vol_ratio = df['Volume_Ratio'].iloc[-1] if not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1
        
        if current_price > resistance and vol_ratio > 1.5:
            return True, round(resistance, 8)
        return False, 0
    except:
        return False, 0

def detect_support_resistance(df):
    if df is None or len(df) < 20:
        return None, None
    try:
        current_price = df['close'].iloc[-1]
        lows = df['low'].tail(50)
        supports = [low for low in lows if low < current_price]
        nearest_support = max(supports) if supports else current_price * 0.95
        highs = df['high'].tail(50)
        resistances = [high for high in highs if high > current_price]
        nearest_resistance = min(resistances) if resistances else current_price * 1.05
        return round(nearest_support, 8), round(nearest_resistance, 8)
    except:
        return None, None

def get_entry_stop_loss(df, current_price, nearest_support):
    if df is None or len(df) < 20:
        return current_price * 0.95, current_price * 0.90
    
    atr = df['ATR'].iloc[-1] if not pd.isna(df['ATR'].iloc[-1]) else current_price * 0.02
    conservative_entry = nearest_support if nearest_support and nearest_support < current_price else current_price * 0.97
    recent_lows = df['low'].tail(20)
    swing_low = recent_lows.min()
    stop_loss = swing_low * 0.98 if swing_low < conservative_entry else conservative_entry * 0.95
    if conservative_entry - stop_loss > conservative_entry * 0.15:
        stop_loss = conservative_entry * 0.85
    
    return round(conservative_entry, 8), round(stop_loss, 8)

def calculate_targets(entry_price, atr, trend_strength=2):
    tp1 = entry_price + (atr * 2)
    tp2 = entry_price + (atr * 3)
    tp3 = entry_price + (atr * 5)
    return round(tp1, 8), round(tp2, 8), round(tp3, 8)

# ======================== POLA CHART ========================
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
        current_price = df['close'].iloc[-1]
        return current_price > middle_high * 0.98
    except:
        return False

def detect_bullish_flag(df):
    if df is None or len(df) < 30:
        return False
    try:
        first_10 = df.iloc[-30:-20]
        pole_change = (first_10['close'].iloc[-1] - first_10['close'].iloc[0]) / first_10['close'].iloc[0]
        last_20 = df.tail(20)
        flag_range = (last_20['high'].max() - last_20['low'].min()) / last_20['low'].min()
        vol_ratio_mean = last_20['Volume_Ratio'].mean()
        return pole_change > 0.10 and flag_range < 0.08 and vol_ratio_mean < 0.8
    except:
        return False

def detect_cup_handle(df):
    if df is None or len(df) < 50:
        return False
    try:
        min_price = df['low'].tail(50).min()
        min_idx = df['low'].tail(50).idxmin()
        after_bottom = df.loc[min_idx:].tail(20)
        recovery = (after_bottom['close'].iloc[-1] - min_price) / min_price
        last_10 = df.tail(10)
        handle_range = (last_10['high'].max() - last_10['low'].min()) / last_10['low'].min()
        return recovery > 0.15 and handle_range < 0.05
    except:
        return False

# ======================== FUNGSI UTAMA ANALISIS ========================
def analyze_coin_spot(symbol, exchange_name, btc_df=None):
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
        
        # Deteksi berbagai kondisi
        is_accum, acc_reason = detect_accumulation(daily)
        has_volume_spike, spike_ratio = detect_volume_spike(daily)
        is_squeeze, squeeze_pct = detect_volatility_squeeze(daily)
        has_breakout, breakout_level = detect_breakout(daily)
        
        # Support & Resistance
        nearest_support, nearest_resistance = detect_support_resistance(daily)
        if nearest_support is None:
            nearest_support = current_price * 0.95
        if nearest_resistance is None:
            nearest_resistance = current_price * 1.05
        
        # Entry & Stop Loss
        conservative_entry, stop_loss = get_entry_stop_loss(daily, current_price, nearest_support)
        atr_value = last['ATR'] if not pd.isna(last['ATR']) else current_price * 0.02
        tp1, tp2, tp3 = calculate_targets(conservative_entry, atr_value)
        
        # Hitung Risk/Reward
        risk = conservative_entry - stop_loss
        reward = tp1 - conservative_entry
        rr_ratio = reward / risk if risk > 0 else 0
        
        # ==================== WHALE DETECTION (FITUR BARU!) ====================
        # OBV Divergence
        has_divergence, div_type, div_reason = detect_obv_divergence(daily)
        
        # Wick Manipulation
        wick_signals, wick_score = detect_wick_manipulation(daily)
        
        # Volume Profile Whale
        has_vol_concentration, vol_conc_pct, vol_profile_reason = detect_volume_profile_whale(daily)
        
        # Large Transactions
        large_tx, tx_count = detect_large_transactions(daily)
        
        # Fakeout Detection
        is_fakeout, fake_score, fake_signals = detect_fakeout(daily, current_price, nearest_resistance)
        
        # Entry Timing (APAKAH TURUN DULU ATAU LANGSUNG NAIK?)
        timing_status, timing_reason, timing_confidence = detect_entry_timing(
            daily, current_price, conservative_entry, None, nearest_resistance, has_breakout
        )
        
        # Hitung total whale confidence
        whale_confidence = 0
        whale_signals = []
        
        if has_divergence and div_type == "bullish":
            whale_confidence += 3
            whale_signals.append(f"🐋 {div_reason}")
        elif has_divergence and div_type == "bearish":
            whale_confidence -= 1
            whale_signals.append(f"⚠️ {div_reason}")
        
        if wick_score >= 2:
            whale_confidence += 2
            for w in wick_signals[:2]:
                whale_signals.append(f"📊 {w}")
        
        if has_vol_concentration:
            whale_confidence += 2
            whale_signals.append(f"📊 {vol_profile_reason}")
        
        if tx_count >= 1:
            whale_confidence += 1
            for tx in large_tx[:1]:
                whale_signals.append(f"🔊 {tx}")
        
        if is_fakeout:
            whale_confidence -= 2
            for fs in fake_signals[:1]:
                whale_signals.append(f"⚠️ {fs}")
        
        # SCORING SYSTEM (0-10)
        score = 0
        reasons = []
        
        # 1. Accumulation (bobot 3)
        if is_accum:
            score += 3
            reasons.append(acc_reason)
        
        # 2. Whale Detection (bobot 3 - BARU!)
        if whale_confidence >= 5:
            score += 3
            reasons.append(f"🐋 WHALE TERKONFIRMASI! Confidence {whale_confidence}/7")
        elif whale_confidence >= 3:
            score += 2
            reasons.append(f"🐋 Ada aktivitas whale (confidence {whale_confidence}/7)")
        elif whale_confidence >= 1:
            score += 1
            reasons.append(f"🔍 Indikasi whale lemah (confidence {whale_confidence}/7)")
        
        # Tambahkan sinyal whale ke reasons
        for ws in whale_signals[:3]:
            reasons.append(f"   {ws}")
        
        # 3. Trend (bobot 2)
        if last['close'] > last['MA200']:
            score += 2
            reasons.append("✅ Harga di atas MA200 (bullish jangka panjang)")
        elif last['close'] > last['MA50']:
            score += 1
            reasons.append("📈 Harga di atas MA50 (trend naik jangka pendek)")
        else:
            reasons.append("⚠️ Harga di bawah MA50 (trend netral/bearish)")
        
        # 4. Ichimoku Cloud (bobot 2)
        above_cloud = last['close'] > last['senkou_a'] and last['close'] > last['senkou_b']
        if above_cloud:
            score += 2
            reasons.append("✅ Harga di atas Cloud Ichimoku (bullish)")
        else:
            reasons.append("☁️ Harga di bawah Cloud Ichimoku (masih bearish/netral)")
        
        # 5. TK Cross (bobot 1)
        tk_cross = last['tenkan'] > last['kijun']
        if tk_cross:
            score += 1
            reasons.append("✅ Tenkan di atas Kijun (golden cross)")
        else:
            reasons.append("⚠️ Tenkan di bawah Kijun (masih konsolidasi)")
        
        # 6. RSI (bobot 1)
        if 40 <= last['RSI'] <= 60:
            score += 1
            reasons.append(f"📊 RSI {last['RSI']:.1f} (netral)")
        elif last['RSI'] < 30:
            reasons.append(f"💀 RSI {last['RSI']:.1f} (oversold)")
        elif last['RSI'] > 70:
            reasons.append(f"⚠️ RSI {last['RSI']:.1f} (overbought)")
        else:
            score += 0.5
            reasons.append(f"📊 RSI {last['RSI']:.1f}")
        
        # 7. MACD (bobot 1)
        if last['MACD'] > last['Signal']:
            score += 1
            reasons.append("✅ MACD bullish")
        else:
            reasons.append("📉 MACD bearish")
        
        # 8. Volume Analysis (bobot 1)
        if has_volume_spike:
            score += 1
            reasons.append(f"🔊 Volume spike {spike_ratio}x!")
        elif last['Volume_Ratio'] < 0.7:
            score += 0.5
            reasons.append(f"🔇 Volume rendah {last['Volume_Ratio']:.2f}x")
        
        # 9. Squeeze (bobot 1.5)
        if is_squeeze:
            score += 1.5
            reasons.append(f"🔥 Volatility Squeeze {squeeze_pct}%!")
        
        # 10. Breakout (bobot 1.5)
        if has_breakout:
            score += 1.5
            reasons.append(f"🚀 Breakout resistance di {breakout_level:.8f}!")
        
        # 11. Pola chart
        if detect_double_bottom(daily):
            score += 1
            reasons.append("📉 Pola Double Bottom")
        if detect_bullish_flag(daily):
            score += 1
            reasons.append("🚩 Pola Bullish Flag")
        if detect_cup_handle(daily):
            score += 1
            reasons.append("🏆 Pola Cup and Handle")
        
        # 12. OBV trend
        if daily['OBV_trend'].iloc[-1]:
            score += 1
            reasons.append(f"🐋 OBV naik {daily['OBV_change_pct'].iloc[-1]:.1f}%")
        
        # Korelasi BTC
        correlation_text = ""
        if btc_df is not None:
            corr, corr_interp = calculate_correlation(daily, btc_df)
            if corr:
                correlation_text = f"Korelasi BTC: {corr} ({corr_interp})"
                reasons.append(f"🔄 {correlation_text}")
        
        # STATUS BERDASARKAN SKOR
        if score >= 7:
            status = "🚀 STRONG BUY - SIAP PUMP!"
            confidence = "Tinggi"
            action = "BELI"
        elif score >= 5:
            status = "📈 BUY - POTENSI PUMP"
            confidence = "Sedang"
            action = "BELI (Spekulatif)"
        elif score >= 3:
            status = "⏳ ACCUMULATION - PANTAU TERUS"
            confidence = "Rendah"
            action = "PANTAU"
        else:
            status = "⏸️ HOLD / WAIT - HINDARI DULU"
            confidence = "Sangat Rendah"
            action = "HOLD / TUNGGU"
        
        # Potensi pump
        if score >= 7:
            pump_potential = "50-100%+ dalam 1-2 minggu"
        elif score >= 5:
            pump_potential = "25-50% dalam 1-2 minggu"
        elif score >= 3:
            pump_potential = "10-25% dalam 2-4 minggu"
        else:
            pump_potential = "Belum jelas"
        
        # Hitung persentase keuntungan tiap target
        tp1_pct = ((tp1 - conservative_entry) / conservative_entry) * 100
        tp2_pct = ((tp2 - conservative_entry) / conservative_entry) * 100
        tp3_pct = ((tp3 - conservative_entry) / conservative_entry) * 100
        
        beginner_summary = f"""
        **Gampangnya:** 
        - 📊 Skor {score:.1f}/10 → {status}
        - 🎯 Potensi pump: {pump_potential}
        - ⏰ Timing: {timing_status} - {timing_reason}
        - 💰 Entry: {conservative_entry:.8f}
        - 🛑 Stop loss: {stop_loss:.8f}
        - 🎯 Target: {tp1:.8f} (+{tp1_pct:.1f}%) → {tp2:.8f} (+{tp2_pct:.1f}%) → {tp3:.8f} (+{tp3_pct:.1f}%)
        """
        
        return {
            'symbol': symbol,
            'current_price': round(current_price, 8),
            'entry_price': conservative_entry,
            'stop_loss': stop_loss,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'tp1_pct': round(tp1_pct, 1),
            'tp2_pct': round(tp2_pct, 1),
            'tp3_pct': round(tp3_pct, 1),
            'rr_ratio': round(rr_ratio, 2),
            'score': round(score, 1),
            'status': status,
            'action': action,
            'confidence': confidence,
            'pump_potential': pump_potential,
            'is_accum': is_accum,
            'has_volume_spike': has_volume_spike,
            'is_squeeze': is_squeeze,
            'has_breakout': has_breakout,
            'tk_cross': tk_cross,
            'above_cloud': above_cloud,
            'rsi': round(last['RSI'], 1),
            'volume_ratio': round(last['Volume_Ratio'], 2) if not pd.isna(last['Volume_Ratio']) else 0,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'reasons': reasons,
            'beginner_summary': beginner_summary,
            'correlation': correlation_text,
            'daily_df': daily,
            # FITUR BARU
            'whale_confidence': whale_confidence,
            'whale_signals': whale_signals,
            'wick_signals': wick_signals,
            'timing_status': timing_status,
            'timing_reason': timing_reason,
            'is_fakeout': is_fakeout,
            'has_divergence': has_divergence,
            'div_type': div_type
        }
        
    except Exception as e:
        print(f"Error analyze {symbol}: {e}")
        return None

def quick_scan(symbol, exchange_name):
    try:
        daily = fetch_ohlcv_cached(symbol, exchange_name, '1d', limit=100)
        if daily is None or len(daily) < 50:
            return None
        daily = calculate_indicators(daily)
        daily = calculate_obv(daily)
        last = daily.iloc[-1]
        
        score = 0
        if daily['OBV_trend'].iloc[-1]:
            score += 2
        if last['close'] > last['MA200']:
            score += 1
        if last['close'] > last['MA50']:
            score += 1
        if last['Volume_Ratio'] < 0.7:
            score += 1
        if 40 <= last['RSI'] <= 60:
            score += 0.5
        
        return {
            'symbol': symbol,
            'price': round(last['close'], 8),
            'score': round(score, 1),
            'obv_trend': daily['OBV_trend'].iloc[-1]
        }
    except:
        return None

# ======================== PLOT CHART ========================
def plot_simple_chart(df, symbol, support=None, resistance=None):
    if df is None or len(df) < 50:
        return go.Figure()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f"{symbol} - Harga & Indikator", "Volume"))
    
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Harga'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MA50'], 
                            name='MA50', line=dict(color='orange', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MA200'], 
                            name='MA200', line=dict(color='red', width=1.5)), row=1, col=1)
    
    if support:
        fig.add_hline(y=support, line_dash="dash", line_color="green", 
                     annotation_text="Support", row=1, col=1)
    if resistance:
        fig.add_hline(y=resistance, line_dash="dot", line_color="red", 
                     annotation_text="Resistance", row=1, col=1)
    
    colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color=colors), row=2, col=1)
    
    fig.update_layout(template="plotly_dark", height=600, hovermode='x unified')
    fig.update_xaxes(title_text="Tanggal", row=2, col=1)
    fig.update_yaxes(title_text="Harga (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

# ======================== STREAMLIT UI ========================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    exchange_name = st.selectbox(
        "Pilih Exchange", 
        ["binance", "bybit", "okx", "kucoin", "gate", "coinbase", "bitget", "kraken"], 
        index=0,
        help="Pilih exchange yang ingin di-scan"
    )
    
    scan_limit = st.slider("Jumlah coin yang di-scan", 50, 300, 150)
    min_score = st.slider("Minimal skor untuk ditampilkan", 0, 10, 3)
    auto_refresh = st.checkbox("Auto refresh setiap 30 menit", value=True)
    
    st.markdown("---")
    st.header("🔍 Manual Check")
    manual_symbol = st.text_input("Cek coin tertentu", placeholder="Contoh: BTC/USDT, ETH/USDT")
    manual_button = st.button("🔍 Analisis Manual", use_container_width=True)
    
    st.markdown("---")
    st.header("📖 Panduan Skor & Whale")
    st.markdown("""
    **Skor:**
    - **7-10**: 🚀 Siap pump
    - **5-7**: 📈 Potensi pump
    - **3-5**: ⏳ Akumulasi
    - **0-3**: ⏸️ Hindari
    
    **Whale Signals:**
    - 🐋 OBV Divergence = Paling akurat
    - 📊 Wick panjang = Manipulasi
    - 🔊 Volume spike = Whale gerak
    """)
    
    st.caption("💡 Tips: Cari coin dengan whale confidence tinggi + timing 'BELI SEKARANG'")

# Session state
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'last_scan' not in st.session_state:
    st.session_state.last_scan = None
if 'manual_result' not in st.session_state:
    st.session_state.manual_result = None

# Tombol scan
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    scan_button = st.button("🔍 MULAI SCAN", type="primary", use_container_width=True)

# Proses scan
if scan_button or (auto_refresh and st.session_state.scan_results is not None and 
                   (st.session_state.last_scan is None or 
                    (datetime.now() - st.session_state.last_scan).seconds > 1800)):
    
    with st.spinner("📡 Mengambil daftar coin..."):
        pairs = get_all_usdt_pairs(exchange_name)
        if len(pairs) > scan_limit:
            pairs = pairs[:scan_limit]
    
    st.info(f"🔍 Scanning {len(pairs)} coin...")
    
    with st.spinner("📊 Ambil data BTC..."):
        btc_df = fetch_ohlcv_cached('BTC/USDT', exchange_name, '1d', limit=100)
        if btc_df is not None:
            btc_df = calculate_indicators(btc_df)
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, sym in enumerate(pairs):
        status_text.text(f"Scanning {i+1}/{len(pairs)}: {sym}")
        res = quick_scan(sym, exchange_name)
        if res and res['score'] >= min_score:
            results.append(res)
        progress_bar.progress((i+1)/len(pairs))
        time.sleep(0.03)
    
    status_text.empty()
    progress_bar.empty()
    
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('score', ascending=False)
        st.session_state.scan_results = df_results
        st.session_state.last_scan = datetime.now()
        st.success(f"✅ Selesai! Ditemukan {len(results)} coin")
        st.rerun()
    else:
        st.warning(f"Tidak ada coin dengan skor >= {min_score}")
        st.session_state.scan_results = pd.DataFrame()

# Tampilkan hasil scan
if st.session_state.scan_results is not None and not st.session_state.scan_results.empty:
    df = st.session_state.scan_results
    
    st.subheader(f"📊 Hasil Scan - {st.session_state.last_scan.strftime('%H:%M:%S') if st.session_state.last_scan else 'Sekarang'}")
    
    for _, row in df.head(10).iterrows():
        if row['score'] >= 7:
            bg = "#00ff0015"
            badge = "🚀 SIAP PUMP"
        elif row['score'] >= 5:
            bg = "#44ff0015"
            badge = "📈 POTENSI PUMP"
        else:
            bg = "#ffaa0015"
            badge = "⏳ AKUMULASI"
        
        st.markdown(f"""
        <div style="background-color: {bg}; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
            <table style="width: 100%;">
                <tr><td style="width: 25%;"><b>{row['symbol']}</b></td>
                    <td style="width: 15%;">💰 ${row['price']:.6f}</td>
                    <td style="width: 15%;">⭐ Skor: {row['score']}</td>
                    <td style="width: 25%;"><span class="pump-badge">{badge}</span></td>
                    <td style="width: 20%;">{'🐋 OBV Naik' if row['obv_trend'] else ''}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🔍 Analisis Detail")
    selected_coin = st.selectbox("Pilih coin untuk analisis lengkap", df['symbol'].tolist())
    
    if selected_coin:
        with st.spinner(f"Menganalisis {selected_coin}..."):
            btc_df = fetch_ohlcv_cached('BTC/USDT', exchange_name, '1d', limit=100)
            if btc_df is not None:
                btc_df = calculate_indicators(btc_df)
            detail = analyze_coin_spot(selected_coin, exchange_name, btc_df)
            
            if detail:
                # Status box
                if detail['score'] >= 7:
                    box_class = "buy-box"
                elif detail['score'] >= 5:
                    box_class = "buy-box"
                elif detail['score'] >= 3:
                    box_class = "wait-box"
                else:
                    box_class = "avoid-box"
                
                st.markdown(f"""
                <div class="{box_class}">
                    <h3>{detail['status']}</h3>
                    <p><b>Skor:</b> {detail['score']}/10 | <b>Konfidensi:</b> {detail['confidence']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # WHALE BOX (FITUR BARU!)
                if detail['whale_confidence'] >= 3:
                    st.markdown(f"""
                    <div class="whale-box">
                        <h4>🐋 WHALE DETECTION</h4>
                        <p><b>Confidence:</b> {detail['whale_confidence']}/7</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for ws in detail['whale_signals']:
                        st.write(f"- {ws}")
                
                # TIMING BOX (FITUR BARU!)
                if detail['timing_status'] == "✅ BELI SEKARANG":
                    st.success(f"⏰ **{detail['timing_status']}** - {detail['timing_reason']}")
                elif detail['timing_status'] == "⬇️ TUNGGU PULLBACK":
                    st.warning(f"⏰ **{detail['timing_status']}** - {detail['timing_reason']}")
                elif detail['timing_status'] == "⚡ AGGRESSIVE ENTRY":
                    st.info(f"⏰ **{detail['timing_status']}** - {detail['timing_reason']}")
                else:
                    st.info(f"⏰ **{detail['timing_status']}** - {detail['timing_reason']}")
                
                # Info penting
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("💰 Harga Saat Ini", f"${detail['current_price']:.6f}")
                with col2:
                    st.metric("🎯 Entry Point", f"${detail['entry_price']:.6f}")
                with col3:
                    st.metric("🛑 Stop Loss", f"${detail['stop_loss']:.6f}")
                with col4:
                    st.metric("📊 Risk/Reward", f"1:{detail['rr_ratio']:.1f}")
                
                # Target profit dengan PERSENTASE (FITUR BARU!)
                st.write("**🎯 Target Profit (ambil bertahap):**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"Target 1: ${detail['tp1']:.6f} (+{detail['tp1_pct']:.1f}%)")
                with col2:
                    st.info(f"Target 2: ${detail['tp2']:.6f} (+{detail['tp2_pct']:.1f}%)")
                with col3:
                    st.success(f"Target 3: ${detail['tp3']:.6f} (+{detail['tp3_pct']:.1f}%)")
                
                st.info(detail['beginner_summary'])
                
                st.write(f"**📊 Support:** ${detail['nearest_support']:.6f} | **Resistance:** ${detail['nearest_resistance']:.6f}")
                
                # Signal badges
                st.write("**📡 Signal yang terdeteksi:**")
                sig_col1, sig_col2, sig_col3, sig_col4, sig_col5 = st.columns(5)
                with sig_col1:
                    if detail['is_accum']:
                        st.success("🐋 Akumulasi")
                    else:
                        st.info("⏳ Belum akumulasi")
                with sig_col2:
                    if detail['has_volume_spike']:
                        st.warning("🔊 Volume Spike!")
                    else:
                        st.info("📊 Volume normal")
                with sig_col3:
                    if detail['is_squeeze']:
                        st.error("🔥 Squeeze!")
                    else:
                        st.info("📈 Volatility normal")
                with sig_col4:
                    if detail['has_breakout']:
                        st.success("🚀 Breakout!")
                    else:
                        st.info("🔒 Belum breakout")
                with sig_col5:
                    if detail['has_divergence'] and detail['div_type'] == 'bullish':
                        st.success("🐋 OBV Divergence!")
                    elif detail['has_divergence']:
                        st.warning("⚠️ OBV Divergence")
                    else:
                        st.info("📊 OBV normal")
                
                # Wick warning
                if detail['wick_signals']:
                    with st.expander("⚠️ Peringatan Wick Panjang (Manipulasi Whale)"):
                        for w in detail['wick_signals']:
                            st.write(f"- {w}")
                
                # Alasan lengkap
                with st.expander("📝 Lihat semua alasan analisis"):
                    for reason in detail['reasons']:
                        st.write(f"- {reason}")
                
                if detail['correlation']:
                    st.caption(detail['correlation'])
                
                st.plotly_chart(plot_simple_chart(
                    detail['daily_df'], 
                    detail['symbol'],
                    detail['nearest_support'],
                    detail['nearest_resistance']
                ), use_container_width=True)
                
                st.caption("⚠️ **Disclaimer:** Analisis ini berdasarkan data historis. Bukan jaminan keuntungan. Selalu gunakan manajemen risiko!")
                
else:
    if st.session_state.scan_results is not None:
        st.warning("Belum ada hasil scan. Klik 'MULAI SCAN' untuk mulai.")
    else:
        st.info("👋 Selamat datang! Klik 'MULAI SCAN' di atas untuk mulai mencari coin.")

# Manual analysis
if manual_button and manual_symbol:
    symbol = manual_symbol.strip().upper()
    if not symbol.endswith('/USDT'):
        symbol += '/USDT'
    
    with st.spinner(f"Menganalisis {symbol}..."):
        btc_df = fetch_ohlcv_cached('BTC/USDT', exchange_name, '1d', limit=100)
        if btc_df is not None:
            btc_df = calculate_indicators(btc_df)
        result = analyze_coin_spot(symbol, exchange_name, btc_df)
        
        if result:
            st.session_state.manual_result = result
            st.success(f"✅ Analisis {symbol} selesai!")
        else:
            st.error(f"❌ Gagal menganalisis {symbol}")
        st.rerun()

# Tampilkan manual result
if st.session_state.manual_result:
    detail = st.session_state.manual_result
    
    st.markdown("---")
    st.subheader(f"🔍 Hasil Manual Analysis: {detail['symbol']}")
    
    if detail['score'] >= 7:
        box_class = "buy-box"
    elif detail['score'] >= 5:
        box_class = "buy-box"
    elif detail['score'] >= 3:
        box_class = "wait-box"
    else:
        box_class = "avoid-box"
    
    st.markdown(f"""
    <div class="{box_class}">
        <h3>{detail['status']}</h3>
        <p><b>Skor:</b> {detail['score']}/10 | <b>Konfidensi:</b> {detail['confidence']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # WHALE BOX
    if detail['whale_confidence'] >= 3:
        st.markdown(f"""
        <div class="whale-box">
            <h4>🐋 WHALE DETECTION (Confidence {detail['whale_confidence']}/7)</h4>
        </div>
        """, unsafe_allow_html=True)
        for ws in detail['whale_signals']:
            st.write(f"- {ws}")
    
    # TIMING
    if detail['timing_status'] == "✅ BELI SEKARANG":
        st.success(f"⏰ **{detail['timing_status']}** - {detail['timing_reason']}")
    elif detail['timing_status'] == "⬇️ TUNGGU PULLBACK":
        st.warning(f"⏰ **{detail['timing_status']}** - {detail['timing_reason']}")
    else:
        st.info(f"⏰ **{detail['timing_status']}** - {detail['timing_reason']}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Harga Saat Ini", f"${detail['current_price']:.6f}")
    with col2:
        st.metric("🎯 Entry Point", f"${detail['entry_price']:.6f}")
    with col3:
        st.metric("🛑 Stop Loss", f"${detail['stop_loss']:.6f}")
    with col4:
        st.metric("📊 Risk/Reward", f"1:{detail['rr_ratio']:.1f}")
    
    st.write("**🎯 Target Profit:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"Target 1: ${detail['tp1']:.6f} (+{detail['tp1_pct']:.1f}%)")
    with col2:
        st.info(f"Target 2: ${detail['tp2']:.6f} (+{detail['tp2_pct']:.1f}%)")
    with col3:
        st.success(f"Target 3: ${detail['tp3']:.6f} (+{detail['tp3_pct']:.1f}%)")
    
    st.info(detail['beginner_summary'])
    st.write(f"**📊 Support:** ${detail['nearest_support']:.6f} | **Resistance:** ${detail['nearest_resistance']:.6f}")
    
    with st.expander("📝 Detail Analisis"):
        for reason in detail['reasons']:
            st.write(f"- {reason}")
    
    st.plotly_chart(plot_simple_chart(
        detail['daily_df'], 
        detail['symbol'],
        detail['nearest_support'],
        detail['nearest_resistance']
    ), use_container_width=True)

# Footer
st.markdown("---")
st.caption("""
**Fitur Baru:**
- 🐋 **Whale Detection** - OBV Divergence, Volume Profile, Wick Manipulation
- ⏰ **Entry Timing** - Kasih tahu bakal turun dulu atau langsung naik
- 📊 **Target Persentase** - TP1, TP2, TP3 dalam bentuk persen

**Cara Baca:**
- ✅ BELI SEKARANG = Harga di bawah entry, langsung beli
- ⬇️ TUNGGU PULLBACK = Harga bakal turun dulu, pasang limit order
- ⚡ AGGRESSIVE ENTRY = Udah breakout, bisa beli di harga sekarang
""")
